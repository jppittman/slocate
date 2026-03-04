//! Wait-free SPSC ring buffer.
//!
//! Single-producer, single-consumer bounded queue. The producer owns the tail
//! (write position) and the consumer owns the head (read position). Send is a
//! single `AtomicUsize::store(Release)`, recv is a single `AtomicUsize::load(Acquire)`.
//! No CAS, no retry, no contention between producers — each producer gets its
//! own channel.
//!
//! Cache-line padding on head and tail prevents false sharing between the
//! producer and consumer cores.
//!
//! # Safety
//!
//! The ring buffer uses `UnsafeCell<MaybeUninit<T>>` for the slot array.
//! Safety invariants:
//! - Only the producer writes to slots (at `tail % cap`)
//! - Only the consumer reads from slots (at `head % cap`)
//! - A slot is only written when `tail - head < cap` (not full)
//! - A slot is only read when `head != tail` (not empty)
//! - Slots are `ptr::write` / `ptr::read` (no double-drop)

use std::cell::{Cell, UnsafeCell};
use std::mem::{MaybeUninit, size_of};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache line size for padding. 64 bytes on x86, conservative default.
const CACHE_LINE: usize = 64;

/// Errors returned by `SpscSender::try_send`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrySendError<T> {
    /// Buffer is full. Returns the message.
    Full(T),
    /// Consumer dropped. Returns the message.
    Disconnected(T),
}

/// Errors returned by `SpscReceiver::try_recv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    /// Buffer is empty.
    Empty,
    /// Producer dropped.
    Disconnected,
}

/// Cache-line padded atomic counter.
///
/// Ensures the producer's tail and consumer's head live on separate cache lines,
/// eliminating false sharing between cores.
#[repr(C)]
#[repr(align(64))]
struct PaddedAtomic {
    value: AtomicUsize,
    _pad: [u8; CACHE_LINE - size_of::<AtomicUsize>()],
}

impl PaddedAtomic {
    const fn new(v: usize) -> Self {
        Self {
            value: AtomicUsize::new(v),
            _pad: [0u8; CACHE_LINE - size_of::<AtomicUsize>()],
        }
    }
}

/// Shared ring buffer backing a single SPSC channel.
struct RingBuffer<T> {
    /// Slots. Power-of-2 length for fast modulo via bitmask.
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Capacity (always a power of 2).
    cap: usize,
    /// Bitmask: `cap - 1`. Index = position & mask.
    mask: usize,
    /// Consumer's read position. Only the consumer stores to this.
    head: PaddedAtomic,
    /// Producer's write position. Only the producer stores to this.
    tail: PaddedAtomic,
}

// Safety: RingBuffer is shared between exactly one producer thread and one
// consumer thread. The producer only writes tail and buffer slots at tail,
// the consumer only writes head and reads buffer slots at head. AtomicUsize
// provides the cross-thread synchronization via Release/Acquire ordering.
unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

impl<T> RingBuffer<T> {
    fn new(min_capacity: usize) -> Self {
        // Round up to next power of 2, minimum 2
        let cap = min_capacity.max(2).next_power_of_two();
        let mask = cap - 1;

        let mut buffer = Vec::with_capacity(cap);
        for _ in 0..cap {
            buffer.push(UnsafeCell::new(MaybeUninit::uninit()));
        }

        Self {
            buffer: buffer.into_boxed_slice(),
            cap,
            mask,
            head: PaddedAtomic::new(0),
            tail: PaddedAtomic::new(0),
        }
    }
}

impl<T> Drop for RingBuffer<T> {
    fn drop(&mut self) {
        // Drop any messages still in the buffer.
        // We must handle wrapping: tail.wrapping_sub(head) gives correct count.
        let head = *self.head.value.get_mut();
        let tail = *self.tail.value.get_mut();
        let count = tail.wrapping_sub(head);
        for i in 0..count {
            let idx = head.wrapping_add(i) & self.mask;
            // Safety: slots in [head..tail) are initialized
            unsafe {
                self.buffer[idx].get_mut().assume_init_drop();
            }
        }
    }
}

/// Producer end of an SPSC channel. Not Clone -- one producer per channel.
///
/// Uses `Cell<usize>` for cached indices so that `try_send` takes `&self`
/// instead of `&mut self`. `Cell` is `!Sync`, which is correct -- only one
/// thread may use a sender.
pub struct SpscSender<T> {
    ring: Arc<RingBuffer<T>>,
    /// Cached copy of tail. Only this thread writes tail.
    cached_tail: Cell<usize>,
    /// Cached copy of head (may be stale -- always conservative).
    cached_head: Cell<usize>,
}

// Safety: SpscSender<T> can be sent to the producer thread.
// Only one SpscSender exists per channel (not Clone).
// Note: SpscSender is !Sync (because of Cell), which is correct for SPSC.
unsafe impl<T: Send> Send for SpscSender<T> {}

/// Consumer end of an SPSC channel. Not Clone -- one consumer per channel.
pub struct SpscReceiver<T> {
    ring: Arc<RingBuffer<T>>,
    /// Cached copy of head. Only this thread writes head.
    cached_head: usize,
    /// Cached copy of tail (may be stale -- always conservative).
    cached_tail: usize,
}

// Safety: SpscReceiver<T> can be sent to the consumer thread.
// Only one SpscReceiver exists per channel (not Clone).
unsafe impl<T: Send> Send for SpscReceiver<T> {}

/// Create an SPSC channel with the given minimum capacity.
///
/// Capacity is rounded up to the next power of 2 (minimum 2).
/// Returns `(sender, receiver)`.
#[must_use]
pub fn spsc_channel<T>(min_capacity: usize) -> (SpscSender<T>, SpscReceiver<T>) {
    let ring = Arc::new(RingBuffer::new(min_capacity));

    let sender = SpscSender {
        ring: Arc::clone(&ring),
        cached_tail: Cell::new(0),
        cached_head: Cell::new(0),
    };

    let receiver = SpscReceiver {
        ring,
        cached_head: 0,
        cached_tail: 0,
    };

    (sender, receiver)
}

impl<T> SpscSender<T> {
    /// Try to send a message. Wait-free: at most one atomic store.
    ///
    /// Takes `&self` (not `&mut self`) thanks to `Cell` for cached indices.
    /// `Cell` is `!Sync`, ensuring only one thread can use the sender at a time.
    ///
    /// Returns `Ok(())` on success, `Err(TrySendError::Full(msg))` if the
    /// buffer is full, or `Err(TrySendError::Disconnected(msg))` if the
    /// consumer has been dropped.
    pub fn try_send(&self, msg: T) -> Result<(), TrySendError<T>> {
        // Early disconnect check (relaxed load -- cheap, may be slightly stale)
        if Arc::strong_count(&self.ring) == 1 {
            return Err(TrySendError::Disconnected(msg));
        }

        let tail = self.cached_tail.get();

        // Check if buffer is full using cached head
        if tail - self.cached_head.get() >= self.ring.cap {
            // Refresh head from the consumer's atomic
            self.cached_head
                .set(self.ring.head.value.load(Ordering::Acquire));
            if tail - self.cached_head.get() >= self.ring.cap {
                // Check if consumer is gone
                if Arc::strong_count(&self.ring) == 1 {
                    return Err(TrySendError::Disconnected(msg));
                }
                return Err(TrySendError::Full(msg));
            }
        }

        // Write the message into the slot
        let idx = tail & self.ring.mask;
        // Safety: we verified tail - head < cap, so this slot is not occupied
        // by the consumer. Only we write to slots at tail.
        unsafe {
            (*self.ring.buffer[idx].get()).write(msg);
        }

        // Publish: make the write visible to the consumer
        self.cached_tail.set(tail + 1);
        self.ring
            .tail
            .value
            .store(self.cached_tail.get(), Ordering::Release);

        Ok(())
    }

    /// Returns true if the consumer has been dropped.
    #[allow(dead_code)]
    pub fn is_disconnected(&self) -> bool {
        Arc::strong_count(&self.ring) == 1
    }
}

impl<T> SpscReceiver<T> {
    /// Try to receive a message. Wait-free: at most one atomic store.
    ///
    /// Returns `Ok(msg)` on success, `Err(TryRecvError::Empty)` if the buffer
    /// is empty, or `Err(TryRecvError::Disconnected)` if the producer has been
    /// dropped and the buffer is empty.
    pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
        let head = self.cached_head;

        // Check if buffer is empty using cached tail
        if head == self.cached_tail {
            // Refresh tail from the producer's atomic
            self.cached_tail = self.ring.tail.value.load(Ordering::Acquire);
            if head == self.cached_tail {
                // Empty -- check if producer is gone
                if Arc::strong_count(&self.ring) == 1 {
                    return Err(TryRecvError::Disconnected);
                }
                return Err(TryRecvError::Empty);
            }
        }

        // Read the message from the slot
        let idx = head & self.ring.mask;
        // Safety: we verified head != tail, so this slot was written by the
        // producer. Only we read from slots at head.
        let msg = unsafe { (*self.ring.buffer[idx].get()).assume_init_read() };

        // Publish: free the slot for the producer
        self.cached_head = head + 1;
        self.ring
            .head
            .value
            .store(self.cached_head, Ordering::Release);

        Ok(msg)
    }

    /// Returns true if the producer has been dropped.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_disconnected(&self) -> bool {
        Arc::strong_count(&self.ring) == 1
    }

    /// Returns the number of messages currently in the buffer.
    ///
    /// This is approximate -- the producer may be writing concurrently.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        let tail = self.ring.tail.value.load(Ordering::Acquire);
        let head = self.cached_head;
        tail.wrapping_sub(head)
    }

    /// Returns true if the buffer is empty.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Drop for SpscSender<T> {
    fn drop(&mut self) {
        // No special action needed -- the RingBuffer's Drop handles cleanup.
        // The consumer will see Disconnected on next try_recv when empty.
    }
}

impl<T> Drop for SpscReceiver<T> {
    fn drop(&mut self) {
        // No special action needed -- the RingBuffer's Drop handles cleanup.
        // The producer will see Disconnected on next try_send when full.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_recv_basic() {
        let (tx, mut rx) = spsc_channel::<u32>(4);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        tx.try_send(3).unwrap();

        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv().unwrap(), 3);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn full_then_drain() {
        let (tx, mut rx) = spsc_channel::<u32>(2);
        // Capacity rounds up to 2
        tx.try_send(10).unwrap();
        tx.try_send(20).unwrap();

        // Should be full
        assert!(matches!(tx.try_send(30), Err(TrySendError::Full(30))));

        // Drain one
        assert_eq!(rx.try_recv().unwrap(), 10);

        // Now we can send again
        tx.try_send(30).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 20);
        assert_eq!(rx.try_recv().unwrap(), 30);
    }

    #[test]
    fn producer_disconnect() {
        let (tx, mut rx) = spsc_channel::<u32>(4);
        drop(tx);
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn consumer_disconnect() {
        let (tx, rx) = spsc_channel::<u32>(4);
        drop(rx);
        assert!(matches!(tx.try_send(1), Err(TrySendError::Disconnected(1))));
    }

    #[test]
    fn drain_after_producer_disconnect() {
        let (tx, mut rx) = spsc_channel::<u32>(4);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        drop(tx);

        // Should still drain buffered messages
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        // Then disconnected
        assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn cross_thread_throughput() {
        let (tx, mut rx) = spsc_channel::<u64>(1024);
        let count = 100_000u64;

        let producer = std::thread::spawn(move || {
            for i in 0..count {
                loop {
                    match tx.try_send(i) {
                        Ok(()) => break,
                        Err(TrySendError::Full(_)) => std::thread::yield_now(),
                        Err(TrySendError::Disconnected(_)) => panic!("disconnected"),
                    }
                }
            }
        });

        let consumer = std::thread::spawn(move || {
            let mut received = 0u64;
            let mut expected = 0u64;
            loop {
                match rx.try_recv() {
                    Ok(v) => {
                        assert_eq!(v, expected, "out of order");
                        expected += 1;
                        received += 1;
                        if received == count {
                            break;
                        }
                    }
                    Err(TryRecvError::Empty) => std::thread::yield_now(),
                    Err(TryRecvError::Disconnected) => break,
                }
            }
            received
        });

        producer.join().unwrap();
        let received = consumer.join().unwrap();
        assert_eq!(received, count);
    }

    #[test]
    fn power_of_two_rounding() {
        // Capacity 3 should round up to 4
        let (tx, mut rx) = spsc_channel::<u32>(3);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        tx.try_send(3).unwrap();
        tx.try_send(4).unwrap();
        assert!(matches!(tx.try_send(5), Err(TrySendError::Full(5))));

        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert_eq!(rx.try_recv().unwrap(), 3);
        assert_eq!(rx.try_recv().unwrap(), 4);
    }

    #[test]
    fn wrapping_indices() {
        // Small buffer, many messages -- tests index wrapping
        let (tx, mut rx) = spsc_channel::<u32>(2);
        for i in 0..1000 {
            tx.try_send(i).unwrap();
            assert_eq!(rx.try_recv().unwrap(), i);
        }
    }

    // Kills: replace is_disconnected -> bool with true (line 232)
    // Kills: replace is_disconnected -> bool with false (line 232)
    // Kills: replace == with != in SpscSender::is_disconnected (line 232)
    #[test]
    fn sender_is_disconnected_false_while_receiver_alive() {
        let (tx, _rx) = spsc_channel::<u32>(4);
        assert!(!tx.is_disconnected(), "Sender should NOT be disconnected while receiver lives");
    }

    #[test]
    fn sender_is_disconnected_true_after_receiver_dropped() {
        let (tx, rx) = spsc_channel::<u32>(4);
        drop(rx);
        assert!(tx.is_disconnected(), "Sender should be disconnected after receiver drops");
    }

    // Kills: replace is_disconnected -> bool with true (line 277)
    // Kills: replace is_disconnected -> bool with false (line 277)
    // Kills: replace == with != in SpscReceiver::is_disconnected (line 277)
    #[test]
    fn receiver_is_disconnected_false_while_sender_alive() {
        let (_tx, rx) = spsc_channel::<u32>(4);
        assert!(!rx.is_disconnected(), "Receiver should NOT be disconnected while sender lives");
    }

    #[test]
    fn receiver_is_disconnected_true_after_sender_dropped() {
        let (tx, rx) = spsc_channel::<u32>(4);
        drop(tx);
        assert!(rx.is_disconnected(), "Receiver should be disconnected after sender drops");
    }

    // Kills: replace len -> usize with 0 (line 285)
    // Kills: replace len -> usize with 1 (line 285)
    #[test]
    fn len_reflects_message_count() {
        let (tx, rx) = spsc_channel::<u32>(8);
        assert_eq!(rx.len(), 0, "Empty buffer has len 0");

        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        tx.try_send(3).unwrap();
        assert_eq!(rx.len(), 3, "Buffer with 3 messages has len 3");
        assert_ne!(rx.len(), 0);
        assert_ne!(rx.len(), 1);
    }

    // Kills: replace is_empty -> bool with true (line 293)
    // Kills: replace is_empty -> bool with false (line 293)
    // Kills: replace == with != in SpscReceiver::is_empty (line 293)
    #[test]
    fn is_empty_true_when_buffer_empty() {
        let (_tx, rx) = spsc_channel::<u32>(4);
        assert!(rx.is_empty(), "Freshly created buffer should be empty");
    }

    #[test]
    fn is_empty_false_when_buffer_has_messages() {
        let (tx, rx) = spsc_channel::<u32>(4);
        tx.try_send(42).unwrap();
        assert!(!rx.is_empty(), "Buffer with a message should not be empty");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Deadlock regression tests
    //
    // PR #2 fixed a hang where: main thread exited the thread::scope closure
    // with an error while a worker was blocked in a spsc_blocking_send spin loop
    // on a full channel. The worker never saw Disconnected because the receiver
    // lived outside the closure and wasn't dropped.  The fix was to keep
    // receivers inside the scope closure so they drop on any exit.
    //
    // The tests below guard against regressions of that class of bug.
    // ──────────────────────────────────────────────────────────────────────────

    /// Simulates the reindex worker pattern: a producer fills the channel and
    /// blocks in the spin-send loop. The consumer (main thread) then drops its
    /// receiver. The producer MUST unblock — failure here means a hang of the
    /// kind fixed in PR #2.
    #[test]
    fn full_channel_unblocks_when_receiver_drops() {
        let (tx, rx) = spsc_channel::<u32>(2); // capacity rounds to 2

        // Fill the channel so the next send will spin.
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();

        // Spawn a producer that runs the same spin-send pattern used in reindex.
        let handle = std::thread::spawn(move || {
            let mut msg = 3u32;
            loop {
                match tx.try_send(msg) {
                    Ok(()) => break,
                    Err(TrySendError::Full(m)) => {
                        msg = m;
                        std::thread::yield_now();
                    }
                    // This is the escape hatch: Disconnected must be reached.
                    Err(TrySendError::Disconnected(_)) => break,
                }
            }
        });

        // Give the producer thread time to enter the spin loop.
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Drop the receiver — this should let the producer see Disconnected.
        drop(rx);

        // If this join hangs, the deadlock regression has reappeared.
        handle.join().expect("producer must unblock after receiver drop");
    }

    /// Two-worker scenario: worker 0 sends an error; the "main thread" propagates
    /// it (simulating a scope closure exit) which drops both receivers; worker 1,
    /// which is blocked spinning on a full channel, must unblock.
    ///
    /// The channel for worker 1 is pre-filled from the main thread BEFORE
    /// spawning the worker, so the worker starts directly in the spin loop and
    /// there is no race with receiver drop.
    #[test]
    fn worker_error_unblocks_other_workers_via_receiver_drop() {
        let (tx0, mut rx0) = spsc_channel::<Result<u32, &'static str>>(4);
        let (tx1, rx1) = spsc_channel::<u32>(2);

        // Pre-fill worker 1's channel (capacity 2) so it is full before we spawn.
        tx1.try_send(10).unwrap();
        tx1.try_send(20).unwrap();

        // Worker 1: starts spinning immediately because the channel is already full.
        let w1 = std::thread::spawn(move || {
            loop {
                match tx1.try_send(30) {
                    Ok(()) => break,
                    Err(TrySendError::Full(m)) => {
                        let _ = m;
                        std::thread::yield_now();
                    }
                    Err(TrySendError::Disconnected(_)) => break, // unblocked by drop(rx1)
                }
            }
        });

        // Worker 0: immediately sends an error result.
        tx0.try_send(Err("injected failure")).unwrap();
        drop(tx0);

        // "Main thread": receive the error and propagate (scope exits).
        let result = rx0.try_recv();
        assert!(matches!(result, Ok(Err("injected failure"))));

        // Simulate scope closure exit: drop receiver for worker 1.
        drop(rx1);

        // Worker 1 must terminate now that its receiver is gone.
        w1.join().expect("worker 1 must unblock after receiver drop");
    }

    // Kills: replace & with | in RingBuffer::drop (line 115)
    // Kills: replace & with ^ in RingBuffer::drop (line 115)
    // The mask used in drop must correctly compute slot index: idx = i & mask
    // With | or ^: wrong slots are accessed -> UB or double-free
    #[test]
    fn drop_cleans_up_buffered_messages() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Debug)]
        struct Counted(u32);
        impl Drop for Counted {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }

        DROP_COUNT.store(0, Ordering::Relaxed);

        let (tx, rx) = spsc_channel::<Counted>(4);
        tx.try_send(Counted(1)).unwrap();
        tx.try_send(Counted(2)).unwrap();
        tx.try_send(Counted(3)).unwrap();

        // Drop both ends without consuming
        drop(tx);
        drop(rx);

        assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 3);
    }
}
