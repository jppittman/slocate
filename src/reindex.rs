use crate::config::Config;
use crate::vdb::{self, Hnsw};
use crate::{embed, leiden, parse, registry, store};

/// Drop guard that sets a cancellation flag when it is dropped.
///
/// Used inside the `thread::scope` closure so that the cancel flag is set on
/// any exit (normal return, `?` propagation, or panic). Workers check this
/// flag before stealing new work batches, preventing wasteful embed work after
/// the main thread has errored.
struct CancelGuard(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// FNV-1a 64-bit hash of text. Stable across process restarts and Rust versions,
/// which is required for embed-cache keys to survive between reindex runs.
///
/// `std::collections::hash_map::DefaultHasher` is deliberately non-deterministic
/// (randomised per-process since Rust 1.36) and must NOT be used for cache keys.
fn content_hash(text: &str) -> String {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", h)
}

/// flock()-based reindex guard. The kernel releases the lock when the fd closes,
/// which happens on normal exit, panic, or kill -9. Lock file lives on tmpfs
/// (XDG_RUNTIME_DIR on Linux, TMPDIR on macOS) so it's cleaned on reboot.
struct ReindexLock {
    _file: std::fs::File,
}

impl ReindexLock {
    fn acquire(workspace_root: &std::path::Path) -> crate::error::Result<Self> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        workspace_root.hash(&mut h);
        let hash = format!("{:016x}", h.finish());

        let lock_dir = runtime_dir().join("slocate");
        std::fs::create_dir_all(&lock_dir)?;

        let lock_path = lock_dir.join(format!("{hash}.lock"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;

        if !try_flock_exclusive(&file) {
            return Err(crate::error::Error::Config(format!(
                "another reindex is already running for {}",
                workspace_root.display()
            )));
        }

        Ok(Self { _file: file })
    }
}

/// Non-blocking exclusive flock. Returns true if acquired.
fn try_flock_exclusive(file: &std::fs::File) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        extern "C" {
            fn flock(fd: i32, operation: i32) -> i32;
        }
        const LOCK_EX: i32 = 2;
        const LOCK_NB: i32 = 4;
        unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) == 0 }
    }
    #[cfg(not(unix))]
    {
        let _ = file;
        true // No locking on non-unix; best-effort.
    }
}

/// Per-user runtime directory on tmpfs.
/// Linux: $XDG_RUNTIME_DIR (/run/user/<uid>)
/// macOS: $TMPDIR (/var/folders/xx/.../T/)
/// Fallback: /tmp
fn runtime_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("XDG_RUNTIME_DIR") {
        return std::path::PathBuf::from(d);
    }
    if let Ok(d) = std::env::var("TMPDIR") {
        return std::path::PathBuf::from(d);
    }
    std::path::PathBuf::from("/tmp")
}

pub fn reindex_workspace(
    embedder: &embed::Embedder,
    config: &Config,
    workspace_root: &std::path::Path,
) -> crate::error::Result<()> {
    let _lock = ReindexLock::acquire(workspace_root)?;
    let index_dir = registry::index_dir(workspace_root)?;
    let mut db = store::Store::open(&index_dir)?;
    db.ensure_file_meta_table()?;

    let old_meta = db.load_file_meta()?;

    // ── Phase 1: walk + classify ─────────────────────────────────────────────
    let (to_embed, unchanged_paths, deleted_paths) =
        classify_files(config, workspace_root, &old_meta)?;

    let new_count = to_embed.len();
    let unchanged_count = unchanged_paths.len();
    let deleted_count = deleted_paths.len();

    if new_count == 0 && deleted_count == 0 {
        eprintln!(
            "[reindex] {}: {} files unchanged, nothing to do",
            workspace_root.display(),
            unchanged_count
        );
        return Ok(());
    }

    eprintln!(
        "[reindex] {}: {} changed/new, {} unchanged, {} deleted",
        workspace_root.display(),
        new_count,
        unchanged_count,
        deleted_count
    );

    // ── Phase 2: remove deleted files ────────────────────────────────────────
    if !deleted_paths.is_empty() {
        db.remove_files(&deleted_paths)?;
    }

    // ── Phase 3+4: work-stealing parallel embed + incremental commit ────────
    //
    // N worker threads steal batches from a shared work queue (atomic counter).
    // Each worker: grab batch → check embed cache → embed misses → send results.
    // Main thread: receive batches → commit chunks + cache entries to SQLite.
    // GPU forces 1 worker (shared resource). CPU gets min(parallelism, 4).
    const COMMIT_BATCH: usize = 50;

    let n_workers = if embedder.is_gpu() {
        1
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(4)
    };

    // Pre-split work into batches for stealing.
    let batches: Vec<&[(String, Vec<parse::RawChunk>, store::FileMeta)]> =
        to_embed.chunks(COMMIT_BATCH).collect();

    struct BatchResult {
        embedded: Vec<(store::Chunk, Vec<f32>)>,
        meta: Vec<(String, store::FileMeta)>,
        /// New (hash, vector) pairs to write back to the embed cache.
        new_cache: Vec<(String, Vec<f32>)>,
        cache_hits: usize,
    }

    let mut all_embedded: Vec<(store::Chunk, Vec<f32>)> = Vec::new();
    let mut committed_files = 0usize;
    let mut committed_chunks = 0usize;
    let mut total_cache_hits = 0usize;

    if batches.is_empty() {
        // Nothing to embed — skip to HNSW phase.
    } else {
        let work_idx = std::sync::atomic::AtomicUsize::new(0);

        // Cancellation flag: set by CancelGuard when the scope closure exits for
        // any reason (success, error, or panic). Workers check this before stealing
        // the next batch, preventing them from starting unnecessary embed work after
        // the main thread has errored. The receiver-drop is still the primary
        // unblocking mechanism for workers already in spsc_blocking_send.
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        eprintln!("[reindex] embedding with {n_workers} worker(s), {} batch(es)", batches.len());

        // Scoped threads: workers steal batches, main thread commits.
        // Each worker opens its own read-only SQLite connection for cache
        // lookups. WAL mode allows concurrent readers + one writer (main thread).
        //
        // IMPORTANT: receivers must be created INSIDE the scope closure so they
        // are dropped when the closure returns (even on early return via `?`).
        // If receivers lived outside the closure, a worker error would cause the
        // closure to exit while other workers are blocked in spsc_blocking_send
        // on a full channel — they would never see Disconnected, and
        // thread::scope would hang forever waiting for them to finish.
        let commit_err: crate::error::Result<()> = std::thread::scope(|s| {
            // CancelGuard: fires on any exit from this closure (?, panic, normal).
            // This signals workers to stop stealing new batches, capping wasted
            // embed work after the main thread has errored.
            let _cancel_guard = CancelGuard(std::sync::Arc::clone(&cancel));

            // Per-worker SPSC channels: zero contention on send, each worker's
            // hot path is completely independent. Main thread polls all receivers.
            // Receivers are declared here so they drop when this closure returns,
            // unblocking any worker stuck in spsc_blocking_send.
            let mut receivers: Vec<crate::spsc::SpscReceiver<crate::error::Result<BatchResult>>> =
                Vec::with_capacity(n_workers);

            // Spawn N embed workers, each with its own SPSC sender.
            for worker_id in 0..n_workers {
                let (tx, rx) = crate::spsc::spsc_channel(4);
                receivers.push(rx);
                let work_idx = &work_idx;
                let batches = &batches;
                let index_dir = &index_dir;
                let cancel = std::sync::Arc::clone(&cancel);
                s.spawn(move || {
                    set_background_qos();

                    // Each worker gets its own read-only DB connection for cache.
                    let cache_db = match store::Store::open(index_dir) {
                        Ok(db) => db,
                        Err(e) => {
                            spsc_blocking_send(&tx, Err(e));
                            return;
                        }
                    };

                    loop {
                        // Check cancellation before stealing next batch. The main
                        // thread sets this via CancelGuard on any scope exit.
                        if cancel.load(std::sync::atomic::Ordering::Acquire) {
                            break;
                        }
                        let idx = work_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if idx >= batches.len() {
                            break;
                        }
                        let group = batches[idx];

                        // Build embed inputs for this batch.
                        let mut embed_inputs: Vec<String> = Vec::new();
                        let mut chunk_meta: Vec<(parse::RawChunk, String, store::FileMeta)> = Vec::new();

                        for (rel_path, raw_chunks, file_meta) in group {
                            eprintln!("[worker {worker_id}] {rel_path}: {} chunks", raw_chunks.len());
                            for raw in raw_chunks {
                                embed_inputs.push(format!(
                                    "{} {} in {}\n\n{}",
                                    raw.kind, raw.name, rel_path, raw.embed_text
                                ));
                                chunk_meta.push((raw.clone(), rel_path.clone(), *file_meta));
                            }
                        }

                        if embed_inputs.is_empty() {
                            continue;
                        }

                        // Check embed cache: split into hits and misses.
                        let hashes: Vec<String> = embed_inputs.iter()
                            .map(|t| content_hash(t))
                            .collect();

                        let cached = match cache_db.cache_get_batch(&hashes) {
                            Ok(c) => c,
                            Err(e) => {
                                eprintln!("[worker {worker_id}] cache read failed, embedding all: {e}");
                                std::collections::HashMap::new()
                            }
                        };

                        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; embed_inputs.len()];
                        let mut miss_indices: Vec<usize> = Vec::new();
                        let mut miss_texts: Vec<String> = Vec::new();
                        let mut cache_hits = 0usize;

                        for (i, hash) in hashes.iter().enumerate() {
                            if let Some(vec) = cached.get(hash) {
                                vectors[i] = Some(vec.clone());
                                cache_hits += 1;
                            } else {
                                miss_indices.push(i);
                                miss_texts.push(embed_inputs[i].clone());
                            }
                        }

                        // Embed only the cache misses.
                        let mut new_cache: Vec<(String, Vec<f32>)> = Vec::new();
                        if !miss_texts.is_empty() {
                            let miss_vectors = match embedder.embed_batch(&miss_texts) {
                                Ok(v) => v,
                                Err(e) => {
                                    spsc_blocking_send(&tx, Err(crate::error::Error::Embed(format!(
                                        "worker {worker_id} embed failed: {e}"
                                    ))));
                                    return;
                                }
                            };

                            if miss_vectors.len() != miss_indices.len() {
                                spsc_blocking_send(&tx, Err(crate::error::Error::Embed(format!(
                                    "worker {worker_id}: embed_batch returned {} vectors, \
                                     expected {}",
                                    miss_vectors.len(), miss_indices.len()
                                ))));
                                return;
                            }

                            for (j, vec) in miss_indices.into_iter().zip(miss_vectors) {
                                new_cache.push((hashes[j].clone(), vec.clone()));
                                vectors[j] = Some(vec);
                            }
                        }

                        // Build Chunk structs + file meta.
                        let mut embedded = Vec::with_capacity(vectors.len());
                        let mut meta: Vec<(String, store::FileMeta)> = Vec::new();
                        let mut seen_paths = std::collections::HashSet::new();

                        for ((raw, rel_path, file_meta), vector) in
                            chunk_meta.into_iter().zip(vectors)
                        {
                            let vec = vector.expect(
                                "BUG: vector slot unfilled after cache + embed"
                            );
                            let id = store::chunk_id(&rel_path, &raw.name, raw.kind.as_str());
                            if seen_paths.insert(rel_path.clone()) {
                                meta.push((rel_path.clone(), file_meta));
                            }
                            embedded.push((
                                store::Chunk {
                                    id,
                                    kind: raw.kind,
                                    name: raw.name,
                                    source_path: rel_path,
                                    source: raw.source,
                                    community_id: None,
                                },
                                vec,
                            ));
                        }

                        spsc_blocking_send(&tx, Ok(BatchResult { embedded, meta, new_cache, cache_hits }));
                    }
                    // tx drops here → receiver sees Disconnected after draining.
                });
            }

            // Main thread: poll all per-worker receivers, commit to SQLite.
            let mut active = vec![true; n_workers];
            loop {
                let mut any_active = false;
                for (i, rx) in receivers.iter_mut().enumerate() {
                    if !active[i] {
                        continue;
                    }
                    match rx.try_recv() {
                        Ok(result) => {
                            any_active = true;
                            let batch = result?;

                            db.begin()?;
                            // Execute all batch writes inside the transaction.
                            // On failure, explicitly roll back before propagating
                            // the error so the connection is left in a clean state.
                            let write_result = (|| -> crate::error::Result<()> {
                                for (path, _) in &batch.meta {
                                    db.remove_chunks_for_file(path)?;
                                }
                                for (chunk, _) in &batch.embedded {
                                    db.insert_chunk(chunk)?;
                                }
                                db.update_file_meta(&batch.meta)?;
                                if !batch.new_cache.is_empty() {
                                    db.cache_put(&batch.new_cache)?;
                                }
                                Ok(())
                            })();
                            match write_result {
                                Ok(()) => db.commit()?,
                                Err(e) => {
                                    let _ = db.rollback(); // best-effort; don't mask original error
                                    return Err(e);
                                }
                            }

                            committed_files += batch.meta.len();
                            committed_chunks += batch.embedded.len();
                            total_cache_hits += batch.cache_hits;
                            eprintln!(
                                "[reindex] committed {committed_files}/{new_count} files \
                                 ({committed_chunks} chunks, {total_cache_hits} cache hits)"
                            );

                            all_embedded.extend(batch.embedded);
                        }
                        Err(crate::spsc::TryRecvError::Empty) => {
                            any_active = true;
                        }
                        Err(crate::spsc::TryRecvError::Disconnected) => {
                            active[i] = false;
                        }
                    }
                }
                if !any_active {
                    break;
                }
                std::thread::yield_now();
            }

            Ok(())
        });

        commit_err?;
    }

    let new_embedded = all_embedded;

    // ── Phase 5: update HNSW ────────────────────────────────────────────────
    let mut old_hnsw = db.load_hnsw(vdb::dot)?;

    if new_embedded.is_empty() && deleted_count == 0 {
        // Nothing changed at all — shouldn't reach here (early return above), but be safe.
    } else if deleted_count == 0 && !new_embedded.is_empty() && !old_hnsw.is_empty() {
        // Fast path: no deletions → keep old graph, remove stale nodes for modified
        // files, then insert new/modified vectors incrementally.
        let modified_ids: std::collections::HashSet<String> =
            new_embedded.iter().map(|(c, _)| c.id.clone()).collect();
        old_hnsw.remove_ids(&modified_ids);

        for (chunk, vector) in &new_embedded {
            old_hnsw.insert(&chunk.id, vector.clone());
        }
    } else {
        // Slow path: deletions happened → rebuild from scratch to avoid stale nodes.
        let mut fresh = Hnsw::new(vdb::dot);

        // Collect IDs of deleted/modified chunks to skip.
        let modified_ids: std::collections::HashSet<String> =
            new_embedded.iter().map(|(c, _)| c.id.clone()).collect();
        let deleted_path_set: std::collections::HashSet<&str> =
            deleted_paths.iter().map(|s| s.as_str()).collect();

        // Re-insert surviving nodes. Use chunk DB to check path membership for deleted files.
        for node in &old_hnsw.nodes {
            if modified_ids.contains(&node.id) {
                continue;
            }
            // Check if this node's chunk was from a deleted file.
            let chunks = db.get_chunks_by_ids(&[node.id.clone()]).unwrap_or_default();
            let is_deleted = chunks.first()
                .map(|c| deleted_path_set.contains(c.source_path.as_str()))
                .unwrap_or(true); // missing chunk = stale, skip
            if !is_deleted {
                fresh.insert(&node.id, node.vector.clone());
            }
        }

        for (chunk, vector) in &new_embedded {
            fresh.insert(&chunk.id, vector.clone());
        }
        old_hnsw = fresh;
    }

    let hnsw_ids: Vec<String> = old_hnsw.nodes.iter().map(|n| n.id.clone()).collect();

    let total_chunks = hnsw_ids.len();

    // ── Phase 6: Leiden community detection ──────────────────────────────────
    let partition = leiden::run(&old_hnsw, leiden::DEFAULT_GAMMA);
    let community_assignments: Vec<(String, usize)> = hnsw_ids
        .iter()
        .zip(partition.assignment.iter())
        .map(|(id, &cid)| (id.clone(), cid))
        .collect();
    db.set_community_ids(&community_assignments)?;
    db.save_hnsw(&old_hnsw)?;

    let total_files = unchanged_count + committed_files;
    eprintln!(
        "[reindex] {}: {total_files} files ({new_count} re-embedded, {deleted_count} deleted), \
         {total_chunks} chunks, {} communities",
        workspace_root.display(),
        partition.n_communities,
    );
    Ok(())
}

fn classify_files(
    config: &Config,
    workspace_root: &std::path::Path,
    old_meta: &std::collections::HashMap<String, store::FileMeta>,
) -> crate::error::Result<(
    Vec<(String, Vec<parse::RawChunk>, store::FileMeta)>,  // (rel_path, chunks, meta)
    Vec<String>,                                            // unchanged rel_paths
    Vec<String>,                                            // deleted rel_paths
)> {
    let all_files = collect_file_paths(config, workspace_root)?;

    let mut to_embed = Vec::new();
    let mut unchanged = Vec::new();
    let mut current_paths = std::collections::HashSet::new();

    for (rel_path, abs_path) in &all_files {
        current_paths.insert(rel_path.clone());

        let fs_meta = std::fs::metadata(abs_path)?;

        let mtime_ns = fs_meta
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let size = fs_meta.len() as i64;
        let file_meta = store::FileMeta { mtime_ns, size };

        // Check if file is unchanged.
        if let Some(old) = old_meta.get(rel_path) {
            if old.mtime_ns == mtime_ns && old.size == size {
                unchanged.push(rel_path.clone());
                continue;
            }
        }

        // File is new or modified — parse it.
        let source = match std::fs::read_to_string(abs_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping {}: {e}", abs_path.display());
                continue;
            }
        };

        let raw_chunks = parse::parse_file(abs_path, &source);
        if raw_chunks.is_empty() {
            // File has no parseable chunks but still exists — track it so we don't
            // re-read it every time.
            unchanged.push(rel_path.clone());
            continue;
        }

        to_embed.push((rel_path.clone(), raw_chunks, file_meta));
    }

    // Files in old_meta but not on disk anymore → deleted.
    let deleted: Vec<String> = old_meta
        .keys()
        .filter(|k| !current_paths.contains(k.as_str()))
        .cloned()
        .collect();

    Ok((to_embed, unchanged, deleted))
}

fn collect_file_paths(
    config: &Config,
    workspace_root: &std::path::Path,
) -> crate::error::Result<Vec<(String, std::path::PathBuf)>> {
    let skip_dirs: &[&str] = &[
        "target", ".git", ".slocate", "vendor", "node_modules",
        // OS/platform junk — symlink cycles, caches, irrelevant data.
        "Library", "Applications", ".Trash",
    ];
    let extensions: Vec<&str> = config.index.extensions.iter().map(|s| s.as_str()).collect();

    let mut files = Vec::new();
    let mut stack = vec![workspace_root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[walk] skipping {}: {e}", dir.display());
                continue;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("[walk] skipping entry in {}: {e}", dir.display());
                    continue;
                }
            };
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();

            // Skip symlinks to avoid cycles.
            if path.symlink_metadata().map(|m| m.file_type().is_symlink()).unwrap_or(false) {
                continue;
            }

            if path.is_dir() {
                if file_name.starts_with('.') || skip_dirs.contains(&file_name) {
                    continue;
                }
                stack.push(path);
                continue;
            }

            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
            if !extensions.contains(&ext) {
                continue;
            }

            if is_binary_magic(&path) {
                continue;
            }

            match std::fs::metadata(&path) {
                Ok(m) if m.len() > config.index.max_file_bytes => continue,
                Err(_) => continue,
                Ok(_) => {}
            }

            let rel_path = path
                .strip_prefix(workspace_root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            files.push((rel_path, path));
        }
    }

    Ok(files)
}

/// Blocking send on an SPSC channel. Spins with yield on Full.
/// Silently returns if the receiver is dropped (main thread hit an error).
pub(crate) fn spsc_blocking_send<T>(tx: &crate::spsc::SpscSender<T>, mut msg: T) {
    loop {
        match tx.try_send(msg) {
            Ok(()) => return,
            Err(crate::spsc::TrySendError::Full(m)) => {
                msg = m;
                std::thread::yield_now();
            }
            Err(crate::spsc::TrySendError::Disconnected(_)) => return,
        }
    }
}

/// Lower the calling thread to background QoS so reindex doesn't compete
/// with interactive work. macOS: QOS_CLASS_BACKGROUND via pthread.
/// Linux: IOPRIO_CLASS_IDLE + nice 19. No-op on other platforms.
fn set_background_qos() {
    #[cfg(target_os = "macos")]
    {
        // QOS_CLASS_BACKGROUND = 0x09
        // https://developer.apple.com/documentation/dispatch/qos_class_t
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        let ret = unsafe { pthread_set_qos_class_self_np(0x09, 0) };
        if ret != 0 {
            eprintln!("[reindex] warning: failed to set background QoS (errno {ret})");
        }
    }

    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn nice(inc: i32) -> i32;
            fn syscall(num: std::ffi::c_long, ...) -> std::ffi::c_long;
        }

        // nice 19 = lowest CPU scheduling priority.
        unsafe { nice(19); }

        // ioprio_set(IOPRIO_WHO_PROCESS, 0 /*self*/, IOPRIO_CLASS_IDLE << 13)
        // https://man7.org/linux/man-pages/man2/ioprio_set.2.html
        #[cfg(target_arch = "x86_64")]
        const SYS_IOPRIO_SET: std::ffi::c_long = 251;
        #[cfg(target_arch = "aarch64")]
        const SYS_IOPRIO_SET: std::ffi::c_long = 30;
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            const IOPRIO_WHO_PROCESS: std::ffi::c_long = 1;
            const IOPRIO_PRIO_VALUE: std::ffi::c_long = 3 << 13; // class=IDLE, data=0
            unsafe { syscall(SYS_IOPRIO_SET, IOPRIO_WHO_PROCESS, 0 as std::ffi::c_long, IOPRIO_PRIO_VALUE); }
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        // No-op on unsupported platforms.
    }
}

fn is_binary_magic(path: &std::path::Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 4];
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let n = f.read(&mut buf).unwrap_or(0);
    if n < 4 {
        return false;
    }
    matches!(
        buf,
        [0x7f, b'E', b'L', b'F']       // ELF (Linux executables, .so)
        | [0xFE, 0xED, 0xFA, 0xCE]     // Mach-O 32-bit BE
        | [0xCE, 0xFA, 0xED, 0xFE]     // Mach-O 32-bit LE
        | [0xFE, 0xED, 0xFA, 0xCF]     // Mach-O 64-bit BE
        | [0xCF, 0xFA, 0xED, 0xFE]     // Mach-O 64-bit LE
        | [0xCA, 0xFE, 0xBA, 0xBE]     // Mach-O fat binary / universal
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── content_hash ─────────────────────────────────────────────────────────

    #[test]
    fn content_hash_is_deterministic() {
        // Must produce the same value across calls (no randomness).
        let h1 = content_hash("fn foo() { 42 }");
        let h2 = content_hash("fn foo() { 42 }");
        assert_eq!(h1, h2, "content_hash must be deterministic");
    }

    #[test]
    fn content_hash_is_sensitive_to_input() {
        // Different inputs → different hashes.
        let h1 = content_hash("fn foo() {}");
        let h2 = content_hash("fn bar() {}");
        assert_ne!(h1, h2, "content_hash must distinguish different inputs");
    }

    #[test]
    fn content_hash_empty_string() {
        // Empty input should not panic and should produce a fixed value.
        let h = content_hash("");
        assert_eq!(h.len(), 16, "hash should be 16 hex chars");
        // FNV-1a of empty string is the offset basis: 14695981039346656037 = cbf29ce484222325
        assert_eq!(h, "cbf29ce484222325");
    }

    #[test]
    fn content_hash_known_value() {
        // Pin a known FNV-1a value so any accidental switch back to DefaultHasher
        // (which would produce a different, non-deterministic result) is caught.
        let h = content_hash("hello");
        assert_eq!(h, "a430d84680aabd0b", "FNV-1a(\"hello\") must be stable");
    }

    // ── spsc_blocking_send ───────────────────────────────────────────────────

    #[test]
    fn spsc_blocking_send_delivers_when_space_available() {
        let (tx, mut rx) = crate::spsc::spsc_channel::<u32>(4);
        spsc_blocking_send(&tx, 42);
        assert_eq!(rx.try_recv().unwrap(), 42);
    }

    #[test]
    fn spsc_blocking_send_returns_on_receiver_drop_while_full() {
        // This is the core deadlock regression test for the reindex pipeline.
        //
        // Scenario: a worker fills its SPSC channel and enters the spin-send
        // loop; the "main thread" then drops the receiver (simulating an early
        // scope exit due to a commit error). The worker MUST unblock.
        let (tx, rx) = crate::spsc::spsc_channel::<u32>(2);
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap(); // channel is now full

        let handle = std::thread::spawn(move || {
            // This will spin until rx is dropped.
            spsc_blocking_send(&tx, 3);
        });

        std::thread::sleep(std::time::Duration::from_millis(10));

        // Drop the receiver — simulates the scope closure exiting.
        drop(rx);

        // Must terminate in finite time. A hang here = deadlock regression.
        handle.join().expect("spsc_blocking_send must return when receiver drops");
    }

    // ── CancelGuard ──────────────────────────────────────────────────────────

    #[test]
    fn cancel_guard_sets_flag_on_drop() {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let _guard = CancelGuard(std::sync::Arc::clone(&flag));
            assert!(!flag.load(std::sync::atomic::Ordering::Acquire), "flag before drop");
        }
        assert!(flag.load(std::sync::atomic::Ordering::Acquire), "flag after drop");
    }

    #[test]
    fn cancel_guard_sets_flag_on_error_propagation() {
        // Simulates `?` exiting a closure that owns a CancelGuard.
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag_clone = std::sync::Arc::clone(&flag);

        let result: Result<(), &str> = (|| {
            let _guard = CancelGuard(std::sync::Arc::clone(&flag_clone));
            Err("simulated error")?;
            Ok(())
        })();

        assert!(result.is_err());
        assert!(flag.load(std::sync::atomic::Ordering::Acquire),
            "cancel flag must be set even when closure exits via ?");
    }

    // ── classify_files ───────────────────────────────────────────────────────

    #[test]
    fn classify_files_detects_new_files() {
        let dir = tempdir();
        std::fs::write(dir.join("main.rs"), "fn main() {}").unwrap();

        let config = crate::config::Config::default();
        let old_meta = std::collections::HashMap::new();
        let (to_embed, unchanged, deleted) =
            classify_files(&config, &dir, &old_meta).unwrap();

        assert!(!to_embed.is_empty(), "new .rs file must be classified for embedding");
        assert!(unchanged.is_empty());
        assert!(deleted.is_empty());
    }

    #[test]
    fn classify_files_detects_deleted_files() {
        let dir = tempdir();

        let config = crate::config::Config::default();
        let mut old_meta = std::collections::HashMap::new();
        old_meta.insert("gone.rs".to_string(), store::FileMeta { mtime_ns: 1, size: 1 });

        let (to_embed, unchanged, deleted) =
            classify_files(&config, &dir, &old_meta).unwrap();

        assert!(to_embed.is_empty());
        assert!(unchanged.is_empty());
        assert_eq!(deleted, vec!["gone.rs".to_string()],
            "file in old_meta but not on disk must be classified as deleted");
    }

    #[test]
    fn classify_files_unchanged_file_skipped() {
        let dir = tempdir();
        let file_path = dir.join("lib.rs");
        std::fs::write(&file_path, "fn foo() {}").unwrap();

        // Record the current mtime and size.
        let meta = std::fs::metadata(&file_path).unwrap();
        let mtime_ns = meta.modified().unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let size = meta.len() as i64;

        let config = crate::config::Config::default();
        let mut old_meta = std::collections::HashMap::new();
        old_meta.insert("lib.rs".to_string(), store::FileMeta { mtime_ns, size });

        let (to_embed, unchanged, deleted) =
            classify_files(&config, &dir, &old_meta).unwrap();

        assert!(to_embed.is_empty(), "unchanged file must not be re-embedded");
        assert!(!unchanged.is_empty(), "unchanged file must be in unchanged list");
        assert!(deleted.is_empty());
    }

    #[test]
    fn classify_files_ignores_target_directory() {
        let dir = tempdir();
        let target = dir.join("target");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("artifact.rs"), "fn ignored() {}").unwrap();

        let config = crate::config::Config::default();
        let old_meta = std::collections::HashMap::new();
        let (to_embed, _, _) = classify_files(&config, &dir, &old_meta).unwrap();

        assert!(to_embed.iter().all(|(p, _, _)| !p.contains("target")),
            "files inside target/ must be ignored");
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "slocate_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
