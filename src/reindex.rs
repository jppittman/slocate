use crate::config::Config;
use crate::embed::Embedder;
use crate::store::Db;
use crate::{parse, registry, store};

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

/// Maximum chars per embedding window. BGE supports 512 tokens (~1800 chars of code).
const EMBED_WINDOW: usize = 1800;

/// Split `source` into non-overlapping windows of at most `EMBED_WINDOW` chars,
/// each prefixed with `prefix`. Always returns at least one window.
fn split_windows(source: &str, prefix: &str) -> Vec<String> {
    if source.len() <= EMBED_WINDOW {
        return vec![format!("{prefix}{source}")];
    }
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < source.len() {
        let raw_end = (pos + EMBED_WINDOW).min(source.len());
        let end = (pos..=raw_end)
            .rev()
            .find(|&i| source.is_char_boundary(i))
            .unwrap_or(raw_end);
        if end <= pos {
            break;
        }
        out.push(format!("{prefix}{}", &source[pos..end]));
        pos = end;
    }
    out
}

/// Log-weighted sum of unit vectors. Weight = log(1 + char_len) per window.
/// Result is NOT normalized — caller must call `l2_normalize`.
fn log_weighted_sum(vecs: &[Vec<f32>], texts: &[String]) -> Vec<f32> {
    assert!(
        !vecs.is_empty(),
        "log_weighted_sum requires at least one vector"
    );
    let dim = vecs[0].len();
    let mut out = vec![0f32; dim];
    for (v, t) in vecs.iter().zip(texts.iter()) {
        let w = (1.0 + t.len() as f32).ln();
        for (o, &vi) in out.iter_mut().zip(v.iter()) {
            *o += w * vi;
        }
    }
    out
}

/// L2-normalize a vector in place. No-op for near-zero vectors.
fn l2_normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
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

/// Accumulates per-chunk metadata alongside its window range into the flat
/// `texts` / `hashes` / `vectors` arrays built during batch preparation.
struct ChunkItem {
    rel_path: String,
    name: String,
    kind: parse::ChunkKind,
    source: String,
    file_meta: store::FileMeta,
    /// Index of the first window for this chunk in the flat `texts`/`hashes`/`vectors` arrays.
    win_start: usize,
    /// Number of contiguous windows for this chunk (`win_start..win_start + win_count`).
    win_count: usize,
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
    embedder: &dyn Embedder,
    config: &Config,
    cache: &dyn store::CacheBackend,
    workspace_root: &std::path::Path,
    force: bool,
) -> crate::error::Result<()> {
    let _lock = ReindexLock::acquire(workspace_root)?;
    let index_dir = registry::index_dir(workspace_root)?;

    if force {
        log::info!(
            "{}: forcing full reindex — deleting index",
            workspace_root.display()
        );
        let db_path = index_dir.join("index.db");
        for ext in &["", "-shm", "-wal"] {
            let p = index_dir.join(format!("index.db{ext}"));
            if p.exists() {
                std::fs::remove_file(&p)?;
            }
        }
        drop(db_path); // suppress unused warning
    }

    let mut db = store::SqliteDb::open(&index_dir)?;
    db.ensure_file_meta_table()?;

    let old_meta = db.load_file_meta()?;

    // ── Phase 1: walk + classify ─────────────────────────────────────────────
    let (to_embed, unchanged_paths, deleted_paths) =
        classify_files(config, workspace_root, &old_meta)?;

    let new_count = to_embed.len();
    let unchanged_count = unchanged_paths.len();
    let deleted_count = deleted_paths.len();

    if new_count == 0 && deleted_count == 0 {
        log::info!(
            "{}: {} files unchanged, nothing to do",
            workspace_root.display(),
            unchanged_count
        );
        return Ok(());
    }

    log::info!(
        "{}: {} changed/new, {} unchanged, {} deleted",
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

    let mut committed_files = 0usize;
    let mut committed_chunks = 0usize;
    let mut total_cache_hits = 0usize;

    if batches.is_empty() {
        // Nothing to embed.
    } else {
        let work_idx = std::sync::atomic::AtomicUsize::new(0);

        // Cancellation flag: set by CancelGuard when the scope closure exits for
        // any reason (success, error, or panic). Workers check this before stealing
        // the next batch, preventing them from starting unnecessary embed work after
        // the main thread has errored. The receiver-drop is still the primary
        // unblocking mechanism for workers already in spsc_blocking_send.
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        log::info!(
            "embedding with {n_workers} worker(s), {} batch(es)",
            batches.len()
        );

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
            let _cancel_guard = CancelGuard(std::sync::Arc::clone(&cancel));

            let mut receivers: Vec<crate::spsc::SpscReceiver<crate::error::Result<BatchResult>>> =
                Vec::with_capacity(n_workers);

            for worker_id in 0..n_workers {
                let (tx, rx) = crate::spsc::spsc_channel::<crate::error::Result<BatchResult>>(4);
                receivers.push(rx);
                let work_idx_ref = &work_idx;
                let batches_ref = &batches;
                let cancel_ref = &cancel;

                // Create worker-local cache connection before spawning —
                // CacheBackend is Send but not necessarily Sync, so we
                // can't share &cache across threads.
                let worker_cache = match cache.open_new() {
                    Ok(c) => c,
                    Err(e) => {
                        // Can't spawn if cache open fails. Send error and skip.
                        spsc_blocking_send(
                            &tx,
                            Err(crate::error::Error::Embed(format!(
                                "worker {worker_id} cache open failed: {e}"
                            ))),
                        );
                        continue;
                    }
                };

                s.spawn(move || {
                    set_background_qos();

                    loop {
                        if cancel_ref.load(std::sync::atomic::Ordering::Acquire) {
                            break;
                        }
                        let idx = work_idx_ref.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                        if idx >= batches_ref.len() {
                            break;
                        }

                        let batch = batches_ref[idx];

                        // Split each chunk into windows and flatten for batch embedding.
                        // chunk_items tracks window ranges into the flat texts/hashes arrays.
                        let mut chunk_items: Vec<ChunkItem> = Vec::new();
                        let mut texts: Vec<String> = Vec::new();
                        let mut hashes: Vec<String> = Vec::new();

                        for (rel_path, raw_chunks, file_meta) in batch {
                            for rc in raw_chunks {
                                let windows = split_windows(&rc.source, "code: ");
                                let win_start = texts.len();
                                let win_count = windows.len();
                                for w in windows {
                                    hashes.push(content_hash(&w));
                                    texts.push(w);
                                }
                                chunk_items.push(ChunkItem {
                                    rel_path: rel_path.clone(),
                                    name: rc.name.clone(),
                                    kind: rc.kind,
                                    source: rc.source.clone(),
                                    file_meta: *file_meta,
                                    win_start,
                                    win_count,
                                });
                            }
                        }

                        // Batch cache lookup from shared embed cache.
                        let cached = worker_cache.get_batch(&hashes).unwrap_or_default();
                        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
                        let mut miss_indices = Vec::new();
                        let mut miss_texts = Vec::new();
                        let mut cache_hits = 0usize;

                        for (j, hash) in hashes.iter().enumerate() {
                            if let Some(v) = cached.get(hash) {
                                vectors[j] = Some(v.clone());
                                cache_hits += 1;
                            } else {
                                miss_indices.push(j);
                                miss_texts.push(texts[j].clone());
                            }
                        }

                        // Embed only the cache misses.
                        let mut new_cache: Vec<(String, Vec<f32>)> = Vec::new();
                        if !miss_texts.is_empty() {
                            let miss_vectors = match embedder.embed_batch(&miss_texts) {
                                Ok(v) => v,
                                Err(e) => {
                                    spsc_blocking_send(
                                        &tx,
                                        Err(crate::error::Error::Embed(format!(
                                            "worker {worker_id} embed failed: {e}"
                                        ))),
                                    );
                                    return;
                                }
                            };

                            if miss_vectors.len() != miss_indices.len() {
                                spsc_blocking_send(
                                    &tx,
                                    Err(crate::error::Error::Embed(format!(
                                        "worker {worker_id}: embed_batch returned {} vectors, \
                                     expected {}",
                                        miss_vectors.len(),
                                        miss_indices.len()
                                    ))),
                                );
                                return;
                            }

                            for (j, vec) in miss_indices.into_iter().zip(miss_vectors) {
                                new_cache.push((hashes[j].clone(), vec.clone()));
                                vectors[j] = Some(vec);
                            }
                        }

                        // Build Chunk structs: combine per-chunk windows into a centroid.
                        let mut embedded = Vec::with_capacity(chunk_items.len());
                        let mut meta: Vec<(String, store::FileMeta)> = Vec::new();
                        let mut seen_files = std::collections::HashSet::new();

                        for item in &chunk_items {
                            let win_end = item.win_start + item.win_count;

                            // Collect window vectors; skip chunk on any missing vector (BUG guard).
                            let win_vecs: Vec<Vec<f32>> = match (item.win_start..win_end)
                                .map(|j| vectors[j].clone())
                                .collect::<Option<Vec<_>>>()
                            {
                                Some(v) => v,
                                None => {
                                    log::error!(
                                        "worker {worker_id}: missing vector for a window of \
                                         chunk '{}' — BUG",
                                        item.name
                                    );
                                    continue;
                                }
                            };

                            let win_texts = &texts[item.win_start..win_end];
                            let vec = if win_vecs.len() == 1 {
                                win_vecs.into_iter().next().unwrap()
                            } else {
                                let mut c = log_weighted_sum(&win_vecs, win_texts);
                                l2_normalize(&mut c);
                                c
                            };

                            let id =
                                store::chunk_id(&item.rel_path, &item.name, item.kind.as_str());
                            if seen_files.insert(item.rel_path.clone()) {
                                meta.push((item.rel_path.clone(), item.file_meta));
                            }
                            embedded.push((
                                store::Chunk {
                                    id,
                                    kind: item.kind,
                                    name: item.name.clone(),
                                    source_path: item.rel_path.clone(),
                                    source: item.source.clone(),
                                },
                                vec,
                            ));
                        }

                        if !spsc_blocking_send(
                            &tx,
                            Ok(BatchResult {
                                embedded,
                                meta,
                                new_cache,
                                cache_hits,
                            }),
                        ) {
                            break; // Receiver dropped, stop immediately.
                        }
                    }
                    // tx drops here → receiver sees Disconnected after draining.
                });
            }

            // Main thread: poll all per-worker receivers, accumulate results,
            // and flush to SQLite in coalesced transactions. This amortises the
            // fsync cost of COMMIT across many batches instead of paying it per
            // BatchResult.
            const FLUSH_THRESHOLD: usize = 8; // flush after this many batches

            let mut active = vec![true; n_workers];
            let mut pending: Vec<BatchResult> = Vec::new();

            // Flush all pending batches to SQLite in a single transaction.
            // Chunks go to the per-workspace DB; new cache entries go to the
            // shared embed cache (separate SQLite, idempotent writes).
            let flush = |db: &mut dyn store::Db,
                         cache: &dyn store::CacheBackend,
                         pending: &mut Vec<BatchResult>,
                         committed_files: &mut usize,
                         committed_chunks: &mut usize,
                         total_cache_hits: &mut usize|
             -> crate::error::Result<()> {
                if pending.is_empty() {
                    return Ok(());
                }
                db.begin()?;
                let write_result = (|| -> crate::error::Result<()> {
                    for batch in pending.iter() {
                        for (rel_path, _) in &batch.meta {
                            db.remove_chunks_for_file(rel_path)?;
                        }
                        for (chunk, vec) in &batch.embedded {
                            db.insert_chunk(chunk, vec)?;
                        }
                        db.update_file_meta(&batch.meta)?;
                        *committed_files += batch.meta.len();
                        *committed_chunks += batch.embedded.len();
                        *total_cache_hits += batch.cache_hits;
                    }
                    Ok(())
                })();
                match write_result {
                    Ok(()) => {
                        db.commit()?;
                    }
                    Err(e) => {
                        let _ = db.rollback();
                        return Err(e);
                    }
                }
                // Write new cache entries to the shared embed cache (outside
                // the per-workspace transaction). These are idempotent, so
                // losing them on crash just means re-embedding next time.
                for batch in pending.iter() {
                    if !batch.new_cache.is_empty() {
                        cache.put(&batch.new_cache)?;
                    }
                }
                pending.clear();
                Ok(())
            };

            loop {
                let any_connected = active.iter().any(|&a| a);
                let mut received_this_pass = false;

                for (i, rx) in receivers.iter_mut().enumerate() {
                    if !active[i] {
                        continue;
                    }
                    match rx.try_recv() {
                        Ok(result) => {
                            received_this_pass = true;
                            pending.push(result?);
                        }
                        Err(crate::spsc::TryRecvError::Empty) => {}
                        Err(crate::spsc::TryRecvError::Disconnected) => {
                            active[i] = false;
                        }
                    }
                }

                // Flush when we've accumulated enough batches, or when all
                // workers are idle/done and we have anything pending.
                if pending.len() >= FLUSH_THRESHOLD || (!received_this_pass && !pending.is_empty())
                {
                    flush(
                        &mut db,
                        cache,
                        &mut pending,
                        &mut committed_files,
                        &mut committed_chunks,
                        &mut total_cache_hits,
                    )?;
                }

                if !any_connected {
                    // Final flush for any stragglers.
                    flush(
                        &mut db,
                        cache,
                        &mut pending,
                        &mut committed_files,
                        &mut committed_chunks,
                        &mut total_cache_hits,
                    )?;
                    break;
                }

                if !received_this_pass {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }

            Ok(())
        });

        commit_err?;
    }

    let total_files = unchanged_count + committed_files;
    log::info!(
        "{}: {total_files} files ({new_count} re-embedded, {deleted_count} deleted), \
         {committed_chunks} chunks indexed ({total_cache_hits} cache hits)",
        workspace_root.display()
    );
    Ok(())
}

fn classify_files(
    config: &Config,
    workspace_root: &std::path::Path,
    old_meta: &std::collections::HashMap<String, store::FileMeta>,
) -> crate::error::Result<(
    Vec<(String, Vec<parse::RawChunk>, store::FileMeta)>, // (rel_path, chunks, meta)
    Vec<String>,                                          // unchanged rel_paths
    Vec<String>,                                          // deleted rel_paths
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
                log::error!("skipping {}: {e}", abs_path.display());
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
        "target",
        ".git",
        ".slocate",
        "vendor",
        "node_modules",
        // OS/platform junk — symlink cycles, caches, irrelevant data.
        "Library",
        "Applications",
        ".Trash",
    ];
    let extensions: Vec<&str> = config.index.extensions.iter().map(|s| s.as_str()).collect();

    let mut files = Vec::new();
    let mut stack = vec![workspace_root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                log::error!("[walk] skipping {}: {e}", dir.display());
                continue;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    log::error!("[walk] skipping entry in {}: {e}", dir.display());
                    continue;
                }
            };
            let path = entry.path();
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();

            // Skip symlinks to avoid cycles.
            if path
                .symlink_metadata()
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false)
            {
                continue;
            }

            if path.is_dir() {
                if file_name.starts_with('.') || skip_dirs.contains(&file_name) {
                    continue;
                }
                stack.push(path);
                continue;
            }

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default();
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
/// Returns true on success, false if the receiver is dropped.
pub(crate) fn spsc_blocking_send<T>(tx: &crate::spsc::SpscSender<T>, mut msg: T) -> bool {
    loop {
        match tx.try_send(msg) {
            Ok(()) => return true,
            Err(crate::spsc::TrySendError::Full(m)) => {
                msg = m;
                std::thread::yield_now();
            }
            Err(crate::spsc::TrySendError::Disconnected(_)) => return false,
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
            log::warn!("failed to set background QoS (errno {ret})");
        }
    }

    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn nice(inc: i32) -> i32;
            fn syscall(num: std::ffi::c_long, ...) -> std::ffi::c_long;
        }

        // nice 19 = lowest CPU scheduling priority.
        unsafe {
            nice(19);
        }

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
            unsafe {
                syscall(
                    SYS_IOPRIO_SET,
                    IOPRIO_WHO_PROCESS,
                    0 as std::ffi::c_long,
                    IOPRIO_PRIO_VALUE,
                );
            }
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
        | [0xCA, 0xFE, 0xBA, 0xBE] // Mach-O fat binary / universal
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
        handle
            .join()
            .expect("spsc_blocking_send must return when receiver drops");
    }

    // ── CancelGuard ──────────────────────────────────────────────────────────

    #[test]
    fn cancel_guard_sets_flag_on_drop() {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let _guard = CancelGuard(std::sync::Arc::clone(&flag));
            assert!(
                !flag.load(std::sync::atomic::Ordering::Acquire),
                "flag before drop"
            );
        }
        assert!(
            flag.load(std::sync::atomic::Ordering::Acquire),
            "flag after drop"
        );
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
        assert!(
            flag.load(std::sync::atomic::Ordering::Acquire),
            "cancel flag must be set even when closure exits via ?"
        );
    }

    // ── classify_files ───────────────────────────────────────────────────────

    #[test]
    fn classify_files_detects_new_files() {
        let dir = tempdir();
        std::fs::write(dir.join("main.rs"), "fn main() {}").unwrap();

        let config = crate::config::Config::default();
        let old_meta = std::collections::HashMap::new();
        let (to_embed, unchanged, deleted) = classify_files(&config, &dir, &old_meta).unwrap();

        assert!(
            !to_embed.is_empty(),
            "new .rs file must be classified for embedding"
        );
        assert!(unchanged.is_empty());
        assert!(deleted.is_empty());
    }

    #[test]
    fn classify_files_detects_deleted_files() {
        let dir = tempdir();

        let config = crate::config::Config::default();
        let mut old_meta = std::collections::HashMap::new();
        old_meta.insert(
            "gone.rs".to_string(),
            store::FileMeta {
                mtime_ns: 1,
                size: 1,
            },
        );

        let (to_embed, unchanged, deleted) = classify_files(&config, &dir, &old_meta).unwrap();

        assert!(to_embed.is_empty());
        assert!(unchanged.is_empty());
        assert_eq!(
            deleted,
            vec!["gone.rs".to_string()],
            "file in old_meta but not on disk must be classified as deleted"
        );
    }

    #[test]
    fn classify_files_unchanged_file_skipped() {
        let dir = tempdir();
        let file_path = dir.join("lib.rs");
        std::fs::write(&file_path, "fn foo() {}").unwrap();

        // Record the current mtime and size.
        let meta = std::fs::metadata(&file_path).unwrap();
        let mtime_ns = meta
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let size = meta.len() as i64;

        let config = crate::config::Config::default();
        let mut old_meta = std::collections::HashMap::new();
        old_meta.insert("lib.rs".to_string(), store::FileMeta { mtime_ns, size });

        let (to_embed, unchanged, deleted) = classify_files(&config, &dir, &old_meta).unwrap();

        assert!(
            to_embed.is_empty(),
            "unchanged file must not be re-embedded"
        );
        assert!(
            !unchanged.is_empty(),
            "unchanged file must be in unchanged list"
        );
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

        assert!(
            to_embed.iter().all(|(p, _, _)| !p.contains("target")),
            "files inside target/ must be ignored"
        );
    }

    // ── split_windows ─────────────────────────────────────────────────────────

    #[test]
    fn split_windows_small_source_is_single_window() {
        let wins = split_windows("fn foo() {}", "code: ");
        assert_eq!(wins.len(), 1);
        assert_eq!(wins[0], "code: fn foo() {}");
    }

    #[test]
    fn split_windows_large_source_splits_correctly() {
        // Build a source that is exactly 2 * EMBED_WINDOW chars.
        let source = "x".repeat(EMBED_WINDOW * 2);
        let wins = split_windows(&source, "code: ");
        assert_eq!(wins.len(), 2, "should split into exactly 2 windows");
        // Each window except possibly the last should be EMBED_WINDOW chars of source.
        assert_eq!(wins[0], format!("code: {}", "x".repeat(EMBED_WINDOW)));
        assert_eq!(wins[1], format!("code: {}", "x".repeat(EMBED_WINDOW)));
    }

    #[test]
    fn split_windows_partial_last_window() {
        let source = "x".repeat(EMBED_WINDOW + 100);
        let wins = split_windows(&source, "code: ");
        assert_eq!(wins.len(), 2);
        assert_eq!(wins[1], format!("code: {}", "x".repeat(100)));
    }

    // ── l2_normalize ─────────────────────────────────────────────────────────

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let mut v = vec![3.0f32, 4.0f32]; // magnitude 5
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm should be 1.0, got {norm}");
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_is_noop() {
        let mut v = vec![0.0f32, 0.0f32];
        l2_normalize(&mut v); // must not panic
        assert_eq!(v, vec![0.0f32, 0.0f32]);
    }

    // ── log_weighted_sum ─────────────────────────────────────────────────────

    #[test]
    fn log_weighted_sum_single_window_is_scaled_input() {
        // Single window: result = log(1 + len) * v, direction matches v.
        let v = vec![1.0f32, 0.0f32];
        let t = "hello".to_string();
        let result = log_weighted_sum(&[v.clone()], &[t.clone()]);
        let w = (1.0 + t.len() as f32).ln();
        assert!((result[0] - w).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn log_weighted_sum_opposite_vectors_cancel() {
        // Two equal-weight windows pointing in opposite directions → near-zero sum.
        let v1 = vec![1.0f32, 0.0f32];
        let v2 = vec![-1.0f32, 0.0f32];
        let t = "hello".to_string(); // same length → same log weight
        let result = log_weighted_sum(&[v1, v2], &[t.clone(), t]);
        assert!(
            result[0].abs() < 1e-6,
            "opposite unit vectors should cancel"
        );
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
