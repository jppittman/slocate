use crate::backends::HookBackend;
use crate::config::Config;
use crate::vdb::{self, Hnsw};
use crate::{embed, leiden, mcp, parse, store};
use serde_json::Value;


// ─── MCP dispatch ─────────────────────────────────────────────────────────────

pub fn handle(
    embedder: &embed::Embedder,
    config: &Config,
    req: &serde_json::Value,
) -> Option<serde_json::Value> {
    let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");

    match method {
        "initialize" => Some(mcp::ok(
            id,
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "slocate", "version": "0.1.0"}
            }),
        )),

        // Notifications have no id and expect no response.
        "notifications/initialized" => None,

        "tools/list" => Some(mcp::ok(
            id,
            serde_json::json!({"tools": tool_schemas()}),
        )),

        "tools/call" => {
            let params = req
                .get("params")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let tool_name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("(missing)");
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::Object(Default::default()));

            let result_text = match tool_name {
                "index_workspace" => index_workspace(embedder, config, &args),
                "search_code" => search_code(embedder, config, &args),
                "note_to_self" => note_to_self(embedder, &args),
                "check_notes" => check_notes(embedder, config, &args),
                other => Err(format!("unknown tool: {other}")),
            };

            let result_value = match result_text {
                Ok(text) => mcp::tool_result(text),
                Err(e) => {
                    eprintln!("tool error [{tool_name}]: {e}");
                    mcp::tool_error_result(e)
                }
            };

            Some(mcp::ok(id, result_value))
        }

        other => {
            eprintln!("unhandled method: {other}");
            // For unknown methods with an id, return a JSON-RPC error.
            if req.get("id").is_some() {
                Some(mcp::error(
                    id,
                    -32601,
                    format!("method not found: {other}"),
                ))
            } else {
                None
            }
        }
    }
}

fn tool_schemas() -> serde_json::Value {
    serde_json::json!([
        {
            "name": "index_workspace",
            "description": "Incremental reindex: walks the workspace, parses source files with tree-sitter, embeds changed/new chunks with BGE-base-en-v1.5 (in-process ONNX), and updates the HNSW index in .slocate/index.db. Only re-embeds files whose mtime or size changed since the last run. Automatically triggered by git post-commit/post-merge hooks and a periodic daemon, so manual calls are rarely needed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional explicit workspace root path. Defaults to the first workspace in config or the workspace containing the current directory."
                    }
                }
            }
        },
        {
            "name": "search_code",
            "description": "Semantic search over indexed code chunks via HNSW approximate nearest neighbors. Returns the top-k most similar chunks by cosine similarity (BGE-base-en-v1.5 768-dim embeddings). Results include source code, file path, chunk kind (function, impl, struct, etc.), and community label. Use crate_filter to scope to a specific crate.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or code query to search for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default from config)."
                    },
                    "crate_filter": {
                        "type": "string",
                        "description": "Optional prefix filter on source_path (e.g. 'pixelflow-core/')."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "note_to_self",
            "description": "Store a persistent note embedded with BGE-base for later semantic retrieval via check_notes. Notes survive across sessions. Use for design decisions, TODOs, debugging insights, or anything worth remembering.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note content to store."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of string tags for the note."
                    }
                },
                "required": ["text"]
            }
        },
        {
            "name": "check_notes",
            "description": "Semantic search over stored notes. Returns notes ranked by cosine similarity to the query. Use to recall past design decisions, debugging insights, or context from previous sessions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search notes."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default from config)."
                    }
                },
                "required": ["query"]
            }
        }
    ])
}

// ─── Public reindex/query entry points for CLI subcommands ────────────────────

/// Incremental reindex: only re-embed files that changed since last index.
///
/// Pipeline:
///   1. Walk all files, stat each one → compare (mtime, size) to stored metadata
///   2. Classify: unchanged / modified / new / deleted
///   3. Parse + embed only modified/new files (parallel, scoped threads)
///   4. Remove deleted files' chunks, insert/replace modified/new chunks
///   5. Rebuild HNSW from all chunks, run Leiden, persist
///
/// First run (no file_meta table) behaves like a full reindex.
pub fn reindex_workspace(
    embedder: &embed::Embedder,
    config: &Config,
    workspace_root: &std::path::Path,
) -> Result<(), String> {
    let index_dir = store::index_dir(workspace_root)?;
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

    // ── Phase 3: batched embed changed/new files ─────────────────────────────
    // GPU: single worker (GPU handles parallelism internally, bigger batches = better).
    // CPU: N workers, each batches its own bucket via embed_batch.
    let n_workers = if embedder.is_gpu() {
        1
    } else {
        config.index.embed_workers.max(1)
    };
    let mut buckets: Vec<Vec<(String, Vec<parse::RawChunk>, i64, i64)>> =
        (0..n_workers).map(|_| Vec::new()).collect();
    for (i, item) in to_embed.into_iter().enumerate() {
        buckets[i % n_workers].push(item);
    }

    let mut new_embedded: Vec<(store::Chunk, Vec<f32>)> = Vec::new();
    let mut new_meta: Vec<(String, i64, i64)> = Vec::new();

    std::thread::scope(|s| -> Result<(), String> {
        let handles: Vec<_> = buckets
            .into_iter()
            .map(|bucket| {
                s.spawn(move || -> Result<Vec<(store::Chunk, Vec<f32>, String, i64, i64)>, String> {
                    // Flatten all chunks from all files in this bucket into one big
                    // batch so embed_batch can run one (or a few) forward passes
                    // instead of one per chunk.
                    let mut embed_inputs: Vec<String> = Vec::new();
                    let mut chunk_meta: Vec<(parse::RawChunk, String, i64, i64)> = Vec::new();

                    for (rel_path, raw_chunks, mtime, size) in bucket {
                        let n = raw_chunks.len();
                        eprintln!("[embed] {rel_path}: {n} chunks");
                        for raw in raw_chunks {
                            embed_inputs.push(format!(
                                "{} {} in {}\n\n{}",
                                raw.kind, raw.name, rel_path, raw.embed_text
                            ));
                            chunk_meta.push((raw, rel_path.clone(), mtime, size));
                        }
                    }

                    if embed_inputs.is_empty() {
                        return Ok(Vec::new());
                    }

                    let vectors = embedder.embed_batch(&embed_inputs).map_err(|e| {
                        format!("batch embedding failed: {e}")
                    })?;

                    if vectors.len() != chunk_meta.len() {
                        return Err(format!(
                            "embed_batch returned {} vectors but expected {}",
                            vectors.len(),
                            chunk_meta.len()
                        ));
                    }

                    let mut results = Vec::with_capacity(vectors.len());
                    for ((raw, rel_path, mtime, size), vector) in
                        chunk_meta.into_iter().zip(vectors)
                    {
                        let id = store::chunk_id(&rel_path, &raw.name, &raw.kind);
                        results.push((
                            store::Chunk {
                                id,
                                kind: raw.kind,
                                name: raw.name,
                                source_path: rel_path.clone(),
                                source: raw.source,
                                community_id: None,
                            },
                            vector,
                            rel_path,
                            mtime,
                            size,
                        ));
                    }
                    Ok(results)
                })
            })
            .collect();

        for handle in handles {
            let batch = handle
                .join()
                .map_err(|_| "embedding worker thread panicked".to_string())??;
            let mut seen_paths = std::collections::HashSet::new();
            for (chunk, vector, rel_path, mtime, size) in batch {
                if seen_paths.insert(rel_path.clone()) {
                    new_meta.push((rel_path, mtime, size));
                }
                new_embedded.push((chunk, vector));
            }
        }
        Ok(())
    })?;

    // ── Phase 4: remove old chunks for modified files, insert new ones ───────
    {
        let modified_paths: std::collections::HashSet<&str> =
            new_meta.iter().map(|(p, _, _)| p.as_str()).collect();
        for path in &modified_paths {
            db.remove_chunks_for_file(path)?;
        }
    }

    for (chunk, _) in &new_embedded {
        db.insert_chunk(chunk)?;
    }
    db.update_file_meta(&new_meta)?;

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

    let total_files = unchanged_count + new_meta.len();
    eprintln!(
        "[reindex] {}: {total_files} files ({new_count} re-embedded, {deleted_count} deleted), \
         {total_chunks} chunks, {} communities ({n_workers} embed workers)",
        workspace_root.display(),
        partition.n_communities,
    );
    Ok(())
}

/// Classify files into: needs-embedding, unchanged, deleted.
/// Returns (to_embed, unchanged_paths, deleted_paths).
fn classify_files(
    config: &Config,
    workspace_root: &std::path::Path,
    old_meta: &std::collections::HashMap<String, (i64, i64)>,
) -> Result<(
    Vec<(String, Vec<parse::RawChunk>, i64, i64)>,  // (rel_path, chunks, mtime_ns, size)
    Vec<String>,                                       // unchanged rel_paths
    Vec<String>,                                       // deleted rel_paths
), String> {
    let all_files = collect_file_paths(config, workspace_root)?;

    let mut to_embed = Vec::new();
    let mut unchanged = Vec::new();
    let mut current_paths = std::collections::HashSet::new();

    for (rel_path, abs_path) in &all_files {
        current_paths.insert(rel_path.clone());

        let meta = std::fs::metadata(abs_path)
            .map_err(|e| format!("stat failed for {}: {e}", abs_path.display()))?;

        let mtime_ns = meta
            .modified()
            .map_err(|e| format!("mtime failed for {}: {e}", abs_path.display()))?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let size = meta.len() as i64;

        // Check if file is unchanged.
        if let Some(&(old_mtime, old_size)) = old_meta.get(rel_path) {
            if old_mtime == mtime_ns && old_size == size {
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

        to_embed.push((rel_path.clone(), raw_chunks, mtime_ns, size));
    }

    // Files in old_meta but not on disk anymore → deleted.
    let deleted: Vec<String> = old_meta
        .keys()
        .filter(|k| !current_paths.contains(k.as_str()))
        .cloned()
        .collect();

    Ok((to_embed, unchanged, deleted))
}

/// Walk workspace and return (rel_path, abs_path) pairs for all source files.
fn collect_file_paths(
    config: &Config,
    workspace_root: &std::path::Path,
) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    let skip_dirs = ["target", ".git", ".slocate", "vendor", "node_modules"];
    let extensions: Vec<&str> = config.index.extensions.iter().map(|s| s.as_str()).collect();

    let mut files = Vec::new();
    let mut stack = vec![workspace_root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .map_err(|e| format!("read_dir failed for {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("dir entry error: {e}"))?;
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();

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

/// A search result with both similarity score and full chunk metadata.
pub struct ScoredChunk {
    pub score: f32,
    pub chunk: store::Chunk,
}

/// Embed a query, run HNSW search across all configured workspaces, fetch
/// metadata for hits, and drop results below `config.search.min_score`.
/// Returns results sorted descending by score.
pub fn search_workspaces(
    embedder: &embed::Embedder,
    config: &Config,
    prompt: &str,
) -> Result<Vec<ScoredChunk>, String> {
    // BGE responds better when the query has a task hint prefix.
    // "code: " biases embeddings toward source code matches.
    let prefixed = format!("code: {prompt}");
    let query_vec = embedder.embed(&prefixed)?;
    let top_k = config.search.top_k;
    let min_score = config.search.min_score;
    // ef > top_k gives the ANN search more candidates to improve recall.
    let ef = (top_k * 4).max(64);

    let workspaces = config.expanded_workspaces();
    let mut all: Vec<ScoredChunk> = Vec::new();

    for ws in &workspaces {
        let index_dir = store::index_dir(ws)?;
        let db = store::Store::open(&index_dir)?;
        let hnsw = db.load_hnsw(vdb::dot)?;

        if hnsw.is_empty() {
            continue;
        }

        let hits = hnsw.search(&query_vec, top_k, ef);
        let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        let chunks = db.get_chunks_by_ids(&ids)?;

        // Build a lookup so we preserve HNSW's score order.
        let mut by_id: std::collections::HashMap<String, store::Chunk> =
            chunks.into_iter().map(|c| (c.id.clone(), c)).collect();

        for hit in hits {
            if hit.score < min_score {
                break; // hits are sorted descending; everything after is also below threshold
            }
            if let Some(chunk) = by_id.remove(&hit.id) {
                all.push(ScoredChunk { score: hit.score, chunk });
            };
        }
    }

    // Sort across workspaces (each workspace's hits are already sorted,
    // but we merge multiple workspaces here).
    all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(all)
}

/// Search all configured workspaces and format the results via the given backend.
/// Used by CLI subcommands for LLM hook injection.
pub fn query_all_workspaces(
    embedder: &embed::Embedder,
    config: &Config,
    prompt: &str,
    backend: &dyn HookBackend,
) -> Result<String, String> {
    let results = search_workspaces(embedder, config, prompt)?;
    Ok(backend.format_results(&results, config.search.top_k))
}

// ─── MCP tool implementations ─────────────────────────────────────────────────

/// Walk workspace, parse supported files, embed each chunk, write chunks.jsonl.
fn index_workspace(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> Result<String, String> {
    let workspace_root = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            // Use first configured workspace, or fall back to workspace root detection.
            match config.expanded_workspaces().into_iter().next() {
                Some(ws) => ws,
                None => find_workspace_root()?,
            }
        }
    };

    reindex_workspace(embedder, config, &workspace_root)?;
    Ok(format!(
        "Indexing complete. Written to {}/.slocate/index.db",
        workspace_root.display()
    ))
}

/// Embed query, search HNSW, return top_k results with optional path filter.
fn search_code(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "search_code requires a 'query' string argument".to_string())?;

    let top_k = args
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(config.search.top_k);

    let crate_filter = args
        .get("crate_filter")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let workspace_root = match config.expanded_workspaces().into_iter().next() {
        Some(ws) => ws,
        None => find_workspace_root()?,
    };
    let index_dir = store::index_dir(&workspace_root)?;
    let db = store::Store::open(&index_dir)?;
    let hnsw = db.load_hnsw(vdb::dot)?;

    if hnsw.is_empty() {
        return Ok("No chunks indexed yet. Run `index_workspace` first.".to_string());
    }

    let query_vec = embedder.embed(query)?;
    // Request more candidates when filtering so we have enough after the filter.
    let ef = (top_k * 8).max(64);
    let hits = hnsw.search(&query_vec, ef, ef);

    let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    let chunks = db.get_chunks_by_ids(&ids)?;
    let by_id: std::collections::HashMap<String, store::Chunk> =
        chunks.into_iter().map(|c| (c.id.clone(), c)).collect();

    let mut out = String::new();
    let mut count = 0;
    for hit in &hits {
        if count >= top_k {
            break;
        }
        let chunk = match by_id.get(&hit.id) {
            Some(c) => c,
            None => continue,
        };
        if crate_filter
            .as_deref()
            .map(|f| !chunk.source_path.starts_with(f))
            .unwrap_or(false)
        {
            continue;
        }
        let preview = if chunk.source.len() > 300 {
            &chunk.source[..300]
        } else {
            &chunk.source
        };
        let community = chunk
            .community_id
            .map(|c| format!(" [community {c}]"))
            .unwrap_or_default();
        out.push_str(&format!(
            "[{:.2}]{} {} `{}` — {}\n{}\n\n",
            hit.score, community, chunk.kind, chunk.name, chunk.source_path, preview
        ));
        count += 1;
    }

    if out.is_empty() {
        Ok("No matching chunks found.".to_string())
    } else {
        Ok(out.trim_end().to_string())
    }
}

/// Embed and append a note to notes.jsonl.
fn note_to_self(embedder: &embed::Embedder, args: &Value) -> Result<String, String> {
    let text = args
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "note_to_self requires a 'text' string argument".to_string())?;

    let tags: Vec<String> = args
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let vector = embedder.embed(text)?;

    let workspace_root = find_workspace_root()?;
    let index_dir = store::index_dir(&workspace_root)?;
    let db = store::Store::open(&index_dir)?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string());
    let id = store::chunk_id(text, &timestamp, "note");

    let note = store::Note {
        id: id.clone(),
        text: text.to_string(),
        tags,
        timestamp,
        vector,
    };

    db.upsert_note(&note)?;

    Ok(format!("Note saved (id: {id})"))
}

/// Embed query, score notes by cosine similarity, return top_k results.
fn check_notes(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "check_notes requires a 'query' string argument".to_string())?;

    let top_k = args
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(config.search.top_k);

    let workspace_root = find_workspace_root()?;
    let index_dir = store::index_dir(&workspace_root)?;
    let db = store::Store::open(&index_dir)?;
    let notes = db.load_notes()?;
    if notes.is_empty() {
        return Ok("No notes yet. Use `note_to_self` to add one.".to_string());
    }

    let query_vec = embedder.embed(query)?;

    let mut scored: Vec<(f32, &store::Note)> = notes
        .iter()
        .map(|n| (vdb::dot(&query_vec, &n.vector), n))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = String::new();
    for (score, note) in scored.iter().take(top_k) {
        let tag_str = if note.tags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", note.tags.join(", "))
        };
        out.push_str(&format!(
            "[{:.2}]{} (ts: {}) {}\n\n",
            score, tag_str, note.timestamp, note.text
        ));
    }

    Ok(out.trim_end().to_string())
}

// ─── Workspace detection ───────────────────────────────────────────────────────

/// Returns true if the first 4 bytes of `path` match an ELF or Mach-O magic number.
/// Called before `read_to_string` so we never pull a binary blob into memory.
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

/// Walk up from cwd to find the directory containing a Cargo.toml with [workspace].
fn find_workspace_root() -> Result<std::path::PathBuf, String> {
    let cwd = std::env::current_dir()
        .map_err(|e| format!("cannot determine current directory: {e}"))?;

    let mut dir = cwd.as_path();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = std::fs::read_to_string(&cargo_toml)
                .map_err(|e| format!("failed to read {}: {e}", cargo_toml.display()))?;
            if content.contains("[workspace]") {
                return Ok(dir.to_path_buf());
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => {
                return Err(
                    "could not find workspace root (no Cargo.toml with [workspace] found)"
                        .to_string(),
                );
            }
        }
    }
}
