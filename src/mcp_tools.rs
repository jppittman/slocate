use crate::config::Config;
use crate::vdb;
use crate::{embed, mcp, registry, store, reindex};
use serde_json::Value;

/// Pair of embedders for MCP tool dispatch.
///
/// `index` is used for bulk embedding during workspace indexing — typically
/// GPU-backed when available. `query` is used for single-vector lookups at
/// search time — CPU is fine here and avoids contention with ongoing indexing.
/// When no GPU is present, both fields point to the same CPU embedder.
pub struct Embedders<'a> {
    pub index: &'a embed::Embedder,
    pub query: &'a embed::Embedder,
}

pub fn handle(
    embedders: &Embedders<'_>,
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

            let result_text: crate::error::Result<String> = match tool_name {
                "index_workspace" => index_workspace(embedders.index, config, &args),
                "search_code" => search_code(embedders.query, config, &args),
                "note_to_self" => note_to_self(embedders.query, &args),
                "check_notes" => check_notes(embedders.query, config, &args),
                other => Err(crate::error::Error::NotFound(format!("unknown tool: {other}"))),
            };

            let result_value = match result_text {
                Ok(text) => mcp::tool_result(text),
                Err(e) => {
                    eprintln!("tool error [{tool_name}]: {e}");
                    mcp::tool_error_result(e.to_string())
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
            "description": "Incremental reindex: walks the workspace, parses source files with tree-sitter, embeds changed/new chunks with BGE-small-en-v1.5 (in-process candle), and updates the HNSW index in .slocate/index.db. Only re-embeds files whose mtime or size changed since the last run. Automatically triggered by git post-commit/post-merge hooks and a periodic daemon, so manual calls are rarely needed.",
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
            "description": "Hybrid BM25+semantic search over indexed code chunks. Combines FTS5 full-text (BM25) with HNSW approximate nearest-neighbor cosine similarity (BGE-small-en-v1.5 384-dim). BM25 catches exact identifier and keyword matches; semantic search catches conceptual queries. Final score = bm25_weight·bm25 + (1−bm25_weight)·cosine. Results include source code, file path, chunk kind (function, impl, struct, etc.), and community label.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or code query (identifiers, function names, concepts)."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default from config)."
                    },
                    "crate_filter": {
                        "type": "string",
                        "description": "Optional prefix filter on source_path (e.g. 'pixelflow-core/')."
                    },
                    "mmr_lambda": {
                        "type": "number",
                        "description": "MMR diversity parameter 0–1. 1.0 = pure relevance, 0.0 = pure diversity. Default 0.8 keeps related functions from the same file together."
                    },
                    "bm25_weight": {
                        "type": "number",
                        "description": "Weight for BM25 in hybrid score (0–1). 0.0 = pure semantic, 1.0 = pure BM25. Default 0.5."
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "note_to_self",
            "description": "Store a persistent note embedded with BGE-small for later semantic retrieval via check_notes. Notes survive across sessions. Use for design decisions, TODOs, debugging insights, or anything worth remembering.",
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

fn index_workspace(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> crate::error::Result<String> {
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

    reindex::reindex_workspace(embedder, config, &workspace_root)?;
    Ok(format!(
        "Indexing complete. Written to {}/.slocate/index.db",
        workspace_root.display()
    ))
}

fn search_code(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> crate::error::Result<String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| crate::error::Error::Config("search_code requires a 'query' string argument".into()))?;

    let top_k = args
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(config.search.top_k);

    let crate_filter = args
        .get("crate_filter")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let mmr_lambda = args
        .get("mmr_lambda")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(config.search.mmr_lambda);

    let bm25_weight = args
        .get("bm25_weight")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(config.search.bm25_weight);

    let workspace_root = match config.expanded_workspaces().into_iter().next() {
        Some(ws) => ws,
        None => find_workspace_root()?,
    };
    let index_dir = registry::index_dir(&workspace_root)?;
    let db = store::Store::open(&index_dir)?;
    let hnsw = db.load_hnsw(vdb::dot)?;

    if hnsw.is_empty() {
        return Ok("No chunks indexed yet. Run `index_workspace` first.".to_string());
    }

    let query_vec = embedder.embed(query)?;

    if let Some(first) = hnsw.nodes.first() {
        if first.vector.len() != query_vec.len() {
            return Err(crate::error::Error::Embed(format!(
                "dimension mismatch: index has {}-dim vectors but query is {}-dim \
                 (reindex needed after model change: `slocate reindex`)",
                first.vector.len(),
                query_vec.len(),
            )));
        }
    }

    // ── Semantic search via HNSW ──────────────────────────────────────────
    let n_candidates = (top_k * 8).max(64);
    let hits = hnsw.search(&query_vec, n_candidates, n_candidates);

    let vec_by_id: std::collections::HashMap<&str, &[f32]> = hnsw
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.vector.as_slice()))
        .collect();

    let semantic_scores: std::collections::HashMap<String, f32> =
        hits.iter().map(|h| (h.id.clone(), h.score)).collect();

    // ── BM25 search via FTS5 ──────────────────────────────────────────────
    let bm25_scores: std::collections::HashMap<String, f32> = if bm25_weight > 0.0 {
        db.bm25_search(query, n_candidates)
            .unwrap_or_default()
            .into_iter()
            .collect()
    } else {
        std::collections::HashMap::new()
    };

    // ── Merge + hybrid score ──────────────────────────────────────────────
    let mut seen = std::collections::HashSet::new();
    let mut candidate_ids: Vec<String> = Vec::new();
    let min_score = config.search.min_score;

    for h in &hits {
        let has_bm25 = bm25_scores.contains_key(&h.id);
        if h.score >= min_score || has_bm25 {
            if seen.insert(h.id.clone()) {
                candidate_ids.push(h.id.clone());
            }
        }
    }
    for id in bm25_scores.keys() {
        if seen.insert(id.clone()) {
            candidate_ids.push(id.clone());
        }
    }

    // Apply crate_filter before fetching chunks.
    let chunks = db.get_chunks_by_ids(&candidate_ids)?;
    let by_id: std::collections::HashMap<String, store::Chunk> =
        chunks.into_iter().map(|c| (c.id.clone(), c)).collect();

    // Build scored, filtered candidates for MMR.
    let mut candidates: Vec<(f32, &store::Chunk, Vec<f32>)> = candidate_ids
        .iter()
        .filter_map(|id| {
            let chunk = by_id.get(id)?;
            if crate_filter
                .as_deref()
                .map(|f| !chunk.source_path.starts_with(f))
                .unwrap_or(false)
            {
                return None;
            }
            let cosine = semantic_scores.get(id).copied().unwrap_or(0.0);
            let bm25 = bm25_scores.get(id).copied().unwrap_or(0.0);
            let hybrid_score = bm25_weight * bm25 + (1.0 - bm25_weight) * cosine;
            let vec = vec_by_id
                .get(id.as_str())
                .map(|v| v.to_vec())
                .unwrap_or_else(|| vec![0.0f32; query_vec.len()]);
            Some((hybrid_score, chunk, vec))
        })
        .collect();

    // MMR reranking.
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let selected = if mmr_lambda >= 1.0 || candidates.len() <= 1 {
        candidates.into_iter().take(top_k).collect::<Vec<_>>()
    } else {
        mmr_select(&query_vec, candidates, top_k, mmr_lambda)
    };

    let mut out = String::new();
    for (score, chunk, _) in &selected {
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
            score, community, chunk.kind, chunk.name, chunk.source_path, preview
        ));
    }

    if out.is_empty() {
        Ok("No matching chunks found.".to_string())
    } else {
        Ok(out.trim_end().to_string())
    }
}

/// MMR selection returning `(score, chunk_ref, vector)` tuples.
/// Mirrors the logic in `search::mmr` but works over borrowed chunk refs.
fn mmr_select<'a>(
    query_vec: &[f32],
    mut candidates: Vec<(f32, &'a store::Chunk, Vec<f32>)>,
    top_k: usize,
    lambda: f32,
) -> Vec<(f32, &'a store::Chunk, Vec<f32>)> {
    let _ = query_vec;
    let mut selected_vecs: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    let mut results = Vec::with_capacity(top_k);

    while results.len() < top_k && !candidates.is_empty() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, (query_sim, _, vec)) in candidates.iter().enumerate() {
            let max_redundancy = selected_vecs
                .iter()
                .map(|sel| vdb::dot(vec, sel))
                .fold(f32::NEG_INFINITY, f32::max);
            let mmr_score = if selected_vecs.is_empty() {
                *query_sim
            } else {
                lambda * query_sim - (1.0 - lambda) * max_redundancy
            };
            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = i;
            }
        }

        let (score, chunk, vec) = candidates.swap_remove(best_idx);
        selected_vecs.push(vec.clone());
        results.push((score, chunk, vec));
    }

    results
}

fn note_to_self(embedder: &embed::Embedder, args: &Value) -> crate::error::Result<String> {
    let text = args
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| crate::error::Error::Config("note_to_self requires a 'text' string argument".into()))?;

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
    let index_dir = registry::index_dir(&workspace_root)?;
    let db = store::Store::open(&index_dir)?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let ts_str = timestamp.to_string();
    let id = store::chunk_id(text, &ts_str, "note");

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

fn check_notes(
    embedder: &embed::Embedder,
    config: &Config,
    args: &Value,
) -> crate::error::Result<String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| crate::error::Error::Config("check_notes requires a 'query' string argument".into()))?;

    let top_k = args
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(config.search.top_k);

    let workspace_root = find_workspace_root()?;
    let index_dir = registry::index_dir(&workspace_root)?;
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

fn find_workspace_root() -> crate::error::Result<std::path::PathBuf> {
    let cwd = std::env::current_dir()?;

    let mut dir = cwd.as_path();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = std::fs::read_to_string(&cargo_toml)?;
            if content.contains("[workspace]") {
                return Ok(dir.to_path_buf());
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => {
                return Err(crate::error::Error::NotFound(
                    "could not find workspace root (no Cargo.toml with [workspace] found)".into(),
                ));
            }
        }
    }
}
