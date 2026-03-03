use crate::backends::HookBackend;
use crate::config::Config;
use crate::vdb;
use crate::{embed, registry, store};

pub struct ScoredChunk {
    pub score: f32,
    pub chunk: store::Chunk,
}

pub fn search_workspaces(
    embedder: &embed::Embedder,
    config: &Config,
    prompt: &str,
) -> crate::error::Result<Vec<ScoredChunk>> {
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
        let index_dir = registry::index_dir(ws)?;
        let db = store::Store::open(&index_dir)?;
        let hnsw = db.load_hnsw(vdb::dot)?;

        if hnsw.is_empty() {
            continue;
        }

        // Detect dimension mismatch (e.g. index built with old 768-dim model,
        // query from new 384-dim model). Silently skipping would give wrong
        // results; fail loud so the user knows to reindex.
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

pub fn query_all_workspaces(
    embedder: &embed::Embedder,
    config: &Config,
    prompt: &str,
    backend: &dyn HookBackend,
) -> crate::error::Result<String> {
    let results = search_workspaces(embedder, config, prompt)?;
    Ok(backend.format_results(&results, config.search.top_k))
}
