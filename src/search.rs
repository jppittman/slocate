use crate::backends::HookBackend;
use crate::config::Config;
use crate::embed::Embedder;
use crate::store::Db;
use crate::{registry, store};

/// Dot product. Equals cosine similarity on L2-normalized vectors.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[derive(serde::Serialize)]
pub struct ScoredChunk {
    pub score: f32,
    pub chunk: store::Chunk,
}

pub fn search_workspaces(
    embedder: &dyn Embedder,
    config: &Config,
    prompt: &str,
) -> crate::error::Result<Vec<ScoredChunk>> {
    let query_vec = embedder.embed(prompt)?;
    let top_k = config.search.top_k;
    let min_score = config.search.min_score;
    let mmr_lambda = config.search.mmr_lambda;

    let workspaces = config.expanded_workspaces();
    let mut candidates: Vec<(f32, store::Chunk, Vec<f32>)> = Vec::new();

    for ws in &workspaces {
        let index_dir = registry::index_dir(ws)?;
        let db = store::SqliteDb::open(&index_dir)?;

        // Flat scan: score all chunk vectors against the query.
        let all_chunks = db.load_all_chunks_with_vectors()?;
        for (chunk, vector) in all_chunks {
            let sim = dot(&query_vec, &vector);
            if sim >= min_score {
                candidates.push((sim, chunk, vector));
            }
        }

        // BM25 fallback/boost
        if config.search.bm25_weight > 0.0 {
            let bm25_hits = db.bm25_search(prompt, top_k * 4).unwrap_or_default();
            let mut bm25_map: std::collections::HashMap<String, f32> = bm25_hits.into_iter().collect();
            let w = config.search.bm25_weight;

            // Update existing candidates' scores with BM25 blend
            for (score, chunk, _vec) in candidates.iter_mut() {
                let bm25 = bm25_map.remove(&chunk.id).unwrap_or(0.0);
                *score = w * bm25 + (1.0 - w) * (*score);
            }

            // Add remaining BM25 hits not already in candidates
            for (id, bm25_score) in bm25_map {
                if let Ok(chunks) = db.get_chunks_with_vectors_by_ids(&[id]) {
                    if let Some((chunk, vec)) = chunks.into_iter().next() {
                        let sim = dot(&query_vec, &vec);
                        let final_score = w * bm25_score + (1.0 - w) * sim;
                        candidates.push((final_score, chunk, vec));
                    }
                }
            }
        }
    }

    // Dedup by chunk id, keeping highest score per chunk.
    {
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut i = 0;
        while i < candidates.len() {
            let id = candidates[i].1.id.clone();
            match seen.get(&id) {
                Some(&j) if candidates[j].0 >= candidates[i].0 => { candidates.swap_remove(i); }
                Some(&j) => { candidates.swap_remove(j); seen.insert(id, i.min(candidates.len() - 1)); }
                None => { seen.insert(id, i); i += 1; }
            }
        }
    }

    // MMR reranking
    let results = mmr(&query_vec, candidates, top_k, mmr_lambda);
    Ok(results)
}

/// Maximal Marginal Relevance reranking.
fn mmr(
    _query_vec: &[f32],
    mut candidates: Vec<(f32, store::Chunk, Vec<f32>)>,
    top_k: usize,
    lambda: f32,
) -> Vec<ScoredChunk> {
    if lambda >= 1.0 || candidates.len() <= 1 {
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        return candidates
            .into_iter()
            .take(top_k)
            .map(|(score, chunk, _)| ScoredChunk { score, chunk })
            .collect();
    }

    let mut selected: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    let mut results: Vec<ScoredChunk> = Vec::with_capacity(top_k);

    while results.len() < top_k && !candidates.is_empty() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, (query_sim, _, vec)) in candidates.iter().enumerate() {
            let max_redundancy = selected
                .iter()
                .map(|sel| dot(vec, sel))
                .fold(f32::NEG_INFINITY, f32::max);

            let mmr_score = if selected.is_empty() {
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
        selected.push(vec);
        results.push(ScoredChunk { score, chunk });
    }

    results
}

pub fn query_all_workspaces(
    embedder: &dyn Embedder,
    config: &Config,
    prompt: &str,
    backend: &dyn HookBackend,
) -> crate::error::Result<String> {
    let results = search_workspaces(embedder, config, prompt)?;
    Ok(backend.format_results(&results, config.search.top_k))
}
