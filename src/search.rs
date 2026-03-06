use crate::backends::HookBackend;
use crate::config::Config;
use crate::vdb;
use crate::{embed, registry, store};

pub struct ScoredChunk {
    pub score: f32,
    pub chunk: store::Chunk,
    pub lineage: Vec<String>,
}

pub fn search_workspaces(
    embedder: &embed::Embedder,
    config: &Config,
    prompt: &str,
) -> crate::error::Result<Vec<ScoredChunk>> {
    let prefixed = format!("code: {prompt}");
    let query_vec = embedder.embed(&prefixed)?;
    let top_k = config.search.top_k;
    let min_score = config.search.min_score;
    let mmr_lambda = config.search.mmr_lambda;

    let workspaces = config.expanded_workspaces();
    let mut candidates: Vec<(f32, store::Chunk, Vec<f32>, Vec<String>)> = Vec::new();

    for ws in &workspaces {
        let index_dir = registry::index_dir(ws)?;
        let db = store::Store::open(&index_dir)?;

        // 1. Start Tournament at the root (bundles with no parent)
        let mut current_bundle_ids: Vec<i64> = db.load_bundles_by_parent(None)?
            .into_iter()
            .map(|(id, _, _, _)| id)
            .collect();

        if current_bundle_ids.is_empty() {
            continue;
        }

        // 2. Descend the tree recursively
        let mut final_leaf_bundle_ids = Vec::new();
        let mut level_nodes = current_bundle_ids;

        while !level_nodes.is_empty() {
            let mut bundles = Vec::new();
            for &id in &level_nodes {
                if let Ok(b) = db.get_bundle_by_id(id) {
                    bundles.push(b);
                }
            }

            if bundles.is_empty() { break; }

            // Score candidate bundles against the query
            let mut scores: Vec<(f32, i64, i32)> = bundles.iter()
                .map(|b| (vdb::dot(&query_vec, &b.vector), b.id, b.level))
                .collect();
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Pick Top-2 winners to branch down (Tournament selection)
            let top_n = 2.min(scores.len());
            let winners = &scores[..top_n];

            let mut next_level_nodes = Vec::new();
            for &(_score, id, level) in winners {
                if level == 0 {
                    final_leaf_bundle_ids.push(id);
                } else {
                    let children = db.load_bundles_by_parent(Some(id))?;
                    for (child_id, _, _, _) in children {
                        next_level_nodes.push(child_id);
                    }
                }
            }
            level_nodes = next_level_nodes;
        }

        // 3. Brute force search within the chosen leaf communities
        let mut leaf_candidates: Vec<(f32, store::Chunk, Vec<f32>)> = Vec::new();
        for id in final_leaf_bundle_ids {
            let chunks = db.load_chunks_with_vectors_by_bundle(id)?;
            for (chunk, vector) in chunks {
                let sim = vdb::dot(&query_vec, &vector);
                if sim >= min_score {
                    leaf_candidates.push((sim, chunk, vector));
                }
            }
        }

        // 4. BM25 Fallback/Boost
        if config.search.bm25_weight > 0.0 {
            let bm25_hits = db.bm25_search(prompt, top_k * 4).unwrap_or_default();
            for (id, bm25_score) in bm25_hits {
                if !leaf_candidates.iter().any(|(_, c, _)| c.id == id) {
                    if let Ok(chunks) = db.get_chunks_by_ids(&[id]) {
                        if let Some(chunk) = chunks.into_iter().next() {
                            // Fetch raw vector from DB for MMR inter-result similarity
                            if let Ok(full_chunks) = db.get_chunks_with_vectors_by_ids(&[chunk.id.clone()]) {
                                if let Some((_, vec)) = full_chunks.into_iter().next() {
                                    leaf_candidates.push((bm25_score, chunk, vec));
                                }
                            }
                        }
                    }
                }
            }
        }

        for (sim, chunk, vector) in leaf_candidates {
            let lineage = if let Some(bid) = chunk.community_id {
                db.get_bundle_lineage(bid as i64).unwrap_or_default()
            } else {
                Vec::new()
            };
            candidates.push((sim, chunk, vector, lineage));
        }
    }

    // 5. MMR reranking
    let results = mmr(&query_vec, candidates, top_k, mmr_lambda);
    Ok(results)
}

/// Maximal Marginal Relevance reranking.
///
/// `candidates` is a list of `(query_similarity, chunk, vector, lineage)` tuples,
/// pre-filtered by `min_score`. Returns up to `top_k` results ordered by
/// the MMR selection sequence (most relevant/diverse first).
fn mmr(
    query_vec: &[f32],
    mut candidates: Vec<(f32, store::Chunk, Vec<f32>, Vec<String>)>,
    top_k: usize,
    lambda: f32,
) -> Vec<ScoredChunk> {
    // Pure top-k fast path — skip MMR bookkeeping entirely.
    if lambda >= 1.0 || candidates.len() <= 1 {
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        return candidates
            .into_iter()
            .take(top_k)
            .map(|(score, chunk, _, lineage)| ScoredChunk { score, chunk, lineage })
            .collect();
    }

    let _ = query_vec; // query similarity already in candidates[i].0
    let mut selected: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    let mut results: Vec<ScoredChunk> = Vec::with_capacity(top_k);

    while results.len() < top_k && !candidates.is_empty() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, (query_sim, _, vec, _)) in candidates.iter().enumerate() {
            // Max similarity to any already-selected result.
            let max_redundancy = selected
                .iter()
                .map(|sel| vdb::dot(vec, sel))
                .fold(f32::NEG_INFINITY, f32::max);

            let mmr_score = if selected.is_empty() {
                // Nothing selected yet — pure relevance.
                *query_sim
            } else {
                lambda * query_sim - (1.0 - lambda) * max_redundancy
            };

            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = i;
            }
        }

        let (score, chunk, vec, lineage) = candidates.swap_remove(best_idx);
        selected.push(vec);
        results.push(ScoredChunk { score, chunk, lineage });
    }

    results
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
