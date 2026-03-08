use crate::backends::HookBackend;
use crate::config::Config;
use crate::vdb;
use crate::{embed, registry, store};

/// Load a bundle matrix file. Returns (bundle_ids, matrix_flat, n_bundles, dim).
fn load_bundle_matrix(path: &std::path::Path) -> Option<(Vec<i64>, Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 8 { return None; }
    let n = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let ids_end = 8 + n * 8;
    let vecs_end = ids_end + n * dim * 4;
    if bytes.len() < vecs_end { return None; }
    let ids: Vec<i64> = (0..n).map(|i| {
        i64::from_le_bytes(bytes[8 + i*8 .. 8 + i*8 + 8].try_into().unwrap())
    }).collect();
    let vecs: Vec<f32> = (0..n*dim).map(|i| {
        let off = ids_end + i * 4;
        f32::from_le_bytes(bytes[off..off+4].try_into().unwrap())
    }).collect();
    Some((ids, vecs, n, dim))
}

#[derive(serde::Serialize)]
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
    let raw_query_vec = embedder.embed(prompt)?;
    let top_k = config.search.top_k;
    let min_score = config.search.min_score;
    let mmr_lambda = config.search.mmr_lambda;
    let beam_width = config.search.beam_width;

    let workspaces = config.expanded_workspaces();
    let mut candidates: Vec<(f32, store::Chunk, Vec<f32>, Vec<String>)> = Vec::new();

    for ws in &workspaces {
        let index_dir = registry::index_dir(ws)?;
        let db = store::Store::open(&index_dir)?;

        // Center the query using the global mean from this workspace
        let query_vec = if let Ok(Some(json)) = db.get_meta("global_mean") {
            let mean: Vec<f32> = serde_json::from_str(&json).unwrap_or_default();
            if mean.len() == raw_query_vec.len() {
                let mut centered = vec![0.0f32; mean.len()];
                for i in 0..mean.len() {
                    centered[i] = raw_query_vec[i] - mean[i];
                }
                // Re-normalize
                let norm = centered.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                for val in centered.iter_mut() {
                    *val /= norm;
                }
                centered
            } else {
                raw_query_vec.clone()
            }
        } else {
            raw_query_vec.clone()
        };

        // Flat scan: score all leaf path vectors against the query in one pass.
        // leaf_paths.bin = [n:u32][dim:u32][ids: i64×n][vecs: f32×n×dim]
        // Keep bundle scores so they can be blended with chunk cosines below.
        let community_weight = config.search.community_score_weight;
        let final_leaf_bundles: Vec<(f32, i64)> = if let Some((lp_ids, lp_vecs, lp_n, lp_dim)) =
            load_bundle_matrix(&index_dir.join("leaf_paths.bin"))
        {
            if lp_n > 0 && lp_dim == query_vec.len() {
                let mut scored: Vec<(f32, i64)> = lp_ids.into_iter().enumerate()
                    .map(|(i, id)| {
                        let row = &lp_vecs[i * lp_dim..(i + 1) * lp_dim];
                        (vdb::dot(&query_vec, row), id)
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                scored.into_iter().take(beam_width).collect()
            } else {
                // Dimension mismatch or empty — fall back to all root bundles
                db.load_bundles_by_parent(None)?.into_iter().map(|(id, _, _, _)| (0.0f32, id)).collect()
            }
        } else {
            // No leaf_paths.bin yet (first run before reindex) — fall back to DB bundles
            db.load_bundles_by_parent(None)?.into_iter().map(|(id, _, _, _)| (0.0f32, id)).collect()
        };

        if final_leaf_bundles.is_empty() {
            continue;
        }

        // Exact search within chosen leaf communities.
        // If community_score_weight > 0, blend the bundle's relevance score into each
        // chunk's final score: chunks in the best-matching community are promoted.
        for (bundle_score, id) in final_leaf_bundles {
            let lineage = db.get_bundle_lineage(id).unwrap_or_default();
            for (chunk, vector) in db.load_chunks_with_vectors_by_bundle_including_secondary(id)? {
                let sim = vdb::dot(&query_vec, &vector);
                let score = if community_weight > 0.0 {
                    (1.0 - community_weight) * sim + community_weight * bundle_score
                } else {
                    sim
                };
                if score >= min_score { candidates.push((score, chunk, vector, lineage.clone())); }
            }
        }

        // 4. BM25 Fallback/Boost
        if config.search.bm25_weight > 0.0 {
            let bm25_hits = db.bm25_search(prompt, top_k * 4).unwrap_or_default();
            let mut bm25_map: std::collections::HashMap<String, f32> = bm25_hits.into_iter().collect();
            let w = config.search.bm25_weight;

            // Update existing candidates' scores with BM25 blend
            for (score, chunk, _vec, _lineage) in candidates.iter_mut() {
                let bm25 = bm25_map.remove(&chunk.id).unwrap_or(0.0);
                *score = w * bm25 + (1.0 - w) * (*score);
            }

            // Add remaining BM25 hits
            for (id, bm25_score) in bm25_map {
                if let Ok(chunks) = db.get_chunks_with_vectors_by_ids(&[id]) {
                    if let Some((chunk, vec)) = chunks.into_iter().next() {
                        let lineage = chunk.community_id.map(|bid| db.get_bundle_lineage(bid as i64).unwrap_or_default()).unwrap_or_default();
                        let sim = vdb::dot(&query_vec, &vec);
                        let final_score = w * bm25_score + (1.0 - w) * sim;
                        candidates.push((final_score, chunk, vec, lineage));
                    }
                }
            }
        }
    }

    // 5. Dedup by chunk id, keeping highest score per chunk.
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

    // 6. MMR reranking
    let results = mmr(&raw_query_vec, candidates, top_k, mmr_lambda);
    Ok(results)
}

/// Maximal Marginal Relevance reranking.
fn mmr(
    query_vec: &[f32],
    mut candidates: Vec<(f32, store::Chunk, Vec<f32>, Vec<String>)>,
    top_k: usize,
    lambda: f32,
) -> Vec<ScoredChunk> {
    if lambda >= 1.0 || candidates.len() <= 1 {
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        return candidates
            .into_iter()
            .take(top_k)
            .map(|(score, chunk, _, lineage)| ScoredChunk { score, chunk, lineage })
            .collect();
    }

    let mut selected: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    let mut results: Vec<ScoredChunk> = Vec::with_capacity(top_k);

    while results.len() < top_k && !candidates.is_empty() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, (query_sim, _, vec, _)) in candidates.iter().enumerate() {
            let max_redundancy = selected
                .iter()
                .map(|sel| vdb::dot(vec, sel))
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
