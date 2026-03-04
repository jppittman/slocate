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
    let mmr_lambda = config.search.mmr_lambda;
    let bm25_weight = config.search.bm25_weight;
    // Fetch more candidates than top_k so MMR has a pool to diversify from.
    let n_candidates = (top_k * 8).max(64);
    let ef = n_candidates;

    let workspaces = config.expanded_workspaces();
    // Candidates: (hybrid_score, chunk, chunk_vector)
    let mut candidates: Vec<(f32, store::Chunk, Vec<f32>)> = Vec::new();

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

        // ── Semantic search via HNSW ──────────────────────────────────────
        let hits = hnsw.search(&query_vec, n_candidates, ef);

        // Build id→vector from HNSW nodes for MMR inter-result similarity.
        let vec_by_id: std::collections::HashMap<&str, &[f32]> = hnsw
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), n.vector.as_slice()))
            .collect();

        let semantic_scores: std::collections::HashMap<String, f32> =
            hits.iter().map(|h| (h.id.clone(), h.score)).collect();

        // ── BM25 search via FTS5 ──────────────────────────────────────────
        // Only run if bm25_weight > 0. Errors are non-fatal — fall back to
        // pure semantic (e.g. FTS5 not compiled in, malformed query).
        let bm25_scores: std::collections::HashMap<String, f32> = if bm25_weight > 0.0 {
            db.bm25_search(prompt, n_candidates).unwrap_or_default().into_iter().collect()
        } else {
            std::collections::HashMap::new()
        };

        // ── Merge candidate sets ──────────────────────────────────────────
        // Union: semantic hits (filtered by min_score unless BM25 also found
        // them) plus any BM25-only hits.
        let mut seen: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut candidate_ids: Vec<String> = Vec::new();

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

        if candidate_ids.is_empty() {
            continue;
        }

        let chunks = db.get_chunks_by_ids(&candidate_ids)?;
        let mut chunk_by_id: std::collections::HashMap<String, store::Chunk> =
            chunks.into_iter().map(|c| (c.id.clone(), c)).collect();

        for id in &candidate_ids {
            let cosine = semantic_scores.get(id).copied().unwrap_or(0.0);
            let bm25 = bm25_scores.get(id).copied().unwrap_or(0.0);
            let hybrid_score = bm25_weight * bm25 + (1.0 - bm25_weight) * cosine;

            // For BM25-only hits the HNSW vector may be absent; fall back to a
            // zero vector so MMR can still include the chunk (zero dot-product
            // means it competes on relevance alone, not redundancy).
            let vec = vec_by_id
                .get(id.as_str())
                .map(|v| v.to_vec())
                .unwrap_or_else(|| vec![0.0f32; query_vec.len()]);

            if let Some(chunk) = chunk_by_id.remove(id) {
                candidates.push((hybrid_score, chunk, vec));
            }
        }
    }

    // MMR reranking: iteratively select the candidate that maximises
    //   λ · sim(doc, query) − (1−λ) · max_j sim(doc, selected_j)
    // λ=1.0 → pure top-k, λ=0.0 → pure diversity.
    let results = mmr(&query_vec, candidates, top_k, mmr_lambda);
    Ok(results)
}

/// Maximal Marginal Relevance reranking.
///
/// `candidates` is a list of `(query_similarity, chunk, vector)` tuples,
/// pre-filtered by `min_score`. Returns up to `top_k` results ordered by
/// the MMR selection sequence (most relevant/diverse first).
fn mmr(
    query_vec: &[f32],
    mut candidates: Vec<(f32, store::Chunk, Vec<f32>)>,
    top_k: usize,
    lambda: f32,
) -> Vec<ScoredChunk> {
    // Pure top-k fast path — skip MMR bookkeeping entirely.
    if lambda >= 1.0 || candidates.len() <= 1 {
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        return candidates
            .into_iter()
            .take(top_k)
            .map(|(score, chunk, _)| ScoredChunk { score, chunk })
            .collect();
    }

    let _ = query_vec; // query similarity already in candidates[i].0
    let mut selected: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    let mut results: Vec<ScoredChunk> = Vec::with_capacity(top_k);

    while results.len() < top_k && !candidates.is_empty() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, (query_sim, _, vec)) in candidates.iter().enumerate() {
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

        let (score, chunk, vec) = candidates.swap_remove(best_idx);
        selected.push(vec);
        results.push(ScoredChunk { score, chunk });
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
