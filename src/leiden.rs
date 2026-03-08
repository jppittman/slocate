//! Leiden community detection on the HNSW layer-0 graph.
//!
//! Quality function: Constant Potts Model (CPM)
//!   Q = Σ_c [ w_c - γ · n_c · (n_c−1) / 2 ]
//! where w_c = total internal edge weight, n_c = |community c|, γ = resolution.
//!
//! Reference: Traag, Waltman, van Eck (2019) — "From Louvain to Leiden: guaranteeing
//! well-connected communities." Scientific Reports 9, 5233.
//!
//! Key improvement over Louvain: the refinement phase guarantees that every community
//! is internally well-connected (no "peninsular" nodes attached only through other
//! communities). Well-connectedness condition for merging singleton {v} into sub-community S:
//!   W(S∪{v}, C \ (S∪{v})) >= γ · |S∪{v}| · |C \ (S∪{v})|
//!
//! Algorithm per iteration:
//!   1. Local moving  — greedy CPM node reassignment until stable.
//!   2. Refinement    — within each coarse community, merge singletons into
//!                      well-connected sub-communities (single pass).
//!   3. Aggregation   — collapse refined sub-communities into super-nodes; recurse.

use std::collections::HashMap;

use crate::vdb::{self, Hnsw};

/// Resolution parameter γ. Smaller → larger communities.
/// 0.05 gives roughly "module/file level" clusters for code chunks.
pub const DEFAULT_GAMMA: f64 = 0.05;

// ─── Public output ────────────────────────────────────────────────────────────

/// Dense community assignment: `assignment[node_idx] = community_id` (0-based).
#[derive(Clone)]
pub struct Partition {
    pub assignment: Vec<usize>,
    pub n_communities: usize,
}

// ─── Internal graph ───────────────────────────────────────────────────────────

/// Pre-built semantic adjacency graph. Build once, run Leiden multiple times.
pub struct SemGraph {
    n: usize,
    /// Symmetrized adjacency list. `adj[v]` = `(neighbor, weight)` pairs.
    adj: Vec<Vec<(usize, f32)>>,
    /// Original node count represented by each super-node (all 1.0 at level 0;
    /// sums of constituent sizes after aggregation). Used for correct CPM scaling.
    node_size: Vec<f64>,
}

// Internal alias for the same type used within this module.
type Graph = SemGraph;

impl Graph {
    /// Build a dense semantic graph using all-to-all similarity computation.
    /// Uses chunked matrix multiplication to stay memory-safe on GPUs.
    /// `abs_sim`: if true, edges are included where |sim| > threshold (weight = |sim| − threshold).
    /// Used for the final graph after iterative centering — complementary pairs (negative cosine)
    /// attract just as similar pairs do, so they land in the same community.
    fn from_raw_vecs(vecs: &[Vec<f32>], mut threshold: f32, target_neighbors: usize, threshold_ceiling: f32, relative_weights: bool, device: &Device, source_paths: Option<&[&str]>, colocation_sigma: f32, colocation_lambda: f32, abs_sim: bool) -> Self {
        let n = vecs.len();
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];

        if n == 0 {
            return Self { n, adj, node_size: vec![1.0f64; n] };
        }

        let dim = vecs[0].len();
        log::info!("[leiden] building semantic graph for {n} nodes (dim={dim}) on {device:?}");

        // 1. Prepare vectors on device
        let mut flat_vecs = Vec::with_capacity(n * dim);
        for vec in vecs {
            flat_vecs.extend_from_slice(vec);
        }
        let v = match Tensor::from_vec(flat_vecs, (n, dim), device) {
            Ok(t) => t,
            Err(e) => {
                log::error!("[leiden] failed to load vectors to device: {e}");
                return Self { n, adj, node_size: vec![1.0f64; n] };
            }
        };
        let v_t = v.t().expect("transpose failed");

        // 2. Auto-thresholding (if threshold is 0.0)
        if threshold <= 0.0 {
            let sample_size = n.min(512);
            let v_sample = v.narrow(0, 0, sample_size).expect("narrow failed");
            let s_sample = v_sample.matmul(&v_t).expect("sample matmul failed");
            let mut flat_sample: Vec<f32> = s_sample.flatten_all().expect("flatten failed").to_vec1().expect("to_vec1 failed");

            // Filter out the diagonal (1.0 self-similarity) to avoid biasing the threshold.
            flat_sample.retain(|&x| x < 0.999);

            if relative_weights {
                // For bundle-vector graphs (level 1+), all pairwise similarities cluster
                // tightly (e.g. [0.6, 0.9]). Set threshold at the mean so only above-average
                // pairs get edges, and use sim - threshold as weight to give Leiden variance.
                let mean_sim = if flat_sample.is_empty() { 0.5 } else {
                    flat_sample.iter().sum::<f32>() / flat_sample.len() as f32
                };
                threshold = mean_sim.max(0.01);
                log::info!("[leiden] relative-weight threshold = mean_sim={:.4} (n={})", threshold, n);
            } else {
                flat_sample.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

                let effective_neighbors = if n < 500 {
                    (n / 4).min(target_neighbors).max(2)
                } else {
                    target_neighbors
                };

                let target_idx = (sample_size * effective_neighbors).min(flat_sample.len().saturating_sub(1));
                threshold = if flat_sample.is_empty() { 0.5 } else { flat_sample[target_idx] };
                threshold = threshold.min(threshold_ceiling).max(0.01);

                log::info!("[leiden] auto-threshold set to {:.4} (targeting ~{} neighbors/node, ceiling={:.2}, n={})", threshold, effective_neighbors, threshold_ceiling, n);
            }
        }

        // 3. Chunked computation to avoid OOM
        const CHUNK_SIZE: usize = 1024;
        for i_start in (0..n).step_by(CHUNK_SIZE) {
            let i_end = (i_start + CHUNK_SIZE).min(n);
            let chunk_len = i_end - i_start;
            
            let v_chunk = v.narrow(0, i_start, chunk_len).expect("narrow failed");
            let s_chunk = v_chunk.matmul(&v_t).expect("chunk matmul failed");
            
            // Download only this chunk's rows (N_chunk * N entries)
            let sims: Vec<f32> = s_chunk.flatten_all().expect("flatten failed").to_vec1().expect("to_vec1 failed");

            for i in 0..chunk_len {
                let global_i = i_start + i;
                let offset = i * n;
                for j in 0..n {
                    if global_i == j {
                        continue;
                    }
                    let sim = sims[offset + j];
                    let boost = if let Some(paths) = source_paths {
                        let dist = lca_distance(paths[global_i], paths[j]);
                        colocation_sigma * (-colocation_lambda * dist as f32).exp()
                    } else {
                        0.0
                    };
                    let effective = if abs_sim { sim.abs() } else { sim };
                    let weight = (effective - threshold) + boost;
                    if weight > 0.0 {
                        adj[global_i].push((j, weight));
                    }
                }
            }
            
            if n > 5000 {
                log::info!("[leiden] graph progress: {}/{}", i_end, n);
            }
        }

        let edge_count: usize = adj.iter().map(|v| v.len()).sum();
        let neg_count: usize = adj.iter().flat_map(|nbrs| nbrs.iter()).filter(|&&(_, w)| w < 0.0).count();
        let neg_pct = if edge_count > 0 { 100.0 * neg_count as f32 / edge_count as f32 } else { 0.0 };
        log::info!("[leiden] built graph with {} nodes, {} edges (avg degree {:.2}), neg={:.1}%", n, edge_count / 2, edge_count as f32 / n as f32, neg_pct);

        Self { n, adj, node_size: vec![1.0f64; n] }
    }

    /// Collapse the current graph according to `partition`, producing a new graph
    /// where each community is a super-node. Edge weights are summed.
    /// node_size is accumulated so CPM penalties stay in original-node units.
    fn aggregate(&self, partition: &[usize], n_comm: usize) -> Self {
        let mut edge_map: HashMap<(usize, usize), f64> = HashMap::new();
        let mut new_node_size = vec![0.0f64; n_comm];
        for v in 0..self.n {
            let cv = partition[v];
            new_node_size[cv] += self.node_size[v];
            for &(u, w) in &self.adj[v] {
                let cu = partition[u];
                if cv == cu {
                    continue; // internal edge
                }
                let key = if cv < cu { (cv, cu) } else { (cu, cv) };
                *edge_map.entry(key).or_insert(0.0) += w as f64;
            }
        }

        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_comm];
        for ((a, b), w) in edge_map {
            let wf = w as f32;
            adj[a].push((b, wf));
            adj[b].push((a, wf));
        }

        Self { n: n_comm, adj, node_size: new_node_size }
    }
}

// ─── Phase 1: Local moving ────────────────────────────────────────────────────

/// Greedy CPM node reassignment with randomised visit order. Returns `true` if any node moved.
///
/// ΔQ for moving v (original size s_v) from community c_old (total size n_old)
/// to c_new (total size n_new), sizes measured in original-node units:
///   ΔQ = w(v, c_new) − γ·s_v·n_new  −  ( w(v, c_old\{v}) − γ·s_v·(n_old−s_v) )
///
/// Randomised visit order breaks the sequential cascade that causes all nodes to
/// pile into the first large community. Uses a simple XorShift64 RNG seeded per call.
fn local_moving(graph: &Graph, partition: &mut Vec<usize>, gamma: f64, rng: &mut XorShift64) -> bool {
    let n = graph.n;

    // Community sizes in original-node units (correct CPM scaling across levels).
    let mut com_size: Vec<f64> = vec![0.0; n];
    for (i, &c) in partition.iter().enumerate() {
        com_size[c] += graph.node_size[i];
    }

    // Randomised visit order to break sequential cascade into one giant community.
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next() as usize) % (i + 1);
        order.swap(i, j);
    }

    let mut moved = false;

    for &v in &order {
        let c_old = partition[v];
        let s_v = graph.node_size[v];
        let n_old = com_size[c_old];

        // Tally edge weight from v to its own community (excluding v) and to others.
        let mut w_internal = 0.0f64;
        let mut candidates: HashMap<usize, f64> = HashMap::new();

        for &(u, w) in &graph.adj[v] {
            let cu = partition[u];
            if cu == c_old {
                w_internal += w as f64;
            } else {
                *candidates.entry(cu).or_insert(0.0) += w as f64;
            }
        }

        // CPM score of staying: w_internal − γ · s_v · (n_old − s_v)
        let score_stay = w_internal - gamma * s_v * (n_old - s_v);

        let mut best_c = c_old;
        let mut best_score = score_stay;

        for (c, w_to_c) in candidates {
            // CPM score of moving to c: w(v→c) − γ · s_v · n_c
            let score = w_to_c - gamma * s_v * com_size[c];
            // Epsilon prevents oscillation from tiny floating-point gains.
            if score > best_score + 1e-10 {
                best_score = score;
                best_c = c;
            }
        }

        if best_c != c_old {
            partition[v] = best_c;
            com_size[c_old] -= s_v;
            com_size[best_c] += s_v;
            moved = true;
        }
    }

    moved
}

/// Minimal XorShift64 RNG — no external dependency, deterministic, fast.
struct XorShift64 { state: u64 }
impl XorShift64 {
    fn new(seed: u64) -> Self { Self { state: seed.max(1) } }
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

// ─── Phase 2: Refinement ──────────────────────────────────────────────────────

/// Within each coarse community, start with singletons and greedily merge each
/// singleton {v} into a neighboring sub-community S if:
///   (a) CPM gain > 0: w(v, S) > γ · |S|
///   (b) Well-connectedness: W(S∪{v}, C\(S∪{v})) >= γ · |S∪{v}| · |C\(S∪{v})|
///
/// Only singletons are eligible to move (one pass per community).
/// Returns the refined partition and the number of sub-communities.
fn refine(graph: &Graph, coarse: &[usize], n_coarse: usize, gamma: f64, rng: &mut XorShift64) -> (Vec<usize>, usize) {
    let n = graph.n;

    // Group nodes by coarse community.
    let mut comm_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_coarse];
    for v in 0..n {
        comm_nodes[coarse[v]].push(v);
    }

    // `sub[v]`      = representative node of v's sub-community (initially v itself).
    // `sub_size[r]` = original-node total size of sub-community with representative r.
    // `w_ext[r]`    = Σ weight(r's subcommunity ↔ coarse_community \ r's subcommunity).
    let mut sub: Vec<usize> = (0..n).collect();
    let mut sub_size: Vec<f64> = graph.node_size.clone();
    let mut w_ext: Vec<f64> = vec![0.0; n];

    // Precompute w_ext for each singleton.
    for v in 0..n {
        let c_id = coarse[v];
        w_ext[v] = graph.adj[v]
            .iter()
            .filter(|&&(u, _)| coarse[u] == c_id)
            .map(|&(_, w)| w as f64)
            .sum();
    }

    for nodes in &comm_nodes {
        if nodes.len() <= 1 {
            continue;
        }
        let c_id = coarse[nodes[0]];
        let c_size: f64 = nodes.iter().map(|&v| graph.node_size[v]).sum();

        // Single pass over nodes in this coarse community.
        for &v in nodes {
            let sv = sub[v];
            let s_v = graph.node_size[v];

            // Only singletons (sub-community of original size = node_size[v]) move.
            if (sub_size[sv] - s_v).abs() > 1e-9 {
                continue;
            }

            // Aggregate edge weight from v to each neighboring sub-community within C.
            let mut candidates: HashMap<usize, f64> = HashMap::new();
            for &(u, w) in &graph.adj[v] {
                if coarse[u] != c_id {
                    continue;
                }
                let su = sub[u];
                if su != sv {
                    *candidates.entry(su).or_insert(0.0) += w as f64;
                }
            }

            // Collect all well-connected candidates with positive CPM gain.
            // leidenalg uses stochastic selection here: randomly choosing among
            // positive-gain sub-communities prevents the greedy cascade that collapses
            // all singletons into one giant sub-community.
            let mut eligible: Vec<(usize, f64)> = Vec::new();

            for (&s, &w_vs) in &candidates {
                let s_size = sub_size[s];
                let merged = s_size + s_v;
                let complement = c_size - merged;

                // Well-connectedness: W(S∪{v}, C\(S∪{v})) >= γ · merged · complement.
                let w_ext_merged = w_ext[s] + w_ext[sv] - 2.0 * w_vs;
                if complement > 1e-9 && w_ext_merged < gamma * merged * complement {
                    continue; // not well-connected
                }

                // CPM gain: w(v, S) − γ · s_v · s_size
                let gain = w_vs - gamma * s_v * s_size;
                if gain > 0.0 {
                    eligible.push((s, w_vs));
                }
            }

            if !eligible.is_empty() {
                // Random selection among positive-gain candidates.
                let chosen_idx = (rng.next() as usize) % eligible.len();
                let (best_s, w_vs) = eligible[chosen_idx];
                w_ext[best_s] = w_ext[best_s] + w_ext[sv] - 2.0 * w_vs;
                sub[v] = best_s;
                sub_size[best_s] += s_v;
                sub_size[sv] = 0.0;
            }
        }
    }

    let n_refined = relabel_in_place(&mut sub);
    (sub, n_refined)
}

use candle_core::{Device, Tensor};

// ─── File co-location helpers ─────────────────────────────────────────────────

/// Patristic (LCA) distance between two file paths.
/// dist(a, b) = depth(a) + depth(b) - 2 * depth(LCA(a, b))
/// Same file → 0, same directory → 2, one level apart → 4, etc.
fn lca_distance(a: &str, b: &str) -> usize {
    let a_parts: Vec<&str> = a.split('/').collect();
    let b_parts: Vec<&str> = b.split('/').collect();
    let lca_depth = a_parts.iter().zip(b_parts.iter())
        .take_while(|(x, y)| x == y)
        .count();
    (a_parts.len() + b_parts.len()).saturating_sub(2 * lca_depth)
}

// ─── Centering ────────────────────────────────────────────────────────────────

/// Center vectors within their communities: subtract community mean, L2-renormalize.
/// After centering, anti-correlated pairs within the same community produce negative
/// cosine similarity — enabling signed Leiden to fire.
fn center_vectors(vecs: &[Vec<f32>], assignment: &[usize], n_comm: usize) -> Vec<Vec<f32>> {
    let n = vecs.len();
    if n == 0 { return Vec::new(); }
    let dim = vecs[0].len();

    // Accumulate community means in f64 to avoid catastrophic cancellation.
    let mut means: Vec<Vec<f64>> = vec![vec![0.0; dim]; n_comm];
    let mut counts: Vec<f64> = vec![0.0; n_comm];
    for (i, &c) in assignment.iter().enumerate() {
        counts[c] += 1.0;
        for d in 0..dim {
            means[c][d] += vecs[i][d] as f64;
        }
    }
    for c in 0..n_comm {
        let inv = if counts[c] > 0.0 { 1.0 / counts[c] } else { 1.0 };
        for d in 0..dim { means[c][d] *= inv; }
    }

    // Subtract mean and renormalize each vector.
    let mut centered: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; n];
    for (i, &c) in assignment.iter().enumerate() {
        let mut norm_sq = 0.0f64;
        for d in 0..dim {
            let v = vecs[i][d] as f64 - means[c][d];
            centered[i][d] = v as f32;
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt() as f32;
        if norm > 1e-8 {
            for d in 0..dim { centered[i][d] /= norm; }
        }
        // If norm ≈ 0 (all identical within community), leave as-is — degenerate case.
    }
    centered
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/// Build the semantic graph once (expensive matmul), then run Leiden separately.
///
/// If `center_iters > 0`, performs iterative local mean-centering before the final
/// graph build: run Leiden at `center_gamma` → subtract community means → renormalize →
/// repeat. This breaks the BGE anisotropic cone so that anti-correlated pairs produce
/// negative edges when `signed = true`.
pub fn build_graph(nodes: &[vdb::HnswNode], threshold: f32, target_neighbors: usize, threshold_ceiling: f32, relative_weights: bool, device: &Device, source_paths: Option<&[&str]>, colocation_sigma: f32, colocation_lambda: f32, center_iters: usize, center_gamma: f64) -> SemGraph {
    if nodes.is_empty() {
        return SemGraph { n: 0, adj: Vec::new(), node_size: Vec::new() };
    }

    let mut vecs: Vec<Vec<f32>> = nodes.iter().map(|nd| nd.vector.clone()).collect();
    let n = vecs.len();

    if center_iters > 0 {
        let mut prev_n_comm: Option<usize> = None;
        let mut stable_for = 0usize;

        for iter in 0..center_iters {
            // Unsigned graph for intermediate partitioning — auto-threshold adapts to centered distribution.
            let g = Graph::from_raw_vecs(&vecs, 0.0, target_neighbors, threshold_ceiling, relative_weights, device, source_paths, colocation_sigma, colocation_lambda, false /* unsigned: only positive sim during centering iterations */);
            let part = run_leiden(SemGraph { n: g.n, adj: g.adj.clone(), node_size: g.node_size.clone() }, n, center_gamma);
            log::info!("[leiden] centering iter {}: {} communities", iter, part.n_communities);

            let nc = part.n_communities;
            if prev_n_comm == Some(nc) {
                stable_for += 1;
                if stable_for >= 2 {
                    log::info!("[leiden] centering converged (n_comm stable at {} for 2 iters)", nc);
                    // Apply one final centering pass before breaking.
                    vecs = center_vectors(&vecs, &part.assignment, nc);
                    break;
                }
            } else {
                stable_for = 0;
            }
            prev_n_comm = Some(nc);

            if nc < 2 {
                log::warn!("[leiden] centering: only {} community, cannot center meaningfully", nc);
                break;
            }
            vecs = center_vectors(&vecs, &part.assignment, nc);
        }
    }

    // After centering, similarity distribution shifts — the configured threshold is calibrated
    // for raw BGE vectors and would prune nearly all edges. Use auto-threshold instead.
    // Also use |sim| so complementary pairs (negative cosine after centering) attract rather
    // than repel — they belong in the same community, just express different poles of it.
    let final_threshold = if center_iters > 0 { 0.0 } else { threshold };
    let abs_sim = center_iters > 0;
    Graph::from_raw_vecs(&vecs, final_threshold, target_neighbors, threshold_ceiling, relative_weights, device, source_paths, colocation_sigma, colocation_lambda, abs_sim)
}

/// Compute soft community memberships from a converged Leiden partition.
///
/// For each node v, computes the fraction of its total edge weight that flows to each
/// non-primary community c. Returns (node_idx, community_id) pairs where that fraction
/// exceeds `min_fraction`. This measures: "what share of this node's graph connections
/// belong to community c?" — graph-derived, scale-invariant, gamma-independent.
///
/// At Leiden convergence the CPM delta to non-primary communities is always ≤ 0 by
/// definition, so a fraction-of-weight criterion is used rather than CPM score.
pub fn soft_memberships(graph: &SemGraph, partition: &Partition, min_fraction: f32) -> Vec<(usize, usize)> {
    let n = graph.n;
    if n == 0 { return Vec::new(); }

    let mut result = Vec::new();
    for v in 0..n {
        let primary = partition.assignment[v];
        let mut total_w = 0.0f64;
        let mut neighbor_weights: HashMap<usize, f64> = HashMap::new();

        for &(u, w) in &graph.adj[v] {
            let cu = partition.assignment[u];
            total_w += w as f64;
            if cu != primary {
                *neighbor_weights.entry(cu).or_insert(0.0) += w as f64;
            }
        }

        if total_w <= 0.0 { continue; }

        for (c, w_to_c) in neighbor_weights {
            if (w_to_c / total_w) as f32 >= min_fraction {
                result.push((v, c));
            }
        }
    }
    result
}

/// Run Leiden on a pre-built graph (no matmul). Reuse for gamma sweeps.
pub fn run_on_graph(graph: &SemGraph, n_orig: usize, gamma: f64) -> Partition {
    if graph.n == 0 {
        return Partition { assignment: Vec::new(), n_communities: 0 };
    }
    // Clone adjacency so Leiden can mutate its working state per gamma.
    let working = SemGraph { n: graph.n, adj: graph.adj.clone(), node_size: graph.node_size.clone() };
    run_leiden(working, n_orig, gamma)
}

fn run_leiden(mut graph: SemGraph, n: usize, gamma: f64) -> Partition {
    let mut super_node_of: Vec<usize> = (0..n).collect();
    let mut node_community: Vec<usize> = (0..n).collect();
    let mut prev_graph_n = graph.n;
    // Seed with gamma bits for reproducible but gamma-varied shuffles.
    let mut rng = XorShift64::new(gamma.to_bits() ^ 0xdeadbeefcafe1234);

    loop {
        let mut coarse: Vec<usize> = (0..graph.n).collect();
        loop {
            if !local_moving(&graph, &mut coarse, gamma, &mut rng) {
                break;
            }
        }
        let n_coarse = relabel_in_place(&mut coarse);
        for orig in 0..n {
            node_community[orig] = coarse[super_node_of[orig]];
        }
        if n_coarse >= graph.n {
            break;
        }
        let (refined, n_refined) = refine(&graph, &coarse, n_coarse, gamma, &mut rng);
        graph = graph.aggregate(&refined, n_refined);
        for orig in 0..n {
            super_node_of[orig] = refined[super_node_of[orig]];
        }
        if n_refined >= prev_graph_n {
            break;
        }
        prev_graph_n = n_refined;
    }

    let n_communities = relabel_in_place(&mut node_community);
    Partition { assignment: node_community, n_communities }
}

/// Leiden community detection on an arbitrary set of nodes.
pub fn run_on_nodes(nodes: &[vdb::HnswNode], gamma: f64, threshold: f32, target_neighbors: usize, threshold_ceiling: f32, relative_weights: bool, device: &Device) -> Partition {
    let n = nodes.len();
    if n == 0 {
        return Partition { assignment: Vec::new(), n_communities: 0 };
    }

    let vecs: Vec<Vec<f32>> = nodes.iter().map(|nd| nd.vector.clone()).collect();
    let graph = Graph::from_raw_vecs(&vecs, threshold, target_neighbors, threshold_ceiling, relative_weights, device, None, 0.0, 0.0, false);
    run_leiden(graph, n, gamma)
}

/// Leiden community detection on an HNSW index.
pub fn run(hnsw: &Hnsw, gamma: f64, threshold: f32, target_neighbors: usize, threshold_ceiling: f32, relative_weights: bool, device: &Device) -> Partition {
    run_on_nodes(&hnsw.nodes, gamma, threshold, target_neighbors, threshold_ceiling, relative_weights, device)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vdb::{dot, Hnsw};

    fn build_hnsw(vecs: &[Vec<f32>], ids: &[&str]) -> Hnsw {
        // M=4 so small graphs still get well-connected layer-0 graphs.
        let mut hnsw = Hnsw::with_params(4, 20, dot);
        for (id, vec) in ids.iter().zip(vecs.iter()) {
            hnsw.insert(id, vec.clone());
        }
        hnsw
    }

    #[test]
    fn empty_graph() {
        let hnsw = Hnsw::new(dot);
        let p = run(&hnsw, DEFAULT_GAMMA, 0.0, 15, 0.90, false, &Device::Cpu);
        assert_eq!(p.n_communities, 0);
        assert!(p.assignment.is_empty());
    }

    #[test]
    fn single_node() {
        let mut hnsw = Hnsw::new(dot);
        hnsw.insert("only", vec![1.0, 0.0]);
        let p = run(&hnsw, DEFAULT_GAMMA, 0.0, 15, 0.90, false, &Device::Cpu);
        assert_eq!(p.n_communities, 1);
        assert_eq!(p.assignment, vec![0]);
    }

    #[test]
    fn two_disconnected_clusters() {
        // 6 unit 2D vectors: 3 near (1,0), 3 near (-1,0).
        // Within-cluster cosine ~ 0.94; between-cluster cosine ~ -0.94.
        // HNSW layer-0 should have no cross-cluster edges → 2 communities.
        let vecs: Vec<Vec<f32>> = vec![
            vec![ 1.000_f32,  0.000],  // 0
            vec![ 0.940,      0.342],  // 1  } cluster A
            vec![ 0.940,     -0.342],  // 2
            vec![-1.000,      0.000],  // 3
            vec![-0.940,      0.342],  // 4  } cluster B
            vec![-0.940,     -0.342],  // 5
        ];
        let ids = &["a0", "a1", "a2", "b0", "b1", "b2"];
        let hnsw = build_hnsw(&vecs, ids);

        let p = run(&hnsw, DEFAULT_GAMMA, 0.0, 15, 0.90, false, &Device::Cpu);

        assert_eq!(p.n_communities, 2, "expected 2 communities, got {}", p.n_communities);
        // All A nodes in one community.
        assert_eq!(p.assignment[0], p.assignment[1], "a0 and a1 should share a community");
        assert_eq!(p.assignment[1], p.assignment[2], "a1 and a2 should share a community");
        // All B nodes in the other.
        assert_eq!(p.assignment[3], p.assignment[4], "b0 and b1 should share a community");
        assert_eq!(p.assignment[4], p.assignment[5], "b1 and b2 should share a community");
        // Clusters must be distinct.
        assert_ne!(p.assignment[0], p.assignment[3], "clusters A and B must be different");
    }

    #[test]
    fn three_clusters() {
        // 9 unit 2D vectors at 0°, 120°, 240°, with 3 nodes per cluster.
        use std::f32::consts::PI;
        let angles: &[f32] = &[0.0, 120.0, 240.0];
        let spread: &[f32] = &[-10.0, 0.0, 10.0]; // ±10° within each cluster
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for &base in angles {
            for &offset in spread {
                let rad = (base + offset) * PI / 180.0;
                vecs.push(vec![rad.cos(), rad.sin()]);
            }
        }
        let ids: Vec<String> = (0..9).map(|i| i.to_string()).collect();
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let hnsw = build_hnsw(&vecs, &id_refs);

        let p = run(&hnsw, DEFAULT_GAMMA, 0.0, 15, 0.90, false, &Device::Cpu);

        // Should recover 3 clusters. The exact label assignments may vary, but
        // all nodes within a cluster (same base angle) must share a community.
        assert_eq!(p.n_communities, 3, "expected 3 communities, got {}", p.n_communities);
        assert_eq!(p.assignment[0], p.assignment[1]);
        assert_eq!(p.assignment[1], p.assignment[2]);
        assert_eq!(p.assignment[3], p.assignment[4]);
        assert_eq!(p.assignment[4], p.assignment[5]);
        assert_eq!(p.assignment[6], p.assignment[7]);
        assert_eq!(p.assignment[7], p.assignment[8]);
        assert_ne!(p.assignment[0], p.assignment[3]);
        assert_ne!(p.assignment[3], p.assignment[6]);
        assert_ne!(p.assignment[0], p.assignment[6]);
    }

    #[test]
    fn all_identical_vectors_form_one_community() {
        // All vectors the same → maximal internal similarity → one community.
        let vecs: Vec<Vec<f32>> = vec![vec![1.0f32, 0.0]; 10];
        let ids: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let hnsw = build_hnsw(&vecs, &id_refs);

        let p = run(&hnsw, DEFAULT_GAMMA, 0.0, 15, 0.90, false, &Device::Cpu);
        // All 10 should end up in the same community.
        let first = p.assignment[0];
        assert!(
            p.assignment.iter().all(|&c| c == first),
            "all identical vectors should be in one community: {:?}",
            p.assignment
        );
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Relabel community ids to a dense 0..k range.
/// Returns the number of distinct communities k.
fn relabel_in_place(partition: &mut Vec<usize>) -> usize {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    for c in partition.iter_mut() {
        let new_c = *map.entry(*c).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        *c = new_c;
    }
    next
}
