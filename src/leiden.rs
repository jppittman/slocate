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
pub struct Partition {
    pub assignment: Vec<usize>,
    pub n_communities: usize,
}

// ─── Internal graph ───────────────────────────────────────────────────────────

struct Graph {
    n: usize,
    /// Symmetrized adjacency list. `adj[v]` = `(neighbor, weight)` pairs.
    adj: Vec<Vec<(usize, f32)>>,
}

impl Graph {
    /// Build from HNSW layer-0 neighbors. Edges are symmetrized and deduplicated
    /// (taking the max weight when HNSW produces both directions).
    fn from_hnsw(hnsw: &Hnsw) -> Self {
        let n = hnsw.nodes.len();
        // Use edge map to symmetrize and dedup.
        let mut edge_map: HashMap<(usize, usize), f32> = HashMap::new();
        for (i, node) in hnsw.nodes.iter().enumerate() {
            let neighbors = node.neighbors.get(0).map(|v| v.as_slice()).unwrap_or(&[]);
            for &j in neighbors {
                if i == j {
                    continue;
                }
                let sim = vdb::dot(&node.vector, &hnsw.nodes[j].vector);
                let key = if i < j { (i, j) } else { (j, i) };
                let e = edge_map.entry(key).or_insert(0.0);
                *e = e.max(sim);
            }
        }

        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for ((i, j), w) in edge_map {
            adj[i].push((j, w));
            adj[j].push((i, w));
        }

        Self { n, adj }
    }

    /// Collapse the current graph according to `partition`, producing a new graph
    /// where each community is a super-node. Edge weights are summed.
    fn aggregate(&self, partition: &[usize], n_comm: usize) -> Self {
        let mut edge_map: HashMap<(usize, usize), f64> = HashMap::new();
        for v in 0..self.n {
            let cv = partition[v];
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

        Self { n: n_comm, adj }
    }
}

// ─── Phase 1: Local moving ────────────────────────────────────────────────────

/// Greedy CPM node reassignment. Returns `true` if any node moved.
///
/// ΔQ for moving v from c_old (of size n_old) to c_new (of size n_new):
///   ΔQ = w(v, c_new) − γ·n_new  −  ( w(v, c_old\{v}) − γ·(n_old−1) )
/// Move iff ΔQ > 0.
fn local_moving(graph: &Graph, partition: &mut Vec<usize>, gamma: f64) -> bool {
    let n = graph.n;
    let mut com_size: Vec<usize> = vec![0; n];
    for &c in partition.iter() {
        com_size[c] += 1;
    }

    let mut moved = false;

    for v in 0..n {
        let c_old = partition[v];
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

        // Score of staying put.
        let score_stay = w_internal - gamma * (n_old as f64 - 1.0);

        let mut best_c = c_old;
        let mut best_score = score_stay;

        for (c, w_to_c) in candidates {
            let score = w_to_c - gamma * com_size[c] as f64;
            // Use a small epsilon to prevent oscillation from tiny floating-point gains.
            if score > best_score + 1e-10 {
                best_score = score;
                best_c = c;
            }
        }

        if best_c != c_old {
            partition[v] = best_c;
            com_size[c_old] -= 1;
            com_size[best_c] += 1;
            moved = true;
        }
    }

    moved
}

// ─── Phase 2: Refinement ──────────────────────────────────────────────────────

/// Within each coarse community, start with singletons and greedily merge each
/// singleton {v} into a neighboring sub-community S if:
///   (a) CPM gain > 0: w(v, S) > γ · |S|
///   (b) Well-connectedness: W(S∪{v}, C\(S∪{v})) >= γ · |S∪{v}| · |C\(S∪{v})|
///
/// Only singletons are eligible to move (one pass per community).
/// Returns the refined partition and the number of sub-communities.
fn refine(graph: &Graph, coarse: &[usize], n_coarse: usize, gamma: f64) -> (Vec<usize>, usize) {
    let n = graph.n;

    // Group nodes by coarse community.
    let mut comm_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_coarse];
    for v in 0..n {
        comm_nodes[coarse[v]].push(v);
    }

    // `sub[v]`     = representative node of v's sub-community (initially v itself).
    // `sub_size[r]` = size of the sub-community whose representative is r.
    // `w_ext[r]`   = Σ weight(r's subcommunity ↔ coarse_community \ r's subcommunity).
    let mut sub: Vec<usize> = (0..n).collect();
    let mut sub_size: Vec<usize> = vec![1; n];
    let mut w_ext: Vec<f64> = vec![0.0; n];

    // Precompute w_ext for each singleton: total weight from v to the rest of its coarse comm.
    for v in 0..n {
        let c_id = coarse[v];
        w_ext[v] = graph.adj[v]
            .iter()
            .filter(|&&(u, _)| coarse[u] == c_id)
            .map(|&(_, w)| w as f64)
            .sum();
    }

    for nodes in &comm_nodes {
        let c_size = nodes.len();
        if c_size <= 1 {
            continue;
        }
        let c_id = coarse[nodes[0]];

        // Single pass over nodes in this coarse community.
        for &v in nodes {
            let sv = sub[v];

            // Only singletons move.
            if sub_size[sv] != 1 {
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

            let mut best_s = sv;
            let mut best_gain = 0.0f64;

            for (&s, &w_vs) in &candidates {
                let s_size = sub_size[s];
                let merged = s_size + 1; // +1 for singleton {v}
                let complement = c_size.saturating_sub(merged);

                // Well-connectedness: W(S∪{v}, C\(S∪{v})) >= γ · merged · complement.
                // W(merged, complement) = w_ext[s] + w_ext[sv] − 2·w(v,S)
                let w_ext_merged = w_ext[s] + w_ext[sv] - 2.0 * w_vs;
                if complement > 0 && w_ext_merged < gamma * merged as f64 * complement as f64 {
                    continue; // not well-connected
                }

                // CPM gain: w(v, S) − γ · 1 · |S|
                let gain = w_vs - gamma * s_size as f64;
                if gain > best_gain {
                    best_gain = gain;
                    best_s = s;
                }
            }

            if best_s != sv {
                let w_vs = candidates[&best_s];
                // Update w_ext for the merged sub-community.
                w_ext[best_s] = w_ext[best_s] + w_ext[sv] - 2.0 * w_vs;
                // Absorb singleton sv into best_s.
                sub[v] = best_s;
                sub_size[best_s] += 1;
                sub_size[sv] = 0; // sv is now empty
            }
        }
    }

    let n_refined = relabel_in_place(&mut sub);
    (sub, n_refined)
}

// ─── Main entry point ─────────────────────────────────────────────────────────

pub fn run(hnsw: &Hnsw, gamma: f64) -> Partition {
    let n = hnsw.nodes.len();
    if n == 0 {
        return Partition { assignment: Vec::new(), n_communities: 0 };
    }

    let mut graph = Graph::from_hnsw(hnsw);

    // `super_node_of[orig]` = which node in the current aggregated graph
    // corresponds to original node `orig`. Initially the identity.
    let mut super_node_of: Vec<usize> = (0..n).collect();

    // `node_community[orig]` = final coarse community of `orig`.
    // Updated at each level from the coarse (local-moving) partition.
    let mut node_community: Vec<usize> = (0..n).collect();

    let mut prev_graph_n = graph.n;

    loop {
        // ── 1. Local moving ──────────────────────────────────────────────────
        let mut coarse: Vec<usize> = (0..graph.n).collect();
        loop {
            if !local_moving(&graph, &mut coarse, gamma) {
                break;
            }
        }
        let n_coarse = relabel_in_place(&mut coarse);

        // Commit the coarse partition as the current best community assignment.
        for orig in 0..n {
            node_community[orig] = coarse[super_node_of[orig]];
        }

        // Converged: every node is its own community — no grouping happened.
        if n_coarse >= graph.n {
            break;
        }

        // ── 2. Refinement ────────────────────────────────────────────────────
        let (refined, n_refined) = refine(&graph, &coarse, n_coarse, gamma);

        // ── 3. Aggregation ───────────────────────────────────────────────────
        graph = graph.aggregate(&refined, n_refined);

        // Update super_node_of to point into the new aggregated graph.
        for orig in 0..n {
            super_node_of[orig] = refined[super_node_of[orig]];
        }

        // Converged: aggregation didn't reduce the graph.
        if n_refined >= prev_graph_n {
            break;
        }
        prev_graph_n = n_refined;
    }

    let n_communities = relabel_in_place(&mut node_community);
    Partition { assignment: node_community, n_communities }
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
        let p = run(&hnsw, DEFAULT_GAMMA);
        assert_eq!(p.n_communities, 0);
        assert!(p.assignment.is_empty());
    }

    #[test]
    fn single_node() {
        let mut hnsw = Hnsw::new(dot);
        hnsw.insert("only", vec![1.0, 0.0]);
        let p = run(&hnsw, DEFAULT_GAMMA);
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

        let p = run(&hnsw, DEFAULT_GAMMA);

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

        let p = run(&hnsw, DEFAULT_GAMMA);

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

        let p = run(&hnsw, DEFAULT_GAMMA);
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
