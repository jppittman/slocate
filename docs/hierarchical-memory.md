# Hierarchical Memory Architecture

A design for replacing slocate's flat HNSW index with a three-level hierarchical
memory model inspired by VSA, Leiden community detection, and complementary
learning systems theory.

---

## The Problem with the Current Architecture

The current design runs a single global HNSW index over all chunks and uses Leiden
post-hoc to label results. Two issues:

1. **Leiden labels are decorative.** Community IDs appear in output but don't
   influence retrieval. The structure Leiden discovers is thrown away after labeling.

2. **HNSW and Leiden are redundant.** Both are discovering the same latent
   hierarchical structure in the data — small-world communities with hub nodes
   bridging them. HNSW finds this implicitly via random level assignment. Leiden
   finds it explicitly by optimizing a graph quality criterion. We're doing the
   work twice and using neither result well.

---

## Core Insight: Leiden Gives You the Optimal Sharding

In VSA (Vector Symbolic Architecture), superposing N vectors into a bundle gives
signal-to-noise ratio ≈ √(D/N). At D=384 (BGE-small) and N=10,000 chunks,
SNR ≈ 0.06 — indistinguishable from noise. Superposition over the full corpus
is useless.

But Leiden partitions the corpus into communities. If the mean community size
is ~50 chunks, each shard bundle has SNR ≈ √(384/50) ≈ 2.8 — clean enough
for reliable similarity queries. The community structure isn't just a label;
it's telling you the granularity at which superposition becomes tractable.

**Leiden is the optimal sharding strategy for VSA.**

---

## The Three-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│  Routing Layer                                       │
│  N_communities VSA bundles (superposition of shard) │
│  Query cost: O(N_communities) dot products           │
└───────────────────┬─────────────────────────────────┘
                    │ top-k shards
┌───────────────────▼─────────────────────────────────┐
│  Shard Layer                                         │
│  One local HNSW per community (layer-0 only)         │
│  Query cost: O(log N_shard) per shard                │
└───────────────────┬─────────────────────────────────┘
                    │ candidates
┌───────────────────▼─────────────────────────────────┐
│  Chunk Layer                                         │
│  Raw embeddings + source text + metadata             │
│  MMR reranking across shard results                  │
└─────────────────────────────────────────────────────┘
```

**Query flow:**

1. Embed query → 384-dim vector
2. Dot product against all community bundles → rank shards → take top-k
3. Run local HNSW search within each matched shard
4. Merge candidate sets, apply MMR, return top results

This replaces the multi-layer global HNSW. The hierarchy that HNSW was building
across layers is now explicit: community bundles handle coarse routing,
local HNSW handles fine-grained search within a region.

---

## Why Drop Multi-Layer HNSW

HNSW's upper layers exist to solve the same routing problem that community bundles
solve. High-layer nodes in HNSW are exactly the community hubs Leiden identifies —
they're the high-degree nodes that bridge local neighborhoods. HNSW discovers them
stochastically (insertion order + random level sampling). Leiden discovers them from
the graph structure.

With explicit community routing above, within-shard indexes only need layer 0:
dense local connections, no long-range shortcuts needed. Smaller indexes, faster
construction, faster search.

---

## Community Bundles

A community bundle is the L2-normalized sum of its member vectors:

```
bundle_c = normalize( Σ_{i ∈ community_c} v_i )
```

Properties inherited from VSA superposition:
- `dot(query, bundle_c)` is proportional to the average similarity of the query
  to community members — it measures how "in this neighborhood" the query is
- Bundles are incrementally updatable: adding a chunk is `bundle += normalize(v_new)`
  followed by renormalization. O(1), no reindex needed
- Subtraction approximates removal, though it drifts with many operations (see Sleep)

The community hub (highest within-community degree node) is a useful companion:
it's the most semantically central chunk, and its text serves as a human-readable
community label. The bundle is the routing signal; the hub is the explanation.

---

## Wake / Sleep

The system operates in two modes, analogous to complementary learning systems:

### Wake (incremental, online)

Triggered by: file change, daemon reindex tick, `slocate reindex`

- Embed changed chunks
- Insert into the appropriate shard's local HNSW (community assignment from
  the existing Leiden partition — no Leiden rerun)
- Update the shard bundle: `bundle += new_vec; normalize(bundle)`
- Fast. Doesn't touch community structure.

Tradeoff: bundles drift slightly from true superposition as vectors accumulate
without a full recompute. Acceptable for moderate update volumes.

### Sleep (consolidation, offline)

Triggered by: scheduled maintenance window, large batch reindex, explicit `slocate gc`

- Re-run Leiden on the full layer-0 graph → discover if community structure
  has shifted (files added/deleted, codebase restructured)
- Rebalance shards: split communities that have grown beyond the VSA SNR
  threshold, merge communities that have shrunk below minimum useful size
- Recompute bundles from scratch (clears accumulated drift)
- Rebuild local HNSW indexes cleanly (removes tombstones from deletions,
  re-optimizes edge set)
- Recompute hub vectors

Sleep is the existing `reindex` pipeline, made semantically explicit. The
systemd/launchd timer is already a sleep schedule.

**`slocate gc` is the glymphatic system** — runs during or after sleep,
clears orphaned embeddings, stale registry links, and chunks whose source
files no longer exist.

---

## Shard Rebalancing Policy

Leiden's resolution parameter γ controls community granularity. Rather than
a fixed γ, use an adaptive value:

```
γ = mean_layer0_edge_weight × 0.8
```

This self-scales to the embedding space — if all chunks in a repo are
semantically similar (a focused library), γ rises and communities stay fine-grained.
If the repo spans many topics, γ falls and communities broaden.

A shard should be split when: `N_shard > D / min_snr²`
At D=384, min_snr=2: split when `N_shard > 96`. Merge when `N_shard < 10`.

---

## Open Questions

**VSA binding for compositional queries.** Superposition (bundles) works on raw
BGE vectors. Binding — `bind(file_symbol, content_vec)` to represent "this
concept within this file" — requires quasi-orthogonal symbol atoms, which BGE
vectors aren't. A fixed random projection to a larger HDC space would enable
binding, but it's a bigger lift and the use case (structured queries like
"async functions in the search module") may be better served by metadata
filtering.

**Bundle drift tolerance.** How many incremental wake-cycle updates before
bundle fidelity degrades enough to hurt routing accuracy? Needs empirical
measurement. Hypothesis: fine for up to ~20% shard membership turnover;
sleep should trigger at that threshold.

**Shard HNSW vs. brute force.** For small shards (N < ~200), brute-force
cosine over shard members is faster than HNSW traversal. Cutover should
be dynamic.

**Conceptualizing communities.** The sleep phase does generalize in the
relevant sense: new chunks shift community structure, Leiden reruns, and
the routing layer gets updated abstract representations reflecting patterns
across many specific examples — specific-to-abstract, bottom-up. What is
not yet present is *conceptualization*: synthesizing a natural language
proposition about what a community means. The bundle and hub give a routing
signal and a representative example, but not a claim. "This community is
about concurrent queue implementations" requires a generative model reading
the members and producing that description. The structural generalization is
in scope; language-level semantics are the next layer above.

The zero-shot approximation: pre-embed a vocabulary of programming terms,
find terms with highest cosine similarity to the bundle. No generation
required, interpretable output, zero training cost. Good enough for labels;
not the same as understanding.
