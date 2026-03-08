# slocate Architecture

slocate is a semantic code search tool that acts as a RAG context provider for LLM coding assistants. It parses source files into semantic chunks, embeds them with a local BERT model, organizes them into a hierarchical community structure using Leiden clustering, and retrieves the most relevant chunks at query time using a combination of path-vector matching, exact cosine search, BM25, and MMR reranking.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Pipeline](#data-pipeline)
3. [Parsing](#parsing)
4. [Embedding](#embedding)
5. [Leiden Community Detection](#leiden-community-detection)
6. [Hierarchical VSA Tree](#hierarchical-vsa-tree)
7. [Index Files](#index-files)
8. [Query Execution](#query-execution)
9. [VSA Concepts Used](#vsa-concepts-used)
10. [Key Design Decisions](#key-design-decisions)
11. [Module Reference](#module-reference)
12. [Configuration Reference](#configuration-reference)

---

## System Overview

```
Workspaces (source files)
        │
        ▼
  Phase 1: classify (mtime+size check)
        │
        ▼
  Phase 2: delete stale chunks from DB
        │
        ▼
  Phase 3+4: parse → embed (parallel SPSC workers) → commit to SQLite
        │
        ▼
  Phase 5: Leiden VSA tree rollup
        │   chunks → L0 bundles → L1 bundles → ... → root
        │   gamma sweep per level (graph built once, Leiden run at multiple gammas)
        │   secondary community membership via VSA stability analysis
        │
        ▼
  Phase 6: write binary matrix files
        │   bundle_matrix_L{n}.bin per level
        │   leaf_paths.bin (Hadamard path vectors for L0 bundles)
        │
        ▼
  Query time
        │   embed query → center on global mean
        │   flat scan over leaf_paths.bin → top beam_width L0 bundles
        │   exact cosine search within chosen bundles (primary + secondary members)
        │   BM25 boost (hybrid score blend)
        │   MMR reranking
        │
        ▼
  Formatted results → LLM context
```

---

## Data Pipeline

### Phase 1: Classify

`classify_files` walks the workspace directory tree (skipping `target`, `.git`, `.slocate`, `vendor`, `node_modules`, `Library`, `Applications`, `.Trash`) and compares each file's `mtime_ns` and `size` against the `file_meta` table. Files are sorted into three buckets:

- **to_embed**: new or modified files (parsed here, embedded in Phase 3)
- **unchanged**: mtime+size match the stored record
- **deleted**: present in `file_meta` but gone from disk

Only extensions listed in `config.index.extensions` are considered (default: `rs`, `py`, `ts`, `go`, `c`, `h`, `md`, `yaml`, `yml`, `bzl`). Files larger than `max_file_bytes` (default 1 MB) are skipped.

### Phase 2: Delete

Stale chunks for deleted and modified files are removed from the `chunks` table and the FTS5 index. `file_meta` rows for deleted files are also purged.

### Phase 3+4: Parallel Embed and Commit

N worker threads (1 for GPU, up to `min(parallelism, 4)` for CPU) steal 50-file batches from a shared atomic counter. Each worker:

1. Formats each chunk's embed text as `"{kind} {name} in {rel_path}\n\n{embed_text}"`.
2. Computes FNV-1a content hashes and checks the `embed_cache` table for hits.
3. Calls `embed_batch` for cache misses (batched forward pass through BGE).
4. Sends a `BatchResult` to the main thread via a wait-free SPSC channel.

The main thread polls all per-worker SPSC receivers and flushes batches to SQLite in coalesced transactions (every 8 batches or when workers go idle). A `CancelGuard` sets an atomic cancellation flag on any scope exit so workers stop stealing new batches after an error.

Chunk IDs are a djb2 hash of `(rel_path, name, kind)` — stable across reindex runs.

### Phase 5: Leiden VSA Tree Rollup

After all chunks are embedded, the full Leiden tree is rebuilt from scratch. See the [Hierarchical VSA Tree](#hierarchical-vsa-tree) section for details.

### Phase 6: Write Binary Matrix Files

Bundle vectors and leaf path vectors are written to flat binary files for fast query-time scoring. See [Index Files](#index-files).

---

## Parsing

`parse.rs` extracts semantic chunks from source files.

**Supported languages and node types:**

| Extension | Parser | Chunk types |
|-----------|--------|-------------|
| `.rs` | tree-sitter-rust | function, struct, enum, trait, impl, mod |
| `.py` | tree-sitter-python | function, class |
| `.go` | tree-sitter-go | function, method, type_decl |
| `.yaml` / `.yml` | tree-sitter-yaml | block_mapping_pair |
| `.bzl` / `BUILD` / `WORKSPACE` | tree-sitter-starlark | function, expression_statement (rule calls) |
| `.md` / `.markdown` | custom | section (split at `#` headings) |

For tree-sitter languages, `parse_with_treesitter` runs a language-specific query to identify landmark nodes (named functions, structs, classes, etc.) and then traverses the AST iteratively with a chunking strategy:

- **Small nodes** (byte length ≤ 2000): emitted as a single chunk.
- **Terminal landmark nodes** (non-container): emitted whole even if large; embed text is truncated to 2000 bytes.
- **Container landmarks** (impl, class, module, trait, section) or oversized non-landmarks: children are recursed into. Adjacent small non-landmark children are bucketed together into one chunk.

The result is a `Vec<RawChunk>` with fields `name` (hierarchical, e.g. `"MyStruct > my_method"`), `source` (full text), `embed_text` (truncated to 2000 bytes), and `kind`.

For Markdown, sections are split at `#` headings; the heading text becomes the chunk name.

---

## Embedding

`embed.rs` wraps BGE-small-en-v1.5 (33M parameter BERT, 384-dim output) loaded via `candle` with mmap'd safetensors weights. No Python, no subprocess.

**Single embed (`embed`):**
1. Tokenize with the BGE tokenizer, truncate to 512 tokens.
2. Forward pass: `[1, seq_len, 384]` hidden states.
3. Masked mean pooling over non-padding positions.
4. L2-normalize → `Vec<f32>` of length 384.

**Batch embed (`embed_batch`):**
Same as single, but all inputs are padded to the longest sequence and run in one forward pass of shape `[batch, max_seq, 384]`. Sub-batched at 64 to bound GPU memory.

All output vectors are L2-normalized at emit time, so cosine similarity = dot product throughout the system.

**Device selection:**
- Linux: CUDA if available, else CPU with MKL.
- macOS: CPU with Accelerate (Metal only if `SLOCATE_DEVICE=metal`).
- Override with `SLOCATE_DEVICE=cpu`.

In MCP server mode (`serve`), if the index embedder is on GPU, a separate CPU embedder is allocated for query-time embedding to avoid contention.

**Embed cache:**
Vectors are cached in SQLite keyed by FNV-1a hash of the formatted embed text. The cache stores f16 BLOBs and survives across reindex runs. Entries older than 30 days are pruned by `gc`.

---

## Leiden Community Detection

`leiden.rs` implements the Leiden algorithm with the Constant Potts Model (CPM) quality function:

```
Q = Σ_c [ w_c − γ · n_c · (n_c − 1) / 2 ]
```

where `w_c` is total internal edge weight and `n_c` is community size. Higher `γ` yields smaller, more granular communities.

### Graph Construction

`build_graph` computes all-pairs cosine similarities via GPU-accelerated chunked matrix multiplication (chunk size 1024 rows to avoid OOM). An edge `(i, j)` exists when `sim(i, j) > threshold`, with weight `sim - threshold`.

**Auto-thresholding** (when `leiden_threshold = 0.0`, the default):
- At level 0 (chunks): sample 512 nodes, collect all pairwise similarities excluding the diagonal, sort descending, take the value at index `sample_size × target_neighbors` as threshold. Capped at `threshold_ceiling` (default 0.90).
- At level 1+ (bundle vectors): use relative weighting — threshold = mean pairwise similarity, weight = `sim - mean`. This gives Leiden meaningful variance in an otherwise tight similarity distribution.

### Algorithm Phases (per iteration)

**Local moving:** For each node `v`, compute the CPM gain of moving to each neighboring community vs. staying put:

```
ΔQ(v → c_new) = w(v, c_new) − γ·|c_new|  −  (w(v, c_old\{v}) − γ·(|c_old| − 1))
```

Move if `ΔQ > 1e-10`. Repeat until stable.

**Refinement:** Within each coarse community, start from singletons. Greedily merge singleton `{v}` into neighboring sub-community `S` if:
- CPM gain: `w(v, S) > γ · |S|`
- Well-connectedness: `W(S∪{v}, C\(S∪{v})) ≥ γ · |S∪{v}| · |C\(S∪{v})|`

**Aggregation:** Collapse refined sub-communities into super-nodes (sum edge weights across collapsed boundaries). Recurse on the aggregated graph until no further merging occurs.

The graph is built **once per level** and reused across the gamma sweep — graph structure is gamma-independent, only the CPM objective changes.

---

## Hierarchical VSA Tree

Phase 5 of reindex builds a recursive hierarchy of communities. Each iteration takes the current set of nodes (chunks at level 0, bundle centroids at level 1+), clusters them with Leiden, creates bundle records, and repeats with the bundles as the next level's nodes. The loop terminates when the node count reaches 1 or fails to shrink.

### Level 0: Gamma Sweep and Secondary Membership

At level 0, a gamma sweep finds the partition closest to the branching factor target (`n_nodes / branching_factor`, default 10x reduction). The sweep reduces gamma by `leiden_gamma_step` (default 0.5×) each iteration until the community count falls into the target range or gamma drops below 0.0001.

**Graph is built once before the sweep.** All gamma trials reuse the same `SemGraph`.

**VSA membership accumulation** runs in parallel with the sweep. For each "useful" partition (more than one community, fewer than n_nodes):

```rust
// For each community in this trial partition:
let trial_bundle = normalize(Σ member_vecs);  // per-community superposition

// Accumulate into each node's membership vector:
membership_accum[i] += trial_bundle[assignment[i]];
```

After the sweep, the accumulated vector for each node is L2-normalized to produce a `membership_vec`. This vector points toward the centroid of the node's average community across all useful gamma trials.

**Secondary community assignment:** For each chunk `i` with primary community `p`, check all other bundles `b`:

```rust
let score = dot(membership_vec[i], bundle_vec[b]);
if score > 0.3 {
    // chunk i is also a secondary member of bundle b
}
```

These pairs are stored in `chunk_secondary_communities`. At query time, when a bundle is selected, its chunk set includes both primary and secondary members.

### Bundle Computation (all levels)

For each community at any level:

1. **Superposition vector** (the bundle): `normalize(Σ member_vecs)` — the mean direction of all members.
2. **Hub** (prototype): the member with highest dot product against the bundle vector — the most representative node.
3. The bundle is stored with both the normalized vector and the raw sum (for potential re-normalization).

The hub's name is stored in the `hubs` field and used to populate lineage labels in search results.

### Parent Pointer Assignment

- Level 0 bundles: `chunks.community_id` is set to the bundle ID.
- Level 1+ bundles: `bundles.parent_id` is set to the parent bundle ID.

This forms a tree rooted at bundles with `parent_id IS NULL`.

### Leaf Path Vectors

After the full tree is built, `leaf_paths.bin` is computed for all L0 bundles. For each L0 bundle, the path vector is computed by traversing the ancestor chain and progressively binding (Hadamard product) each ancestor's vector:

```rust
let mut path_vec = leaf_bundle.vector.clone();  // start with the leaf

let mut current_parent_id = leaf_bundle.parent_id;
while let Some(pid) = current_parent_id {
    let ancestor = db.get_bundle_by_id(pid)?;

    // Renormalize before each bind so the leaf remains dominant
    let norm = path_vec.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-10);
    for v in path_vec.iter_mut() { *v /= norm; }

    // Bind: element-wise multiply with ancestor vector
    for (i, &av) in ancestor.vector.iter().enumerate() {
        path_vec[i] *= av;
    }

    current_parent_id = ancestor.parent_id;
}

// Final normalization
let norm = path_vec.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-10);
for v in path_vec.iter_mut() { *v /= norm; }
```

The renormalization before each bind step prevents distant ancestors from dominating — the leaf's direction is anchored at the start, and ancestors progressively modulate it.

---

## Index Files

All per-workspace index data lives in `<workspace>/.slocate/`.

### `index.db` (SQLite, WAL mode)

| Table | Contents |
|-------|----------|
| `chunks` | `id TEXT PK`, `kind`, `name`, `source_path`, `source`, `community_id INTEGER`, `vector BLOB` (f16) |
| `bundles` | `id INTEGER PK`, `parent_id INTEGER`, `level INTEGER`, `vector BLOB` (f16), `vector_sum BLOB` (f16), `chunk_count INTEGER`, `hubs TEXT` |
| `chunk_secondary_communities` | `(chunk_id TEXT, bundle_id INTEGER)` — many-to-many secondary membership |
| `chunks_fts` | FTS5 virtual table on `(name, source)` for BM25 search |
| `file_meta` | `rel_path TEXT PK`, `mtime_ns INTEGER`, `size INTEGER` |
| `embed_cache` | `content_hash TEXT PK`, `vector BLOB` (f16), `created_at INTEGER` |
| `meta` | key-value store; `global_mean` holds the JSON-encoded global mean vector |
| `hnsw_nodes` / `hnsw_edges` / `hnsw_meta` | HNSW tables (present in schema; HNSW insert/load is currently stubbed — the tree serves the retrieval role) |

Vectors in `chunks` and `embed_cache` are stored as f16 BLOBs (2 bytes per dimension, 768 bytes for 384-dim). Bundle vectors in `bundles` are also f16.

### `bundle_matrix_L{n}.bin`

One file per tree level (L0, L1, ...). Flat binary format:

```
[n: u32 LE]          — number of bundles at this level
[dim: u32 LE]        — vector dimension (384)
[id_0: i64 LE]       — bundle row IDs from `bundles` table
...
[id_{n-1}: i64 LE]
[vec_0: f32 LE × dim]  — L2-normalized bundle vectors
...
[vec_{n-1}: f32 LE × dim]
```

These files are not used directly during query time (search uses `leaf_paths.bin`), but they are present for potential use.

### `leaf_paths.bin`

Same binary format as `bundle_matrix_L{n}.bin`, but contains one entry per L0 bundle. Each vector is the progressive renormalized Hadamard product of the bundle's own vector with all ancestor bundle vectors up to the root — encoding the full hierarchical context of each L0 community into a single 384-dim vector.

---

## Query Execution

`search.rs` implements `search_workspaces`, which runs against all configured workspaces and merges results.

### Step 1: Embed the Query

The raw query string is embedded with BGE (no "code: " prefix or other prompt engineering). The resulting vector is centered by subtracting the workspace's `global_mean` and re-normalizing:

```rust
centered[i] = raw_query_vec[i] - global_mean[i];
// then L2-normalize centered
```

This corrects for anisotropy — the global mean represents "generic code noise" shared across all chunks.

### Step 2: Flat Scan over `leaf_paths.bin`

All leaf path vectors are scored against the centered query via dot product in a single pass:

```rust
let scored: Vec<(f32, i64)> = lp_ids.iter().zip(lp_vecs.chunks(dim))
    .map(|(id, row)| (dot(&query_vec, row), *id))
    .collect();
scored.sort_by(|a, b| b.0.partial_cmp(&a.0)...);
let top_bundle_ids = scored.into_iter().take(beam_width).map(|(_, id)| id);
```

`beam_width` (default 8) controls how many L0 bundles are selected. This is a flat scan, not a tree traversal — there is no hierarchical routing that could cascade errors. If `leaf_paths.bin` is absent or dimension-mismatched, the search falls back to all root bundles from the DB.

### Step 3: Exact Search within Selected Bundles

For each selected L0 bundle ID, all primary and secondary member chunks are loaded with their vectors:

```sql
SELECT ... FROM chunks
WHERE vector IS NOT NULL AND (
    community_id = ?1
    OR id IN (SELECT chunk_id FROM chunk_secondary_communities WHERE bundle_id = ?1)
)
```

Each chunk vector is dot-producted against the centered query. Chunks scoring below `min_score` (default 0.01) are dropped.

### Step 4: BM25 Boost

When `bm25_weight > 0.0` (default 0.5), the FTS5 index is queried with the raw prompt. The query is sanitized: tokens are quoted and joined with `OR`, with common stopwords removed (`for`, `the`, `and`, etc.). BM25 scores are normalized to the top hit's score, then blended:

```
final_score = bm25_weight × bm25_score + (1 − bm25_weight) × cosine_score
```

BM25 hits not already in the candidate set are fetched from the DB and added.

### Step 5: MMR Reranking

Maximal Marginal Relevance reranks the candidate set to balance relevance and diversity:

```
MMR_score(d) = λ × sim(query, d) − (1 − λ) × max_{s ∈ selected} sim(d, s)
```

`mmr_lambda` (default 0.9) controls the tradeoff — 1.0 is pure relevance, 0.0 is pure diversity. At each step the highest-scoring remaining candidate is greedily selected. If `mmr_lambda ≥ 1.0`, MMR is bypassed and results are returned in similarity order.

The final `top_k` (default 5) results are returned as `ScoredChunk` structs with `score`, the `Chunk`, and a `lineage` (the `hubs` fields of all ancestors from root to leaf, for display context).

---

## VSA Concepts Used

slocate uses three operations from Vector Symbolic Architectures (VSA):

### Superposition: Community Bundle Vector

A bundle vector is the L2-normalized sum of all member vectors:

```
bundle = normalize(v_1 + v_2 + ... + v_n)
```

In the VSA sense, superposition produces a vector that is approximately similar to all of its components. The dot product between a query and a bundle is a proxy for the query's relevance to any member of the community.

### Binding: Leaf Path Vector

The Hadamard product (element-wise multiplication) is the VSA binding operator. A leaf path vector binds the L0 bundle with all its ancestors:

```
path = normalize(normalize(normalize(leaf ⊙ parent) ⊙ grandparent) ⊙ ...)
```

Renormalization before each Hadamard step keeps the leaf's direction as the primary signal; each ancestor modulates but does not overwhelm it. The result encodes both the local community's identity and its position in the global hierarchy. Querying against path vectors simultaneously tests cluster relevance at all levels.

### Secondary Membership: VSA Stability Analysis

The membership accumulator tracks how consistently each chunk is grouped with the same community across different gamma values:

```
membership_accum[i] += normalize(Σ_{j in community(i,γ)} v_j)   for each useful γ
membership_vec[i] = normalize(membership_accum[i])
```

Chunks with unstable community assignment (boundary chunks that land in different communities at different gammas) produce a `membership_vec` that points between multiple bundles. The threshold `score > 0.3` detects this and registers secondary membership, ensuring boundary chunks are found regardless of which community "wins" in the final partition.

---

## Key Design Decisions

### Build-once gamma sweep

The Leiden graph is a function of the node vectors and the similarity threshold, not of gamma. The CPM quality function changes with gamma, but the graph structure does not. Building the all-pairs similarity matrix (the expensive matmul) once and running Leiden multiple times on the same graph reduces the cost of the sweep from O(k × n²) matmuls to O(n²) + O(k × n) Leiden iterations.

### Flat scan over `leaf_paths.bin` instead of tree traversal

A top-down tree traversal routes the query through intermediate nodes at each level. Routing errors at level 1 permanently exclude entire subtrees. The flat scan over leaf path vectors scores all L0 communities directly in one pass. Because path vectors encode hierarchical context via Hadamard binding, the flat scan achieves the same discrimination as tree traversal but without cascading routing failures. The cost is O(n_leaves × dim), which for typical codebases (hundreds to low thousands of L0 bundles × 384 dims) is fast enough that it dominates neither CPU nor wall time.

### Renormalize at each Hadamard step

Without renormalization, the path vector after k steps has magnitude ≤ 1 per dimension (product of unit-vector components), and distant ancestors contribute with the same weight as the leaf. By renormalizing before each multiplication, the current path vector is kept unit-length so the next ancestor modulates it multiplicatively rather than overwriting it. The leaf vector is bound first and has the strongest influence on the final direction.

### Secondary community membership via VSA stability analysis

Any fixed gamma partitions the chunk space deterministically, but boundary chunks — those semantically equidistant between two communities — land in one community or the other arbitrarily. The gamma sweep is already required to find a good partition count. Accumulating the normalized trial bundles across all useful gammas is a free by-product of that sweep. The resulting membership vector reveals which bundles a chunk has affinity for across gamma values, enabling secondary assignment without any additional clustering cost.

### Global mean centering at query time (not at embed time)

Subtracting the global mean before running Leiden would collapse inter-chunk similarity variance (making the graph near-edgeless), breaking clustering. Instead, the global mean is computed and stored at reindex time but applied only to query vectors at search time. This removes the "generic code" bias from queries without disturbing the structure used for community detection.

---

## Module Reference

| Module | Role |
|--------|------|
| `main.rs` | CLI (`clap` derive), subcommand dispatch |
| `error.rs` | Unified `Error` enum, `Result<T>` alias |
| `config.rs` | TOML config at `$XDG_CONFIG_HOME/slocate/config.toml`, tilde expansion |
| `parse.rs` | Tree-sitter + Markdown chunking |
| `embed.rs` | BGE-small-en-v1.5 via `candle`; mmap'd weights; batched forward pass |
| `store.rs` | SQLite persistence: all tables, encode/decode f16 vectors, BM25 search |
| `vdb.rs` | HNSW implementation (M=16, ef_construction=200); `dot` product similarity function |
| `leiden.rs` | Leiden algorithm (CPM); `build_graph` / `run_on_graph` separation for gamma sweeps |
| `reindex.rs` | 5-phase incremental pipeline + Leiden tree rollup + binary file output |
| `search.rs` | Query execution: embed → center → flat scan → exact search → BM25 → MMR |
| `registry.rs` | Workspace symlink registry at `$XDG_DATA_HOME/slocate/`; index co-located at `<workspace>/.slocate/index.db` |
| `backends/` | `HookBackend` trait; `ClaudeBackend` and `GeminiBackend` formatters |
| `mcp_tools.rs` | MCP tool handlers: `search_code`, `index_workspace`, `note_to_self`, `check_notes` |
| `spsc.rs` | Wait-free single-producer/single-consumer channel used by embed workers |
| `download.rs` | HuggingFace model downloader with BERT config validation |
| `install.rs` | One-time setup: model download, config write, daemon install, first reindex |
| `platform/` | Daemon: launchd plist (macOS) / systemd user timer (Linux) |
| `fastmath.rs` | `FastMathGuard`: sets FTZ/DAZ flush-to-zero flags for the embed thread |

---

## Configuration Reference

Config file: `$XDG_CONFIG_HOME/slocate/config.toml` (default: `~/.config/slocate/config.toml`)

### `[index]`

| Key | Default | Description |
|-----|---------|-------------|
| `workspaces` | `[]` | Workspace directories to index |
| `extensions` | `["rs","py","ts","go","c","h","md","yaml","yml","bzl"]` | File extensions to index |
| `max_file_bytes` | `1048576` (1 MB) | Skip files larger than this |
| `embed_workers` | `4` | Max parallel embed threads (GPU forces 1) |
| `reindex_interval_minutes` | `10` | Daemon reindex cadence |
| `leiden_gamma` | `1.13` | Starting CPM resolution; higher = smaller communities |
| `leiden_threshold` | `0.0` | Graph edge threshold; 0.0 = auto |
| `leiden_target_neighbors` | `15` | Avg neighbors per node for auto-threshold |
| `leiden_threshold_ceiling` | `0.90` | Max allowed auto-threshold |
| `leiden_gamma_step` | `0.5` | Gamma reduction factor per sweep step |
| `tree_branching_factor` | `10` | Target n_nodes/n_communities per level |

### `[search]`

| Key | Default | Description |
|-----|---------|-------------|
| `top_k` | `5` | Number of results to return |
| `min_score` | `0.01` | Minimum cosine similarity to include |
| `beam_width` | `8` | Number of L0 bundles selected from `leaf_paths.bin` |
| `bm25_weight` | `0.5` | α in `α·bm25 + (1−α)·cosine` hybrid score |
| `mmr_lambda` | `0.9` | MMR relevance/diversity tradeoff (1.0 = pure relevance) |
| `whitening` | `0.24` | Not currently applied in search; present in config |

### `[model]`

| Key | Default | Description |
|-----|---------|-------------|
| `dir` | `~/.local/share/slocate/models/bge-small-en-v1.5` | Path to BGE model directory |
