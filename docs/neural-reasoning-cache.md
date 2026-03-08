# Neural Reasoning Cache

A design for an external memory system for language models that supports
persistent, correctable, introspectable reasoning across sessions.

---

## Motivation

Transformer attention is bounded by context length. External memory (RAG, MemGPT)
retrieves facts but not reasoning structure. Truth Maintenance Systems (JTMS, ATMS)
handle belief revision in symbolic systems but require hand-engineered justifications
and don't scale. This design combines the strengths of all three:

- **Unbounded persistence** across sessions (external commit log)
- **Semantic retrieval** of past reasoning (HNSW over commit embeddings)
- **Correctable chains** without destructive mutation (union-find over immutable DAG)
- **End-to-end trainable** retrieval and write policy (RL over CoT episodes)

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  HNSW Index                                          │
│  Semantic entry points into the reasoning graph      │
│  Query: embed(current_thought) → top-k similar nodes │
└───────────────────┬─────────────────────────────────┘
                    │ node SHAs
┌───────────────────▼─────────────────────────────────┐
│  Union-Find                                          │
│  Canonical correction pointers                       │
│  find(sha) → canonical chain entry point             │
└───────────────────┬─────────────────────────────────┘
                    │ canonical SHA
┌───────────────────▼─────────────────────────────────┐
│  Commit DAG (FOSSIL-style)                           │
│  Immutable, content-addressed reasoning history      │
│  Parent pointers encode derivation chains            │
└─────────────────────────────────────────────────────┘
```

The three layers are fully decoupled. HNSW doesn't know about union-find or
the DAG structure. Union-find doesn't know about commit content. The DAG is
append-only and never mutated.

---

## Commit Structure

```
commit {
    sha:     hash(content)          // content-addressed, dedup is free
    parent:  sha | null             // derivation link: "I reached this from that"
    content: string                 // the thought/observation/conclusion
    vector:  f32[D]                 // embed(content), indexed in HNSW
}
```

Each CoT step appends one commit with `parent = previous step's SHA`. A reasoning
episode produces a linked chain. Branching (when re-deriving from an earlier point)
creates a new chain that shares ancestor nodes with the original.

---

## Memory Head

The memory system is a separate head in the model architecture, not in-band
token generation:

```
Forward pass:
  h (current hidden state)
    → HNSW query: top-k similar past commits
    → union-find: find(sha) → canonical commits
    → cross-attention: softmax(M·W_K^T·h / √d) · M·W_V^T → m_out
    → m_out added to h before next layer

Post-generation (downstream write):
  embed(current thought) → append commit to DAG + HNSW
```

The read is differentiable (soft attention over retrieved vectors, learned W
matrices). The write is discrete and post-hoc — outside the forward pass, no
gradient through the write operation. The write policy is trained via RL reward.

The learned W_K and W_V matrices align what the model queries against what's
in memory. The model learns to write commits that it can retrieve effectively.

---

## Correction: Union-Find as Soft Retraction

When the model discovers a committed reasoning chain was wrong, it does not
delete or mutate the chain. Instead:

1. Identify the divergence point: `bad_end` (last node before the error)
2. Identify the correction entry: `good_root` (where correct reasoning starts)
3. `union(bad_end, good_root)`

The topology becomes:
```
bad_0 → bad_1 → bad_end ─[union-find]→ good_root → good_end
```

`find(bad_0)` path-compresses through to `good_root`. Future retrievals that
land in the bad chain are automatically redirected to the correct chain.

The bad chain remains fully traversable via DAG parent pointers — it is not
erased. This is intentional:
- **Inspection**: the error trace is preserved for debugging
- **Training signal**: (bad_end, good_root) is a contrastive pair with
  the same retrieval context — GRPO uses this to learn when chains go wrong

---

## Training

**Architecture**: GPT-2 base (or comparable small model) with memory head.

**Objective**: RL (GRPO) over multi-turn CoT episodes. Terminal reward = task accuracy.

**Training curriculum**:

1. **SFT warm-up**: pre-train on synthetic (CoT, commit sequence) pairs to teach
   the model to produce well-formed commits at all. The RL phase cannot learn
   commit format from scratch.

2. **Memory head pre-training** (optional): train the cross-attention W matrices in
   isolation using write-then-read reconstruction loss. Ensures retrieval is useful
   before joint training.

3. **Joint RL**: full model + memory head trained together. Episode reward
   propagates to both the main model policy and the write gate.

**Cold start**: the commit log is empty at training start, making retrieval
uninformative. Options:
- Suppress memory head for the first K episodes until the log has density
- Pre-populate with synthetic reasoning traces

**Contrastive signal from corrections**: when a union-find correction is written,
the `(bad_chain, good_chain)` pair at the junction is a free contrastive training
example. The model receives negative advantage for reasoning steps that required
correction.

---

## Relationship to Prior Work

| System | Read | Write | Correction | Scale |
|--------|------|-------|------------|-------|
| JTMS | symbolic lookup | hand-engineered | retraction propagation (destructive) | small |
| ATMS | label propagation | hand-engineered | multi-world maintenance | small |
| MemGPT | vector retrieval | explicit API call | none | medium |
| Titans | gradient-updated params | implicit (online SGD) | none | medium |
| **This** | HNSW + cross-attention | every CoT step | union-find (non-destructive) | large |

Key differences from JTMS/ATMS:
- Continuous truth values (attention weights) instead of crisp IN/OUT
- Learned justification queries instead of hand-engineered rules
- Non-destructive correction via union-find (bad chains preserved as training signal)
- Scales via approximate nearest-neighbor search

Key differences from Titans:
- Discrete commits (inspectable, persistent across sessions)
- Explicit DAG structure encodes reasoning chains (not just a parameter matrix)
- Correction is explicit (union-find pointer) not implicit (gradient)

---

## Open Questions

**Embedder alignment**: the HNSW embedder (BGE or similar) is not trained jointly
with the memory head. The model learns to write commits that help *it* reason, but
the embedder determines what gets retrieved. Misalignment = useful commits that
aren't retrievable. Mitigation: fine-tune embedder jointly, or use the model's own
hidden states as commit vectors.

**Union-find granularity**: union at `bad_end → good_root` requires the model to
identify the exact divergence point. In practice the model may only know "somewhere
in this chain I went wrong." Coarser option: union at chain roots. Loses the precise
contrastive pair but simpler to implement.

**Memory size at inference**: HNSW grows unbounded. For large commit logs, retrieval
latency increases. Mitigation: Leiden community detection over the commit graph
(same algorithm as slocate's code index) enables hierarchical routing — beam search
through community bundles before exact search within a community.

**Multi-session identity**: commits from session A should be retrievable in session B.
The HNSW index and union-find state must be persisted and versioned. FOSSIL's
content-addressed storage is the natural backend — the commit log IS a FOSSIL
repository.
