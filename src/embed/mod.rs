use crate::error::Result;

/// Embedding backend. Implement this to swap in cloud APIs (Vertex AI, OpenAI,
/// etc.) alongside the default local BGE model.
///
/// Must be `Send + Sync` — the reindex pipeline shares embedders across threads.
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batched variant. Default calls `embed` in a loop; override for efficiency.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// True if running on a GPU; used to cap worker concurrency.
    fn is_gpu(&self) -> bool {
        false
    }
}

pub mod bge;
pub use bge::BgeEmbedder;
