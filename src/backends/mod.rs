pub mod claude;
pub mod gemini;

use crate::search::ScoredChunk;

/// Formats scored code chunks for injection into a specific LLM's hook system.
pub trait HookBackend {
    /// Consume the top-k scored results and produce the output written to stdout.
    fn format_results(&self, results: &[ScoredChunk], top_k: usize) -> String;
}
