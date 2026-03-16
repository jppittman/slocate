use super::HookBackend;
use crate::search::ScoredChunk;

pub struct GeminiBackend;

impl HookBackend for GeminiBackend {
    fn format_results(&self, results: &[ScoredChunk], top_k: usize) -> String {
        if results.is_empty() {
            return String::new();
        }

        let mut out = String::from("Relevant Workspace Context (RAG):\n\n");
        for sc in results.iter().take(top_k) {
            out.push_str(&format!(
                "--- {} `{}` in {} (Relevance: {:.2}) ---\n{}\n\n",
                sc.chunk.kind, sc.chunk.name, sc.chunk.source_path, sc.score, sc.chunk.source
            ));
        }
        out.trim_end().to_string()
    }
}
