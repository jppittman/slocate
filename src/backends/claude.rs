use super::HookBackend;
use crate::search::ScoredChunk;

pub struct ClaudeBackend;

impl HookBackend for ClaudeBackend {
    fn format_results(&self, results: &[ScoredChunk], top_k: usize) -> String {
        let mut out = String::new();
        for sc in results.iter().take(top_k) {
            out.push_str(&format!(
                "[{:.2}] {} `{}` — {}\n{}\n\n",
                sc.score, sc.chunk.kind, sc.chunk.name, sc.chunk.source_path, sc.chunk.source
            ));
        }
        out.trim_end().to_string()
    }
}
