use super::HookBackend;
use crate::tools::ScoredChunk;

pub struct ClaudeBackend;

impl HookBackend for ClaudeBackend {
    fn format_results(&self, results: &[ScoredChunk], top_k: usize) -> String {
        // Group by community so related chunks appear together.
        let mut by_community: Vec<(Option<usize>, Vec<&ScoredChunk>)> = Vec::new();
        for sc in results.iter().take(top_k) {
            let cid = sc.chunk.community_id;
            match by_community.iter_mut().find(|(c, _)| *c == cid) {
                Some((_, group)) => group.push(sc),
                None => by_community.push((cid, vec![sc])),
            }
        }

        let mut out = String::new();
        for (cid, group) in &by_community {
            if let Some(id) = cid {
                out.push_str(&format!("── Community {id} ──\n"));
            }
            for sc in group {
                let preview = if sc.chunk.source.len() > 300 {
                    &sc.chunk.source[..300]
                } else {
                    &sc.chunk.source
                };
                out.push_str(&format!(
                    "[{:.2}] {} `{}` — {}\n{}\n\n",
                    sc.score, sc.chunk.kind, sc.chunk.name, sc.chunk.source_path, preview
                ));
            }
        }
        out.trim_end().to_string()
    }
}
