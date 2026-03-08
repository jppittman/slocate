use super::HookBackend;
use crate::search::ScoredChunk;

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
                let lineage = if let Some(first) = group.first() {
                    if first.lineage.is_empty() {
                        format!("Community {id}")
                    } else {
                        format!("Community {id} ({})", first.lineage.join(" > "))
                    }
                } else {
                    format!("Community {id}")
                };
                out.push_str(&format!("── {lineage} ──\n"));
            }
            for sc in group {
                out.push_str(&format!(
                    "[{:.2}] {} `{}` — {}\n{}\n\n",
                    sc.score, sc.chunk.kind, sc.chunk.name, sc.chunk.source_path, sc.chunk.source
                ));
            }
        }
        out.trim_end().to_string()
    }
}
