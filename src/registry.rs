use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

/// Returns the index directory for a workspace.
///
/// Data lives at `<workspace>/.slocate/` -- delete the workspace and the
/// index goes with it. No leaked space.
///
/// A symlink from `~/.local/share/slocate/<hash>` points back to the
/// workspace's `.slocate/` dir so we can enumerate all indexed dirs
/// without scanning the filesystem. Broken symlinks = workspace was deleted,
/// `slocate gc` prunes them.
pub fn index_dir(workspace_root: &Path) -> crate::error::Result<PathBuf> {
    let dir = workspace_root.join(".slocate");
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }

    // Maintain the reverse symlink registry (best-effort).
    let _ = ensure_registry_link(workspace_root, &dir);

    Ok(dir)
}

/// Remove the index for a workspace and its registry symlink.
pub fn remove_index(workspace_root: &Path) -> crate::error::Result<()> {
    let dir = workspace_root.join(".slocate");
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    // Remove the registry symlink.
    if let Ok(link) = registry_link_path(workspace_root) {
        if link.symlink_metadata().is_ok() {
            let _ = std::fs::remove_file(&link);
        }
    }
    Ok(())
}

/// Remove registry symlinks whose targets no longer exist (workspace deleted).
pub fn gc_registry() -> crate::error::Result<usize> {
    let base = registry_base()?;
    if !base.exists() {
        return Ok(0);
    }
    let mut removed = 0;
    let entries = std::fs::read_dir(&base)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        // Symlink whose target is gone -> orphan.
        if path.symlink_metadata().is_ok() && !path.exists() {
            let _ = std::fs::remove_file(&path);
            removed += 1;
        }
    }
    Ok(removed)
}

/// Wipe all symlinks in the registry.
pub fn wipe_registry() -> crate::error::Result<()> {
    let base = registry_base()?;
    if !base.exists() {
        return Ok(());
    }
    let entries = std::fs::read_dir(&base)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.symlink_metadata().is_ok() {
            let _ = std::fs::remove_file(&path);
        }
    }
    Ok(())
}

fn registry_base() -> crate::error::Result<PathBuf> {
    Ok(crate::config::data_dir().join("registry"))
}

fn registry_link_path(workspace_root: &Path) -> crate::error::Result<PathBuf> {
    let base = registry_base()?;
    Ok(base.join(dir_hash(workspace_root)))
}

fn ensure_registry_link(workspace_root: &Path, index_dir: &Path) -> crate::error::Result<()> {
    let base = registry_base()?;
    std::fs::create_dir_all(&base)?;
    let link = base.join(dir_hash(workspace_root));
    if !link.exists() {
        #[cfg(unix)]
        std::os::unix::fs::symlink(index_dir, &link)?;
    }
    Ok(())
}

/// Stable hash of a workspace path -> 16 hex chars.
fn dir_hash(path: &Path) -> String {
    let mut h: u64 = 5381;
    for b in path.display().to_string().bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    format!("{:016x}", h)
}

// ─── PathTrie ────────────────────────────────────────────────────────────────

/// Prefix trie over absolute filesystem paths (split by [`Component`]).
///
/// Each node corresponds to one path component. A node whose `workspace` field
/// is `Some` is a registered workspace root. The trie lets us answer two
/// overlap questions in O(depth) time:
///
/// - Is a candidate path *below* an existing workspace?
///   → its files are already indexed; adding it would double-index them.
/// - Does a candidate path *cover* existing workspaces?
///   → those workspaces' files would be double-indexed under the new root.
#[derive(Default)]
pub struct PathTrie {
    root: TrieNode,
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<OsString, TrieNode>,
    /// Set when this exact path is a registered workspace root.
    workspace: Option<PathBuf>,
}

impl PathTrie {
    /// Build a trie from a slice of already-canonicalized workspace roots.
    pub fn from_workspaces(paths: &[PathBuf]) -> Self {
        let mut trie = PathTrie::default();
        for p in paths {
            trie.insert(p);
        }
        trie
    }

    /// Insert a canonicalized path as a workspace root.
    pub fn insert(&mut self, path: &Path) {
        let mut node = &mut self.root;
        for comp in path.components() {
            node = node
                .children
                .entry(comp.as_os_str().to_owned())
                .or_default();
        }
        node.workspace = Some(path.to_path_buf());
    }

    /// Return the nearest ancestor workspace that already covers `path`, if any.
    ///
    /// "Ancestor" means the registered path is a proper prefix of `path`.
    /// An exact match (same path) is not considered an ancestor here — that is
    /// the duplicate check in `cmd_add_repo`.
    pub fn ancestor_of(&self, path: &Path) -> Option<&Path> {
        let mut node = &self.root;
        let mut found: Option<&Path> = None;
        let components: Vec<Component> = path.components().collect();
        for (i, comp) in components.iter().enumerate() {
            // A node marked as workspace that is a *proper* prefix (not the
            // final component) is an ancestor workspace.
            if let Some(ws) = &node.workspace {
                // Proper prefix: there are still components left.
                if i < components.len() {
                    found = Some(ws.as_path());
                }
            }
            match node.children.get(comp.as_os_str()) {
                Some(child) => node = child,
                None => break,
            }
        }
        found
    }

    /// Collect all registered workspace roots that are proper descendants of `path`.
    ///
    /// "Descendant" means `path` is a proper prefix of the registered path.
    pub fn descendants_of(&self, path: &Path) -> Vec<&Path> {
        // Walk down to the node for `path`.
        let mut node = &self.root;
        for comp in path.components() {
            match node.children.get(comp.as_os_str()) {
                Some(child) => node = child,
                None => return vec![],
            }
        }
        // `node` is now the node for `path` itself. Collect all workspace
        // nodes in the subtree *below* it (not `path` itself, which is the
        // duplicate check).
        let mut results = Vec::new();
        for child in node.children.values() {
            collect_workspaces(child, &mut results);
        }
        results
    }
}

fn collect_workspaces<'a>(node: &'a TrieNode, out: &mut Vec<&'a Path>) {
    if let Some(ws) = &node.workspace {
        out.push(ws.as_path());
    }
    for child in node.children.values() {
        collect_workspaces(child, out);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn no_overlap_disjoint_workspaces() {
        let trie = PathTrie::from_workspaces(&[p("/home/user/a"), p("/home/user/b")]);
        assert!(trie.ancestor_of(&p("/home/user/c")).is_none());
        assert!(trie.descendants_of(&p("/home/user/c")).is_empty());
    }

    #[test]
    fn ancestor_detected() {
        let trie = PathTrie::from_workspaces(&[p("/home/user/projects")]);
        assert_eq!(
            trie.ancestor_of(&p("/home/user/projects/myrepo")),
            Some(p("/home/user/projects").as_path())
        );
    }

    #[test]
    fn exact_match_is_not_an_ancestor() {
        // Exact duplicates are handled separately; ancestor_of must not fire.
        let trie = PathTrie::from_workspaces(&[p("/home/user/projects")]);
        assert!(trie.ancestor_of(&p("/home/user/projects")).is_none());
    }

    #[test]
    fn descendants_detected() {
        let trie =
            PathTrie::from_workspaces(&[p("/home/user/projects/a"), p("/home/user/projects/b")]);
        let mut desc: Vec<&Path> = trie.descendants_of(&p("/home/user/projects"));
        desc.sort();
        assert_eq!(desc.len(), 2);
        assert!(desc.contains(&p("/home/user/projects/a").as_path()));
        assert!(desc.contains(&p("/home/user/projects/b").as_path()));
    }

    #[test]
    fn exact_match_not_a_descendant() {
        let trie = PathTrie::from_workspaces(&[p("/home/user/projects")]);
        assert!(trie.descendants_of(&p("/home/user/projects")).is_empty());
    }

    #[test]
    fn sibling_is_not_ancestor_or_descendant() {
        let trie = PathTrie::from_workspaces(&[p("/home/user/a")]);
        assert!(trie.ancestor_of(&p("/home/user/b")).is_none());
        assert!(trie.descendants_of(&p("/home/user/b")).is_empty());
    }

    #[test]
    fn deep_nesting() {
        let trie = PathTrie::from_workspaces(&[p("/a/b/c/d")]);
        assert_eq!(
            trie.ancestor_of(&p("/a/b/c/d/e/f")),
            Some(p("/a/b/c/d").as_path())
        );
        assert!(trie.ancestor_of(&p("/a/b/c")).is_none());
    }
}
