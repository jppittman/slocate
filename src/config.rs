use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub search: SearchConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub workspaces: Vec<String>,
    pub reindex_interval_minutes: u64,
    pub extensions: Vec<String>,
    /// Number of threads used for parallel embedding during reindex.
    /// Each thread shares the single Embedder (no per-thread model copy needed).
    pub embed_workers: usize,
    /// Skip files larger than this many bytes before reading them.
    /// Protects against huge generated files, data blobs, etc.
    pub max_file_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub top_k: usize,
    /// Minimum cosine similarity to inject. Results below this are dropped.
    pub min_score: f32,
    /// MMR lambda: relevance/diversity tradeoff for result reranking.
    /// 1.0 = pure top-k by similarity, 0.0 = pure diversity, 0.5 = balanced.
    #[serde(default = "default_mmr_lambda")]
    pub mmr_lambda: f32,
}

fn default_mmr_lambda() -> f32 {
    0.5
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            workspaces: vec![],
            reindex_interval_minutes: 10,
            extensions: vec![
                "rs".into(),
                "py".into(),
                "ts".into(),
                "go".into(),
                "c".into(),
                "h".into(),
                "md".into(),
                "yaml".into(),
                "yml".into(),
                "bzl".into(),
            ],
            embed_workers: 4,
            max_file_bytes: 1024 * 1024, // 1 MB
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dir: data_dir()
                .join("models/bge-small-en-v1.5")
                .display()
                .to_string(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            // BGE-small-en-v1.5 produces lower raw similarity scores than
            // BGE-base. 0.65 is a reasonable default; tune after reindexing.
            min_score: 0.65,
            mmr_lambda: default_mmr_lambda(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            index: IndexConfig::default(),
            model: ModelConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

impl Config {
    pub fn load() -> crate::error::Result<Self> {
        let path = config_file();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)?;
        toml::from_str(&text)
            .map_err(|e| crate::error::Error::Config(
                format!("invalid config at {}: {e}", path.display()),
            ))
    }

    pub fn save(&self) -> crate::error::Result<()> {
        let path = config_file();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)
            .map_err(|e| crate::error::Error::Config(
                format!("failed to serialize config: {e}"),
            ))?;
        std::fs::write(&path, text)?;
        Ok(())
    }

    pub fn expanded_workspaces(&self) -> Vec<PathBuf> {
        self.index
            .workspaces
            .iter()
            .map(|w| expand_tilde(w))
            .collect()
    }

    pub fn model_dir(&self) -> PathBuf {
        expand_tilde(&self.model.dir)
    }
}

// ─── XDG directory helpers ──────────────────────────────────────────────────

/// $XDG_CONFIG_HOME/slocate  (default: ~/.config/slocate)
pub fn config_dir() -> PathBuf {
    xdg_dir("XDG_CONFIG_HOME", ".config").join("slocate")
}

/// $XDG_DATA_HOME/slocate  (default: ~/.local/share/slocate)
pub fn data_dir() -> PathBuf {
    xdg_dir("XDG_DATA_HOME", ".local/share").join("slocate")
}

/// $XDG_STATE_HOME/slocate  (default: ~/.local/state/slocate)
pub fn state_dir() -> PathBuf {
    xdg_dir("XDG_STATE_HOME", ".local/state").join("slocate")
}

/// Config file path.
pub fn config_file() -> PathBuf {
    config_dir().join("config.toml")
}

/// Log file path (used by launchd/systemd daemon).
pub fn log_file() -> PathBuf {
    state_dir().join("daemon.log")
}

/// Expand `~/` to `$HOME/` in a path string.
pub fn expand_tilde_pub(path: &str) -> PathBuf {
    expand_tilde(path)
}

fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(&path[2..])
    } else {
        PathBuf::from(path)
    }
}

fn xdg_dir(env_var: &str, fallback_rel: &str) -> PathBuf {
    if let Ok(val) = std::env::var(env_var) {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(fallback_rel)
}
