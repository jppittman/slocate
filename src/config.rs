use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub search: SearchConfig,
}

/// Controls CPU intensity during reindex.
///
/// - `quiet`:       max 2 workers, 50 ms sleep between batches — minimal fan noise.
/// - `balanced`:    max 4 workers, no sleep — default, good for background daemon.
/// - `performance`: max 8 workers, no sleep — fastest, uses more cores.
///
/// Override at runtime with `SLOCATE_POWER=quiet|balanced|performance`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PowerMode {
    Quiet,
    #[default]
    Balanced,
    Performance,
}

impl PowerMode {
    /// Parse from string (case-insensitive). Returns `None` on unrecognised input.
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "quiet" => Some(Self::Quiet),
            "balanced" => Some(Self::Balanced),
            "performance" => Some(Self::Performance),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub workspaces: Vec<String>,
    pub reindex_interval_minutes: u64,
    pub extensions: Vec<String>,
    /// Explicit worker-count override. When set to 0 (the default), the effective
    /// count is determined by `power_mode`.
    pub embed_workers: usize,
    /// Skip files larger than this many bytes before reading them.
    /// Protects against huge generated files, data blobs, etc.
    pub max_file_bytes: u64,
    /// CPU intensity profile for reindex. See [`PowerMode`].
    #[serde(default)]
    pub power_mode: PowerMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub top_k: usize,
    /// Minimum cosine similarity to inject. Results below this are dropped
    /// when there is no BM25 match. BM25 hits always bypass this threshold.
    pub min_score: f32,
    /// MMR lambda: relevance/diversity tradeoff for result reranking.
    /// 1.0 = pure top-k by similarity, 0.0 = pure diversity.
    /// Default 0.8 keeps related functions from the same file together.
    #[serde(default = "default_mmr_lambda")]
    pub mmr_lambda: f32,
    /// BM25 weight α in hybrid score: α·bm25 + (1−α)·cosine.
    /// 0.0 = pure semantic, 1.0 = pure BM25, 0.5 = equal blend.
    #[serde(default = "default_bm25_weight")]
    pub bm25_weight: f32,
}

fn default_mmr_lambda() -> f32 {
    0.9
}

fn default_bm25_weight() -> f32 {
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
            embed_workers: 0,            // 0 = auto (determined by power_mode)
            max_file_bytes: 1024 * 1024, // 1 MB
            power_mode: PowerMode::default(),
        }
    }
}

impl IndexConfig {
    /// Resolve the active power mode. `SLOCATE_POWER` env var takes precedence
    /// over the config file value.
    pub fn effective_power_mode(&self) -> PowerMode {
        if let Ok(val) = std::env::var("SLOCATE_POWER") {
            if let Some(mode) = PowerMode::from_str_loose(&val) {
                return mode;
            }
            log::warn!(
                "SLOCATE_POWER={val:?} not recognised (use quiet/balanced/performance), \
                 falling back to config"
            );
        }
        self.power_mode
    }

    /// Effective worker count for CPU embedding.
    /// If `embed_workers` is explicitly set (>0), use it.
    /// Otherwise derive from `effective_power_mode()`.
    pub fn effective_embed_workers(&self) -> usize {
        if self.embed_workers > 0 {
            return self.embed_workers;
        }
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let cap = match self.effective_power_mode() {
            PowerMode::Quiet => 2,
            PowerMode::Balanced => 4,
            PowerMode::Performance => 8,
        };
        available.min(cap)
    }

    /// Optional sleep between embedding batches for thermal throttling.
    pub fn batch_sleep(&self) -> Option<std::time::Duration> {
        match self.effective_power_mode() {
            PowerMode::Quiet => Some(std::time::Duration::from_millis(50)),
            PowerMode::Balanced | PowerMode::Performance => None,
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
            min_score: 0.01,
            mmr_lambda: default_mmr_lambda(),
            bm25_weight: default_bm25_weight(),
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
        toml::from_str(&text).map_err(|e| {
            crate::error::Error::Config(format!("invalid config at {}: {e}", path.display()))
        })
    }

    pub fn save(&self) -> crate::error::Result<()> {
        let path = config_file();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)
            .map_err(|e| crate::error::Error::Config(format!("failed to serialize config: {e}")))?;
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
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(stripped)
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── PowerMode parsing ───────────────────────────────────────────────

    #[test]
    fn power_mode_from_str_exact() {
        assert_eq!(PowerMode::from_str_loose("quiet"), Some(PowerMode::Quiet));
        assert_eq!(
            PowerMode::from_str_loose("balanced"),
            Some(PowerMode::Balanced)
        );
        assert_eq!(
            PowerMode::from_str_loose("performance"),
            Some(PowerMode::Performance)
        );
    }

    #[test]
    fn power_mode_from_str_case_insensitive() {
        assert_eq!(PowerMode::from_str_loose("QUIET"), Some(PowerMode::Quiet));
        assert_eq!(
            PowerMode::from_str_loose("Balanced"),
            Some(PowerMode::Balanced)
        );
        assert_eq!(
            PowerMode::from_str_loose("PERFORMANCE"),
            Some(PowerMode::Performance)
        );
    }

    #[test]
    fn power_mode_from_str_trims_whitespace() {
        assert_eq!(
            PowerMode::from_str_loose("  quiet  "),
            Some(PowerMode::Quiet)
        );
        assert_eq!(
            PowerMode::from_str_loose("\tbalanced\n"),
            Some(PowerMode::Balanced)
        );
    }

    #[test]
    fn power_mode_from_str_rejects_garbage() {
        assert_eq!(PowerMode::from_str_loose(""), None);
        assert_eq!(PowerMode::from_str_loose("turbo"), None);
        assert_eq!(PowerMode::from_str_loose("qui et"), None);
    }

    #[test]
    fn power_mode_default_is_balanced() {
        assert_eq!(PowerMode::default(), PowerMode::Balanced);
    }

    // ── effective_embed_workers ─────────────────────────────────────────

    #[test]
    fn effective_workers_explicit_override() {
        let cfg = IndexConfig {
            embed_workers: 7,
            ..Default::default()
        };
        assert_eq!(cfg.effective_embed_workers(), 7);
    }

    #[test]
    fn effective_workers_auto_quiet_caps_at_2() {
        let cfg = IndexConfig {
            embed_workers: 0,
            power_mode: PowerMode::Quiet,
            ..Default::default()
        };
        assert!(cfg.effective_embed_workers() <= 2);
        assert!(cfg.effective_embed_workers() >= 1);
    }

    #[test]
    fn effective_workers_auto_balanced_caps_at_4() {
        let cfg = IndexConfig {
            embed_workers: 0,
            power_mode: PowerMode::Balanced,
            ..Default::default()
        };
        assert!(cfg.effective_embed_workers() <= 4);
        assert!(cfg.effective_embed_workers() >= 1);
    }

    #[test]
    fn effective_workers_auto_performance_caps_at_8() {
        let cfg = IndexConfig {
            embed_workers: 0,
            power_mode: PowerMode::Performance,
            ..Default::default()
        };
        assert!(cfg.effective_embed_workers() <= 8);
        assert!(cfg.effective_embed_workers() >= 1);
    }

    // ── batch_sleep ─────────────────────────────────────────────────────

    #[test]
    fn batch_sleep_quiet_returns_some() {
        let cfg = IndexConfig {
            power_mode: PowerMode::Quiet,
            ..Default::default()
        };
        let sleep = cfg.batch_sleep();
        assert!(sleep.is_some());
        assert_eq!(sleep.unwrap().as_millis(), 50);
    }

    #[test]
    fn batch_sleep_balanced_returns_none() {
        let cfg = IndexConfig {
            power_mode: PowerMode::Balanced,
            ..Default::default()
        };
        assert!(cfg.batch_sleep().is_none());
    }

    #[test]
    fn batch_sleep_performance_returns_none() {
        let cfg = IndexConfig {
            power_mode: PowerMode::Performance,
            ..Default::default()
        };
        assert!(cfg.batch_sleep().is_none());
    }

    // ── TOML round-trip ─────────────────────────────────────────────────

    #[test]
    fn power_mode_toml_round_trip() {
        let cfg = IndexConfig {
            power_mode: PowerMode::Quiet,
            ..Default::default()
        };
        let toml_str = toml::to_string_pretty(&cfg).unwrap();
        assert!(
            toml_str.contains("power_mode = \"quiet\""),
            "expected lowercase serde rename, got: {toml_str}"
        );
        let parsed: IndexConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.power_mode, PowerMode::Quiet);
    }

    #[test]
    fn power_mode_missing_from_toml_defaults_to_balanced() {
        // Simulates an old config file that doesn't have power_mode at all.
        let toml_str = r#"
            workspaces = []
            reindex_interval_minutes = 10
            extensions = ["rs"]
            embed_workers = 0
            max_file_bytes = 1048576
        "#;
        let parsed: IndexConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(parsed.power_mode, PowerMode::Balanced);
    }

    // ── IndexConfig default ─────────────────────────────────────────────

    #[test]
    fn default_embed_workers_is_auto() {
        let cfg = IndexConfig::default();
        assert_eq!(cfg.embed_workers, 0, "default should be 0 (auto)");
    }
}
