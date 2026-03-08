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
    /// Leiden resolution parameter: higher = smaller, more granular communities.
    /// Default 1.0. Increase to 2.0 or 5.0 to break up large blobs.
    #[serde(default = "default_leiden_gamma")]
    pub leiden_gamma: f64,
    /// Minimum similarity threshold for community edges.
    /// Only pairs with cosine similarity > threshold are considered connected.
    /// Default 0.5.
    #[serde(default = "default_leiden_threshold")]
    pub leiden_threshold: f32,
    /// Target number of neighbors per node when auto-thresholding the Leiden graph.
    /// Higher → denser graph → larger communities. Default 15.
    #[serde(default = "default_leiden_target_neighbors")]
    pub leiden_target_neighbors: usize,
    /// Maximum similarity threshold cap during auto-thresholding.
    /// Prevents the threshold from being set so high that the graph becomes a clique.
    /// Default 0.90.
    #[serde(default = "default_leiden_threshold_ceiling")]
    pub leiden_threshold_ceiling: f32,
    /// Gamma reduction factor per retry in the auto-gamma sweep.
    /// gamma *= leiden_gamma_step each time communities are too coarse.
    /// Smaller values (e.g. 0.5) give finer search; 0.1 is too coarse. Default 0.5.
    #[serde(default = "default_leiden_gamma_step")]
    pub leiden_gamma_step: f64,
    /// Branching factor for the VSA tree hierarchy.
    /// Each level targets n_nodes / branching_factor communities. Default 10.
    #[serde(default = "default_tree_branching_factor")]
    pub tree_branching_factor: usize,
    /// Additive edge weight boost for co-located chunks (same file/directory).
    /// boost = sigma * exp(-lambda * lca_distance). Default 0.1 (same-file boost).
    #[serde(default = "default_colocation_sigma")]
    pub structural_colocation_sigma: f32,
    /// Decay rate for file co-location boost. λ = ln(2)/2 ≈ 0.347 halves per dir level.
    #[serde(default = "default_colocation_lambda")]
    pub structural_colocation_lambda: f32,
    /// Number of iterative local mean-centering passes before building the final graph.
    /// Each pass: run Leiden → subtract community mean from member vectors → renormalize.
    /// Breaks the BGE anisotropic cone so anti-correlated pairs produce negative edges.
    /// 0 = no centering (default). 3-5 is typically enough to converge.
    #[serde(default)]
    pub leiden_center_iters: usize,
    /// Minimum fraction of a node's total edge weight that must flow to a non-primary
    /// community for it to receive secondary membership there.
    /// 0.05 = 5% of connections. Lower → more secondaries, higher → fewer but more precise.
    #[serde(default = "default_secondary_min_fraction")]
    pub secondary_min_fraction: f32,
}

fn default_leiden_gamma() -> f64 {
    1.13
}

fn default_leiden_threshold() -> f32 {
    0.0
}

fn default_leiden_target_neighbors() -> usize {
    15
}

fn default_leiden_threshold_ceiling() -> f32 {
    0.90
}

fn default_leiden_gamma_step() -> f64 {
    0.5
}

fn default_tree_branching_factor() -> usize {
    10
}

fn default_colocation_sigma() -> f32 {
    0.1
}

fn default_colocation_lambda() -> f32 {
    0.347 // ln(2)/2: halves per directory level
}

fn default_secondary_min_fraction() -> f32 {
    0.05
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
    /// Beam width for hierarchical tree traversal during search.
    /// Number of top-scoring bundles kept at each level of the Leiden tree.
    #[serde(default = "default_beam_width")]
    pub beam_width: usize,
    /// Whitening strength applied to the query after each tree level.
    /// Subtracts whitening * best_bundle_vec from the query to sharpen
    /// within-cluster discrimination at the next level. 0.0 = off.
    #[serde(default = "default_whitening")]
    pub whitening: f32,
    /// Weight given to the community (bundle) score when blending with chunk cosine.
    /// final_score = (1-w) * chunk_cosine + w * bundle_score
    /// 0.0 = pure chunk similarity, higher values favor chunks in the best-matching community.
    #[serde(default)]
    pub community_score_weight: f32,
}

fn default_mmr_lambda() -> f32 {
    0.9
}

fn default_bm25_weight() -> f32 {
    0.5
}

fn default_beam_width() -> usize {
    8
}

fn default_whitening() -> f32 {
    0.24
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
            leiden_gamma: default_leiden_gamma(),
            leiden_threshold: default_leiden_threshold(),
            leiden_target_neighbors: default_leiden_target_neighbors(),
            leiden_threshold_ceiling: default_leiden_threshold_ceiling(),
            leiden_gamma_step: default_leiden_gamma_step(),
            tree_branching_factor: default_tree_branching_factor(),
            structural_colocation_sigma: default_colocation_sigma(),
            structural_colocation_lambda: default_colocation_lambda(),
            leiden_center_iters: 0,
            secondary_min_fraction: default_secondary_min_fraction(),
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
            beam_width: default_beam_width(),
            whitening: default_whitening(),
            community_score_weight: 0.0,
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
