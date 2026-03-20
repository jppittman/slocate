use std::fs;
use std::io;
use std::path::Path;

const FILES: &[(&str, &str)] = &[
    ("config.json", "config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("model.safetensors", "model.safetensors"),
];

/// Infer the BAAI HuggingFace repo name from the model directory name.
/// Expects the directory to be named after the model, e.g. "bge-small-en-v1.5".
/// Falls back to bge-small-en-v1.5 if the directory name isn't recognized.
fn repo_from_dir(model_dir: &Path) -> String {
    let dir_name = model_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("bge-small-en-v1.5");
    format!("BAAI/{dir_name}")
}

/// Ensure all model files are present in `model_dir` and belong to the
/// expected BERT architecture.
///
/// Downloads from HuggingFace via HTTPS if any are missing or if a previous
/// model occupies the directory. The HuggingFace repo is inferred from the
/// model directory name (e.g. `bge-base-en-v1.5` → `BAAI/bge-base-en-v1.5`).
pub fn ensure_model(model_dir: &Path) -> crate::error::Result<()> {
    if FILES
        .iter()
        .all(|(_, local)| model_dir.join(local).exists())
    {
        if is_bert_config(model_dir) {
            return Ok(());
        }
        log::warn!("Model type changed — re-downloading.");
    }
    let repo = repo_from_dir(model_dir);
    let base = format!("https://huggingface.co/{repo}/resolve/main");
    log::info!("Downloading {repo} to {} ...", model_dir.display());
    fs::create_dir_all(model_dir)?;
    for (remote, local) in FILES {
        log::info!("  {local}");
        let url = format!("{base}/{remote}");
        let dest_path = model_dir.join(local);
        let resp = ureq::get(&url).call().map_err(|e| {
            crate::error::Error::Download(format!("failed to download {local}: {e}"))
        })?;
        let mut dest = fs::File::create(&dest_path)?;
        let mut reader = resp.into_body().into_reader();
        io::copy(&mut reader, &mut dest)?;
    }
    log::info!("Download complete.");
    Ok(())
}

fn is_bert_config(model_dir: &Path) -> bool {
    let config_path = model_dir.join("config.json");
    let Ok(bytes) = std::fs::read(&config_path) else {
        return false;
    };
    let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    val.get("model_type").and_then(|v| v.as_str()) == Some("bert")
}
