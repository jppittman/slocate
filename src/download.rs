use std::fs;
use std::io;
use std::path::Path;

const BASE: &str = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main";

/// (remote path relative to BASE, local filename in model_dir)
const FILES: &[(&str, &str)] = &[
    ("config.json", "config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("model.safetensors", "model.safetensors"),
];

/// Ensure all model files are present in `model_dir` and belong to the
/// expected BERT architecture.
///
/// Downloads from HuggingFace via HTTPS if any are missing or if a previous
/// model (e.g. EmbeddingGemma) occupies the directory.
pub fn ensure_model(model_dir: &Path) -> crate::error::Result<()> {
    if FILES.iter().all(|(_, local)| model_dir.join(local).exists()) {
        // Verify the config.json is actually BERT — catches leftover files
        // from a previous model without a confusing VarBuilder error.
        if is_bert_config(model_dir) {
            return Ok(());
        }
        eprintln!("[slocate] Model type changed — re-downloading.");
    }
    eprintln!(
        "[slocate] Downloading BGE-small-en-v1.5 to {} ...",
        model_dir.display()
    );
    fs::create_dir_all(model_dir)?;
    for (remote, local) in FILES {
        eprintln!("[slocate]   {local}");
        let url = format!("{BASE}/{remote}");
        let dest_path = model_dir.join(local);
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| crate::error::Error::Download(
                format!("failed to download {local}: {e}"),
            ))?;
        let mut dest = fs::File::create(&dest_path)?;
        let mut reader = resp.into_body().into_reader();
        io::copy(&mut reader, &mut dest)?;
    }
    eprintln!("[slocate] Download complete.");
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
