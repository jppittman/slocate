use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

/// Sentence embedder backed by BGE-base-en-v1.5 (BERT, 109M params, 768-dim).
///
/// `BertModel::forward` takes `&self`, so the struct is naturally `Send + Sync`
/// as long as all fields are — `Tensor` and `Tokenizer` both are.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

/// Maximum batch size for `embed_batch`. Larger = more GPU utilization,
/// but BERT attention is O(batch * seq^2) so memory grows fast.
const MAX_BATCH: usize = 64;

impl Embedder {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        crate::download::ensure_model(model_dir)?;

        let device = pick_device();
        let dtype = pick_dtype(&device);

        let config_path = model_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path)
            .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
        let config: Config = serde_json::from_slice(&config_bytes)
            .map_err(|e| format!("failed to parse config.json: {e}"))?;

        let weights_path = model_dir.join("model.safetensors");
        // SAFETY: the file is read-only and its lifetime covers the VarBuilder usage here.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], dtype, &device)
                .map_err(|e| format!("failed to load model weights: {e}"))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| format!("failed to build BertModel: {e}"))?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("failed to load tokenizer from {}: {e}", tokenizer_path.display()))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Embed `text` and return an L2-normalised f32 vector.
    ///
    /// Truncates input to 512 tokens (BERT max). Safe to call concurrently
    /// from multiple threads (no internal mutation).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenization failed: {e}"))?;

        let seq_len = encoding.get_ids().len().min(512);
        if seq_len == 0 {
            return Err("tokenization produced zero tokens".to_string());
        }

        let ids: Vec<i64> = encoding.get_ids()[..seq_len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mask: Vec<i64> = encoding.get_attention_mask()[..seq_len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        let type_ids: Vec<i64> = vec![0i64; seq_len];

        let input_ids = Tensor::from_vec(ids, (1usize, seq_len), &self.device)
            .map_err(|e| format!("input_ids tensor failed: {e}"))?;
        let attention_mask = Tensor::from_vec(mask, (1usize, seq_len), &self.device)
            .map_err(|e| format!("attention_mask tensor failed: {e}"))?;
        let token_type_ids = Tensor::from_vec(type_ids, (1usize, seq_len), &self.device)
            .map_err(|e| format!("token_type_ids tensor failed: {e}"))?;

        // Forward pass → [1, seq, hidden_size]
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| format!("model forward failed: {e}"))?;

        // Mean pooling over non-padding tokens.
        let pooled = mean_pool(&hidden, &attention_mask)
            .map_err(|e| format!("mean pooling failed: {e}"))?;

        // Flatten [1, hidden] → [hidden], convert to f32.
        let mut vec: Vec<f32> = pooled
            .flatten_all()
            .map_err(|e| format!("pooled flatten failed: {e}"))?
            .to_dtype(DType::F32)
            .map_err(|e| format!("dtype cast failed: {e}"))?
            .to_vec1()
            .map_err(|e| format!("to_vec1 failed: {e}"))?;

        if vec.is_empty() {
            return Err("model returned zero-length embedding".to_string());
        }

        // L2 normalise so cosine similarity reduces to dot product.
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter_mut().for_each(|x| *x /= norm);
        }

        Ok(vec)
    }

    /// Embed multiple texts in a single batched forward pass.
    ///
    /// Pad-and-batch: tokenize all inputs, pad to the longest sequence,
    /// run one forward pass with shape [batch, max_seq], then split the
    /// output back into per-input vectors.
    ///
    /// For large inputs, chunks into sub-batches of MAX_BATCH to bound
    /// GPU memory.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() == 1 {
            return Ok(vec![self.embed(&texts[0])?]);
        }

        let mut all_vecs = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(MAX_BATCH) {
            let batch_vecs = self.embed_batch_inner(chunk)?;
            all_vecs.extend(batch_vecs);
        }

        Ok(all_vecs)
    }

    fn embed_batch_inner(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        // Tokenize all inputs.
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                self.tokenizer
                    .encode(t.as_str(), true)
                    .map_err(|e| format!("tokenization failed: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let seq_lens: Vec<usize> = encodings
            .iter()
            .map(|e| e.get_ids().len().min(512).max(1))
            .collect();
        let max_seq = *seq_lens.iter().max().unwrap_or(&1);
        let batch_size = encodings.len();

        // Pad to max_seq and flatten into [batch, max_seq] tensors.
        let mut all_ids = Vec::with_capacity(batch_size * max_seq);
        let mut all_mask = Vec::with_capacity(batch_size * max_seq);
        let mut all_type_ids = Vec::with_capacity(batch_size * max_seq);

        for (enc, &slen) in encodings.iter().zip(seq_lens.iter()) {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for i in 0..max_seq {
                if i < slen {
                    all_ids.push(ids[i] as i64);
                    all_mask.push(mask[i] as i64);
                } else {
                    all_ids.push(0i64);   // PAD token
                    all_mask.push(0i64);  // masked out
                }
                all_type_ids.push(0i64);
            }
        }

        let input_ids = Tensor::from_vec(all_ids, (batch_size, max_seq), &self.device)
            .map_err(|e| format!("batch input_ids failed: {e}"))?;
        let attention_mask = Tensor::from_vec(all_mask, (batch_size, max_seq), &self.device)
            .map_err(|e| format!("batch attention_mask failed: {e}"))?;
        let token_type_ids = Tensor::from_vec(all_type_ids, (batch_size, max_seq), &self.device)
            .map_err(|e| format!("batch token_type_ids failed: {e}"))?;

        // Single forward pass → [batch, max_seq, hidden_size]
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| format!("batch forward failed: {e}"))?;

        // Mean pooling → [batch, hidden_size]
        let pooled = mean_pool(&hidden, &attention_mask)
            .map_err(|e| format!("batch mean_pool failed: {e}"))?;

        // Split into per-input vectors and L2-normalize.
        let pooled_f32 = pooled
            .to_dtype(DType::F32)
            .map_err(|e| format!("batch dtype cast failed: {e}"))?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let row = pooled_f32
                .get(i)
                .map_err(|e| format!("batch row {i} failed: {e}"))?;
            let mut vec: Vec<f32> = row
                .to_vec1()
                .map_err(|e| format!("batch to_vec1 row {i} failed: {e}"))?;

            if vec.is_empty() {
                return Err(format!("batch row {i} returned zero-length embedding"));
            }

            let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            results.push(vec);
        }

        Ok(results)
    }

    /// True if running on Metal GPU (callers may want to adjust batch strategy).
    pub fn is_gpu(&self) -> bool {
        matches!(self.device, Device::Metal(_))
    }
}

/// Masked mean pooling.
///
/// hidden: [batch, seq, hidden]
/// mask:   [batch, seq]  — i64, 1 for real tokens, 0 for padding
///
/// Returns: [batch, hidden]
fn mean_pool(hidden: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
    let mask_f = mask.to_dtype(hidden.dtype())?;
    let mask_f = mask_f.unsqueeze(D::Minus1)?;

    let hidden_masked = hidden.broadcast_mul(&mask_f)?;
    let sum = hidden_masked.sum(1)?;

    let count = mask_f
        .sum(1)?
        .broadcast_as(sum.shape())?;

    sum.broadcast_div(&count)
}

fn pick_device() -> Device {
    // BGE-base (109M params) is too small for Metal GPU to win over
    // CPU + Accelerate BLAS on Apple Silicon — same ~200ms latency,
    // but Metal has a 3.4s cold-start for shader compilation.
    // Use CPU by default; set SLOCATE_DEVICE=metal to force GPU
    // (useful for larger models).
    match std::env::var("SLOCATE_DEVICE").as_deref() {
        #[cfg(target_os = "macos")]
        Ok("metal") => match Device::new_metal(0) {
            Ok(d) => {
                eprintln!("[embed] Using Metal GPU");
                return d;
            }
            Err(e) => {
                eprintln!("[embed] Metal failed ({e}), falling back to CPU");
            }
        },
        _ => {}
    }
    Device::Cpu
}

fn pick_dtype(_device: &Device) -> DType {
    // F32 everywhere. Metal still accelerates the matmuls via GPU.
    // BF16 causes NaN scores when mixed with an F32-built HNSW index.
    DType::F32
}
