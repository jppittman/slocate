use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use slocate::embed::{available_devices, Embedder};

fn device_label(device: &candle_core::Device) -> &'static str {
    match device {
        candle_core::Device::Cpu => "cpu",
        candle_core::Device::Cuda(_) => "cuda",
        candle_core::Device::Metal(_) => "metal",
    }
}

fn sample_texts() -> Vec<String> {
    vec![
        // Short: function signature
        "fn embed_batch(texts: &[String]) -> Result<Vec<Vec<f32>>>".to_string(),
        // Medium: function body
        r#"pub fn reindex_workspace(embedder: &Embedder, config: &Config, workspace_root: &Path) -> Result<()> {
    let _lock = ReindexLock::acquire(workspace_root)?;
    let index_dir = registry::index_dir(workspace_root)?;
    let mut db = store::Store::open(&index_dir)?;
    db.ensure_file_meta_table()?;
    let old_meta = db.load_file_meta()?;
    let (to_embed, unchanged_paths, deleted_paths) = classify_files(config, workspace_root, &old_meta)?;
    Ok(())
}"#.to_string(),
        // Long: module-level context
        r#"//! Wait-free SPSC ring buffer.
//! Single-producer, single-consumer bounded queue. The producer owns the tail
//! (write position) and the consumer owns the head (read position). Send is a
//! single AtomicUsize::store(Release), recv is a single AtomicUsize::load(Acquire).
//! No CAS, no retry, no contention between producers — each producer gets its own channel.
//! Cache-line padding on head and tail prevents false sharing between producer and consumer cores.
pub struct SpscSender<T> {
    ring: Arc<RingBuffer<T>>,
    cached_tail: Cell<usize>,
    cached_head: Cell<usize>,
}
pub struct SpscReceiver<T> {
    ring: Arc<RingBuffer<T>>,
    cached_head: usize,
    cached_tail: usize,
}"#.to_string(),
    ]
}

fn bench_embed_single(c: &mut Criterion) {
    let config = slocate::config::Config::load().unwrap_or_default();
    let model_dir = config.model_dir();
    let text = "fn process_batch(items: &[Item]) -> Vec<Result<Embedding, Error>>";

    let mut group = c.benchmark_group("embed_single");
    for device in available_devices() {
        let label = device_label(&device);
        let embedder = match Embedder::load_on(&model_dir, device) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skipping {label}: {e}");
                continue;
            }
        };
        // Warmup: amortise shader compilation / CUDA JIT on first call.
        let _ = embedder.embed(text);

        group.bench_function(label, |b| {
            b.iter(|| embedder.embed(text).unwrap())
        });
    }
    group.finish();
}

fn bench_embed_batch(c: &mut Criterion) {
    let config = slocate::config::Config::load().unwrap_or_default();
    let model_dir = config.model_dir();
    let texts = sample_texts();

    for batch_size in [1usize, 16, 64] {
        let mut group = c.benchmark_group(format!("embed_batch/{batch_size}"));
        let batch: Vec<String> = texts.iter().cycle().take(batch_size).cloned().collect();

        for device in available_devices() {
            let label = device_label(&device);
            let embedder = match Embedder::load_on(&model_dir, device) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("skipping {label}: {e}");
                    continue;
                }
            };
            // Warmup pass.
            let _ = embedder.embed_batch(&batch);

            group.bench_with_input(BenchmarkId::new(label, batch_size), &batch, |b, batch| {
                b.iter(|| embedder.embed_batch(batch).unwrap())
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_embed_single, bench_embed_batch);
criterion_main!(benches);
