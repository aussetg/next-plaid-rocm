//! Diagnostic encoder for isolating ONNX Runtime execution-provider behavior.
//!
//! This intentionally avoids ColGREP indexing so we can separate model/session
//! creation, tokenization/batching, and ONNX `session.run` performance.

use anyhow::{bail, Context, Result};
use next_plaid_onnx::{Colbert, ExecutionProvider};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    provider: ExecutionProvider,
    batch_size: Option<usize>,
    parallel: usize,
    dynamic_batch: bool,
    quantized: bool,
    input: Option<PathBuf>,
    docs: usize,
    repeat_text: String,
    repetitions: usize,
    query: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            provider: ExecutionProvider::Cpu,
            batch_size: None,
            parallel: 1,
            dynamic_batch: true,
            quantized: false,
            input: None,
            docs: 1,
            repeat_text:
                "fn configure_migraphx_provider() { /* AMD ROCm MIGraphX execution provider */ }"
                    .to_string(),
            repetitions: 1,
            query: false,
        }
    }
}

fn parse_provider(value: &str) -> Result<ExecutionProvider> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(ExecutionProvider::Auto),
        "cpu" => Ok(ExecutionProvider::Cpu),
        "cuda" => Ok(ExecutionProvider::Cuda),
        "tensorrt" => Ok(ExecutionProvider::TensorRT),
        "coreml" => Ok(ExecutionProvider::CoreML),
        "directml" => Ok(ExecutionProvider::DirectML),
        "migraphx" | "rocm" => Ok(ExecutionProvider::MIGraphX),
        _ => bail!("unknown provider: {value}"),
    }
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut iter = env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => args.model = iter.next().context("--model needs a value")?.into(),
            "--provider" => {
                args.provider = parse_provider(&iter.next().context("--provider needs a value")?)?;
            }
            "--batch-size" => {
                args.batch_size = Some(iter.next().context("--batch-size needs a value")?.parse()?);
            }
            "--parallel" => {
                args.parallel = iter.next().context("--parallel needs a value")?.parse()?;
            }
            "--static-batch" => args.dynamic_batch = false,
            "--dynamic-batch" => args.dynamic_batch = true,
            "--quantized" | "--int8" => args.quantized = true,
            "--fp32" => args.quantized = false,
            "--input" => args.input = Some(iter.next().context("--input needs a value")?.into()),
            "--docs" => args.docs = iter.next().context("--docs needs a value")?.parse()?,
            "--repeat-text" => {
                args.repeat_text = iter.next().context("--repeat-text needs a value")?;
            }
            "--repetitions" => {
                args.repetitions = iter
                    .next()
                    .context("--repetitions needs a value")?
                    .parse()?;
            }
            "--query" => args.query = true,
            "--help" | "-h" => {
                println!(
                    "Usage: diagnose_encode --model PATH [--provider cpu|rocm|migraphx|auto] \
                     [--batch-size N] [--parallel N] [--static-batch] [--quantized|--fp32] \
                     [--input texts.json-or-lines] [--docs N] [--repeat-text TEXT] [--repetitions N] [--query]"
                );
                std::process::exit(0);
            }
            _ => bail!("unknown argument: {arg}"),
        }
    }

    if args.model.as_os_str().is_empty() {
        bail!("--model is required");
    }
    args.parallel = args.parallel.max(1);
    args.docs = args.docs.max(1);
    args.repetitions = args.repetitions.max(1);
    Ok(args)
}

fn load_texts(args: &Args) -> Result<Vec<String>> {
    if let Some(input) = &args.input {
        let content = fs::read_to_string(input)
            .with_context(|| format!("failed to read input file {}", input.display()))?;
        if content.trim_start().starts_with('[') {
            serde_json::from_str(&content).context("failed to parse input as JSON string array")
        } else {
            Ok(content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_string)
                .collect())
        }
    } else {
        Ok((0..args.docs)
            .map(|i| format!("{} // diagnostic document {i}", args.repeat_text))
            .collect())
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let texts = load_texts(&args)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    eprintln!(
        "diagnose_encode config provider={} quantized={} parallel={} batch_size={:?} dynamic_batch={} query={} texts={}",
        args.provider.display_name(),
        args.quantized,
        args.parallel,
        args.batch_size,
        args.dynamic_batch,
        args.query,
        refs.len()
    );

    let build_start = Instant::now();
    let mut builder = Colbert::builder(&args.model)
        .with_quantized(args.quantized)
        .with_parallel(args.parallel)
        .with_dynamic_batch(args.dynamic_batch)
        .with_execution_provider(args.provider);
    if let Some(batch_size) = args.batch_size {
        builder = builder.with_batch_size(batch_size);
    }
    let model = builder.build()?;
    eprintln!(
        "diagnose_encode model_built_ms={:.3} effective_batch_size={} sessions={} doc_len={} query_len={} embedding_dim={}",
        elapsed_ms(build_start),
        model.batch_size(),
        model.num_sessions(),
        model.config().document_length,
        model.config().query_length,
        model.embedding_dim()
    );

    for repetition in 0..args.repetitions {
        if args.query {
            let start = Instant::now();
            let embeddings = model.encode_queries(&refs)?;
            let tokens: usize = embeddings.iter().map(|embedding| embedding.nrows()).sum();
            eprintln!(
                "diagnose_encode repetition={} query_total_ms={:.3} embeddings={} tokens={}",
                repetition,
                elapsed_ms(start),
                embeddings.len(),
                tokens
            );
        } else {
            let tokenize_start = Instant::now();
            let prepared = model.tokenize_documents_in_batches(&refs)?;
            eprintln!(
                "diagnose_encode repetition={} tokenize_total_ms={:.3}",
                repetition,
                elapsed_ms(tokenize_start)
            );

            let encode_start = Instant::now();
            let embeddings = model.encode_prepared_document_batches(prepared)?;
            let tokens: usize = embeddings.iter().map(|embedding| embedding.nrows()).sum();
            eprintln!(
                "diagnose_encode repetition={} encode_total_ms={:.3} embeddings={} tokens={}",
                repetition,
                elapsed_ms(encode_start),
                embeddings.len(),
                tokens
            );
        }
    }

    Ok(())
}
