use anyhow::Result;

#[cfg(feature = "migraphx")]
use crate::commands::search::resolve_model;
#[cfg(feature = "migraphx")]
use anyhow::Context;
#[cfg(feature = "migraphx")]
use colgrep::{config, onnx_runtime};
#[cfg(feature = "migraphx")]
use colgrep::{ensure_model, Config};
#[cfg(feature = "migraphx")]
use next_plaid_onnx::{Colbert, ExecutionProvider};

#[cfg(feature = "migraphx")]
use colgrep::acceleration::{apply_acceleration_mode, AccelerationMode};

pub fn cmd_warm_rocm_cache(cli_model: Option<&str>) -> Result<()> {
    #[cfg(not(feature = "migraphx"))]
    {
        let _ = cli_model;
        anyhow::bail!(
            "ROCm/MIGraphX support is not compiled. Rebuild colgrep with --features rocm."
        );
    }

    #[cfg(feature = "migraphx")]
    {
        apply_acceleration_mode(AccelerationMode::ForceGpu);
        onnx_runtime::ensure_onnx_runtime().context("Failed to initialize ONNX Runtime")?;

        let model_id = resolve_model(cli_model);
        let config = Config::load().unwrap_or_default();
        let quantized = !config.use_fp32();
        let model_path = ensure_model(Some(&model_id), false)?;

        eprintln!("🤖 Model: {model_id}");
        eprintln!("🔥 Warming ROCm/MIGraphX static-shape cache...");

        let model = Colbert::builder(&model_path)
            .with_quantized(quantized)
            .with_parallel(1)
            .with_batch_size(config::DEFAULT_BATCH_SIZE_MIGRAPHX)
            .with_execution_provider(ExecutionProvider::MIGraphX)
            .build()
            .context("Failed to load ColBERT model for ROCm cache warming")?;

        let shapes = model.migraphx_static_shapes();
        eprintln!("Planned shapes: {shapes:?}");
        let warmed = model.warm_migraphx_static_shape_cache()?;
        eprintln!(
            "✅ Warmed {warmed} ROCm/MIGraphX shape cache(s). \
             Set NEXT_PLAID_MIGRAPHX_WARM_MAX_SEQUENCE_LEN=2048 to include long-document shapes."
        );
        Ok(())
    }
}
