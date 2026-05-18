//! Quantization op family — Category P.
//!
//! Phase 8 splits across two parallel milestones:
//!
//! - **Milestone 8.1** (sibling): per-tensor + per-channel quantize /
//!   dequantize plus `fake_quantize`. Owns
//!   `crates/baracuda-kernels-sys/kernels/quantize/per_tensor.cu` /
//!   `per_channel.cu` / `fake_quantize.cu` and the Rust plans for those
//!   ops in this `quantize/` module.
//!
//! - **Milestone 8.2** (this work): per-token + per-group quantize /
//!   dequantize plus their STE backwards. Used by LLM activation (W8A8
//!   per-row) and weight (GPTQ-style INT4 per-group, `g=128`) quant.
//!   Owns
//!   `crates/baracuda-kernels-sys/kernels/quantize/per_token.cu` /
//!   `per_group.cu` and the plans in this module.
//!
//! The two milestones share **append-only** edits to this file, to
//! `crate::lib`'s re-exports, and to `baracuda-kernels-sys/src/lib.rs`.
//! No existing entry is rewritten.
//!
//! Trailblazer dtype coverage: input FP ∈ {`f32`, `f64`, `f16`, `bf16`};
//! output int ∈ {`s8`, `u8`}. Sub-byte packed types (`s4` / `u4`) are
//! deferred to a later milestone.
//!
//! Backward convention is the Straight-Through Estimator (STE):
//! `dx = (dy / scale) * 1[qmin < q < qmax]`. The in-range mask is
//! recomputed inside the BW kernel from the saved input — callers must
//! retain the input tensor for autograd (which they would do anyway).

// --- Milestone 8.1 modules (per-tensor + per-channel + fake_quantize). ---
pub mod dequantize_per_channel;
pub mod dequantize_per_channel_backward;
pub mod dequantize_per_tensor;
pub mod dequantize_per_tensor_backward;
pub mod fake_quantize;
pub mod fake_quantize_backward;
pub mod per_channel;
pub mod per_channel_backward;
pub mod per_tensor;
pub mod per_tensor_backward;

// --- Milestone 8.2 modules (per-token + per-group). Full coverage of
//     FW + STE BW + dequant FW + straight-through dequant BW for both
//     per-token and per-group quant. ---
pub mod dequantize_per_group;
pub mod dequantize_per_group_backward;
pub mod dequantize_per_token;
pub mod dequantize_per_token_backward;
pub mod per_group;
pub mod per_group_backward;
pub mod per_token;
pub mod per_token_backward;

// --- Milestone 8.3 modules — composing ops on top of 8.1 / 8.2. ----
pub mod dynamic_range;
pub mod quantized_linear;

// --- Milestone 8.4 module — GGUF block-format quant family (vendored
//     from llama.cpp via fuel-cuda-kernels). Full block-format coverage
//     for both dequant and MMVQ. Phase 11.4 added a bespoke Q8_K MMVQ
//     (upstream llama.cpp / Fuel ship only Q8_K dequant). ----
pub mod gguf;

pub use dequantize_per_channel::{
    DequantizePerChannelArgs, DequantizePerChannelDescriptor, DequantizePerChannelPlan,
};
pub use dequantize_per_channel_backward::{
    DequantizePerChannelBackwardArgs, DequantizePerChannelBackwardDescriptor,
    DequantizePerChannelBackwardPlan,
};
pub use dequantize_per_tensor::{
    DequantizePerTensorArgs, DequantizePerTensorDescriptor, DequantizePerTensorPlan,
};
pub use dequantize_per_tensor_backward::{
    DequantizePerTensorBackwardArgs, DequantizePerTensorBackwardDescriptor,
    DequantizePerTensorBackwardPlan,
};
pub use fake_quantize::{FakeQuantizeArgs, FakeQuantizeDescriptor, FakeQuantizePlan};
pub use fake_quantize_backward::{
    FakeQuantizeBackwardArgs, FakeQuantizeBackwardDescriptor, FakeQuantizeBackwardPlan,
};
pub use per_channel::{QuantizePerChannelArgs, QuantizePerChannelDescriptor, QuantizePerChannelPlan};
pub use per_channel_backward::{
    QuantizePerChannelBackwardArgs, QuantizePerChannelBackwardDescriptor,
    QuantizePerChannelBackwardPlan,
};
pub use per_tensor::{QuantizePerTensorArgs, QuantizePerTensorDescriptor, QuantizePerTensorPlan};
pub use per_tensor_backward::{
    QuantizePerTensorBackwardArgs, QuantizePerTensorBackwardDescriptor,
    QuantizePerTensorBackwardPlan,
};

pub use per_token::{QuantizePerTokenArgs, QuantizePerTokenDescriptor, QuantizePerTokenPlan};
pub use per_token_backward::{
    QuantizePerTokenBackwardArgs, QuantizePerTokenBackwardDescriptor, QuantizePerTokenBackwardPlan,
};
pub use dequantize_per_token::{
    DequantizePerTokenArgs, DequantizePerTokenDescriptor, DequantizePerTokenPlan,
};
pub use dequantize_per_token_backward::{
    DequantizePerTokenBackwardArgs, DequantizePerTokenBackwardDescriptor,
    DequantizePerTokenBackwardPlan,
};
pub use per_group::{QuantizePerGroupArgs, QuantizePerGroupDescriptor, QuantizePerGroupPlan};
pub use per_group_backward::{
    QuantizePerGroupBackwardArgs, QuantizePerGroupBackwardDescriptor, QuantizePerGroupBackwardPlan,
};
pub use dequantize_per_group::{
    DequantizePerGroupArgs, DequantizePerGroupDescriptor, DequantizePerGroupPlan,
};
pub use dequantize_per_group_backward::{
    DequantizePerGroupBackwardArgs, DequantizePerGroupBackwardDescriptor,
    DequantizePerGroupBackwardPlan,
};

// --- Milestone 8.3 exports ---
pub use dynamic_range::{
    DynamicRangeMode, DynamicRangeQuantizeArgs, DynamicRangeQuantizeDescriptor,
    DynamicRangeQuantizePlan, DynamicRangeScope,
};
pub use quantized_linear::{
    QuantizedLinearArgs, QuantizedLinearDescriptor, QuantizedLinearPlan,
};

// --- Milestone 8.4 exports — GGUF block-format dequant + MMVQ ---
pub use gguf::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8K, GgufDequantizeArgs, GgufDequantizeDescriptor, GgufDequantizePlan,
    GgufMmvqArgs, GgufMmvqDescriptor, GgufMmvqPlan,
};

use baracuda_cutlass::{Error, Result};

/// Shared status-code mapper (mirrors `indexing::gather::map_status` and
/// `segment::map_status`).
pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}

/// Element-kind check shared across the per-token / per-group plans.
/// Returns Ok if `tin_kind` is one of the four supported FP dtypes.
pub(crate) fn validate_input_element(
    tin_kind: baracuda_kernels_types::ElementKind,
    plan_name: &'static str,
) -> Result<()> {
    use baracuda_kernels_types::ElementKind;
    if !matches!(
        tin_kind,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(plan_name));
    }
    Ok(())
}

/// Output-kind check shared across the per-token / per-group plans.
/// Returns Ok if `tout_kind` is `S8` or `U8`.
pub(crate) fn validate_output_element(
    tout_kind: baracuda_kernels_types::ElementKind,
    plan_name: &'static str,
) -> Result<()> {
    use baracuda_kernels_types::ElementKind;
    if !matches!(tout_kind, ElementKind::S8 | ElementKind::U8) {
        return Err(Error::Unsupported(plan_name));
    }
    Ok(())
}

/// Default `qmin` / `qmax` for an output integer dtype. Today wired for
/// the two trailblazer output kinds — [`baracuda_kernels_types::S8`]
/// (`[-128, 127]`) and [`baracuda_kernels_types::U8`] (`[0, 255]`).
///
/// Returns `None` for unsupported kinds; the plan's `select()` returns
/// `Error::Unsupported` in that case.
#[inline]
pub fn default_q_range(out_kind: baracuda_kernels_types::ElementKind) -> Option<(i32, i32)> {
    use baracuda_kernels_types::ElementKind;
    match out_kind {
        ElementKind::S8 => Some((-128, 127)),
        ElementKind::U8 => Some((0, 255)),
        _ => None,
    }
}
