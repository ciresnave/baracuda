//! GGUF block-format quantization plans — Phase 8 Milestone 8.4.
//!
//! GGUF (GPT-Generated Unified Format) is the weight-storage format used
//! by [llama.cpp](https://github.com/ggerganov/llama.cpp) and the
//! broader local-inference ecosystem. This module ships two ops:
//!
//! - [`GgufDequantizePlan`]: unpack a GGUF-packed weight buffer into a
//!   dense f32 tensor. Used for ahead-of-time weight unpacking and as
//!   a fallback path when batched FP matmul is preferred over the
//!   dequant-fused MMVQ path.
//! - [`GgufMmvqPlan`]: fused dequant + matrix-vector multiply
//!   (`out[r] = Σ_c W_q[r, c] · y[c]`). This is the inference-time
//!   "decode-step" matmul — single activation vector, full FP precision
//!   output, no intermediate dequant materialization.
//!
//! ## Block formats
//!
//! Both plans accept any of the eleven GGUF block formats via the
//! [`GgufBlockFormat`] enum on the descriptor:
//!
//! - **Type-0/1** (32 elements per block, single scale ± min):
//!   `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`.
//! - **k-quants** (256 elements per super-block, multi-level scales):
//!   `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`.
//!
//! [`GgufMmvqPlan`] does NOT support `Q8_K` — llama.cpp / Fuel reserve
//! it as a CPU-side intermediate (no MMVQ kernel exists upstream).
//! [`GgufDequantizePlan`] supports all eleven formats.
//!
//! ## Dtype coverage
//!
//! - Dequant output: `f32`. (`f16` output deferred.)
//! - MMVQ activation / output: `f32`. (`f16` / `bf16` activation deferred.)
//!
//! ## Lineage
//!
//! Vendored from [llama.cpp](https://github.com/ggerganov/llama.cpp)
//! via fuel-cuda-kernels. See
//! `crates/baracuda-kernels-sys/LICENSE-thirdparty.md` for the full
//! attribution chain and `kernels/include/baracuda_gguf.cuh` for kernel-
//! level lineage notes.

pub mod block_formats;
pub mod dequantize;
pub mod mmvq;

pub use block_formats::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8K,
};
pub use dequantize::{GgufDequantizeArgs, GgufDequantizeDescriptor, GgufDequantizePlan};
pub use mmvq::{GgufMmvqArgs, GgufMmvqDescriptor, GgufMmvqPlan};
