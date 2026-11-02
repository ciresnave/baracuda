//! GGUF block-format quantization plans — Phase 8 Milestone 8.4.
//!
//! GGUF (GPT-Generated Unified Format) is the weight-storage format used
//! by [llama.cpp](https://github.com/ggerganov/llama.cpp) and the
//! broader local-inference ecosystem. This module ships:
//!
//! - [`GgufDequantizePlan`]: unpack a GGUF-packed weight buffer into a
//!   dense f32 tensor, for dequantize-then-multiply with a float GEMM.
//! - [`GgufMmvqPlan`] / [`GgufMmvqBatchedPlan`]: fused dequant + matrix-
//!   vector multiply with FP activations (`out[r] = Σ_c W_q[r, c] · y[c]`).
//!
//! The Q8_1-staged multi-M launchers and the MoE-GGUF kernels (integer-dot)
//! were removed in 0.1.0-alpha.80 (board 307).
//!
//! ## Block formats
//!
//! Both plans accept any of the eleven GGUF block formats via the
//! `GgufBlockFormat` enum on the descriptor:
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
pub mod mmvq_batched;

pub use block_formats::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8K,
};
pub use dequantize::{GgufDequantizeArgs, GgufDequantizeDescriptor, GgufDequantizePlan};
pub use mmvq::{GgufMmvqArgs, GgufMmvqDescriptor, GgufMmvqPlan};
pub use mmvq_batched::{
    GgufMmvqBatchedActivation, GgufMmvqBatchedArgs, GgufMmvqBatchedDescriptor,
    GgufMmvqBatchedFormat, GgufMmvqBatchedPlan,
};
