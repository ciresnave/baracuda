//! # baracuda-kernels
//!
//! Unified ML op facade for the baracuda CUDA ecosystem.
//!
//! Exposes every primitive an ML framework would expect (union of
//! PyTorch `torch.*` + `nn.functional` and JAX `lax.*` / `numpy` ops)
//! through a single Plan-based Rust surface, internally dispatching to:
//!
//! 1. An NVIDIA-library wrapper crate when one already covers the op
//!    (`baracuda-cublas`, `baracuda-cudnn`, `baracuda-cufft`,
//!    `baracuda-cusparse`, `baracuda-cusolver`, `baracuda-curand`,
//!    `baracuda-cutensor`, `baracuda-npp`, `baracuda-cvcuda`,
//!    `baracuda-cutlass`).
//! 2. A bespoke `.cu` kernel shipped in
//!    [`baracuda-kernels-sys`](https://docs.rs/baracuda-kernels-sys)
//!    when no NVIDIA library covers it (or covers it poorly at relevant
//!    shapes).
//!
//! Callers import **one** crate and reach for **one** API style; the
//! dispatch decision is an internal detail driven by `select`.
//!
//! ## Status
//!
//! Phase 0 scaffolding: the facade currently re-exports the existing
//! `baracuda-cutlass` plan types so downstream callers can switch their
//! import paths now (`use baracuda_kernels::IntGemmPlan;` instead of
//! `use baracuda_cutlass::IntGemmPlan;`) and gain the new layouts /
//! dtypes as later phases land — no API breakage at the switch.
//!
//! The first bespoke kernels (int8 GEMM RRR — `LayoutSku::Rrr` over
//! `{S8, U8} × {Identity, Bias, BiasRelu, BiasGelu, BiasSilu} × {f32, i32}` bias)
//! land in workspace alpha.16.

#![deny(missing_docs)]

// Re-export the shared type vocabulary.
pub use baracuda_kernels_types::{
    ActivationKind, ArchSku, BiasElement, BiasElementKind, Element, ElementKind, EpilogueKind,
    F32Strict, IntElement, LayoutSku, MathPrecision, MatrixMut, MatrixRef, PlanPreference,
    PrecisionGuarantee, S8, ScalarType, U8, VectorRef, Workspace,
};

// Re-export the float-GEMM plan types from baracuda-cutlass unchanged —
// no bespoke path exists for float GEMM yet, the CUTLASS surface is
// the one true entry.
pub use baracuda_cutlass::{
    BatchedGemmArgs, BatchedGemmDescriptor, BatchedGemmPlan, Error, GemmArgs, GemmDescriptor,
    GemmPlan, GemmSku, GroupedGemmPlan, GroupedPlanPreference, GroupedProblem, GroupedScheduleMode,
    PreparedGroupedGemm, Result,
};

// Unified GEMM plan dispatchers. Today exposes only `IntGemmPlan` (RCR
// → CUTLASS, RRR → bespoke); float GEMM and the FP8 / int4 / bin
// dispatchers join later.
pub mod gemm;

pub use gemm::{IntGemmArgs, IntGemmDescriptor, IntGemmPlan};
