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

// Phase 0 facade: re-export the existing CUTLASS plan types so callers
// can switch import paths. The actual unified-dispatcher wrappers
// (`IntGemmPlan` with a `Backend::Cutlass | Backend::Bespoke` enum) land
// in Phase 1 once the bespoke int8 RRR kernels exist.
pub use baracuda_cutlass::{
    BatchedGemmArgs, BatchedGemmDescriptor, BatchedGemmPlan, Error, GemmArgs, GemmDescriptor,
    GemmPlan, GemmSku, GroupedGemmPlan, GroupedPlanPreference, GroupedProblem, GroupedScheduleMode,
    IntGemmArgs, IntGemmDescriptor, IntGemmPlan, PreparedGroupedGemm, Result,
};
