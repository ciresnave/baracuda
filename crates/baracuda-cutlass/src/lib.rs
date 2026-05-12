//! # baracuda-cutlass
//!
//! Safe Rust wrapper for compiled CUTLASS kernels in the baracuda
//! ecosystem. Plan-based GEMM and grouped-GEMM API with caller-supplied
//! workspace, typed device-buffer arguments, and capture-safe launches.
//!
//! See the crate `README.md` for the v0 scope and the design rationale.
//! See [`baracuda-cutlass-kernels-sys`] for the underlying compiled
//! template instantiations.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use baracuda_cutlass::{
//!     EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku,
//!     MatrixMut, MatrixRef, PlanPreference, Workspace,
//! };
//! use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
//! use half::f16;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = Context::new(&Device::get(0)?)?;
//! let stream = Stream::new(&ctx)?;
//!
//! let m = 128i32; let n = 128i32; let k = 128i32;
//! let dev_a: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * k) as usize)?;
//! let dev_b: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (k * n) as usize)?;
//! let mut dev_d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, (m * n) as usize)?;
//!
//! let desc = GemmDescriptor {
//!     m, n, k,
//!     layout: LayoutSku::Rcr,
//!     epilogue: EpilogueKind::Identity,
//! };
//! let plan = GemmPlan::<f16>::select(&stream, &desc, PlanPreference::default())?;
//! let args = GemmArgs::<f16> {
//!     a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
//!     b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: k as i64 },
//!     c: None,
//!     d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
//!     bias: None,
//!     alpha: 1.0,
//!     beta: 0.0,
//! };
//! plan.can_implement(&args)?;
//! plan.run(&stream, Workspace::None, args)?;
//! # Ok(()) }
//! ```
//!
//! [`baracuda-cutlass-kernels-sys`]: https://docs.rs/baracuda-cutlass-kernels-sys

#![deny(missing_docs)]

pub mod error;
pub mod plan;
pub mod types;

pub use error::{Error, Result};
pub use plan::{BatchedGemmPlan, GemmPlan, GroupedGemmPlan, IntGemmPlan, PreparedGroupedGemm};
pub use types::{
    ActivationKind, ArchSku, BatchedGemmArgs, BatchedGemmDescriptor, BiasElement, BiasElementKind,
    CutlassElement, ElementKind, EpilogueKind, F32Strict, GemmArgs, GemmDescriptor, GemmSku,
    GroupedPlanPreference, GroupedProblem, GroupedScheduleMode, IntElement, IntGemmArgs,
    IntGemmDescriptor, LayoutSku, MathPrecision, MatrixMut, MatrixRef, PlanPreference,
    PrecisionGuarantee, S8, ScalarType, U8, VectorRef, Workspace,
};
