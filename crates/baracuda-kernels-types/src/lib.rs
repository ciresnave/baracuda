//! # baracuda-kernels-types
//!
//! Shared type vocabulary for the baracuda ML kernel facade.
//!
//! This crate has no behavior of its own — it ships pure-data types that
//! are common to every member of the kernel facade ecosystem:
//!
//! - The [`KernelDtype`] umbrella marker trait + the [`Element`] /
//!   [`IntElement`] / [`FpElement`] / [`BinElement`] / [`BiasElement`]
//!   trait hierarchy plus the [`ScalarType`] alpha/beta projection.
//! - Wrapper types ([`S8`], [`U8`], [`S4`], [`U4`], [`Bin`],
//!   [`F32Strict`], [`Fp8E4M3`], [`Fp8E5M2`]) that drive kernel
//!   selection at the Rust type level.
//! - Tag enums ([`ElementKind`], [`MathPrecision`], [`BiasElementKind`],
//!   [`LayoutSku`], [`ArchSku`], [`EpilogueKind`], [`ActivationKind`]).
//! - Borrowed device-buffer views ([`MatrixRef`], [`MatrixMut`],
//!   [`VectorRef`]).
//! - Plan-layer descriptors ([`PlanPreference`], [`PrecisionGuarantee`],
//!   [`Workspace`]).
//!
//! # 1.0-freeze stability (Phase 28)
//!
//! Most op-family discriminant enums (`BinaryKind`, `UnaryKind`,
//! `ReduceKind`, `ScanKind`, `SoftmaxKind`, `NormalizationKind`,
//! `LossKind`, `LinalgKind`, `ConvKind`, `PoolKind`, `AttentionKind`,
//! `IndexingKind`, `SegmentKind`, `EmbeddingKind`, `QuantizeKind`,
//! `GgufBlockFormat`, `MoeKind`, `SortKind`, `ImageKind`, `FftKind`,
//! `RandomKind`, `ShapeLayoutKind`, `BinaryCmpKind`, `TernaryKind`,
//! `GatedActivationKind`, `ArgReduceKind`, `PadMode`) plus the category
//! / backend tag enums (`OpCategory`, `BackendKind`) and the auxiliary
//! index-dtype tag enums (`IndexElementKind`, `IndexOutputKind`) are
//! marked `#[non_exhaustive]` as of Phase 28. Downstream `match` arms
//! must include a `_ =>` catch-all to remain forward-compatible with
//! new variants.
//!
//! The kernel-dispatch-keying enums `ElementKind`, `BiasElementKind`,
//! `LayoutSku`, `ArchSku`, `EpilogueKind`, `ActivationKind`, and
//! `Workspace<'a>` are **intentionally** left exhaustive — they're
//! the keys per-arch / per-layout / per-epilogue / per-bias-dtype
//! kernel dispatchers exhaustively match on, so adding a new variant
//! is a deliberate workspace-wide event that ought to surface as a
//! build break across every match site.
//!
//! The types here were previously defined in `baracuda-cutlass::types`;
//! they were lifted out so that `baracuda-kernels` (the unified ML op
//! facade) and any sibling wrapper crate (`baracuda-cublas`,
//! `baracuda-cudnn`, …) can share one vocabulary instead of each
//! re-declaring its own.
//!
//! The trait `Element` was previously named `CutlassElement`;
//! `baracuda-cutlass` keeps the old name available as a re-export for
//! back-compat. The semantics are unchanged.

#![deny(missing_docs)]

pub mod element;
pub mod layout;
pub mod matrix;
pub mod ops;
pub mod plan;
pub mod sku;
pub mod tensor;

pub use element::{
    BiasElement, BiasElementKind, Bin, BinElement, Bool, Complex32, Complex64, Element,
    ElementKind, F32Strict, Fp8E4M3, Fp8E5M2, FpElement, IndexElement, IndexElementKind,
    IndexOutputElement, IndexOutputKind, IntElement, KernelDtype, MathPrecision, S4, S8,
    ScalarType, U4, U8,
};
pub use layout::{ActivationKind, ArchSku, EpilogueKind, LayoutSku};
pub use matrix::{MatrixMut, MatrixRef, VectorRef};
pub use ops::{
    ArgReduceKind, AttentionKind, BinaryCmpKind, BinaryKind, ConvKind, CrossEntropyTargetKind,
    EmbeddingKind, FftKind, FillMode, GatedActivationKind, GgufBlockFormat, ImageKind,
    IndexingKind, LinalgKind, LossKind, LossReduction, MoeKind, NormalizationKind, PadMode,
    PoolKind, QuantizeKind, RandomKind, ReduceKind, ScanKind, SegmentKind, ShapeLayoutKind,
    SoftmaxKind, SortKind, TernaryKind, UnaryKind,
};
pub use plan::{PlanPreference, PrecisionGuarantee, Workspace};
pub use sku::{BackendKind, KernelSku, OpCategory};
pub use tensor::{contiguous_stride, strides_equal, TensorMut, TensorRef};
