//! # baracuda-kernel-vocab
//!
//! The **driver-free classifier vocabulary** for the baracuda kernel facade:
//! the pure-data types that describe *what a kernel operates on* and *how it is
//! keyed for dispatch*, with **no dependency on the CUDA driver**.
//!
//! - The [`KernelDtype`] umbrella trait + the [`Element`] / [`IntElement`] /
//!   [`FpElement`] / [`BinElement`] / [`BiasElement`] hierarchy and the dtype
//!   wrapper types ([`S8`], [`U8`], [`S4`], [`U4`], [`Bin`], [`F32Strict`],
//!   [`Fp8E4M3`], [`Fp8E5M2`]).
//! - Tag enums ([`ElementKind`], [`MathPrecision`], [`ArchSku`], [`LayoutSku`],
//!   [`EpilogueKind`], [`ActivationKind`], [`OpCategory`], [`BackendKind`], the
//!   op-family discriminants in [`ops`], …).
//! - The structure-key vocabulary ([`StructureKey`], [`OperandDesc`],
//!   [`structure_key`], the per-operand axes) — the classifier INPUT both
//!   Baracuda and Fuel key on.
//! - The dispatch-table types ([`DispatchTable`], [`DispatchEntry`],
//!   [`Implementor`], [`Provenance`], …) and the plan descriptors
//!   ([`PlanPreference`], [`PrecisionGuarantee`]).
//!
//! ## Why this crate exists
//!
//! These types were carved out of `baracuda-kernels-types`, which also holds the
//! **device-view** half (`MatrixRef` / `TensorRef` / `Workspace` + the
//! `OperandDesc::from_tensor_ref` adapter). Those views pull `baracuda-driver`
//! → `baracuda-cuda-sys` (the CUDA driver FFI). Neutral consumers — the kernel
//! generator, kernel selectors, Fuel's seam — need only the *vocabulary*, not
//! the driver. Depending on this leaf crate drops that transitive CUDA-driver
//! pull entirely; the plain enums and POD structs here need only
//! `baracuda-types` (for the [`DeviceRepr`](baracuda_types::DeviceRepr) memory
//! marker) plus `half` / `float8`.
//!
//! `baracuda-kernels-types` re-exports this crate wholesale, so its public API —
//! both flat items and `::element` / `::layout` / … module paths — is unchanged
//! for existing consumers.
//!
//! This crate seeds the reference implementation of the **classifier-vocabulary**
//! sub-standard of KISS (the Kernel Interface Standards Suite).
//!
//! # 1.0-freeze stability
//!
//! The op-family discriminant enums plus the category / backend tags and the
//! auxiliary index-dtype tags are `#[non_exhaustive]`; downstream `match` arms
//! need a `_ =>` catch-all. The kernel-dispatch-keying enums (`ElementKind`,
//! `BiasElementKind`, `LayoutSku`, `ArchSku`, `EpilogueKind`, `ActivationKind`)
//! are **intentionally** exhaustive — adding a variant is a deliberate
//! workspace-wide event that should surface as a build break at every match
//! site.

#![deny(missing_docs)]

pub mod dispatch;
pub mod element;
pub mod layout;
pub mod ops;
pub mod plan;
pub mod sku;
pub mod structure_key;

pub use dispatch::{
    CandidateResult, DispatchEntry, DispatchTable, HwStamp, Implementor, MIN_FLIP_MARGIN,
    Provenance, ReportedCandidate, merge, reported_entry, seed_winner, winner_of,
};
pub use element::{
    BiasElement, BiasElementKind, Bin, BinElement, Bool, Complex32, Complex64, Element,
    ElementKind, F32Strict, Fp8E4M3, Fp8E5M2, FpElement, IndexElement, IndexElementKind,
    IndexOutputElement, IndexOutputKind, IntElement, KernelDtype, MathPrecision, S4, S8,
    ScalarType, U4, U8,
};
pub use layout::{ActivationKind, ArchSku, EpilogueKind, LayoutSku};
pub use ops::{
    ArgReduceKind, AttentionKind, BinaryCmpKind, BinaryKind, ConvKind, CrossEntropyTargetKind,
    EmbeddingKind, FftKind, FillMode, GatedActivationKind, GgufBlockFormat, ImageKind,
    IndexingKind, LinalgKind, LossKind, LossReduction, MoeKind, NormalizationKind, PadMode,
    PoolKind, QuantizeKind, RandomKind, ReduceKind, ReduceToOp, ScanKind, SegmentKind,
    ShapeLayoutKind, SoftmaxKind, SortKind, TernaryKind, UnaryKind,
};
pub use plan::{PlanPreference, PrecisionGuarantee};
pub use sku::{BackendKind, KernelSku, OpCategory};
pub use structure_key::{
    AxisMask, Contiguity, ContractionKey, DivBucket, IdxWidth, MAX_OPERANDS, MAX_RANK, MpCode,
    OperandDesc, OperandKey, QuantFacts, QuantFamily, STRUCTURE_KEY_VERSION, ScalePlacement,
    SizeClass, StructureKey, SymExtent, SymKind, VecWidth, WorkClass, dtype_token, structure_key,
    structure_key_token,
};
