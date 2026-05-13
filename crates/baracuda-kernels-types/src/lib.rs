//! # baracuda-kernels-types
//!
//! Shared type vocabulary for the baracuda ML kernel facade.
//!
//! This crate has no behavior of its own — it ships pure-data types that
//! are common to every member of the kernel facade ecosystem:
//!
//! - The [`Element`] / [`IntElement`] / [`BiasElement`] trait hierarchy
//!   plus the [`ScalarType`] alpha/beta projection.
//! - Wrapper types ([`S8`], [`U8`], [`F32Strict`]) that drive kernel
//!   selection at the Rust type level.
//! - Tag enums ([`ElementKind`], [`MathPrecision`], [`BiasElementKind`],
//!   [`LayoutSku`], [`ArchSku`], [`EpilogueKind`], [`ActivationKind`]).
//! - Borrowed device-buffer views ([`MatrixRef`], [`MatrixMut`],
//!   [`VectorRef`]).
//! - Plan-layer descriptors ([`PlanPreference`], [`PrecisionGuarantee`],
//!   [`Workspace`]).
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
pub mod plan;

pub use element::{
    BiasElement, BiasElementKind, Element, ElementKind, F32Strict, IntElement, MathPrecision,
    S8, ScalarType, U8,
};
pub use layout::{ActivationKind, ArchSku, EpilogueKind, LayoutSku};
pub use matrix::{MatrixMut, MatrixRef, VectorRef};
pub use plan::{PlanPreference, PrecisionGuarantee, Workspace};
