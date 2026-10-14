//! # baracuda-kernels-types
//!
//! The **device-coupled** half of the baracuda kernel-facade vocabulary: the
//! borrowed device VIEWS ([`MatrixRef`] / [`MatrixMut`] / [`VectorRef`],
//! [`TensorRef`] / [`TensorMut`]), the [`Workspace`] scratch handle, and the
//! [`OperandDescExt::from_tensor_ref`] adapter that projects a device tensor
//! view into an [`OperandDesc`].
//!
//! The **driver-free classifier vocabulary** — the dtype / layout / op-family
//! tags, the [`Element`] trait hierarchy, [`StructureKey`] / [`OperandDesc`] +
//! [`structure_key`](mod@structure_key), the dispatch-table types, and the plan descriptors — was
//! carved into the leaf crate [`unpopped_vocab`] so that neutral
//! consumers (the kernel generator, kernel selectors, Fuel's seam) can depend on
//! the vocabulary WITHOUT this crate's `baracuda-driver` → `baracuda-cuda-sys`
//! transitive pull. This crate **re-exports all of it** (`pub use
//! unpopped_vocab::*` plus the vocabulary's module paths), so the public
//! API — flat items AND `baracuda_kernels_types::{element, layout, sku,
//! dispatch, structure_key}::*` — is unchanged for every existing consumer.
//!
//! # 1.0-freeze stability
//!
//! See [`unpopped_vocab`] for the `#[non_exhaustive]` / exhaustive tag
//! policy — those enums live there now and are re-exported here.
//!
//! # `T: Element` vs `T: DeviceRepr + Copy`
//!
//! Two bounds coexist **on purpose** (both traits are sealed in `unpopped-vocab`):
//! `Element: KernelDtype + sealed` adds the α/β `Scalar` projection that arithmetic
//! plans need, while `KernelDtype` is `DeviceRepr + Copy + 'static` plus `KIND`. Carriers
//! ([`TensorRef`] / [`TensorMut`] / [`MatrixRef`] / [`MatrixMut`] / [`VectorRef`]) and
//! layout/pack plans bound on `DeviceRepr + Copy + 'static` so sub-byte and packed
//! types can ride them; compute plans bound on `T: Element`. Unifying them would be an
//! `unpopped-vocab` change, so `Element` / `KernelDtype` are an Unpopped-owned
//! stability dependency of every `Plan<T>`. See `ARCHITECTURE.md`, "Trait layering".
//!
//! The trait `Element` was previously named `CutlassElement`;
//! `baracuda-cutlass` keeps the old name available as a re-export for
//! back-compat. The semantics are unchanged.

#![deny(missing_docs)]

pub mod matrix;
pub mod operand_desc_ext;
pub mod plan;
pub mod tensor;

// The classifier vocabulary now lives in `unpopped-vocab`. Re-export it
// wholesale — flat items and module paths both — so this crate's public surface
// is unchanged. The local `plan` module (device-coupled `Workspace`) shadows the
// glob-imported vocabulary `plan` module and re-exports its descriptors.
pub use unpopped_vocab::*;

pub use matrix::{MatrixMut, MatrixRef, VectorRef};
pub use operand_desc_ext::OperandDescExt;
pub use plan::Workspace;
pub use tensor::{TensorMut, TensorRef, contiguous_stride, strides_equal};
