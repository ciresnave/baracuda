//! Borrowed views of device-resident matrices and vectors.
//!
//! Pure data — no hidden device allocations, no driver state. Plans
//! cache *selection metadata* on top of these descriptors but never own
//! device memory.
//!
//! The type parameter `T` is bounded by [`DeviceRepr`] (not by the
//! [`crate::Element`] / [`crate::IntElement`] traits) so the same view
//! structs can be re-used across float kernels, integer kernels, and
//! arbitrary scalar-typed bias / aux buffers. Semantic enforcement
//! (which element types a given plan accepts) happens at the plan layer
//! via the appropriate trait bound.

use baracuda_driver::{DeviceSlice, DeviceSliceMut};
use baracuda_types::DeviceRepr;

/// Read-only view of a device-resident matrix.
///
/// `ld` is the leading dimension in **elements** (not bytes), measured
/// along the major axis dictated by the layout: row-stride for row-major
/// matrices, column-stride for column-major matrices.
#[derive(Debug)]
pub struct MatrixRef<'a, T: DeviceRepr + Copy + 'static> {
    /// Device-resident element storage.
    pub data: DeviceSlice<'a, T>,
    /// Number of rows.
    pub rows: i32,
    /// Number of columns.
    pub cols: i32,
    /// Leading dimension in elements.
    pub ld: i64,
}

/// Mutable view of a device-resident matrix (used for the output `D`).
///
/// See [`MatrixRef`] for the rationale behind the relaxed `T` bound.
#[derive(Debug)]
pub struct MatrixMut<'a, T: DeviceRepr + Copy + 'static> {
    /// Device-resident element storage.
    pub data: DeviceSliceMut<'a, T>,
    /// Number of rows.
    pub rows: i32,
    /// Number of columns.
    pub cols: i32,
    /// Leading dimension in elements.
    pub ld: i64,
}

/// Read-only view of a device-resident vector.
///
/// See [`MatrixRef`] for the rationale behind the relaxed `T` bound.
#[derive(Debug)]
pub struct VectorRef<'a, T: DeviceRepr + Copy + 'static> {
    /// Device-resident element storage.
    pub data: DeviceSlice<'a, T>,
    /// Number of elements.
    pub len: i32,
    /// Stride in elements.
    pub stride: i64,
}
