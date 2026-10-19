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

use crate::tensor::{ViewError, elements_per_slot};

/// Smallest slice that can hold a `rows x cols` matrix in ANY layout: the
/// matrix has `rows * cols` distinct logical elements, packed `elements_per_slot`
/// to a slot. Plans check the exact footprint of the layout they use
/// (`(rows-1)*ld + cols` row-major, `(cols-1)*ld + rows` column-major, bytes for
/// packed int4 / bit operands); this lower bound is what a constructor can
/// promise without knowing the layout.
fn matrix_floor<T: 'static>(rows: i32, cols: i32, ld: i64, got: usize) -> Result<(), ViewError> {
    if rows < 0 || cols < 0 || ld < 0 {
        return Err(ViewError::NotAddressable);
    }
    let elems = rows as u128 * cols as u128;
    let needed = usize::try_from(elems.div_ceil(elements_per_slot::<T>() as u128))
        .map_err(|_| ViewError::NotAddressable)?;
    if needed > got {
        return Err(ViewError::TooSmall { needed, got });
    }
    Ok(())
}

/// Read-only view of a device-resident matrix.
///
/// `ld` is the leading dimension in **elements** (not bytes), measured
/// along the major axis dictated by the layout: row-stride for row-major
/// matrices, column-stride for column-major matrices.
///
/// The fields are private: build one with [`MatrixRef::new`] and read it back
/// with `data()` / `rows()` / `cols()` / `ld()`. `new` rejects negative
/// dimensions and a slice too short to hold `rows * cols` elements; the plan
/// that consumes the matrix checks the exact footprint of its layout.
#[derive(Debug)]
pub struct MatrixRef<'a, T: DeviceRepr + Copy + 'static> {
    data: DeviceSlice<'a, T>,
    rows: i32,
    cols: i32,
    ld: i64,
}

impl<'a, T: DeviceRepr + Copy + 'static> MatrixRef<'a, T> {
    /// Build a view, checking the dimensions and the slice length (see the
    /// type docs for what is, and is not, checked here).
    pub fn new(
        data: DeviceSlice<'a, T>,
        rows: i32,
        cols: i32,
        ld: i64,
    ) -> Result<Self, ViewError> {
        matrix_floor::<T>(rows, cols, ld, data.len())?;
        Ok(Self {
            data,
            rows,
            cols,
            ld,
        })
    }

    /// Build a view WITHOUT any check.
    ///
    /// # Safety
    ///
    /// The slice must hold the footprint the consuming plan will read for this
    /// `rows` / `cols` / `ld` in its layout; otherwise a launch reads outside it.
    #[inline]
    pub unsafe fn new_unchecked(data: DeviceSlice<'a, T>, rows: i32, cols: i32, ld: i64) -> Self {
        Self {
            data,
            rows,
            cols,
            ld,
        }
    }

    /// Device-resident element storage.
    #[inline]
    pub fn data(&self) -> DeviceSlice<'a, T> {
        self.data
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> i32 {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> i32 {
        self.cols
    }

    /// Leading dimension in elements.
    #[inline]
    pub fn ld(&self) -> i64 {
        self.ld
    }
}

/// Mutable view of a device-resident matrix (used for the output `D`).
///
/// See [`MatrixRef`] for the rationale behind the relaxed `T` bound and for
/// the private fields.
#[derive(Debug)]
pub struct MatrixMut<'a, T: DeviceRepr + Copy + 'static> {
    data: DeviceSliceMut<'a, T>,
    rows: i32,
    cols: i32,
    ld: i64,
}

impl<'a, T: DeviceRepr + Copy + 'static> MatrixMut<'a, T> {
    /// Build a view, checking the dimensions and the slice length (see
    /// [`MatrixRef`]).
    pub fn new(
        data: DeviceSliceMut<'a, T>,
        rows: i32,
        cols: i32,
        ld: i64,
    ) -> Result<Self, ViewError> {
        matrix_floor::<T>(rows, cols, ld, data.len())?;
        Ok(Self {
            data,
            rows,
            cols,
            ld,
        })
    }

    /// Build a view WITHOUT any check.
    ///
    /// # Safety
    ///
    /// As for [`MatrixRef::new_unchecked`].
    #[inline]
    pub unsafe fn new_unchecked(
        data: DeviceSliceMut<'a, T>,
        rows: i32,
        cols: i32,
        ld: i64,
    ) -> Self {
        Self {
            data,
            rows,
            cols,
            ld,
        }
    }

    /// Device-resident element storage.
    #[inline]
    pub fn data(&self) -> &DeviceSliceMut<'a, T> {
        &self.data
    }

    /// Device-resident element storage, mutably.
    #[inline]
    pub fn data_mut(&mut self) -> &mut DeviceSliceMut<'a, T> {
        &mut self.data
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> i32 {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> i32 {
        self.cols
    }

    /// Leading dimension in elements.
    #[inline]
    pub fn ld(&self) -> i64 {
        self.ld
    }
}

/// Read-only view of a device-resident vector.
///
/// See [`MatrixRef`] for the rationale behind the relaxed `T` bound. The fields
/// are private; [`VectorRef::new`] checks that `len` elements at `stride` fit in
/// the slice.
#[derive(Debug)]
pub struct VectorRef<'a, T: DeviceRepr + Copy + 'static> {
    data: DeviceSlice<'a, T>,
    len: i32,
    stride: i64,
}

impl<'a, T: DeviceRepr + Copy + 'static> VectorRef<'a, T> {
    /// Build a view, checking that `len` elements at `stride` stay inside `data`
    /// (the highest element reached is `(len - 1) * stride`).
    pub fn new(data: DeviceSlice<'a, T>, len: i32, stride: i64) -> Result<Self, ViewError> {
        if len < 0 || stride < 0 {
            return Err(ViewError::NotAddressable);
        }
        let needed = if len == 0 {
            0
        } else {
            usize::try_from((len as i128 - 1) * stride as i128 + 1)
                .map_err(|_| ViewError::NotAddressable)?
                .div_ceil(elements_per_slot::<T>())
        };
        if needed > data.len() {
            return Err(ViewError::TooSmall {
                needed,
                got: data.len(),
            });
        }
        Ok(Self { data, len, stride })
    }

    /// Build a view WITHOUT any check.
    ///
    /// # Safety
    ///
    /// `len` elements at `stride` must lie inside `data`.
    #[inline]
    pub unsafe fn new_unchecked(data: DeviceSlice<'a, T>, len: i32, stride: i64) -> Self {
        Self { data, len, stride }
    }

    /// Device-resident element storage.
    #[inline]
    pub fn data(&self) -> DeviceSlice<'a, T> {
        self.data
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> i32 {
        self.len
    }

    /// `true` iff the vector has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stride in elements.
    #[inline]
    pub fn stride(&self) -> i64 {
        self.stride
    }
}
