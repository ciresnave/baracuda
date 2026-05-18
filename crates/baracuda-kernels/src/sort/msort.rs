//! `msort` (stable sort) plan + BW.
//!
//! Same as [`crate::sort::SortPlan`] but with stability guarantee —
//! equal keys preserve input order via tie-break on original index.
//! PyTorch `torch.msort`.
//!
//! Trailblazer dtype coverage: FW `f32, f64, i32, i64`; BW `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::sort::{build_sku, validate_sort_args_2, validate_sort_desc};

/// Descriptor for an `msort` op.
#[derive(Copy, Clone, Debug)]
pub struct MsortDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each row. Trailblazer cap: `≤ 1024`.
    pub row_len: i32,
    /// `true` = sort largest-first.
    pub descending: bool,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `msort` launch.
pub struct MsortArgs<'a, T: Element> {
    /// Input `[batch, row_len]`.
    pub input: TensorRef<'a, T, 2>,
    /// Sorted values `[batch, row_len]`.
    pub values: TensorMut<'a, T, 2>,
    /// Sorted indices `[batch, row_len]` — saved for BW.
    pub indices: TensorMut<'a, i32, 2>,
}

/// `msort` plan.
pub struct MsortPlan<T: Element> {
    desc: MsortDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> MsortPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &MsortDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_sort_desc(desc.batch, desc.row_len, desc.element, T::KIND, "MsortPlan")?;
        let sku = build_sku::<T>(SortKind::Msort);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &MsortArgs<'_, T>) -> Result<()> {
        validate_sort_args_2(
            self.desc.batch,
            self.desc.row_len,
            args.input.shape,
            args.values.shape,
            args.indices.shape,
            "MsortPlan",
        )
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: MsortArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 || self.desc.row_len == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let vals_ptr = args.values.data.as_raw().0 as *mut c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let desc_flag = if self.desc.descending { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_msort_f32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_msort_f64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_msort_i32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_msort_i64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MsortPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}

// ---- BW ----

/// Descriptor for an `msort_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct MsortBackwardDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each row.
    pub row_len: i32,
    /// Grad element type.
    pub element: ElementKind,
}

/// Args bundle for an `msort_backward` launch.
pub struct MsortBackwardArgs<'a, T: Element> {
    /// Upstream grad of sorted-values output `[batch, row_len]`.
    pub dy: TensorRef<'a, T, 2>,
    /// Saved indices from FW `[batch, row_len]`.
    pub indices: TensorRef<'a, i32, 2>,
    /// Grad of the input `[batch, row_len]`.
    pub dx: TensorMut<'a, T, 2>,
}

/// `msort_backward` plan.
pub struct MsortBackwardPlan<T: Element> {
    desc: MsortBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> MsortBackwardPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &MsortBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_sort_desc(
            desc.batch,
            desc.row_len,
            desc.element,
            T::KIND,
            "MsortBackwardPlan",
        )?;
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::MsortBackwardPlan: today only f32 / f64 grads supported",
            ));
        }
        let sku = build_sku::<T>(SortKind::MsortBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &MsortBackwardArgs<'_, T>) -> Result<()> {
        let expected = [self.desc.batch, self.desc.row_len];
        if args.dy.shape != expected
            || args.indices.shape != expected
            || args.dx.shape != expected
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MsortBackwardPlan: tensor shapes != [batch, row_len]",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: MsortBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 || self.desc.row_len == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_msort_backward_f32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    dy_ptr,
                    idx_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_msort_backward_f64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    dy_ptr,
                    idx_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MsortBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
