//! `topk_backward` plan — scatter k-wide `dy` into row_len-wide `dx`
//! via the saved indices. Launcher zeros `dx` first.
//!
//! Trailblazer dtype coverage: `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::sort::build_sku;
use super::{SORT_MAX_ROW, TOPK_MAX_K};

/// Descriptor for a `topk_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct TopkBackwardDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each input row.
    pub row_len: i32,
    /// `k` used in the corresponding FW.
    pub k: i32,
    /// Grad element type.
    pub element: ElementKind,
}

/// Args bundle for a `topk_backward` launch.
pub struct TopkBackwardArgs<'a, T: Element> {
    /// Upstream grad of top-k values `[batch, k]`.
    pub dy: TensorRef<'a, T, 2>,
    /// Saved indices from FW `[batch, k]`.
    pub indices: TensorRef<'a, i32, 2>,
    /// Grad of the input `[batch, row_len]`.
    pub dx: TensorMut<'a, T, 2>,
}

/// `topk_backward` plan.
pub struct TopkBackwardPlan<T: Element> {
    desc: TopkBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> TopkBackwardPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &TopkBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TopkBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if desc.batch < 0 || desc.row_len < 0 || desc.k < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkBackwardPlan: batch / row_len / k must be non-negative",
            ));
        }
        if desc.row_len > SORT_MAX_ROW {
            return Err(Error::Unsupported(
                "baracuda-kernels::TopkBackwardPlan: row_len > 1024 not supported",
            ));
        }
        if desc.k > TOPK_MAX_K {
            return Err(Error::Unsupported(
                "baracuda-kernels::TopkBackwardPlan: k > 64 not supported",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::TopkBackwardPlan: today only f32 / f64 grads supported",
            ));
        }
        let sku = build_sku::<T>(SortKind::TopkBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &TopkBackwardArgs<'_, T>) -> Result<()> {
        let dy_shape = [self.desc.batch, self.desc.k];
        let dx_shape = [self.desc.batch, self.desc.row_len];
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkBackwardPlan: dy shape != [batch, k]",
            ));
        }
        if args.indices.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkBackwardPlan: indices shape != [batch, k]",
            ));
        }
        if args.dx.shape != dx_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkBackwardPlan: dx shape != [batch, row_len]",
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
        args: TopkBackwardArgs<'_, T>,
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
                baracuda_kernels_sys::baracuda_kernels_topk_backward_f32_run(
                    self.desc.batch,
                    self.desc.k,
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
                baracuda_kernels_sys::baracuda_kernels_topk_backward_f64_run(
                    self.desc.batch,
                    self.desc.k,
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
                    "baracuda-kernels::TopkBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
