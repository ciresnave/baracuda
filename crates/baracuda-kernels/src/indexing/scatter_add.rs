//! `scatter_add` plan — Category L.
//!
//! `out[..., index[..., j, ...], ...] += updates[..., j, ...]` along
//! the `scatter_dim` axis (atomicAdd, dup-safe). PyTorch
//! `torch.Tensor.scatter_add_`.
//!
//! Trailblazer dtype coverage: `f32, f64` only (uses `atomicAdd`).
//! Integer scatter_add is supported by CUDA but kept out-of-scope.
//!
//! The backward of `scatter_add` is `gather` — callers wire that via
//! [`crate::indexing::GatherPlan`], so no separate BW plan exists here.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexingKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::gather::map_status;

/// Descriptor for a `scatter_add` op.
#[derive(Copy, Clone, Debug)]
pub struct ScatterAddDescriptor<const N: usize> {
    /// Shape of `updates` / `index`.
    pub upd_shape: [i32; N],
    /// Scatter axis (must be in `[0, N)`).
    pub scatter_dim: i32,
    /// Extent of `out` along `scatter_dim` (in-bounds check on indices).
    pub out_dim_size: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `scatter_add` launch.
pub struct ScatterAddArgs<'a, T: Element, const N: usize> {
    /// Update values.
    pub updates: TensorRef<'a, T, N>,
    /// Index tensor (i32). Same shape as `updates`.
    pub index: TensorRef<'a, i32, N>,
    /// Output. Accumulated into (atomicAdd) — caller must pre-zero or
    /// pre-populate.
    pub out: TensorMut<'a, T, N>,
}

/// `scatter_add` plan.
pub struct ScatterAddPlan<T: Element, const N: usize> {
    desc: ScatterAddDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> ScatterAddPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &ScatterAddDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScatterAddPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterAddPlan: rank-0 tensors not supported",
            ));
        }
        if desc.scatter_dim < 0 || desc.scatter_dim >= N as i32 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterAddPlan: scatter_dim out of range [0, N)",
            ));
        }
        if desc.out_dim_size < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterAddPlan: out_dim_size must be non-negative",
            ));
        }
        for &d in desc.upd_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ScatterAddPlan: upd_shape dims must be non-negative",
                ));
            }
        }

        let supported = matches!(T::KIND, ElementKind::F32 | ElementKind::F64);
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScatterAddPlan: today only `f32`, `f64` wired",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            // atomicAdd → non-deterministic order.
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::ScatterAdd as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I32),
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &ScatterAddArgs<'_, T, N>) -> Result<()> {
        if args.updates.shape != self.desc.upd_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterAddPlan: updates shape mismatch with descriptor",
            ));
        }
        if args.index.shape != self.desc.upd_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterAddPlan: index shape must equal updates shape",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScatterAddPlan: tensor rank > 8 not supported",
            ));
        }
        let upd_numel = args.updates.numel();
        let upd_len = args.updates.data.len() as i64;
        let idx_len = args.index.data.len() as i64;
        if upd_len < upd_numel {
            return Err(Error::BufferTooSmall {
                needed: upd_numel as usize,
                got: upd_len as usize,
            });
        }
        if idx_len < upd_numel {
            return Err(Error::BufferTooSmall {
                needed: upd_numel as usize,
                got: idx_len as usize,
            });
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
        args: ScatterAddArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let upd_numel = args.updates.numel();
        if upd_numel == 0 {
            return Ok(());
        }
        let upd_ptr = args.updates.data.as_raw().0 as *const c_void;
        let idx_ptr = args.index.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let upd_shape = self.desc.upd_shape;
        let stride_upd = args.updates.stride;
        let stride_index = args.index.stride;
        let stride_out = args.out.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_add_f32_run(
                    upd_numel,
                    rank,
                    self.desc.scatter_dim,
                    self.desc.out_dim_size,
                    upd_shape.as_ptr(),
                    stride_upd.as_ptr(),
                    stride_index.as_ptr(),
                    stride_out.as_ptr(),
                    upd_ptr,
                    idx_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_add_f64_run(
                    upd_numel,
                    rank,
                    self.desc.scatter_dim,
                    self.desc.out_dim_size,
                    upd_shape.as_ptr(),
                    stride_upd.as_ptr(),
                    stride_index.as_ptr(),
                    stride_out.as_ptr(),
                    upd_ptr,
                    idx_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ScatterAddPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
