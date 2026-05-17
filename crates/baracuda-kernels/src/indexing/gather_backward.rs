//! `gather_backward` plan — Category L BW.
//!
//! Adjoint of [`crate::indexing::GatherPlan`]:
//! `dsrc[..., index[..., j, ...], ...] += dout[..., j, ...]` along the
//! gather dim. Uses `atomicAdd`, so dtype coverage is restricted to
//! native-FP-atomic types: `f32, f64` (trailblazer). Integer atomic-add
//! is supported by CUDA but kept out-of-scope for the trailblazer.
//!
//! The caller is responsible for zeroing `dsrc` before launch (or
//! accumulating into a pre-existing gradient buffer).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexingKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::gather::map_status;

/// Descriptor for a `gather_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct GatherBackwardDescriptor<const N: usize> {
    /// Shape of `dout` / `index` (== shape of FW output).
    pub out_shape: [i32; N],
    /// Gather axis (must match the FW pass).
    pub gather_dim: i32,
    /// Extent of `dsrc` along `gather_dim` (== `src_dim_size` from FW).
    pub src_dim_size: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `gather_backward` launch.
pub struct GatherBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dout: TensorRef<'a, T, N>,
    /// Index tensor from the FW pass.
    pub index: TensorRef<'a, i32, N>,
    /// Gradient w.r.t. `src`. Caller MUST zero this before launch
    /// (or pre-populate to accumulate).
    pub dsrc: TensorMut<'a, T, N>,
}

/// `gather_backward` plan.
pub struct GatherBackwardPlan<T: Element, const N: usize> {
    desc: GatherBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> GatherBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &GatherBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatherBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherBackwardPlan: rank-0 tensors not supported",
            ));
        }
        if desc.gather_dim < 0 || desc.gather_dim >= N as i32 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherBackwardPlan: gather_dim out of range [0, N)",
            ));
        }
        if desc.src_dim_size < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherBackwardPlan: src_dim_size must be non-negative",
            ));
        }

        let supported = matches!(T::KIND, ElementKind::F32 | ElementKind::F64);
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatherBackwardPlan: today only `f32`, `f64` wired \
                 (BW uses atomicAdd — restricted to native-FP atomic types)",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            // atomicAdd makes the order non-deterministic.
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::GatherBackward as u16,
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
    pub fn can_implement(&self, args: &GatherBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dout.shape != self.desc.out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherBackwardPlan: dout shape mismatch with descriptor",
            ));
        }
        if args.index.shape != self.desc.out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherBackwardPlan: index shape must equal dout shape",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatherBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let out_numel = args.dout.numel();
        let idx_numel = args.index.numel();
        let dout_len = args.dout.data.len() as i64;
        let idx_len = args.index.data.len() as i64;
        if dout_len < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: dout_len as usize,
            });
        }
        if idx_len < idx_numel {
            return Err(Error::BufferTooSmall {
                needed: idx_numel as usize,
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
        args: GatherBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let out_numel = args.dout.numel();
        if out_numel == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let idx_ptr = args.index.data.as_raw().0 as *const c_void;
        let dsrc_ptr = args.dsrc.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let out_shape = self.desc.out_shape;
        let stride_dout = args.dout.stride;
        let stride_index = args.index.stride;
        let stride_dsrc = args.dsrc.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gather_backward_f32_run(
                    out_numel,
                    rank,
                    self.desc.gather_dim,
                    self.desc.src_dim_size,
                    out_shape.as_ptr(),
                    stride_dout.as_ptr(),
                    stride_index.as_ptr(),
                    stride_dsrc.as_ptr(),
                    dout_ptr,
                    idx_ptr,
                    dsrc_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gather_backward_f64_run(
                    out_numel,
                    rank,
                    self.desc.gather_dim,
                    self.desc.src_dim_size,
                    out_shape.as_ptr(),
                    stride_dout.as_ptr(),
                    stride_index.as_ptr(),
                    stride_dsrc.as_ptr(),
                    dout_ptr,
                    idx_ptr,
                    dsrc_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GatherBackwardPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
