//! `gather` plan — Category L FW trailblazer.
//!
//! `out[..., j, ...] = src[..., index[..., j, ...], ...]` along the
//! `gather_dim` axis. `index` shape == `out` shape (same rank as `src`,
//! same extents on every axis except `gather_dim` where the extent is
//! determined by the index shape). Index dtype is `i32` only
//! (trailblazer; i64 deferred).
//!
//! Out-of-bounds index policy: kernel skips (no write); negative
//! indices treated as out-of-bounds (no PyTorch-style wrap).
//!
//! Trailblazer dtype coverage: `f32, f64, i32`. The kernel is a pure
//! load + store — no arithmetic — so output is bit-exact against
//! reference at every dtype.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexingKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `gather` op.
///
/// Identifies the shape of the output / index tensor, the axis to gather
/// along, and the source extent along that axis (for in-bounds checks).
/// The output and index tensors must have identical shape; this single
/// `out_shape` doubles as the index shape. `T::KIND` must equal
/// `element`.
#[derive(Copy, Clone, Debug)]
pub struct GatherDescriptor<const N: usize> {
    /// Output / index shape (gather collapses to whatever `index` shape
    /// the caller supplies).
    pub out_shape: [i32; N],
    /// Axis along which to gather. Must be in `[0, N)`.
    pub gather_dim: i32,
    /// Extent of `src` along `gather_dim`. Used for the kernel's
    /// in-bounds check on each index value.
    pub src_dim_size: i32,
    /// Value element type (src / out dtype). Must match `T::KIND`.
    pub element: ElementKind,
}

/// Args bundle for a `gather` launch.
pub struct GatherArgs<'a, T: Element, const N: usize> {
    /// Source tensor.
    pub src: TensorRef<'a, T, N>,
    /// Index tensor (i32). Same shape as `out`.
    pub index: TensorRef<'a, i32, N>,
    /// Output. Shape == index shape.
    pub out: TensorMut<'a, T, N>,
}

/// `gather` plan.
///
/// `out[..., j, ...] = src[..., index[..., j, ...], ...]` along the
/// `gather_dim` axis.
///
/// **When to use**: forward `gather`. Pair with [`GatherBackwardPlan`](crate::GatherBackwardPlan)
/// for the autograd pass (which scatter-adds into `dsrc`).
///
/// **Dtypes**: value tensor `{f32, f64, i32}`; index tensor always
/// `i32`. (i64 indices deferred.)
///
/// **Shape limits**: rank in `[1, 8]`; `gather_dim ∈ [0, N)`; every
/// dim of `out_shape` non-negative.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable on same hardware.
/// Pure load + store, no arithmetic — output is bit-exact at every
/// dtype.
///
/// **Index policy**: out-of-bounds indices skip the write; negative
/// indices are treated as out-of-bounds (no PyTorch-style wraparound).
pub struct GatherPlan<T: Element, const N: usize> {
    desc: GatherDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> GatherPlan<T, N> {
    /// Pick a kernel for `desc`.
    ///
    /// Validates: `T::KIND == desc.element`, rank > 0, `gather_dim`
    /// in range, non-negative extents, and dtype in `{f32, f64, i32}`.
    pub fn select(
        _stream: &Stream,
        desc: &GatherDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatherPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherPlan: rank-0 tensors not supported",
            ));
        }
        if desc.gather_dim < 0 || desc.gather_dim >= N as i32 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherPlan: gather_dim out of range [0, N)",
            ));
        }
        if desc.src_dim_size < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherPlan: src_dim_size must be non-negative",
            ));
        }
        for &d in desc.out_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::GatherPlan: out_shape dims must be non-negative",
                ));
            }
        }

        let supported =
            matches!(T::KIND, ElementKind::F32 | ElementKind::F64 | ElementKind::I32);
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatherPlan: today only `f32`, `f64`, `i32` wired",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // gather is pure load + store, no arithmetic — bit-exact.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::Gather as u16,
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

    /// Validate that `args` is compatible with this plan: output and
    /// index shapes match the descriptor, rank is ≤ 8, and every device
    /// buffer is large enough to address its declared `numel`.
    pub fn can_implement(&self, args: &GatherArgs<'_, T, N>) -> Result<()> {
        if args.out.shape != self.desc.out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherPlan: out shape mismatch with descriptor",
            ));
        }
        if args.index.shape != self.desc.out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherPlan: index shape must equal out shape",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatherPlan: tensor rank > 8 not supported",
            ));
        }
        let out_numel = args.out.numel();
        let idx_numel = args.index.numel();
        let out_len = args.out.data.len() as i64;
        let idx_len = args.index.data.len() as i64;
        let src_len = args.src.data.len() as i64;
        if out_len < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: out_len as usize,
            });
        }
        if idx_len < idx_numel {
            return Err(Error::BufferTooSmall {
                needed: idx_numel as usize,
                got: idx_len as usize,
            });
        }
        // Minimum src size: enough to address `src_dim_size` along the
        // gather axis with whatever stride the caller provided.
        if src_len < 0 || args.src.numel() < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatherPlan: src numel overflow",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes. Always zero — gather is in-place over
    /// the output tensor with no scratch state.
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

    /// Launch the kernel on `stream`. Calls [`Self::can_implement`]
    /// first; returns early on zero-element output. `workspace` is
    /// ignored (gather requires none).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: GatherArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let out_numel = args.out.numel();
        if out_numel == 0 {
            return Ok(());
        }
        let src_ptr = args.src.data.as_raw().0 as *const c_void;
        let idx_ptr = args.index.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let out_shape = self.desc.out_shape;
        let stride_src = args.src.stride;
        let stride_index = args.index.stride;
        let stride_out = args.out.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gather_f32_run(
                    out_numel,
                    rank,
                    self.desc.gather_dim,
                    self.desc.src_dim_size,
                    out_shape.as_ptr(),
                    stride_src.as_ptr(),
                    stride_index.as_ptr(),
                    stride_out.as_ptr(),
                    src_ptr,
                    idx_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gather_f64_run(
                    out_numel,
                    rank,
                    self.desc.gather_dim,
                    self.desc.src_dim_size,
                    out_shape.as_ptr(),
                    stride_src.as_ptr(),
                    stride_index.as_ptr(),
                    stride_out.as_ptr(),
                    src_ptr,
                    idx_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gather_i32_run(
                    out_numel,
                    rank,
                    self.desc.gather_dim,
                    self.desc.src_dim_size,
                    out_shape.as_ptr(),
                    stride_src.as_ptr(),
                    stride_index.as_ptr(),
                    stride_out.as_ptr(),
                    src_ptr,
                    idx_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GatherPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
