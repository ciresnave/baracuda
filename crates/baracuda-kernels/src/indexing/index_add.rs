//! `index_add` plan — Category L (Phase 39).
//!
//! `dst[idx[i], ...] += src[i, ...]` along the `add_dim` axis. `idx` is
//! a 1-D index tensor of length `src.shape[add_dim]`. PyTorch
//! `torch.Tensor.index_add_`.
//!
//! Algorithmically identical to [`crate::IndexSelectBackwardPlan`] (a
//! 1-D-idx atomic-Σ scatter along a single axis) but exposed under a
//! non-autograd-flavored name + with f16 / bf16 dtype fanout that the
//! autograd plan deliberately stops short of.
//!
//! **Dtype coverage (Phase 39 Tier 1)**: `{f32, f64, f16, bf16}` × index
//! `{i32, i64}` = 8 FFI symbols. f16 / bf16 use `atomicCAS`-via-
//! `baracuda::atomic::add<T>` (Phase 11.3 helper); f32 / f64 use native
//! `atomicAdd`. Per-thread arithmetic is deterministic; accumulation
//! order is not.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexElement, IndexElementKind, IndexingKind,
    KernelSku, MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::gather::map_status;

/// Descriptor for an `index_add` op.
#[derive(Copy, Clone, Debug)]
pub struct IndexAddDescriptor<const N: usize> {
    /// Shape of `src` (the per-row values added into `dst`).
    pub src_shape: [i32; N],
    /// Axis along which `dst` is indexed (must be in `[0, N)`).
    pub add_dim: i32,
    /// Extent of `dst` along `add_dim` (bounds check on `idx` entries).
    pub dst_dim_size: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `index_add` launch.
pub struct IndexAddArgs<'a, T: Element, const N: usize, I: IndexElement = i32> {
    /// Source tensor (values to add into `dst`).
    pub src: TensorRef<'a, T, N>,
    /// Index tensor (1-D). `idx.numel()` must equal `src.shape[add_dim]`.
    pub idx: TensorRef<'a, I, 1>,
    /// Destination. Accumulated into via atomicAdd-Σ — caller pre-zeroes
    /// (for pure index_add semantics) or pre-populates (for
    /// `dst += accumulate(src, idx)` semantics).
    pub dst: TensorMut<'a, T, N>,
}

/// `index_add` plan.
///
/// `dst[idx[i], ...] += src[i, ...]` along `add_dim` via atomicAdd-Σ
/// (duplicate-index safe).
///
/// **When to use**: forward `index_add` (PyTorch
/// `torch.Tensor.index_add_`). For the inverse — copying `dst[idx[j]]`
/// rows out into a contiguous tensor — use
/// [`IndexSelectPlan`](crate::IndexSelectPlan).
///
/// **Dtypes**: `{f32, f64, f16, bf16}`. f16 / bf16 use the CAS-based
/// `baracuda::atomic::add<T>` helper for deterministic per-thread
/// arithmetic.
///
/// **Shape limits**: rank in `[1, 8]`; `add_dim ∈ [0, N)`; idx 1-D
/// with `idx.numel() == src.shape[add_dim]`.
///
/// **Workspace**: none. Caller pre-zeros (or pre-populates) `dst`.
///
/// **Precision guarantee**: **non-deterministic accumulation order**
/// (atomicAdd). Per-thread arithmetic is bit-stable on same hardware.
///
/// **Index policy**: out-of-bounds and negative indices skipped.
pub struct IndexAddPlan<T: Element, const N: usize> {
    desc: IndexAddDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> IndexAddPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &IndexAddDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexAddPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexAddPlan: rank-0 tensors not supported",
            ));
        }
        if desc.add_dim < 0 || desc.add_dim >= N as i32 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexAddPlan: add_dim out of range [0, N)",
            ));
        }
        if desc.dst_dim_size < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexAddPlan: dst_dim_size must be non-negative",
            ));
        }
        for &d in desc.src_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::IndexAddPlan: src_shape dims must be non-negative",
                ));
            }
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexAddPlan: today only `f32`, `f64`, `f16`, `bf16` wired",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            // atomicAdd order is non-deterministic across launches.
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::IndexAdd as u16,
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

    /// Validate `args` against the descriptor.
    pub fn can_implement<I: IndexElement>(&self, args: &IndexAddArgs<'_, T, N, I>) -> Result<()> {
        if args.src.shape != self.desc.src_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexAddPlan: src shape mismatch with descriptor",
            ));
        }
        let expected_idx = self.desc.src_shape[self.desc.add_dim as usize];
        if args.idx.shape[0] != expected_idx {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexAddPlan: idx.shape[0] must equal \
                 src_shape[add_dim]",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::IndexAddPlan: tensor rank > 8 not supported",
            ));
        }
        let src_numel = args.src.numel();
        let idx_numel = args.idx.numel();
        let src_len = args.src.data.len() as i64;
        let idx_len = args.idx.data.len() as i64;
        if src_len < src_numel {
            return Err(Error::BufferTooSmall {
                needed: src_numel as usize,
                got: src_len as usize,
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

    /// Workspace size in bytes. Always zero.
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

    /// Launch the kernel on `stream`. Caller must have zeroed (or
    /// pre-populated) `dst` before this call. `workspace` ignored.
    pub fn run<I: IndexElement>(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: IndexAddArgs<'_, T, N, I>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let src_numel = args.src.numel();
        if src_numel == 0 {
            return Ok(());
        }
        let src_ptr = args.src.data.as_raw().0 as *const c_void;
        let idx_ptr = args.idx.data.as_raw().0 as *const c_void;
        let dst_ptr = args.dst.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let src_shape = self.desc.src_shape;
        let stride_src = args.src.stride;
        let stride_dst = args.dst.stride;
        let rank = N as i32;

        let status = match (T::KIND, I::KIND) {
            (ElementKind::F32, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_add_f32_run(
                    src_numel, rank, self.desc.add_dim, self.desc.dst_dim_size,
                    src_shape.as_ptr(), stride_src.as_ptr(), stride_dst.as_ptr(),
                    src_ptr, idx_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_add_f64_run(
                    src_numel, rank, self.desc.add_dim, self.desc.dst_dim_size,
                    src_shape.as_ptr(), stride_src.as_ptr(), stride_dst.as_ptr(),
                    src_ptr, idx_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_add_f16_run(
                    src_numel, rank, self.desc.add_dim, self.desc.dst_dim_size,
                    src_shape.as_ptr(), stride_src.as_ptr(), stride_dst.as_ptr(),
                    src_ptr, idx_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_add_bf16_run(
                    src_numel, rank, self.desc.add_dim, self.desc.dst_dim_size,
                    src_shape.as_ptr(), stride_src.as_ptr(), stride_dst.as_ptr(),
                    src_ptr, idx_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_add_i64idx_f32_run(
                    src_numel, rank, self.desc.add_dim, self.desc.dst_dim_size,
                    src_shape.as_ptr(), stride_src.as_ptr(), stride_dst.as_ptr(),
                    src_ptr, idx_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_add_i64idx_f64_run(
                    src_numel, rank, self.desc.add_dim, self.desc.dst_dim_size,
                    src_shape.as_ptr(), stride_src.as_ptr(), stride_dst.as_ptr(),
                    src_ptr, idx_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_add_i64idx_f16_run(
                    src_numel, rank, self.desc.add_dim, self.desc.dst_dim_size,
                    src_shape.as_ptr(), stride_src.as_ptr(), stride_dst.as_ptr(),
                    src_ptr, idx_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_add_i64idx_bf16_run(
                    src_numel, rank, self.desc.add_dim, self.desc.dst_dim_size,
                    src_shape.as_ptr(), stride_src.as_ptr(), stride_dst.as_ptr(),
                    src_ptr, idx_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::IndexAddPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
