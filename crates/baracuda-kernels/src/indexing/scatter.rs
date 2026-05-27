//! `scatter` (pure assign) plan — Category L (Phase 39).
//!
//! `out[..., index[..., j, ...], ...] = updates[..., j, ...]` along the
//! `scatter_dim` axis. **No accumulation** — if multiple updates target
//! the same output cell, the **last writer wins** (race; the result is
//! non-deterministic but the per-cell value is always one of the
//! contributing writes — never a partial / torn store, since one
//! element fits in a single 32/64-bit write).
//!
//! PyTorch `torch.Tensor.scatter_` (the in-place pure-assign variant).
//! Distinct from [`crate::ScatterAddPlan`] which atomically Σ-accumulates.
//!
//! **Dtype coverage (Phase 39 Tier 1)**: `{f32, f64, f16, bf16}` × index
//! `{i32, i64}` = 8 FFI symbols. The kernel does no arithmetic, only
//! stores, so all four dtypes ship in the trailblazer.
//!
//! **Tests should use disjoint targets** to keep results deterministic.
//! Duplicate-target writes are an advisory feature of the op semantics,
//! not something callers should rely on for any specific outcome.

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

/// Descriptor for a `scatter` (pure assign) op.
///
/// Identifies the shape of `updates` (== `index` shape), the axis, and
/// the extent of `out` along that axis. `T::KIND` must equal `element`.
#[derive(Copy, Clone, Debug)]
pub struct ScatterDescriptor<const N: usize> {
    /// Shape of `updates` / `index`.
    pub upd_shape: [i32; N],
    /// Scatter axis (must be in `[0, N)`).
    pub scatter_dim: i32,
    /// Extent of `out` along `scatter_dim` (in-bounds check on indices).
    pub out_dim_size: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `scatter` (pure assign) launch.
pub struct ScatterArgs<'a, T: Element, const N: usize, I: IndexElement = i32> {
    /// Update values.
    pub updates: TensorRef<'a, T, N>,
    /// Index tensor. Same shape as `updates`. `i32` (legacy) or `i64`
    /// (PyTorch default).
    pub index: TensorRef<'a, I, N>,
    /// Output. **Overwritten** (not accumulated). Caller pre-populates
    /// any cells that should retain their value when no index targets
    /// them (the kernel only touches cells named by `index`).
    pub out: TensorMut<'a, T, N>,
}

/// `scatter` (pure assign) plan.
///
/// `out[..., index[..., j, ...], ...] = updates[..., j, ...]` along
/// `scatter_dim` — **no accumulation**. Last writer wins on
/// duplicate-target races (caller-aware non-determinism).
///
/// **When to use**: forward `scatter` (PyTorch
/// `torch.Tensor.scatter_`). For Σ-accumulation use
/// [`ScatterAddPlan`](crate::ScatterAddPlan).
///
/// **Dtypes**: `{f32, f64, f16, bf16}`. Pure store, no arithmetic.
///
/// **Shape limits**: rank in `[1, 8]`; `scatter_dim ∈ [0, N)`;
/// `out_dim_size ≥ 0`. `updates` and `index` must share shape.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: **non-deterministic** on duplicate-target
/// indices (race condition). For disjoint-index workloads the output
/// is deterministic and bit-exact (pure copy, no arithmetic).
///
/// **Index policy**: out-of-bounds and negative indices are skipped
/// (no PyTorch-style wraparound).
pub struct ScatterPlan<T: Element, const N: usize> {
    desc: ScatterDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> ScatterPlan<T, N> {
    /// Pick a kernel for `desc`. Validates element-type alignment,
    /// rank, axis, non-negative extents, and dtype in
    /// `{f32, f64, f16, bf16}`.
    pub fn select(
        _stream: &Stream,
        desc: &ScatterDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScatterPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterPlan: rank-0 tensors not supported",
            ));
        }
        if desc.scatter_dim < 0 || desc.scatter_dim >= N as i32 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterPlan: scatter_dim out of range [0, N)",
            ));
        }
        if desc.out_dim_size < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterPlan: out_dim_size must be non-negative",
            ));
        }
        for &d in desc.upd_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ScatterPlan: upd_shape dims must be non-negative",
                ));
            }
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScatterPlan: today only `f32`, `f64`, `f16`, `bf16` wired",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: T::KIND,
            // Pure store: bit-stable per-cell when targets are disjoint,
            // but the op semantics permit duplicate targets which race.
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::Scatter as u16,
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
    pub fn can_implement<I: IndexElement>(&self, args: &ScatterArgs<'_, T, N, I>) -> Result<()> {
        if args.updates.shape != self.desc.upd_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterPlan: updates shape mismatch with descriptor",
            ));
        }
        if args.index.shape != self.desc.upd_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScatterPlan: index shape must equal updates shape",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScatterPlan: tensor rank > 8 not supported",
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

    /// Launch the kernel on `stream`. Caller pre-populates `out`; the
    /// kernel only writes cells named by `index`. `workspace` ignored.
    pub fn run<I: IndexElement>(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: ScatterArgs<'_, T, N, I>,
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

        let status = match (T::KIND, I::KIND) {
            (ElementKind::F32, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_f32_run(
                    upd_numel, rank, self.desc.scatter_dim, self.desc.out_dim_size,
                    upd_shape.as_ptr(), stride_upd.as_ptr(), stride_index.as_ptr(),
                    stride_out.as_ptr(), upd_ptr, idx_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_f64_run(
                    upd_numel, rank, self.desc.scatter_dim, self.desc.out_dim_size,
                    upd_shape.as_ptr(), stride_upd.as_ptr(), stride_index.as_ptr(),
                    stride_out.as_ptr(), upd_ptr, idx_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_f16_run(
                    upd_numel, rank, self.desc.scatter_dim, self.desc.out_dim_size,
                    upd_shape.as_ptr(), stride_upd.as_ptr(), stride_index.as_ptr(),
                    stride_out.as_ptr(), upd_ptr, idx_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_bf16_run(
                    upd_numel, rank, self.desc.scatter_dim, self.desc.out_dim_size,
                    upd_shape.as_ptr(), stride_upd.as_ptr(), stride_index.as_ptr(),
                    stride_out.as_ptr(), upd_ptr, idx_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_i64idx_f32_run(
                    upd_numel, rank, self.desc.scatter_dim, self.desc.out_dim_size,
                    upd_shape.as_ptr(), stride_upd.as_ptr(), stride_index.as_ptr(),
                    stride_out.as_ptr(), upd_ptr, idx_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_i64idx_f64_run(
                    upd_numel, rank, self.desc.scatter_dim, self.desc.out_dim_size,
                    upd_shape.as_ptr(), stride_upd.as_ptr(), stride_index.as_ptr(),
                    stride_out.as_ptr(), upd_ptr, idx_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_i64idx_f16_run(
                    upd_numel, rank, self.desc.scatter_dim, self.desc.out_dim_size,
                    upd_shape.as_ptr(), stride_upd.as_ptr(), stride_index.as_ptr(),
                    stride_out.as_ptr(), upd_ptr, idx_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_scatter_i64idx_bf16_run(
                    upd_numel, rank, self.desc.scatter_dim, self.desc.out_dim_size,
                    upd_shape.as_ptr(), stride_upd.as_ptr(), stride_index.as_ptr(),
                    stride_out.as_ptr(), upd_ptr, idx_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ScatterPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
