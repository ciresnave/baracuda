//! `embedding` BW plan — Category M.
//!
//! Adjoint of [`crate::embedding::EmbeddingPlan`]:
//! `dweight[indices[n], :] += dout[n, :]` along the row axis (atomicAdd).
//! Rows where `indices[n] == padding_idx` (or where `indices[n]` is
//! negative / out-of-range) are skipped — no contribution.
//!
//! Trailblazer dtype coverage: `f32, f64`. atomicAdd is native-FP.
//!
//! Caller MUST zero `dweight` before launch (or pre-populate to
//! accumulate into a running gradient).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, EmbeddingKind, IndexElement, IndexElementKind,
    KernelSku, MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use crate::indexing::gather::map_status;

use super::PADDING_DISABLED;

/// Descriptor for an `embedding_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct EmbeddingBackwardDescriptor {
    /// Vocabulary size — extent of `dweight` along axis 0.
    pub num_embeddings: i32,
    /// Embedding dimension — extent of `dweight` along axis 1.
    pub embedding_dim: i32,
    /// Number of indices — extent of `indices` and of `dout` along axis 0.
    pub num_indices: i32,
    /// Optional padding index. When `Some(p)`, rows where
    /// `indices[n] == p` are skipped (no contribution).
    pub padding_idx: Option<i32>,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `embedding_backward` launch.
///
/// Phase 11.5: `I: IndexElement` generic (`i32` or `i64`).
pub struct EmbeddingBackwardArgs<'a, T: Element, I: IndexElement = i32> {
    /// Upstream gradient `[N, D]`. Row-major contiguous.
    pub dout: TensorRef<'a, T, 2>,
    /// Index tensor `[N]` (same as FW pass).
    pub indices: TensorRef<'a, I, 1>,
    /// Gradient w.r.t. `weight` `[V, D]`. Caller MUST pre-zero this.
    pub dweight: TensorMut<'a, T, 2>,
}

/// `embedding_backward` plan.
///
/// Adjoint of [`crate::EmbeddingPlan`]:
/// `dweight[indices[n], :] += dout[n, :]` via atomicAdd. Rows where
/// `indices[n] == padding_idx` (or negative / out-of-range) are skipped.
///
/// **When to use**: backward for [`EmbeddingPlan`](crate::EmbeddingPlan).
///
/// **Dtypes**: `{f32, f64}` only — atomicAdd is native-FP.
///
/// **Shape limits**: `dweight` is `[V, D]`, `dout` is `[N, D]`,
/// `indices` is `[N]`, all extents non-negative.
///
/// **Workspace**: none. Caller MUST zero `dweight` before launch
/// (or pre-populate to accumulate into a running gradient).
///
/// **Precision guarantee**: **non-deterministic** — atomicAdd
/// ordering varies between launches.
pub struct EmbeddingBackwardPlan<T: Element> {
    desc: EmbeddingBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> EmbeddingBackwardPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &EmbeddingBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if desc.num_embeddings < 0
            || desc.embedding_dim < 0
            || desc.num_indices < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBackwardPlan: num_embeddings / embedding_dim / \
                 num_indices must be non-negative",
            ));
        }
        let supported = matches!(T::KIND, ElementKind::F32 | ElementKind::F64);
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBackwardPlan: today only `f32`, `f64` wired \
                 (BW uses atomicAdd)",
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
            category: OpCategory::Embedding,
            op: EmbeddingKind::EmbeddingBackward as u16,
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
    pub fn can_implement<I: IndexElement>(&self, args: &EmbeddingBackwardArgs<'_, T, I>) -> Result<()> {
        if args.dout.shape[0] != self.desc.num_indices
            || args.dout.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBackwardPlan: dout shape must be \
                 [num_indices, embedding_dim]",
            ));
        }
        if args.indices.shape[0] != self.desc.num_indices {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBackwardPlan: indices.shape[0] mismatch with descriptor",
            ));
        }
        if args.dweight.shape[0] != self.desc.num_embeddings
            || args.dweight.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBackwardPlan: dweight shape must be \
                 [num_embeddings, embedding_dim]",
            ));
        }
        let dout_len = args.dout.data.len() as i64;
        let idx_len = args.indices.data.len() as i64;
        let dw_len = args.dweight.data.len() as i64;
        let dout_numel = args.dout.numel();
        let idx_numel = args.indices.numel();
        let dw_numel = args.dweight.numel();
        if dout_len < dout_numel {
            return Err(Error::BufferTooSmall {
                needed: dout_numel as usize,
                got: dout_len as usize,
            });
        }
        if idx_len < idx_numel {
            return Err(Error::BufferTooSmall {
                needed: idx_numel as usize,
                got: idx_len as usize,
            });
        }
        if dw_len < dw_numel {
            return Err(Error::BufferTooSmall {
                needed: dw_numel as usize,
                got: dw_len as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes (zero).
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
    ///
    /// Phase 11.5: generic over `I: IndexElement`.
    pub fn run<I: IndexElement>(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: EmbeddingBackwardArgs<'_, T, I>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let num_indices = self.desc.num_indices as i64;
        if num_indices == 0 || self.desc.embedding_dim == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *const c_void;
        let dweight_ptr = args.dweight.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        // Phase 11.5: padding_idx widens to i64 across FFI.
        let padding_idx: i64 = self.desc.padding_idx.unwrap_or(PADDING_DISABLED) as i64;

        let status = match (T::KIND, I::KIND) {
            (ElementKind::F32, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_backward_f32_run(
                    num_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    padding_idx, dout_ptr, idx_ptr, dweight_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_backward_f64_run(
                    num_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    padding_idx, dout_ptr, idx_ptr, dweight_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_backward_i64idx_f32_run(
                    num_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    padding_idx, dout_ptr, idx_ptr, dweight_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_backward_i64idx_f64_run(
                    num_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    padding_idx, dout_ptr, idx_ptr, dweight_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::EmbeddingBackwardPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
