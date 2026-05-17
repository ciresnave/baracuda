//! `embedding_bag` BW plan — Category M.
//!
//! Adjoint of [`crate::embedding::EmbeddingBagPlan`]. For each bag `b`
//! and each non-padded / in-bounds index `k` in `[offsets[b], end_b)`:
//! - Sum-mode:  `dweight[indices[k], :] += dout[b, :]`
//! - Mean-mode: `dweight[indices[k], :] += dout[b, :] / bag_size(b)`,
//!   where `bag_size(b)` is the count of non-padded / in-bounds entries
//!   in the bag (matches the FW divisor).
//!
//! Trailblazer dtype coverage: `f32, f64`. atomicAdd is native-FP.
//!
//! Caller MUST zero `dweight` before launch.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, EmbeddingKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use crate::indexing::gather::map_status;

use super::embedding_bag::EmbeddingBagMode;
use super::PADDING_DISABLED;

/// Descriptor for an `embedding_bag_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct EmbeddingBagBackwardDescriptor {
    /// Vocabulary size — extent of `dweight` along axis 0.
    pub num_embeddings: i32,
    /// Embedding dimension — extent of `dweight` along axis 1.
    pub embedding_dim: i32,
    /// Number of bags — extent of `offsets` and of `dout` along axis 0.
    pub num_bags: i32,
    /// Total flat-index length — extent of `indices`.
    pub total_indices: i32,
    /// Reduction mode (must match the FW pass).
    pub mode: EmbeddingBagMode,
    /// Optional padding index (must match the FW pass).
    pub padding_idx: Option<i32>,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `embedding_bag_backward` launch.
pub struct EmbeddingBagBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_bags, D]`. Row-major contiguous.
    pub dout: TensorRef<'a, T, 2>,
    /// Flat index tensor `[total_indices]`, i32 (same as FW).
    pub indices: TensorRef<'a, i32, 1>,
    /// Per-bag offset table `[num_bags]`, i32 (same as FW).
    pub offsets: TensorRef<'a, i32, 1>,
    /// Gradient w.r.t. `weight` `[num_embeddings, D]`. Caller MUST pre-
    /// zero this.
    pub dweight: TensorMut<'a, T, 2>,
}

/// `embedding_bag_backward` plan.
pub struct EmbeddingBagBackwardPlan<T: Element> {
    desc: EmbeddingBagBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> EmbeddingBagBackwardPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &EmbeddingBagBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBagBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if desc.num_embeddings < 0
            || desc.embedding_dim < 0
            || desc.num_bags < 0
            || desc.total_indices < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagBackwardPlan: num_embeddings / embedding_dim / \
                 num_bags / total_indices must be non-negative",
            ));
        }
        let supported = matches!(T::KIND, ElementKind::F32 | ElementKind::F64);
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBagBackwardPlan: today only `f32`, `f64` wired \
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
        let op = match desc.mode {
            EmbeddingBagMode::Sum => EmbeddingKind::EmbeddingBagSumBackward,
            EmbeddingBagMode::Mean => EmbeddingKind::EmbeddingBagMeanBackward,
        };
        let sku = KernelSku {
            category: OpCategory::Embedding,
            op: op as u16,
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
    pub fn can_implement(&self, args: &EmbeddingBagBackwardArgs<'_, T>) -> Result<()> {
        if args.dout.shape[0] != self.desc.num_bags
            || args.dout.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagBackwardPlan: dout shape must be \
                 [num_bags, embedding_dim]",
            ));
        }
        if args.indices.shape[0] != self.desc.total_indices {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagBackwardPlan: indices.shape[0] != total_indices",
            ));
        }
        if args.offsets.shape[0] != self.desc.num_bags {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagBackwardPlan: offsets.shape[0] != num_bags",
            ));
        }
        if args.dweight.shape[0] != self.desc.num_embeddings
            || args.dweight.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagBackwardPlan: dweight shape must be \
                 [num_embeddings, embedding_dim]",
            ));
        }
        let dout_len = args.dout.data.len() as i64;
        let idx_len = args.indices.data.len() as i64;
        let off_len = args.offsets.data.len() as i64;
        let dw_len = args.dweight.data.len() as i64;
        let dout_numel = args.dout.numel();
        let idx_numel = args.indices.numel();
        let off_numel = args.offsets.numel();
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
        if off_len < off_numel {
            return Err(Error::BufferTooSmall {
                needed: off_numel as usize,
                got: off_len as usize,
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
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: EmbeddingBagBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.num_bags == 0 || self.desc.embedding_dim == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *const c_void;
        let off_ptr = args.offsets.data.as_raw().0 as *const c_void;
        let dw_ptr = args.dweight.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let padding_idx = self.desc.padding_idx.unwrap_or(PADDING_DISABLED);
        let mode = self.desc.mode.ffi_tag();

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_backward_f32_run(
                    self.desc.total_indices,
                    self.desc.num_embeddings,
                    self.desc.embedding_dim,
                    self.desc.num_bags,
                    mode,
                    padding_idx,
                    dout_ptr,
                    idx_ptr,
                    off_ptr,
                    dw_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_backward_f64_run(
                    self.desc.total_indices,
                    self.desc.num_embeddings,
                    self.desc.embedding_dim,
                    self.desc.num_bags,
                    mode,
                    padding_idx,
                    dout_ptr,
                    idx_ptr,
                    off_ptr,
                    dw_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::EmbeddingBagBackwardPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
