//! `embedding_bag` Max-mode FW plan — Category M. Phase 25.
//!
//! Per-bag max-reduction with per-feature argmax tracking. For each
//! bag `b` and each feature `d`:
//!
//! - `out[b, d]      = max(weight[indices[k], d])` for `k ∈ bag b`,
//!   excluding padding / OOB indices.
//! - `out_index[b, d] = the (first-occurrence) indices[k]` that
//!   contributed the max value, or `-1` if the bag was empty / all
//!   padded.
//!
//! The `out_index` tensor is the saved-state input to
//! [`crate::embedding::EmbeddingBagMaxBackwardPlan`].
//!
//! Trailblazer dtype coverage: `f32, f64, f16, bf16` (matches the
//! Sum/Mean FWs). Indices: `i32` + `i64`. f16 / bf16 accumulate in f32.
//!
//! **Tie-break**: first occurrence (lowest `k` in the bag). PyTorch
//! chooses the last occurrence; we document the divergence here.

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

/// Descriptor for an `embedding_bag` Max-mode op.
#[derive(Copy, Clone, Debug)]
pub struct EmbeddingBagMaxDescriptor {
    /// Vocabulary size — extent of `weight` along axis 0.
    pub num_embeddings: i32,
    /// Embedding dimension — extent of `weight` along axis 1.
    pub embedding_dim: i32,
    /// Number of bags — extent of `offsets` and of `out`, `out_index`
    /// along axis 0.
    pub num_bags: i32,
    /// Total flat-index length — extent of `indices`.
    pub total_indices: i32,
    /// Optional padding index. Indices matching `p` (or negative / OOB)
    /// are dropped from the bag.
    pub padding_idx: Option<i32>,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `embedding_bag` Max-mode launch.
pub struct EmbeddingBagMaxArgs<'a, T: Element, I: IndexElement = i32> {
    /// Weight matrix `[V, D]`.
    pub weight: TensorRef<'a, T, 2>,
    /// Flat index tensor `[total_indices]`.
    pub indices: TensorRef<'a, I, 1>,
    /// Per-bag start offset table `[num_bags]`, i32.
    pub offsets: TensorRef<'a, i32, 1>,
    /// Output max values `[num_bags, D]`.
    pub output: TensorMut<'a, T, 2>,
    /// Output per-(b, d) contributing-row index `[num_bags, D]`,
    /// always i32. `-1` for empty / all-padded bags.
    pub output_index: TensorMut<'a, i32, 2>,
}

/// `embedding_bag` Max-mode FW plan. Phase 25.
///
/// Per-bag max with per-feature argmax tracking. Pair with
/// [`crate::EmbeddingBagMaxBackwardPlan`] for autograd — the BW pass
/// scatters `dout[b, :]` into `dweight[output_index[b, :], :]`.
///
/// **Dtypes**: weight / output `{f32, f64, f16, bf16}`; index buffers
/// `i32` / `i64`. `output_index` is always `i32`.
///
/// **Shape limits**: same as the Sum/Mean FW plus
/// `output_index` `[num_bags, D]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable. No atomics.
///
/// **Index policy**: padding / OOB indices skipped; empty / all-padded
/// bag emits zero output and `-1` in every `output_index` cell.
///
/// **Tie-break**: first occurrence — diverges from PyTorch (last).
pub struct EmbeddingBagMaxPlan<T: Element> {
    desc: EmbeddingBagMaxDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> EmbeddingBagMaxPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &EmbeddingBagMaxDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBagMaxPlan: descriptor element != type parameter T",
            ));
        }
        if desc.num_embeddings < 0
            || desc.embedding_dim < 0
            || desc.num_bags < 0
            || desc.total_indices < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxPlan: num_embeddings / embedding_dim / \
                 num_bags / total_indices must be non-negative",
            ));
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBagMaxPlan: today only `f32`, `f64`, `f16`, `bf16` wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: if T::KIND == ElementKind::F64 {
                ElementKind::F64
            } else {
                ElementKind::F32
            },
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Embedding,
            op: EmbeddingKind::EmbeddingBagMax as u16,
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
    pub fn can_implement<I: IndexElement>(
        &self,
        args: &EmbeddingBagMaxArgs<'_, T, I>,
    ) -> Result<()> {
        if args.weight.shape[0] != self.desc.num_embeddings
            || args.weight.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxPlan: weight shape mismatch with descriptor",
            ));
        }
        if args.indices.shape[0] != self.desc.total_indices {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxPlan: indices.shape[0] != total_indices",
            ));
        }
        if args.offsets.shape[0] != self.desc.num_bags {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxPlan: offsets.shape[0] != num_bags",
            ));
        }
        if args.output.shape[0] != self.desc.num_bags
            || args.output.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxPlan: output shape must be [num_bags, embedding_dim]",
            ));
        }
        if args.output_index.shape[0] != self.desc.num_bags
            || args.output_index.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxPlan: output_index shape must be [num_bags, embedding_dim]",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes (zero).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run<I: IndexElement>(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: EmbeddingBagMaxArgs<'_, T, I>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.num_bags == 0 || self.desc.embedding_dim == 0 {
            return Ok(());
        }
        let weight_ptr = args.weight.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *const c_void;
        let off_ptr = args.offsets.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let out_idx_ptr = args.output_index.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let padding_idx: i64 = self.desc.padding_idx.unwrap_or(PADDING_DISABLED) as i64;

        let status = match (T::KIND, I::KIND) {
            (ElementKind::F32, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_f32_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr, out_idx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_f64_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr, out_idx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_f16_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr, out_idx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_bf16_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr, out_idx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_i64idx_f32_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr, out_idx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_i64idx_f64_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr, out_idx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_i64idx_f16_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr, out_idx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_i64idx_bf16_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr, out_idx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::EmbeddingBagMaxPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
