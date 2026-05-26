//! `embedding_bag` FW plan — Category M.
//!
//! Per-bag reduction over a flat `indices` array partitioned by the
//! `offsets` table. For each bag `b`:
//! - `start = offsets[b]`
//! - `end   = offsets[b + 1] if b + 1 < num_bags else total_indices`
//! - `out[b, :] = reduce(weight[indices[k], :] for k in start..end)`
//!
//! Reducer is selected by [`EmbeddingBagMode`]:
//! - `Sum`: pure addition.
//! - `Mean`: sum, then divide by the post-skip bag size. If every
//!   entry in the bag was padding / OOB the row is emitted as zero
//!   (no divide by zero).
//!
//! `padding_idx` rows are dropped from the reduction (also excluded
//! from the Mean divisor). Empty bags (`start == end`) emit zero rows.
//!
//! `Max`-mode is deferred — it needs per-feature argmax tracking on FW
//! so the BW can scatter into the contributing rows.
//!
//! Trailblazer dtype coverage: `f32, f64, f16, bf16`. f16 / bf16
//! accumulate in f32 internally before casting back to T at write.

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

/// Reduction mode for `embedding_bag`.
///
/// **Intentionally NOT `#[non_exhaustive]`** — Sum / Mean is the
/// closed set for the EmbeddingBag op family today; Max mode lives on
/// its own [`super::EmbeddingBagMaxPlan`] plan (separate FFI surface)
/// rather than as a third variant here. New variants would be a
/// deliberate breaking-change event.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EmbeddingBagMode {
    /// `out[b, :] = Σ weight[indices[k], :]` for k in bag b.
    Sum,
    /// `out[b, :] = (Σ weight[indices[k], :]) / bag_size(b)` where
    /// `bag_size` counts only non-padded / in-bounds indices.
    Mean,
}

impl EmbeddingBagMode {
    /// FFI tag matching `kModeSum` / `kModeMean` in the .cuh header.
    #[inline]
    pub(crate) fn ffi_tag(self) -> i32 {
        match self {
            EmbeddingBagMode::Sum => 0,
            EmbeddingBagMode::Mean => 1,
        }
    }

    /// Maps to the corresponding [`EmbeddingKind`] discriminant for SKU
    /// tagging.
    #[inline]
    fn kind(self) -> EmbeddingKind {
        match self {
            EmbeddingBagMode::Sum => EmbeddingKind::EmbeddingBagSum,
            EmbeddingBagMode::Mean => EmbeddingKind::EmbeddingBagMean,
        }
    }
}

/// Descriptor for an `embedding_bag` op.
#[derive(Copy, Clone, Debug)]
pub struct EmbeddingBagDescriptor {
    /// Vocabulary size — extent of `weight` along axis 0.
    pub num_embeddings: i32,
    /// Embedding dimension — extent of `weight` along axis 1.
    pub embedding_dim: i32,
    /// Number of bags — extent of `offsets` and of `output` along axis 0.
    pub num_bags: i32,
    /// Total flat-index length — extent of `indices` along axis 0.
    pub total_indices: i32,
    /// Reduction mode.
    pub mode: EmbeddingBagMode,
    /// Optional padding index. When `Some(p)`, indices equal to `p` (or
    /// negative / OOB) are dropped from the bag's reduction. Excluded
    /// from the Mean divisor.
    pub padding_idx: Option<i32>,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `embedding_bag` launch.
///
/// Phase 11.5: `I: IndexElement` generic (`i32` or `i64`) for the
/// index tensor. `offsets` stays i32 — bag boundaries fit comfortably.
pub struct EmbeddingBagArgs<'a, T: Element, I: IndexElement = i32> {
    /// Weight matrix `[V, D]`. Row-major contiguous.
    pub weight: TensorRef<'a, T, 2>,
    /// Flat index tensor `[total_indices]`. `i32` (legacy) or `i64`
    /// (PyTorch default).
    pub indices: TensorRef<'a, I, 1>,
    /// Per-bag start offset table `[num_bags]`, i32. `offsets[0]` should
    /// be 0; `offsets[b+1] - offsets[b]` is bag `b`'s size; the last
    /// bag's implicit end is `total_indices`.
    pub offsets: TensorRef<'a, i32, 1>,
    /// Output `[num_bags, D]`. Row-major contiguous.
    pub output: TensorMut<'a, T, 2>,
}

/// `embedding_bag` plan.
///
/// Per-bag reduction over a flat `indices` array partitioned by the
/// `offsets` table (PyTorch `torch.nn.functional.embedding_bag`). For
/// each bag `b`: `out[b, :] = reduce(weight[indices[k], :])` for
/// `k ∈ offsets[b]..offsets[b+1]` (last bag's end is `total_indices`).
///
/// **When to use**: forward pooled embedding lookup (e.g. continuous
/// bag-of-words). Pair with
/// [`EmbeddingBagBackwardPlan`](crate::EmbeddingBagBackwardPlan) for
/// autograd. For non-pooled lookup, use
/// [`EmbeddingPlan`](crate::EmbeddingPlan).
///
/// **Dtypes**: weight / output `{f32, f64, f16, bf16}`; indices and
/// offsets always `i32`. f16 / bf16 accumulate in f32 internally
/// before the cast back to T at write.
///
/// **Shape limits**: `weight` is `[V, D]`, `indices` is
/// `[total_indices]`, `offsets` is `[num_bags]`, `output` is
/// `[num_bags, D]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable on same
/// hardware. No atomics on FW.
///
/// **Index policy**: `padding_idx` (or negative / OOB) indices are
/// dropped from the bag's reduction; excluded from the Mean divisor.
/// Empty bags (`start == end`) emit zero rows; an all-padding bag in
/// Mean mode also emits zero (no divide-by-zero).
///
/// **Known limitations**: `Max` mode is deferred — it needs
/// per-feature argmax tracking on FW so the BW can scatter into the
/// contributing rows.
pub struct EmbeddingBagPlan<T: Element> {
    desc: EmbeddingBagDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> EmbeddingBagPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &EmbeddingBagDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBagPlan: descriptor element != type parameter T",
            ));
        }
        if desc.num_embeddings < 0
            || desc.embedding_dim < 0
            || desc.num_bags < 0
            || desc.total_indices < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagPlan: num_embeddings / embedding_dim / num_bags / \
                 total_indices must be non-negative",
            ));
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBagPlan: today only `f32`, `f64`, `f16`, `bf16` wired",
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
            // No atomics on FW — same input → same output bit pattern.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Embedding,
            op: desc.mode.kind() as u16,
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
    pub fn can_implement<I: IndexElement>(&self, args: &EmbeddingBagArgs<'_, T, I>) -> Result<()> {
        if args.weight.shape[0] != self.desc.num_embeddings
            || args.weight.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagPlan: weight shape mismatch with descriptor",
            ));
        }
        if args.indices.shape[0] != self.desc.total_indices {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagPlan: indices.shape[0] != total_indices",
            ));
        }
        if args.offsets.shape[0] != self.desc.num_bags {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagPlan: offsets.shape[0] != num_bags",
            ));
        }
        if args.output.shape[0] != self.desc.num_bags
            || args.output.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagPlan: output shape must be [num_bags, embedding_dim]",
            ));
        }
        let weight_len = args.weight.data.len() as i64;
        let idx_len = args.indices.data.len() as i64;
        let off_len = args.offsets.data.len() as i64;
        let out_len = args.output.data.len() as i64;
        let weight_numel = args.weight.numel();
        let idx_numel = args.indices.numel();
        let off_numel = args.offsets.numel();
        let out_numel = args.output.numel();
        if weight_len < weight_numel {
            return Err(Error::BufferTooSmall {
                needed: weight_numel as usize,
                got: weight_len as usize,
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
        if out_len < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: out_len as usize,
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
        args: EmbeddingBagArgs<'_, T, I>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.num_bags == 0 || self.desc.embedding_dim == 0 {
            return Ok(());
        }
        let weight_ptr = args.weight.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *const c_void;
        let off_ptr = args.offsets.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        // Phase 11.5: padding_idx widens to i64 across FFI.
        let padding_idx: i64 = self.desc.padding_idx.unwrap_or(PADDING_DISABLED) as i64;
        let mode = self.desc.mode.ffi_tag();

        let status = match (T::KIND, I::KIND) {
            (ElementKind::F32, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_f32_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, mode, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_f64_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, mode, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_f16_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, mode, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_bf16_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, mode, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_i64idx_f32_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, mode, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_i64idx_f64_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, mode, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_i64idx_f16_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, mode, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_i64idx_bf16_run(
                    self.desc.total_indices, self.desc.num_embeddings, self.desc.embedding_dim,
                    self.desc.num_bags, mode, padding_idx,
                    weight_ptr, idx_ptr, off_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::EmbeddingBagPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
