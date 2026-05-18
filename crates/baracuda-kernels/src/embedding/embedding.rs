//! `embedding` FW plan — Category M trailblazer.
//!
//! `out[n, :] = weight[indices[n], :]` for n in `0..num_indices`. When
//! `padding_idx` is `Some(p)`, rows where `indices[n] == p` are emitted
//! as all-zero (and the kernel does not read from `weight` for that row).
//! Out-of-range / negative `indices` entries are also emitted as
//! all-zero (no PyTorch-style negative-wrap).
//!
//! This is mathematically equivalent to `index_select` along axis 0
//! with `padding_idx`-aware zeroing fused into the same kernel pass.
//!
//! Trailblazer dtype coverage: `f32, f64, f16, bf16` — pure copy, no
//! arithmetic, so output is bit-exact at every dtype.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, EmbeddingKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use crate::indexing::gather::map_status;

use super::PADDING_DISABLED;

/// Descriptor for an `embedding` op.
#[derive(Copy, Clone, Debug)]
pub struct EmbeddingDescriptor {
    /// Vocabulary size — extent of `weight` along axis 0.
    pub num_embeddings: i32,
    /// Embedding dimension — extent of `weight` along axis 1.
    pub embedding_dim: i32,
    /// Number of indices to look up — extent of `indices` and of `out`
    /// along axis 0.
    pub num_indices: i32,
    /// Optional padding index. When `Some(p)`, rows where
    /// `indices[n] == p` are zeroed (and the kernel skips the read).
    pub padding_idx: Option<i32>,
    /// Value element type. Must match the type parameter on the plan.
    pub element: ElementKind,
}

/// Args bundle for an `embedding` launch.
pub struct EmbeddingArgs<'a, T: Element> {
    /// Weight matrix `[V, D]`. Row-major contiguous.
    pub weight: TensorRef<'a, T, 2>,
    /// Index tensor `[N]`, i32. Negative / OOB → all-zero row.
    pub indices: TensorRef<'a, i32, 1>,
    /// Output `[N, D]`. Row-major contiguous.
    pub output: TensorMut<'a, T, 2>,
}

/// `embedding` plan.
///
/// `out[n, :] = weight[indices[n], :]` (PyTorch
/// `torch.nn.functional.embedding`). When `padding_idx == Some(p)`,
/// rows where `indices[n] == p` are zeroed (no read from `weight`).
///
/// **When to use**: forward embedding-table lookup. Mathematically
/// equivalent to `index_select` along axis 0 with `padding_idx`-aware
/// zeroing fused into the same pass. Pair with
/// [`EmbeddingBackwardPlan`](crate::EmbeddingBackwardPlan) for
/// autograd; use [`EmbeddingBagPlan`](crate::EmbeddingBagPlan) for
/// bag-reductions.
///
/// **Dtypes**: weight / output `{f32, f64, f16, bf16}`; indices
/// always `i32`.
///
/// **Shape limits**: `weight` is `[V, D]`, `indices` is `[N]`,
/// `output` is `[N, D]`. All extents non-negative.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable on same
/// hardware. Pure copy, no arithmetic.
///
/// **Index policy**: negative or out-of-range indices emit an
/// all-zero row (no PyTorch-style wrap-around).
pub struct EmbeddingPlan<T: Element> {
    desc: EmbeddingDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> EmbeddingPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &EmbeddingDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingPlan: descriptor element != type parameter T",
            ));
        }
        if desc.num_embeddings < 0
            || desc.embedding_dim < 0
            || desc.num_indices < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingPlan: num_embeddings / embedding_dim / num_indices \
                 must be non-negative",
            ));
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingPlan: today only `f32`, `f64`, `f16`, `bf16` wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            // Pure copy — bit-stable, deterministic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Embedding,
            op: EmbeddingKind::Embedding as u16,
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
    pub fn can_implement(&self, args: &EmbeddingArgs<'_, T>) -> Result<()> {
        if args.weight.shape[0] != self.desc.num_embeddings
            || args.weight.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingPlan: weight shape mismatch with descriptor",
            ));
        }
        if args.indices.shape[0] != self.desc.num_indices {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingPlan: indices.shape[0] mismatch with descriptor",
            ));
        }
        if args.output.shape[0] != self.desc.num_indices
            || args.output.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingPlan: output shape must be [num_indices, embedding_dim]",
            ));
        }
        let weight_numel = args.weight.numel();
        let indices_numel = args.indices.numel();
        let out_numel = args.output.numel();
        let weight_len = args.weight.data.len() as i64;
        let indices_len = args.indices.data.len() as i64;
        let out_len = args.output.data.len() as i64;
        if weight_len < weight_numel {
            return Err(Error::BufferTooSmall {
                needed: weight_numel as usize,
                got: weight_len as usize,
            });
        }
        if indices_len < indices_numel {
            return Err(Error::BufferTooSmall {
                needed: indices_numel as usize,
                got: indices_len as usize,
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

    /// Workspace size in bytes (zero — this op has none).
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
        args: EmbeddingArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let num_indices = self.desc.num_indices as i64;
        if num_indices == 0 || self.desc.embedding_dim == 0 {
            return Ok(());
        }
        let weight_ptr = args.weight.data.as_raw().0 as *const c_void;
        let indices_ptr = args.indices.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let padding_idx = self.desc.padding_idx.unwrap_or(PADDING_DISABLED);

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_f32_run(
                    num_indices,
                    self.desc.num_embeddings,
                    self.desc.embedding_dim,
                    padding_idx,
                    weight_ptr,
                    indices_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_f64_run(
                    num_indices,
                    self.desc.num_embeddings,
                    self.desc.embedding_dim,
                    padding_idx,
                    weight_ptr,
                    indices_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_f16_run(
                    num_indices,
                    self.desc.num_embeddings,
                    self.desc.embedding_dim,
                    padding_idx,
                    weight_ptr,
                    indices_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bf16_run(
                    num_indices,
                    self.desc.num_embeddings,
                    self.desc.embedding_dim,
                    padding_idx,
                    weight_ptr,
                    indices_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::EmbeddingPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
