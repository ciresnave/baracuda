//! `embedding_bag` Max-mode BW plan — Category M. Phase 25.
//!
//! Adjoint of [`crate::embedding::EmbeddingBagMaxPlan`]:
//! `dweight[output_index[b, d], d] += dout[b, d]`
//! (atomicAdd over `b` since multiple bags may have the same
//! contributing row).
//!
//! Skips cells where `output_index[b, d] < 0` (empty / all-padded
//! bags).
//!
//! Dtype coverage: `f32, f64` (atomicAdd is native-FP).
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

/// Descriptor for an `embedding_bag` Max-mode BW op.
#[derive(Copy, Clone, Debug)]
pub struct EmbeddingBagMaxBackwardDescriptor {
    /// Vocabulary size — extent of `dweight` along axis 0.
    pub num_embeddings: i32,
    /// Embedding dimension.
    pub embedding_dim: i32,
    /// Number of bags.
    pub num_bags: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for an `embedding_bag` Max-mode BW launch.
pub struct EmbeddingBagMaxBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_bags, D]`.
    pub dout: TensorRef<'a, T, 2>,
    /// FW saved per-(b, d) contributing-row index `[num_bags, D]`, i32.
    pub output_index: TensorRef<'a, i32, 2>,
    /// Gradient w.r.t. `weight` `[num_embeddings, D]`. Caller MUST
    /// pre-zero.
    pub dweight: TensorMut<'a, T, 2>,
}

/// `embedding_bag` Max-mode BW plan. Phase 25.
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Workspace**: none. Caller MUST zero `dweight` before launch.
///
/// **Precision guarantee**: **non-deterministic** — atomicAdd ordering
/// varies across launches when multiple bags share a contributing row.
pub struct EmbeddingBagMaxBackwardPlan<T: Element> {
    desc: EmbeddingBagMaxBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> EmbeddingBagMaxBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &EmbeddingBagMaxBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBagMaxBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if desc.num_embeddings < 0 || desc.embedding_dim < 0 || desc.num_bags < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxBackwardPlan: extents must be non-negative",
            ));
        }
        let supported = matches!(T::KIND, ElementKind::F32 | ElementKind::F64);
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::EmbeddingBagMaxBackwardPlan: today only `f32`, `f64` wired (BW uses atomicAdd)",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Embedding,
            op: EmbeddingKind::EmbeddingBagMaxBackward as u16,
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
    pub fn can_implement(&self, args: &EmbeddingBagMaxBackwardArgs<'_, T>) -> Result<()> {
        if args.dout.shape[0] != self.desc.num_bags
            || args.dout.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxBackwardPlan: dout shape != [num_bags, embedding_dim]",
            ));
        }
        if args.output_index.shape[0] != self.desc.num_bags
            || args.output_index.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxBackwardPlan: output_index shape != [num_bags, embedding_dim]",
            ));
        }
        if args.dweight.shape[0] != self.desc.num_embeddings
            || args.dweight.shape[1] != self.desc.embedding_dim
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::EmbeddingBagMaxBackwardPlan: dweight shape != [num_embeddings, embedding_dim]",
            ));
        }
        Ok(())
    }

    /// Workspace size — zero.
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
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: EmbeddingBagMaxBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.num_bags == 0 || self.desc.embedding_dim == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let idx_ptr = args.output_index.data.as_raw().0 as *const c_void;
        let dw_ptr = args.dweight.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_backward_f32_run(
                    self.desc.num_embeddings,
                    self.desc.embedding_dim,
                    self.desc.num_bags,
                    dout_ptr,
                    idx_ptr,
                    dw_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_embedding_bag_max_backward_f64_run(
                    self.desc.num_embeddings,
                    self.desc.embedding_dim,
                    self.desc.num_bags,
                    dout_ptr,
                    idx_ptr,
                    dw_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::EmbeddingBagMaxBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
