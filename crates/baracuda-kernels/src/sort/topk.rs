//! `topk` plan — block-bitonic top-k select.
//!
//! `topk(x, k, dim=last, largest)` returns the top-k values along the
//! last dim and their indices. PyTorch `torch.topk`.
//!
//! Trailblazer dtype coverage: `f32, f64`.
//! Trailblazer caps:
//!   * `row_len ≤ 1024` (one block per row, bitonic).
//!   * `k ≤ 64` (LLM-inference range — common top-k for
//!     speculative decoding, sampling, MoE expert dispatch).
//!
//! BW: see [`crate::sort::TopkBackwardPlan`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::sort::build_sku;
use super::{SORT_MAX_ROW, TOPK_MAX_K};

/// Descriptor for a `topk` op.
#[derive(Copy, Clone, Debug)]
pub struct TopkDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each row. Trailblazer cap: `≤ 1024`.
    pub row_len: i32,
    /// Number of cells to retain per row. Trailblazer cap: `≤ 64`.
    pub k: i32,
    /// `true` = top-k by value; `false` = bottom-k.
    pub largest: bool,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `topk` launch.
pub struct TopkArgs<'a, T: Element> {
    /// Input `[batch, row_len]`.
    pub input: TensorRef<'a, T, 2>,
    /// Top-k values `[batch, k]`.
    pub values: TensorMut<'a, T, 2>,
    /// Top-k indices `[batch, k]` — saved for BW.
    pub indices: TensorMut<'a, i32, 2>,
}

/// `topk` plan.
///
/// `topk(x, k, dim=last, largest)` — returns the top-k values along
/// the last dim and their indices (PyTorch `torch.topk`).
///
/// **When to use**: top-k select for sampling, speculative decoding,
/// MoE expert dispatch. Pair with
/// [`TopkBackwardPlan`](crate::TopkBackwardPlan).
///
/// **Dtypes**: `{f32, f64}`; indices always `i32`.
///
/// **Shape limits**: rank-2 input `[batch, row_len]`; outputs
/// `[batch, k]`. `row_len ≤ 1024`; `k ≤ 64` (LLM-inference range).
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable on identical
/// hardware. Block-bitonic select is a fixed comparator network.
///
/// **Saved-indices contract**: FW emits both `values` and `indices`;
/// BW reads saved indices verbatim. Retain `indices` for autograd.
pub struct TopkPlan<T: Element> {
    desc: TopkDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> TopkPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &TopkDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TopkPlan: descriptor element != type parameter T",
            ));
        }
        if desc.batch < 0 || desc.row_len < 0 || desc.k < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkPlan: batch / row_len / k must be non-negative",
            ));
        }
        if desc.row_len > SORT_MAX_ROW {
            return Err(Error::Unsupported(
                "baracuda-kernels::TopkPlan: row_len > 1024 not supported in the \
                 block-bitonic trailblazer",
            ));
        }
        if desc.k > TOPK_MAX_K {
            return Err(Error::Unsupported(
                "baracuda-kernels::TopkPlan: k > 64 not supported in the trailblazer \
                 (LLM-inference range)",
            ));
        }
        if desc.k > desc.row_len {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkPlan: k must be <= row_len",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::TopkPlan: today only f32 / f64 wired",
            ));
        }
        let sku = build_sku::<T>(SortKind::Topk);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &TopkArgs<'_, T>) -> Result<()> {
        let in_shape = [self.desc.batch, self.desc.row_len];
        let out_shape = [self.desc.batch, self.desc.k];
        if args.input.shape != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkPlan: input shape != [batch, row_len]",
            ));
        }
        if args.values.shape != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkPlan: values shape != [batch, k]",
            ));
        }
        if args.indices.shape != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TopkPlan: indices shape != [batch, k]",
            ));
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
        args: TopkArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 || self.desc.k == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let vals_ptr = args.values.data.as_raw().0 as *mut c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let largest_flag = if self.desc.largest { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_topk_f32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    self.desc.k,
                    largest_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_topk_f64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    self.desc.k,
                    largest_flag,
                    in_ptr,
                    vals_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TopkPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
