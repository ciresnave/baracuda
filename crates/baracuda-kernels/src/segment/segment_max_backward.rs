//! `segment_max_backward` plan — Category S, sorted variant. Phase 25.
//!
//! Adjoint of [`crate::segment::SegmentMaxPlan`]:
//! `d_input[k, d] = d_output[seg, d]` iff `k` is the (first-occurrence)
//! argmax of segment `seg` in column `d`, else `0`.
//!
//! Implementation notes:
//!
//! - Argmax is **recomputed in the BW kernel** rather than saved from
//!   the FW pass. This preserves the FW API source-compat (no paired-
//!   index tensor in the FW signature). The trade-off is one extra
//!   segment-scan per (n, d) cell in the BW; for the typical embedding
//!   workload where segments are short this is negligible compared to
//!   the global-memory traffic.
//! - Tie-break: **first occurrence** (lowest `k`). PyTorch chooses the
//!   *last* occurrence; the divergence is documented but not yet patched.
//! - Out-of-range / empty segments: `d_input` for the row is left as 0.
//!
//! Dtype coverage: `f32, f64`.
//!
//! This plan reuses [`crate::segment::SegmentMaxPlan`]'s descriptor
//! and arg shape; the only new tensor in BW is the upstream `d_output`
//! and the gradient target `d_input`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SegmentKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::segment_sum::{build_sku, validate_desc, SegDescView};

/// Descriptor for a `segment_max_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct SegmentMaxBackwardDescriptor {
    /// Number of input rows (matches the FW `num_inputs`).
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for SegmentMaxBackwardDescriptor {
    #[inline]
    fn view(&self) -> (i32, i32, i32, ElementKind) {
        (
            self.num_inputs,
            self.embedding_dim,
            self.num_segments,
            self.element,
        )
    }
}

/// Args bundle for a `segment_max_backward` launch.
pub struct SegmentMaxBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_segments, D]`.
    pub d_output: TensorRef<'a, T, 2>,
    /// FW input `[N, D]` — re-scanned to recompute argmax.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]` (same as FW).
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Gradient w.r.t. input `[N, D]`. Overwritten by the launch.
    pub d_input: TensorMut<'a, T, 2>,
}

/// `segment_max_backward` plan (sorted). Phase 25.
///
/// Adjoint of [`crate::SegmentMaxPlan`]. For each `(n, d)`:
/// `d_input[n, d] = d_output[seg, d]` iff `n` is the **first-occurrence**
/// argmax of segment `seg` in column `d`, else `0`.
///
/// **When to use**: BW pass for [`SegmentMaxPlan`](crate::SegmentMaxPlan).
/// Pair with the FW pass — both descriptors must agree on
/// `num_inputs`, `embedding_dim`, `num_segments`.
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: `d_output` `[num_segments, D]`; `input` and
/// `d_input` `[N, D]`; `segment_ids` `[N]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable. The BW kernel
/// uses a single thread per `(n, d)` cell and no atomics.
///
/// **Tie-break**: first occurrence (lowest `k`) — differs from
/// PyTorch which picks the last occurrence.
pub struct SegmentMaxBackwardPlan<T: Element> {
    desc: SegmentMaxBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SegmentMaxBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SegmentMaxBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "SegmentMaxBackwardPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_sku::<T>(SegmentKind::SegmentMaxBackward),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SegmentMaxBackwardArgs<'_, T>) -> Result<()> {
        if args.d_output.shape != [self.desc.num_segments, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentMaxBackwardPlan: d_output shape != [num_segments, D]",
            ));
        }
        if args.input.shape != [self.desc.num_inputs, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentMaxBackwardPlan: input shape != [num_inputs, D]",
            ));
        }
        if args.segment_ids.shape != [self.desc.num_inputs] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentMaxBackwardPlan: segment_ids shape != [num_inputs]",
            ));
        }
        if args.d_input.shape != [self.desc.num_inputs, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentMaxBackwardPlan: d_input shape != [num_inputs, D]",
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
        args: SegmentMaxBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.num_inputs as i64) * (self.desc.embedding_dim as i64);
        if total == 0 {
            return Ok(());
        }
        let do_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let id_ptr = args.segment_ids.data.as_raw().0 as *const c_void;
        let di_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_segment_max_backward_f32_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    do_ptr,
                    in_ptr,
                    id_ptr,
                    di_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_segment_max_backward_f64_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    do_ptr,
                    in_ptr,
                    id_ptr,
                    di_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SegmentMaxBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
