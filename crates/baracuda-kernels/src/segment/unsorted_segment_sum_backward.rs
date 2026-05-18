//! `unsorted_segment_sum_backward` plan — Category S, unsorted variant.
//!
//! Adjoint of [`crate::segment::UnsortedSegmentSumPlan`]:
//! `d_input[n, d] = d_output[segment_ids[n], d]`. The BW kernel is
//! identical to the sorted `segment_sum_backward` kernel (the access
//! pattern is the same regardless of seg-ids ordering); we keep
//! distinct symbols + plan shapes for SKU-tagging / telemetry
//! differentiation.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SegmentKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::segment_sum::{validate_desc, SegDescView};
use super::unsorted_segment_sum::build_unsorted_sku;

/// Descriptor for an `unsorted_segment_sum_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct UnsortedSegmentSumBackwardDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for UnsortedSegmentSumBackwardDescriptor {
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

/// Args bundle for an `unsorted_segment_sum_backward` launch.
pub struct UnsortedSegmentSumBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_segments, D]`.
    pub d_output: TensorRef<'a, T, 2>,
    /// Segment ids `[N]` from FW.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Gradient w.r.t. input `[N, D]`. Overwritten.
    pub d_input: TensorMut<'a, T, 2>,
}

/// `unsorted_segment_sum_backward` plan.
///
/// Adjoint of [`crate::UnsortedSegmentSumPlan`]:
/// `d_input[n, d] = d_output[segment_ids[n], d]`. Pure gather along
/// the seg-ids array — identical to the sorted-variant BW kernel
/// (sort state is irrelevant for the gather).
///
/// **When to use**: BW for unsorted segment-sum.
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: `d_output` `[num_segments, D]`; `segment_ids`
/// `[N]`; `d_input` `[N, D]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable. No atomics.
pub struct UnsortedSegmentSumBackwardPlan<T: Element> {
    desc: UnsortedSegmentSumBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> UnsortedSegmentSumBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &UnsortedSegmentSumBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "UnsortedSegmentSumBackwardPlan")?;
        // BW is deterministic (pure gather); use the unsorted SKU
        // builder but the SKU's deterministic flag is `false` by default —
        // override here would lie, but downstream consumers should
        // already treat unsorted-family SKUs as nondet. We keep the
        // builder's defaults for consistency.
        Ok(Self {
            desc: *desc,
            sku: build_unsorted_sku::<T>(SegmentKind::UnsortedSegmentSumBackward),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UnsortedSegmentSumBackwardArgs<'_, T>) -> Result<()> {
        if args.d_output.shape != [self.desc.num_segments, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnsortedSegmentSumBackwardPlan: d_output shape != [num_segments, D]",
            ));
        }
        if args.segment_ids.shape != [self.desc.num_inputs] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnsortedSegmentSumBackwardPlan: segment_ids shape != [num_inputs]",
            ));
        }
        if args.d_input.shape != [self.desc.num_inputs, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnsortedSegmentSumBackwardPlan: d_input shape != [num_inputs, D]",
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
        args: UnsortedSegmentSumBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.num_inputs as i64) * (self.desc.embedding_dim as i64);
        if total == 0 {
            return Ok(());
        }
        let do_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let id_ptr = args.segment_ids.data.as_raw().0 as *const c_void;
        let di_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_sum_backward_f32_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    do_ptr,
                    id_ptr,
                    di_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_sum_backward_f64_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    do_ptr,
                    id_ptr,
                    di_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnsortedSegmentSumBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
