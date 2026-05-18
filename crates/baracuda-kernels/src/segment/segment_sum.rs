//! `segment_sum` plan — Category S, sorted variant.
//!
//! `out[s, d] = Σ_{n : segment_ids[n] == s} input[n, d]`. Requires
//! `segment_ids` to be monotonically non-decreasing. TF / JAX
//! `segment_sum`.
//!
//! Trailblazer dtype coverage: `f32, f64`.
//!
//! BW: see [`crate::segment::SegmentSumBackwardPlan`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SegmentKind, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a `segment_sum` op.
#[derive(Copy, Clone, Debug)]
pub struct SegmentSumDescriptor {
    /// Number of input rows (length of `segment_ids`).
    pub num_inputs: i32,
    /// Embedding / feature dim — second axis of `input` and `output`.
    pub embedding_dim: i32,
    /// Total number of segments — first axis of `output`. Output is
    /// allocated for this many rows even when some segments are empty.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `segment_sum` launch.
pub struct SegmentSumArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]`, i32, sorted non-decreasing, values in
    /// `[0, num_segments)`.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`. Overwritten by the launch — no
    /// accumulation into pre-existing state.
    pub output: TensorMut<'a, T, 2>,
}

/// `segment_sum` plan (sorted).
///
/// `out[s, d] = Σ_{n : segment_ids[n] == s} input[n, d]` (TF / JAX
/// `segment_sum`). Requires `segment_ids` to be monotonically
/// non-decreasing.
///
/// **When to use**: forward sorted segment-sum. For unsorted IDs use
/// [`UnsortedSegmentSumPlan`](crate::UnsortedSegmentSumPlan). Pair
/// with [`SegmentSumBackwardPlan`](crate::SegmentSumBackwardPlan)
/// for autograd.
///
/// **Dtypes**: `{f32, f64}` (matches the family — kernels rely on
/// FP atomic primitives even in the sorted variant for some paths).
///
/// **Shape limits**: `input` is `[N, D]`, `segment_ids` is `[N]`
/// with values in `[0, num_segments)`; `output` is `[num_segments, D]`.
/// All extents non-negative.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: **deterministic, bit-stable** — single
/// thread per output cell sweeps the segment's row range in order.
///
/// **Index policy**: out-of-range segment IDs (`< 0` or
/// `≥ num_segments`) are silently dropped (TF / JAX semantic).
/// Output buffer is fully overwritten (no accumulation into prior
/// state).
pub struct SegmentSumPlan<T: Element> {
    desc: SegmentSumDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SegmentSumPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &SegmentSumDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "SegmentSumPlan")?;
        let sku = build_sku::<T>(SegmentKind::SegmentSum);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SegmentSumArgs<'_, T>) -> Result<()> {
        validate_args(
            &self.desc,
            args.input.shape,
            args.segment_ids.shape,
            args.output.shape,
            "SegmentSumPlan",
        )
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
        args: SegmentSumArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total_out = (self.desc.num_segments as i64) * (self.desc.embedding_dim as i64);
        if total_out == 0 {
            return Ok(());
        }
        run_sorted_fw::<T>(
            stream,
            self.desc.num_inputs,
            self.desc.embedding_dim,
            self.desc.num_segments,
            &args.input,
            &args.segment_ids,
            &args.output,
            SortedFwOp::Sum,
        )
    }
}

/// Validate descriptor fields shared across the sorted-family plans.
/// The `_plan_name` parameter is unused today; reserved for richer
/// error messages without churning every call site when we wire it in.
pub(crate) fn validate_desc(
    desc_num_inputs_dim_seg: impl SegDescView,
    expected_element: ElementKind,
    _plan_name: &'static str,
) -> Result<()> {
    let (n, d, ns, el) = desc_num_inputs_dim_seg.view();
    if el != expected_element {
        return Err(Error::Unsupported(
            "baracuda-kernels::segment: descriptor element != type parameter T",
        ));
    }
    if n < 0 || d < 0 || ns < 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::segment: num_inputs / embedding_dim / num_segments must be non-negative",
        ));
    }
    if !matches!(el, ElementKind::F32 | ElementKind::F64) {
        return Err(Error::Unsupported(
            "baracuda-kernels::segment: today only f32, f64 wired (atomicAdd / atomic-CAS restricted to native-FP-atomic types)",
        ));
    }
    Ok(())
}

/// Trait abstracting the four descriptor fields shared by every sorted
/// + unsorted segment plan. Lets `validate_desc` accept any descriptor
/// without forcing a concrete type.
pub(crate) trait SegDescView {
    fn view(&self) -> (i32, i32, i32, ElementKind);
}

impl SegDescView for SegmentSumDescriptor {
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

/// Validate args shared across the sorted-family FW plans.
pub(crate) fn validate_args(
    desc: &SegmentSumDescriptor,
    input_shape: [i32; 2],
    seg_shape: [i32; 1],
    output_shape: [i32; 2],
    _plan_name: &'static str,
) -> Result<()> {
    if input_shape != [desc.num_inputs, desc.embedding_dim] {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::segment: input shape != [num_inputs, embedding_dim]",
        ));
    }
    if seg_shape != [desc.num_inputs] {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::segment: segment_ids shape != [num_inputs]",
        ));
    }
    if output_shape != [desc.num_segments, desc.embedding_dim] {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::segment: output shape != [num_segments, embedding_dim]",
        ));
    }
    Ok(())
}

/// Construct a `KernelSku` for the segment-family plan.
pub(crate) fn build_sku<T: Element>(op: SegmentKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if T::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: T::KIND,
        // Sorted: deterministic (single thread per output cell, in-order
        // sweep). Unsorted: atomic accumulation → not deterministic.
        // We set conservative defaults here and let unsorted plans
        // re-tag via their own builder when they need to differ.
        bit_stable_on_same_hardware: matches!(
            op,
            SegmentKind::SegmentSum
                | SegmentKind::SegmentMean
                | SegmentKind::SegmentMax
                | SegmentKind::SegmentMin
                | SegmentKind::SegmentProd
                | SegmentKind::SegmentSumBackward
                | SegmentKind::SegmentMeanBackward
                | SegmentKind::UnsortedSegmentSumBackward
                | SegmentKind::UnsortedSegmentMeanBackward
        ),
        deterministic: matches!(
            op,
            SegmentKind::SegmentSum
                | SegmentKind::SegmentMean
                | SegmentKind::SegmentMax
                | SegmentKind::SegmentMin
                | SegmentKind::SegmentProd
                | SegmentKind::SegmentSumBackward
                | SegmentKind::SegmentMeanBackward
                | SegmentKind::UnsortedSegmentSumBackward
                | SegmentKind::UnsortedSegmentMeanBackward
        ),
    };
    KernelSku {
        category: OpCategory::SegmentOps,
        op: op as u16,
        element: T::KIND,
        aux_element: Some(ElementKind::I32),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}

/// Sorted FW op tag — picks the launcher symbol at `run` time.
#[derive(Copy, Clone, Debug)]
pub(crate) enum SortedFwOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

/// Shared sorted-FW launch helper.
pub(crate) fn run_sorted_fw<T: Element>(
    stream: &Stream,
    n: i32,
    d: i32,
    num_segments: i32,
    input: &TensorRef<'_, T, 2>,
    segment_ids: &TensorRef<'_, i32, 1>,
    output: &TensorMut<'_, T, 2>,
    op: SortedFwOp,
) -> Result<()> {
    let in_ptr = input.data.as_raw().0 as *const c_void;
    let id_ptr = segment_ids.data.as_raw().0 as *const c_void;
    let out_ptr = output.data.as_raw().0 as *mut c_void;
    let stream_ptr = stream.as_raw() as *mut c_void;

    let status = match (T::KIND, op) {
        (ElementKind::F32, SortedFwOp::Sum) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_sum_f32_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F64, SortedFwOp::Sum) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_sum_f64_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F32, SortedFwOp::Mean) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_mean_f32_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F64, SortedFwOp::Mean) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_mean_f64_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F32, SortedFwOp::Max) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_max_f32_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F64, SortedFwOp::Max) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_max_f64_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F32, SortedFwOp::Min) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_min_f32_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F64, SortedFwOp::Min) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_min_f64_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F32, SortedFwOp::Prod) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_prod_f32_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        (ElementKind::F64, SortedFwOp::Prod) => unsafe {
            baracuda_kernels_sys::baracuda_kernels_segment_prod_f64_run(
                n, d, num_segments, in_ptr, id_ptr, out_ptr,
                core::ptr::null_mut(), 0, stream_ptr,
            )
        },
        _ => {
            return Err(Error::Unsupported(
                "baracuda-kernels::segment::run_sorted_fw reached an unimplemented dtype \
                 — select() should have caught this",
            ));
        }
    };
    map_status(status)
}
