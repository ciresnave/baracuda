//! `segment_mean` plan — Category S, sorted variant.
//!
//! `out[s, d] = mean_{n : segment_ids[n] == s} input[n, d]`. Requires
//! `segment_ids` to be monotonically non-decreasing. TF / JAX
//! `segment_mean`.
//!
//! Empty segments emit zero (no NaN — division is guarded inside the
//! kernel). BW: see [`crate::segment::SegmentMeanBackwardPlan`].

use core::marker::PhantomData;

use baracuda_cutlass::Result;
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SegmentKind, TensorMut,
    TensorRef, Workspace,
};

use super::segment_sum::{
    build_sku, run_sorted_fw, validate_args, validate_desc, SegDescView, SegmentSumDescriptor,
    SortedFwOp,
};

/// Descriptor for a `segment_mean` op.
#[derive(Copy, Clone, Debug)]
pub struct SegmentMeanDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for SegmentMeanDescriptor {
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

/// Args bundle for a `segment_mean` launch.
pub struct SegmentMeanArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]` — sorted non-decreasing.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`.
    pub output: TensorMut<'a, T, 2>,
}

/// `segment_mean` plan (sorted).
///
/// `out[s, d] = mean_{n : segment_ids[n] == s} input[n, d]` (TF / JAX
/// `segment_mean`). Requires `segment_ids` monotonically non-decreasing.
///
/// **When to use**: forward sorted segment-mean. Pair with
/// [`SegmentMeanBackwardPlan`](crate::SegmentMeanBackwardPlan).
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: `input` `[N, D]`; `segment_ids` `[N]` with values
/// in `[0, num_segments)`; `output` `[num_segments, D]`.
///
/// **Workspace**: none — segment counts derived inline via binary
/// search.
///
/// **Precision guarantee**: deterministic, bit-stable.
///
/// **Index policy**: out-of-range IDs dropped. Empty segments emit
/// zero (division is guarded; no NaN).
pub struct SegmentMeanPlan<T: Element> {
    desc: SegmentMeanDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SegmentMeanPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SegmentMeanDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "SegmentMeanPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_sku::<T>(SegmentKind::SegmentMean),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SegmentMeanArgs<'_, T>) -> Result<()> {
        let proxy = SegmentSumDescriptor {
            num_inputs: self.desc.num_inputs,
            embedding_dim: self.desc.embedding_dim,
            num_segments: self.desc.num_segments,
            element: self.desc.element,
        };
        validate_args(
            &proxy,
            args.input.shape,
            args.segment_ids.shape,
            args.output.shape,
            "SegmentMeanPlan",
        )
    }

    /// Workspace size — zero (count computed inline via binary search).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
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
        args: SegmentMeanArgs<'_, T>,
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
            SortedFwOp::Mean,
        )
    }
}
