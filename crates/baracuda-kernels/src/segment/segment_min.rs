//! `segment_min` plan — Category S, sorted variant.
//!
//! `out[s, d] = min_{n : segment_ids[n] == s} input[n, d]`. Requires
//! `segment_ids` to be monotonically non-decreasing. TF / JAX
//! `segment_min`.
//!
//! FW only. BW deferred (argmin tracking).

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

/// Descriptor for a `segment_min` op.
#[derive(Copy, Clone, Debug)]
pub struct SegmentMinDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for SegmentMinDescriptor {
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

/// Args bundle for a `segment_min` launch.
pub struct SegmentMinArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]` — sorted non-decreasing.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`.
    pub output: TensorMut<'a, T, 2>,
}

/// `segment_min` plan (sorted, FW only).
///
/// `out[s, d] = min_{n : segment_ids[n] == s} input[n, d]`. Sorted
/// segment-min mirror of [`SegmentMaxPlan`](crate::SegmentMaxPlan).
///
/// **When to use**: forward sorted segment-min. **No BW plan** —
/// argmin tracking deferred.
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: `input` `[N, D]`; `segment_ids` `[N]`;
/// `output` `[num_segments, D]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
///
/// **Index policy**: out-of-range IDs dropped. Empty segments emit
/// the per-op identity sentinel.
pub struct SegmentMinPlan<T: Element> {
    desc: SegmentMinDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SegmentMinPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SegmentMinDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "SegmentMinPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_sku::<T>(SegmentKind::SegmentMin),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SegmentMinArgs<'_, T>) -> Result<()> {
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
            "SegmentMinPlan",
        )
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
        args: SegmentMinArgs<'_, T>,
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
            SortedFwOp::Min,
        )
    }
}
