//! `segment_max` plan — Category S, sorted variant.
//!
//! `out[s, d] = max_{n : segment_ids[n] == s} input[n, d]`. Requires
//! `segment_ids` to be monotonically non-decreasing. TF / JAX
//! `segment_max`.
//!
//! Empty segments emit zero (the kernel writes a per-op identity
//! sentinel — see `segment_sorted_kernel`).
//!
//! FW only. BW deferred — would need argmax tracking from FW.

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

/// Descriptor for a `segment_max` op.
#[derive(Copy, Clone, Debug)]
pub struct SegmentMaxDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for SegmentMaxDescriptor {
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

/// Args bundle for a `segment_max` launch.
pub struct SegmentMaxArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]` — sorted non-decreasing.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`.
    pub output: TensorMut<'a, T, 2>,
}

/// `segment_max` plan (sorted, FW only).
pub struct SegmentMaxPlan<T: Element> {
    desc: SegmentMaxDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SegmentMaxPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SegmentMaxDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "SegmentMaxPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_sku::<T>(SegmentKind::SegmentMax),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SegmentMaxArgs<'_, T>) -> Result<()> {
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
            "SegmentMaxPlan",
        )
    }

    /// Workspace size — zero.
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
        args: SegmentMaxArgs<'_, T>,
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
            SortedFwOp::Max,
        )
    }
}
