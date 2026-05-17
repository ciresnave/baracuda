//! `segment_prod` plan — Category S, sorted variant.
//!
//! `out[s, d] = prod_{n : segment_ids[n] == s} input[n, d]`. Requires
//! `segment_ids` to be monotonically non-decreasing. TF / JAX
//! `segment_prod`.
//!
//! Empty segments emit `1` (multiplicative identity).
//!
//! FW only. BW deferred (the analytic gradient is
//! `d_input[n, d] = d_output[seg, d] * (prod[seg, d] / input[n, d])`
//! which is unstable when any input is near zero — would need a
//! safer-divide path or saved-running-prods).

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

/// Descriptor for a `segment_prod` op.
#[derive(Copy, Clone, Debug)]
pub struct SegmentProdDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for SegmentProdDescriptor {
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

/// Args bundle for a `segment_prod` launch.
pub struct SegmentProdArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]` — sorted non-decreasing.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`.
    pub output: TensorMut<'a, T, 2>,
}

/// `segment_prod` plan (sorted, FW only).
pub struct SegmentProdPlan<T: Element> {
    desc: SegmentProdDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SegmentProdPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SegmentProdDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "SegmentProdPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_sku::<T>(SegmentKind::SegmentProd),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SegmentProdArgs<'_, T>) -> Result<()> {
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
            "SegmentProdPlan",
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
        args: SegmentProdArgs<'_, T>,
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
            SortedFwOp::Prod,
        )
    }
}
