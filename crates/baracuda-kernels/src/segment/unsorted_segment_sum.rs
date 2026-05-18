//! `unsorted_segment_sum` plan — Category S, unsorted variant.
//!
//! `out[s, d] = Σ_{n : segment_ids[n] == s} input[n, d]` with arbitrary
//! `segment_ids` ordering. The kernel zero-fills `output` then performs
//! `atomicAdd(output[seg[n], d], input[n, d])` for every input cell.
//!
//! TF `unsorted_segment_sum`. Output is non-deterministic (atomic
//! accumulation order); on a fixed problem the magnitude of the
//! float-summation drift is bounded by `O(eps · N)`.
//!
//! Dtype coverage: `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SegmentKind, TensorMut, TensorRef, Workspace,
};

use super::map_status;
use super::segment_sum::{validate_desc, SegDescView};

/// Descriptor for an `unsorted_segment_sum` op.
#[derive(Copy, Clone, Debug)]
pub struct UnsortedSegmentSumDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments (output rows). Out-of-range seg-ids in
    /// the input are silently dropped.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for UnsortedSegmentSumDescriptor {
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

/// Args bundle for an `unsorted_segment_sum` launch.
pub struct UnsortedSegmentSumArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]`, i32, in any order.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`. Overwritten — kernel zero-fills
    /// before the atomic accumulation phase.
    pub output: TensorMut<'a, T, 2>,
}

/// `unsorted_segment_sum` plan.
///
/// `out[s, d] = Σ_{n : segment_ids[n] == s} input[n, d]` with
/// arbitrary `segment_ids` ordering. The kernel zero-fills `output`
/// then performs `atomicAdd(output[seg[n], d], input[n, d])` per
/// input cell. TF `unsorted_segment_sum`.
///
/// **When to use**: forward unsorted segment-sum. For sorted IDs
/// prefer the deterministic
/// [`SegmentSumPlan`](crate::SegmentSumPlan). BW pass shares
/// [`SegmentSumBackwardPlan`](crate::SegmentSumBackwardPlan) with
/// the sorted variant.
///
/// **Dtypes**: `{f32, f64}` (native FP atomicAdd only).
///
/// **Shape limits**: `input` `[N, D]`; `segment_ids` `[N]` (any
/// order); `output` `[num_segments, D]`. Out-of-range IDs dropped.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: **non-deterministic** — atomicAdd
/// ordering varies. On a fixed problem the magnitude of float-
/// summation drift is bounded by `O(eps · N)`.
pub struct UnsortedSegmentSumPlan<T: Element> {
    desc: UnsortedSegmentSumDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> UnsortedSegmentSumPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &UnsortedSegmentSumDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "UnsortedSegmentSumPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_unsorted_sku::<T>(SegmentKind::UnsortedSegmentSum),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UnsortedSegmentSumArgs<'_, T>) -> Result<()> {
        validate_unsorted_args(
            self.desc.num_inputs,
            self.desc.embedding_dim,
            self.desc.num_segments,
            args.input.shape,
            args.segment_ids.shape,
            args.output.shape,
            "UnsortedSegmentSumPlan",
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
        args: UnsortedSegmentSumArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.num_segments as i64) * (self.desc.embedding_dim as i64);
        if total == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let id_ptr = args.segment_ids.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_sum_f32_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    in_ptr,
                    id_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_sum_f64_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    in_ptr,
                    id_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnsortedSegmentSumPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}

/// Build a `KernelSku` for an unsorted-segment plan. Unsorted variants
/// are non-deterministic (atomic accumulation order).
pub(crate) fn build_unsorted_sku<T: Element>(op: SegmentKind) -> KernelSku {
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

/// Validate args for an unsorted-segment FW plan.
pub(crate) fn validate_unsorted_args(
    num_inputs: i32,
    embedding_dim: i32,
    num_segments: i32,
    input_shape: [i32; 2],
    seg_shape: [i32; 1],
    output_shape: [i32; 2],
    _plan_name: &'static str,
) -> Result<()> {
    if input_shape != [num_inputs, embedding_dim] {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::segment: input shape != [num_inputs, embedding_dim]",
        ));
    }
    if seg_shape != [num_inputs] {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::segment: segment_ids shape != [num_inputs]",
        ));
    }
    if output_shape != [num_segments, embedding_dim] {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::segment: output shape != [num_segments, embedding_dim]",
        ));
    }
    Ok(())
}
