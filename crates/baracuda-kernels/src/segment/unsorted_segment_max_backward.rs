//! `unsorted_segment_max_backward` plan — Category S, unsorted. Phase 25.
//!
//! Same recompute-argmax pattern as the sorted variant, but the BW
//! kernel scans the full `[0, N)` input range per (n, d) cell because
//! seg-ids aren't sorted. O(N) per cell; slow on big N. For small
//! segments the cost is acceptable.
//!
//! Determinism: matches the FW pass — the FW writes `out[s, d]` via
//! `atomicMax`-via-CAS, which is non-deterministic w.r.t. tie-break
//! across launches. The BW kernel here uses a deterministic
//! first-occurrence tie-break, but that means the (BW argmax) is not
//! guaranteed to equal (FW argmax) when there are ties — in tie-free
//! cases both pick the same cell.
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

/// Descriptor for an `unsorted_segment_max_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct UnsortedSegmentMaxBackwardDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for UnsortedSegmentMaxBackwardDescriptor {
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

/// Args bundle for an `unsorted_segment_max_backward` launch.
pub struct UnsortedSegmentMaxBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_segments, D]`.
    pub d_output: TensorRef<'a, T, 2>,
    /// FW input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]` (any order).
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Gradient w.r.t. input `[N, D]`. Overwritten by the launch.
    pub d_input: TensorMut<'a, T, 2>,
}

/// `unsorted_segment_max_backward` plan. Phase 25.
///
/// **Dtypes**: `{f32, f64}`. **Tie-break**: first occurrence (lowest k).
/// **Precision**: deterministic at the BW level (single thread per
/// cell, no atomics) but inconsistent with the FW when the FW's CAS
/// ordering broke a tie differently — see module-level note.
pub struct UnsortedSegmentMaxBackwardPlan<T: Element> {
    desc: UnsortedSegmentMaxBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> UnsortedSegmentMaxBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &UnsortedSegmentMaxBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "UnsortedSegmentMaxBackwardPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_unsorted_bw_sku::<T>(SegmentKind::UnsortedSegmentMaxBackward),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UnsortedSegmentMaxBackwardArgs<'_, T>) -> Result<()> {
        if args.d_output.shape != [self.desc.num_segments, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnsortedSegmentMaxBackwardPlan: d_output shape != [num_segments, D]",
            ));
        }
        if args.input.shape != [self.desc.num_inputs, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnsortedSegmentMaxBackwardPlan: input shape != [num_inputs, D]",
            ));
        }
        if args.segment_ids.shape != [self.desc.num_inputs] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnsortedSegmentMaxBackwardPlan: segment_ids shape != [num_inputs]",
            ));
        }
        if args.d_input.shape != [self.desc.num_inputs, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UnsortedSegmentMaxBackwardPlan: d_input shape != [num_inputs, D]",
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
        args: UnsortedSegmentMaxBackwardArgs<'_, T>,
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
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_max_backward_f32_run(
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
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_max_backward_f64_run(
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
                    "baracuda-kernels::UnsortedSegmentMaxBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}

/// SKU helper — Phase 25 unsorted BW. Deterministic at the kernel
/// level (single thread per cell, no atomics) but consistency with
/// the FW depends on tie-break behavior.
pub(crate) fn build_unsorted_bw_sku<T: Element>(op: SegmentKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if T::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: T::KIND,
        // Kernel itself is deterministic; FW-vs-BW consistency depends
        // on the tie-break alignment which the unsorted FW doesn't
        // guarantee. We flag deterministic=true (true at the per-
        // kernel level) and let docs cover the FW tie-break caveat.
        bit_stable_on_same_hardware: true,
        deterministic: true,
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
