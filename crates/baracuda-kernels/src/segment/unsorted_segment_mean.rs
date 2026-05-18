//! `unsorted_segment_mean` plan — Category S, unsorted variant.
//!
//! `out[s, d] = mean_{n : segment_ids[n] == s} input[n, d]`. The
//! launcher runs three sub-kernels:
//!  1. zero-fill `output` and atomic-add the unsorted-sum.
//!  2. atomic-add per-segment counts into a workspace buffer.
//!  3. divide each `output` cell by its count (guarded for empty
//!     segments — they emit zero, not NaN).
//!
//! Workspace: `num_segments * sizeof(i32)` for the per-segment count
//! buffer.
//!
//! Dtype coverage: `f32, f64`.

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
use super::unsorted_segment_sum::{build_unsorted_sku, validate_unsorted_args};

/// Descriptor for an `unsorted_segment_mean` op.
#[derive(Copy, Clone, Debug)]
pub struct UnsortedSegmentMeanDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for UnsortedSegmentMeanDescriptor {
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

/// Args bundle for an `unsorted_segment_mean` launch.
pub struct UnsortedSegmentMeanArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]`, any order.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`.
    pub output: TensorMut<'a, T, 2>,
}

/// `unsorted_segment_mean` plan.
///
/// `out[s, d] = Σ input[n, d] / count[s]` with `segment_ids` in any
/// order. Two-phase: (1) atomicAdd accumulate, (2) divide by
/// per-segment count derived via a separate atomic-counter pass.
///
/// **When to use**: forward unsorted segment-mean. For sorted IDs
/// use [`SegmentMeanPlan`](crate::SegmentMeanPlan). BW pass shares
/// [`SegmentMeanBackwardPlan`](crate::SegmentMeanBackwardPlan).
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: `input` `[N, D]`; `segment_ids` `[N]` (any
/// order); `output` `[num_segments, D]`. Out-of-range IDs dropped;
/// empty segments emit zero.
///
/// **Workspace**: `num_segments * sizeof(i32)` bytes for the per-
/// segment count buffer. Use [`Self::workspace_size`].
///
/// **Precision guarantee**: **non-deterministic** — atomicAdd
/// ordering during accumulation.
pub struct UnsortedSegmentMeanPlan<T: Element> {
    desc: UnsortedSegmentMeanDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> UnsortedSegmentMeanPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &UnsortedSegmentMeanDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "UnsortedSegmentMeanPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_unsorted_sku::<T>(SegmentKind::UnsortedSegmentMean),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UnsortedSegmentMeanArgs<'_, T>) -> Result<()> {
        validate_unsorted_args(
            self.desc.num_inputs,
            self.desc.embedding_dim,
            self.desc.num_segments,
            args.input.shape,
            args.segment_ids.shape,
            args.output.shape,
            "UnsortedSegmentMeanPlan",
        )
    }

    /// Workspace size — `num_segments * sizeof(i32)`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        (self.desc.num_segments as usize).saturating_mul(core::mem::size_of::<i32>())
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
        workspace: Workspace<'_>,
        args: UnsortedSegmentMeanArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.num_segments as i64) * (self.desc.embedding_dim as i64);
        if total == 0 {
            return Ok(());
        }
        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes): (*mut c_void, usize) = match workspace {
            Workspace::None => {
                if needed == 0 {
                    (core::ptr::null_mut(), 0)
                } else {
                    return Err(Error::WorkspaceTooSmall { needed, got: 0 });
                }
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: slice.len(),
                    });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let id_ptr = args.segment_ids.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_mean_f32_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    in_ptr,
                    id_ptr,
                    out_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_mean_f64_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    in_ptr,
                    id_ptr,
                    out_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnsortedSegmentMeanPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
