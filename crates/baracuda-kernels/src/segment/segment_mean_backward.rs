//! `segment_mean_backward` plan — Category S, sorted variant.
//!
//! Adjoint of [`crate::segment::SegmentMeanPlan`]:
//! `d_input[n, d] = d_output[segment_ids[n], d] / count[segment_ids[n]]`.
//! The kernel computes the per-segment count on-the-fly into a
//! caller-supplied workspace, then divides the gathered `d_output`
//! cell by the count.
//!
//! Workspace: `num_segments * sizeof(i32)` bytes. Sorted and unsorted
//! variants share this kernel (the count computation is order-
//! independent).
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
use super::segment_sum::{build_sku, validate_desc, SegDescView};

/// Descriptor for a `segment_mean_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct SegmentMeanBackwardDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for SegmentMeanBackwardDescriptor {
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

/// Args bundle for a `segment_mean_backward` launch.
pub struct SegmentMeanBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_segments, D]`.
    pub d_output: TensorRef<'a, T, 2>,
    /// Segment ids `[N]` from FW.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Gradient w.r.t. input `[N, D]`. Overwritten.
    pub d_input: TensorMut<'a, T, 2>,
}

/// `segment_mean_backward` plan.
pub struct SegmentMeanBackwardPlan<T: Element> {
    desc: SegmentMeanBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SegmentMeanBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SegmentMeanBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "SegmentMeanBackwardPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_sku::<T>(SegmentKind::SegmentMeanBackward),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SegmentMeanBackwardArgs<'_, T>) -> Result<()> {
        if args.d_output.shape != [self.desc.num_segments, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentMeanBackwardPlan: d_output shape != [num_segments, D]",
            ));
        }
        if args.segment_ids.shape != [self.desc.num_inputs] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentMeanBackwardPlan: segment_ids shape != [num_inputs]",
            ));
        }
        if args.d_input.shape != [self.desc.num_inputs, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentMeanBackwardPlan: d_input shape != [num_inputs, D]",
            ));
        }
        Ok(())
    }

    /// Workspace size — `num_segments * sizeof(i32)` bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        (self.desc.num_segments as usize).saturating_mul(core::mem::size_of::<i32>())
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
        workspace: Workspace<'_>,
        args: SegmentMeanBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.num_inputs as i64) * (self.desc.embedding_dim as i64);
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

        let do_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let id_ptr = args.segment_ids.data.as_raw().0 as *const c_void;
        let di_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_segment_mean_backward_f32_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    do_ptr,
                    id_ptr,
                    di_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_segment_mean_backward_f64_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    do_ptr,
                    id_ptr,
                    di_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SegmentMeanBackwardPlan::run reached an unimplemented dtype",
                ))
            }
        };
        map_status(status)
    }
}
