//! `unsorted_segment_max` plan — Category S, unsorted variant.
//!
//! `out[s, d] = max_{n : segment_ids[n] == s} input[n, d]`. Output is
//! pre-initialized to `-∞` by the launcher; then atomic-max-via-CAS.
//!
//! FW only. BW deferred (argmax tracking).

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

/// Descriptor for an `unsorted_segment_max` op.
#[derive(Copy, Clone, Debug)]
pub struct UnsortedSegmentMaxDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for UnsortedSegmentMaxDescriptor {
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

/// Args bundle for an `unsorted_segment_max` launch.
pub struct UnsortedSegmentMaxArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]`, any order.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`.
    pub output: TensorMut<'a, T, 2>,
}

/// `unsorted_segment_max` plan.
pub struct UnsortedSegmentMaxPlan<T: Element> {
    desc: UnsortedSegmentMaxDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> UnsortedSegmentMaxPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &UnsortedSegmentMaxDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "UnsortedSegmentMaxPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_unsorted_sku::<T>(SegmentKind::UnsortedSegmentMax),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UnsortedSegmentMaxArgs<'_, T>) -> Result<()> {
        validate_unsorted_args(
            self.desc.num_inputs,
            self.desc.embedding_dim,
            self.desc.num_segments,
            args.input.shape,
            args.segment_ids.shape,
            args.output.shape,
            "UnsortedSegmentMaxPlan",
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
        args: UnsortedSegmentMaxArgs<'_, T>,
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
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_max_f32_run(
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
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_max_f64_run(
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
                    "baracuda-kernels::UnsortedSegmentMaxPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
