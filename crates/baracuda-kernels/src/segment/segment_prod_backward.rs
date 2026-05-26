//! `segment_prod_backward` plan — Category S, sorted variant. Phase 25.
//!
//! Adjoint of [`crate::segment::SegmentProdPlan`]:
//! `d_input[k, d] = d_output[seg, d] * (output[seg, d] / input[k, d])`
//! where `output[seg, d]` is the FW `prod` (caller must save it).
//!
//! Direct division — caller MUST avoid zero-valued inputs in any
//! segment or accept NaN / Inf in the gradient. For workloads where
//! a zero input is possible, a safer-divide path (saved running prods
//! split around the zero) is needed; we ship the simple variant here.
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

/// Descriptor for a `segment_prod_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct SegmentProdBackwardDescriptor {
    /// Number of input rows (matches the FW `num_inputs`).
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for SegmentProdBackwardDescriptor {
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

/// Args bundle for a `segment_prod_backward` launch.
pub struct SegmentProdBackwardArgs<'a, T: Element> {
    /// Upstream gradient `[num_segments, D]`.
    pub d_output: TensorRef<'a, T, 2>,
    /// FW input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// FW output `[num_segments, D]` (the saved `prod`).
    pub output: TensorRef<'a, T, 2>,
    /// Segment ids `[N]`.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Gradient w.r.t. input `[N, D]`. Overwritten by the launch.
    pub d_input: TensorMut<'a, T, 2>,
}

/// `segment_prod_backward` plan (sorted). Phase 25.
///
/// Adjoint of [`crate::SegmentProdPlan`]:
/// `d_input[n, d] = d_output[seg, d] * (output[seg, d] / input[n, d])`.
///
/// **When to use**: BW pass for [`SegmentProdPlan`](crate::SegmentProdPlan).
/// Caller MUST pass the FW `output` (the saved `prod`).
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: `d_output` and `output` `[num_segments, D]`;
/// `input` and `d_input` `[N, D]`; `segment_ids` `[N]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable (single thread
/// per cell, no atomics).
///
/// **Zero-input limitation**: when `input[n, d] == 0`, the gradient
/// is NaN or `±Inf` (direct division). Use a safer-divide path
/// (saved running prods) if zeros are in scope; we ship the simple
/// variant.
pub struct SegmentProdBackwardPlan<T: Element> {
    desc: SegmentProdBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SegmentProdBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SegmentProdBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "SegmentProdBackwardPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_sku::<T>(SegmentKind::SegmentProdBackward),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &SegmentProdBackwardArgs<'_, T>) -> Result<()> {
        if args.d_output.shape != [self.desc.num_segments, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentProdBackwardPlan: d_output shape != [num_segments, D]",
            ));
        }
        if args.input.shape != [self.desc.num_inputs, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentProdBackwardPlan: input shape != [num_inputs, D]",
            ));
        }
        if args.output.shape != [self.desc.num_segments, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentProdBackwardPlan: output shape != [num_segments, D]",
            ));
        }
        if args.segment_ids.shape != [self.desc.num_inputs] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentProdBackwardPlan: segment_ids shape != [num_inputs]",
            ));
        }
        if args.d_input.shape != [self.desc.num_inputs, self.desc.embedding_dim] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SegmentProdBackwardPlan: d_input shape != [num_inputs, D]",
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
        args: SegmentProdBackwardArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.num_inputs as i64) * (self.desc.embedding_dim as i64);
        if total == 0 {
            return Ok(());
        }
        let do_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *const c_void;
        let id_ptr = args.segment_ids.data.as_raw().0 as *const c_void;
        let di_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_segment_prod_backward_f32_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    do_ptr,
                    in_ptr,
                    out_ptr,
                    id_ptr,
                    di_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_segment_prod_backward_f64_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    do_ptr,
                    in_ptr,
                    out_ptr,
                    id_ptr,
                    di_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SegmentProdBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
