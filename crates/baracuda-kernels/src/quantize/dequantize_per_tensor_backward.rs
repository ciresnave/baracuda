//! `dequantize_per_tensor` backward plan — `dq = dy * scale`.
//!
//! Linear identity scaled by `scale`. The integer `q` operand is
//! non-differentiable; the BW returns a gradient in the FW input's FP
//! dtype (the gradient continues to flow in FP space upstream of the
//! dequant boundary).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, IntElement, KernelSku, PlanPreference, PrecisionGuarantee, QuantizeKind,
    ScalarType, TensorMut, TensorRef, Workspace,
};

use super::map_status;
use super::per_tensor::build_sku;
use super::{validate_input_element, validate_output_element};

/// Descriptor for a `dequantize_per_tensor` backward op.
#[derive(Copy, Clone, Debug)]
pub struct DequantizePerTensorBackwardDescriptor {
    /// Total element count.
    pub numel: i32,
    /// FP element kind (input gradient dtype).
    pub input_element: ElementKind,
    /// FW's int input element kind (s8 or u8).
    pub output_element: ElementKind,
}

/// Args bundle for a `dequantize_per_tensor` backward launch.
pub struct DequantizePerTensorBackwardArgs<'a, TIn: Element, TOut: IntElement> {
    /// Scalar scale (same value used in FW).
    pub scale: <TIn as Element>::Scalar,
    /// Upstream gradient `[numel]` in FP.
    pub d_output: TensorRef<'a, TIn, 1>,
    /// Output `[numel]` in FP (gradient w.r.t. the FW's q-input,
    /// surfaced as FP — same dtype as `d_output`).
    pub d_input: TensorMut<'a, TIn, 1>,
    /// Phantom for the int input dtype carried by the plan type
    /// parameter (parity with FW Plan signature).
    pub _phantom: PhantomData<TOut>,
}

/// `dequantize_per_tensor` backward plan.
pub struct DequantizePerTensorBackwardPlan<TIn: Element, TOut: IntElement> {
    desc: DequantizePerTensorBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DequantizePerTensorBackwardPlan<TIn, TOut> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &DequantizePerTensorBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "DequantizePerTensorBackwardPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "DequantizePerTensorBackwardPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(
            TIn::KIND,
            "DequantizePerTensorBackwardPlan: unsupported TIn dtype",
        )?;
        validate_output_element(
            TOut::KIND,
            "DequantizePerTensorBackwardPlan: unsupported TOut dtype",
        )?;
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerTensorBackwardPlan: numel must be non-negative",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::DequantizePerTensorBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(
        &self,
        args: &DequantizePerTensorBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        let expected = [self.desc.numel];
        if args.d_output.shape != expected || args.d_input.shape != expected {
            return Err(Error::InvalidProblem(
                "DequantizePerTensorBackwardPlan: tensor shape != [numel]",
            ));
        }
        Ok(())
    }

    /// Workspace bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity.
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
        args: DequantizePerTensorBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let dq_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = if <TIn::Scalar as ScalarType>::IS_F64 {
            let scale_f64 = args.scale.to_f64();
            unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_backward_f64_run(
                    numel, scale_f64, dy_ptr, dq_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            }
        } else {
            let scale_f32 = args.scale.to_f32();
            match TIn::KIND {
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_backward_f32_run(
                        numel, scale_f32, dy_ptr, dq_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_backward_f16_run(
                        numel, scale_f32, dy_ptr, dq_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_backward_bf16_run(
                        numel, scale_f32, dy_ptr, dq_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => return Err(Error::Unsupported(
                    "DequantizePerTensorBackwardPlan: unsupported TIn at run()",
                )),
            }
        };
        map_status(status)
    }
}
