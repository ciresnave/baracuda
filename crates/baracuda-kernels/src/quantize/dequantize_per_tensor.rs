//! `dequantize_per_tensor` forward plan.
//!
//! `x = scale * (q - zero_point)`. Linear; exactly invertible (up to
//! FW rounding) against [`super::QuantizePerTensorPlan`]. Output is FP-
//! typed (`TIn`); the int input is `TOut` (the same int dtype the FW
//! quantized into).

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

/// Descriptor for a `dequantize_per_tensor` op.
#[derive(Copy, Clone, Debug)]
pub struct DequantizePerTensorDescriptor {
    /// Total element count.
    pub numel: i32,
    /// Output FP element kind (same as FW input).
    pub input_element: ElementKind,
    /// Input int element kind (s8 or u8 — the FW output dtype).
    pub output_element: ElementKind,
}

/// Args bundle for a `dequantize_per_tensor` launch.
pub struct DequantizePerTensorArgs<'a, TIn: Element, TOut: IntElement> {
    /// Input int tensor `[numel]`.
    pub input: TensorRef<'a, TOut, 1>,
    /// Scalar scale (FP).
    pub scale: <TIn as Element>::Scalar,
    /// Scalar zero point.
    pub zero_point: i32,
    /// Output FP tensor `[numel]`.
    pub output: TensorMut<'a, TIn, 1>,
}

/// `dequantize_per_tensor` plan.
pub struct DequantizePerTensorPlan<TIn: Element, TOut: IntElement> {
    desc: DequantizePerTensorDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DequantizePerTensorPlan<TIn, TOut> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &DequantizePerTensorDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "DequantizePerTensorPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "DequantizePerTensorPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(TIn::KIND, "DequantizePerTensorPlan: unsupported TIn dtype")?;
        validate_output_element(TOut::KIND, "DequantizePerTensorPlan: unsupported TOut dtype")?;
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerTensorPlan: numel must be non-negative",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::DequantizePerTensor);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &DequantizePerTensorArgs<'_, TIn, TOut>) -> Result<()> {
        let expected = [self.desc.numel];
        if args.input.shape != expected || args.output.shape != expected {
            return Err(Error::InvalidProblem(
                "DequantizePerTensorPlan: tensor shape != [numel]",
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
        args: DequantizePerTensorArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let q_ptr = args.input.data.as_raw().0 as *const c_void;
        let x_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let zp = args.zero_point;

        let status = if <TIn::Scalar as ScalarType>::IS_F64 {
            let scale_f64 = args.scale.to_f64();
            match TOut::KIND {
                ElementKind::S8 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_f64_s8_run(
                        numel, scale_f64, zp, q_ptr, x_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::U8 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_f64_u8_run(
                        numel, scale_f64, zp, q_ptr, x_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => return Err(Error::Unsupported(
                    "DequantizePerTensorPlan: unsupported TOut at run()",
                )),
            }
        } else {
            let scale_f32 = args.scale.to_f32();
            match (TIn::KIND, TOut::KIND) {
                (ElementKind::F32, ElementKind::S8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_f32_s8_run(
                        numel, scale_f32, zp, q_ptr, x_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::F32, ElementKind::U8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_f32_u8_run(
                        numel, scale_f32, zp, q_ptr, x_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::F16, ElementKind::S8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_f16_s8_run(
                        numel, scale_f32, zp, q_ptr, x_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::F16, ElementKind::U8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_f16_u8_run(
                        numel, scale_f32, zp, q_ptr, x_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::Bf16, ElementKind::S8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_bf16_s8_run(
                        numel, scale_f32, zp, q_ptr, x_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::Bf16, ElementKind::U8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_dequantize_per_tensor_bf16_u8_run(
                        numel, scale_f32, zp, q_ptr, x_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => return Err(Error::Unsupported(
                    "DequantizePerTensorPlan: unsupported (TIn, TOut) at run()",
                )),
            }
        };
        map_status(status)
    }
}
