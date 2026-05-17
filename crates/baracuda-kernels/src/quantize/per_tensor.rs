//! `quantize_per_tensor` forward plan — Category P FW trailblazer.
//!
//! `q = clamp(round(x / scale) + zero_point, q_min, q_max)`. One scalar
//! `scale` (FP) and `zero_point` (i32) for the whole tensor. PyTorch
//! `torch.quantize_per_tensor`.
//!
//! ## Trailblazer dtype coverage
//!
//! Input FP × output int:
//! - Input FP `TIn`: `f32, f64, f16, bf16`.
//! - Output int `TOut`: [`baracuda_kernels_types::S8`] (`[-128, 127]`) or
//!   [`baracuda_kernels_types::U8`] (`[0, 255]`).
//!
//! `scale` is carried in the input FP dtype's [`Element::Scalar`]
//! projection — `f32` for the 16/32-bit FP family, `f64` for `f64`. The
//! plan dispatches to a `_f32` or `_f64` FFI flavor based on `TIn`.
//!
//! The trailblazer flattens to a 1-D layout — the caller is expected to
//! collapse a multi-D tensor down to its flat `numel` (per-tensor
//! quantization doesn't depend on axis structure).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IntElement, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, ScalarType, TensorMut, TensorRef, Workspace,
};

use super::{map_status, validate_input_element, validate_output_element};

/// Descriptor for a `quantize_per_tensor` forward op.
#[derive(Copy, Clone, Debug)]
pub struct QuantizePerTensorDescriptor {
    /// Total element count of the input / output tensors.
    pub numel: i32,
    /// Quantization range lower bound (e.g. `-128` for s8, `0` for u8).
    pub q_min: i32,
    /// Quantization range upper bound (e.g. `127` for s8, `255` for u8).
    pub q_max: i32,
    /// Input FP element kind. Must match `TIn::KIND`.
    pub input_element: ElementKind,
    /// Output int element kind (s8 or u8). Must match `TOut::KIND`.
    pub output_element: ElementKind,
}

/// Args bundle for a `quantize_per_tensor` forward launch.
///
/// The input / output tensors are 1-D for the trailblazer — the caller
/// flattens a multi-D tensor down to its `numel` (per-tensor quantization
/// is axis-agnostic). The 1-D shape is `[numel]`.
pub struct QuantizePerTensorArgs<'a, TIn: Element, TOut: IntElement> {
    /// Input tensor in FP. Flat `[numel]`.
    pub input: TensorRef<'a, TIn, 1>,
    /// Scalar scale (FP), passed by value. The plan converts to the
    /// appropriate FFI scalar (`f32` or `f64`) based on `TIn`.
    pub scale: <TIn as Element>::Scalar,
    /// Scalar zero point (i32), passed by value.
    pub zero_point: i32,
    /// Output tensor in int. Flat `[numel]`.
    pub output: TensorMut<'a, TOut, 1>,
}

/// `quantize_per_tensor` forward plan.
pub struct QuantizePerTensorPlan<TIn: Element, TOut: IntElement> {
    desc: QuantizePerTensorDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> QuantizePerTensorPlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizePerTensorDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizePerTensorPlan: descriptor input_element != type parameter TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "QuantizePerTensorPlan: descriptor output_element != type parameter TOut",
            ));
        }
        validate_input_element(TIn::KIND, "QuantizePerTensorPlan: unsupported TIn dtype")?;
        validate_output_element(TOut::KIND, "QuantizePerTensorPlan: unsupported TOut dtype")?;
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerTensorPlan: numel must be non-negative",
            ));
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem(
                "QuantizePerTensorPlan: q_max < q_min",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::PerTensor);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args at run time.
    pub fn can_implement(&self, args: &QuantizePerTensorArgs<'_, TIn, TOut>) -> Result<()> {
        if args.input.shape != [self.desc.numel] {
            return Err(Error::InvalidProblem(
                "QuantizePerTensorPlan: input shape != [numel]",
            ));
        }
        if args.output.shape != [self.desc.numel] {
            return Err(Error::InvalidProblem(
                "QuantizePerTensorPlan: output shape != [numel]",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — none.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the selected kernel.
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
        args: QuantizePerTensorArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let q_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let zp = args.zero_point;
        let qmin = self.desc.q_min;
        let qmax = self.desc.q_max;

        let status = if <TIn::Scalar as ScalarType>::IS_F64 {
            // f64 input — use the f64-scale FFI flavor.
            let scale_f64 = args.scale.to_f64();
            match TOut::KIND {
                ElementKind::S8 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_f64_s8_run(
                        numel, scale_f64, zp, qmin, qmax,
                        x_ptr, q_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::U8 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_f64_u8_run(
                        numel, scale_f64, zp, qmin, qmax,
                        x_ptr, q_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => return Err(Error::Unsupported(
                    "QuantizePerTensorPlan: unsupported TOut at run() (select should have caught)",
                )),
            }
        } else {
            // f32 / f16 / bf16 input — f32 scale flavor.
            let scale_f32 = args.scale.to_f32();
            match (TIn::KIND, TOut::KIND) {
                (ElementKind::F32, ElementKind::S8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_f32_s8_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, q_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::F32, ElementKind::U8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_f32_u8_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, q_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::F16, ElementKind::S8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_f16_s8_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, q_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::F16, ElementKind::U8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_f16_u8_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, q_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::Bf16, ElementKind::S8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_bf16_s8_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, q_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                (ElementKind::Bf16, ElementKind::U8) => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_bf16_u8_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, q_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => return Err(Error::Unsupported(
                    "QuantizePerTensorPlan: unsupported (TIn, TOut) at run()",
                )),
            }
        };
        map_status(status)
    }
}

/// Build the [`KernelSku`] for a per-tensor quantize-family plan.
pub(crate) fn build_sku<TIn: Element, TOut: IntElement>(op: QuantizeKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if TIn::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: ElementKind::F32,
        bit_stable_on_same_hardware: true,
        deterministic: true,
    };
    KernelSku {
        category: OpCategory::Quantization,
        op: op as u16,
        element: TIn::KIND,
        aux_element: Some(TOut::KIND),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}
