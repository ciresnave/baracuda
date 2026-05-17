//! `fake_quantize` forward plan — per-tensor, FP roundtrip.
//!
//! `y = scale * (clamp(round(x / scale) + zp, q_min, q_max) - zp)`. The
//! roundtrip of `quantize` followed by `dequantize`, executed entirely
//! in FP — produces a lossy FP output of the same dtype as the input.
//! No integer storage involved. PyTorch
//! `torch.fake_quantize_per_tensor_affine`.
//!
//! The descriptor carries the int range (`q_min` / `q_max`) but not an
//! output dtype — the int range is what defines the lossy precision
//! step. Caller picks the int range matching their downstream `qint`
//! storage (`[-128, 127]` for s8, `[0, 255]` for u8) but no `TOut` plan
//! parameter is needed.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, ScalarType, TensorMut, TensorRef, Workspace,
};

use super::{map_status, validate_input_element};

/// Descriptor for a `fake_quantize` forward op.
#[derive(Copy, Clone, Debug)]
pub struct FakeQuantizeDescriptor {
    /// Total element count.
    pub numel: i32,
    /// Lower clip bound.
    pub q_min: i32,
    /// Upper clip bound.
    pub q_max: i32,
    /// Input FP element kind.
    pub input_element: ElementKind,
}

/// Args bundle for a `fake_quantize` forward launch.
pub struct FakeQuantizeArgs<'a, TIn: Element> {
    /// Input FP tensor `[numel]`.
    pub input: TensorRef<'a, TIn, 1>,
    /// Scalar scale (FP).
    pub scale: <TIn as Element>::Scalar,
    /// Scalar zero point.
    pub zero_point: i32,
    /// Output FP tensor `[numel]` — same dtype as input.
    pub output: TensorMut<'a, TIn, 1>,
}

/// `fake_quantize` forward plan.
pub struct FakeQuantizePlan<TIn: Element> {
    desc: FakeQuantizeDescriptor,
    sku: KernelSku,
    _marker: PhantomData<TIn>,
}

impl<TIn: Element> FakeQuantizePlan<TIn> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FakeQuantizeDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "FakeQuantizePlan: descriptor input_element != TIn",
            ));
        }
        validate_input_element(TIn::KIND, "FakeQuantizePlan: unsupported TIn dtype")?;
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "FakeQuantizePlan: numel must be non-negative",
            ));
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem("FakeQuantizePlan: q_max < q_min"));
        }
        let sku = build_sku::<TIn>(QuantizeKind::FakeQuantize);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &FakeQuantizeArgs<'_, TIn>) -> Result<()> {
        let expected = [self.desc.numel];
        if args.input.shape != expected || args.output.shape != expected {
            return Err(Error::InvalidProblem(
                "FakeQuantizePlan: tensor shape != [numel]",
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
        args: FakeQuantizeArgs<'_, TIn>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let y_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let zp = args.zero_point;
        let qmin = self.desc.q_min;
        let qmax = self.desc.q_max;

        let status = if <TIn::Scalar as ScalarType>::IS_F64 {
            let scale_f64 = args.scale.to_f64();
            unsafe {
                baracuda_kernels_sys::baracuda_kernels_fake_quantize_f64_run(
                    numel, scale_f64, zp, qmin, qmax, x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            }
        } else {
            let scale_f32 = args.scale.to_f32();
            match TIn::KIND {
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fake_quantize_f32_run(
                        numel, scale_f32, zp, qmin, qmax, x_ptr, y_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fake_quantize_f16_run(
                        numel, scale_f32, zp, qmin, qmax, x_ptr, y_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fake_quantize_bf16_run(
                        numel, scale_f32, zp, qmin, qmax, x_ptr, y_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => return Err(Error::Unsupported(
                    "FakeQuantizePlan: unsupported TIn at run()",
                )),
            }
        };
        map_status(status)
    }
}

/// Build the [`KernelSku`] for a fake-quantize-family plan. Sibling of
/// [`super::per_tensor::build_sku`]; no TOut surfaces in the SKU because
/// fake_quantize stays in FP space.
pub(crate) fn build_sku<TIn: Element>(op: QuantizeKind) -> KernelSku {
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
        aux_element: None,
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}
