//! `fake_quantize` backward plan via STE.
//!
//! `dx = dy * 1[qmin <= round(x/scale)+zp <= qmax]`. **No `1/scale`
//! factor** — the FW's dequant-side multiply by `scale` exactly cancels
//! the STE's `1/scale`. This is the key difference from
//! [`super::QuantizePerTensorBackwardPlan`], which DOES include `1/scale`.
//!
//! The in-range mask is recomputed in the kernel from the saved input
//! `x`. Callers must retain `x` from the FW pass.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, QuantizeKind, ScalarType,
    TensorMut, TensorRef, Workspace,
};

use super::fake_quantize::build_sku;
use super::{map_status, validate_input_element};

/// Descriptor for a `fake_quantize` backward op.
#[derive(Copy, Clone, Debug)]
pub struct FakeQuantizeBackwardDescriptor {
    /// Total element count.
    pub numel: i32,
    /// Lower clip bound from FW.
    pub q_min: i32,
    /// Upper clip bound from FW.
    pub q_max: i32,
    /// Input FP element kind.
    pub input_element: ElementKind,
}

/// Args bundle for a `fake_quantize` backward launch.
pub struct FakeQuantizeBackwardArgs<'a, TIn: Element> {
    /// Saved FW input `[numel]` — required for mask recomputation.
    pub input: TensorRef<'a, TIn, 1>,
    /// Scalar scale (same value used in FW).
    pub scale: <TIn as Element>::Scalar,
    /// Scalar zero point (same value used in FW).
    pub zero_point: i32,
    /// Upstream gradient `[numel]` in FP.
    pub d_output: TensorRef<'a, TIn, 1>,
    /// Output `[numel]` in FP.
    pub d_input: TensorMut<'a, TIn, 1>,
}

/// `fake_quantize` backward plan.
pub struct FakeQuantizeBackwardPlan<TIn: Element> {
    desc: FakeQuantizeBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<TIn>,
}

impl<TIn: Element> FakeQuantizeBackwardPlan<TIn> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &FakeQuantizeBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "FakeQuantizeBackwardPlan: descriptor input_element != TIn",
            ));
        }
        validate_input_element(TIn::KIND, "FakeQuantizeBackwardPlan: unsupported TIn dtype")?;
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "FakeQuantizeBackwardPlan: numel must be non-negative",
            ));
        }
        let sku = build_sku::<TIn>(QuantizeKind::FakeQuantizeBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &FakeQuantizeBackwardArgs<'_, TIn>) -> Result<()> {
        let expected = [self.desc.numel];
        if args.input.shape != expected
            || args.d_output.shape != expected
            || args.d_input.shape != expected
        {
            return Err(Error::InvalidProblem(
                "FakeQuantizeBackwardPlan: tensor shape != [numel]",
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
        args: FakeQuantizeBackwardArgs<'_, TIn>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let dy_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let dx_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let zp = args.zero_point;
        let qmin = self.desc.q_min;
        let qmax = self.desc.q_max;

        let status = if <TIn::Scalar as ScalarType>::IS_F64 {
            let scale_f64 = args.scale.to_f64();
            unsafe {
                baracuda_kernels_sys::baracuda_kernels_fake_quantize_backward_f64_run(
                    numel, scale_f64, zp, qmin, qmax,
                    x_ptr, dy_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            }
        } else {
            let scale_f32 = args.scale.to_f32();
            match TIn::KIND {
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fake_quantize_backward_f32_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, dy_ptr, dx_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fake_quantize_backward_f16_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, dy_ptr, dx_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_fake_quantize_backward_bf16_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, dy_ptr, dx_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => return Err(Error::Unsupported(
                    "FakeQuantizeBackwardPlan: unsupported TIn at run()",
                )),
            }
        };
        map_status(status)
    }
}
