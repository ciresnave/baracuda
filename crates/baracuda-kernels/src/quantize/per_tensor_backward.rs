//! `quantize_per_tensor` backward plan (Straight-Through Estimator).
//!
//! `dx = (dy / scale) * 1[qmin <= round(x/scale)+zp <= qmax]`. The
//! in-range mask is **recomputed in the BW kernel from the saved input
//! tensor `x`** — there is no separate "mask" output from the FW. Callers
//! must retain the original input `x` for the BW pass (which they would
//! do anyway for autograd).
//!
//! Math note: the `1/scale` factor is from differentiating `x/scale` in
//! the FW. It is NOT optional — leaving it out is the single most common
//! STE quant-grad bug.

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

/// Descriptor for a `quantize_per_tensor` backward op. Mirrors the FW
/// descriptor with `numel` / `q_min` / `q_max`. The output dtype field
/// records which int output the FW targeted (needed to keep the kernel
/// SKU's `aux_element` consistent between FW and BW) — though the BW
/// itself produces an FP gradient.
#[derive(Copy, Clone, Debug)]
pub struct QuantizePerTensorBackwardDescriptor {
    /// Total element count.
    pub numel: i32,
    /// Lower clip bound from FW.
    pub q_min: i32,
    /// Upper clip bound from FW.
    pub q_max: i32,
    /// Input FP element kind.
    pub input_element: ElementKind,
    /// FW's output int element kind (s8 or u8). Recorded for SKU
    /// consistency; the BW kernel itself doesn't consume it.
    pub output_element: ElementKind,
}

/// Args bundle for a `quantize_per_tensor` backward launch.
pub struct QuantizePerTensorBackwardArgs<'a, TIn: Element, TOut: IntElement> {
    /// Saved FW input `[numel]` in FP — required for mask recomputation.
    pub input: TensorRef<'a, TIn, 1>,
    /// Scalar scale (FP, same value used in FW).
    pub scale: <TIn as Element>::Scalar,
    /// Scalar zero point (same value used in FW).
    pub zero_point: i32,
    /// Upstream gradient `[numel]` in FP.
    pub d_output: TensorRef<'a, TIn, 1>,
    /// Output `[numel]` in FP — same dtype as `d_output`.
    pub d_input: TensorMut<'a, TIn, 1>,
    /// Phantom for the int-output dtype carried by the plan type
    /// parameter (so the BW plan has the same `<TIn, TOut>` shape as the
    /// FW plan, even though the BW kernel doesn't consume an int operand).
    pub _phantom: PhantomData<TOut>,
}

/// `quantize_per_tensor` backward plan.
///
/// Straight-Through Estimator (STE):
/// `dx = (dy / scale) * 1[qmin ≤ round(x/scale)+zp ≤ qmax]`. The
/// in-range mask is recomputed in-kernel from the saved input `x`
/// (no separate mask is saved on FW).
///
/// **When to use**: backward for
/// [`QuantizePerTensorPlan`](crate::QuantizePerTensorPlan). Caller
/// must retain the original input `x` from the FW pass.
///
/// **Dtypes**: gradient `dy` and `dx` in input FP `{f32, f64, f16, bf16}`.
/// `TOut` is the FW output int dtype, carried for SKU consistency
/// only — BW kernel does not consume an int operand.
///
/// **Shape limits**: flat `[numel]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable. The `1/scale`
/// factor is mandatory (omitting it is the most common STE-grad bug).
pub struct QuantizePerTensorBackwardPlan<TIn: Element, TOut: IntElement> {
    desc: QuantizePerTensorBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> QuantizePerTensorBackwardPlan<TIn, TOut> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizePerTensorBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizePerTensorBackwardPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "QuantizePerTensorBackwardPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(
            TIn::KIND,
            "QuantizePerTensorBackwardPlan: unsupported TIn dtype",
        )?;
        validate_output_element(
            TOut::KIND,
            "QuantizePerTensorBackwardPlan: unsupported TOut dtype",
        )?;
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerTensorBackwardPlan: numel must be non-negative",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::PerTensorBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(
        &self,
        args: &QuantizePerTensorBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        let expected = [self.desc.numel];
        if args.input.shape != expected
            || args.d_output.shape != expected
            || args.d_input.shape != expected
        {
            return Err(Error::InvalidProblem(
                "QuantizePerTensorBackwardPlan: tensor shape != [numel]",
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
        args: QuantizePerTensorBackwardArgs<'_, TIn, TOut>,
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
                baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_backward_f64_run(
                    numel, scale_f64, zp, qmin, qmax,
                    x_ptr, dy_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            }
        } else {
            let scale_f32 = args.scale.to_f32();
            match TIn::KIND {
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_backward_f32_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, dy_ptr, dx_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_backward_f16_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, dy_ptr, dx_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_quantize_per_tensor_backward_bf16_run(
                        numel, scale_f32, zp, qmin, qmax,
                        x_ptr, dy_ptr, dx_ptr,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                _ => return Err(Error::Unsupported(
                    "QuantizePerTensorBackwardPlan: unsupported TIn at run()",
                )),
            }
        };
        map_status(status)
    }
}
