//! `quantize_per_channel` backward plan via STE.
//!
//! `dx[i] = (dy[i] / scale[c]) * 1[qmin <= round(x[i]/scale[c])+zp[c] <= qmax]`,
//! where `c = coord(i)[axis]`. In-range mask recomputed from saved `x`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, IntElement, KernelSku, PlanPreference, PrecisionGuarantee, QuantizeKind,
    TensorMut, TensorRef, Workspace,
};

use super::map_status;
use super::per_channel::MAX_RANK;
use super::per_tensor::build_sku;
use super::{validate_input_element, validate_output_element};

/// Descriptor for a `quantize_per_channel` backward op.
#[derive(Copy, Clone, Debug)]
pub struct QuantizePerChannelBackwardDescriptor {
    /// 4-D shape (caller pads rank).
    pub shape: [i32; MAX_RANK],
    /// Logical rank.
    pub rank: u8,
    /// Axis index in `[0, 4)`.
    pub axis: u8,
    /// Lower clip bound from FW.
    pub q_min: i32,
    /// Upper clip bound from FW.
    pub q_max: i32,
    /// Input FP element kind.
    pub input_element: ElementKind,
    /// FW output int element kind (s8 or u8) — recorded for SKU parity.
    pub output_element: ElementKind,
}

/// Args bundle for a `quantize_per_channel` backward launch.
pub struct QuantizePerChannelBackwardArgs<'a, TIn: Element, TOut: IntElement> {
    /// Saved FW input `[D0, D1, D2, D3]` in FP — required for mask
    /// recomputation.
    pub input: TensorRef<'a, TIn, 4>,
    /// Per-channel scale `[C]` (same values used in FW).
    pub scale: TensorRef<'a, TIn, 1>,
    /// Per-channel zero point `[C]` (same values used in FW).
    pub zero_point: TensorRef<'a, i32, 1>,
    /// Upstream gradient `[D0, D1, D2, D3]` in FP.
    pub d_output: TensorRef<'a, TIn, 4>,
    /// Output `[D0, D1, D2, D3]` in FP.
    pub d_input: TensorMut<'a, TIn, 4>,
    /// Phantom for the int-output dtype.
    pub _phantom: PhantomData<TOut>,
}

/// `quantize_per_channel` backward plan.
///
/// STE: `dx[..., c, ...] = (dy[..., c, ...] / scale[c]) * 1[qmin ≤ round(x/scale[c])+zp[c] ≤ qmax]`.
/// Mask recomputed in-kernel from the saved FW input.
///
/// **When to use**: backward for
/// [`QuantizePerChannelPlan`](crate::QuantizePerChannelPlan). Caller
/// must retain FW input `x`, `scale[]`, `zero_point[]`.
///
/// **Dtypes**: gradients in `{f32, f64, f16, bf16}`; `TOut` carried
/// for SKU consistency only (BW kernel does not consume an int operand).
///
/// **Shape limits**: rank-4 contiguous; `axis ∈ [0, 4)`; per-channel
/// vectors of length `shape[axis]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct QuantizePerChannelBackwardPlan<TIn: Element, TOut: IntElement> {
    desc: QuantizePerChannelBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> QuantizePerChannelBackwardPlan<TIn, TOut> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizePerChannelBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizePerChannelBackwardPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "QuantizePerChannelBackwardPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(
            TIn::KIND,
            "QuantizePerChannelBackwardPlan: unsupported TIn dtype",
        )?;
        validate_output_element(
            TOut::KIND,
            "QuantizePerChannelBackwardPlan: unsupported TOut dtype",
        )?;
        if (desc.axis as usize) >= MAX_RANK {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelBackwardPlan: axis out of range",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::PerChannelBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(
        &self,
        args: &QuantizePerChannelBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        if args.input.shape != self.desc.shape
            || args.d_output.shape != self.desc.shape
            || args.d_input.shape != self.desc.shape
        {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelBackwardPlan: tensor shape mismatch",
            ));
        }
        let c = self.desc.shape[self.desc.axis as usize];
        if args.scale.shape != [c] || args.zero_point.shape != [c] {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelBackwardPlan: scale / zero_point shape != [C]",
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
        args: QuantizePerChannelBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.d_output.numel();
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let zp_ptr = args.zero_point.data.as_raw().0 as *const c_void;
        let dy_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let dx_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let shape4 = self.desc.shape.as_ptr();
        let axis = self.desc.axis as i32;
        let qmin = self.desc.q_min;
        let qmax = self.desc.q_max;

        let status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_backward_f32_run(
                    numel, shape4, axis, qmin, qmax,
                    x_ptr, sc_ptr, zp_ptr, dy_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_backward_f16_run(
                    numel, shape4, axis, qmin, qmax,
                    x_ptr, sc_ptr, zp_ptr, dy_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_backward_bf16_run(
                    numel, shape4, axis, qmin, qmax,
                    x_ptr, sc_ptr, zp_ptr, dy_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_backward_f64_run(
                    numel, shape4, axis, qmin, qmax,
                    x_ptr, sc_ptr, zp_ptr, dy_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "QuantizePerChannelBackwardPlan: unsupported TIn at run()",
            )),
        };
        map_status(status)
    }
}
