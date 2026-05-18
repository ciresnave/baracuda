//! `dequantize_per_channel` backward plan — `dq[i] = dy[i] * scale[c]`.

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

/// Descriptor for a `dequantize_per_channel` backward op.
#[derive(Copy, Clone, Debug)]
pub struct DequantizePerChannelBackwardDescriptor {
    /// 4-D shape.
    pub shape: [i32; MAX_RANK],
    /// Logical rank.
    pub rank: u8,
    /// Axis index.
    pub axis: u8,
    /// FP element kind.
    pub input_element: ElementKind,
    /// FW's int input element kind.
    pub output_element: ElementKind,
}

/// Args bundle for a `dequantize_per_channel` backward launch.
pub struct DequantizePerChannelBackwardArgs<'a, TIn: Element, TOut: IntElement> {
    /// Per-channel scale `[C]` (same values used in FW).
    pub scale: TensorRef<'a, TIn, 1>,
    /// Upstream gradient `[D0, D1, D2, D3]` in FP.
    pub d_output: TensorRef<'a, TIn, 4>,
    /// Output `[D0, D1, D2, D3]` in FP.
    pub d_input: TensorMut<'a, TIn, 4>,
    /// Phantom for the int dtype.
    pub _phantom: PhantomData<TOut>,
}

/// `dequantize_per_channel` backward plan.
///
/// Straight-through linear: `dq_FP[..., c, ...] = scale[c] * dy[..., c, ...]`.
///
/// **When to use**: backward for
/// [`DequantizePerChannelPlan`](crate::DequantizePerChannelPlan).
///
/// **Dtypes**: gradients in `{f32, f64, f16, bf16}`.
///
/// **Shape limits**: rank-4 contiguous; `axis ∈ [0, 4)`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct DequantizePerChannelBackwardPlan<TIn: Element, TOut: IntElement> {
    desc: DequantizePerChannelBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DequantizePerChannelBackwardPlan<TIn, TOut> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &DequantizePerChannelBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "DequantizePerChannelBackwardPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "DequantizePerChannelBackwardPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(
            TIn::KIND,
            "DequantizePerChannelBackwardPlan: unsupported TIn dtype",
        )?;
        validate_output_element(
            TOut::KIND,
            "DequantizePerChannelBackwardPlan: unsupported TOut dtype",
        )?;
        if (desc.axis as usize) >= MAX_RANK {
            return Err(Error::InvalidProblem(
                "DequantizePerChannelBackwardPlan: axis out of range",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::DequantizePerChannelBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(
        &self,
        args: &DequantizePerChannelBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        if args.d_output.shape != self.desc.shape || args.d_input.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "DequantizePerChannelBackwardPlan: tensor shape mismatch",
            ));
        }
        let c = self.desc.shape[self.desc.axis as usize];
        if args.scale.shape != [c] {
            return Err(Error::InvalidProblem(
                "DequantizePerChannelBackwardPlan: scale shape != [C]",
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
        args: DequantizePerChannelBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.d_output.numel();
        if numel == 0 {
            return Ok(());
        }
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let dy_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let dq_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let shape4 = self.desc.shape.as_ptr();
        let axis = self.desc.axis as i32;

        let status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_backward_f32_run(
                    numel, shape4, axis, sc_ptr, dy_ptr, dq_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_backward_f16_run(
                    numel, shape4, axis, sc_ptr, dy_ptr, dq_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_backward_bf16_run(
                    numel, shape4, axis, sc_ptr, dy_ptr, dq_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_backward_f64_run(
                    numel, shape4, axis, sc_ptr, dy_ptr, dq_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "DequantizePerChannelBackwardPlan: unsupported TIn at run()",
            )),
        };
        map_status(status)
    }
}
