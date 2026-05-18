//! `dequantize_per_channel` forward plan.
//!
//! `x[..., c, ...] = scale[c] * (q[..., c, ...] - zero_point[c])`. Linear
//! per-axis-slice; exact inverse (up to rounding) of
//! [`super::QuantizePerChannelPlan`].

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

/// Descriptor for a `dequantize_per_channel` op.
#[derive(Copy, Clone, Debug)]
pub struct DequantizePerChannelDescriptor {
    /// 4-D shape.
    pub shape: [i32; MAX_RANK],
    /// Logical rank.
    pub rank: u8,
    /// Axis index.
    pub axis: u8,
    /// Output FP element kind (matches FW input).
    pub input_element: ElementKind,
    /// Input int element kind (s8 or u8, matches FW output).
    pub output_element: ElementKind,
}

/// Args bundle for a `dequantize_per_channel` launch.
pub struct DequantizePerChannelArgs<'a, TIn: Element, TOut: IntElement> {
    /// Input int tensor `[D0, D1, D2, D3]`.
    pub input: TensorRef<'a, TOut, 4>,
    /// Per-channel scale `[C]` in FP.
    pub scale: TensorRef<'a, TIn, 1>,
    /// Per-channel zero point `[C]` in i32.
    pub zero_point: TensorRef<'a, i32, 1>,
    /// Output FP tensor `[D0, D1, D2, D3]`.
    pub output: TensorMut<'a, TIn, 4>,
}

/// `dequantize_per_channel` plan.
///
/// `x[..., c, ...] = scale[c] * (q[..., c, ...] - zero_point[c])`.
/// Exactly invertible (up to FW rounding) against
/// [`QuantizePerChannelPlan`](crate::QuantizePerChannelPlan).
///
/// **When to use**: FP recovery from a per-channel-quantized weight
/// buffer. Pair with [`DequantizePerChannelBackwardPlan`](crate::DequantizePerChannelBackwardPlan).
///
/// **Dtypes**: input int `{s8, u8}`; output FP `{f32, f64, f16, bf16}`.
///
/// **Shape limits**: rank-4 contiguous; `axis ∈ [0, 4)`; per-channel
/// vectors of length `shape[axis]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct DequantizePerChannelPlan<TIn: Element, TOut: IntElement> {
    desc: DequantizePerChannelDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DequantizePerChannelPlan<TIn, TOut> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &DequantizePerChannelDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "DequantizePerChannelPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "DequantizePerChannelPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(TIn::KIND, "DequantizePerChannelPlan: unsupported TIn dtype")?;
        validate_output_element(
            TOut::KIND,
            "DequantizePerChannelPlan: unsupported TOut dtype",
        )?;
        if (desc.axis as usize) >= MAX_RANK {
            return Err(Error::InvalidProblem(
                "DequantizePerChannelPlan: axis out of range",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::DequantizePerChannel);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &DequantizePerChannelArgs<'_, TIn, TOut>) -> Result<()> {
        if args.input.shape != self.desc.shape || args.output.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "DequantizePerChannelPlan: tensor shape mismatch",
            ));
        }
        let c = self.desc.shape[self.desc.axis as usize];
        if args.scale.shape != [c] || args.zero_point.shape != [c] {
            return Err(Error::InvalidProblem(
                "DequantizePerChannelPlan: scale / zero_point shape != [C]",
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
        args: DequantizePerChannelArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.output.numel();
        if numel == 0 {
            return Ok(());
        }
        let q_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let zp_ptr = args.zero_point.data.as_raw().0 as *const c_void;
        let x_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let shape4 = self.desc.shape.as_ptr();
        let axis = self.desc.axis as i32;

        let status = match (TIn::KIND, TOut::KIND) {
            (ElementKind::F32, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_f32_s8_run(
                    numel, shape4, axis, q_ptr, sc_ptr, zp_ptr, x_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_f32_u8_run(
                    numel, shape4, axis, q_ptr, sc_ptr, zp_ptr, x_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_f16_s8_run(
                    numel, shape4, axis, q_ptr, sc_ptr, zp_ptr, x_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_f16_u8_run(
                    numel, shape4, axis, q_ptr, sc_ptr, zp_ptr, x_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_bf16_s8_run(
                    numel, shape4, axis, q_ptr, sc_ptr, zp_ptr, x_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_bf16_u8_run(
                    numel, shape4, axis, q_ptr, sc_ptr, zp_ptr, x_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_f64_s8_run(
                    numel, shape4, axis, q_ptr, sc_ptr, zp_ptr, x_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_channel_f64_u8_run(
                    numel, shape4, axis, q_ptr, sc_ptr, zp_ptr, x_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "DequantizePerChannelPlan: unsupported (TIn, TOut) at run()",
            )),
        };
        map_status(status)
    }
}
