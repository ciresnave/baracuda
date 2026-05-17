//! `quantize_per_channel` forward plan.
//!
//! Per-axis-slice quantization: `q[..., c, ...] = clamp(round(x[..., c, ...]
//! / scale[c]) + zero_point[c], q_min, q_max)`. `scale` and `zero_point`
//! are 1-D tensors of length `C` (the extent of the channel axis). The
//! typical use is per-output-channel weight quantization for conv /
//! linear layers. PyTorch `torch.quantize_per_channel`.
//!
//! Trailblazer layout: the input / output are **rank-4 contiguous** —
//! the caller pads lower-rank tensors to 4 with leading or trailing
//! extents of `1`. `axis` selects which of the 4 dims indexes
//! `scale[]` / `zero_point[]`. Strided per-channel is deferred.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, IntElement, KernelSku, PlanPreference, PrecisionGuarantee, QuantizeKind,
    TensorMut, TensorRef, Workspace,
};

use super::map_status;
use super::per_tensor::build_sku;
use super::{validate_input_element, validate_output_element};

/// Max rank supported by the per-channel kernels (matches `MAX_RANK` in
/// `kernels/include/baracuda_quantize.cuh`).
pub const MAX_RANK: usize = 4;

/// Descriptor for a `quantize_per_channel` forward op.
#[derive(Copy, Clone, Debug)]
pub struct QuantizePerChannelDescriptor {
    /// 4-D shape (caller pads rank with `1`'s).
    pub shape: [i32; MAX_RANK],
    /// Logical rank (used for validation; the kernel always sees rank 4).
    pub rank: u8,
    /// Axis index in `[0, 4)` that indexes the per-channel `scale[]` /
    /// `zero_point[]` vectors.
    pub axis: u8,
    /// Quantization range lower bound.
    pub q_min: i32,
    /// Quantization range upper bound.
    pub q_max: i32,
    /// Input FP element kind.
    pub input_element: ElementKind,
    /// Output int element kind.
    pub output_element: ElementKind,
}

/// Args bundle for a `quantize_per_channel` forward launch.
pub struct QuantizePerChannelArgs<'a, TIn: Element, TOut: IntElement> {
    /// Input `[D0, D1, D2, D3]` in FP.
    pub input: TensorRef<'a, TIn, 4>,
    /// Per-channel scale `[C]` in FP, where `C = shape[axis]`.
    pub scale: TensorRef<'a, TIn, 1>,
    /// Per-channel zero point `[C]` in i32.
    pub zero_point: TensorRef<'a, i32, 1>,
    /// Output `[D0, D1, D2, D3]` in int.
    pub output: TensorMut<'a, TOut, 4>,
}

/// `quantize_per_channel` forward plan.
pub struct QuantizePerChannelPlan<TIn: Element, TOut: IntElement> {
    desc: QuantizePerChannelDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> QuantizePerChannelPlan<TIn, TOut> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizePerChannelDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizePerChannelPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "QuantizePerChannelPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(TIn::KIND, "QuantizePerChannelPlan: unsupported TIn dtype")?;
        validate_output_element(TOut::KIND, "QuantizePerChannelPlan: unsupported TOut dtype")?;
        if (desc.axis as usize) >= MAX_RANK {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelPlan: axis out of range [0, MAX_RANK)",
            ));
        }
        if (desc.rank as usize) == 0 || (desc.rank as usize) > MAX_RANK {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelPlan: rank must be in [1, MAX_RANK]",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "QuantizePerChannelPlan: shape dims must be non-negative",
                ));
            }
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem("QuantizePerChannelPlan: q_max < q_min"));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::PerChannel);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &QuantizePerChannelArgs<'_, TIn, TOut>) -> Result<()> {
        if args.input.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelPlan: input shape mismatch with descriptor",
            ));
        }
        if args.output.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelPlan: output shape mismatch with descriptor",
            ));
        }
        let c = self.desc.shape[self.desc.axis as usize];
        if args.scale.shape != [c] {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelPlan: scale shape != [shape[axis]]",
            ));
        }
        if args.zero_point.shape != [c] {
            return Err(Error::InvalidProblem(
                "QuantizePerChannelPlan: zero_point shape != [shape[axis]]",
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
        args: QuantizePerChannelArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.output.numel();
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let zp_ptr = args.zero_point.data.as_raw().0 as *const c_void;
        let q_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let shape4 = self.desc.shape.as_ptr();
        let axis = self.desc.axis as i32;
        let qmin = self.desc.q_min;
        let qmax = self.desc.q_max;

        let status = match (TIn::KIND, TOut::KIND) {
            (ElementKind::F32, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_f32_s8_run(
                    numel, shape4, axis, qmin, qmax, x_ptr, sc_ptr, zp_ptr, q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_f32_u8_run(
                    numel, shape4, axis, qmin, qmax, x_ptr, sc_ptr, zp_ptr, q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_f16_s8_run(
                    numel, shape4, axis, qmin, qmax, x_ptr, sc_ptr, zp_ptr, q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_f16_u8_run(
                    numel, shape4, axis, qmin, qmax, x_ptr, sc_ptr, zp_ptr, q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_bf16_s8_run(
                    numel, shape4, axis, qmin, qmax, x_ptr, sc_ptr, zp_ptr, q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_bf16_u8_run(
                    numel, shape4, axis, qmin, qmax, x_ptr, sc_ptr, zp_ptr, q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_f64_s8_run(
                    numel, shape4, axis, qmin, qmax, x_ptr, sc_ptr, zp_ptr, q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_channel_f64_u8_run(
                    numel, shape4, axis, qmin, qmax, x_ptr, sc_ptr, zp_ptr, q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "QuantizePerChannelPlan: unsupported (TIn, TOut) at run()",
            )),
        };
        map_status(status)
    }
}
