//! `dequantize_per_group` forward plan.
//!
//! `y[i, j] = (q[i, j] - zp[i, g_idx]) * scale[i, g_idx]` where
//! `g_idx = j / group_size`. Exact inverse of
//! [`super::QuantizePerGroupPlan`] (up to FW rounding).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, IntElement, KernelSku, PlanPreference, PrecisionGuarantee, QuantizeKind,
    TensorMut, TensorRef, Workspace,
};

use super::map_status;
use super::per_group::build_sku_group;
use super::{validate_input_element, validate_output_element};

/// Descriptor for `dequantize_per_group`.
#[derive(Copy, Clone, Debug)]
pub struct DequantizePerGroupDescriptor {
    /// Flattened-prefix size (same as FW).
    pub outer_size: i32,
    /// Quant axis length.
    pub axis_size: i32,
    /// Group size.
    pub group_size: i32,
    /// Output FP element kind.
    pub input_element: ElementKind,
    /// Input int element kind.
    pub output_element: ElementKind,
}

impl DequantizePerGroupDescriptor {
    /// Number of groups along the quant axis.
    #[inline]
    pub fn num_groups(&self) -> i32 {
        if self.group_size <= 0 {
            0
        } else {
            self.axis_size / self.group_size
        }
    }
}

/// Args for the dequant-per-group launch.
pub struct DequantizePerGroupArgs<'a, TIn: Element, TOut: IntElement> {
    /// Quantized input `[outer, axis_size]` in int.
    pub input: TensorRef<'a, TOut, 2>,
    /// Per-group scale `[outer, num_groups]` in FP.
    pub scale: TensorRef<'a, TIn, 2>,
    /// Per-group zero-point `[outer, num_groups]` in i32.
    pub zero_point: TensorRef<'a, i32, 2>,
    /// Output `[outer, axis_size]` in FP.
    pub output: TensorMut<'a, TIn, 2>,
}

/// `dequantize_per_group` plan.
///
/// `x[..., g] = scale[outer, g] * (q[..., g] - zero_point[outer, g])`
/// with `g` the group index along the (rightmost) quant axis.
/// Inverse of [`QuantizePerGroupPlan`](crate::QuantizePerGroupPlan).
///
/// **When to use**: FP recovery from INT4/INT8 grouped weight blobs
/// (GPTQ / AWQ / GGML). Pair with
/// [`DequantizePerGroupBackwardPlan`](crate::DequantizePerGroupBackwardPlan).
///
/// **Dtypes**: input int `{s8, u8}`; output FP `{f32, f64, f16, bf16}`.
///
/// **Shape limits**: rank-2 `[outer, axis_size]` with
/// `axis_size % group_size == 0`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct DequantizePerGroupPlan<TIn: Element, TOut: IntElement> {
    desc: DequantizePerGroupDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DequantizePerGroupPlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &DequantizePerGroupDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "DequantizePerGroupPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "DequantizePerGroupPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(TIn::KIND, "DequantizePerGroupPlan: unsupported TIn dtype")?;
        validate_output_element(TOut::KIND, "DequantizePerGroupPlan: unsupported TOut dtype")?;
        if desc.outer_size < 0 || desc.axis_size < 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupPlan: outer_size and axis_size must be non-negative",
            ));
        }
        if desc.group_size <= 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupPlan: group_size must be > 0",
            ));
        }
        if desc.axis_size % desc.group_size != 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupPlan: axis_size must be a multiple of group_size",
            ));
        }
        let sku = build_sku_group::<TIn, TOut>(QuantizeKind::DequantizePerGroup);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &DequantizePerGroupArgs<'_, TIn, TOut>) -> Result<()> {
        let expect_io = [self.desc.outer_size, self.desc.axis_size];
        if args.input.shape != expect_io || args.output.shape != expect_io {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupPlan: I/O tensor shape != [outer, axis_size]",
            ));
        }
        let expect_sg = [self.desc.outer_size, self.desc.num_groups()];
        if args.scale.shape != expect_sg || args.zero_point.shape != expect_sg {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupPlan: scale / zp shape != [outer, num_groups]",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — none.
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
        args: DequantizePerGroupArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.outer_size as i64) * (self.desc.axis_size as i64);
        if total == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let zp_ptr = args.zero_point.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let (outer, axis, g) = (
            self.desc.outer_size,
            self.desc.axis_size,
            self.desc.group_size,
        );
        let status = match (TIn::KIND, TOut::KIND) {
            (ElementKind::F32, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_f32_s8_run(
                    outer, axis, g, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_f32_u8_run(
                    outer, axis, g, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_f64_s8_run(
                    outer, axis, g, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_f64_u8_run(
                    outer, axis, g, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_f16_s8_run(
                    outer, axis, g, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_f16_u8_run(
                    outer, axis, g, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_bf16_s8_run(
                    outer, axis, g, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_bf16_u8_run(
                    outer, axis, g, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "DequantizePerGroupPlan::run unsupported (TIn, TOut)",
                ))
            }
        };
        map_status(status)
    }
}
