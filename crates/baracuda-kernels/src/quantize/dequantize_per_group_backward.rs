//! `dequantize_per_group` backward plan — straight-through.
//!
//! `dq[i, j] = dy[i, j] * scale[i, g_idx]` where `g_idx = j / group_size`.

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
use super::validate_input_element;

/// Descriptor for `dequantize_per_group` backward.
#[derive(Copy, Clone, Debug)]
pub struct DequantizePerGroupBackwardDescriptor {
    /// Flattened-prefix size (same as FW).
    pub outer_size: i32,
    /// Quant axis length.
    pub axis_size: i32,
    /// Group size.
    pub group_size: i32,
}

impl DequantizePerGroupBackwardDescriptor {
    /// Number of groups.
    #[inline]
    pub fn num_groups(&self) -> i32 {
        if self.group_size <= 0 {
            0
        } else {
            self.axis_size / self.group_size
        }
    }
}

/// Args for the dequant-per-group BW launch.
///
/// `TOut` is a phantom mirroring [`super::DequantizePerGroupPlan`]'s
/// `(TIn, TOut)` signature so a caller can carry the FW tuple straight
/// through to the BW plan type.
pub struct DequantizePerGroupBackwardArgs<'a, TIn: Element, TOut: IntElement> {
    /// Upstream gradient `[outer, axis_size]`.
    pub d_output: TensorRef<'a, TIn, 2>,
    /// Saved scale `[outer, num_groups]`.
    pub scale: TensorRef<'a, TIn, 2>,
    /// Output gradient `[outer, axis_size]`.
    pub d_input: TensorMut<'a, TIn, 2>,
    /// Phantom for the int output dtype carried by the plan type
    /// parameter.
    pub _phantom: PhantomData<TOut>,
}

/// `dequantize_per_group` backward plan.
pub struct DequantizePerGroupBackwardPlan<TIn: Element, TOut: IntElement> {
    desc: DequantizePerGroupBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DequantizePerGroupBackwardPlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &DequantizePerGroupBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_input_element(
            TIn::KIND,
            "DequantizePerGroupBackwardPlan: unsupported TIn dtype",
        )?;
        if !matches!(TOut::KIND, ElementKind::S8 | ElementKind::U8) {
            return Err(Error::Unsupported(
                "DequantizePerGroupBackwardPlan: TOut must be S8 or U8",
            ));
        }
        if desc.outer_size < 0 || desc.axis_size < 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupBackwardPlan: outer_size and axis_size must be non-negative",
            ));
        }
        if desc.group_size <= 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupBackwardPlan: group_size must be > 0",
            ));
        }
        if desc.axis_size % desc.group_size != 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupBackwardPlan: axis_size must be a multiple of group_size",
            ));
        }
        let sku = build_sku_group::<TIn, TOut>(QuantizeKind::DequantizePerGroupBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(
        &self,
        args: &DequantizePerGroupBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        let expect_io = [self.desc.outer_size, self.desc.axis_size];
        if args.d_output.shape != expect_io || args.d_input.shape != expect_io {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupBackwardPlan: I/O tensor shape != [outer, axis_size]",
            ));
        }
        let expect_sg = [self.desc.outer_size, self.desc.num_groups()];
        if args.scale.shape != expect_sg {
            return Err(Error::InvalidProblem(
                "DequantizePerGroupBackwardPlan: scale shape != [outer, num_groups]",
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
        args: DequantizePerGroupBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.outer_size as i64) * (self.desc.axis_size as i64);
        if total == 0 {
            return Ok(());
        }
        let dy_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let dx_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let (outer, axis, g) = (
            self.desc.outer_size,
            self.desc.axis_size,
            self.desc.group_size,
        );

        let status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_backward_f32_run(
                    outer, axis, g, dy_ptr, sc_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_backward_f64_run(
                    outer, axis, g, dy_ptr, sc_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_backward_f16_run(
                    outer, axis, g, dy_ptr, sc_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_group_backward_bf16_run(
                    outer, axis, g, dy_ptr, sc_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "DequantizePerGroupBackwardPlan::run unsupported TIn dtype",
                ))
            }
        };
        map_status(status)
    }
}
