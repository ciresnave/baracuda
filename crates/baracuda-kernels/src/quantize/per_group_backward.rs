//! `quantize_per_group` backward plan (Straight-Through Estimator).
//!
//! For index `j` along the quant axis, group index is `g = j / group_size`:
//! `dx[i, j] = (dy[i, j] / scale[i, g]) * 1[in-range]`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::per_group::build_sku_group;
use super::validate_input_element;

/// Descriptor for the per-group BW.
#[derive(Copy, Clone, Debug)]
pub struct QuantizePerGroupBackwardDescriptor {
    /// Flattened-prefix size (same as FW).
    pub outer_size: i32,
    /// Quant axis length (same as FW).
    pub axis_size: i32,
    /// Group size (same as FW).
    pub group_size: i32,
    /// FW's qmin.
    pub q_min: i32,
    /// FW's qmax.
    pub q_max: i32,
    /// Input FP element kind.
    pub input_element: ElementKind,
}

impl QuantizePerGroupBackwardDescriptor {
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

/// Args for the per-group BW launch.
pub struct QuantizePerGroupBackwardArgs<'a, TIn: Element> {
    /// Upstream gradient `[outer, axis_size]`.
    pub d_output: TensorRef<'a, TIn, 2>,
    /// Saved input from FW `[outer, axis_size]`.
    pub input: TensorRef<'a, TIn, 2>,
    /// Saved scale `[outer, num_groups]`.
    pub scale: TensorRef<'a, TIn, 2>,
    /// Saved zero-point `[outer, num_groups]`.
    pub zero_point: TensorRef<'a, i32, 2>,
    /// Output `dx` `[outer, axis_size]`.
    pub d_input: TensorMut<'a, TIn, 2>,
}

/// `quantize_per_group` backward plan.
pub struct QuantizePerGroupBackwardPlan<TIn: Element> {
    desc: QuantizePerGroupBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<TIn>,
}

impl<TIn: Element> QuantizePerGroupBackwardPlan<TIn> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizePerGroupBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizePerGroupBackwardPlan: descriptor input_element != TIn",
            ));
        }
        validate_input_element(
            TIn::KIND,
            "QuantizePerGroupBackwardPlan: unsupported TIn dtype",
        )?;
        if desc.outer_size < 0 || desc.axis_size < 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupBackwardPlan: outer_size and axis_size must be non-negative",
            ));
        }
        if desc.group_size <= 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupBackwardPlan: group_size must be > 0",
            ));
        }
        if desc.axis_size % desc.group_size != 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupBackwardPlan: axis_size must be a multiple of group_size",
            ));
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupBackwardPlan: q_max < q_min",
            ));
        }
        let sku =
            build_sku_group::<TIn, baracuda_kernels_types::S8>(QuantizeKind::PerGroupBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &QuantizePerGroupBackwardArgs<'_, TIn>) -> Result<()> {
        let expect_io = [self.desc.outer_size, self.desc.axis_size];
        if args.d_output.shape != expect_io
            || args.input.shape != expect_io
            || args.d_input.shape != expect_io
        {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupBackwardPlan: I/O tensor shape != [outer, axis_size]",
            ));
        }
        let expect_sg = [self.desc.outer_size, self.desc.num_groups()];
        if args.scale.shape != expect_sg || args.zero_point.shape != expect_sg {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupBackwardPlan: scale / zp shape != [outer, num_groups]",
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
        args: QuantizePerGroupBackwardArgs<'_, TIn>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.outer_size as i64) * (self.desc.axis_size as i64);
        if total == 0 {
            return Ok(());
        }
        let dy_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let zp_ptr = args.zero_point.data.as_raw().0 as *const c_void;
        let dx_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let (outer, axis, g, qmin, qmax) = (
            self.desc.outer_size,
            self.desc.axis_size,
            self.desc.group_size,
            self.desc.q_min,
            self.desc.q_max,
        );

        let status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_backward_f32_run(
                    outer, axis, g, qmin, qmax,
                    dy_ptr, x_ptr, sc_ptr, zp_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_backward_f64_run(
                    outer, axis, g, qmin, qmax,
                    dy_ptr, x_ptr, sc_ptr, zp_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_backward_f16_run(
                    outer, axis, g, qmin, qmax,
                    dy_ptr, x_ptr, sc_ptr, zp_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_backward_bf16_run(
                    outer, axis, g, qmin, qmax,
                    dy_ptr, x_ptr, sc_ptr, zp_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "QuantizePerGroupBackwardPlan::run unsupported TIn dtype",
                ))
            }
        };
        map_status(status)
    }
}
