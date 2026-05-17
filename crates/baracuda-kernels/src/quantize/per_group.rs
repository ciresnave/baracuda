//! `quantize_per_group` forward plan.
//!
//! Per-group quantization: input is `[outer, axis_size]` (the higher-
//! rank tensor is flattened by the caller before this plan), and the
//! quantization axis (the rightmost dim) is partitioned into contiguous
//! groups of size `group_size`. Each group gets its own `(scale, zp)`,
//! so `scale` and `zero_point` have shape `[outer, num_groups]` where
//! `num_groups = axis_size / group_size`.
//!
//! Used by INT4 LLM weight quantization (GPTQ / AWQ / GGML), typically
//! with `group_size = 128`.
//!
//! Trailblazer scope: quant axis must be the **rightmost** axis so the
//! layout is naturally group-contiguous. Higher-rank tensors with a
//! non-last quant axis require a permute first (caller's
//! responsibility).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IntElement, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace,
};

use super::{map_status, validate_input_element, validate_output_element};

/// Descriptor for a `quantize_per_group` forward op.
#[derive(Copy, Clone, Debug)]
pub struct QuantizePerGroupDescriptor {
    /// Product of all dims except the quant axis (the flattened
    /// non-quant prefix).
    pub outer_size: i32,
    /// Length of the quant axis. Must be `>= 0` and divisible by
    /// `group_size`.
    pub axis_size: i32,
    /// Group size — number of consecutive elements along the quant
    /// axis that share a `(scale, zp)` pair. Typical: `128` for GPTQ
    /// INT4 weights.
    pub group_size: i32,
    /// Quantization range lower bound.
    pub q_min: i32,
    /// Quantization range upper bound.
    pub q_max: i32,
    /// Input FP element kind.
    pub input_element: ElementKind,
    /// Output int element kind.
    pub output_element: ElementKind,
}

impl QuantizePerGroupDescriptor {
    /// Number of groups along the quant axis. Equals
    /// `axis_size / group_size` (validated `axis_size % group_size == 0`).
    #[inline]
    pub fn num_groups(&self) -> i32 {
        if self.group_size <= 0 {
            0
        } else {
            self.axis_size / self.group_size
        }
    }
}

/// Args bundle for a `quantize_per_group` forward launch.
pub struct QuantizePerGroupArgs<'a, TIn: Element, TOut: IntElement> {
    /// Input `[outer_size, axis_size]` in FP.
    pub input: TensorRef<'a, TIn, 2>,
    /// Per-group scale `[outer_size, num_groups]` in FP.
    pub scale: TensorRef<'a, TIn, 2>,
    /// Per-group zero-point `[outer_size, num_groups]` in i32.
    pub zero_point: TensorRef<'a, i32, 2>,
    /// Output `[outer_size, axis_size]` in int.
    pub output: TensorMut<'a, TOut, 2>,
}

/// `quantize_per_group` forward plan.
pub struct QuantizePerGroupPlan<TIn: Element, TOut: IntElement> {
    desc: QuantizePerGroupDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> QuantizePerGroupPlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizePerGroupDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizePerGroupPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "QuantizePerGroupPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(TIn::KIND, "QuantizePerGroupPlan: unsupported TIn dtype")?;
        validate_output_element(TOut::KIND, "QuantizePerGroupPlan: unsupported TOut dtype")?;
        if desc.outer_size < 0 || desc.axis_size < 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupPlan: outer_size and axis_size must be non-negative",
            ));
        }
        if desc.group_size <= 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupPlan: group_size must be > 0",
            ));
        }
        if desc.axis_size % desc.group_size != 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupPlan: axis_size must be a multiple of group_size",
            ));
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupPlan: q_max < q_min",
            ));
        }
        let sku = build_sku_group::<TIn, TOut>(QuantizeKind::PerGroup);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &QuantizePerGroupArgs<'_, TIn, TOut>) -> Result<()> {
        let expect_io = [self.desc.outer_size, self.desc.axis_size];
        if args.input.shape != expect_io || args.output.shape != expect_io {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupPlan: I/O tensor shape != [outer, axis_size]",
            ));
        }
        let expect_sg = [self.desc.outer_size, self.desc.num_groups()];
        if args.scale.shape != expect_sg || args.zero_point.shape != expect_sg {
            return Err(Error::InvalidProblem(
                "QuantizePerGroupPlan: scale / zp shape != [outer, num_groups]",
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
        args: QuantizePerGroupArgs<'_, TIn, TOut>,
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
        let (outer, axis, g, qmin, qmax) = (
            self.desc.outer_size,
            self.desc.axis_size,
            self.desc.group_size,
            self.desc.q_min,
            self.desc.q_max,
        );
        let status = match (TIn::KIND, TOut::KIND) {
            (ElementKind::F32, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_f32_s8_run(
                    outer, axis, g, qmin, qmax, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_f32_u8_run(
                    outer, axis, g, qmin, qmax, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_f64_s8_run(
                    outer, axis, g, qmin, qmax, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_f64_u8_run(
                    outer, axis, g, qmin, qmax, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_f16_s8_run(
                    outer, axis, g, qmin, qmax, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_f16_u8_run(
                    outer, axis, g, qmin, qmax, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_bf16_s8_run(
                    outer, axis, g, qmin, qmax, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_group_bf16_u8_run(
                    outer, axis, g, qmin, qmax, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "QuantizePerGroupPlan::run unsupported (TIn, TOut)",
                ))
            }
        };
        map_status(status)
    }
}

/// Build the [`KernelSku`] for a quantize-per-group-family plan.
pub(crate) fn build_sku_group<TIn: Element, TOut: IntElement>(op: QuantizeKind) -> KernelSku {
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
        aux_element: Some(TOut::KIND),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}
