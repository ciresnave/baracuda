//! `dequantize_per_token` backward plan — straight-through
//! (`dq = dy * scale[n]`).
//!
//! Phase 8 Milestone 8.2. The plan is parameterized on both `TIn` (the
//! FP element type the gradient flows in) and `TOut` (the int storage
//! type the FW would have produced). `TOut` is unused by the BW kernel
//! itself — the gradient continues in FP — but lives in the type
//! signature for parity with [`super::DequantizePerTokenPlan`], so a
//! caller can ascribe an autograd node's BW Plan with the same
//! `(TIn, TOut)` tuple it used for FW.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, IntElement, KernelSku, PlanPreference, PrecisionGuarantee, QuantizeKind,
    TensorMut, TensorRef, Workspace,
};

use super::map_status;
use super::per_token::build_sku;
use super::validate_input_element;

/// Descriptor for a `dequantize_per_token` backward op.
#[derive(Copy, Clone, Debug)]
pub struct DequantizePerTokenBackwardDescriptor {
    /// Number of token rows.
    pub n: i32,
    /// Feature dim.
    pub d: i32,
}

/// Args bundle for a `dequantize_per_token` backward launch.
pub struct DequantizePerTokenBackwardArgs<'a, TIn: Element, TOut: IntElement> {
    /// Per-row scale `[N]` in FP. Used to scale dy.
    pub scale: TensorRef<'a, TIn, 1>,
    /// Upstream gradient `[N, D]` in FP.
    pub d_output: TensorRef<'a, TIn, 2>,
    /// Output `[N, D]` in FP — same dtype as `d_output` (the q-input is
    /// integer but the gradient continues in FP).
    pub d_input: TensorMut<'a, TIn, 2>,
    /// Phantom for the int output dtype carried by the plan type
    /// parameter (needed so the plan can be parametric the same way the
    /// sibling FW plan is).
    pub _phantom: PhantomData<TOut>,
}

/// `dequantize_per_token` backward plan.
pub struct DequantizePerTokenBackwardPlan<TIn: Element, TOut: IntElement> {
    desc: DequantizePerTokenBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DequantizePerTokenBackwardPlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &DequantizePerTokenBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_input_element(
            TIn::KIND,
            "DequantizePerTokenBackwardPlan: unsupported TIn dtype",
        )?;
        if !matches!(TOut::KIND, ElementKind::S8 | ElementKind::U8) {
            return Err(Error::Unsupported(
                "DequantizePerTokenBackwardPlan: TOut must be S8 or U8",
            ));
        }
        if desc.n < 0 || desc.d < 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerTokenBackwardPlan: n and d must be non-negative",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::DequantizePerTokenBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(
        &self,
        args: &DequantizePerTokenBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        let expect = [self.desc.n, self.desc.d];
        if args.d_output.shape != expect || args.d_input.shape != expect {
            return Err(Error::InvalidProblem(
                "DequantizePerTokenBackwardPlan: tensor shape != [n, d]",
            ));
        }
        if args.scale.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "DequantizePerTokenBackwardPlan: scale shape != [n]",
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
        args: DequantizePerTokenBackwardArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.n as i64) * (self.desc.d as i64);
        if total == 0 {
            return Ok(());
        }
        let dy_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let dx_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_backward_f32_run(
                    self.desc.n, self.desc.d, dy_ptr, sc_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_backward_f64_run(
                    self.desc.n, self.desc.d, dy_ptr, sc_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_backward_f16_run(
                    self.desc.n, self.desc.d, dy_ptr, sc_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_backward_bf16_run(
                    self.desc.n, self.desc.d, dy_ptr, sc_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "DequantizePerTokenBackwardPlan::run unsupported TIn dtype",
                ))
            }
        };
        map_status(status)
    }
}
