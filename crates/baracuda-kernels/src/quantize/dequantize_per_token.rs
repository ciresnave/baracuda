//! `dequantize_per_token` forward plan.
//!
//! Per-row dequantization: `y[n, d] = (q[n, d] - zp[n]) * scale[n]`.
//! Exact inverse of [`super::QuantizePerTokenPlan`] (up to FW rounding).

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
use super::{validate_input_element, validate_output_element};

/// Descriptor for a `dequantize_per_token` op.
#[derive(Copy, Clone, Debug)]
pub struct DequantizePerTokenDescriptor {
    /// Number of token rows.
    pub n: i32,
    /// Feature dim.
    pub d: i32,
    /// Output FP element kind (matches `TIn::KIND` since the FP type is
    /// the result here).
    pub input_element: ElementKind,
    /// Input int element kind (s8 or u8). Matches `TOut::KIND`.
    pub output_element: ElementKind,
}

/// Args bundle for the dequant-per-token launch.
///
/// The `TIn` / `TOut` type parameters mirror the FW plan to keep the
/// type vocabulary consistent: `TIn` is the FP type the FW consumed
/// (and the BW + dequant produce), `TOut` is the int storage type the
/// FW produced (and the dequant consumes).
pub struct DequantizePerTokenArgs<'a, TIn: Element, TOut: IntElement> {
    /// Quantized input `[N, D]` in int.
    pub input: TensorRef<'a, TOut, 2>,
    /// Per-row scale `[N]` in FP.
    pub scale: TensorRef<'a, TIn, 1>,
    /// Per-row zero-point `[N]` in i32.
    pub zero_point: TensorRef<'a, i32, 1>,
    /// Output `[N, D]` in FP.
    pub output: TensorMut<'a, TIn, 2>,
}

/// `dequantize_per_token` plan.
pub struct DequantizePerTokenPlan<TIn: Element, TOut: IntElement> {
    desc: DequantizePerTokenDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DequantizePerTokenPlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &DequantizePerTokenDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "DequantizePerTokenPlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "DequantizePerTokenPlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(TIn::KIND, "DequantizePerTokenPlan: unsupported TIn dtype")?;
        validate_output_element(TOut::KIND, "DequantizePerTokenPlan: unsupported TOut dtype")?;
        if desc.n < 0 || desc.d < 0 {
            return Err(Error::InvalidProblem(
                "DequantizePerTokenPlan: n and d must be non-negative",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::DequantizePerToken);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &DequantizePerTokenArgs<'_, TIn, TOut>) -> Result<()> {
        if args.input.shape != [self.desc.n, self.desc.d] {
            return Err(Error::InvalidProblem(
                "DequantizePerTokenPlan: input shape != [n, d]",
            ));
        }
        if args.output.shape != [self.desc.n, self.desc.d] {
            return Err(Error::InvalidProblem(
                "DequantizePerTokenPlan: output shape != [n, d]",
            ));
        }
        if args.scale.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "DequantizePerTokenPlan: scale shape != [n]",
            ));
        }
        if args.zero_point.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "DequantizePerTokenPlan: zero_point shape != [n]",
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
        args: DequantizePerTokenArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.n as i64) * (self.desc.d as i64);
        if total == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let zp_ptr = args.zero_point.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match (TIn::KIND, TOut::KIND) {
            (ElementKind::F32, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_f32_s8_run(
                    self.desc.n, self.desc.d, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_f32_u8_run(
                    self.desc.n, self.desc.d, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_f64_s8_run(
                    self.desc.n, self.desc.d, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_f64_u8_run(
                    self.desc.n, self.desc.d, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_f16_s8_run(
                    self.desc.n, self.desc.d, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_f16_u8_run(
                    self.desc.n, self.desc.d, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_bf16_s8_run(
                    self.desc.n, self.desc.d, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dequantize_per_token_bf16_u8_run(
                    self.desc.n, self.desc.d, in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "DequantizePerTokenPlan::run unsupported (TIn, TOut)",
                ))
            }
        };
        map_status(status)
    }
}
