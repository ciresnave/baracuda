//! `quantize_per_token` backward plan (Straight-Through Estimator).
//!
//! `dx[n, d] = (dy[n, d] / scale[n]) * 1[qmin < round(x/s)+zp < qmax]`.
//! The in-range mask is recomputed in the kernel from the saved input
//! tensor — no separate "mask" output from FW.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::per_token::build_sku;
use super::validate_input_element;

/// Descriptor for a `quantize_per_token` backward op.
#[derive(Copy, Clone, Debug)]
pub struct QuantizePerTokenBackwardDescriptor {
    /// Number of token rows.
    pub n: i32,
    /// Feature dim.
    pub d: i32,
    /// Lower clip bound (FW's qmin).
    pub q_min: i32,
    /// Upper clip bound (FW's qmax).
    pub q_max: i32,
    /// Input FP element kind.
    pub input_element: ElementKind,
}

/// Args bundle for the per-token BW launch.
pub struct QuantizePerTokenBackwardArgs<'a, TIn: Element> {
    /// Upstream gradient `[N, D]`.
    pub d_output: TensorRef<'a, TIn, 2>,
    /// Saved input from FW (needed for the in-range mask) `[N, D]`.
    pub input: TensorRef<'a, TIn, 2>,
    /// Saved scale `[N]`.
    pub scale: TensorRef<'a, TIn, 1>,
    /// Saved zero-point `[N]`.
    pub zero_point: TensorRef<'a, i32, 1>,
    /// Output `dx` `[N, D]`.
    pub d_input: TensorMut<'a, TIn, 2>,
}

/// `quantize_per_token` backward plan.
///
/// STE: `dx[n, d] = (dy[n, d] / scale[n]) * 1[qmin ≤ round(x[n,d]/scale[n])+zp[n] ≤ qmax]`.
/// Mask recomputed in-kernel.
///
/// **When to use**: backward for
/// [`QuantizePerTokenPlan`](crate::QuantizePerTokenPlan). Caller
/// retains FW input, scale, zero_point.
///
/// **Dtypes**: gradients in `{f32, f64, f16, bf16}`; no int output —
/// hence the single-type-parameter signature.
///
/// **Shape limits**: rank-2 `[N, D]`; per-row `scale` and `zp` of
/// length `N`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct QuantizePerTokenBackwardPlan<TIn: Element> {
    desc: QuantizePerTokenBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<TIn>,
}

// Phantom 2nd type for the SKU build. We pick S8 as a stand-in for
// `aux_element` slot — the BW kernel is TOut-agnostic (no integer
// storage is touched), but the SKU expects a concrete tag.
impl<TIn: Element> QuantizePerTokenBackwardPlan<TIn> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizePerTokenBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizePerTokenBackwardPlan: descriptor input_element != type parameter TIn",
            ));
        }
        validate_input_element(
            TIn::KIND,
            "QuantizePerTokenBackwardPlan: unsupported TIn dtype",
        )?;
        if desc.n < 0 || desc.d < 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenBackwardPlan: n and d must be non-negative",
            ));
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenBackwardPlan: q_max < q_min",
            ));
        }
        // SKU's aux_element slot reflects "the output int kind FW would
        // have used". BW doesn't actually touch int storage, but
        // selectors / telemetry treat the BW SKU as related to its FW
        // peer — we publish S8 as the default.
        let sku = build_sku::<TIn, baracuda_kernels_types::S8>(QuantizeKind::PerTokenBackward);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &QuantizePerTokenBackwardArgs<'_, TIn>) -> Result<()> {
        let expect = [self.desc.n, self.desc.d];
        if args.d_output.shape != expect
            || args.input.shape != expect
            || args.d_input.shape != expect
        {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenBackwardPlan: tensor shape != [n, d]",
            ));
        }
        if args.scale.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenBackwardPlan: scale shape != [n]",
            ));
        }
        if args.zero_point.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenBackwardPlan: zero_point shape != [n]",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — none.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel.
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
        args: QuantizePerTokenBackwardArgs<'_, TIn>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.n as i64) * (self.desc.d as i64);
        if total == 0 {
            return Ok(());
        }
        let dy_ptr = args.d_output.data.as_raw().0 as *const c_void;
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let zp_ptr = args.zero_point.data.as_raw().0 as *const c_void;
        let dx_ptr = args.d_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_backward_f32_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    dy_ptr, x_ptr, sc_ptr, zp_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_backward_f64_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    dy_ptr, x_ptr, sc_ptr, zp_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_backward_f16_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    dy_ptr, x_ptr, sc_ptr, zp_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_backward_bf16_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    dy_ptr, x_ptr, sc_ptr, zp_ptr, dx_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "QuantizePerTokenBackwardPlan::run reached unsupported TIn dtype",
                ))
            }
        };
        map_status(status)
    }
}
