//! LPPool1d backward (Phase 16.2). Pairs with [`super::LpPool1dPlan`].
//!
//! `y = (Σ_{k ∈ window} |x_k|^p)^(1/p)`
//! `dx_k = dy · |x_k|^(p-1) · sgn(x_k) · y^(1-p)`
//!
//! BW math handles four edge cases:
//! - `y == 0` (all window cells zero): gradient zero (definition).
//! - `x_k == 0` with `p > 1`: contribution exactly zero (|x|^(p-1) = 0).
//! - `x_k == 0` with `p == 1`: contribution zero (sgn(0) = 0).
//! - `x_k == 0` with `p < 1`: |x|^(p-1) is +inf — kernel clamps to zero
//!   per PyTorch convention.
//!
//! Implementation: one thread per output cell + `atomicAdd` scatter
//! into `dx` (matches the rest of pool BW; half/bf16 route through
//! CAS-loop atomicAdd from `baracuda_atomic.cuh`). **Caller must zero
//! `dx` before launch.**

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::lp_pool1d::{build_lp1d_sku, compute_l_out, validate_lp1d, LpPool1dDescriptor};

/// Args bundle for an `LpPool1d` backward launch.
///
/// Caller is responsible for zeroing `dx` before the launch — the
/// kernel uses `atomicAdd` scatter and does not clear the buffer.
pub struct LpPool1dBwArgs<'a, T: Element> {
    /// Saved forward input `[N, C, L_in]`.
    pub x: TensorRef<'a, T, 3>,
    /// Saved forward output `[N, C, L_out]`.
    pub y: TensorRef<'a, T, 3>,
    /// Upstream gradient `[N, C, L_out]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Output gradient w.r.t. input `[N, C, L_in]`. Caller-zeroed.
    pub dx: TensorMut<'a, T, 3>,
}

/// `LPPool1d` backward plan — bespoke fused BW with atomicAdd scatter.
pub struct LpPool1dBackwardPlan<T: Element> {
    desc: LpPool1dDescriptor,
    l_out: i32,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> LpPool1dBackwardPlan<T> {
    /// Pick a kernel + validate the descriptor (same shape as the FW plan).
    pub fn select(
        _stream: &Stream,
        desc: &LpPool1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_lp1d::<T>(desc)?;
        let l_out = compute_l_out(desc)?;
        let sku = build_lp1d_sku::<T>(PoolKind::LpPool1dBackward);
        Ok(Self {
            desc: *desc,
            l_out,
            sku,
            _marker: PhantomData,
        })
    }

    /// Kernel SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees. atomicAdd scatter → deterministic but not
    /// bit-stable across runs.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Workspace size in bytes. Always `0`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Run the backward pass. Caller must zero `dx` first.
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: LpPool1dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, self.l_out, &args)?;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let ceil_flag = if self.desc.ceil_mode { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_1d_f32_backward_run(
                    x_ptr, y_ptr, dy_ptr, dx_ptr,
                    self.desc.batch, self.desc.channels, self.desc.l_in,
                    self.desc.window, self.desc.stride, self.l_out,
                    self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_1d_f64_backward_run(
                    x_ptr, y_ptr, dy_ptr, dx_ptr,
                    self.desc.batch, self.desc.channels, self.desc.l_in,
                    self.desc.window, self.desc.stride, self.l_out,
                    self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_1d_f16_backward_run(
                    x_ptr, y_ptr, dy_ptr, dx_ptr,
                    self.desc.batch, self.desc.channels, self.desc.l_in,
                    self.desc.window, self.desc.stride, self.l_out,
                    self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_1d_bf16_backward_run(
                    x_ptr, y_ptr, dy_ptr, dx_ptr,
                    self.desc.batch, self.desc.channels, self.desc.l_in,
                    self.desc.window, self.desc.stride, self.l_out,
                    self.desc.p, ceil_flag, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::LpPool1dBackwardPlan: unexpected dtype after select()",
                ));
            }
        };
        super::map_lp_pool_status(status)
    }
}

fn check_bw_args<T: Element>(
    d: &LpPool1dDescriptor,
    l_out: i32,
    args: &LpPool1dBwArgs<'_, T>,
) -> Result<()> {
    let want_x = [d.batch, d.channels, d.l_in];
    let want_y = [d.batch, d.channels, l_out];
    if args.x.shape != want_x {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dBackwardPlan: x shape != [N, C, L_in]",
        ));
    }
    if args.dx.shape != want_x {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dBackwardPlan: dx shape != [N, C, L_in]",
        ));
    }
    if args.y.shape != want_y {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dBackwardPlan: y shape != [N, C, L_out]",
        ));
    }
    if args.dy.shape != want_y {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dBackwardPlan: dy shape != [N, C, L_out]",
        ));
    }
    Ok(())
}
