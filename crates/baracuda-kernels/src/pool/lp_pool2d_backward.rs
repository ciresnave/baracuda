//! LPPool2d backward (Phase 16.2). Pairs with [`super::LpPool2dPlan`].
//!
//! Same gradient math as the 1d sibling — see
//! [`super::lp_pool1d_backward`] for the derivation and edge-case
//! handling.
//!
//! Implementation: one thread per output cell + `atomicAdd` scatter
//! into `dx`. **Caller must zero `dx` before launch.**

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::lp_pool2d::{build_lp2d_sku, compute_out_2d, validate_lp2d, LpPool2dDescriptor};

/// Args bundle for an `LpPool2d` backward launch.
///
/// Caller is responsible for zeroing `dx` before the launch.
pub struct LpPool2dBwArgs<'a, T: Element> {
    /// Saved forward input `[N, C, H_in, W_in]`.
    pub x: TensorRef<'a, T, 4>,
    /// Saved forward output `[N, C, H_out, W_out]`.
    pub y: TensorRef<'a, T, 4>,
    /// Upstream gradient `[N, C, H_out, W_out]`.
    pub dy: TensorRef<'a, T, 4>,
    /// Output gradient w.r.t. input `[N, C, H_in, W_in]`. Caller-zeroed.
    pub dx: TensorMut<'a, T, 4>,
}

/// `LPPool2d` backward plan — bespoke fused BW with atomicAdd scatter.
pub struct LpPool2dBackwardPlan<T: Element> {
    desc: LpPool2dDescriptor,
    h_out: i32,
    w_out: i32,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> LpPool2dBackwardPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &LpPool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_lp2d::<T>(desc)?;
        let (h_out, w_out) = compute_out_2d(desc)?;
        let sku = build_lp2d_sku::<T>(PoolKind::LpPool2dBackward);
        Ok(Self {
            desc: *desc,
            h_out,
            w_out,
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
        args: LpPool2dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, self.h_out, self.w_out, &args)?;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let ceil_flag = if self.desc.ceil_mode { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_2d_f32_backward_run(
                    x_ptr, y_ptr, dy_ptr, dx_ptr,
                    self.desc.batch, self.desc.channels, self.desc.h_in, self.desc.w_in,
                    self.desc.window_h, self.desc.window_w,
                    self.desc.stride_h, self.desc.stride_w,
                    self.h_out, self.w_out, self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_2d_f64_backward_run(
                    x_ptr, y_ptr, dy_ptr, dx_ptr,
                    self.desc.batch, self.desc.channels, self.desc.h_in, self.desc.w_in,
                    self.desc.window_h, self.desc.window_w,
                    self.desc.stride_h, self.desc.stride_w,
                    self.h_out, self.w_out, self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_2d_f16_backward_run(
                    x_ptr, y_ptr, dy_ptr, dx_ptr,
                    self.desc.batch, self.desc.channels, self.desc.h_in, self.desc.w_in,
                    self.desc.window_h, self.desc.window_w,
                    self.desc.stride_h, self.desc.stride_w,
                    self.h_out, self.w_out, self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_2d_bf16_backward_run(
                    x_ptr, y_ptr, dy_ptr, dx_ptr,
                    self.desc.batch, self.desc.channels, self.desc.h_in, self.desc.w_in,
                    self.desc.window_h, self.desc.window_w,
                    self.desc.stride_h, self.desc.stride_w,
                    self.h_out, self.w_out, self.desc.p, ceil_flag, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::LpPool2dBackwardPlan: unexpected dtype after select()",
                ));
            }
        };
        super::map_lp_pool_status(status)
    }
}

fn check_bw_args<T: Element>(
    d: &LpPool2dDescriptor,
    h_out: i32,
    w_out: i32,
    args: &LpPool2dBwArgs<'_, T>,
) -> Result<()> {
    let want_x = [d.batch, d.channels, d.h_in, d.w_in];
    let want_y = [d.batch, d.channels, h_out, w_out];
    if args.x.shape != want_x {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dBackwardPlan: x shape != [N, C, H_in, W_in]",
        ));
    }
    if args.dx.shape != want_x {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dBackwardPlan: dx shape != [N, C, H_in, W_in]",
        ));
    }
    if args.y.shape != want_y {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dBackwardPlan: y shape != [N, C, H_out, W_out]",
        ));
    }
    if args.dy.shape != want_y {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dBackwardPlan: dy shape != [N, C, H_out, W_out]",
        ));
    }
    Ok(())
}
