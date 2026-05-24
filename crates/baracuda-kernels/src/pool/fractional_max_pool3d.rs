//! FractionalMaxPool3d — bespoke kernel (Phase 16.3).
//!
//! 3-D sibling of [`super::fractional_max_pool2d`]. See that module for
//! the window-placement formula (evenly-spaced base + α perturbation),
//! the random-samples ABI, and the saved-`indices` BW pattern.
//!
//! Input shape: `[N, C, D_in, H_in, W_in]`. Output shape:
//! `[N, C, D_out, H_out, W_out]`. `random_samples` shape:
//! `[N, C, 3]` f32 (one α per axis per (batch, channel)).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_fractional_max_pool_3d_bw_bf16_run,
    baracuda_kernels_fractional_max_pool_3d_bw_f16_run,
    baracuda_kernels_fractional_max_pool_3d_bw_f32_run,
    baracuda_kernels_fractional_max_pool_3d_bw_f64_run,
    baracuda_kernels_fractional_max_pool_3d_fw_bf16_run,
    baracuda_kernels_fractional_max_pool_3d_fw_f16_run,
    baracuda_kernels_fractional_max_pool_3d_fw_f32_run,
    baracuda_kernels_fractional_max_pool_3d_fw_f64_run,
};
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::fractional_max_pool2d::{build_sku, ffi_status};

/// Descriptor for `FractionalMaxPool3d`.
#[derive(Copy, Clone, Debug)]
pub struct FractionalMaxPool3dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input depth.
    pub d_in: i32,
    /// Input height.
    pub h_in: i32,
    /// Input width.
    pub w_in: i32,
    /// Window depth.
    pub window_d: i32,
    /// Window height.
    pub window_h: i32,
    /// Window width.
    pub window_w: i32,
    /// Desired output depth.
    pub d_out: i32,
    /// Desired output height.
    pub h_out: i32,
    /// Desired output width.
    pub w_out: i32,
    /// PRNG seed. **Unused** in Phase 16.3 — caller supplies samples.
    pub seed: u64,
    /// Element dtype.
    pub element: ElementKind,
}

/// Args bundle for the 3-D forward launch.
pub struct FractionalMaxPool3dFwArgs<'a, T: Element> {
    /// Input `[N, C, D_in, H_in, W_in]`.
    pub x: TensorRef<'a, T, 5>,
    /// Output `[N, C, D_out, H_out, W_out]`.
    pub y: TensorMut<'a, T, 5>,
    /// Per-window argmax linear-index output `[N, C, D_out, H_out, W_out]` i64.
    pub indices: TensorMut<'a, i64, 5>,
    /// Per-(batch, channel, axis) samples `[N, C, 3]` f32.
    pub random_samples: TensorRef<'a, f32, 3>,
}

/// Args bundle for the 3-D backward launch.
///
/// **Caller must zero `dx` before calling.**
pub struct FractionalMaxPool3dBwArgs<'a, T: Element> {
    /// Upstream gradient `[N, C, D_out, H_out, W_out]`.
    pub dy: TensorRef<'a, T, 5>,
    /// Saved forward argmax indices `[N, C, D_out, H_out, W_out]` i64.
    pub indices: TensorRef<'a, i64, 5>,
    /// Output gradient `[N, C, D_in, H_in, W_in]`. Must be pre-zeroed.
    pub dx: TensorMut<'a, T, 5>,
}

/// 3-D fractional max-pool plan (bespoke kernel).
pub struct FractionalMaxPool3dPlan<T: Element> {
    desc: FractionalMaxPool3dDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FractionalMaxPool3dPlan<T> {
    /// Validate the descriptor and pick a kernel SKU.
    pub fn select(
        _stream: &Stream,
        desc: &FractionalMaxPool3dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let sku = build_sku::<T>(PoolKind::FractionalMaxPool3d, true);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Kernel SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Workspace size in bytes. Always `0`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FractionalMaxPool3dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args(&self.desc, &args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x = args.x.data.as_raw().0 as *const c_void;
        let y = args.y.data.as_raw().0 as *mut c_void;
        let indices = args.indices.data.as_raw().0 as *mut c_void;
        let rs = args.random_samples.data.as_raw().0 as *const f32;
        let status = unsafe {
            match T::KIND {
                ElementKind::F32 => baracuda_kernels_fractional_max_pool_3d_fw_f32_run(
                    x, y, indices, rs,
                    self.desc.batch, self.desc.channels,
                    self.desc.d_in, self.desc.h_in, self.desc.w_in,
                    self.desc.d_out, self.desc.h_out, self.desc.w_out,
                    self.desc.window_d, self.desc.window_h, self.desc.window_w,
                    stream_ptr,
                ),
                ElementKind::F64 => baracuda_kernels_fractional_max_pool_3d_fw_f64_run(
                    x, y, indices, rs,
                    self.desc.batch, self.desc.channels,
                    self.desc.d_in, self.desc.h_in, self.desc.w_in,
                    self.desc.d_out, self.desc.h_out, self.desc.w_out,
                    self.desc.window_d, self.desc.window_h, self.desc.window_w,
                    stream_ptr,
                ),
                ElementKind::F16 => baracuda_kernels_fractional_max_pool_3d_fw_f16_run(
                    x, y, indices, rs,
                    self.desc.batch, self.desc.channels,
                    self.desc.d_in, self.desc.h_in, self.desc.w_in,
                    self.desc.d_out, self.desc.h_out, self.desc.w_out,
                    self.desc.window_d, self.desc.window_h, self.desc.window_w,
                    stream_ptr,
                ),
                ElementKind::Bf16 => baracuda_kernels_fractional_max_pool_3d_fw_bf16_run(
                    x, y, indices, rs,
                    self.desc.batch, self.desc.channels,
                    self.desc.d_in, self.desc.h_in, self.desc.w_in,
                    self.desc.d_out, self.desc.h_out, self.desc.w_out,
                    self.desc.window_d, self.desc.window_h, self.desc.window_w,
                    stream_ptr,
                ),
                _ => return Err(Error::Unsupported(
                    "baracuda-kernels::FractionalMaxPool3dPlan: dtype not in {f16, bf16, f32, f64}",
                )),
            }
        };
        ffi_status(status)
    }

    /// Run the backward pass. **Caller must zero `dx` before this call.**
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FractionalMaxPool3dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, &args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy = args.dy.data.as_raw().0 as *const c_void;
        let indices = args.indices.data.as_raw().0 as *const c_void;
        let dx = args.dx.data.as_raw().0 as *mut c_void;
        let status = unsafe {
            match T::KIND {
                ElementKind::F32 => baracuda_kernels_fractional_max_pool_3d_bw_f32_run(
                    dy, indices, dx,
                    self.desc.batch, self.desc.channels,
                    self.desc.d_in, self.desc.h_in, self.desc.w_in,
                    self.desc.d_out, self.desc.h_out, self.desc.w_out,
                    stream_ptr,
                ),
                ElementKind::F64 => baracuda_kernels_fractional_max_pool_3d_bw_f64_run(
                    dy, indices, dx,
                    self.desc.batch, self.desc.channels,
                    self.desc.d_in, self.desc.h_in, self.desc.w_in,
                    self.desc.d_out, self.desc.h_out, self.desc.w_out,
                    stream_ptr,
                ),
                ElementKind::F16 => baracuda_kernels_fractional_max_pool_3d_bw_f16_run(
                    dy, indices, dx,
                    self.desc.batch, self.desc.channels,
                    self.desc.d_in, self.desc.h_in, self.desc.w_in,
                    self.desc.d_out, self.desc.h_out, self.desc.w_out,
                    stream_ptr,
                ),
                ElementKind::Bf16 => baracuda_kernels_fractional_max_pool_3d_bw_bf16_run(
                    dy, indices, dx,
                    self.desc.batch, self.desc.channels,
                    self.desc.d_in, self.desc.h_in, self.desc.w_in,
                    self.desc.d_out, self.desc.h_out, self.desc.w_out,
                    stream_ptr,
                ),
                _ => return Err(Error::Unsupported(
                    "baracuda-kernels::FractionalMaxPool3dPlan: dtype not in {f16, bf16, f32, f64}",
                )),
            }
        };
        ffi_status(status)
    }
}

fn validate_descriptor<T: Element>(desc: &FractionalMaxPool3dDescriptor) -> Result<()> {
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::FractionalMaxPool3dPlan: descriptor.element != T::KIND",
        ));
    }
    if !matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::FractionalMaxPool3dPlan: dtype not in {f16, bf16, f32, f64}",
        ));
    }
    if desc.batch <= 0 || desc.channels <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: batch / channels must be > 0",
        ));
    }
    if desc.d_in <= 0 || desc.h_in <= 0 || desc.w_in <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: d_in / h_in / w_in must be > 0",
        ));
    }
    if desc.d_out <= 0 || desc.h_out <= 0 || desc.w_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: d_out / h_out / w_out must be > 0",
        ));
    }
    if desc.window_d <= 0 || desc.window_h <= 0 || desc.window_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: window extents must be > 0",
        ));
    }
    if desc.window_d > desc.d_in || desc.window_h > desc.h_in || desc.window_w > desc.w_in {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: window must fit within input",
        ));
    }
    Ok(())
}

fn check_fw_args<T: Element>(
    desc: &FractionalMaxPool3dDescriptor,
    args: &FractionalMaxPool3dFwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.d_in, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, desc.d_out, desc.h_out, desc.w_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: x shape != [N, C, D_in, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: y shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    if args.indices.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: indices shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    if args.random_samples.shape != [desc.batch, desc.channels, 3] {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: random_samples shape != [N, C, 3]",
        ));
    }
    Ok(())
}

fn check_bw_args<T: Element>(
    desc: &FractionalMaxPool3dDescriptor,
    args: &FractionalMaxPool3dBwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.d_in, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, desc.d_out, desc.h_out, desc.w_out];
    if args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: dy shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    if args.indices.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: indices shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    if args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool3dPlan: dx shape != [N, C, D_in, H_in, W_in]",
        ));
    }
    Ok(())
}
