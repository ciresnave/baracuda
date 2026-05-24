//! LPPool2d — bespoke fused 2-D LP-pool (Phase 16.2).
//!
//! `y[..., i, j] = (Σ_{k, l ∈ window} |x[..., i*sh + k, j*sw + l]|^p)^(1/p)`
//! over an NCHW tensor. PyTorch's `nn.LPPool2d(p, (kh, kw), (sh, sw),
//! ceil_mode)`.
//!
//! See [`super::lp_pool1d`] for the rationale (cuDNN has no LP-pool;
//! the fused kernel sidesteps the missing parameterized `Pow(p)` unary
//! plan and saves 2 launches).
//!
//! **No padding** (matches PyTorch). `ceil_mode = true` rounds output
//! extents up; partially-overhanging windows are truncated.
//!
//! **Dtypes**: `f16, bf16, f32, f64`.
//!
//! Pair with [`super::lp_pool2d_backward::LpPool2dBackwardPlan`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PoolKind, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for `LPPool2d`.
///
/// Input shape: `[batch, channels, h_in, w_in]`. Output shape:
/// `[batch, channels, h_out, w_out]`.
#[derive(Copy, Clone, Debug)]
pub struct LpPool2dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input height.
    pub h_in: i32,
    /// Input width.
    pub w_in: i32,
    /// Window height.
    pub window_h: i32,
    /// Window width.
    pub window_w: i32,
    /// Stride along the height axis.
    pub stride_h: i32,
    /// Stride along the width axis.
    pub stride_w: i32,
    /// Norm exponent `p`. Must be `> 0` and finite. Use
    /// [`super::MaxPool2dPlan`] for the `p = ∞` case.
    pub p: f32,
    /// `false` → floor formula on output extents (default).
    /// `true`  → ceil formula on output extents.
    pub ceil_mode: bool,
    /// Element dtype.
    pub element: ElementKind,
}

/// Args bundle for an `LpPool2d` forward launch.
pub struct LpPool2dFwArgs<'a, T: Element> {
    /// Input `[N, C, H_in, W_in]` NCHW contiguous.
    pub x: TensorRef<'a, T, 4>,
    /// Output `[N, C, H_out, W_out]` NCHW contiguous.
    pub y: TensorMut<'a, T, 4>,
}

/// `LPPool2d` plan — bespoke fused FW.
pub struct LpPool2dPlan<T: Element> {
    pub(super) desc: LpPool2dDescriptor,
    pub(super) h_out: i32,
    pub(super) w_out: i32,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> LpPool2dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &LpPool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_lp2d::<T>(desc)?;
        let (h_out, w_out) = compute_out_2d(desc)?;
        let sku = build_lp2d_sku::<T>(PoolKind::LpPool2d);
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

    /// `(H_out, W_out)` output spatial extents.
    #[inline]
    pub fn output_dims(&self) -> (i32, i32) {
        (self.h_out, self.w_out)
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: LpPool2dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args_lp2d(&self.desc, self.h_out, self.w_out, &args)?;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let ceil_flag = if self.desc.ceil_mode { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_2d_f32_run(
                    x_ptr, y_ptr, self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.window_h, self.desc.window_w,
                    self.desc.stride_h, self.desc.stride_w,
                    self.h_out, self.w_out, self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_2d_f64_run(
                    x_ptr, y_ptr, self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.window_h, self.desc.window_w,
                    self.desc.stride_h, self.desc.stride_w,
                    self.h_out, self.w_out, self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_2d_f16_run(
                    x_ptr, y_ptr, self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.window_h, self.desc.window_w,
                    self.desc.stride_h, self.desc.stride_w,
                    self.h_out, self.w_out, self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_2d_bf16_run(
                    x_ptr, y_ptr, self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.window_h, self.desc.window_w,
                    self.desc.stride_h, self.desc.stride_w,
                    self.h_out, self.w_out, self.desc.p, ceil_flag, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::LpPool2dPlan: unexpected dtype after select()",
                ));
            }
        };
        super::map_lp_pool_status(status)
    }
}

// =============================================================================
// Shared helpers — also used by the BW sibling.
// =============================================================================

pub(super) fn validate_lp2d<T: Element>(d: &LpPool2dDescriptor) -> Result<()> {
    if d.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::LpPool2dPlan: descriptor.element != T::KIND",
        ));
    }
    if !matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::LpPool2dPlan: dtype must be f32 / f64 / f16 / bf16",
        ));
    }
    if d.batch <= 0 || d.channels <= 0 || d.h_in <= 0 || d.w_in <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dPlan: batch/channels/h_in/w_in must be > 0",
        ));
    }
    if d.window_h <= 0 || d.window_w <= 0 || d.stride_h <= 0 || d.stride_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dPlan: window/stride must be > 0",
        ));
    }
    if d.window_h > d.h_in || d.window_w > d.w_in {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dPlan: window > input dimension produces zero-sized output",
        ));
    }
    if !d.p.is_finite() || d.p <= 0.0 {
        return Err(Error::Unsupported(
            "baracuda-kernels::LpPool2dPlan: p must be finite and > 0 \
             (use MaxPool2dPlan for the p=∞ case)",
        ));
    }
    Ok(())
}

pub(super) fn compute_out_2d(d: &LpPool2dDescriptor) -> Result<(i32, i32)> {
    let diff_h = d.h_in - d.window_h;
    let diff_w = d.w_in - d.window_w;
    let (h_out, w_out) = if d.ceil_mode {
        (
            (diff_h + d.stride_h - 1) / d.stride_h + 1,
            (diff_w + d.stride_w - 1) / d.stride_w + 1,
        )
    } else {
        (diff_h / d.stride_h + 1, diff_w / d.stride_w + 1)
    };
    if h_out <= 0 || w_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dPlan: computed (h_out, w_out) <= 0",
        ));
    }
    Ok((h_out, w_out))
}

pub(super) fn check_fw_args_lp2d<T: Element>(
    d: &LpPool2dDescriptor,
    h_out: i32,
    w_out: i32,
    args: &LpPool2dFwArgs<'_, T>,
) -> Result<()> {
    let want_x = [d.batch, d.channels, d.h_in, d.w_in];
    let want_y = [d.batch, d.channels, h_out, w_out];
    if args.x.shape != want_x {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dPlan: x shape != [N, C, H_in, W_in]",
        ));
    }
    if args.y.shape != want_y {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool2dPlan: y shape != [N, C, H_out, W_out]",
        ));
    }
    Ok(())
}

pub(super) fn build_lp2d_sku<T: Element>(op: PoolKind) -> KernelSku {
    let math_precision = match T::KIND {
        ElementKind::F64 => MathPrecision::F64,
        ElementKind::F16 => MathPrecision::F16,
        ElementKind::Bf16 => MathPrecision::Bf16,
        _ => MathPrecision::F32,
    };
    let accumulator = match T::KIND {
        ElementKind::F64 => ElementKind::F64,
        _ => ElementKind::F32,
    };
    let precision_guarantee = PrecisionGuarantee {
        math_precision,
        accumulator,
        bit_stable_on_same_hardware: false,
        deterministic: true,
    };
    KernelSku {
        category: OpCategory::Pooling,
        op: op as u16,
        element: T::KIND,
        aux_element: None,
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}
