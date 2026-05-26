//! AdaptiveAvgPool1d — NCL adaptive average-pool, bit-exact PyTorch.
//!
//! **Phase 16.1 rewrite.** Previously this plan dispatched through cuDNN
//! with a uniform-window approximation (`kernel = ceil(L_in/L_out)`,
//! `stride = floor(L_in/L_out)`) — bit-exact only when `L_in % L_out
//! == 0`. The new path routes to a bespoke CUDA kernel that implements
//! PyTorch's non-uniform per-output-cell window convention:
//!
//! ```text
//! start_i = floor(i * L_in / L_out)
//! end_i   = ceil((i + 1) * L_in / L_out)
//! y[i]    = mean(x[start_i..end_i])     (denominator = end_i - start_i)
//! ```
//!
//! The plan struct and public API (descriptor, args, `select`, `run_fw`,
//! `run_bw`) are unchanged; only the internal dispatch flipped.
//!
//! **Dtypes**: `{f16, bf16, f32, f64}`. Integer adaptive pool has no
//! Fuel ask and the rounding semantics are dtype-dependent.
//!
//! **BW**: one thread per output cell scatters `dy[i] / win_size_i`
//! into every input cell in the window via `atomicAdd`. Boundary
//! windows overlap (when non-divisible), so atomic scatter is required.
//! half / bf16 atomics route through `baracuda_atomic.cuh`'s CAS
//! helper for determinism on those dtypes. `dx` is zeroed internally
//! by the launcher before the scatter — callers do NOT need to
//! pre-zero.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PoolKind, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for an adaptive 1-D pooling op over NCL tensors.
///
/// Input `[N, C, L_in]` → output `[N, C, L_out]` for caller-supplied
/// `L_out`. PyTorch's bit-exact window-bounds formula is applied
/// internally — no caller-side kernel/stride knobs.
///
/// `#[non_exhaustive]` (Phase 32) — future-proofs against added
/// fields (e.g. ceil_mode, padding hints). All present fields are
/// required; [`Self::new`] is the constructor.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct AdaptivePool1dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input length `L_in`.
    pub l_in: i32,
    /// Desired output length `L_out`.
    pub l_out: i32,
    /// Element dtype.
    pub element: ElementKind,
}

impl AdaptivePool1dDescriptor {
    /// Build a descriptor. All fields are required — adaptive pooling
    /// has no optional knobs at present.
    pub fn new(
        batch: i32,
        channels: i32,
        l_in: i32,
        l_out: i32,
        element: ElementKind,
    ) -> Self {
        Self {
            batch,
            channels,
            l_in,
            l_out,
            element,
        }
    }
}

/// FW args (shape `[N, C, L_in]` → `[N, C, L_out]`).
pub struct AdaptivePool1dFwArgs<'a, T: Element> {
    /// Input `[N, C, L_in]`.
    pub x: TensorRef<'a, T, 3>,
    /// Output `[N, C, L_out]`.
    pub y: TensorMut<'a, T, 3>,
}

/// BW args. `y` / `x` are retained for API symmetry with the prior
/// cuDNN-driven plan but the bespoke AvgPool BW kernel only reads
/// `dy` + writes `dx`.
pub struct AdaptivePool1dBwArgs<'a, T: Element> {
    /// Saved FW output `[N, C, L_out]` (unused by AvgPool BW; retained
    /// for API symmetry with [`super::AdaptiveMaxPool1dPlan`]).
    pub y: TensorRef<'a, T, 3>,
    /// Upstream gradient `[N, C, L_out]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Saved FW input `[N, C, L_in]` (unused by AvgPool BW; retained
    /// for API symmetry).
    pub x: TensorRef<'a, T, 3>,
    /// Output gradient `[N, C, L_in]`. Fully overwritten by the launch
    /// (kernel zeros it internally before atomic-scattering into it).
    pub dx: TensorMut<'a, T, 3>,
}

/// Adaptive 1-D average-pool plan (bit-exact PyTorch, bespoke kernel).
pub struct AdaptiveAvgPool1dPlan<T: Element> {
    desc: AdaptivePool1dDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> AdaptiveAvgPool1dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    ///
    /// Returns `Error::Unsupported` if the dtype isn't `f32` / `f64` /
    /// `f16` / `bf16`, or `Error::InvalidProblem` if `l_in <= 0` /
    /// `l_out <= 0` / `l_out > l_in`.
    pub fn select(
        _stream: &Stream,
        desc: &AdaptivePool1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let sku = build_sku::<T>(PoolKind::AdaptiveAvgPool1d);
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

    /// Deprecated. Returned the cuDNN-approximation `(kernel, stride)`
    /// derived from `(L_in, L_out)` in the previous implementation.
    /// Phase 16.1 implements PyTorch's per-output-cell variable window
    /// directly — no single `(kernel, stride)` pair represents the op.
    /// Always returns `(0, 0)` as a stable poison value.
    #[inline]
    #[deprecated(
        since = "0.0.1-alpha.33",
        note = "Phase 16.1 uses bit-exact per-output-cell windows; no single (kernel, stride) pair applies."
    )]
    pub fn derived_kernel_stride(&self) -> (i32, i32) {
        (0, 0)
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool1dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args(&self.desc, &args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let nc = self.desc.batch * self.desc.channels;
        let status = dispatch_avg_fw::<T>(
            x_ptr,
            y_ptr,
            nc,
            1, // spatial_rank
            1, 1, self.desc.l_in,
            1, 1, self.desc.l_out,
            stream_ptr,
        );
        map_status(status)
    }

    /// Run the backward pass. Kernel internally zeros `dx` before
    /// atomic-scattering — callers do not need to pre-zero.
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool1dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, &args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let nc = self.desc.batch * self.desc.channels;
        let status = dispatch_avg_bw::<T>(
            dy_ptr,
            dx_ptr,
            nc,
            1,
            1, 1, self.desc.l_in,
            1, 1, self.desc.l_out,
            stream_ptr,
        );
        map_status(status)
    }
}

// =============================================================================
// Shared adaptive-1d helpers (used by sibling adaptive_max_pool1d)
// =============================================================================

pub(crate) fn validate_descriptor<T: Element>(
    desc: &AdaptivePool1dDescriptor,
) -> Result<()> {
    validate_dtype::<T>()?;
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::AdaptivePool1dPlan: descriptor.element != T::KIND",
        ));
    }
    if desc.batch <= 0 || desc.channels <= 0 || desc.l_in <= 0 || desc.l_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: extents must be > 0",
        ));
    }
    if desc.l_out > desc.l_in {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: l_out must be <= l_in (upsampling \
             is not a pooling op)",
        ));
    }
    Ok(())
}

pub(crate) fn check_fw_args<T: Element>(
    desc: &AdaptivePool1dDescriptor,
    args: &AdaptivePool1dFwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.l_in];
    let y_shape = [desc.batch, desc.channels, desc.l_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: x shape != [N, C, L_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: y shape != [N, C, L_out]",
        ));
    }
    Ok(())
}

pub(crate) fn check_bw_args<T: Element>(
    desc: &AdaptivePool1dDescriptor,
    args: &AdaptivePool1dBwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.l_in];
    let y_shape = [desc.batch, desc.channels, desc.l_out];
    if args.x.shape != x_shape || args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: x/dx shape != [N, C, L_in]",
        ));
    }
    if args.y.shape != y_shape || args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: y/dy shape != [N, C, L_out]",
        ));
    }
    Ok(())
}

// =============================================================================
// Shared dispatch helpers used by all six adaptive-pool plans.
// =============================================================================

pub(crate) fn validate_dtype<T: Element>() -> Result<()> {
    if !matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::AdaptivePoolPlan: adaptive pooling supports f32 / f64 / f16 / bf16",
        ));
    }
    Ok(())
}

pub(crate) fn build_sku<T: Element>(op: PoolKind) -> KernelSku {
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
        // BW uses atomicAdd scatter — non-deterministic ordering across
        // launches, but per-window mean / max arithmetic is bit-stable.
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_avg_fw<T: Element>(
    x: *const c_void,
    y: *mut c_void,
    nc: i32,
    spatial_rank: i32,
    in_d: i32,
    in_h: i32,
    in_w: i32,
    out_d: i32,
    out_h: i32,
    out_w: i32,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_avg_pool_f16_fw_run(
                x, y, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::Bf16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_avg_pool_bf16_fw_run(
                x, y, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::F32 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_avg_pool_f32_fw_run(
                x, y, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::F64 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_avg_pool_f64_fw_run(
                x, y, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        _ => 3, // unsupported — should be unreachable post-select
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_avg_bw<T: Element>(
    dy: *const c_void,
    dx: *mut c_void,
    nc: i32,
    spatial_rank: i32,
    in_d: i32,
    in_h: i32,
    in_w: i32,
    out_d: i32,
    out_h: i32,
    out_w: i32,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_avg_pool_f16_bw_run(
                dy, dx, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::Bf16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_avg_pool_bf16_bw_run(
                dy, dx, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::F32 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_avg_pool_f32_bw_run(
                dy, dx, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::F64 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_avg_pool_f64_bw_run(
                dy, dx, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        _ => 3,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_max_fw<T: Element>(
    x: *const c_void,
    y: *mut c_void,
    nc: i32,
    spatial_rank: i32,
    in_d: i32,
    in_h: i32,
    in_w: i32,
    out_d: i32,
    out_h: i32,
    out_w: i32,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_max_pool_f16_fw_run(
                x, y, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::Bf16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_max_pool_bf16_fw_run(
                x, y, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::F32 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_max_pool_f32_fw_run(
                x, y, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::F64 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_max_pool_f64_fw_run(
                x, y, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        _ => 3,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_max_bw<T: Element>(
    x: *const c_void,
    dy: *const c_void,
    dx: *mut c_void,
    nc: i32,
    spatial_rank: i32,
    in_d: i32,
    in_h: i32,
    in_w: i32,
    out_d: i32,
    out_h: i32,
    out_w: i32,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_max_pool_f16_bw_run(
                x, dy, dx, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::Bf16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_max_pool_bf16_bw_run(
                x, dy, dx, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::F32 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_max_pool_f32_bw_run(
                x, dy, dx, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        ElementKind::F64 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_adaptive_max_pool_f64_bw_run(
                x, dy, dx, nc, spatial_rank, in_d, in_h, in_w, out_d, out_h, out_w, stream,
            )
        },
        _ => 3,
    }
}

pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem (adaptive pool)",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration (adaptive pool)",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
