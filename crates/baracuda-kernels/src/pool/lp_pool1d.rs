//! LPPool1d — bespoke fused 1-D LP-pool (Phase 16.2).
//!
//! `y[..., i] = (Σ_{k ∈ window} |x[..., i*stride + k]|^p)^(1/p)` along
//! the spatial axis of an NCL tensor. PyTorch's `nn.LPPool1d(p, kernel,
//! stride, ceil_mode)`.
//!
//! **Why bespoke**: cuDNN doesn't expose LP-pool natively. Composing as
//! `pow → avg_pool1d → pow` requires a parameterized `Pow(p)` unary
//! plan (today's Pow takes a tensor, Phase 12's PowI takes an integer
//! exponent) and pays 3× launch overhead. The fused kernel does the
//! full pool in one launch.
//!
//! **No padding** (matches PyTorch — `LpPool` has no `pad` argument).
//! `ceil_mode = true` lets the output extent round up; windows whose
//! trailing edge would overhang the input boundary are *truncated*
//! (window clamps to `[start, min(start + kernel, in))`).
//!
//! **Dtypes**: `f16, bf16, f32, f64`.
//!
//! **Special cases**:
//! - `p == 1`: `y = Σ |x|` (sum-of-abs pool).
//! - `p == 2`: `y = √(Σ x²)` (L2-norm pool, most common).
//! - `p == ∞`: equivalent to MaxPool; *not* handled here — use
//!   [`super::MaxPool1dPlan`] directly. `select` rejects `p` that is
//!   non-positive or non-finite.
//!
//! Pair with [`super::lp_pool1d_backward::LpPool1dBackwardPlan`] for
//! autograd.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PoolKind, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for `LPPool1d`.
///
/// Input shape: `[batch, channels, l_in]`. Output shape:
/// `[batch, channels, l_out]` where `l_out` follows the floor/ceil
/// formula below.
///
/// `#[non_exhaustive]` (Phase 32) — Phase 16 already added `ceil_mode`;
/// future fields (e.g. padding) may follow. Use [`Self::new`] + the
/// `with_*` setters from downstream code.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct LpPool1dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input length.
    pub l_in: i32,
    /// Pool window length.
    pub window: i32,
    /// Stride.
    pub stride: i32,
    /// Norm exponent `p`. Must be `> 0` and finite. Use
    /// [`super::MaxPool1dPlan`] for the `p = ∞` case.
    pub p: f32,
    /// `false` → `l_out = floor((l_in - window) / stride) + 1` (default).
    /// `true`  → `l_out = ceil ((l_in - window) / stride) + 1`.
    pub ceil_mode: bool,
    /// Element dtype.
    pub element: ElementKind,
}

impl LpPool1dDescriptor {
    /// Build a descriptor with `stride` defaulted to `window` (PyTorch
    /// pooling default) and `ceil_mode = false`. Chain with
    /// [`Self::with_stride`] / [`Self::with_ceil_mode`] to override.
    pub fn new(
        batch: i32,
        channels: i32,
        l_in: i32,
        window: i32,
        p: f32,
        element: ElementKind,
    ) -> Self {
        Self {
            batch,
            channels,
            l_in,
            window,
            stride: window,
            p,
            ceil_mode: false,
            element,
        }
    }

    /// Override the stride. Default `window`.
    #[inline]
    pub fn with_stride(mut self, stride: i32) -> Self {
        self.stride = stride;
        self
    }

    /// Override `ceil_mode`. Default `false` (floor formula).
    #[inline]
    pub fn with_ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.ceil_mode = ceil_mode;
        self
    }
}

/// Args bundle for an `LpPool1d` forward launch.
pub struct LpPool1dFwArgs<'a, T: Element> {
    /// Input `[N, C, L_in]` NCL contiguous.
    pub x: TensorRef<'a, T, 3>,
    /// Output `[N, C, L_out]` NCL contiguous.
    pub y: TensorMut<'a, T, 3>,
}

/// `LPPool1d` plan — bespoke fused FW.
pub struct LpPool1dPlan<T: Element> {
    pub(super) desc: LpPool1dDescriptor,
    pub(super) l_out: i32,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> LpPool1dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &LpPool1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_lp1d::<T>(desc)?;
        let l_out = compute_l_out(desc)?;
        let sku = build_lp1d_sku::<T>(PoolKind::LpPool1d);
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

    /// Computed output length under the configured window / stride /
    /// ceil_mode.
    #[inline]
    pub fn output_length(&self) -> i32 {
        self.l_out
    }

    /// Run the forward pass. Computes `y := lp_pool1d(x)`.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: LpPool1dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args_lp1d(&self.desc, self.l_out, &args)?;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let ceil_flag = if self.desc.ceil_mode { 1 } else { 0 };
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_1d_f32_run(
                    x_ptr, y_ptr, self.desc.batch, self.desc.channels, self.desc.l_in,
                    self.desc.window, self.desc.stride, self.l_out,
                    self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_1d_f64_run(
                    x_ptr, y_ptr, self.desc.batch, self.desc.channels, self.desc.l_in,
                    self.desc.window, self.desc.stride, self.l_out,
                    self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_1d_f16_run(
                    x_ptr, y_ptr, self.desc.batch, self.desc.channels, self.desc.l_in,
                    self.desc.window, self.desc.stride, self.l_out,
                    self.desc.p, ceil_flag, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_lp_pool_1d_bf16_run(
                    x_ptr, y_ptr, self.desc.batch, self.desc.channels, self.desc.l_in,
                    self.desc.window, self.desc.stride, self.l_out,
                    self.desc.p, ceil_flag, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::LpPool1dPlan: unexpected dtype after select()",
                ));
            }
        };
        super::map_lp_pool_status(status)
    }
}

// =============================================================================
// Shared helpers (also reused by the backward sibling).
// =============================================================================

pub(super) fn validate_lp1d<T: Element>(d: &LpPool1dDescriptor) -> Result<()> {
    if d.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::LpPool1dPlan: descriptor.element != T::KIND",
        ));
    }
    if !matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::LpPool1dPlan: dtype must be f32 / f64 / f16 / bf16",
        ));
    }
    if d.batch <= 0 || d.channels <= 0 || d.l_in <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dPlan: batch/channels/l_in must be > 0",
        ));
    }
    if d.window <= 0 || d.stride <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dPlan: window/stride must be > 0",
        ));
    }
    if d.window > d.l_in {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dPlan: window > l_in produces zero-sized output",
        ));
    }
    if !d.p.is_finite() || d.p <= 0.0 {
        return Err(Error::Unsupported(
            "baracuda-kernels::LpPool1dPlan: p must be finite and > 0 \
             (use MaxPool1dPlan for the p=∞ case)",
        ));
    }
    Ok(())
}

pub(super) fn compute_l_out(d: &LpPool1dDescriptor) -> Result<i32> {
    let diff = d.l_in - d.window;
    let out = if d.ceil_mode {
        (diff + d.stride - 1) / d.stride + 1
    } else {
        diff / d.stride + 1
    };
    if out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dPlan: computed l_out <= 0",
        ));
    }
    Ok(out)
}

pub(super) fn check_fw_args_lp1d<T: Element>(
    d: &LpPool1dDescriptor,
    l_out: i32,
    args: &LpPool1dFwArgs<'_, T>,
) -> Result<()> {
    let want_x = [d.batch, d.channels, d.l_in];
    let want_y = [d.batch, d.channels, l_out];
    if args.x.shape != want_x {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dPlan: x shape != [N, C, L_in]",
        ));
    }
    if args.y.shape != want_y {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::LpPool1dPlan: y shape != [N, C, L_out]",
        ));
    }
    Ok(())
}

pub(super) fn build_lp1d_sku<T: Element>(op: PoolKind) -> KernelSku {
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
        // FW is deterministic + bit-stable (no atomicAdd, fixed
        // accumulation order per output cell). BW uses atomicAdd
        // scatter → deterministic but not bit-stable across runs.
        // The plan tags the same SKU for both; the BW plan owns the
        // weaker guarantee in its rustdoc.
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
