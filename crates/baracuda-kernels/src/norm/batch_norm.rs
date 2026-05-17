//! BatchNorm forward plan.
//!
//! Per-channel normalization across `(N, *spatial*)`:
//!   `y[n, c, ...] = (x[n, c, ...] - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]`
//!
//! `channel_axis` is the axis carrying the per-channel affine. PyTorch's
//! `BatchNorm1d/2d/3d` all use `channel_axis = 1` on `[N, C, ...]`
//! layouts.
//!
//! ## Training mode only
//!
//! This plan supports **training mode** only: it computes batch
//! statistics on every call and writes them to `saved_mean` /
//! `saved_rstd` for BW reuse. Inference mode (using running statistics)
//! reduces to a per-channel affine multiply and is deferred — it will
//! ship as a thin wrapper.
//!
//! ## Deferred
//!
//! `WeightNorm` is a parameterization, not a plain op; ships separately.
//! `LocalResponseNorm` is rarely used and explicitly deferred.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, NormalizationKind,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::rms_norm::map_status;

/// Descriptor for a BatchNorm forward op (training mode).
#[derive(Copy, Clone, Debug)]
pub struct BatchNormDescriptor<const N: usize> {
    /// Input tensor shape. Channel axis must be axis 1 in this
    /// trailblazer (`[N, C, *spatial]`). Future channel-last support
    /// will add an explicit `channel_axis` field.
    pub input_shape: [i32; N],
    /// Channel axis index. Must equal 1 in this trailblazer.
    pub channel_axis: u8,
    /// Epsilon for variance stabilization.
    pub eps: f32,
    /// Whether `gamma + beta` participate (PyTorch convention: both or
    /// neither).
    pub has_affine: bool,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> BatchNormDescriptor<N> {
    /// Number of channels.
    #[inline]
    pub fn num_channels(&self) -> i32 {
        if N >= 2 {
            self.input_shape[self.channel_axis as usize]
        } else {
            1
        }
    }
}

/// Args bundle for BatchNorm forward.
pub struct BatchNormArgs<'a, T: Element, const N: usize> {
    /// Input tensor `[N, C, ...]`.
    pub x: TensorRef<'a, T, N>,
    /// Per-channel affine weight `[C]` (length == num_channels).
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Per-channel affine bias `[C]`.
    pub beta: Option<TensorRef<'a, T, 1>>,
    /// Output tensor `[N, C, ...]`.
    pub y: TensorMut<'a, T, N>,
    /// Saved per-channel mean `[C]` — written by FW for BW reuse.
    pub saved_mean: TensorMut<'a, T, 1>,
    /// Saved per-channel inverse standard deviation `[C]`.
    pub saved_rstd: TensorMut<'a, T, 1>,
}

/// BatchNorm forward plan.
pub struct BatchNormPlan<T: Element, const N: usize> {
    desc: BatchNormDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> BatchNormPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &BatchNormDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchNormPlan: descriptor element != T",
            ));
        }
        if N < 2 || N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchNormPlan: rank must be in 2..=8 (got N)",
            ));
        }
        if desc.channel_axis != 1 {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchNormPlan: channel_axis must be 1 in this trailblazer",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchNormPlan: shape dims must be non-negative",
                ));
            }
        }
        if !(desc.eps.is_finite() && desc.eps >= 0.0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchNormPlan: eps must be finite and non-negative",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchNormPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Normalization,
            op: NormalizationKind::BatchNorm as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &BatchNormArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape || args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchNormPlan: x / y shape mismatch",
            ));
        }
        let c = self.desc.num_channels() as i64;
        if args.saved_mean.shape[0] as i64 != c || args.saved_rstd.shape[0] as i64 != c {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchNormPlan: saved_mean / saved_rstd length != num_channels",
            ));
        }
        if let Some(g) = &args.gamma {
            if g.shape[0] as i64 != c {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchNormPlan: gamma length != num_channels",
                ));
            }
        }
        if let Some(b) = &args.beta {
            if b.shape[0] as i64 != c {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchNormPlan: beta length != num_channels",
                ));
            }
        }
        match (args.gamma.is_some(), args.beta.is_some(), self.desc.has_affine) {
            (true, true, true) | (false, false, false) => {}
            _ => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchNormPlan: gamma + beta must both be present iff \
                     has_affine=true",
                ));
            }
        }
        Ok(())
    }

    /// Workspace bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
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

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: BatchNormArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let n_extent = self.desc.input_shape[0];
        let c_extent = self.desc.input_shape[1];
        let mut s_extent: i32 = 1;
        for d in 2..N {
            s_extent = s_extent.saturating_mul(self.desc.input_shape[d]);
        }
        if n_extent == 0 || c_extent == 0 || s_extent == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let mean_ptr = args.saved_mean.data.as_raw().0 as *mut c_void;
        let rstd_ptr = args.saved_rstd.data.as_raw().0 as *mut c_void;
        let gamma_ptr = args.gamma.as_ref().map(|g| g.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let beta_ptr = args.beta.as_ref().map(|b| b.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_batch_norm_f32_run(
                    n_extent, c_extent, s_extent,
                    c_extent, 0, self.desc.eps,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, rstd_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_batch_norm_f16_run(
                    n_extent, c_extent, s_extent,
                    c_extent, 0, self.desc.eps,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, rstd_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_batch_norm_bf16_run(
                    n_extent, c_extent, s_extent,
                    c_extent, 0, self.desc.eps,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, rstd_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_batch_norm_f64_run(
                    n_extent, c_extent, s_extent,
                    c_extent, 0, self.desc.eps,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, rstd_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BatchNormPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
