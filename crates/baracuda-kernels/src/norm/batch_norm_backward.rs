//! BatchNorm backward plan.
//!
//! Three-stage scheme:
//!   1. Per-group `(sum_dxh, sum_dxhxh)` reduction (deterministic).
//!   2. Per-cell `dx[i] = inv_std · (dx_hat[i] - sum_dxh/M - x_hat[i]·sum_dxhxh/M)`.
//!   3. Per-channel `dgamma` / `dbeta` reduction.
//!
//! Workspace: `2 * C * sizeof(f32)` bytes for the stage-1 partial sums.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, NormalizationKind,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::rms_norm::map_status;

/// Descriptor for a BatchNorm backward op.
#[derive(Copy, Clone, Debug)]
pub struct BatchNormBackwardDescriptor<const N: usize> {
    /// Input tensor shape `[N, C, ...]`.
    pub input_shape: [i32; N],
    /// Channel axis (must equal 1 in this trailblazer).
    pub channel_axis: u8,
    /// Whether gamma + beta participate.
    pub has_affine: bool,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> BatchNormBackwardDescriptor<N> {
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

/// Args bundle for BatchNorm BW.
pub struct BatchNormBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input.
    pub x: TensorRef<'a, T, N>,
    /// Per-channel gamma.
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Saved per-channel mean from FW.
    pub saved_mean: TensorRef<'a, T, 1>,
    /// Saved per-channel inv_std from FW.
    pub saved_rstd: TensorRef<'a, T, 1>,
    /// Gradient w.r.t. the forward input.
    pub dx: TensorMut<'a, T, N>,
    /// Gradient w.r.t. gamma.
    pub dgamma: Option<TensorMut<'a, T, 1>>,
    /// Gradient w.r.t. beta.
    pub dbeta: Option<TensorMut<'a, T, 1>>,
}

/// BatchNorm backward plan.
pub struct BatchNormBackwardPlan<T: Element, const N: usize> {
    desc: BatchNormBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> BatchNormBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &BatchNormBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchNormBackwardPlan: descriptor element != T",
            ));
        }
        if N < 2 || N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchNormBackwardPlan: rank must be in 2..=8",
            ));
        }
        if desc.channel_axis != 1 {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchNormBackwardPlan: channel_axis must be 1 in this \
                 trailblazer",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchNormBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchNormBackwardPlan: wired today: `{f32, f16, bf16, f64}`",
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
    pub fn can_implement(&self, args: &BatchNormBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.input_shape
            || args.x.shape != self.desc.input_shape
            || args.dx.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchNormBackwardPlan: shape mismatch",
            ));
        }
        let c = self.desc.num_channels() as i64;
        if args.saved_mean.shape[0] as i64 != c || args.saved_rstd.shape[0] as i64 != c {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchNormBackwardPlan: saved buffers length != num_channels",
            ));
        }
        match (
            args.gamma.is_some(),
            args.dgamma.is_some(),
            args.dbeta.is_some(),
            self.desc.has_affine,
        ) {
            (true, true, true, true) | (false, false, false, false) => {}
            _ => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BatchNormBackwardPlan: gamma + dgamma + dbeta must all \
                     be present iff has_affine=true",
                ));
            }
        }
        Ok(())
    }

    /// Workspace bytes — `2 * C * sizeof(f32)`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        2 * self.desc.num_channels() as usize * core::mem::size_of::<f32>()
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
        workspace: Workspace<'_>,
        mut args: BatchNormBackwardArgs<'_, T, N>,
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
        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes): (*mut c_void, usize) = match workspace {
            Workspace::None => {
                if needed > 0 {
                    return Err(Error::WorkspaceTooSmall { needed, got: 0 });
                }
                (core::ptr::null_mut(), 0)
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: slice.len(),
                    });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };

        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let mean_ptr = args.saved_mean.data.as_raw().0 as *const c_void;
        let rstd_ptr = args.saved_rstd.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let gamma_ptr = args.gamma.as_ref().map(|g| g.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let dgamma_ptr = args.dgamma.as_mut().map(|g| g.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let dbeta_ptr = args.dbeta.as_mut().map(|b| b.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_batch_norm_backward_f32_run(
                    n_extent, c_extent, s_extent, c_extent, 0,
                    dy_ptr, x_ptr, gamma_ptr, mean_ptr, rstd_ptr,
                    dx_ptr, dgamma_ptr, dbeta_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_batch_norm_backward_f16_run(
                    n_extent, c_extent, s_extent, c_extent, 0,
                    dy_ptr, x_ptr, gamma_ptr, mean_ptr, rstd_ptr,
                    dx_ptr, dgamma_ptr, dbeta_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_batch_norm_backward_bf16_run(
                    n_extent, c_extent, s_extent, c_extent, 0,
                    dy_ptr, x_ptr, gamma_ptr, mean_ptr, rstd_ptr,
                    dx_ptr, dgamma_ptr, dbeta_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_batch_norm_backward_f64_run(
                    n_extent, c_extent, s_extent, c_extent, 0,
                    dy_ptr, x_ptr, gamma_ptr, mean_ptr, rstd_ptr,
                    dx_ptr, dgamma_ptr, dbeta_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BatchNormBackwardPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
