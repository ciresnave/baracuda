//! LayerNorm backward plan — multi-axis.
//!
//! Standard layer-norm gradient (biased variance):
//!
//!   `x_hat[i] = (x[i] - mean) * inv_std`
//!   `dx_hat[i] = dy[i] · gamma[i]`   (or `dy[i]` if no gamma)
//!   `dx[i] = inv_std · (dx_hat[i] - Σ_j dx_hat[j] / M - x_hat[i] · Σ_j dx_hat[j] · x_hat[j] / M)`
//!
//! where `M = norm_total_extent`.
//!
//! Affine gradients:
//!   `dgamma[i] = Σ over outer cells dy[..., i] · x_hat[..., i]`
//!   `dbeta[i]  = Σ over outer cells dy[..., i]`

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, NormalizationKind,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::rms_norm::{map_status, validate_mask_suffix};

/// Descriptor for a LayerNorm backward op.
#[derive(Copy, Clone, Debug)]
pub struct LayerNormBackwardDescriptor<const N: usize> {
    /// Tensor shape (shared by `dy`, `x`, `dx`).
    pub input_shape: [i32; N],
    /// Bitmask of normalized axes (must match the FW pass).
    pub norm_axes_mask: u8,
    /// Whether `gamma` (and thus `dgamma`) participate.
    pub has_gamma: bool,
    /// Whether `beta` (and thus `dbeta`) participate.
    pub has_beta: bool,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> LayerNormBackwardDescriptor<N> {
    /// Shape of the saved mean / inv_std buffers.
    #[inline]
    pub fn save_shape(&self) -> [i32; N] {
        let mut s = self.input_shape;
        for d in 0..N {
            if (self.norm_axes_mask >> d) & 1 == 1 {
                s[d] = 1;
            }
        }
        s
    }

    /// Total normalized extent.
    #[inline]
    pub fn norm_total_extent(&self) -> i32 {
        let mut p: i32 = 1;
        for d in 0..N {
            if (self.norm_axes_mask >> d) & 1 == 1 {
                p = p.saturating_mul(self.input_shape[d]);
            }
        }
        p
    }
}

/// Args bundle for a LayerNorm BW launch.
pub struct LayerNormBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input.
    pub x: TensorRef<'a, T, N>,
    /// Per-feature gamma.
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Saved forward mean.
    pub mean: TensorRef<'a, T, N>,
    /// Saved forward inverse standard deviation.
    pub inv_std: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input.
    pub dx: TensorMut<'a, T, N>,
    /// Gradient w.r.t. gamma.
    pub dgamma: Option<TensorMut<'a, T, 1>>,
    /// Gradient w.r.t. beta.
    pub dbeta: Option<TensorMut<'a, T, 1>>,
}

/// LayerNorm backward plan.
pub struct LayerNormBackwardPlan<T: Element, const N: usize> {
    desc: LayerNormBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> LayerNormBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &LayerNormBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::LayerNormBackwardPlan: descriptor element != T",
            ));
        }
        if !validate_mask_suffix(desc.norm_axes_mask, N) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormBackwardPlan: norm_axes_mask must be a non-empty \
                 suffix of [0, N)",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::LayerNormBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if N == 0 || N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::LayerNormBackwardPlan: tensor rank must be in 1..=8",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::LayerNormBackwardPlan: wired today: `{f32, f16, bf16, f64}`",
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
            op: NormalizationKind::LayerNorm as u16,
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
    pub fn can_implement(&self, args: &LayerNormBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormBackwardPlan: dy shape mismatch",
            ));
        }
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormBackwardPlan: x shape mismatch",
            ));
        }
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormBackwardPlan: dx shape mismatch",
            ));
        }
        let save_shape = self.desc.save_shape();
        if args.mean.shape != save_shape || args.inv_std.shape != save_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormBackwardPlan: mean / inv_std shape mismatch",
            ));
        }
        if args.mean.stride != args.inv_std.stride {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormBackwardPlan: mean and inv_std must share stride",
            ));
        }
        let total_extent = self.desc.norm_total_extent() as i64;
        match (&args.gamma, &args.dgamma, self.desc.has_gamma) {
            (Some(g), Some(dg), true) => {
                if g.shape[0] as i64 != total_extent || dg.shape[0] as i64 != total_extent {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::LayerNormBackwardPlan: gamma / dgamma length != \
                         norm_total_extent",
                    ));
                }
                if (g.data.len() as i64) < total_extent || (dg.data.len() as i64) < total_extent {
                    return Err(Error::BufferTooSmall {
                        needed: total_extent as usize,
                        got: g.data.len().min(dg.data.len()),
                    });
                }
            }
            (None, None, false) => {}
            _ => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::LayerNormBackwardPlan: gamma / dgamma must both be \
                     present iff desc.has_gamma=true",
                ));
            }
        }
        match (&args.dbeta, self.desc.has_beta) {
            (Some(db), true) => {
                if db.shape[0] as i64 != total_extent {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::LayerNormBackwardPlan: dbeta length != \
                         norm_total_extent",
                    ));
                }
                if (db.data.len() as i64) < total_extent {
                    return Err(Error::BufferTooSmall {
                        needed: total_extent as usize,
                        got: db.data.len(),
                    });
                }
            }
            (None, false) => {}
            _ => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::LayerNormBackwardPlan: dbeta present iff desc.has_beta=true",
                ));
            }
        }
        let numel = args.dx.numel();
        let dy_len = args.dy.data.len() as i64;
        let x_len = args.x.data.len() as i64;
        let dx_len = args.dx.data.len() as i64;
        let save_numel = args.mean.numel();
        let mean_len = args.mean.data.len() as i64;
        let std_len = args.inv_std.data.len() as i64;
        if dy_len < numel || x_len < numel || dx_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(x_len).min(dx_len) as usize,
            });
        }
        if mean_len < save_numel || std_len < save_numel {
            return Err(Error::BufferTooSmall {
                needed: save_numel as usize,
                got: mean_len.min(std_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
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
        mut args: LayerNormBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dx.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let mean_ptr = args.mean.data.as_raw().0 as *const c_void;
        let std_ptr = args.inv_std.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let gamma_ptr = match &args.gamma {
            Some(g) => g.data.as_raw().0 as *const c_void,
            None => core::ptr::null(),
        };
        let dgamma_ptr = match &mut args.dgamma {
            Some(dg) => dg.data.as_raw().0 as *mut c_void,
            None => core::ptr::null_mut(),
        };
        let dbeta_ptr = match &mut args.dbeta {
            Some(db) => db.data.as_raw().0 as *mut c_void,
            None => core::ptr::null_mut(),
        };

        let shape = self.desc.input_shape;
        let stride_dy = args.dy.stride;
        let stride_x = args.x.stride;
        let stride_save = args.mean.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;
        let mask = self.desc.norm_axes_mask as i32;
        let total_extent = self.desc.norm_total_extent();

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_layer_norm_backward_f32_run(
                    numel, rank, shape.as_ptr(),
                    stride_dy.as_ptr(), stride_x.as_ptr(), stride_save.as_ptr(), stride_dx.as_ptr(),
                    mask, total_extent,
                    dy_ptr, x_ptr, gamma_ptr, mean_ptr, std_ptr, dx_ptr, dgamma_ptr, dbeta_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_layer_norm_backward_f16_run(
                    numel, rank, shape.as_ptr(),
                    stride_dy.as_ptr(), stride_x.as_ptr(), stride_save.as_ptr(), stride_dx.as_ptr(),
                    mask, total_extent,
                    dy_ptr, x_ptr, gamma_ptr, mean_ptr, std_ptr, dx_ptr, dgamma_ptr, dbeta_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_layer_norm_backward_bf16_run(
                    numel, rank, shape.as_ptr(),
                    stride_dy.as_ptr(), stride_x.as_ptr(), stride_save.as_ptr(), stride_dx.as_ptr(),
                    mask, total_extent,
                    dy_ptr, x_ptr, gamma_ptr, mean_ptr, std_ptr, dx_ptr, dgamma_ptr, dbeta_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_layer_norm_backward_f64_run(
                    numel, rank, shape.as_ptr(),
                    stride_dy.as_ptr(), stride_x.as_ptr(), stride_save.as_ptr(), stride_dx.as_ptr(),
                    mask, total_extent,
                    dy_ptr, x_ptr, gamma_ptr, mean_ptr, std_ptr, dx_ptr, dgamma_ptr, dbeta_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::LayerNormBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
