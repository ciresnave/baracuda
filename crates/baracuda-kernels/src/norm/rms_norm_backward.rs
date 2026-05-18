//! RMSNorm backward plan — multi-axis.
//!
//! Computes:
//!   `dx[..., i] = (dy[..., i] · gamma[i]) / rms`
//!              `- x[..., i] · (Σ_j dy[..., j] · gamma[j] · x[..., j]) / (rms³ · M)`
//!
//! with `M = norm_total_extent` (product of normalized axes' extents).
//!
//! and (when gamma is supplied):
//!   `dgamma[i] = Σ over outer cells dy[..., i] · (x[..., i] / rms)`
//!
//! The single launcher fires both kernels — per-cell `dx` followed by a
//! deterministic one-block-per-feature reduction for `dgamma`. No
//! atomic-adds.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, NormalizationKind,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::rms_norm::{map_status, validate_mask_suffix};

/// Descriptor for an RMSNorm backward op.
#[derive(Copy, Clone, Debug)]
pub struct RMSNormBackwardDescriptor<const N: usize> {
    /// Tensor shape (shared by `dy`, `x`, `dx`).
    pub input_shape: [i32; N],
    /// Bitmask of normalized axes (must match the FW pass and form a
    /// suffix of `[0, N)`).
    pub norm_axes_mask: u8,
    /// Whether `gamma` (and thus `dgamma`) participate.
    pub has_gamma: bool,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> RMSNormBackwardDescriptor<N> {
    /// Shape of the saved-rms buffer.
    #[inline]
    pub fn rms_shape(&self) -> [i32; N] {
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

/// Args bundle for an RMSNorm BW launch.
pub struct RMSNormBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input.
    pub x: TensorRef<'a, T, N>,
    /// Per-feature gamma.
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Saved forward RMS.
    pub rms: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input.
    pub dx: TensorMut<'a, T, N>,
    /// Gradient w.r.t. gamma.
    pub dgamma: Option<TensorMut<'a, T, 1>>,
}

/// RMSNorm backward plan — see module docs for the formula.
///
/// **When to use**: autograd backward for [`super::RMSNormPlan`].
/// Caller saves `x` and the per-row RMS from the FW pass.
///
/// **Dtypes / shape**: `{f32, f16, bf16, f64}` × rank `1..=8`. Mask
/// must match the FW pass.
///
/// **Workspace**: none.
///
/// **Precision**: deterministic, bit-stable on the same hardware. The
/// `dgamma` reduction uses a one-block-per-feature kernel with warp
/// shuffles + smem (no atomic-add), so it's bit-stable regardless of the
/// half / bf16 atomicAdd arch quirks. f16 / bf16 accumulate in f32.
pub struct RMSNormBackwardPlan<T: Element, const N: usize> {
    desc: RMSNormBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> RMSNormBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &RMSNormBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RMSNormBackwardPlan: descriptor element != T",
            ));
        }
        if !validate_mask_suffix(desc.norm_axes_mask, N) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormBackwardPlan: norm_axes_mask must be a non-empty \
                 suffix of [0, N)",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RMSNormBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if N == 0 || N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::RMSNormBackwardPlan: tensor rank must be in 1..=8",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::RMSNormBackwardPlan: wired today: `{f32, f16, bf16, f64}`",
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
            op: NormalizationKind::RMSNorm as u16,
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
    pub fn can_implement(&self, args: &RMSNormBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormBackwardPlan: dy shape mismatch",
            ));
        }
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormBackwardPlan: x shape mismatch",
            ));
        }
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormBackwardPlan: dx shape mismatch",
            ));
        }
        let rms_shape = self.desc.rms_shape();
        if args.rms.shape != rms_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormBackwardPlan: rms shape mismatch",
            ));
        }
        let total_extent = self.desc.norm_total_extent() as i64;
        match (&args.gamma, &args.dgamma, self.desc.has_gamma) {
            (Some(g), Some(dg), true) => {
                if g.shape[0] as i64 != total_extent || dg.shape[0] as i64 != total_extent {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::RMSNormBackwardPlan: gamma / dgamma length != norm_total_extent",
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
                    "baracuda-kernels::RMSNormBackwardPlan: gamma / dgamma must both be \
                     present iff desc.has_gamma=true",
                ));
            }
        }
        let numel = args.dx.numel();
        let dy_len = args.dy.data.len() as i64;
        let x_len = args.x.data.len() as i64;
        let dx_len = args.dx.data.len() as i64;
        let rms_len = args.rms.data.len() as i64;
        let rms_numel = args.rms.numel();
        if dy_len < numel || x_len < numel || dx_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(x_len).min(dx_len) as usize,
            });
        }
        if rms_len < rms_numel {
            return Err(Error::BufferTooSmall {
                needed: rms_numel as usize,
                got: rms_len as usize,
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
        mut args: RMSNormBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dx.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let rms_ptr = args.rms.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let gamma_ptr = match &args.gamma {
            Some(g) => g.data.as_raw().0 as *const c_void,
            None => core::ptr::null(),
        };
        let dgamma_ptr = match &mut args.dgamma {
            Some(dg) => dg.data.as_raw().0 as *mut c_void,
            None => core::ptr::null_mut(),
        };

        let shape = self.desc.input_shape;
        let stride_dy = args.dy.stride;
        let stride_x = args.x.stride;
        let stride_rms = args.rms.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;
        let mask = self.desc.norm_axes_mask as i32;
        let total_extent = self.desc.norm_total_extent();

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rms_norm_backward_f32_run(
                    numel, rank, shape.as_ptr(),
                    stride_dy.as_ptr(), stride_x.as_ptr(), stride_rms.as_ptr(), stride_dx.as_ptr(),
                    mask, total_extent,
                    dy_ptr, x_ptr, gamma_ptr, rms_ptr, dx_ptr, dgamma_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rms_norm_backward_f16_run(
                    numel, rank, shape.as_ptr(),
                    stride_dy.as_ptr(), stride_x.as_ptr(), stride_rms.as_ptr(), stride_dx.as_ptr(),
                    mask, total_extent,
                    dy_ptr, x_ptr, gamma_ptr, rms_ptr, dx_ptr, dgamma_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rms_norm_backward_bf16_run(
                    numel, rank, shape.as_ptr(),
                    stride_dy.as_ptr(), stride_x.as_ptr(), stride_rms.as_ptr(), stride_dx.as_ptr(),
                    mask, total_extent,
                    dy_ptr, x_ptr, gamma_ptr, rms_ptr, dx_ptr, dgamma_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rms_norm_backward_f64_run(
                    numel, rank, shape.as_ptr(),
                    stride_dy.as_ptr(), stride_x.as_ptr(), stride_rms.as_ptr(), stride_dx.as_ptr(),
                    mask, total_extent,
                    dy_ptr, x_ptr, gamma_ptr, rms_ptr, dx_ptr, dgamma_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RMSNormBackwardPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
