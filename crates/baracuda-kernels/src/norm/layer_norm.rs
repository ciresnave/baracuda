//! LayerNorm forward plan — multi-axis.
//!
//! `y = (x - mean) / sqrt(var + eps) * gamma + beta` where mean and var
//! are taken over the normalized axes (biased / "population" variance,
//! matching PyTorch's `torch.nn.LayerNorm`). `gamma` and `beta` are
//! independently optional (PyTorch's `elementwise_affine=False`).
//!
//! ## Multi-axis spec
//!
//! `norm_axes_mask: u8` is a bitmask: bit `d` set means axis `d` is
//! normalized. Must be a **suffix** of `[0, N)` (PyTorch convention).
//! Single-axis is `mask = 1 << k`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, NormalizationKind,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::rms_norm::{map_status, validate_mask_suffix};

/// Descriptor for a LayerNorm forward op.
#[derive(Copy, Clone, Debug)]
pub struct LayerNormDescriptor<const N: usize> {
    /// Tensor shape.
    pub input_shape: [i32; N],
    /// Bitmask of normalized axes. Bit `d` set ⇒ axis `d` is
    /// normalized. Must be a suffix of `[0, N)`.
    pub norm_axes_mask: u8,
    /// Epsilon added to variance before the square root.
    pub eps: f32,
    /// Whether `gamma` is supplied.
    pub has_gamma: bool,
    /// Whether `beta` is supplied.
    pub has_beta: bool,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> LayerNormDescriptor<N> {
    /// Shape of the save buffers (`mean`, `inv_std`).
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

/// Args bundle for a LayerNorm forward launch.
pub struct LayerNormArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Per-feature affine weight — `None` when `desc.has_gamma == false`.
    /// Length == `desc.norm_total_extent()`.
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Per-feature affine bias.
    pub beta: Option<TensorRef<'a, T, 1>>,
    /// Output tensor.
    pub y: TensorMut<'a, T, N>,
    /// Save buffer for per-row mean — shape `desc.save_shape()`.
    pub mean: TensorMut<'a, T, N>,
    /// Save buffer for per-row inverse standard deviation. Must share
    /// `mean`'s stride.
    pub inv_std: TensorMut<'a, T, N>,
}

/// LayerNorm forward plan.
pub struct LayerNormPlan<T: Element, const N: usize> {
    desc: LayerNormDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> LayerNormPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &LayerNormDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::LayerNormPlan: descriptor element != T",
            ));
        }
        if !validate_mask_suffix(desc.norm_axes_mask, N) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormPlan: norm_axes_mask must be a non-empty suffix \
                 of [0, N)",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::LayerNormPlan: shape dims must be non-negative",
                ));
            }
        }
        if N == 0 || N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::LayerNormPlan: tensor rank must be in 1..=8",
            ));
        }
        if !(desc.eps.is_finite() && desc.eps >= 0.0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormPlan: eps must be finite and non-negative",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::LayerNormPlan: wired today: `{f32, f16, bf16, f64}`",
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
    pub fn can_implement(&self, args: &LayerNormArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormPlan: x shape mismatch",
            ));
        }
        if args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormPlan: y shape mismatch",
            ));
        }
        let save_shape = self.desc.save_shape();
        if args.mean.shape != save_shape || args.inv_std.shape != save_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormPlan: mean / inv_std shape mismatch",
            ));
        }
        if args.mean.stride != args.inv_std.stride {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::LayerNormPlan: mean and inv_std must share stride",
            ));
        }
        let total_extent = self.desc.norm_total_extent() as i64;
        match (&args.gamma, self.desc.has_gamma) {
            (Some(g), true) => {
                if g.shape[0] as i64 != total_extent {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::LayerNormPlan: gamma length != norm_total_extent",
                    ));
                }
                if (g.data.len() as i64) < total_extent {
                    return Err(Error::BufferTooSmall {
                        needed: total_extent as usize,
                        got: g.data.len(),
                    });
                }
            }
            (None, false) => {}
            _ => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::LayerNormPlan: gamma supplied iff desc.has_gamma=true",
                ));
            }
        }
        match (&args.beta, self.desc.has_beta) {
            (Some(b), true) => {
                if b.shape[0] as i64 != total_extent {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::LayerNormPlan: beta length != norm_total_extent",
                    ));
                }
                if (b.data.len() as i64) < total_extent {
                    return Err(Error::BufferTooSmall {
                        needed: total_extent as usize,
                        got: b.data.len(),
                    });
                }
            }
            (None, false) => {}
            _ => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::LayerNormPlan: beta supplied iff desc.has_beta=true",
                ));
            }
        }
        let numel = args.x.numel();
        let save_numel = args.mean.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        let mean_len = args.mean.data.len() as i64;
        let std_len = args.inv_std.data.len() as i64;
        if x_len < numel || y_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: x_len.min(y_len) as usize,
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
        args: LayerNormArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.x.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let mean_ptr = args.mean.data.as_raw().0 as *mut c_void;
        let std_ptr = args.inv_std.data.as_raw().0 as *mut c_void;
        let gamma_ptr = match &args.gamma {
            Some(g) => g.data.as_raw().0 as *const c_void,
            None => core::ptr::null(),
        };
        let beta_ptr = match &args.beta {
            Some(b) => b.data.as_raw().0 as *const c_void,
            None => core::ptr::null(),
        };

        let shape = self.desc.input_shape;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let stride_save = args.mean.stride;
        let rank = N as i32;
        let mask = self.desc.norm_axes_mask as i32;
        let total_extent = self.desc.norm_total_extent();
        let eps = self.desc.eps;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_layer_norm_f32_run(
                    eps, numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(), stride_save.as_ptr(),
                    mask, total_extent,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, std_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_layer_norm_f16_run(
                    eps, numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(), stride_save.as_ptr(),
                    mask, total_extent,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, std_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_layer_norm_bf16_run(
                    eps, numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(), stride_save.as_ptr(),
                    mask, total_extent,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, std_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_layer_norm_f64_run(
                    eps, numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(), stride_save.as_ptr(),
                    mask, total_extent,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, std_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::LayerNormPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
