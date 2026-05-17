//! RMSNorm forward plan — multi-axis.
//!
//! `y = x / sqrt(mean(x², over norm_axes) + eps) * gamma`. `gamma` is an
//! optional per-feature affine vector of rank 1 and size equal to the
//! product of the normalized axes' extents. The FW saves the row-RMS to
//! a caller-allocated buffer (`rms`) for BW reuse — shape equals
//! `input_shape` with normalized axes collapsed to 1.
//!
//! ## Multi-axis spec
//!
//! `norm_axes_mask: u8` is a bitmask: bit `d` set means axis `d` is
//! normalized. The mask must be a **suffix** of `[0, N)` — i.e. axes
//! contiguous from the right (PyTorch's `normalized_shape` convention).
//! Single-axis is the special case `mask = 1 << k`.
//!
//! Wired today: `{f32, f16, bf16, f64}` × ranks `1..=8`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, NormalizationKind,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for an RMSNorm forward op.
#[derive(Copy, Clone, Debug)]
pub struct RMSNormDescriptor<const N: usize> {
    /// Tensor shape — input, output, and (with norm axes collapsed to
    /// 1) rms-save buffer all derive from this.
    pub input_shape: [i32; N],
    /// Bitmask of normalized axes. Bit `d` set ⇒ axis `d` is
    /// normalized. Must be a suffix of `[0, N)` (PyTorch
    /// `normalized_shape` convention).
    pub norm_axes_mask: u8,
    /// Epsilon added to `mean(x²)` before the square root.
    pub eps: f32,
    /// Whether `gamma` is supplied as a tensor operand.
    pub has_gamma: bool,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> RMSNormDescriptor<N> {
    /// Shape of the saved-rms buffer: `input_shape` with normalized
    /// axes collapsed to 1.
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

    /// Total extent of the joint normalized region — product of axes
    /// with mask bit set.
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

/// Args bundle for an RMSNorm forward launch.
pub struct RMSNormArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Optional per-feature affine weight — rank-1, size ==
    /// `desc.norm_total_extent()`. `None` when `desc.has_gamma == false`.
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Output tensor — same shape as input.
    pub y: TensorMut<'a, T, N>,
    /// RMS save buffer — shape `desc.rms_shape()`. Required output for
    /// the BW pass.
    pub rms: TensorMut<'a, T, N>,
}

/// RMSNorm forward plan.
pub struct RMSNormPlan<T: Element, const N: usize> {
    desc: RMSNormDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

/// Verify that `mask` is a non-empty suffix of `[0, n)`. Returns Ok if
/// the set bits form `{n-k, n-k+1, ..., n-1}` for some k ≥ 1.
fn check_mask_is_suffix(mask: u8, n: usize) -> bool {
    if mask == 0 {
        return false;
    }
    // Find the lowest set bit.
    let mut lowest = 0usize;
    while lowest < n && ((mask >> lowest) & 1) == 0 {
        lowest += 1;
    }
    if lowest >= n {
        return false;
    }
    // From `lowest` to `n-1`, every bit must be set; above n-1 none.
    for d in lowest..n {
        if (mask >> d) & 1 == 0 {
            return false;
        }
    }
    // Above n: no bits set.
    let upper_bits = mask >> n;
    if upper_bits != 0 {
        return false;
    }
    true
}

impl<T: Element, const N: usize> RMSNormPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &RMSNormDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RMSNormPlan: descriptor element != T",
            ));
        }
        if !check_mask_is_suffix(desc.norm_axes_mask, N) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormPlan: norm_axes_mask must be a non-empty suffix of [0, N)",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RMSNormPlan: shape dims must be non-negative",
                ));
            }
        }
        if N == 0 || N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::RMSNormPlan: tensor rank must be in 1..=8",
            ));
        }
        if !(desc.eps.is_finite() && desc.eps >= 0.0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormPlan: eps must be finite and non-negative",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::RMSNormPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // FW kernel is fully per-cell deterministic — no atomics.
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
    pub fn can_implement(&self, args: &RMSNormArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormPlan: x shape mismatch",
            ));
        }
        if args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormPlan: y shape mismatch",
            ));
        }
        let rms_shape = self.desc.rms_shape();
        if args.rms.shape != rms_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RMSNormPlan: rms shape mismatch with desc.rms_shape()",
            ));
        }
        let total_extent = self.desc.norm_total_extent() as i64;
        match (&args.gamma, self.desc.has_gamma) {
            (Some(g), true) => {
                if g.shape[0] as i64 != total_extent {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::RMSNormPlan: gamma length != norm_total_extent",
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
            (Some(_), false) => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RMSNormPlan: gamma supplied but desc.has_gamma=false",
                ));
            }
            (None, true) => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RMSNormPlan: desc.has_gamma=true but no gamma supplied",
                ));
            }
        }
        let numel = args.x.numel();
        let rms_numel = args.rms.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        let rms_len = args.rms.data.len() as i64;
        if x_len < numel || y_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: x_len.min(y_len) as usize,
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
        args: RMSNormArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.x.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let rms_ptr = args.rms.data.as_raw().0 as *mut c_void;
        let gamma_ptr = match &args.gamma {
            Some(g) => g.data.as_raw().0 as *const c_void,
            None => core::ptr::null(),
        };

        let shape = self.desc.input_shape;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let stride_rms = args.rms.stride;
        let rank = N as i32;
        let mask = self.desc.norm_axes_mask as i32;
        let total_extent = self.desc.norm_total_extent();
        let eps = self.desc.eps;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rms_norm_f32_run(
                    eps, numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(), stride_rms.as_ptr(),
                    mask, total_extent,
                    x_ptr, gamma_ptr, y_ptr, rms_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rms_norm_f16_run(
                    eps, numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(), stride_rms.as_ptr(),
                    mask, total_extent,
                    x_ptr, gamma_ptr, y_ptr, rms_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rms_norm_bf16_run(
                    eps, numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(), stride_rms.as_ptr(),
                    mask, total_extent,
                    x_ptr, gamma_ptr, y_ptr, rms_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_rms_norm_f64_run(
                    eps, numel, rank, shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(), stride_rms.as_ptr(),
                    mask, total_extent,
                    x_ptr, gamma_ptr, y_ptr, rms_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RMSNormPlan::run reached an unimplemented dtype — \
                     select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}

/// Verify that a mask is a suffix of [0, N) — exposed for use by sibling
/// plans (LayerNorm BW etc.) that share the same validation.
pub(crate) fn validate_mask_suffix(mask: u8, n: usize) -> bool {
    check_mask_is_suffix(mask, n)
}
