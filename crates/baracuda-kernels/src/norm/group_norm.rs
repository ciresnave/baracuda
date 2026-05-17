//! GroupNorm forward plan.
//!
//! Splits the channel axis into `num_groups` groups; normalizes per
//! `(sample, group, *spatial*)`. `num_groups` must divide `num_channels`.
//! Per-channel affine (`gamma`, `beta`) is applied after the
//! normalization.
//!
//! `InstanceNorm` is the special case `num_groups == num_channels` —
//! see [`super::instance_norm::InstanceNormPlan`] for the sugar wrapper.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, NormalizationKind,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::rms_norm::map_status;

/// Descriptor for a GroupNorm forward op.
#[derive(Copy, Clone, Debug)]
pub struct GroupNormDescriptor<const N: usize> {
    /// Input tensor shape `[N, C, *spatial]`. Rank ≥ 2.
    pub input_shape: [i32; N],
    /// Channel axis (must be 1 in this trailblazer).
    pub channel_axis: u8,
    /// Number of groups. Must divide `num_channels` evenly.
    pub num_groups: u32,
    /// Epsilon for variance stabilization.
    pub eps: f32,
    /// Whether gamma + beta participate.
    pub has_affine: bool,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> GroupNormDescriptor<N> {
    /// Number of channels.
    #[inline]
    pub fn num_channels(&self) -> i32 {
        if N >= 2 { self.input_shape[self.channel_axis as usize] } else { 1 }
    }
}

/// Args bundle for GroupNorm FW.
pub struct GroupNormArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Per-channel gamma `[C]`.
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Per-channel beta `[C]`.
    pub beta: Option<TensorRef<'a, T, 1>>,
    /// Output tensor.
    pub y: TensorMut<'a, T, N>,
    /// Per-`(N, group)` saved mean — length == `N * num_groups`.
    pub saved_mean: TensorMut<'a, T, 1>,
    /// Per-`(N, group)` saved inv_std — length == `N * num_groups`.
    pub saved_rstd: TensorMut<'a, T, 1>,
}

/// GroupNorm forward plan.
pub struct GroupNormPlan<T: Element, const N: usize> {
    desc: GroupNormDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> GroupNormPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &GroupNormDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported("baracuda-kernels::GroupNormPlan: descriptor element != T"));
        }
        if N < 2 || N > 8 {
            return Err(Error::Unsupported("baracuda-kernels::GroupNormPlan: rank must be in 2..=8"));
        }
        if desc.channel_axis != 1 {
            return Err(Error::Unsupported(
                "baracuda-kernels::GroupNormPlan: channel_axis must be 1 in this trailblazer",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::GroupNormPlan: shape dims must be non-negative",
                ));
            }
        }
        if desc.num_groups == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GroupNormPlan: num_groups must be > 0",
            ));
        }
        let c = desc.num_channels();
        if c <= 0 || (c as u32) % desc.num_groups != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GroupNormPlan: num_groups must divide num_channels",
            ));
        }
        if !(desc.eps.is_finite() && desc.eps >= 0.0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GroupNormPlan: eps must be finite and non-negative",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::GroupNormPlan: wired today: `{f32, f16, bf16, f64}`",
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
            op: NormalizationKind::GroupNorm as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self { desc: *desc, sku, _marker: PhantomData })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &GroupNormArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape || args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem("baracuda-kernels::GroupNormPlan: x / y shape mismatch"));
        }
        let n = self.desc.input_shape[0] as i64;
        let c = self.desc.num_channels() as i64;
        let g = self.desc.num_groups as i64;
        let group_count = n * g;
        if args.saved_mean.shape[0] as i64 != group_count
            || args.saved_rstd.shape[0] as i64 != group_count
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GroupNormPlan: saved buffers length != N * num_groups",
            ));
        }
        if let Some(gm) = &args.gamma {
            if gm.shape[0] as i64 != c {
                return Err(Error::InvalidProblem("baracuda-kernels::GroupNormPlan: gamma length != C"));
            }
        }
        if let Some(bt) = &args.beta {
            if bt.shape[0] as i64 != c {
                return Err(Error::InvalidProblem("baracuda-kernels::GroupNormPlan: beta length != C"));
            }
        }
        match (args.gamma.is_some(), args.beta.is_some(), self.desc.has_affine) {
            (true, true, true) | (false, false, false) => {}
            _ => {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::GroupNormPlan: gamma + beta must both be present iff has_affine",
                ));
            }
        }
        Ok(())
    }

    /// Workspace bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize { 0 }
    /// Kernel SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku { self.sku }
    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee { self.sku.precision_guarantee }

    /// Launch (also serves InstanceNormPlan via num_groups == c_extent).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: GroupNormArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let n_extent = self.desc.input_shape[0];
        let c_extent = self.desc.input_shape[1];
        let mut s_extent: i32 = 1;
        for d in 2..N { s_extent = s_extent.saturating_mul(self.desc.input_shape[d]); }
        if n_extent == 0 || c_extent == 0 || s_extent == 0 { return Ok(()); }
        let num_groups = self.desc.num_groups as i32;

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
                baracuda_kernels_sys::baracuda_kernels_group_norm_f32_run(
                    n_extent, c_extent, s_extent, num_groups, 1, self.desc.eps,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, rstd_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_group_norm_f16_run(
                    n_extent, c_extent, s_extent, num_groups, 1, self.desc.eps,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, rstd_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_group_norm_bf16_run(
                    n_extent, c_extent, s_extent, num_groups, 1, self.desc.eps,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, rstd_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_group_norm_f64_run(
                    n_extent, c_extent, s_extent, num_groups, 1, self.desc.eps,
                    x_ptr, gamma_ptr, beta_ptr, y_ptr, mean_ptr, rstd_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GroupNormPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
