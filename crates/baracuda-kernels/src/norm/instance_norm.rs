//! InstanceNorm forward plan.
//!
//! Per-`(sample, channel)` normalization across spatial dims. Equivalent
//! to GroupNorm with `num_groups == num_channels` — this plan is sugar
//! that builds a [`super::group_norm::GroupNormPlan`] internally with
//! that setting. **Same kernel symbols** dispatch behind the scenes
//! (no separate `.cu` file).
//!
//! ## Why a separate plan?
//!
//! PyTorch ships `InstanceNorm1d/2d/3d` as their own modules — the API
//! split matches their layer-shape semantics and lets callers be
//! explicit about intent. Internally the kernel is identical to
//! `GroupNorm(num_groups=C)`.

use baracuda_cutlass::Result;
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef,
    Workspace,
};

use super::group_norm::{GroupNormArgs, GroupNormDescriptor, GroupNormPlan};

/// Descriptor for an InstanceNorm forward op.
#[derive(Copy, Clone, Debug)]
pub struct InstanceNormDescriptor<const N: usize> {
    /// Input tensor shape `[N, C, *spatial]`.
    pub input_shape: [i32; N],
    /// Channel axis (must equal 1).
    pub channel_axis: u8,
    /// Epsilon.
    pub eps: f32,
    /// Whether gamma + beta participate.
    pub has_affine: bool,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for InstanceNorm FW.
pub struct InstanceNormArgs<'a, T: Element, const N: usize> {
    /// Input.
    pub x: TensorRef<'a, T, N>,
    /// Per-channel gamma.
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Per-channel beta.
    pub beta: Option<TensorRef<'a, T, 1>>,
    /// Output.
    pub y: TensorMut<'a, T, N>,
    /// Saved per-`(N, C)` mean — length == `N * C`.
    pub saved_mean: TensorMut<'a, T, 1>,
    /// Saved per-`(N, C)` inv_std — length == `N * C`.
    pub saved_rstd: TensorMut<'a, T, 1>,
}

/// InstanceNorm forward plan. Thin wrapper over [`GroupNormPlan`] with
/// `num_groups == num_channels`.
pub struct InstanceNormPlan<T: Element, const N: usize> {
    inner: GroupNormPlan<T, N>,
}

impl<T: Element, const N: usize> InstanceNormPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        stream: &Stream,
        desc: &InstanceNormDescriptor<N>,
        pref: PlanPreference,
    ) -> Result<Self> {
        let c = if N >= 2 { desc.input_shape[desc.channel_axis as usize] } else { 1 };
        let inner_desc = GroupNormDescriptor::<N> {
            input_shape: desc.input_shape,
            channel_axis: desc.channel_axis,
            num_groups: c.max(1) as u32,
            eps: desc.eps,
            has_affine: desc.has_affine,
            element: desc.element,
        };
        let inner = GroupNormPlan::<T, N>::select(stream, &inner_desc, pref)?;
        Ok(Self { inner })
    }

    /// Workspace bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize { self.inner.workspace_size() }
    /// Kernel SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku { self.inner.sku() }
    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee { self.inner.precision_guarantee() }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: InstanceNormArgs<'_, T, N>,
    ) -> Result<()> {
        let inner_args = GroupNormArgs::<T, N> {
            x: args.x,
            gamma: args.gamma,
            beta: args.beta,
            y: args.y,
            saved_mean: args.saved_mean,
            saved_rstd: args.saved_rstd,
        };
        self.inner.run(stream, workspace, inner_args)
    }
}
