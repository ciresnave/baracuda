//! InstanceNorm backward plan — thin wrapper over GroupNorm BW.

use baracuda_cutlass::Result;
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef,
    Workspace,
};

use super::group_norm_backward::{
    GroupNormBackwardArgs, GroupNormBackwardDescriptor, GroupNormBackwardPlan,
};

/// Descriptor for an InstanceNorm BW op.
#[derive(Copy, Clone, Debug)]
pub struct InstanceNormBackwardDescriptor<const N: usize> {
    /// Input shape `[N, C, ...]`.
    pub input_shape: [i32; N],
    /// Channel axis (must equal 1).
    pub channel_axis: u8,
    /// Affine.
    pub has_affine: bool,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for InstanceNorm BW.
pub struct InstanceNormBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input.
    pub x: TensorRef<'a, T, N>,
    /// Per-channel gamma.
    pub gamma: Option<TensorRef<'a, T, 1>>,
    /// Saved per-`(N, C)` mean.
    pub saved_mean: TensorRef<'a, T, 1>,
    /// Saved per-`(N, C)` inv_std.
    pub saved_rstd: TensorRef<'a, T, 1>,
    /// Gradient w.r.t. forward input.
    pub dx: TensorMut<'a, T, N>,
    /// Gradient w.r.t. gamma.
    pub dgamma: Option<TensorMut<'a, T, 1>>,
    /// Gradient w.r.t. beta.
    pub dbeta: Option<TensorMut<'a, T, 1>>,
}

/// InstanceNorm BW plan — wraps [`GroupNormBackwardPlan`].
pub struct InstanceNormBackwardPlan<T: Element, const N: usize> {
    inner: GroupNormBackwardPlan<T, N>,
}

impl<T: Element, const N: usize> InstanceNormBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        stream: &Stream,
        desc: &InstanceNormBackwardDescriptor<N>,
        pref: PlanPreference,
    ) -> Result<Self> {
        let c = if N >= 2 { desc.input_shape[desc.channel_axis as usize] } else { 1 };
        let inner_desc = GroupNormBackwardDescriptor::<N> {
            input_shape: desc.input_shape,
            channel_axis: desc.channel_axis,
            num_groups: c.max(1) as u32,
            has_affine: desc.has_affine,
            element: desc.element,
        };
        let inner = GroupNormBackwardPlan::<T, N>::select(stream, &inner_desc, pref)?;
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
        args: InstanceNormBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        let inner_args = GroupNormBackwardArgs::<T, N> {
            dy: args.dy,
            x: args.x,
            gamma: args.gamma,
            saved_mean: args.saved_mean,
            saved_rstd: args.saved_rstd,
            dx: args.dx,
            dgamma: args.dgamma,
            dbeta: args.dbeta,
        };
        self.inner.run(stream, workspace, inner_args)
    }
}
