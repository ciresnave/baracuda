//! Layout / arch / epilogue / activation tags shared across kernel
//! families.
//!
//! These are pure descriptor enums that don't carry generic parameters;
//! they appear in plan descriptors, in [`crate::KernelSku`] (TBD) /
//! `GemmSku`, and in selector preference fields.

/// Layout SKU. Describes the row/column orientation of A, B, C, and D
/// for matrix-multiply-shaped kernels.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum LayoutSku {
    /// `A` row-major `[M, K]`, `B` column-major `[K, N]`, `C/D` row-major `[M, N]`.
    ///
    /// Useful when a row-major weight tensor stored as `[N, K]` is
    /// reinterpreted as logical column-major `B = [K, N]` without a
    /// transpose copy.
    Rcr,
    /// `A` row-major `[M, K]`, `B` row-major `[K, N]`, `C/D` row-major `[M, N]`.
    ///
    /// The natural shape for activation-row-major @ weight-row-major
    /// matmul (the typical ML graph layout). No transpose pass needed
    /// before launch — both operands stored in their native row-major
    /// form.
    Rrr,
}

/// Compute capability bucket the selected kernel was compiled for.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ArchSku {
    /// Ampere (also runs on Ada and as forward-compatible fallback on Hopper).
    Sm80,
    /// Ada Lovelace specializations (FP8 tensor cores). Requires `sm89`
    /// feature in the consuming kernel crate.
    Sm89,
    /// Hopper-specialized (requires `sm90a` feature in the consuming
    /// kernel crate).
    Sm90a,
}

/// Epilogue applied after the matrix-multiply accumulation.
///
/// The four `Bias*` variants share one kernel family: they all fuse the
/// bias add into the output epilogue and additionally apply the named
/// activation function before the store. `BiasRelu`, `BiasGelu`, and
/// `BiasSilu` therefore deliver the full `y = activation(W·x + b)`
/// transformer-Linear pipeline in a single kernel pass — no extra memory
/// traffic vs plain `Bias`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum EpilogueKind {
    /// `D = α · (A · B) + β · C` (no activation, no bias).
    Identity,
    /// `D = α · (A · B) + β · C + bias_broadcast(N)`. The bias vector
    /// has length `N` (one element per output column) and is broadcast
    /// across rows.
    Bias,
    /// `D = relu(α · (A · B) + β · C + bias_broadcast(N))`.
    /// `relu(x) = max(x, 0)`.
    BiasRelu,
    /// `D = gelu(α · (A · B) + β · C + bias_broadcast(N))` using the
    /// exact (erf-based) GELU — matches PyTorch's default `nn.GELU()`.
    BiasGelu,
    /// `D = silu(α · (A · B) + β · C + bias_broadcast(N))` where
    /// `silu(x) = x · sigmoid(x)`. Also known as Swish-1.
    BiasSilu,
}

impl EpilogueKind {
    /// `true` if a bias broadcast must be supplied for this epilogue.
    /// Equivalent to "any `Bias*` variant".
    #[inline]
    pub const fn requires_bias(self) -> bool {
        matches!(
            self,
            Self::Bias | Self::BiasRelu | Self::BiasGelu | Self::BiasSilu,
        )
    }

    /// Activation function this epilogue applies after the linear
    /// combination, if any.
    ///
    /// Returns `None` for [`Identity`](Self::Identity) and
    /// [`Bias`](Self::Bias) (both apply no activation); returns the
    /// corresponding [`ActivationKind`] for the `Bias*Activation`
    /// variants.
    #[inline]
    pub const fn activation(self) -> Option<ActivationKind> {
        match self {
            Self::Identity | Self::Bias => None,
            Self::BiasRelu => Some(ActivationKind::Relu),
            Self::BiasGelu => Some(ActivationKind::Gelu),
            Self::BiasSilu => Some(ActivationKind::Silu),
        }
    }
}

/// Activation functions implemented by the `Bias*Activation`
/// [`EpilogueKind`] variants. Surfaced for telemetry and selector
/// logic; the kernel selection itself is driven by the enum variant.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ActivationKind {
    /// `relu(x) = max(x, 0)`.
    Relu,
    /// Exact (erf-based) Gaussian Error Linear Unit. Matches
    /// PyTorch's default `nn.GELU()`.
    Gelu,
    /// `silu(x) = x · sigmoid(x)`. Also known as Swish-1.
    Silu,
}
