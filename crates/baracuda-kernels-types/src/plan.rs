//! Plan-layer descriptors shared across kernel families: caller
//! preferences, workspace handles, and the numerical-guarantee record
//! every plan exposes.

use baracuda_driver::DeviceSliceMut;

use crate::element::{ElementKind, MathPrecision};

/// Caller-supplied workspace for a launch.
///
/// Plans never own device memory in baracuda — pass scratch in at
/// `run` time. Pass [`Workspace::None`] for plans whose
/// workspace size is zero.
#[derive(Debug)]
pub enum Workspace<'a> {
    /// No workspace (only valid when the plan reports zero bytes needed).
    None,
    /// Borrowed device scratch. Length must be at least the plan's
    /// reported workspace size.
    Borrowed(DeviceSliceMut<'a, u8>),
}

/// Hints that influence kernel selection inside a plan's `select`
/// method.
///
/// The fields are intentionally generic across kernel families — each
/// op category may layer its own `*PlanPreference` wrapper on top
/// (e.g. `GroupedPlanPreference` adds grouped-specific knobs) that
/// embeds this struct.
#[derive(Copy, Clone, Debug)]
pub struct PlanPreference {
    /// Maximum workspace the caller is willing to provide. The selector
    /// only considers kernels whose workspace size for the descriptor
    /// fits in this budget. Use `usize::MAX` to disable the constraint.
    pub max_workspace_bytes: usize,
    /// Allow Hopper-specialized (`sm_90a`) kernels in selection. Has no
    /// effect when the `sm90a` feature is off in the underlying kernel
    /// crate (no such kernels exist in the build).
    pub allow_sm90a: bool,
}

impl Default for PlanPreference {
    fn default() -> Self {
        Self {
            max_workspace_bytes: usize::MAX,
            allow_sm90a: true,
        }
    }
}

/// Numerical guarantees a kernel provides.
///
/// Surfaces the salient numerical properties consumers need to decide
/// whether a kernel SKU satisfies an op's precision contract — without
/// having to re-derive them from documentation per kernel.
///
/// All fields are intentionally cheap to compare so this struct can be
/// hashed into selection / autotuner caches.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PrecisionGuarantee {
    /// Bit-precision used inside the math instruction.
    pub math_precision: MathPrecision,
    /// Element type of the multiply-accumulate accumulator.
    pub accumulator: ElementKind,
    /// Whether the kernel produces bit-identical results across runs on
    /// the same hardware with the same inputs.
    ///
    /// `false` for tensor-core kernels (F16, BF16, TF32) because the
    /// warp-level reduction order isn't fixed by the spec — adjacent
    /// runs can differ in the last bit even with the same inputs.
    /// `true` for SIMT F32 and for integer kernels.
    pub bit_stable_on_same_hardware: bool,
    /// Whether the kernel produces bit-identical results across runs
    /// from a single thread within a process — i.e. it has no internal
    /// nondeterminism (no atomic accumulation across blocks, no random
    /// tile-schedule decisions).
    pub deterministic: bool,
}
