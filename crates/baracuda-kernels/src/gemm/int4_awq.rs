//! AWQ W4A16 GEMM plan — Phase 48 Goal B.
//!
//! Vendored kernel: `crates/baracuda-kernels-sys/vendor/awq/`
//! (mit-han-lab/llm-awq, MIT — no patent grant). Gated behind the
//! `awq` cargo feature on both `baracuda-kernels-sys` and
//! `baracuda-kernels`.
//!
//! ## What is AWQ?
//!
//! AWQ (Activation-aware Weight Quantization) is the **most-deployed
//! 4-bit format on the Hugging Face Hub**. Llama / Mistral / Qwen
//! prebuilts published as `*-AWQ` use this format. The Phase 48
//! native AWQ kernel lets baracuda load these checkpoints directly
//! without a Marlin-style repack.
//!
//! ## Distinction from sibling 4-bit families
//!
//! - **AWQ (this plan)** — **asymmetric** int4: explicit per-group
//!   zero-points stored as packed int4 (8 zero-points per int32 word).
//!   Loads directly from HF `*-AWQ` checkpoints.
//! - **Marlin ([`super::int4_marlin::Int4MarlinGemmPlan`])** —
//!   **symmetric** int4: zero-points absorbed into scales by the
//!   packer. Faster on decode-batch (M ∈ [1, 64]) but requires a
//!   non-trivial repack from any non-Marlin checkpoint.
//! - **GPTQ-quantized weights** — asymmetric int4 in a DIFFERENT
//!   layout from AWQ. Cannot be directly fed into this plan.
//!
//! ## Scope
//!
//! - **Activation / output dtype**: `f16` only. Upstream AWQ's
//!   dequant magic-number trick is fp16-specific; bf16 deferred.
//! - **Group size**: 64 or 128 (upstream supports both;
//!   configurable per descriptor).
//! - **OC alignment**: OC must be divisible by 64.
//! - **IC alignment**: IC must be divisible by `group_size` and by
//!   `32 * split_k_iters` (the kernel's K-tile bound).
//! - **Hardware**: sm_80+ (Ampere or newer; uses `ldmatrix.sync` +
//!   `mma.sync.m16n8k16`).
//! - **No backwards** — inference-only by convention.
//!
//! ## Numerical contract
//!
//! AWQ has worse accuracy than Marlin at the same bitwidth because
//! the asymmetric int4 layout commits more bits to zero-point
//! storage; the AWQ paper reports the activation-aware
//! pre-quantization step partly compensates. Relative error vs
//! FP16 GEMM is on the order of `~1e-2` (the standard 4-bit
//! quantization error class).
//!
//! Accumulator is f32; output is fp16 cast at store time. The
//! kernel uses a **split-k strategy**: the launcher allocates a
//! `[split_k_iters, padded_M, OC]` staging buffer, launches the
//! kernel that writes per-split partial sums, then runs an
//! axis-0 reduce-sum into the final `[M, OC]` output.
//!
//! Deterministic and bit-stable on identical hardware. No
//! atomic adds (the split-k reduce is a separate kernel pass).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace,
};
use half::f16;

use crate::quantize::map_status;

// =============================================================================
// Sealed activation trait
// =============================================================================

/// Sealed marker for activation / destination dtypes accepted by the
/// AWQ plan. Phase 48 scope is `f16` only.
pub trait AwqActivation: Element + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for half::f16 {}
}

impl AwqActivation for f16 {}

// =============================================================================
// Descriptor
// =============================================================================

/// AWQ GEMM problem descriptor.
///
/// Construct via [`Int4AwqGemmDescriptor::new`] and tune with the
/// `with_*` setters per the Phase 32 builder convention.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct Int4AwqGemmDescriptor {
    /// Output row count (= activation batch / sequence dim).
    pub m: i32,
    /// Inner / contraction dim (= activation feature dim = weight
    /// input dim). Must be a multiple of `group_size` AND of
    /// `32 * split_k_iters`.
    pub ic: i32,
    /// Weight output dim. Must be divisible by 64 (the kernel's
    /// `cta_N` tile bound).
    pub oc: i32,
    /// Quantization group size. Either 64 or 128 (upstream
    /// supports both; the kernel templates on `G`).
    pub group_size: i32,
    /// Split-K iteration count. Caller-chosen; upstream default is
    /// 8. Trade-off: higher → more parallelism (better at large
    /// M), more workspace, slightly more reduce overhead. Lower →
    /// less workspace, less reduce overhead, but worse occupancy.
    pub split_k_iters: i32,
}

impl Int4AwqGemmDescriptor {
    /// HF-AWQ-aligned defaults: group_size = 128 (the dominant
    /// HF setting; alternative is 64), split_k_iters = 8 (matches
    /// the upstream Python reference).
    pub fn new(m: i32, ic: i32, oc: i32) -> Self {
        Self {
            m,
            ic,
            oc,
            group_size: 128,
            split_k_iters: 8,
        }
    }

    /// Override the quantization group size. Must be `64` or `128`.
    #[must_use]
    pub fn with_group_size(mut self, g: i32) -> Self {
        self.group_size = g;
        self
    }

    /// Override the split-K iteration count.
    #[must_use]
    pub fn with_split_k_iters(mut self, s: i32) -> Self {
        self.split_k_iters = s;
        self
    }
}

// =============================================================================
// Args
// =============================================================================

/// Per-launch arguments for [`Int4AwqGemmPlan::run`].
pub struct Int4AwqGemmArgs<'a, T: AwqActivation> {
    /// Activation `[M, IC]` row-major in `T`.
    pub activation: TensorRef<'a, T, 2>,
    /// AWQ-packed int4 weights. Shape `[OC, IC/8]` of `i32` storage
    /// (8 packed nibbles per int32 word, OC-major IC-minor). This is
    /// the transpose of the naive `[K, N]` layout — note when
    /// reshaping from a flat HF checkpoint buffer.
    pub weight_packed: TensorRef<'a, i32, 2>,
    /// Per-group scales `[IC/group_size, OC]` in `T`.
    pub scales: TensorRef<'a, T, 2>,
    /// Per-group packed int4 zero-points. Shape `[IC/group_size, OC/8]`
    /// of `i32` storage (8 zero-points per int32 word).
    pub zeros: TensorRef<'a, i32, 2>,
    /// Output `[M, OC]` row-major in `T`.
    pub output: TensorMut<'a, T, 2>,
}

// =============================================================================
// Plan
// =============================================================================

/// AWQ int4 W4A16 GEMM. fp16-only.
///
/// Use for direct inference on HF `*-AWQ` checkpoints. For the
/// decode-batch hot path on already-repacked symmetric weights,
/// prefer [`super::int4_marlin::Int4MarlinGemmPlan`] (faster at
/// M ∈ [1, 64]).
pub struct Int4AwqGemmPlan<T: AwqActivation> {
    desc: Int4AwqGemmDescriptor,
    sku: KernelSku,
    _phantom: PhantomData<T>,
}

impl<T: AwqActivation> Int4AwqGemmPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &Int4AwqGemmDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.m < 0 || desc.ic <= 0 || desc.oc <= 0 {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: M must be non-negative; IC and OC must be positive",
            ));
        }
        if desc.group_size != 64 && desc.group_size != 128 {
            return Err(Error::Unsupported(
                "Int4AwqGemmPlan: group_size must be 64 or 128",
            ));
        }
        if desc.split_k_iters <= 0 {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: split_k_iters must be positive",
            ));
        }
        if desc.oc % 64 != 0 {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: OC must be divisible by 64 (kernel cta_N tile)",
            ));
        }
        if desc.ic % desc.group_size != 0 {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: IC must be divisible by group_size",
            ));
        }
        if desc.ic % (32 * desc.split_k_iters) != 0 {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: IC must be divisible by 32 * split_k_iters",
            ));
        }
        if !matches!(T::KIND, ElementKind::F16) {
            return Err(Error::Unsupported(
                "Int4AwqGemmPlan: activation dtype must be f16 (upstream is fp16-only)",
            ));
        }
        Ok(Self {
            desc: *desc,
            sku: build_sku(T::KIND),
            _phantom: PhantomData,
        })
    }

    /// Validate args against the plan.
    pub fn can_implement(&self, args: &Int4AwqGemmArgs<'_, T>) -> Result<()> {
        let m = self.desc.m;
        let ic = self.desc.ic;
        let oc = self.desc.oc;
        let g = self.desc.group_size;

        if args.activation.shape != [m, ic] {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: activation shape != [M, IC]",
            ));
        }
        if args.activation.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: activation must be contig along IC",
            ));
        }

        // Packed weight: [OC, IC/8].
        if args.weight_packed.shape != [oc, ic / 8] {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: weight_packed shape != [OC, IC/8] (i32 storage)",
            ));
        }

        // Scales: [IC/g, OC].
        if args.scales.shape != [ic / g, oc] {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: scales shape != [IC/group_size, OC]",
            ));
        }

        // Zeros: [IC/g, OC/8].
        if args.zeros.shape != [ic / g, oc / 8] {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: zeros shape != [IC/group_size, OC/8] (i32 storage)",
            ));
        }

        if args.output.shape != [m, oc] {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: output shape != [M, OC]",
            ));
        }
        if args.output.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "Int4AwqGemmPlan: output must be contig along OC",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — the split-k staging buffer. Sized as
    /// `split_k_iters * ceil(M, 128) * 128 * OC * sizeof(__half)`.
    /// Returns 0 when M == 0.
    pub fn workspace_size(&self) -> usize {
        if self.desc.m <= 0 || self.desc.oc <= 0 || self.desc.split_k_iters <= 0 {
            return 0;
        }
        let padded_m = ((self.desc.m as i64 + 127) / 128) * 128;
        (self.desc.split_k_iters as usize) * (padded_m as usize) * (self.desc.oc as usize) * 2
        // 2 = sizeof(__half) = sizeof(half::f16)
    }

    /// Identity of the selected kernel.
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
        args: Int4AwqGemmArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.m == 0 {
            return Ok(());
        }

        let need = self.workspace_size();
        let (ws_ptr, ws_bytes) = match workspace {
            Workspace::Borrowed(buf) => (
                buf.as_raw().0 as *mut c_void,
                buf.len(),
            ),
            Workspace::None => (core::ptr::null_mut(), 0usize),
        };
        if ws_bytes < need {
            return Err(Error::WorkspaceTooSmall {
                needed: need,
                got: ws_bytes,
            });
        }

        let a_ptr = args.activation.data.as_raw().0 as *const c_void;
        let w_ptr = args.weight_packed.data.as_raw().0 as *const c_void;
        let s_ptr = args.scales.data.as_raw().0 as *const c_void;
        let z_ptr = args.zeros.data.as_raw().0 as *const c_void;
        let o_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = unsafe {
            dispatch_awq::<T>(
                self.desc.m,
                self.desc.ic,
                self.desc.oc,
                self.desc.group_size,
                self.desc.split_k_iters,
                a_ptr,
                w_ptr,
                s_ptr,
                z_ptr,
                o_ptr,
                ws_ptr,
                ws_bytes,
                stream_ptr,
            )
        };
        map_status(status)
    }
}

// =============================================================================
// Internal helpers
// =============================================================================

fn build_sku(act_kind: ElementKind) -> KernelSku {
    KernelSku {
        category: OpCategory::Quantization,
        op: QuantizeKind::GgufMmvq as u16,
        element: act_kind,
        aux_element: Some(ElementKind::U8),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee: PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        },
    }
}

#[cfg(feature = "awq")]
#[inline]
unsafe fn dispatch_awq<T: AwqActivation>(
    m: i32,
    ic: i32,
    oc: i32,
    group_size: i32,
    split_k_iters: i32,
    a: *const c_void,
    w: *const c_void,
    s: *const c_void,
    z: *const c_void,
    out: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_int4_awq_gemm_f16_run(
                m, ic, oc, group_size, split_k_iters,
                a, w, s, z, out, workspace, workspace_bytes, stream,
            )
        },
        _ => 3,
    }
}

#[cfg(not(feature = "awq"))]
#[inline]
unsafe fn dispatch_awq<T: AwqActivation>(
    _: i32, _: i32, _: i32, _: i32, _: i32,
    _: *const c_void, _: *const c_void, _: *const c_void, _: *const c_void,
    _: *mut c_void, _: *mut c_void, _: usize, _: *mut c_void,
) -> i32 {
    // awq cargo feature is off; FFI symbol absent.
    3
}
