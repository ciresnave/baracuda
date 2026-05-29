//! Marlin W4A16 GEMM plan — Phase 48 Goal A.
//!
//! Vendored kernel: `crates/baracuda-kernels-sys/vendor/marlin/`
//! (IST-DASLab/marlin, Apache-2.0 + §3 patent grant). Gated behind
//! the `marlin` cargo feature on both `baracuda-kernels-sys` and
//! `baracuda-kernels`.
//!
//! ## What is Marlin?
//!
//! Marlin is a **state-of-the-art** W4A16 GEMM kernel optimised for
//! the decode-batch regime (M ∈ [1, 64]) on Ampere / Ada GPUs. The
//! upstream paper reports ~3.87× speedup over FP16 GEMM at batch
//! sizes 1-32. The kernel uses `mma.sync.m16n8k16` tensor-core
//! instructions plus async `cp.async.cg.shared.global` weight loads,
//! with a custom packed-int4 layout designed for tensor-core fragment
//! alignment.
//!
//! ## Distinction from sibling 4-bit families
//!
//! - **Marlin (this plan)** — **symmetric** int4: the zero-point is
//!   fused into the dequant by subtracting 8 from the unsigned 4-bit
//!   value at dequant time. No per-group zero-points stored.
//! - **AWQ ([`super::int4_awq::Int4AwqGemmPlan`])** — **asymmetric**
//!   int4: explicit per-group zero-points stored alongside scales.
//!   Directly loadable from HF `*-AWQ` checkpoints.
//! - **GPTQ-quantized weights** — asymmetric int4. Use the
//!   [`super::gptq_to_marlin::repack`] utility to convert
//!   GPTQ → Marlin format at model load time (absorbs the zero-point
//!   into the scale).
//!
//! ## Scope
//!
//! - **Activation / output dtype**: `f16` only. Upstream Marlin is
//!   fp16-only; the dequant magic-number trick is fp16-specific.
//!   bf16 deferred (out of scope for Phase 48).
//! - **Group size**: `-1` (per-channel) or `128`. Other values
//!   rejected at `select`.
//! - **K alignment**: K must be divisible by 128.
//! - **N alignment**: N must be divisible by 256.
//! - **Hardware**: sm_80 / sm_86 / sm_89 (Ampere + Ada). sm_90
//!   (Hopper) is **rejected** at `select` — Marlin requires a WGMMA
//!   rewrite for Hopper (Marlin v2 territory, not in this phase).
//! - **No backwards** — inference-only by convention.
//!
//! ## Numerical contract
//!
//! Marlin operates at ~1-2 ppm MMLU-class accuracy relative to FP16
//! GEMM on properly-quantized weights. The accumulator is f32 (mma
//! instructions accumulate in fp32). Output is fp16 cast at store
//! time.
//!
//! Deterministic and bit-stable on identical hardware — no atomic
//! adds (the per-tile lock array sequences writes but does not
//! perform reductions).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace, U8,
};
use half::f16;

use crate::quantize::map_status;

// =============================================================================
// Sealed activation trait
// =============================================================================

/// Sealed marker for activation / destination dtypes accepted by the
/// Marlin plan. Phase 48 scope is `f16` only.
pub trait MarlinActivation: Element + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for half::f16 {}
}

impl MarlinActivation for f16 {}

// =============================================================================
// Descriptor
// =============================================================================

/// Marlin GEMM problem descriptor.
///
/// Construct via [`Int4MarlinGemmDescriptor::new`] and tune with the
/// `with_*` setters per the Phase 32 builder convention.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct Int4MarlinGemmDescriptor {
    /// Output row count (= activation batch / sequence dim).
    pub m: i32,
    /// Output column count (= weight output dim). Must be a multiple
    /// of 256 (the kernel's `thread_n * parallel-tile` lower bound).
    pub n: i32,
    /// Reduction depth (= activation feature dim = weight input dim).
    /// Must be a multiple of 128 (the kernel's `K_BLOCK` lower bound).
    pub k: i32,
    /// Quantization group size. Either `128` (per-group with
    /// `[K/128, N]` scales) or `-1` (per-channel with `[1, N]` scales).
    pub group_size: i32,
    /// Parallel-tile upper bound. Marlin processes the M-dimension in
    /// outer chunks of `tot_m_blocks > 4` rows by launching up to
    /// `max_par` kernels in parallel. Upstream default is 16; lower
    /// values reduce parallel-tile concurrency, higher values are
    /// wasted work past `M/64`.
    pub max_par: i32,
}

impl Int4MarlinGemmDescriptor {
    /// PyTorch-aligned defaults: group_size = 128 (the dominant
    /// production-LLM setting), max_par = 16 (matches upstream).
    pub fn new(m: i32, n: i32, k: i32) -> Self {
        Self {
            m,
            n,
            k,
            group_size: 128,
            max_par: 16,
        }
    }

    /// Override the quantization group size. Must be `128` (default)
    /// or `-1` (per-channel).
    #[must_use]
    pub fn with_group_size(mut self, g: i32) -> Self {
        self.group_size = g;
        self
    }

    /// Override the parallel-tile bound. Typical values: 8 (smaller
    /// SMs) or 16 (default, matches upstream).
    #[must_use]
    pub fn with_max_par(mut self, mp: i32) -> Self {
        self.max_par = mp;
        self
    }
}

// =============================================================================
// Args
// =============================================================================

/// Per-launch arguments for [`Int4MarlinGemmPlan::run`].
pub struct Int4MarlinGemmArgs<'a, T: MarlinActivation> {
    /// Activation `[M, K]` row-major in `T`.
    pub activation: TensorRef<'a, T, 2>,
    /// Marlin-packed int4 weight buffer. Shape `[K/16, N*16/8]` of
    /// `i32` storage; expressed here as a flat `i32` count `(K/16)
    /// * (N*16/8) = K * N / 8`. The packer (host-side, see
    /// [`crate::gemm::gptq_to_marlin`]) is responsible for placing
    /// the pre-shuffled int4 nibbles in the layout the tensor-core
    /// fragments expect.
    pub weight_packed: TensorRef<'a, i32, 1>,
    /// Per-group scales `[K/group_size, N]` of `T`, or `[1, N]` for
    /// `group_size == -1`. Pre-permuted by the packer along the N
    /// axis to match the tensor-core fragment layout.
    pub scales: TensorRef<'a, T, 2>,
    /// Per-tile lock workspace. Caller-allocated, **zero-initialised**
    /// `i32` buffer with `>= (N / 128) * max_par` entries.
    pub workspace: TensorMut<'a, i32, 1>,
    /// Output `[M, N]` row-major in `T`.
    pub output: TensorMut<'a, T, 2>,
}

// =============================================================================
// Plan
// =============================================================================

/// Marlin int4 W4A16 GEMM. fp16-only.
///
/// Use for the LLM decode-batch hot path (M ∈ [1, 64]) on
/// symmetric-int4-quantized weights. For asymmetric int4 (HF
/// `*-AWQ`) prefer [`super::int4_awq::Int4AwqGemmPlan`].
///
/// GPTQ-quantized weights (asymmetric int4) can be repacked into
/// Marlin's symmetric layout at model load time via
/// [`super::gptq_to_marlin::repack`]; once repacked they're directly
/// usable by this plan.
pub struct Int4MarlinGemmPlan<T: MarlinActivation> {
    desc: Int4MarlinGemmDescriptor,
    sku: KernelSku,
    _phantom: PhantomData<T>,
}

impl<T: MarlinActivation> Int4MarlinGemmPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &Int4MarlinGemmDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.m < 0 || desc.n <= 0 || desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: M, N, K must be non-negative (M) / positive (N, K)",
            ));
        }
        if desc.group_size != -1 && desc.group_size != 128 {
            return Err(Error::Unsupported(
                "Int4MarlinGemmPlan: group_size must be 128 or -1 (per-channel)",
            ));
        }
        if desc.k % 128 != 0 {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: K must be divisible by 128 (kernel K-block bound)",
            ));
        }
        if desc.n % 256 != 0 {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: N must be divisible by 256 (kernel N-tile bound)",
            ));
        }
        if desc.max_par <= 0 {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: max_par must be positive",
            ));
        }
        if !matches!(T::KIND, ElementKind::F16) {
            return Err(Error::Unsupported(
                "Int4MarlinGemmPlan: activation dtype must be f16 (upstream is fp16-only)",
            ));
        }
        // Marlin targets sm_80 / sm_86 / sm_89. We do not reject
        // sm_80/89 here at the Rust layer — the kernel build is
        // gated on the `marlin` cargo feature which implies `sm80`.
        // sm_90 (Hopper) is a separate concern: if the `marlin`
        // feature is enabled in a sm_90a-only build, the kernel
        // won't compile (the inline PTX targets Ampere-class
        // instructions). The Rust plan returns Unsupported in that
        // path; see the cfg-gated dispatch fallback below.
        Ok(Self {
            desc: *desc,
            sku: build_sku(T::KIND),
            _phantom: PhantomData,
        })
    }

    /// Validate args against the plan.
    pub fn can_implement(&self, args: &Int4MarlinGemmArgs<'_, T>) -> Result<()> {
        let m = self.desc.m;
        let n = self.desc.n;
        let k = self.desc.k;
        let g = self.desc.group_size;

        // Activation [M, K] contig along K.
        if args.activation.shape != [m, k] {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: activation shape != [M, K]",
            ));
        }
        if args.activation.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: activation must be contig along K",
            ));
        }

        // Packed weight: K * N / 8 i32 elements.
        let expected_packed_i32 = (k as i64) * (n as i64) / 8;
        if (args.weight_packed.shape[0] as i64) != expected_packed_i32 {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: weight_packed length != K * N / 8 (i32 count)",
            ));
        }

        // Scales [K/g, N] for g > 0 or [1, N] for g == -1.
        let scale_rows = if g == -1 { 1 } else { k / g };
        if args.scales.shape != [scale_rows, n] {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: scales shape != [K/group_size, N] (or [1, N] for per-channel)",
            ));
        }

        // Workspace: int32 with >= (N/128) * max_par entries.
        let need = (n / 128) as i64 * self.desc.max_par as i64;
        if (args.workspace.shape[0] as i64) < need {
            return Err(Error::WorkspaceTooSmall {
                needed: (need as usize) * core::mem::size_of::<i32>(),
                got: (args.workspace.shape[0] as usize) * core::mem::size_of::<i32>(),
            });
        }

        // Output [M, N] contig along N.
        if args.output.shape != [m, n] {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: output shape != [M, N]",
            ));
        }
        if args.output.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "Int4MarlinGemmPlan: output must be contig along N",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — `(N/128) * max_par * 4` (i32 lock array).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        let n_tiles = (self.desc.n / 128).max(0) as usize;
        n_tiles * (self.desc.max_par.max(0) as usize) * core::mem::size_of::<i32>()
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
        _workspace: Workspace<'_>,
        args: Int4MarlinGemmArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.m == 0 {
            return Ok(());
        }

        let a_ptr = args.activation.data.as_raw().0 as *const c_void;
        let b_ptr = args.weight_packed.data.as_raw().0 as *const c_void;
        let s_ptr = args.scales.data.as_raw().0 as *const c_void;
        let ws_ptr = args.workspace.data.as_raw().0 as *mut c_void;
        let c_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = unsafe {
            dispatch_marlin::<T>(
                self.desc.m,
                self.desc.n,
                self.desc.k,
                a_ptr,
                b_ptr,
                c_ptr,
                s_ptr,
                ws_ptr,
                self.desc.group_size,
                self.desc.max_par,
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
        arch: ArchSku::Sm89, // sm_80 / 86 / 89; surface the highest supported on RTX 4070
        backend: BackendKind::Bespoke,
        precision_guarantee: PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        },
    }
}

#[cfg(feature = "marlin")]
#[inline]
unsafe fn dispatch_marlin<T: MarlinActivation>(
    m: i32,
    n: i32,
    k: i32,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    scales: *const c_void,
    workspace: *mut c_void,
    group_size: i32,
    max_par: i32,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_int4_marlin_gemm_f16_run(
                m, n, k, a, b, c, scales, workspace, group_size, max_par, stream,
            )
        },
        _ => 3,
    }
}

#[cfg(not(feature = "marlin"))]
#[inline]
unsafe fn dispatch_marlin<T: MarlinActivation>(
    _: i32, _: i32, _: i32,
    _: *const c_void, _: *const c_void, _: *mut c_void, _: *const c_void,
    _: *mut c_void, _: i32, _: i32, _: *mut c_void,
) -> i32 {
    // marlin cargo feature is off; FFI symbol absent.
    3
}
