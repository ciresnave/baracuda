//! NF4 (NormalFloat 4-bit) dequant + GEMV plans — Phase 53.
//!
//! Vendored kernels: `crates/baracuda-kernels-sys/vendor/bitsandbytes/`
//! (MIT, Dettmers et al. arXiv:2305.14314). Gated behind the
//! `bnb_nf4` cargo feature on both `baracuda-kernels-sys` and
//! `baracuda-kernels`.
//!
//! ## What is NF4?
//!
//! NF4 is the 4-bit format used by **QLoRA** for inference + fine-tuning
//! of large LLMs. It is the dominant 4-bit format for QLoRA-trained
//! Llama / Mistral / Qwen prebuilts on the Hugging Face Hub.
//!
//! Unlike GGUF Q4_0 (symmetric int4 × scale, llama.cpp) or AWQ int4
//! (asymmetric int4 + zero-points, mit-han-lab), NF4 uses a **16-entry
//! non-uniform quantile codebook** derived from the inverse CDF of
//! `N(0, 1)`. Dequant is a 16-entry lookup, not arithmetic. This
//! produces better accuracy than symmetric int4 for normally-distributed
//! weights (which neural-network weights approximately are after
//! normalization).
//!
//! ## Plan families
//!
//! - [`Nf4DequantizePlan<T>`] — bulk unpack `[N/2, K]` u8 → `[N, K]` T.
//!   Used for ahead-of-time dequant or as a debug/correctness reference.
//! - [`Nf4MmvqPlan<T>`] — fused dequant + GEMV. The decode-step matmul:
//!   `out[m, n] = Σ_k codebook[W_q[n, k]] · absmax[n, k/bs] · y[m, k]`.
//!   Supports `M ∈ {1, 2, 4, 8}`; M=1 is single-vector decode,
//!   M ∈ {2, 4, 8} reuses weight gmem reads across the M activation rows
//!   (Phase 33 pattern, applied here to NF4 instead of GGUF Q-formats).
//!
//! ## Scope
//!
//! - **Activation/output dtypes**: `f16`, `bf16` (PyTorch convention —
//!   output dtype matches activation dtype; accumulator stays f32
//!   internally).
//! - **Weight format**: `[N/2, K]` `u8` pair-packed nibbles per the
//!   bitsandbytes upstream convention.
//! - **Block size**: caller-supplied per the descriptor; typically 64.
//!   Must divide `K` evenly.
//! - **N parity**: must be even (pair-packed nibbles).
//! - **No backwards** — inference-only by convention. QLoRA training
//!   uses LoRA adapters; NF4 weights themselves are frozen.
//!
//! ## Numerical contract
//!
//! NF4 is lossy. Relative error vs an fp16 reference matmul is on the
//! order of `~1e-2` (NF4 quantization error class). Specifically:
//!
//! - Weight quantization noise dominates; per the NF4 paper the
//!   per-block normalization keeps the noise floor near the
//!   information-theoretic lower bound for 4-bit codes.
//! - The activation side stays in `f16` / `bf16` throughout. The
//!   accumulator is f32.
//! - The kernel is deterministic and bit-stable on identical hardware
//!   (single-warp reduction per output row, no atomics).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace, U8,
};
use half::{bf16, f16};

use crate::quantize::map_status;

// =============================================================================
// Sealed activation trait
// =============================================================================

/// Sealed marker for activation / destination dtypes accepted by the
/// NF4 plans. The Phase 53 scope is `f16` + `bf16`; an `f32` arm is
/// exposed for the dequant smoke-test path only.
pub trait Nf4Activation: Element + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

impl Nf4Activation for f32 {}
impl Nf4Activation for f16 {}
impl Nf4Activation for bf16 {}

// =============================================================================
// Descriptors
// =============================================================================

/// Descriptor for an NF4 op (dequant or MMVQ). The same descriptor
/// shape covers both plans; only the act dtype generic distinguishes
/// the variants on the Rust side.
#[derive(Copy, Clone, Debug)]
pub struct Nf4Descriptor {
    /// Output row count (= weight matrix output dim). Must be even
    /// (pair-packed nibbles).
    pub n: i32,
    /// Inner / contraction dim (= weight column count = activation
    /// vector length). Must be a multiple of `block_size`.
    pub k: i32,
    /// Per-block element count for the absmax scales. Typically 64.
    /// `k` must be a multiple of this.
    pub block_size: i32,
}

impl Default for Nf4Descriptor {
    fn default() -> Self {
        Self {
            n: 0,
            k: 0,
            block_size: 64,
        }
    }
}

/// Descriptor for the NF4 multi-M MMVQ plan. Same as [`Nf4Descriptor`]
/// plus the compile-time `M` selector.
#[derive(Copy, Clone, Debug)]
pub struct Nf4MmvqMultiMDescriptor {
    /// Inner NF4 op shape.
    pub base: Nf4Descriptor,
    /// Number of activation rows. Must be one of `{1, 2, 4, 8}`. M
    /// values outside that set are rejected at `select` time
    /// (caller-side tiling is the expected handling — Phase 33 pattern).
    pub m: i32,
}

impl Default for Nf4MmvqMultiMDescriptor {
    fn default() -> Self {
        Self {
            base: Nf4Descriptor::default(),
            m: 1,
        }
    }
}

// =============================================================================
// Args bundles
// =============================================================================

/// Args bundle for [`Nf4DequantizePlan`]. Output dtype = `T`.
pub struct Nf4DequantizeArgs<'a, T: Nf4Activation> {
    /// Packed NF4 weight bytes, length `(N/2) * K` (pair-packed nibbles).
    pub weight: TensorRef<'a, U8, 1>,
    /// Per-block absmax scales, length `N * (K / block_size)`.
    pub absmax: TensorRef<'a, f32, 1>,
    /// Output buffer, shape `[N, K]` in `T`.
    pub output: TensorMut<'a, T, 2>,
}

/// Args bundle for [`Nf4MmvqPlan`] (M=1 single-vector decode).
pub struct Nf4MmvqArgs<'a, T: Nf4Activation> {
    /// Packed NF4 weight bytes, length `(N/2) * K`.
    pub weight: TensorRef<'a, U8, 1>,
    /// Per-block absmax, length `N * (K / block_size)`.
    pub absmax: TensorRef<'a, f32, 1>,
    /// Activation vector `[K]` in `T`.
    pub activation: TensorRef<'a, T, 1>,
    /// Output vector `[N]` in `T`.
    pub output: TensorMut<'a, T, 1>,
}

/// Args bundle for [`Nf4MmvqMultiMPlan`] (M ∈ {1, 2, 4, 8}).
pub struct Nf4MmvqMultiMArgs<'a, T: Nf4Activation> {
    /// Packed NF4 weight bytes, length `(N/2) * K`.
    pub weight: TensorRef<'a, U8, 1>,
    /// Per-block absmax, length `N * (K / block_size)`.
    pub absmax: TensorRef<'a, f32, 1>,
    /// Activations, shape `[M, K]` in `T`.
    pub activations: TensorRef<'a, T, 2>,
    /// Output, shape `[M, N]` in `T`.
    pub output: TensorMut<'a, T, 2>,
}

// =============================================================================
// Nf4DequantizePlan
// =============================================================================

/// Bulk-unpack NF4 weights into a dense FP tensor.
///
/// Inference path uses [`Nf4MmvqPlan`] / [`Nf4MmvqMultiMPlan`] instead
/// (fused dequant + GEMV avoids the intermediate `[N, K]` materialization).
/// This plan is primarily a debug / reference / weight-export tool.
pub struct Nf4DequantizePlan<T: Nf4Activation> {
    desc: Nf4Descriptor,
    sku: KernelSku,
    _phantom: PhantomData<T>,
}

impl<T: Nf4Activation> Nf4DequantizePlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &Nf4Descriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(desc, "Nf4DequantizePlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_sku(T::KIND, QuantizeKind::DequantizePerGroup),
            _phantom: PhantomData,
        })
    }

    /// Validate args against the plan.
    pub fn can_implement(&self, args: &Nf4DequantizeArgs<'_, T>) -> Result<()> {
        let n = self.desc.n;
        let k = self.desc.k;
        let bs = self.desc.block_size;
        let expected_packed_bytes = ((n / 2) as i64) * (k as i64);
        if (args.weight.shape[0] as i64) != expected_packed_bytes {
            return Err(Error::InvalidProblem(
                "Nf4DequantizePlan: weight bytes != (N/2) * K",
            ));
        }
        let expected_absmax = (n as i64) * ((k / bs) as i64);
        if (args.absmax.shape[0] as i64) != expected_absmax {
            return Err(Error::InvalidProblem(
                "Nf4DequantizePlan: absmax length != N * (K / block_size)",
            ));
        }
        if args.output.shape != [n, k] {
            return Err(Error::InvalidProblem(
                "Nf4DequantizePlan: output shape != [N, K]",
            ));
        }
        if args.output.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "Nf4DequantizePlan: output must be contiguous along K",
            ));
        }
        Ok(())
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

    /// Workspace bytes — always 0 for the dequant plan.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Nf4DequantizeArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.n == 0 || self.desc.k == 0 {
            return Ok(());
        }

        let w_ptr = args.weight.data.as_raw().0 as *const c_void;
        let amax_ptr = args.absmax.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = unsafe {
            dispatch_dequant::<T>(
                self.desc.n,
                self.desc.k,
                self.desc.block_size,
                w_ptr,
                amax_ptr,
                out_ptr,
                stream_ptr,
            )
        };
        map_status(status)
    }
}

// =============================================================================
// Nf4MmvqPlan (M = 1)
// =============================================================================

/// NF4 GEMV — single-vector decode (M=1) fused dequant + matrix-vector
/// multiply.
///
/// `out[n] = Σ_k codebook[W_q[n, k]] · absmax[n, k/bs] · y[k]`.
///
/// Used for the LLM decode hot path on QLoRA-trained models. For
/// prefill / batched-decode use [`Nf4MmvqMultiMPlan`] with `M > 1`.
pub struct Nf4MmvqPlan<T: Nf4Activation> {
    desc: Nf4Descriptor,
    sku: KernelSku,
    _phantom: PhantomData<T>,
}

impl<T: Nf4Activation> Nf4MmvqPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &Nf4Descriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(desc, "Nf4MmvqPlan")?;
        // M=1 GEMV is only wired for f16/bf16 activations (f32 dequant
        // is a smoke-test path; we don't ship f32 GEMV — caller can
        // upcast bf16 if they need f32 outputs).
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16) {
            return Err(Error::Unsupported(
                "Nf4MmvqPlan: activation dtype must be f16 or bf16",
            ));
        }
        Ok(Self {
            desc: *desc,
            sku: build_sku(T::KIND, QuantizeKind::GgufMmvq),
            _phantom: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &Nf4MmvqArgs<'_, T>) -> Result<()> {
        let n = self.desc.n;
        let k = self.desc.k;
        let bs = self.desc.block_size;
        let expected_packed_bytes = ((n / 2) as i64) * (k as i64);
        if (args.weight.shape[0] as i64) != expected_packed_bytes {
            return Err(Error::InvalidProblem(
                "Nf4MmvqPlan: weight bytes != (N/2) * K",
            ));
        }
        let expected_absmax = (n as i64) * ((k / bs) as i64);
        if (args.absmax.shape[0] as i64) != expected_absmax {
            return Err(Error::InvalidProblem(
                "Nf4MmvqPlan: absmax length != N * (K / block_size)",
            ));
        }
        if args.activation.shape != [k] {
            return Err(Error::InvalidProblem(
                "Nf4MmvqPlan: activation shape != [K]",
            ));
        }
        if args.output.shape != [n] {
            return Err(Error::InvalidProblem("Nf4MmvqPlan: output shape != [N]"));
        }
        if args.activation.stride[0] != 1 {
            return Err(Error::InvalidProblem(
                "Nf4MmvqPlan: activation must be contig",
            ));
        }
        if args.output.stride[0] != 1 {
            return Err(Error::InvalidProblem(
                "Nf4MmvqPlan: output must be contig",
            ));
        }
        Ok(())
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

    /// Workspace bytes — always 0 (no activation staging needed; the
    /// kernel reads activations directly).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Nf4MmvqArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.n == 0 || self.desc.k == 0 {
            return Ok(());
        }

        let w_ptr = args.weight.data.as_raw().0 as *const c_void;
        let amax_ptr = args.absmax.data.as_raw().0 as *const c_void;
        let y_ptr = args.activation.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = unsafe {
            dispatch_gemv_m1::<T>(
                self.desc.n,
                self.desc.k,
                self.desc.block_size,
                w_ptr,
                amax_ptr,
                y_ptr,
                out_ptr,
                stream_ptr,
            )
        };
        map_status(status)
    }
}

// =============================================================================
// Nf4MmvqMultiMPlan (M ∈ {1, 2, 4, 8})
// =============================================================================

/// NF4 multi-M GEMV — reuses one weight matrix across `M` activation
/// rows. Each thread block iterates K once per output row but
/// accumulates `M` partial sums in registers, saving `M`× gmem
/// bandwidth on the weight side.
///
/// `out[m, n] = Σ_k codebook[W_q[n, k]] · absmax[n, k/bs] · y[m, k]`.
///
/// Use for **prefill** (M ∈ [2, 8]) and **batched decode**. M values
/// outside `{1, 2, 4, 8}` are rejected — caller tiles down to these
/// chunk sizes per the Phase 33 pattern.
pub struct Nf4MmvqMultiMPlan<T: Nf4Activation> {
    desc: Nf4MmvqMultiMDescriptor,
    sku: KernelSku,
    _phantom: PhantomData<T>,
}

impl<T: Nf4Activation> Nf4MmvqMultiMPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &Nf4MmvqMultiMDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(&desc.base, "Nf4MmvqMultiMPlan")?;
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16) {
            return Err(Error::Unsupported(
                "Nf4MmvqMultiMPlan: activation dtype must be f16 or bf16",
            ));
        }
        if !matches!(desc.m, 1 | 2 | 4 | 8) {
            return Err(Error::Unsupported(
                "Nf4MmvqMultiMPlan: M must be one of {1, 2, 4, 8}",
            ));
        }
        Ok(Self {
            desc: *desc,
            sku: build_sku(T::KIND, QuantizeKind::GgufMmvq),
            _phantom: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &Nf4MmvqMultiMArgs<'_, T>) -> Result<()> {
        let n = self.desc.base.n;
        let k = self.desc.base.k;
        let bs = self.desc.base.block_size;
        let m = self.desc.m;
        let expected_packed_bytes = ((n / 2) as i64) * (k as i64);
        if (args.weight.shape[0] as i64) != expected_packed_bytes {
            return Err(Error::InvalidProblem(
                "Nf4MmvqMultiMPlan: weight bytes != (N/2) * K",
            ));
        }
        let expected_absmax = (n as i64) * ((k / bs) as i64);
        if (args.absmax.shape[0] as i64) != expected_absmax {
            return Err(Error::InvalidProblem(
                "Nf4MmvqMultiMPlan: absmax length != N * (K / block_size)",
            ));
        }
        if args.activations.shape != [m, k] {
            return Err(Error::InvalidProblem(
                "Nf4MmvqMultiMPlan: activations shape != [M, K]",
            ));
        }
        if args.output.shape != [m, n] {
            return Err(Error::InvalidProblem(
                "Nf4MmvqMultiMPlan: output shape != [M, N]",
            ));
        }
        if args.activations.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "Nf4MmvqMultiMPlan: activations must be contig along K",
            ));
        }
        if args.output.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "Nf4MmvqMultiMPlan: output must be contig along N",
            ));
        }
        Ok(())
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

    /// Workspace bytes — always 0 (no staging buffer needed).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Nf4MmvqMultiMArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.base.n == 0 || self.desc.base.k == 0 || self.desc.m == 0 {
            return Ok(());
        }

        let w_ptr = args.weight.data.as_raw().0 as *const c_void;
        let amax_ptr = args.absmax.data.as_raw().0 as *const c_void;
        let y_ptr = args.activations.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = unsafe {
            dispatch_gemv_multim::<T>(
                self.desc.m,
                self.desc.base.n,
                self.desc.base.k,
                self.desc.base.block_size,
                w_ptr,
                amax_ptr,
                y_ptr,
                out_ptr,
                stream_ptr,
            )
        };
        map_status(status)
    }
}

// =============================================================================
// Internal helpers
// =============================================================================

fn validate_desc(desc: &Nf4Descriptor, plan_name: &'static str) -> Result<()> {
    if desc.n < 0 || desc.k < 0 || desc.block_size <= 0 {
        return Err(Error::InvalidProblem(
            // Static msg per the rest of the kernels surface.
            match plan_name {
                "Nf4DequantizePlan" => "Nf4DequantizePlan: invalid dims",
                "Nf4MmvqPlan" => "Nf4MmvqPlan: invalid dims",
                _ => "Nf4 plan: invalid dims",
            },
        ));
    }
    if (desc.n & 1) != 0 {
        return Err(Error::InvalidProblem(
            "Nf4 plan: N must be even (pair-packed nibbles)",
        ));
    }
    if desc.k % desc.block_size != 0 {
        return Err(Error::InvalidProblem(
            "Nf4 plan: K must be a multiple of block_size",
        ));
    }
    Ok(())
}

fn build_sku(act_kind: ElementKind, op: QuantizeKind) -> KernelSku {
    KernelSku {
        category: OpCategory::Quantization,
        op: op as u16,
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

// =============================================================================
// FFI dispatch wrappers (cfg-gated on `bnb_nf4`)
// =============================================================================

#[cfg(feature = "bnb_nf4")]
#[inline]
unsafe fn dispatch_dequant<T: Nf4Activation>(
    n: i32,
    k: i32,
    block_size: i32,
    w_ptr: *const c_void,
    absmax: *const c_void,
    out: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_nf4_dequantize_f16_run(
                n, k, block_size, w_ptr, absmax, out, stream,
            )
        },
        ElementKind::Bf16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_nf4_dequantize_bf16_run(
                n, k, block_size, w_ptr, absmax, out, stream,
            )
        },
        ElementKind::F32 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_nf4_dequantize_f32_run(
                n, k, block_size, w_ptr, absmax, out, stream,
            )
        },
        _ => 3,
    }
}

#[cfg(not(feature = "bnb_nf4"))]
#[inline]
unsafe fn dispatch_dequant<T: Nf4Activation>(
    _: i32, _: i32, _: i32, _: *const c_void, _: *const c_void, _: *mut c_void, _: *mut c_void,
) -> i32 {
    // bnb_nf4 cargo feature is off; the Rust types compile but launching
    // is a no-op error.
    3
}

#[cfg(feature = "bnb_nf4")]
#[inline]
unsafe fn dispatch_gemv_m1<T: Nf4Activation>(
    n: i32,
    k: i32,
    block_size: i32,
    w_ptr: *const c_void,
    absmax: *const c_void,
    y: *const c_void,
    out: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_nf4_gemv_m1_f16_run(
                n, k, block_size, w_ptr, absmax, y, out, stream,
            )
        },
        ElementKind::Bf16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_nf4_gemv_m1_bf16_run(
                n, k, block_size, w_ptr, absmax, y, out, stream,
            )
        },
        _ => 3,
    }
}

#[cfg(not(feature = "bnb_nf4"))]
#[inline]
unsafe fn dispatch_gemv_m1<T: Nf4Activation>(
    _: i32, _: i32, _: i32, _: *const c_void, _: *const c_void, _: *const c_void,
    _: *mut c_void, _: *mut c_void,
) -> i32 {
    3
}

#[cfg(feature = "bnb_nf4")]
#[inline]
unsafe fn dispatch_gemv_multim<T: Nf4Activation>(
    m: i32,
    n: i32,
    k: i32,
    block_size: i32,
    w_ptr: *const c_void,
    absmax: *const c_void,
    y: *const c_void,
    out: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    use baracuda_kernels_sys as sys;
    match (T::KIND, m) {
        (ElementKind::F16, 1) => unsafe {
            sys::baracuda_kernels_nf4_gemv_m1_f16_run(n, k, block_size, w_ptr, absmax, y, out, stream)
        },
        (ElementKind::F16, 2) => unsafe {
            sys::baracuda_kernels_nf4_gemv_m2_f16_run(n, k, block_size, w_ptr, absmax, y, out, stream)
        },
        (ElementKind::F16, 4) => unsafe {
            sys::baracuda_kernels_nf4_gemv_m4_f16_run(n, k, block_size, w_ptr, absmax, y, out, stream)
        },
        (ElementKind::F16, 8) => unsafe {
            sys::baracuda_kernels_nf4_gemv_m8_f16_run(n, k, block_size, w_ptr, absmax, y, out, stream)
        },
        (ElementKind::Bf16, 1) => unsafe {
            sys::baracuda_kernels_nf4_gemv_m1_bf16_run(n, k, block_size, w_ptr, absmax, y, out, stream)
        },
        (ElementKind::Bf16, 2) => unsafe {
            sys::baracuda_kernels_nf4_gemv_m2_bf16_run(n, k, block_size, w_ptr, absmax, y, out, stream)
        },
        (ElementKind::Bf16, 4) => unsafe {
            sys::baracuda_kernels_nf4_gemv_m4_bf16_run(n, k, block_size, w_ptr, absmax, y, out, stream)
        },
        (ElementKind::Bf16, 8) => unsafe {
            sys::baracuda_kernels_nf4_gemv_m8_bf16_run(n, k, block_size, w_ptr, absmax, y, out, stream)
        },
        _ => 3,
    }
}

#[cfg(not(feature = "bnb_nf4"))]
#[inline]
unsafe fn dispatch_gemv_multim<T: Nf4Activation>(
    _: i32, _: i32, _: i32, _: i32, _: *const c_void, _: *const c_void, _: *const c_void,
    _: *mut c_void, _: *mut c_void,
) -> i32 {
    3
}

// =============================================================================
// Host-side NF4 codebook + quantize helper (test / weight-prep use)
// =============================================================================

/// The 16-entry NF4 codebook (Section 3.1 of Dettmers et al. 2023).
///
/// Reproduced bit-identical to the upstream bitsandbytes `csrc/kernels.cu`
/// device-side switch table. The values come from the inverse CDF of
/// `N(0, 1)` evaluated at the 16-quantile midpoints with the zero
/// quantile pinned to exactly 0.0.
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Quantize a single fp32 value to its nearest NF4 codebook index after
/// dividing by the per-block absmax. Used by the test layer to produce
/// reference weights and by callers that want a host-side reference.
pub fn nf4_quantize_value(x: f32, absmax: f32) -> u8 {
    if absmax == 0.0 {
        // All-zero block → emit code 7 (= 0.0 in the codebook).
        return 7;
    }
    let xn = x / absmax;
    let mut best_i = 0u8;
    let mut best_d = (xn - NF4_CODEBOOK[0]).abs();
    for i in 1..16u8 {
        let d = (xn - NF4_CODEBOOK[i as usize]).abs();
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    best_i
}

/// Pack-quantize a `[N, K]` f32 weight matrix into the bitsandbytes NF4
/// pair-packed `(N/2) * K` byte buffer + the `N * (K / block_size)`
/// per-block absmax buffer. Returns `(packed_bytes, absmax)`.
///
/// `N` must be even; `K` must be a multiple of `block_size`.
pub fn nf4_pack_weight(
    weight_fp: &[f32],
    n: usize,
    k: usize,
    block_size: usize,
) -> (alloc::vec::Vec<u8>, alloc::vec::Vec<f32>) {
    use alloc::vec;
    use alloc::vec::Vec;
    assert!(n % 2 == 0, "N must be even");
    assert!(k % block_size == 0, "K must be a multiple of block_size");
    assert_eq!(weight_fp.len(), n * k);

    let blocks_per_row = k / block_size;
    let num_blocks = n * blocks_per_row;
    let mut absmax: Vec<f32> = vec![0.0; num_blocks];
    let mut packed: Vec<u8> = vec![0; (n / 2) * k];

    // Compute per-(row, block) absmax.
    for row in 0..n {
        for b in 0..blocks_per_row {
            let mut a = 0.0f32;
            for j in 0..block_size {
                let v = weight_fp[row * k + b * block_size + j].abs();
                if v > a {
                    a = v;
                }
            }
            absmax[row * blocks_per_row + b] = a;
        }
    }

    // Quantize + pair-pack.
    for row in 0..n {
        for b in 0..blocks_per_row {
            let a = absmax[row * blocks_per_row + b];
            for j in 0..block_size {
                let kpos = b * block_size + j;
                let code = nf4_quantize_value(weight_fp[row * k + kpos], a);
                // Pack: row `n` lives at byte (n/2)*K + k of the buffer.
                // (row & 1) == 0 → low nibble; == 1 → high nibble.
                let byte_off = (row / 2) * k + kpos;
                let b_ref = &mut packed[byte_off];
                if (row & 1) == 0 {
                    *b_ref = (*b_ref & 0xF0) | (code & 0x0F);
                } else {
                    *b_ref = (*b_ref & 0x0F) | ((code & 0x0F) << 4);
                }
            }
        }
    }

    (packed, absmax)
}

extern crate alloc;
