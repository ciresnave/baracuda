//! int4 GEMM plan.
//!
//! Phase 2 int4 surface — full 36-SKU matrix on `sm_89`:
//!
//!   `{S4, U4} × {RCR, RRR} × {Identity, Bias, BiasRelu, BiasGelu,
//!                              BiasSilu} × {f32 bias, i32 bias}`
//!
//! (The Identity epilogue ignores the bias-element generic, so the
//! full SKU count is `2 × 2 × (1 + 4 × 2) = 36`.) The plan-layer type
//! is parameterized on `T: IntElement` and `BT: BiasElement`; the
//! dispatcher routes on `(T::KIND, layout, epilogue, BT::KIND)` to the
//! matching bespoke kernel in `baracuda-kernels-sys`.
//!
//! All int4 SKUs are bespoke kernels in
//! [`baracuda-kernels-sys`](baracuda_kernels_sys) — CUTLASS 4.2.0 does
//! not ship int4 instantiations on the layouts / epilogues this crate
//! targets (and the int8 RRR rationale carries over: packed-storage
//! int kernels exceed CUTLASS template-vendoring friction past 8-bit).
//!
//! Plan-layer conventions:
//!
//! - Descriptor M / N / K are in **element** counts.
//! - `MatrixRef<S4>::ld` / `MatrixMut<S4>::ld` are in **storage-slot
//!   (= byte) counts**, not element counts. Each storage slot is a
//!   packed-pair byte (low nibble = even index, high nibble = odd
//!   index along the K axis for A/B operands and the N axis for the
//!   D output).
//! - `MatrixRef<S4>::cols` is in **element** counts for shape checks
//!   (= K for A and B, = N for D); the leading dimension `ld` is in
//!   bytes.
//! - `K` must be even (packing is byte-aligned).
//! - `N` must be even (the D output store is byte-granular at packed
//!   pairs along N).
//!
//! Bias element generic mirrors the int8 family: `<T, f32>` for
//! float-scaled bias, `<T, i32>` for TensorRT-style integer bias.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BiasElement, BiasElementKind, ElementKind, EpilogueKind, IntElement, LayoutSku,
    MatrixMut, MatrixRef, PlanPreference, PrecisionGuarantee, S4, U4, VectorRef, Workspace,
};

// Reuse the SKU descriptor from baracuda-cutlass — it carries the same
// (arch, layout, epilogue, element, bias_element) tuple the int-GEMM
// and FP8 dispatchers use, so telemetry / autotuner caches can key off
// one shape across all GEMM backends.
pub use baracuda_cutlass::GemmSku;

/// Problem shape and configuration for [`Int4GemmPlan::select`].
#[derive(Copy, Clone, Debug)]
pub struct Int4GemmDescriptor {
    /// Output row count, in **elements**.
    pub m: i32,
    /// Output column count, in **elements** (must be even).
    pub n: i32,
    /// Reduction depth, in **elements** (must be even).
    pub k: i32,
    /// Layout SKU. Trailblazer SKU is [`LayoutSku::Rcr`]; the RRR
    /// variant joins in subsequent fanout commits.
    pub layout: LayoutSku,
    /// Epilogue kind. Trailblazer SKU is [`EpilogueKind::Identity`];
    /// the `Bias*` family joins in subsequent fanout commits.
    pub epilogue: EpilogueKind,
}

/// Per-launch arguments for an [`Int4GemmPlan::run`] call.
///
/// `c` is optional: when `None`, `β` is forced to `0` at the safe
/// layer and the kernel computes `D = sat_cast_int4(α · A · B)`. The
/// kernel uses an S32 accumulator and saturating-casts back to s4
/// (clamp to `[-8, +7]`) or u4 (clamp to `[0, 15]`) on store.
///
/// `bias` is optional and required-vs-disallowed by the epilogue:
/// [`EpilogueKind::Identity`] must pass `None`; the `Bias*` variants
/// (later fanout) must pass `Some`. The bias element type follows the
/// `BT` plan generic — `f32` for the conventional float-scaled bias,
/// `i32` for the TensorRT-style integer bias convention.
///
/// Pointer / layout contracts (mirrors the FFI; see the
/// `baracuda-kernels-sys` doc comment on
/// `baracuda_kernels_gemm_s4_rcr_sm89_run` for the byte-level details):
///
/// - `a.data` storage holds packed-pair int4 bytes; `a.rows = M`,
///   `a.cols = K` (element counts), `a.ld` is the row stride in
///   **bytes** (≥ `K / 2`).
/// - `b.data` storage holds packed-pair int4 bytes; `b.rows = K`,
///   `b.cols = N`, `b.ld` is the column stride in **bytes** (≥ `K / 2`
///   for `LayoutSku::Rcr`; col-major).
/// - `c.data` / `d.data` storage holds packed-pair int4 bytes;
///   `c.rows = d.rows = M`, `c.cols = d.cols = N`, `c.ld = d.ld` are
///   row strides in **bytes** (≥ `N / 2`).
#[derive(Debug)]
pub struct Int4GemmArgs<'a, T: IntElement, BT: BiasElement = f32> {
    /// Left input. Row-major `[M, K]`. Packed-pair int4 storage; `ld`
    /// in bytes.
    pub a: MatrixRef<'a, T>,
    /// Right input. RCR: col-major `[K, N]` with column stride ≥ K/2
    /// bytes. RRR (later fanout): row-major `[K, N]` with row stride
    /// ≥ N/2 bytes.
    pub b: MatrixRef<'a, T>,
    /// Optional accumulation source. Row-major `[M, N]` packed-pair
    /// int4 storage. `None` skips the `β · C` term internally.
    pub c: Option<MatrixRef<'a, T>>,
    /// Output. Row-major `[M, N]` packed-pair int4 storage.
    pub d: MatrixMut<'a, T>,
    /// Optional bias vector. Required when the descriptor's epilogue
    /// is any `Bias*` variant (later fanout); must be `None` for
    /// [`EpilogueKind::Identity`]. Length-`N`, contiguous device
    /// memory; broadcast across rows of `D`.
    pub bias: Option<VectorRef<'a, BT>>,
    /// Multiplier on the matrix-multiply accumulator. F32 — the
    /// kernel accumulates in S32 and the epilogue scales to F32
    /// before quantizing back to int4.
    pub alpha: f32,
    /// Multiplier on `c`. Forced to `0` internally when `c` is `None`.
    pub beta: f32,
}

/// int4 GEMM plan.
///
/// Parameterized on `T: IntElement` ([`S4`] / [`U4`]; trailblazer SKU
/// is [`S4`] only) and `BT: BiasElement` (`f32` default, or `i32`,
/// matching the int8 family's bias-element convention).
pub struct Int4GemmPlan<T: IntElement, BT: BiasElement = f32> {
    desc: Int4GemmDescriptor,
    sku: GemmSku,
    _element: PhantomData<T>,
    _bias_element: PhantomData<BT>,
}

impl<T: IntElement, BT: BiasElement> Int4GemmPlan<T, BT> {
    /// Pick an int4 GEMM kernel for `desc`.
    ///
    /// Returns [`Error::Unsupported`] for any `(T, layout, epilogue)`
    /// triple outside the shipped trailblazer SKU
    /// (`S4 × RCR × Identity`). The rest of the int4 matrix joins in
    /// subsequent fanout commits.
    pub fn select(
        _stream: &Stream,
        desc: &Int4GemmDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "int4 GEMM problem must have positive M, N, K",
            ));
        }
        if (desc.k & 1) != 0 {
            return Err(Error::InvalidProblem(
                "int4 GEMM requires K to be even (packed-pair storage along K)",
            ));
        }
        if (desc.n & 1) != 0 {
            return Err(Error::InvalidProblem(
                "int4 GEMM requires N to be even (packed-pair storage along N for D output)",
            ));
        }
        if !matches!(T::KIND, ElementKind::S4 | ElementKind::U4) {
            return Err(Error::Unsupported(
                "baracuda-kernels: int4 GEMM: only S4 / U4 elements are accepted",
            ));
        }
        // Shipped today: full {S4, U4} × {RCR, RRR} × {Identity, Bias,
        // BiasRelu, BiasGelu, BiasSilu} × {f32, i32} bias matrix. The
        // `select` accepts every combination; the dispatcher in `run`
        // routes to the matching `baracuda-kernels-sys` entry point.
        // (T::KIND guard above already rejected non-int4 elements.)
        let _ = desc.layout;
        let _ = desc.epilogue;

        let sku = GemmSku {
            arch: ArchSku::Sm89,
            layout: desc.layout,
            epilogue: desc.epilogue,
            element: T::KIND,
            bias_element: if desc.epilogue.requires_bias() {
                Some(BT::KIND)
            } else {
                None
            },
        };
        Ok(Self {
            desc: *desc,
            sku,
            _element: PhantomData,
            _bias_element: PhantomData,
        })
    }

    /// Validate that this plan can launch with `args`.
    pub fn can_implement(&self, args: &Int4GemmArgs<'_, T, BT>) -> Result<()> {
        if self.desc.m <= 0 || self.desc.n <= 0 || self.desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "int4 GEMM problem must have positive M, N, K",
            ));
        }
        if args.a.rows != self.desc.m || args.a.cols != self.desc.k {
            return Err(Error::InvalidProblem(
                "A shape mismatch with descriptor (M, K in elements)",
            ));
        }
        if args.b.rows != self.desc.k || args.b.cols != self.desc.n {
            return Err(Error::InvalidProblem(
                "B shape mismatch with descriptor (K, N in elements)",
            ));
        }
        if args.d.rows != self.desc.m || args.d.cols != self.desc.n {
            return Err(Error::InvalidProblem(
                "D shape mismatch with descriptor (M, N in elements)",
            ));
        }
        // Leading dimensions are in BYTES — minimum is element-count / 2.
        let k_bytes_min = (self.desc.k / 2) as i64;
        let n_bytes_min = (self.desc.n / 2) as i64;
        match self.sku.layout {
            LayoutSku::Rcr => {
                if args.a.ld < k_bytes_min {
                    return Err(Error::InvalidProblem(
                        "A leading dimension (bytes) must be >= K/2 for row-major int4 A",
                    ));
                }
                if args.b.ld < k_bytes_min {
                    return Err(Error::InvalidProblem(
                        "B leading dimension (bytes) must be >= K/2 for col-major int4 B (RCR)",
                    ));
                }
            }
            LayoutSku::Rrr => {
                // Reserved for fanout; reaches here only via a
                // direct construction bypassing `select`.
                if args.a.ld < k_bytes_min {
                    return Err(Error::InvalidProblem(
                        "A leading dimension (bytes) must be >= K/2 for row-major int4 A",
                    ));
                }
                if args.b.ld < n_bytes_min {
                    return Err(Error::InvalidProblem(
                        "B leading dimension (bytes) must be >= N/2 for row-major int4 B (RRR)",
                    ));
                }
            }
        }
        if args.d.ld < n_bytes_min {
            return Err(Error::InvalidProblem(
                "D leading dimension (bytes) must be >= N/2 for row-major int4 D",
            ));
        }
        if let Some(c) = &args.c {
            if c.rows != self.desc.m || c.cols != self.desc.n {
                return Err(Error::InvalidProblem(
                    "C shape mismatch with descriptor (M, N in elements)",
                ));
            }
            if c.ld < n_bytes_min {
                return Err(Error::InvalidProblem(
                    "C leading dimension (bytes) must be >= N/2 for row-major int4 C",
                ));
            }
        }
        // Bias presence must match the epilogue family. Identity
        // (trailblazer) takes no bias.
        let needs_bias = self.sku.epilogue.requires_bias();
        match (needs_bias, &args.bias) {
            (true, None) => {
                return Err(Error::InvalidProblem(
                    "Bias* epilogue requires a bias vector",
                ));
            }
            (false, Some(_)) => {
                return Err(Error::InvalidProblem(
                    "Identity epilogue must not be supplied a bias vector",
                ));
            }
            _ => {}
        }
        if let Some(b) = &args.bias
            && b.len != self.desc.n
        {
            return Err(Error::InvalidProblem(
                "bias length must equal N",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes. Zero across the shipped int4 SKUs —
    /// the kernel does all its work in smem + registers.
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> GemmSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel. int4 → int32
    /// accumulator → f32 epilogue → saturating-cast back to int4
    /// (round-half-to-even, clamp). Bit-stable on the same hardware
    /// (integer MMA has no warp-reduction nondeterminism).
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee()
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: Int4GemmArgs<'_, T, BT>,
    ) -> Result<()> {
        let _ = workspace; // int4 kernels take no scratch

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let (c_ptr, ldc) = match &args.c {
            Some(c) => (c.data.as_raw().0 as *const c_void, c.ld),
            None => (core::ptr::null(), 0i64),
        };
        let bias_ptr: *const c_void = match &args.bias {
            Some(b) => b.data.as_raw().0 as *const c_void,
            None => core::ptr::null(),
        };

        let needs_bias = self.sku.epilogue.requires_bias();
        if needs_bias && bias_ptr.is_null() {
            return Err(Error::InvalidProblem(
                "Bias* epilogue requires a bias vector",
            ));
        }
        if !needs_bias && !bias_ptr.is_null() {
            return Err(Error::InvalidProblem(
                "Identity epilogue must not be supplied a bias vector",
            ));
        }

        let stream_ptr = stream.as_raw() as *mut c_void;
        let m = self.desc.m;
        let n = self.desc.n;
        let k = self.desc.k;
        let lda = args.a.ld;
        let ldb = args.b.ld;
        let ldd = args.d.ld;
        let alpha = args.alpha;
        let beta = if args.c.is_some() { args.beta } else { 0.0 };

        #[cfg(not(feature = "sm89"))]
        {
            let _ = (a_ptr, b_ptr, c_ptr, d_ptr, ldc, lda, ldb, ldd,
                     m, n, k, alpha, beta, stream_ptr);
            return Err(Error::Unsupported(
                "baracuda-kernels: int4 GEMM requires the `sm89` feature \
                 to be enabled in baracuda-kernels-sys",
            ));
        }

        // Identity launchers don't take a bias pointer in their
        // signature; the bias family does and additionally routes on
        // `BT::KIND` ∈ {F32, I32}. The 4-tuple key reflects this:
        // `(T::KIND, layout, epilogue)` for Identity (BT is meaningless)
        // and the same triple delegating to a `BT::KIND` inner match for
        // each bias variant.
        #[cfg(feature = "sm89")]
        let status = match (T::KIND, self.sku.layout, self.sku.epilogue) {
            // ---- Identity ----
            (ElementKind::S4, LayoutSku::Rcr, EpilogueKind::Identity) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    alpha, beta,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            (ElementKind::U4, LayoutSku::Rcr, EpilogueKind::Identity) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    alpha, beta,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            (ElementKind::S4, LayoutSku::Rrr, EpilogueKind::Identity) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    alpha, beta,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            (ElementKind::U4, LayoutSku::Rrr, EpilogueKind::Identity) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    alpha, beta,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },

            // ---- S4 × RCR × Bias family ----
            (ElementKind::S4, LayoutSku::Rcr, EpilogueKind::Bias) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_bias_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_bias_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::S4, LayoutSku::Rcr, EpilogueKind::BiasRelu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_bias_relu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_bias_relu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::S4, LayoutSku::Rcr, EpilogueKind::BiasGelu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_bias_gelu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_bias_gelu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::S4, LayoutSku::Rcr, EpilogueKind::BiasSilu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_bias_silu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rcr_sm89_bias_silu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },

            // ---- U4 × RCR × Bias family ----
            (ElementKind::U4, LayoutSku::Rcr, EpilogueKind::Bias) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_bias_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_bias_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::U4, LayoutSku::Rcr, EpilogueKind::BiasRelu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_bias_relu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_bias_relu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::U4, LayoutSku::Rcr, EpilogueKind::BiasGelu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_bias_gelu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_bias_gelu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::U4, LayoutSku::Rcr, EpilogueKind::BiasSilu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_bias_silu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rcr_sm89_bias_silu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },

            // ---- S4 × RRR × Bias family ----
            (ElementKind::S4, LayoutSku::Rrr, EpilogueKind::Bias) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_bias_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_bias_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::S4, LayoutSku::Rrr, EpilogueKind::BiasRelu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_bias_relu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_bias_relu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::S4, LayoutSku::Rrr, EpilogueKind::BiasGelu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_bias_gelu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_bias_gelu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::S4, LayoutSku::Rrr, EpilogueKind::BiasSilu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_bias_silu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_s4_rrr_sm89_bias_silu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },

            // ---- U4 × RRR × Bias family ----
            (ElementKind::U4, LayoutSku::Rrr, EpilogueKind::Bias) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_bias_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_bias_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::U4, LayoutSku::Rrr, EpilogueKind::BiasRelu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_bias_relu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_bias_relu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::U4, LayoutSku::Rrr, EpilogueKind::BiasGelu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_bias_gelu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_bias_gelu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },
            (ElementKind::U4, LayoutSku::Rrr, EpilogueKind::BiasSilu) => match BT::KIND {
                BiasElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_bias_silu_f32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
                BiasElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_gemm_u4_rrr_sm89_bias_silu_i32_run(
                        m, n, k,
                        a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                        bias_ptr, alpha, beta,
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                },
            },

            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels: int4 GEMM dispatcher reached an \
                     unimplemented (element, layout, epilogue) triple \
                     (T must be S4 / U4)",
                ));
            }
        };

        #[cfg(feature = "sm89")]
        { map_status(status) }
        #[cfg(not(feature = "sm89"))]
        #[allow(unreachable_code)]
        { unreachable!("returned earlier under #[cfg(not(feature = \"sm89\"))]") }
    }
}

// Type-level guard that the int4 element re-exports are in scope.
#[allow(dead_code)]
fn _hold_int4_elements_in_scope() {
    let _ = S4(0);
    let _ = U4(0);
}

#[cfg(feature = "sm89")]
fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration \
             (likely K or N odd — int4 packing requires both even)",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
