//! FP8 GEMM plan.
//!
//! Full Phase 2 surface: 20 SKUs spanning
//! `{Fp8E4M3, Fp8E5M2} × {LayoutSku::Rcr, LayoutSku::Rrr} ×
//!  {Identity, Bias, BiasRelu, BiasGelu, BiasSilu}` on `sm_89`.
//!
//! All FP8 SKUs are bespoke kernels in
//! [`baracuda-kernels-sys`](baracuda_kernels_sys) — CUTLASS 4.2.0
//! does not ship FP8 instantiations on the layouts / epilogues this
//! crate targets, so the dispatch is unconditional (no `Backend` enum;
//! that pattern will return if a CUTLASS FP8 path joins the matrix).
//!
//! The bias element is always `f32` for the FP8 family — FP8 has no
//! int32-bias convention, unlike the integer GEMM family. The plan
//! type intentionally has no `BT` generic; bias is fed as a
//! `VectorRef<'a, f32>` regardless of the matrix element type.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, ElementKind, EpilogueKind, Fp8E4M3, Fp8E5M2, FpElement, LayoutSku, MatrixMut,
    MatrixRef, PlanPreference, PrecisionGuarantee, VectorRef, Workspace,
};

// Reuse the SKU descriptor from baracuda-cutlass — it carries the same
// (arch, layout, epilogue, element, bias_element) tuple the int-GEMM
// dispatcher uses, so telemetry / autotuner caches can key off one
// shape across both backends.
pub use baracuda_cutlass::GemmSku;

/// Problem shape and configuration for [`Fp8GemmPlan::select`].
#[derive(Copy, Clone, Debug)]
pub struct Fp8GemmDescriptor {
    /// Output row count.
    pub m: i32,
    /// Output column count.
    pub n: i32,
    /// Reduction depth.
    pub k: i32,
    /// Layout SKU. Today: [`LayoutSku::Rcr`] or [`LayoutSku::Rrr`].
    pub layout: LayoutSku,
    /// Epilogue kind. Today: any of [`EpilogueKind::Identity`],
    /// [`EpilogueKind::Bias`], [`EpilogueKind::BiasRelu`],
    /// [`EpilogueKind::BiasGelu`], [`EpilogueKind::BiasSilu`].
    pub epilogue: EpilogueKind,
}

/// Per-launch arguments for an [`Fp8GemmPlan::run`] call.
///
/// `c` is optional: when `None`, `β` is ignored at the safe layer
/// (treated as `0`) and the kernel computes `D = α · A · B`. The
/// kernel uses an F32 accumulator and saturating-casts to the
/// element-specific max-finite (E4M3 ±448, E5M2 ±57344) on store.
///
/// `bias` is optional and required-vs-disallowed by the epilogue:
/// [`EpilogueKind::Identity`] must pass `None`; the `Bias*` variants
/// must pass `Some`. The bias vector is always `f32` regardless of
/// `T` — FP8 has no int32-bias convention.
#[derive(Debug)]
pub struct Fp8GemmArgs<'a, T: FpElement> {
    /// Left input. Row-major `[M, K]`.
    pub a: MatrixRef<'a, T>,
    /// Right input. RCR: col-major `[K, N]` (column stride ≥ K).
    /// RRR: row-major `[K, N]` (row stride ≥ N).
    pub b: MatrixRef<'a, T>,
    /// Optional accumulation source. Row-major `[M, N]`.
    pub c: Option<MatrixRef<'a, T>>,
    /// Output. Row-major `[M, N]`.
    pub d: MatrixMut<'a, T>,
    /// Optional bias vector. Required when the descriptor's epilogue
    /// is any `Bias*` variant; must be `None` for
    /// [`EpilogueKind::Identity`]. Length-`N`, contiguous (stride 1)
    /// device memory; broadcast across rows of `D`. Always `f32`
    /// regardless of `T`.
    pub bias: Option<VectorRef<'a, f32>>,
    /// Multiplier on the matrix-multiply accumulator. F32 — the kernel
    /// accumulates in F32 throughout.
    pub alpha: f32,
    /// Multiplier on `c`. Forced to `0` internally when `c` is `None`.
    pub beta: f32,
}

/// FP8 GEMM plan.
///
/// Parameterized on `T: FpElement` ([`Fp8E4M3`] or [`Fp8E5M2`]). No
/// bias-element generic — FP8 bias is always `f32`.
pub struct Fp8GemmPlan<T: FpElement> {
    desc: Fp8GemmDescriptor,
    sku: GemmSku,
    _phantom: PhantomData<T>,
}

impl<T: FpElement> Fp8GemmPlan<T> {
    /// Pick an FP8 GEMM kernel for `desc`.
    ///
    /// Returns [`Error::Unsupported`] only for `(T, layout, epilogue)`
    /// triples outside the shipped 20-SKU matrix — i.e. element types
    /// other than E4M3 / E5M2, or epilogues outside the
    /// `{Identity, Bias, BiasRelu, BiasGelu, BiasSilu}` family.
    pub fn select(
        _stream: &Stream,
        desc: &Fp8GemmDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "FP8 GEMM problem must have positive M, N, K",
            ));
        }
        if !matches!(desc.layout, LayoutSku::Rcr | LayoutSku::Rrr) {
            return Err(Error::Unsupported(
                "baracuda-kernels: FP8 GEMM: only RCR / RRR layouts are shipped",
            ));
        }
        if !matches!(T::KIND, ElementKind::Fp8E4M3 | ElementKind::Fp8E5M2) {
            return Err(Error::Unsupported(
                "baracuda-kernels: FP8 GEMM: only Fp8E4M3 / Fp8E5M2 elements are shipped",
            ));
        }
        // EpilogueKind has 5 variants and we cover all of them — no
        // guard needed beyond the enum match.

        let sku = GemmSku {
            arch: ArchSku::Sm89,
            layout: desc.layout,
            epilogue: desc.epilogue,
            element: T::KIND,
            // FP8 has no int32-bias convention; the bias element is
            // implicitly f32 for `Bias*` epilogues and absent for
            // Identity. We surface this with `None` (Identity) or
            // `Some(F32)` (Bias*) for consistency with the int family.
            bias_element: if desc.epilogue.requires_bias() {
                Some(baracuda_kernels_types::BiasElementKind::F32)
            } else {
                None
            },
        };
        Ok(Self {
            desc: *desc,
            sku,
            _phantom: PhantomData,
        })
    }

    /// Validate that this plan can launch with `args`.
    pub fn can_implement(&self, args: &Fp8GemmArgs<'_, T>) -> Result<()> {
        if self.desc.m <= 0 || self.desc.n <= 0 || self.desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "FP8 GEMM problem must have positive M, N, K",
            ));
        }
        if args.a.rows != self.desc.m || args.a.cols != self.desc.k {
            return Err(Error::InvalidProblem(
                "A shape mismatch with descriptor (M, K)",
            ));
        }
        if args.b.rows != self.desc.k || args.b.cols != self.desc.n {
            return Err(Error::InvalidProblem(
                "B shape mismatch with descriptor (K, N)",
            ));
        }
        // Per-layout stride sanity. RCR: B is col-major [K, N] with
        // column stride = ld ≥ K (kernel walks ldb along K). RRR: B is
        // row-major [K, N] with row stride = ld ≥ N (kernel walks ldb
        // along N).
        match self.sku.layout {
            LayoutSku::Rcr => {
                if args.b.ld < self.desc.k as i64 {
                    return Err(Error::InvalidProblem(
                        "B leading dimension must be >= K for RCR (col-major K-contig)",
                    ));
                }
            }
            LayoutSku::Rrr => {
                if args.b.ld < self.desc.n as i64 {
                    return Err(Error::InvalidProblem(
                        "B leading dimension must be >= N for RRR (row-major N-contig)",
                    ));
                }
            }
        }
        if args.d.rows != self.desc.m || args.d.cols != self.desc.n {
            return Err(Error::InvalidProblem(
                "D shape mismatch with descriptor (M, N)",
            ));
        }
        // Bias presence must match the epilogue family.
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

    /// Workspace size in bytes. Always zero across the FP8 SKU matrix
    /// today — the kernels do all their work in smem + registers.
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> GemmSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee()
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: Fp8GemmArgs<'_, T>,
    ) -> Result<()> {
        let _ = workspace; // FP8 kernels take no scratch

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let (c_ptr, ldc) = match &args.c {
            Some(c) => (c.data.as_raw().0 as *const c_void, c.ld),
            None => (core::ptr::null(), 0i64),
        };
        let bias_ptr = match &args.bias {
            Some(b) => b.data.as_raw().0 as *const c_void,
            None => core::ptr::null(),
        };

        // Reject mis-paired bias / epilogue. Mirrors the contract in
        // `can_implement` so callers that skip the pre-flight still
        // get a precise error.
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
        let beta = args.beta;

        // The 20-SKU matrix is gated entirely on `sm89`. Without it,
        // the FFI symbols don't exist and we must short-circuit before
        // the match arms expand. We early-return here under #[cfg]
        // rather than splitting the function in two so the rest of
        // the body (pointer extraction, bias guards, etc.) is shared.
        #[cfg(not(feature = "sm89"))]
        {
            let _ = (a_ptr, b_ptr, c_ptr, d_ptr, bias_ptr, ldc, lda, ldb, ldd,
                     m, n, k, alpha, beta, stream_ptr);
            return Err(Error::Unsupported(
                "baracuda-kernels: FP8 GEMM requires the `sm89` feature \
                 to be enabled in baracuda-kernels-sys",
            ));
        }

        #[cfg(feature = "sm89")]
        let status = match (T::KIND, self.sku.layout, self.sku.epilogue) {
            // ---------- Identity ----------
            (ElementKind::Fp8E4M3, LayoutSku::Rcr, EpilogueKind::Identity) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    alpha, beta,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, LayoutSku::Rrr, EpilogueKind::Identity) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    alpha, beta,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, LayoutSku::Rcr, EpilogueKind::Identity) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    alpha, beta,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, LayoutSku::Rrr, EpilogueKind::Identity) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    alpha, beta,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },

            // ---------- E4M3 × RCR × Bias family ----------
            (ElementKind::Fp8E4M3, LayoutSku::Rcr, EpilogueKind::Bias) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_bias_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, LayoutSku::Rcr, EpilogueKind::BiasRelu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_bias_relu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, LayoutSku::Rcr, EpilogueKind::BiasGelu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_bias_gelu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, LayoutSku::Rcr, EpilogueKind::BiasSilu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_bias_silu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },

            // ---------- E4M3 × RRR × Bias family ----------
            (ElementKind::Fp8E4M3, LayoutSku::Rrr, EpilogueKind::Bias) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, LayoutSku::Rrr, EpilogueKind::BiasRelu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_relu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, LayoutSku::Rrr, EpilogueKind::BiasGelu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_gelu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, LayoutSku::Rrr, EpilogueKind::BiasSilu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_silu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },

            // ---------- E5M2 × RCR × Bias family ----------
            (ElementKind::Fp8E5M2, LayoutSku::Rcr, EpilogueKind::Bias) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_bias_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, LayoutSku::Rcr, EpilogueKind::BiasRelu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_bias_relu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, LayoutSku::Rcr, EpilogueKind::BiasGelu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_bias_gelu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, LayoutSku::Rcr, EpilogueKind::BiasSilu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_bias_silu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },

            // ---------- E5M2 × RRR × Bias family ----------
            (ElementKind::Fp8E5M2, LayoutSku::Rrr, EpilogueKind::Bias) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_bias_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, LayoutSku::Rrr, EpilogueKind::BiasRelu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_bias_relu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, LayoutSku::Rrr, EpilogueKind::BiasGelu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_bias_gelu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, LayoutSku::Rrr, EpilogueKind::BiasSilu) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_bias_silu_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, c_ptr, ldc, d_ptr, ldd,
                    bias_ptr, alpha, beta,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },

            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels: FP8 GEMM dispatcher reached an \
                     unimplemented (element, layout, epilogue) triple",
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

// Type-level guard that the FP8 element re-exports are in scope; kept
// for clarity in case a future commit adds compile-time
// element-specific dispatch branches that need to refer to them by
// path.
#[allow(dead_code)]
fn _hold_fp8_elements_in_scope() {
    let _ = Fp8E4M3(0);
    let _ = Fp8E5M2(0);
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
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
