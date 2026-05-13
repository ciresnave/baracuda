//! Binary (B1) GEMM plan — Identity-only.
//!
//! Distinct programming model from the float / int / FP8 / int4 GEMM
//! families:
//!
//! - Operation: `D[i, j] = sum_k popcount(A[i, k_byte] XOR B[k_byte, j])`
//!   computed as a raw `int32` accumulator. No re-quantization back to
//!   the input element type (b1), no α / β / bias / activation chain
//!   (the popcount programming model doesn't have a meaningful place
//!   for them).
//!
//! - PTX intrinsic:
//!   `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`.
//!
//! - Output type: **`i32`** (the raw popcount sum). Callers that want
//!   a thresholded binary result or the canonical "binary dot product"
//!   `K - 2 * popcount` (the convention when ±1 is mapped to 1/0)
//!   should post-process externally.
//!
//! Plan-layer conventions:
//!
//! - Descriptor `M / N / K` in **element** counts; `K` must be a
//!   multiple of 8 (packing is byte-aligned).
//! - `MatrixRef<Bin>::ld` in **storage-slot (= byte) counts** — each
//!   storage slot holds 8 packed bits along K (LSB = lowest K index).
//! - `MatrixMut<i32>::ld` in **i32 element counts** — D is a plain
//!   `int32` matrix with no packing.
//!
//! Ships both `LayoutSku::Rcr` and `LayoutSku::Rrr`. The RRR variant
//! uses a **bit-gather B-load** (8 gmem byte reads per smem byte): B
//! is row-major and bit-packed along N in gmem, but the MMA fragment
//! wants B bit-packed along K within each output column. The load
//! gathers the bit at `(col_g & 7)` from each of 8 K-row gmem bytes
//! and OR's them into one K-pair smem byte. Bandwidth-heavy but
//! correct; the bin RRR use case is rare so optimization is future
//! work. The RRR variant additionally requires `N` divisible by 8
//! (the gmem byte boundary for B); the RCR variant only requires `K`
//! divisible by 8.

use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, Bin, ElementKind, EpilogueKind, LayoutSku, MatrixMut, MatrixRef, PlanPreference,
    PrecisionGuarantee, Workspace,
};

pub use baracuda_cutlass::GemmSku;

/// Problem shape and configuration for [`BinGemmPlan::select`].
#[derive(Copy, Clone, Debug)]
pub struct BinGemmDescriptor {
    /// Output row count, in elements.
    pub m: i32,
    /// Output column count, in elements.
    pub n: i32,
    /// Reduction depth, in elements (= bits). Must be a multiple of 8.
    pub k: i32,
    /// Layout SKU. [`LayoutSku::Rcr`] or [`LayoutSku::Rrr`]. The RRR
    /// variant additionally requires `N` divisible by 8 (B is bit-
    /// packed along N in gmem).
    pub layout: LayoutSku,
}

/// Per-launch arguments for a [`BinGemmPlan::run`] call.
///
/// No `c`, `alpha`, `beta`, or `bias` — bin GEMM is Identity-only.
#[derive(Debug)]
pub struct BinGemmArgs<'a> {
    /// Left input. Row-major `[M, K]` packed-bit storage; `ld` in
    /// bytes (≥ `K / 8`).
    pub a: MatrixRef<'a, Bin>,
    /// Right input. RCR: col-major `[K, N]` packed-bit storage; `ld`
    /// in bytes (≥ `K / 8`).
    pub b: MatrixRef<'a, Bin>,
    /// Output. Row-major `[M, N]` plain int32 storage (raw popcount
    /// accumulator); `ld` in **i32 element counts** (≥ `N`).
    pub d: MatrixMut<'a, i32>,
}

/// Binary GEMM plan.
pub struct BinGemmPlan {
    desc: BinGemmDescriptor,
    sku: GemmSku,
}

impl BinGemmPlan {
    /// Pick a binary GEMM kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &BinGemmDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "bin GEMM problem must have positive M, N, K",
            ));
        }
        if (desc.k & 7) != 0 {
            return Err(Error::InvalidProblem(
                "bin GEMM requires K to be a multiple of 8 (packed-bit storage)",
            ));
        }
        if matches!(desc.layout, LayoutSku::Rrr) && (desc.n & 7) != 0 {
            return Err(Error::InvalidProblem(
                "bin GEMM RRR requires N to be a multiple of 8 \
                 (B is bit-packed along N in gmem)",
            ));
        }
        if !matches!(desc.layout, LayoutSku::Rcr | LayoutSku::Rrr) {
            return Err(Error::Unsupported(
                "baracuda-kernels: bin GEMM: only RCR / RRR layouts are shipped",
            ));
        }

        let sku = GemmSku {
            arch: ArchSku::Sm89,
            layout: desc.layout,
            epilogue: EpilogueKind::Identity,
            element: ElementKind::Bin,
            // No bias chain.
            bias_element: None,
        };
        Ok(Self { desc: *desc, sku })
    }

    /// Validate that this plan can launch with `args`.
    pub fn can_implement(&self, args: &BinGemmArgs<'_>) -> Result<()> {
        if self.desc.m <= 0 || self.desc.n <= 0 || self.desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "bin GEMM problem must have positive M, N, K",
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
        let k_bytes_min = (self.desc.k / 8) as i64;
        let n_bytes_min = (self.desc.n / 8) as i64;
        if args.a.ld < k_bytes_min {
            return Err(Error::InvalidProblem(
                "A leading dimension (bytes) must be >= K/8 for row-major bin A",
            ));
        }
        let b_ld_min = match self.sku.layout {
            LayoutSku::Rcr => k_bytes_min,
            LayoutSku::Rrr => n_bytes_min,
        };
        if args.b.ld < b_ld_min {
            return Err(Error::InvalidProblem(
                "B leading dimension (bytes) must be >= K/8 for RCR \
                 (col-major) or >= N/8 for RRR (row-major)",
            ));
        }
        if args.d.ld < self.desc.n as i64 {
            return Err(Error::InvalidProblem(
                "D leading dimension (i32 elements) must be >= N for row-major i32 D",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes. Zero today — the kernel does all its
    /// work in smem + registers.
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> GemmSku {
        self.sku
    }

    /// Numerical guarantees: int1 → int32 popcount-sum accumulator.
    /// Bit-stable on the same hardware (integer popcount has no warp-
    /// reduction nondeterminism).
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee()
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: BinGemmArgs<'_>,
    ) -> Result<()> {
        let _ = workspace; // bin kernel takes no scratch

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;

        let stream_ptr = stream.as_raw() as *mut c_void;
        let m = self.desc.m;
        let n = self.desc.n;
        let k = self.desc.k;
        let lda = args.a.ld;
        let ldb = args.b.ld;
        let ldd = args.d.ld;

        #[cfg(not(feature = "sm89"))]
        {
            let _ = (a_ptr, b_ptr, d_ptr, lda, ldb, ldd, m, n, k, stream_ptr);
            return Err(Error::Unsupported(
                "baracuda-kernels: bin GEMM requires the `sm89` feature \
                 to be enabled in baracuda-kernels-sys",
            ));
        }

        #[cfg(feature = "sm89")]
        let status = match self.sku.layout {
            LayoutSku::Rcr => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_bin_rcr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, d_ptr, ldd,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
            LayoutSku::Rrr => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gemm_bin_rrr_sm89_run(
                    m, n, k,
                    a_ptr, lda, b_ptr, ldb, d_ptr, ldd,
                    core::ptr::null_mut(), 0,
                    stream_ptr,
                )
            },
        };

        #[cfg(feature = "sm89")]
        { map_status(status) }
        #[cfg(not(feature = "sm89"))]
        #[allow(unreachable_code)]
        { unreachable!("returned earlier under #[cfg(not(feature = \"sm89\"))]") }
    }
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
             (bin GEMM requires K to be a multiple of 8)",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
