//! Generalized kernel SKU descriptor — covers every op category.
//!
//! Sibling to (and a generalization of) `baracuda_cutlass::GemmSku`. The
//! GEMM-specific tag struct (with `layout` / `epilogue` / `bias_element`
//! fields populated and an `op` field implicitly `Gemm`) stays in
//! `baracuda-cutlass` for back-compat — [`KernelSku`] here is the new
//! shape used by Phase 3+ Plan types (`UnaryPlan`, `BinaryPlan`, …).
//!
//! Both shapes coexist intentionally during the Phase 3 transition. A
//! later refactor pass can unify them (or fold `GemmSku::sku()` into a
//! `KernelSku::from(GemmSku)` projection) without disturbing existing
//! callers.

use crate::element::ElementKind;
use crate::layout::{ArchSku, EpilogueKind, LayoutSku};
use crate::plan::PrecisionGuarantee;

/// Op category — the top-level taxonomy a kernel SKU belongs to.
///
/// Mirrors the section letters in
/// `~/.claude/plans/baracuda-kernels-comprehensive.md` §6. The
/// `category` field of [`KernelSku`] is the primary axis a selector /
/// telemetry consumer dispatches on; the per-category `op` discriminant
/// (a category-local `u16`) refines further.
///
/// `#[non_exhaustive]` — new categories may land in future phases as
/// op families expand (sparse linear algebra, structured kernels, …).
/// Match arms must include a `_ =>` catch-all.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum OpCategory {
    /// Matrix multiplication / linear algebra (category A). Includes
    /// `matmul`, `bmm`, `addmm`, etc. — anything routed through a
    /// `Gemm*Plan`.
    Gemm,
    /// Element-wise unary math + activations (categories B + B'). Includes
    /// `neg`, `abs`, `sin`, `exp`, `relu`, `gelu`, …
    UnaryElementwise,
    /// Element-wise binary math (category C). Includes `add`, `sub`,
    /// `mul`, `div`, comparisons, `pow`, etc.
    BinaryElementwise,
    /// Element-wise ternary (category D). Includes `where`, `fma`,
    /// `clamp`, `addcmul`, `addcdiv`.
    TernaryElementwise,
    /// Gated activations (category C'). Includes `glu`, `reglu`,
    /// `swiglu`, `geglu`.
    GatedActivation,
    /// Reductions (category E). Includes `sum`, `mean`, `prod`, `max`,
    /// `min`, `argmax`, `argmin`, `var`, `std`, `norm`, etc.
    Reduction,
    /// Scans (category F). Includes `cumsum`, `cumprod`, `cummax`,
    /// `cummin`, `logcumsumexp`, `associative_scan`.
    Scan,
    /// Normalization (category G). Includes `batch_norm`, `layer_norm`,
    /// `group_norm`, `rms_norm`, `weight_norm`, etc.
    Normalization,
    /// Softmax family (category H). Includes `softmax`, `log_softmax`,
    /// `gumbel_softmax`, `sparsemax`.
    Softmax,
    /// Convolution (category I). Includes `conv1d/2d/3d`,
    /// `conv_transpose*`, `depthwise_conv2d`, `unfold` / `fold`.
    Convolution,
    /// Pooling (category J). Includes `max_pool*`, `avg_pool*`,
    /// `adaptive_*_pool*`, `lp_pool*`, `fractional_max_pool*`.
    Pooling,
    /// Attention / transformer (category K). Includes SDPA, flash
    /// attention, RoPE, ALiBi, KV-cache, paged attention.
    Attention,
    /// Indexing / scatter / gather (category L). Includes `gather`,
    /// `scatter`, `scatter_add`, `index_select`, `masked_*`, etc.
    Indexing,
    /// Embedding (category M). Includes `embedding`, `embedding_bag`.
    Embedding,
    /// Shape / layout / permutation (category N). Includes `reshape`
    /// (when materialized), `transpose` / `permute`, `concat`, `pad`,
    /// `tile`, `repeat`, `roll`, `flip`, `meshgrid`, `eye`, `diag`,
    /// `tril`, `triu`.
    ShapeLayout,
    /// Sorting / topk / order statistics (category O). Includes `sort`,
    /// `argsort`, `topk`, `kthvalue`, `unique`, `histogram`,
    /// `searchsorted`.
    Sorting,
    /// Quantization helpers (category P). Includes `quantize_per_*`,
    /// `dequantize_*`, `fake_quantize`, `dynamic_range_quantize`,
    /// `quantized_linear`.
    Quantization,
    /// Random / sampling (category Q). Includes `dropout`, `bernoulli`,
    /// `multinomial`, `uniform_`, `normal_`, RNG state ops.
    Random,
    /// Loss functions (category R). Includes `mse_loss`,
    /// `cross_entropy_loss`, `kl_div`, `bce_loss`, etc.
    Loss,
    /// Segment ops (category S). Includes `segment_sum`,
    /// `segment_mean`, etc.
    SegmentOps,
    /// Image-domain ops (category T). Includes `interpolate`,
    /// `grid_sample`, ROI ops, NMS.
    Image,
    /// FFT (category U). Includes `rfft`, `ifft`, `fftshift`, etc.
    Fft,
    /// Dense linear algebra factorizations / decompositions / solves.
    /// Includes Cholesky, LU, QR, SVD, matrix inverse, eigen-
    /// decomposition, linear solve, least-squares — the cuSOLVER family.
    Linalg,
    /// Mixture-of-Experts inference (category V). Fused per-token
    /// dispatch + expert matmul + accumulate. Covers FP (`MoeKind::Wmma`),
    /// GGUF-quantized weights (`MoeKind::ScalarGguf`), and the combined
    /// WMMA + GGUF hot path (`MoeKind::WmmaGguf`) used by quantized LLM
    /// inference.
    Moe,
}

/// Which underlying compute backend served a kernel SKU.
///
/// Surfaced through [`KernelSku::backend`] for telemetry, autotuner
/// cache keys, and selector debugging.
///
/// `#[non_exhaustive]` — new backends (TensorRT, custom JIT-emitted
/// kernels via baracuda-nvrtc, …) may land in future phases. Match arms
/// must include a `_ =>` catch-all.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum BackendKind {
    /// Hand-rolled kernel in `baracuda-kernels-sys`.
    Bespoke,
    /// CUTLASS template instantiation in `baracuda-cutlass-kernels-sys`.
    Cutlass,
    /// `baracuda-cublas` wrapper of cuBLAS / cuBLASLt.
    Cublas,
    /// `baracuda-cudnn` wrapper of cuDNN (graph or legacy API).
    Cudnn,
    /// `baracuda-cufft` wrapper of cuFFT.
    Cufft,
    /// `baracuda-cusparse` wrapper of cuSPARSE / cuSPARSELt.
    Cusparse,
    /// `baracuda-cusolver` wrapper of cuSOLVER.
    Cusolver,
    /// `baracuda-curand` wrapper of cuRAND.
    Curand,
    /// `baracuda-cutensor` wrapper of cuTENSOR.
    Cutensor,
    /// `baracuda-npp` wrapper of NPP.
    Npp,
    /// `baracuda-cvcuda` wrapper of CV-CUDA.
    Cvcuda,
    /// Vendored Dao-AILab FlashAttention v2 (BSD-3-Clause). Phase 42
    /// added this as a backend choice on `FlashSdpaPlan` for the long-
    /// context regime where FA2's tiling wins over the bespoke kernel.
    FlashAttentionV2,
    /// Vendored FlashInfer (Apache-2.0). Phase 46 added three plan
    /// families backed by FlashInfer cherry-picked headers:
    /// `BatchPagedDecodePlan` (batched paged-KV decode for vLLM-style
    /// serving), `TopKTopPSamplingPlan` (sort-free combined top-K /
    /// top-P / min-P sampling), and `CascadeAttentionPlan` (LSE-merge
    /// for prefix-cache sharing across requests).
    FlashInfer,
    /// Vendored ozIMMU (MIT). Phase 44 backend choice on FP64 `GemmPlan`
    /// that splits each operand into `slices` int8 slices and runs
    /// `slices²` tensor-core matmuls (the Ozaki scheme) to synthesize
    /// a DGEMM on hardware that has no FP64 tensor cores (RTX 4070,
    /// L4, etc.). **Opt-in** — NOT bit-equivalent to native DGEMM;
    /// `slices = 8` is the upstream-recommended sweet spot for
    /// well-conditioned inputs.
    ///
    /// # Slice-count + variant discriminant encoding (Phase 44c)
    ///
    /// The `slices` byte is split into two bit-fields:
    ///
    /// - **Low 5 bits** (`slices & 0x1F`) — slice count `S`:
    ///   - `0` = auto (`fp64_int8_auto`, runtime selection based on
    ///     mantissa-loss histogram).
    ///   - `3..=18` = fixed slice count (`fp64_int8_3` .. `fp64_int8_18`).
    ///
    /// - **High 3 bits** (`slices >> 5`) — Phase 44c variant flag:
    ///   - `0` = Base (original ozIMMU; default for back-compat with
    ///     Phase 44/44b callers).
    ///   - `1` = EF (group-wise error-free summation; ~5–15% faster
    ///     at the same accuracy).
    ///   - `2` = RN (nearest-rounding split; ~2 extra effective bits
    ///     per slice).
    ///   - `3` = H (EF + RN combined).
    ///
    /// Use the [`ozaki_slices`] helper constructors for ergonomic
    /// construction (`ozaki_slices::ef(8)` → `40` = EF variant at
    /// S=8). Values with any other bit pattern are rejected at
    /// plan-select time.
    ///
    /// n-blocking (chunk large-N int8 GEMMs into 8192-wide pieces)
    /// is applied automatically by the C++ shim regardless of the
    /// variant flag.
    Ozaki {
        /// Slice count + variant discriminant — see the
        /// [`BackendKind::Ozaki`] doc-comment for the bit-field
        /// layout. Prefer the [`ozaki_slices`] helpers over raw
        /// integer construction.
        slices: u8,
    },
}

/// Phase 44c — helper constructors for the
/// [`BackendKind::Ozaki { slices }`] discriminant.
///
/// Encodes a slice count + variant pair into a single byte per the
/// `low-5 = slices, high-3 = variant` convention. Each helper takes a
/// raw slice count in `0..=18` and returns the encoded discriminant.
///
/// ```ignore
/// use baracuda_kernels_types::{BackendKind, sku::ozaki_slices};
///
/// // Original ozIMMU at S=8 (Phase 44b default — equivalent to
/// // passing `slices: 8` directly).
/// let base = BackendKind::Ozaki { slices: ozaki_slices::base(8) };
///
/// // RIKEN EF variant at S=8 — ~5-15% faster, same accuracy.
/// let ef = BackendKind::Ozaki { slices: ozaki_slices::ef(8) };
///
/// // RIKEN RN variant at S=4 — similar accuracy to Base at S=5-6.
/// let rn = BackendKind::Ozaki { slices: ozaki_slices::rn(4) };
///
/// // RIKEN H variant at S=8 — best of EF + RN.
/// let h = BackendKind::Ozaki { slices: ozaki_slices::h(8) };
/// ```
pub mod ozaki_slices {
    /// Base ozIMMU at the given slice count `s` (0 = auto, 3..=18 = fixed).
    /// Equivalent to passing `s` directly — preserved for symmetry
    /// with the other variant constructors.
    pub const fn base(s: u8) -> u8 {
        s
    }
    /// EF (group-wise error-free summation) variant.
    pub const fn ef(s: u8) -> u8 {
        (1 << 5) | (s & 0x1F)
    }
    /// RN (nearest-rounding split) variant.
    pub const fn rn(s: u8) -> u8 {
        (2 << 5) | (s & 0x1F)
    }
    /// H (EF + RN combined) variant.
    pub const fn h(s: u8) -> u8 {
        (3 << 5) | (s & 0x1F)
    }
}

/// Generalized kernel SKU — covers every op category.
///
/// Replaces / generalizes the GEMM-specific `GemmSku` used by Phase 1/2
/// plan types. Phase 3+ plan types (`UnaryPlan`, `BinaryPlan`,
/// `TernaryPlan`, …) populate and return this struct from their
/// `sku()` accessor.
///
/// The `op` field is a **category-local discriminant** — its
/// interpretation depends on `category`. For
/// [`OpCategory::BinaryElementwise`] it's a `BinaryKind as u16`; for
/// [`OpCategory::UnaryElementwise`] it's a `UnaryKind as u16`; etc.
/// Surfacing it as a flat `u16` (rather than a per-category enum) keeps
/// the struct shape stable across categories so it can be hashed into
/// autotuner caches uniformly.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct KernelSku {
    /// Op category. Primary axis for telemetry / selector dispatch.
    pub category: OpCategory,
    /// Category-local op discriminant. Interpret against the per-category
    /// `*Kind` enum that the category's Plan type uses.
    pub op: u16,
    /// Primary element type the kernel operates on.
    pub element: ElementKind,
    /// Auxiliary element type, when meaningful (bias element for GEMM
    /// bias epilogues, index element for gather / scatter, output cast
    /// type for ops that produce a different dtype than the input).
    /// `None` when the op has no auxiliary element.
    pub aux_element: Option<ElementKind>,
    /// Layout discriminant for matrix-multiply-shaped kernels. `None`
    /// for op categories that don't have a row/col layout dimension
    /// (elementwise, reduce, scan, …).
    pub layout: Option<LayoutSku>,
    /// Epilogue discriminant for matrix-multiply-shaped kernels. `None`
    /// for op categories that don't fuse a GEMM epilogue chain.
    pub epilogue: Option<EpilogueKind>,
    /// Compute capability the selected kernel was compiled for.
    pub arch: ArchSku,
    /// Which underlying compute path served this SKU.
    pub backend: BackendKind,
    /// Numerical guarantees this SKU provides.
    pub precision_guarantee: PrecisionGuarantee,
}
