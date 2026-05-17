//! Per-category op discriminant enums.
//!
//! Each op category (B, C, D, N, …) gets a `*Kind` enum whose variants
//! correspond to individual PyTorch / JAX ops. The enum value is also
//! the runtime tag stored as a `u16` in [`crate::KernelSku::op`].
//!
//! New enums land alongside the Plan type that consumes them. Today
//! Phase 3 contributes [`BinaryKind`] and [`UnaryKind`];
//! [`TernaryKind`], [`GatedActivationKind`], [`ShapeLayoutKind`] follow
//! as their Plan types ship.

/// Binary elementwise op discriminant.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::BinaryElementwise`. Variants correspond to
/// the union of PyTorch (`torch.<op>` / `torch.Tensor.<op>`) and JAX
/// (`jax.numpy.<op>` / `jax.lax.<op>`) binary elementwise ops.
///
/// Today only [`Self::Add`] is wired — the Phase 3 trailblazer SKU. The
/// other variants are reserved discriminants for the fanout sessions
/// that ship sub / mul / div / pow / comparisons / bitwise.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum BinaryKind {
    /// `y = a + b` — elementwise addition. Trailblazer SKU for
    /// `baracuda-kernels` Phase 3.
    Add = 0,
    /// `y = a - b` — elementwise subtraction.
    Sub = 1,
    /// `y = a * b` — elementwise multiplication.
    Mul = 2,
    /// `y = a / b` — elementwise division.
    Div = 3,
    /// `y = floor(a / b)` — elementwise floor-divide.
    FloorDivide = 4,
    /// `y = a mod b` — elementwise Python-style modulo (sign matches `b`).
    Mod = 5,
    /// `y = remainder(a, b)` — elementwise C-style remainder (sign
    /// matches `a`).
    Remainder = 6,
    /// `y = a ** b` — elementwise power (broadcast scalar exponent OK).
    Pow = 7,
    /// `y = atan2(a, b)`.
    Atan2 = 8,
    /// `y = hypot(a, b) = sqrt(a² + b²)`.
    Hypot = 9,
    /// `y = a` with sign-bit copied from `b`.
    Copysign = 10,
    /// `y` = next representable value from `a` toward `b`.
    Nextafter = 11,
    /// `y = a · 2^b` (integer `b` broadcast as scalar in practice).
    Ldexp = 12,
    /// `y = min(a, b)` — IEEE 754 semantics (NaN-aware).
    Minimum = 13,
    /// `y = max(a, b)` — IEEE 754 semantics (NaN-aware).
    Maximum = 14,
    /// `y = fmin(a, b)` — PyTorch fmin (NaN-propagating-from-other).
    Fmin = 15,
    /// `y = fmax(a, b)` — PyTorch fmax (NaN-propagating-from-other).
    Fmax = 16,
    /// `y = (a == b)` — returns bool.
    Eq = 17,
    /// `y = (a != b)` — returns bool.
    Ne = 18,
    /// `y = (a > b)` — returns bool.
    Gt = 19,
    /// `y = (a >= b)` — returns bool.
    Ge = 20,
    /// `y = (a < b)` — returns bool.
    Lt = 21,
    /// `y = (a <= b)` — returns bool.
    Le = 22,
    /// `y = a && b` — bool only.
    LogicalAnd = 23,
    /// `y = a || b` — bool only.
    LogicalOr = 24,
    /// `y = a ^ b` (logical) — bool only.
    LogicalXor = 25,
    /// `y = a & b` — integer only.
    BitwiseAnd = 26,
    /// `y = a | b` — integer only.
    BitwiseOr = 27,
    /// `y = a ^ b` (bitwise) — integer only.
    BitwiseXor = 28,
    /// `y = a << b` — integer only.
    BitwiseLeftShift = 29,
    /// `y = a >> b` — integer only.
    BitwiseRightShift = 30,
    /// `y = a + (b - a) * weight` (broadcast scalar weight). Per
    /// PyTorch's `torch.lerp` convention.
    Lerp = 31,
}

/// Unary elementwise op discriminant.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::UnaryElementwise`. Variants correspond to
/// the union of PyTorch (`torch.<op>` / `torch.Tensor.<op>`) and JAX
/// (`jax.numpy.<op>` / `jax.lax.<op>`) unary elementwise ops, plus the
/// activation family from PyTorch `nn.functional`.
///
/// Today only [`Self::Neg`] is wired — the Phase 3 unary trailblazer
/// SKU. The other variants are reserved discriminants for the fanout
/// sessions that ship the math (abs / sqrt / exp / log / sin / …) and
/// activation (relu / gelu / silu / …) families.
///
/// Ops that return a different dtype than the input (`isnan`, `isinf`,
/// `isfinite`, `logical_not`) are reserved here but will route through
/// a future `UnaryToBoolPlan` (or similar) with a distinct output type
/// — not through this enum's `UnaryPlan<T, N>`.
///
/// Parameterized activations (`leaky_relu(α)`, `elu(α)`, `threshold(t, v)`,
/// `hardshrink(λ)`, `softshrink(λ)`) carry their parameters via a
/// `UnaryParams` field on the descriptor — landed when the first
/// parameterized op ships, omitted for the trailblazer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum UnaryKind {
    // ---- Category B: elementwise unary (math) — trivial ----
    /// `y = -x` — elementwise negation. Trailblazer SKU.
    Neg = 0,
    /// `y = |x|` — elementwise absolute value.
    Abs = 1,
    /// `y = sign(x)` — `-1` / `0` / `+1` per the input's sign.
    Sign = 2,
    /// `y = 1 / x` — elementwise reciprocal.
    Reciprocal = 3,
    /// `y = x * x` — elementwise square.
    Square = 4,
    /// `y = x * x * x` — elementwise cube.
    Cube = 5,

    // ---- Category B: roots ----
    /// `y = sqrt(x)`.
    Sqrt = 10,
    /// `y = 1 / sqrt(x)` — reciprocal square root.
    Rsqrt = 11,
    /// `y = cbrt(x)` — cube root.
    Cbrt = 12,

    // ---- Category B: exp / log family ----
    /// `y = exp(x)`.
    Exp = 20,
    /// `y = 2^x`.
    Exp2 = 21,
    /// `y = exp(x) - 1`.
    Expm1 = 22,
    /// `y = ln(x)` — natural log.
    Log = 23,
    /// `y = log_2(x)`.
    Log2 = 24,
    /// `y = log_10(x)`.
    Log10 = 25,
    /// `y = ln(1 + x)`.
    Log1p = 26,

    // ---- Category B: trig ----
    /// `y = sin(x)`.
    Sin = 30,
    /// `y = cos(x)`.
    Cos = 31,
    /// `y = tan(x)`.
    Tan = 32,
    /// `y = asin(x)`.
    Asin = 33,
    /// `y = acos(x)`.
    Acos = 34,
    /// `y = atan(x)`.
    Atan = 35,

    // ---- Category B: hyperbolic ----
    /// `y = sinh(x)`.
    Sinh = 40,
    /// `y = cosh(x)`.
    Cosh = 41,
    /// `y = tanh(x)`.
    Tanh = 42,
    /// `y = asinh(x)`.
    Asinh = 43,
    /// `y = acosh(x)`.
    Acosh = 44,
    /// `y = atanh(x)`.
    Atanh = 45,

    // ---- Category B: rounding ----
    /// `y = floor(x)`.
    Floor = 50,
    /// `y = ceil(x)`.
    Ceil = 51,
    /// `y = round(x)` — round-half-to-even (PyTorch convention).
    Round = 52,
    /// `y = trunc(x)` — truncate toward zero.
    Trunc = 53,
    /// `y = x - trunc(x)` — fractional part with sign of `x`.
    Frac = 54,

    // ---- Category B: special functions ----
    /// `y = erf(x)`.
    Erf = 60,
    /// `y = erfc(x) = 1 - erf(x)`.
    Erfc = 61,
    /// `y = erfinv(x)`.
    Erfinv = 62,
    /// `y = lgamma(x) = ln(|Γ(x)|)`.
    Lgamma = 63,
    /// `y = digamma(x) = Γ'(x) / Γ(x)`.
    Digamma = 64,

    // ---- Category B: bitwise / integer (int-typed only) ----
    /// `y = ~x` — bitwise NOT (integer dtypes).
    BitwiseNot = 70,
    /// `y = popcount(x)` — population count of set bits (integer).
    Popcount = 71,
    /// `y = clz(x)` — count leading zeros (integer).
    Clz = 72,
    /// `y = ctz(x)` — count trailing zeros (integer).
    Ctz = 73,

    // ---- Category B': activations (unparameterized) ----
    /// `y = relu(x) = max(x, 0)`.
    Relu = 100,
    /// `y = gelu(x)` — exact (erf-based) Gaussian Error Linear Unit.
    Gelu = 101,
    /// `y = gelu_tanh(x)` — tanh-approximate GELU.
    GeluTanh = 102,
    /// `y = silu(x) = x · sigmoid(x)`. Also known as Swish-1.
    Silu = 103,
    /// `y = mish(x) = x · tanh(softplus(x))`.
    Mish = 104,
    /// `y = sigmoid(x) = 1 / (1 + exp(-x))`.
    Sigmoid = 105,
    /// `y = logit(x) = log(x / (1 - x))`. Inverse of sigmoid.
    Logit = 106,
    /// `y = softplus(x) = ln(1 + exp(x))`.
    Softplus = 107,
    /// `y = softsign(x) = x / (1 + |x|)`.
    Softsign = 108,
    /// `y = tanhshrink(x) = x - tanh(x)`.
    Tanhshrink = 109,
    /// `y = relu6(x) = min(max(x, 0), 6)`.
    Relu6 = 110,
    /// `y = hardswish(x)` — piecewise-linear approximation of swish.
    Hardswish = 111,
    /// `y = hardsigmoid(x)` — piecewise-linear approximation of sigmoid.
    Hardsigmoid = 112,
    /// `y = hardtanh(x, -1, +1)` — piecewise-linear clamp.
    Hardtanh = 113,
    /// `y = selu(x)` — scaled exponential linear unit.
    Selu = 114,
    /// `y = leaky_relu(x) = x if x > 0 else α·x`. Hardcoded α = 0.01 in
    /// the current bespoke kernel; will re-emit as a fanout from a
    /// parameterized-unary plan once that infrastructure lands.
    LeakyRelu = 115,
    /// `y = elu(x) = x if x > 0 else α·(exp(x) - 1)`. Hardcoded α = 1.0
    /// in the current bespoke kernel; same parameterization story as
    /// `LeakyRelu`.
    Elu = 116,
    /// `y = hardshrink(x) = x if |x| > λ else 0`. Hardcoded λ = 0.5 in
    /// the current bespoke kernel; same parameterization story as
    /// `LeakyRelu`.
    Hardshrink = 117,
    /// `y = softshrink(x) = x - λ if x > λ; x + λ if x < -λ; else 0`.
    /// Hardcoded λ = 0.5 in the current bespoke kernel; same
    /// parameterization story as `LeakyRelu`.
    Softshrink = 118,
    /// Reserved — `threshold(x; t, v) = x if x > t else v`. Needs the
    /// parameterized-unary plan (two scalar parameters); not wired yet.
    Threshold = 119,
    /// `prelu(x; α) = x if x > 0 else α·x` with per-channel learnable α
    /// vector (or single scalar α). Uses a distinct plan shape
    /// (`PReluPlan` / `PReluBackwardPlan`) because α is a tensor operand,
    /// not a scalar parameter. Wired in Milestone 5.3.
    PReLU = 120,

    // ---- Category B: dtype / scalar-shape ops ----
    /// `y = (TOut) x` — dtype conversion. Heterogeneous input / output
    /// element types, so it goes through its own `CastPlan` (not the
    /// same-dtype `UnaryPlan<T, N>`). The discriminant lives here for
    /// telemetry / SKU-tagging consistency with the rest of the unary
    /// family. Wired from `fuel-cuda-kernels/cast.cu`.
    Cast = 130,
    /// `y = a * x + b` — fused affine (multiply-add) with scalar
    /// parameters `a` / `b`. Same-dtype input/output but carries two
    /// scalar parameters, so it gets its own `AffinePlan` (the unified
    /// `UnaryPlan<T, N>` doesn't carry kernel parameters). Wired from
    /// `fuel-cuda-kernels/affine.cu`.
    Affine = 131,
}

/// Ternary elementwise op discriminant.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::TernaryElementwise`. Same-dtype-input,
/// same-dtype-output ops only — [`Self::Where`] (which takes a bool
/// cond + two value tensors) is reserved here but won't be wired via
/// the same-dtype `TernaryPlan<T, N>`; it gets its own plan shape in
/// a future session.
///
/// Today only [`Self::Clamp`] on `f32` is wired — the Phase 3 ternary
/// trailblazer SKU. The remaining ops + non-f32 dtypes follow in
/// fanout sessions.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum TernaryKind {
    /// `y = min(max(x, lo), hi)` — clamp `x` to `[lo, hi]`. Trailblazer.
    Clamp = 0,
    /// `y = a * b + c` — fused multiply-add. PyTorch `torch.addcmul(c, a, b)`
    /// with value = 1.
    Fma = 1,
    /// `y = self + value * t1 * t2` — PyTorch `addcmul`. Reserved for
    /// a future parameterized-ternary path (the scalar `value` is a
    /// runtime parameter, not a tensor operand).
    Addcmul = 2,
    /// `y = self + value * t1 / t2` — PyTorch `addcdiv`. Same
    /// parameterization story as `Addcmul`.
    Addcdiv = 3,
    /// `y = cond ? a : b` — element-select. Heterogeneous-dtype inputs
    /// (cond is bool, a / b match output type) — needs its own plan
    /// shape, won't be wired via the same-dtype `TernaryPlan`.
    Where = 4,
}

/// Gated-activation op discriminant (category C').
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::GatedActivation`. All variants follow the
/// same shape: split input `x` along `split_dim` into two halves
/// `(a, b)`, output `y = a · gate(b)`. The `gate` function varies by
/// variant.
///
/// Today the FW + BW are wired for `{Glu, ReGlu, SwiGlu, GeGlu} × {f32,
/// f16, bf16, f64}`. SwiGLU is the trailblazer (highest LLM relevance).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum GatedActivationKind {
    /// `y = a · sigmoid(b)` — PyTorch `torch.nn.functional.glu`.
    Glu = 0,
    /// `y = a · relu(b)`.
    ReGlu = 1,
    /// `y = a · silu(b) = a · b · sigmoid(b)` — Llama / Mistral / Gemma.
    SwiGlu = 2,
    /// `y = a · gelu(b)` (exact, erf-based).
    GeGlu = 3,
}

/// Padding mode for [`crate::ops::ShapeLayoutKind::Pad`].
///
/// Today only [`Self::Constant`] is wired in the Phase 3 trailblazer.
/// Reflect / Replicate / Circular follow in fanout sessions — each
/// changes the kernel body's "what value goes in the pad region"
/// branch but keeps the same plan shape.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum PadMode {
    /// Pad with a constant value (`PadDescriptor::value`).
    Constant = 0,
    /// Reflect input across the boundary (no edge duplication).
    Reflect = 1,
    /// Replicate the boundary value into the pad region.
    Replicate = 2,
    /// Wrap-around padding (also called "circular").
    Circular = 3,
}

/// Shape / layout op discriminant — Category N.
///
/// Tags the kernel SKU for telemetry / autotuner-cache keys. Each
/// variant has its own Plan type today (PadPlan, ConcatPlan, …)
/// because their descriptor / args shapes differ enough that one
/// `ShapeLayoutPlan<T, N>` doesn't fit. The enum exists so all of
/// them populate `KernelSku::op` from a shared discriminant space.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum ShapeLayoutKind {
    /// `F.pad(x, pad, mode='constant', value=v)` — Phase 3 trailblazer.
    Pad = 0,
    /// `torch.cat(tensors, dim)` — variable-arity input. Reserved.
    Concat = 1,
    /// Materialized `torch.permute(x, dims)` (strided-view materialization
    /// when needed). Reserved.
    Permute = 2,
    /// `x.repeat(...)` / `torch.tile(x, ...)`. Reserved.
    Repeat = 3,
    /// `torch.flip(x, dims)` — reverse along axes. Reserved.
    Flip = 4,
    /// `torch.roll(x, shifts, dims)` — shift along axes. Reserved.
    Roll = 5,
    /// `torch.meshgrid(*tensors)` — N rank-1 → N rank-N. Reserved.
    Meshgrid = 6,
    /// `torch.full(shape, value)` / `Tensor.fill_(value)` — fill every
    /// element of an output tensor with a scalar constant. Wired from
    /// `fuel-cuda-kernels/fill.cu`.
    Fill = 7,
}

/// Index-returning reduction discriminant — Phase 4 (`ArgReducePlan`).
///
/// Distinct from [`ReduceKind`] because the output dtype is i64
/// (index), not the input value dtype. Goes through its own plan
/// shape (`ArgReducePlan<T, N>`) for the heterogeneous-output-dtype
/// case.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum ArgReduceKind {
    /// Index of the maximum along the reduced axis. Ties broken by
    /// first occurrence (smallest index wins) — PyTorch convention.
    Argmax = 0,
    /// Index of the minimum along the reduced axis.
    Argmin = 1,
}

/// Reduction op discriminant — Phase 4 (Category E).
///
/// Output shape differs from input: the reduced axis collapses to size
/// 1 (keepdim convention). Other variants are reserved for fanout.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum ReduceKind {
    /// Sum along the reduced axis. Phase 4 trailblazer.
    Sum = 0,
    /// Arithmetic mean along the reduced axis.
    Mean = 1,
    /// Maximum value along the reduced axis.
    Max = 2,
    /// Minimum value along the reduced axis.
    Min = 3,
    /// Product along the reduced axis.
    Prod = 4,
    /// Sample variance (Bessel-corrected) along the reduced axis.
    Var = 5,
    /// Sample standard deviation along the reduced axis.
    Std = 6,
    /// `||x||_2` along the reduced axis.
    Norm2 = 7,
    /// `argmax` along the reduced axis — returns indices (different
    /// output dtype). Will need a separate plan shape, reserved here.
    Argmax = 8,
    /// `argmin` along the reduced axis. Will need a separate plan
    /// shape.
    Argmin = 9,
    /// `any` (logical OR) along the reduced axis.
    Any = 10,
    /// `all` (logical AND) along the reduced axis.
    All = 11,
    /// `logsumexp(x) = log(sum(exp(x - max)))`, numerically stable.
    LogSumExp = 12,
    /// `trace(M) = sum(diag(M))` — sum of the diagonal of a 2-D
    /// square matrix. Reduces *both* axes via the `i == i` constraint
    /// rather than a single reduce-axis, so dispatch goes through a
    /// dedicated `TracePlan` (separate from `ReducePlan`); the
    /// discriminant lives here for telemetry / SKU-tagging consistency
    /// with the rest of the reduction family.
    Trace = 13,
    /// `count_nonzero(x)` along the reduced axis — output is i64
    /// (PyTorch `torch.count_nonzero` returns int64). Heterogeneous
    /// output dtype (always i64 regardless of input), so dispatch
    /// goes through a dedicated `CountReducePlan` (separate from
    /// `ReducePlan`); the discriminant lives here for telemetry /
    /// SKU-tagging consistency with the rest of the reduction family.
    CountNonzero = 14,
}

/// Softmax-family op discriminant — category H from the comprehensive
/// plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Softmax`. All variants apply a
/// length-preserving transform along a single axis (output shape ==
/// input shape — distinct from reductions, like scans).
///
/// Today wired: `{Softmax, LogSoftmax} × {f32, f16, bf16, f64}` —
/// FW + BW. `GumbelSoftmax` (needs RNG state from Phase 4 random) and
/// `Sparsemax` (different gradient — projection onto simplex) are
/// reserved-but-deferred.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum SoftmaxKind {
    /// `y[k] = exp(x[k] - max(x)) / Σ_j exp(x[j] - max(x))`
    /// — numerically stable softmax.
    Softmax = 0,
    /// `y[k] = x[k] - logsumexp(x)` — log-domain softmax, also stable.
    /// Output is the elementwise log of `Softmax(x)`.
    LogSoftmax = 1,
    /// `y = (x + Gumbel(0,1)) / τ → softmax` — reserved.
    GumbelSoftmax = 2,
    /// `y = ProjSimplex(x)` — reserved (different gradient than softmax).
    Sparsemax = 3,
}

/// Scan (associative prefix) op discriminant — category F from the
/// comprehensive plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Scan`. Output shape equals input shape —
/// scans are length-preserving along the scan axis (in contrast with
/// reductions, which collapse the axis to size 1). Inclusive scan by
/// default (PyTorch convention: `y[i] = op(x[0], x[1], …, x[i])`).
/// Direction is controlled by the descriptor's `reverse` flag.
///
/// Today wired: `{Cumsum} × {f32, f16, bf16, f64}` (FW + BW) as the
/// scan trailblazer. Cumprod / Cummax / Cummin land in fanout;
/// LogCumsumExp and the JAX-style generic `associative_scan` are
/// reserved-but-deferred (numerics / generic-functor work).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum ScanKind {
    /// `y[i] = Σ_{j ≤ i} x[j]` — inclusive prefix sum.
    Cumsum = 0,
    /// `y[i] = ∏_{j ≤ i} x[j]` — inclusive prefix product.
    Cumprod = 1,
    /// `y[i] = max(x[0..=i])` — running maximum.
    Cummax = 2,
    /// `y[i] = min(x[0..=i])` — running minimum.
    Cummin = 3,
    /// `y[i] = log(Σ_{j ≤ i} exp(x[j]))` — numerically stable (running
    /// max subtraction). Reserved.
    LogCumsumExp = 4,
}

/// Binary comparison op discriminant.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::BinaryElementwise` and the SKU is from the
/// **comparison family** — distinguished from [`BinaryKind`] because
/// the output dtype is fixed to `u8` (PyTorch / NumPy convention: bool
/// stored as 1 byte, 0 = false, 1 = true) regardless of the input
/// element type.
///
/// Today only [`Self::Eq`] on `f32` is wired — the Phase 3 comparison
/// trailblazer. The other variants are reserved discriminants for the
/// fanout sessions.
///
/// Why a separate enum (rather than reusing [`BinaryKind`]): the
/// dispatch shape differs — these ops produce a different dtype than
/// they consume, so they need their own Plan type
/// (`BinaryCmpPlan<T, N>` with `TensorMut<u8>` output) instead of
/// `BinaryPlan<T, N>` with `TensorMut<T>` output. The reserved Eq /
/// Ne / Gt / Ge / Lt / Le slots in `BinaryKind` are vestigial — they
/// will never be wired into the same-dtype binary path.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum BinaryCmpKind {
    /// `y = (a == b)` — elementwise equality. Trailblazer SKU.
    Eq = 0,
    /// `y = (a != b)`.
    Ne = 1,
    /// `y = (a > b)`.
    Gt = 2,
    /// `y = (a >= b)`.
    Ge = 3,
    /// `y = (a < b)`.
    Lt = 4,
    /// `y = (a <= b)`.
    Le = 5,
}

/// Normalization op discriminant — category G from the comprehensive plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Normalization`. The variants differ in
/// which axes are reduced for the per-row statistics and how the
/// affine parameters (gamma / beta) are indexed.
///
/// Today wired: `{RMSNorm, LayerNorm, BatchNorm, GroupNorm,
/// InstanceNorm} × {f32, f16, bf16, f64}` — FW + BW. RMSNorm /
/// LayerNorm support **multi-axis normalization** via a bitmask
/// (PyTorch's `normalized_shape` — must be a suffix of the input
/// shape). InstanceNorm is implemented as a thin wrapper around
/// GroupNorm with `num_groups == c_extent` (shares kernel symbols).
///
/// BatchNorm is **training-mode-only** for the trailblazer — it
/// computes per-channel stats from the batch and saves them for BW.
/// Inference mode (use of running statistics, reducing to a per-
/// channel affine multiply) is reserved for a follow-up. `WeightNorm`
/// (a parameterization rather than a plain op) and `LocalResponseNorm`
/// (rarely used today) are explicitly deferred.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum NormalizationKind {
    /// `y = x / sqrt(mean(x², over norm_axes) + eps) * gamma`.
    /// Llama / Mistral / Gemma block-pre-norm. Trailblazer SKU.
    RMSNorm = 0,
    /// `y = (x - mean) / sqrt(var + eps) * gamma + beta`. PyTorch's
    /// `torch.nn.LayerNorm` with biased / "population" variance.
    LayerNorm = 1,
    /// Per-group-of-channels statistics. `y[n, c, ...] = (x[n, c, ...] -
    /// mean[n, g]) / sqrt(var[n, g] + eps) * gamma[c] + beta[c]`,
    /// `g = c / (C / num_groups)`. PyTorch `torch.nn.GroupNorm`.
    GroupNorm = 2,
    /// Per-channel statistics across batch + spatial. Training-mode
    /// only — saves `(saved_mean, saved_rstd)` of shape `[C]`. Inference
    /// mode (running stats) deferred. PyTorch `torch.nn.BatchNormNd`.
    BatchNorm = 3,
    /// Per-`(sample, channel)` statistics across spatial only. PyTorch
    /// `torch.nn.InstanceNormNd`. Equivalent to GroupNorm with
    /// `num_groups == num_channels`; same kernel symbols.
    InstanceNorm = 4,
}

/// Loss op discriminant — category R from the comprehensive plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Loss`. Each variant has its own Plan type
/// today (different argument shapes — MSE / BCE / KLDiv take two
/// same-dtype tensor inputs, NLL / CrossEntropy take a `T` input plus an
/// `i64` target index tensor) but they share the [`LossReduction`]
/// enum for selecting per-cell / mean / sum output shape.
///
/// Today wired: `{Mse, Nll, CrossEntropy, Bce, KlDiv} × {f32, f16, bf16,
/// f64}` — FW + BW. `HingeEmbedding`, `L1`, `SmoothL1`, `MarginRanking`,
/// `TripletMargin`, `CtcLoss`, and `PoissonNllLoss` are reserved
/// discriminants for future fanout.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum LossKind {
    /// `y = mean((pred - target)²)` (or sum / per-cell). PyTorch
    /// `torch.nn.functional.mse_loss`.
    Mse = 0,
    /// `y = -mean(input[target_idx[i]])` along the feature axis. PyTorch
    /// `torch.nn.functional.nll_loss`. Heterogeneous-dtype: input `T`,
    /// target `i64`.
    Nll = 1,
    /// `y = NLLLoss(LogSoftmax(input), target)` — fused for numerical
    /// stability. PyTorch `torch.nn.functional.cross_entropy`. Today wired
    /// for class-index target only (`i64`); soft-target CE is reserved.
    CrossEntropy = 2,
    /// `y = -mean(target·log(pred) + (1-target)·log(1-pred))`. PyTorch
    /// `torch.nn.functional.binary_cross_entropy`. Caller ensures
    /// pred ∈ (0, 1).
    Bce = 3,
    /// `y = mean(target·(log(target) - input))`. PyTorch
    /// `torch.nn.functional.kl_div` with the "input is log-prob"
    /// convention.
    KlDiv = 4,
    /// `y = mean(|pred - target|)` (or sum / per-cell). PyTorch
    /// `torch.nn.functional.l1_loss`.
    L1 = 5,
    /// Smooth L1 / "Huber-with-β" loss. PyTorch
    /// `torch.nn.functional.smooth_l1_loss`.
    SmoothL1 = 6,
    /// `y = mean(input if t==1 else max(0, margin - input))`. PyTorch
    /// `torch.nn.functional.hinge_embedding_loss`. Heterogeneous-dtype:
    /// input is `T`, target is `i64` (±1).
    HingeEmbedding = 7,
    /// `y = mean(max(0, -t · (x1 - x2) + margin))`. PyTorch
    /// `torch.nn.functional.margin_ranking_loss`. Target `t` is `T` (±1).
    MarginRanking = 8,
    /// `y = mean(max(0, ||a-p||_p - ||a-n||_p + margin))`. PyTorch
    /// `torch.nn.functional.triplet_margin_loss`. 2-D input `[N, D]`.
    TripletMargin = 9,
    /// Reserved — `torch.nn.functional.ctc_loss`.
    Ctc = 10,
    /// `y = mean(exp(input) - target · input)` (default `log_input=true`).
    /// PyTorch `torch.nn.functional.poisson_nll_loss`.
    PoissonNll = 11,
    /// Huber loss (separate from SmoothL1 — PyTorch
    /// `torch.nn.functional.huber_loss`).
    Huber = 12,
    /// Numerically stable BCE for raw logits. PyTorch
    /// `torch.nn.functional.binary_cross_entropy_with_logits`.
    BceWithLogits = 13,
    /// Gaussian NLL. PyTorch `torch.nn.GaussianNLLLoss`.
    GaussianNll = 14,
    /// `y = (1 - cos(x1, x2))` if `t==1` else `max(0, cos(x1, x2) - margin)`,
    /// then mean. PyTorch `torch.nn.functional.cosine_embedding_loss`.
    /// 2-D input `[N, D]`. Target is `T` (±1.0).
    CosineEmbedding = 15,
    /// `y = mean_i Σ_{j != t_i} max(0, margin - input[i, t_i] + input[i, j])^p / C`.
    /// PyTorch `torch.nn.functional.multi_margin_loss`. Input `[N, C]`,
    /// target `[N]` `i64` class indices.
    MultiMargin = 16,
    /// Multi-label margin loss. PyTorch
    /// `torch.nn.functional.multilabel_margin_loss`. Input `[N, C]`,
    /// target `[N, C]` `i64` (positive class indices followed by -1
    /// padding sentinel).
    MultilabelMargin = 17,
    /// `y = mean(-mean_c(target·log(sigmoid(x)) + (1-target)·log(1-sigmoid(x))))`.
    /// PyTorch `torch.nn.functional.multilabel_soft_margin_loss`.
    /// Input `[N, C]`, target `[N, C]` `T`.
    MultilabelSoftMargin = 18,
}

/// CrossEntropy target-tensor kind. Selects between PyTorch's two
/// target formats: class indices (`i64[N]`) and soft probabilities
/// (`T[N, C]` — used for label smoothing / distillation).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum CrossEntropyTargetKind {
    /// Target is class-index `i64[N]`.
    ClassIndex = 0,
    /// Target is soft probability `T[N, C]` (same dtype as input).
    SoftProb = 1,
}

/// Loss reduction mode. Selects the output shape and the final scalar
/// scaling for a [`LossKind`] plan. PyTorch's `reduction` parameter.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum LossReduction {
    /// Output is per-cell (same shape as the loss surface). No reduction.
    None = 0,
    /// Output is a scalar — sum of per-cell terms divided by element count.
    Mean = 1,
    /// Output is a scalar — sum of per-cell terms (no divide).
    Sum = 2,
}

/// Random / sampling op discriminant.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Random`. Phase 4.5 wires:
/// - [`Self::Uniform`] (f32, f64) — `y ~ U(low, high)` via cuRAND.
/// - [`Self::Normal`] (f32, f64) — `y ~ N(mean, std)` via cuRAND.
/// - [`Self::Bernoulli`] (Bool output) — `y = (rand < p) ? 1 : 0` via
///   cuRAND uniform + custom threshold kernel.
///
/// Multinomial / Randint / exponential / gamma / quasi-random are
/// reserved discriminants for future milestones.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum RandomKind {
    /// `y[i] ~ U(low, high)` — uniform on the half-open interval. Plan
    /// descriptor `param1 = low`, `param2 = high`.
    Uniform = 0,
    /// `y[i] ~ N(mean, std)` — Gaussian. Plan descriptor
    /// `param1 = mean`, `param2 = stddev`.
    Normal = 1,
    /// `y[i] = 1 if uniform < p else 0`, Bool output. Plan descriptor
    /// `param1 = p`. `param2` ignored.
    Bernoulli = 2,
}

/// Linear-algebra (dense) op discriminant — covers the cuSOLVER family
/// shipped in Milestone 6.3.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Linalg`. Today the four canonical PyTorch /
/// JAX dense linalg ops are wired:
///
/// - [`Self::Cholesky`] — `A = L · L^T` (symmetric positive-definite).
///   Batched via `cusolverDnSpotrfBatched` / `cusolverDnDpotrfBatched`.
/// - [`Self::Lu`] — `P · A = L · U`. Batched via
///   `cusolverDnSgetrfBatched` / `cusolverDnDgetrfBatched`.
/// - [`Self::Qr`] — `A = Q · R`. cuSOLVER has no batched variant; 2-D
///   only.
/// - [`Self::Svd`] — `A = U · diag(S) · V^T`. cuSOLVER 2-D only.
///
/// Dtype coverage is `f32` + `f64` — cuSOLVER's dense API does not
/// support `f16` / `bf16` for these factorizations. Reserved variants
/// (`Inverse`, `Eig`, `Solve`, `LeastSquares`, `MatrixExp`) follow in
/// future milestones.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum LinalgKind {
    /// Cholesky factorization `A = L · L^T` (lower) or `A = U^T · U`
    /// (upper). Input must be symmetric positive-definite.
    Cholesky = 0,
    /// LU factorization with partial pivoting `P · A = L · U`. Returns
    /// the packed `LU` factors plus an `i32` pivot vector.
    Lu = 1,
    /// QR factorization `A = Q · R`. Computes full `Q` (`[M, M]`) and
    /// the upper-triangular `R` (`[M, N]`) via `geqrf` + `ormqr`.
    Qr = 2,
    /// Singular value decomposition `A = U · diag(S) · V^T`. cuSOLVER
    /// 2-D only; `full_matrices` controls whether `U`/`V^T` are full
    /// (`[M,M]` / `[N,N]`) or thin (`[M,K]` / `[K,N]`) where
    /// `K = min(M, N)`.
    Svd = 3,
    /// Matrix inverse `A^{-1}` via `getrf` + `getrs` over an identity
    /// RHS. Wired in Milestone 6.9.
    Inverse = 4,
    /// General (non-symmetric) eigen-decomposition `A · v = λ · v`. Wired
    /// via `cusolverDnXgeev` in Milestone 6.12. Always emits complex
    /// eigenvalues (and optional left / right complex eigenvectors).
    Eig = 5,
    /// Linear solve `A · X = B` via `getrf` + `getrs`. Wired in
    /// Milestone 6.9.
    Solve = 6,
    /// Least-squares solve `min ||A·x - b||²` via cuSOLVER's
    /// mixed-precision iterative-refinement `_gels` routine. Wired in
    /// Milestone 6.11.
    LeastSquares = 7,
    /// Reserved — matrix exponential / matrix functions.
    MatrixExp = 8,
    /// Batched QR factorization `A_b = Q_b · R_b` via
    /// `cusolverDn*geqrfBatched`. Wired in Milestone 6.11.
    BatchedQr = 9,
    /// Batched SVD via Jacobi `cusolverDn*gesvdjBatched`. Wired in
    /// Milestone 6.11.
    BatchedSvd = 10,
    /// Symmetric / Hermitian eigen-decomposition `A · v = λ · v` (real
    /// eigenvalues). Wired via `cusolverDn{S,D}syevd` /
    /// `cusolverDn{C,Z}heevd` in Milestone 6.12.
    Eigh = 11,
    /// Rectangular batched approximate-SVD via cuSOLVER's
    /// `gesvdaStridedBatched`. Unlike [`Self::BatchedSvd`] (which is
    /// square-only Jacobi), this routine accepts arbitrary `m × n` per
    /// batch slot, uses element-strides between slots, and reports per-
    /// slot residual Frobenius norms to a host array. Wired in
    /// Milestone 6.15.
    BatchedSvda = 12,
    /// Bespoke batched-`ormqr` — applies the implicit `Q` from a
    /// [`Self::BatchedQr`] packed output to a batch of matrices `C`,
    /// all slots fused into one CUDA launch. cuSOLVER's `ormqr` is
    /// non-batched, so in the small-matrix regime where batched-QR is
    /// most useful the per-slot launch latency dominates; this bespoke
    /// kernel amortizes one launch over the whole batch. Side = Left,
    /// op ∈ {N, T} in the trailblazer (Right + complex variants
    /// deferred). Wired in Milestone 6.14.
    BatchedOrmqr = 13,
    /// Bespoke "materialize dense Q and R from batched-`geqrf` packed
    /// output". Tiny upper-triangle-copy kernel for R; identity-stage
    /// + [`Self::BatchedOrmqr`] for Q. Wired in Milestone 6.14 as the
    /// consumer of `BatchedOrmqrPlan`.
    BatchedQrMaterialize = 14,
    /// WY-blocked batched-`ormqr` — applies the implicit `Q` (or `Q^T`)
    /// from a [`Self::BatchedQr`] packed output to a batch of matrices
    /// `C` at GEMM-rates by fusing groups of `nb` consecutive Householder
    /// reflectors into a block reflector `(I - V·T·V^T)` and applying it
    /// via three cuBLAS strided-batched GEMMs per block. Sibling to
    /// [`Self::BatchedOrmqr`] (the reflector-by-reflector GEMV-rates
    /// variant); callers pick by problem size — WY wins decisively for
    /// `M, N > ~16`, the reflector kernel wins for tiny inputs.
    /// Side = Left, op ∈ {N, T} in the trailblazer. Wired in
    /// Milestone 6.17.
    BatchedOrmqrWy = 15,
}

/// Fill-mode tag for triangular linalg ops (Cholesky / triangular solve).
///
/// Selects whether the factor lives in the lower or upper triangle of
/// the in-place output matrix. The row-major-input → column-major-cuSOLVER
/// adapter at the plan layer flips this when handing the descriptor down
/// to cuSOLVER (a row-major lower-L is bit-identical to a column-major
/// upper-U^T over the same byte storage).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum FillMode {
    /// Lower triangular (the usual PyTorch / scipy default).
    Lower = 0,
    /// Upper triangular.
    Upper = 1,
}

/// FFT-family op discriminant — Category U from the comprehensive plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Fft`. Milestone 6.4 wires the four
/// canonical PyTorch / JAX 1-D FFTs (`fft` / `ifft` / `rfft` / `irfft`)
/// plus the two index-permutation helpers (`fftshift` / `ifftshift`).
///
/// 1-D only for the trailblazer. Multi-D FFTs (`fft2`, `fftn`, …) and
/// arbitrary-axis FFTs follow in fanout sessions — they don't require
/// new cuFFT bindings, just additional descriptor shape + plan glue.
///
/// Dtype coverage: `f32` (single precision) and `f64` (double
/// precision) only. cuFFT's main API does not expose `f16` / `bf16`
/// for native transforms. Callers needing reduced precision must cast
/// on either side. Spectrum-domain tensors use [`crate::Complex32`] /
/// [`crate::Complex64`] for the interleaved real/imag pairs.
///
/// Normalization: forward transforms are unnormalized; inverse
/// transforms are normalized by `1/N` to match PyTorch's
/// `norm="backward"` default. cuFFT itself returns `N · IFFT(x)`; the
/// plan layer multiplies by `1/N` after the inverse exec.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum FftKind {
    /// `y = FFT(x)` — complex-to-complex forward transform (unnormalized).
    /// PyTorch `torch.fft.fft`. Both input and output are complex with
    /// the same shape `[batch, n]`.
    Fft = 0,
    /// `y = IFFT(x)` — complex-to-complex inverse transform, normalized
    /// by `1/N` to match PyTorch's `norm="backward"`. PyTorch
    /// `torch.fft.ifft`. Both input and output are complex `[batch, n]`.
    Ifft = 1,
    /// `y = RFFT(x)` — real-to-complex forward transform (unnormalized).
    /// PyTorch `torch.fft.rfft`. Input is real `[batch, n]`, output is
    /// complex `[batch, n/2 + 1]` (Hermitian-half).
    Rfft = 2,
    /// `y = IRFFT(x, n)` — complex-to-real inverse transform, normalized
    /// by `1/N`. PyTorch `torch.fft.irfft`. Input is complex
    /// `[batch, n/2 + 1]`, output is real `[batch, n]`. The output
    /// length `n` is a required descriptor parameter (cannot be inferred
    /// from the Hermitian-half input shape — both `2*(n/2)` and
    /// `2*(n/2)+1` map to the same Hermitian-half length).
    Irfft = 3,
    /// `fftshift` — shift the zero-frequency component to the center of
    /// the spectrum. PyTorch `torch.fft.fftshift` (matches NumPy's
    /// `np.fft.fftshift`).
    ///
    /// Equivalent to `roll(x, n // 2)`, giving:
    /// `y[i] = x[(i - n // 2) mod n] = x[(i + (n+1) // 2) mod n]`.
    ///
    /// Bit-exact (pure index permutation, no arithmetic on values).
    FftShift = 4,
    /// `ifftshift` — true inverse of `fftshift`:
    /// `ifftshift(fftshift(x)) == x` for any `n`. PyTorch
    /// `torch.fft.ifftshift`.
    ///
    /// Equivalent to `roll(x, -(n // 2))`, giving:
    /// `y[i] = x[(i + n // 2) mod n]`.
    ///
    /// For even `n` this is identical to `fftshift` (the `n/2` offset
    /// is self-inverse mod `n`); for odd `n` the two cyclic offsets
    /// differ by one cell. Bit-exact.
    IfftShift = 5,
}

/// Convolution-family op discriminant — Category I from the
/// comprehensive plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Convolution`. Each variant maps to a
/// distinct cuDNN exec path (forward, data-gradient, filter-gradient)
/// of the underlying convolution descriptor. The dimensional axis
/// (1-D / 2-D / 3-D), padding / stride / dilation, and depthwise /
/// transposed flavors live on the per-plan descriptor — they don't
/// fan out a separate enum slot here.
///
/// Today wired: `Conv2d` × `{f32, f64, f16, bf16}` (FW + BW data +
/// BW filter) via cuDNN. Conv1d / Conv3d / ConvTranspose* / depthwise
/// / `unfold` / `fold` are reserved discriminants for fanout
/// milestones.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum ConvKind {
    /// 2-D convolution forward pass. PyTorch
    /// `torch.nn.functional.conv2d`. Trailblazer for Phase 7.
    Conv2d = 0,
    /// 2-D convolution data-gradient pass (computes `dx` from `dy`
    /// and the filter `w`). PyTorch's autograd-internal
    /// `conv2d_backward_input`.
    Conv2dBackwardData = 1,
    /// 2-D convolution filter-gradient pass (computes `dw` from `x`
    /// and `dy`). PyTorch's autograd-internal
    /// `conv2d_backward_weight`.
    Conv2dBackwardFilter = 2,
    /// 1-D convolution forward. Reserved.
    Conv1d = 3,
    /// 1-D convolution data-gradient. Reserved.
    Conv1dBackwardData = 4,
    /// 1-D convolution filter-gradient. Reserved.
    Conv1dBackwardFilter = 5,
    /// 3-D convolution forward. Reserved.
    Conv3d = 6,
    /// 3-D convolution data-gradient. Reserved.
    Conv3dBackwardData = 7,
    /// 3-D convolution filter-gradient. Reserved.
    Conv3dBackwardFilter = 8,
    /// 2-D transposed convolution (fractionally-strided / "deconv").
    /// Reserved.
    ConvTranspose2d = 9,
    /// 2-D transposed convolution backward. Reserved.
    ConvTranspose2dBackward = 10,
    /// Depthwise 2-D convolution (`groups == c_in`). Reserved — today
    /// callers route through the generic Conv2d plan with cuDNN's
    /// auto-detected depthwise path.
    DepthwiseConv2d = 11,
    /// `torch.nn.functional.unfold` — extract sliding windows. Reserved.
    Unfold = 12,
    /// `torch.nn.functional.fold` — inverse of unfold. Reserved.
    Fold = 13,
}

/// Pooling-family op discriminant — Category J from the comprehensive
/// plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Pooling`. Each variant maps to a distinct
/// cuDNN pooling exec path (forward / backward) for one of three
/// pooling modes: max, average-include-padding, average-exclude-padding.
/// PyTorch's `nn.MaxPool2d` corresponds to [`Self::MaxPool2d`];
/// `nn.AvgPool2d` defaults to `count_include_pad=False` which maps to
/// [`Self::AvgPool2dExcludePad`].
///
/// Today wired: `{MaxPool2d, AvgPool2d} × {f32, f64, f16, bf16}` (FW +
/// BW) via cuDNN. 1-D / 3-D pooling, adaptive pooling, LP-pool, and
/// fractional-max-pool are reserved discriminants for fanout milestones.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum PoolKind {
    /// 2-D max-pool forward. PyTorch `torch.nn.functional.max_pool2d`.
    /// Trailblazer for Phase 7 Milestone 7.2.
    MaxPool2d = 0,
    /// 2-D max-pool backward (data-gradient). PyTorch's autograd-
    /// internal `max_pool2d_with_indices_backward`.
    MaxPool2dBackward = 1,
    /// 2-D average-pool forward, **count-include-padding** denominator.
    /// Matches cuDNN's `*_COUNT_INCLUDE_PADDING` mode.
    AvgPool2dIncludePad = 2,
    /// 2-D average-pool backward, count-include-padding.
    AvgPool2dIncludePadBackward = 3,
    /// 2-D average-pool forward, **count-exclude-padding** denominator
    /// (PyTorch default — `nn.AvgPool2d` with `count_include_pad=False`).
    AvgPool2dExcludePad = 4,
    /// 2-D average-pool backward, count-exclude-padding.
    AvgPool2dExcludePadBackward = 5,
    /// 1-D max-pool forward. Reserved.
    MaxPool1d = 6,
    /// 1-D average-pool forward. Reserved.
    AvgPool1d = 7,
    /// 3-D max-pool forward. Reserved.
    MaxPool3d = 8,
    /// 3-D average-pool forward. Reserved.
    AvgPool3d = 9,
    /// `torch.nn.functional.adaptive_max_pool*` — reserved.
    AdaptiveMaxPool = 10,
    /// `torch.nn.functional.adaptive_avg_pool*` — reserved.
    AdaptiveAvgPool = 11,
    /// `torch.nn.functional.lp_pool*` — reserved.
    LpPool = 12,
    /// `torch.nn.functional.fractional_max_pool*` — reserved.
    FractionalMaxPool = 13,
}

/// Attention-family op discriminant — Category K from the comprehensive
/// plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Attention`. Phase 6.1 wires the two
/// positional-encoding ops [`Self::Rope`] and [`Self::Alibi`]; the rest
/// are reserved discriminants for future milestones (SDPA, FlashAttention,
/// KV-cache, paged attention).
///
/// All variants in this family operate on rank-4 attention-shaped
/// tensors (typically `[batch, num_heads, seq_len, head_dim]` for RoPE
/// or `[batch, num_heads, query_len, key_len]` for attention scores /
/// ALiBi). Plan shapes differ between ops — the discriminant is here
/// for SKU-tagging uniformity, not for shared dispatch.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum AttentionKind {
    /// Rotary position embedding (Llama / Mistral / Gemma / Qwen / Phi).
    /// Rotates pairs of consecutive features `(2i, 2i+1)` of a
    /// `[B, H, S, D]` Q/K tensor by per-position angles
    /// `θ = pos · base^(-2i / D)`. Trailblazer for Phase 6.
    Rope = 0,
    /// Attention with Linear Biases (MPT / BLOOM). Adds the bias
    /// `slope[h] · (j - i)` to attention-score cell `(b, h, i, j)`.
    /// Linear (non-transcendental) FW; BW reduces over the
    /// score-shape axes to recover `dslope[h]`.
    Alibi = 1,
    /// Scaled dot-product attention — reserved.
    Sdpa = 2,
    /// FlashAttention (Tri Dao 2022) — wired in Milestone 6.6. Tiled
    /// fused online-softmax FW kernel that avoids materializing the
    /// `[B, H, Q, K]` attention matrix; instead saves a small
    /// `lse: [B, H, Q]` log-sum-exp tensor for the BW pass. Trailblazer
    /// constraints: `Br = Bc = 64`, `d_k = d_v ≤ 128`, optional causal
    /// mask, no explicit additive mask (use `SdpaPlan` for masked
    /// attention).
    FlashAttention = 3,
    /// KV-cache append — decoder-inference helper that writes
    /// newly-generated `K` / `V` slices into running cache buffers at
    /// per-sample offsets. Wired in Milestone 6.5 (FW only, no BW —
    /// inference-time op).
    KvCache = 4,
    /// Paged attention (vLLM-style) — reserved.
    PagedAttention = 5,
}

/// Indexing / scatter / gather op discriminant — Category L from the
/// comprehensive plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Indexing`. Phase 7 Milestone 7.3 wires:
/// - [`Self::Gather`] (FW + BW): `out[i] = src[index[i]]` along a dim.
/// - [`Self::ScatterAdd`]: `out[index[i]] += updates[i]` along a dim
///   (atomicAdd, dup-safe).
/// - [`Self::IndexSelect`] (FW + BW): `out[..., j, ...] = src[..., idx[j], ...]`
///   with a 1-D i32 idx tensor.
/// - [`Self::MaskedFill`] (FW + BW): `out[i] = mask[i] ? value : src[i]`.
/// - [`Self::OneHot`] (FW only — non-differentiable):
///   `out[..., c] = 1 if c == src[...] else 0`.
/// - [`Self::Nonzero`] (FW only): coordinates where input != 0,
///   returned as an `[k, rank]` i32 table plus a count.
///
/// Index dtype is `i32` only in the trailblazer (i64 deferred).
/// Out-of-bounds and negative indices are treated as no-ops (the kernel
/// skips them — PyTorch-style negative wrap-around is deferred).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum IndexingKind {
    /// `gather(src, dim, index)` — `out[..., j, ...] = src[..., index[..., j, ...], ...]`
    /// along the specified gather dimension. PyTorch `torch.gather`.
    Gather = 0,
    /// Gradient of [`Self::Gather`]: scatters `dout` into `dsrc` along
    /// the gather dim with atomicAdd (dup-safe). Different signature
    /// from [`Self::ScatterAdd`] because the dst is `dsrc` and the
    /// index pattern matches the FW gather coordinates exactly.
    GatherBackward = 1,
    /// `scatter_add(out, dim, index, updates)` —
    /// `out[..., index[..., j, ...], ...] += updates[..., j, ...]`
    /// (atomicAdd). PyTorch `torch.scatter_add_`.
    ScatterAdd = 2,
    /// `index_select(src, dim, idx)` —
    /// `out[..., j, ...] = src[..., idx[j], ...]` with a 1-D i32 idx
    /// tensor. Faster / simpler than `gather` when the index tensor
    /// is 1-D. PyTorch `torch.index_select`.
    IndexSelect = 3,
    /// Gradient of [`Self::IndexSelect`]: scatter-add `dout` into `dsrc`
    /// along `select_dim` using `idx` (atomicAdd).
    IndexSelectBackward = 4,
    /// `masked_fill(src, mask, value)` —
    /// `out[i] = mask[i] ? value : src[i]`. PyTorch
    /// `torch.Tensor.masked_fill`.
    MaskedFill = 5,
    /// Gradient of [`Self::MaskedFill`]: `dsrc[i] = mask[i] ? 0 : dout[i]`.
    /// `value` is a non-differentiable scalar.
    MaskedFillBackward = 6,
    /// `one_hot(src, num_classes)` —
    /// `out[indices..., c] = 1 if c == src[indices...] else 0`. Input
    /// dtype is i32 (class indices); output dtype is configurable.
    /// PyTorch `torch.nn.functional.one_hot`. Non-differentiable.
    OneHot = 7,
    /// `nonzero(x)` — coordinates where `x != 0`. Returns an
    /// `[k, rank]` i32 coordinate table plus a count. PyTorch
    /// `torch.nonzero`. Output ordering is NOT row-major (atomic-counter
    /// races); callers that need sorted output sort afterward.
    Nonzero = 8,
}

/// Segment / scatter-reduce op discriminant — Category S from the
/// comprehensive plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::SegmentOps`. Each variant maps to a
/// distinct kernel symbol — sorted and unsorted families live in the
/// same enum (different `op` slots) because the kernel implementation
/// differs (sorted = binary-search single-pass sweep; unsorted = atomic
/// scatter from the input side).
///
/// Phase 7 Milestone 7.6 wires:
/// - Sorted: [`Self::SegmentSum`], [`Self::SegmentMean`],
///   [`Self::SegmentMax`], [`Self::SegmentMin`], [`Self::SegmentProd`]
///   (FW). Sum / Mean carry a BW variant
///   ([`Self::SegmentSumBackward`], [`Self::SegmentMeanBackward`]).
/// - Unsorted: [`Self::UnsortedSegmentSum`],
///   [`Self::UnsortedSegmentMean`], [`Self::UnsortedSegmentMax`],
///   [`Self::UnsortedSegmentMin`] (FW). Sum / Mean carry a BW variant
///   ([`Self::UnsortedSegmentSumBackward`],
///   [`Self::UnsortedSegmentMeanBackward`]).
///
/// Max / Min / Prod BW is deferred — argmax tracking (Max / Min) and
/// numerically stable division-by-input (Prod) are out-of-scope for
/// the trailblazer. Unsorted Prod is also deferred (no native FP
/// atomicMul; would need a `atomicCAS` retry loop).
///
/// Dtype coverage: `f32, f64` (atomic-supported FP types). f16 / bf16
/// deferred — the kernels use `atomicAdd` / `atomicMax` / `atomicMin`
/// which are restricted to native-FP-atomic types.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum SegmentKind {
    /// `out[s, d] = Σ_{n : seg[n] == s} input[n, d]` — sorted segment
    /// IDs (monotonically non-decreasing). TF / JAX `segment_sum`.
    SegmentSum = 0,
    /// Gradient of [`Self::SegmentSum`]:
    /// `d_input[n, d] = d_output[seg[n], d]` (gather along seg ids).
    SegmentSumBackward = 1,
    /// `out[s, d] = mean_{n : seg[n] == s} input[n, d]` — sorted.
    SegmentMean = 2,
    /// Gradient of [`Self::SegmentMean`]:
    /// `d_input[n, d] = d_output[seg[n], d] / count[seg[n]]`.
    SegmentMeanBackward = 3,
    /// `out[s, d] = max_{n : seg[n] == s} input[n, d]` — sorted.
    /// FW-only; BW deferred (requires argmax tracking).
    SegmentMax = 4,
    /// `out[s, d] = min_{n : seg[n] == s} input[n, d]` — sorted.
    /// FW-only; BW deferred (requires argmin tracking).
    SegmentMin = 5,
    /// `out[s, d] = prod_{n : seg[n] == s} input[n, d]` — sorted.
    /// FW-only; BW deferred (requires numerically stable
    /// division-by-input or saved-running-product).
    SegmentProd = 6,
    /// `out[s, d] = Σ_{n : seg[n] == s} input[n, d]` — unsorted
    /// (seg IDs in any order). TF `unsorted_segment_sum`.
    UnsortedSegmentSum = 7,
    /// Gradient of [`Self::UnsortedSegmentSum`]:
    /// `d_input[n, d] = d_output[seg[n], d]`.
    UnsortedSegmentSumBackward = 8,
    /// `out[s, d] = mean_{n : seg[n] == s} input[n, d]` — unsorted.
    UnsortedSegmentMean = 9,
    /// Gradient of [`Self::UnsortedSegmentMean`]:
    /// `d_input[n, d] = d_output[seg[n], d] / count[seg[n]]`.
    UnsortedSegmentMeanBackward = 10,
    /// `out[s, d] = max_{n : seg[n] == s} input[n, d]` — unsorted.
    /// FW-only; BW deferred.
    UnsortedSegmentMax = 11,
    /// `out[s, d] = min_{n : seg[n] == s} input[n, d]` — unsorted.
    /// FW-only; BW deferred.
    UnsortedSegmentMin = 12,
}

/// Embedding-family op discriminant — Category M from the comprehensive
/// plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Embedding`. Phase 7 Milestone 7.5 wires:
/// - [`Self::Embedding`] (FW + BW): row-lookup
///   `out[i, :] = weight[indices[i], :]` with optional `padding_idx`
///   that emits an all-zero row at FW and skips accumulation at BW.
/// - [`Self::EmbeddingBagSum`] / [`Self::EmbeddingBagMean`] (FW + BW):
///   bag-reduced row lookup —
///   `out[b, :] = reduce(weight[indices[k], :] for k in offsets[b]..offsets[b+1])`.
///   Mode determines the reducer (sum / divide-by-bag-size).
///   `EmbeddingBagMax` is deferred (needs argmax tracking for BW).
///
/// Index dtype is `i32` only (i64 deferred). FW kernels emit
/// `f32, f64, f16, bf16` (pure copy / reduce); BW kernels emit `f32,
/// f64` (atomicAdd).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum EmbeddingKind {
    /// `embedding(weight, indices, padding_idx)` —
    /// `out[i, :] = weight[indices[i], :]`. PyTorch
    /// `torch.nn.functional.embedding`.
    Embedding = 0,
    /// Gradient of [`Self::Embedding`]:
    /// `dweight[indices[i], :] += dout[i, :]` (atomicAdd), skipping
    /// rows where `indices[i] == padding_idx`.
    EmbeddingBackward = 1,
    /// `embedding_bag(weight, indices, offsets, mode=Sum)`.
    /// PyTorch `torch.nn.functional.embedding_bag` with `mode='sum'`.
    EmbeddingBagSum = 2,
    /// `embedding_bag(weight, indices, offsets, mode=Mean)`.
    /// PyTorch `torch.nn.functional.embedding_bag` with `mode='mean'`.
    EmbeddingBagMean = 3,
    /// Gradient of `embedding_bag` (Sum-mode):
    /// `dweight[indices[k], :] += dout[b, :]` for k in bag b (atomicAdd).
    EmbeddingBagSumBackward = 4,
    /// Gradient of `embedding_bag` (Mean-mode):
    /// `dweight[indices[k], :] += dout[b, :] / bag_size(b)` (atomicAdd).
    EmbeddingBagMeanBackward = 5,
    /// `embedding_bag(weight, indices, offsets, mode=Max)` — reserved.
    /// Max-mode requires argmax tracking on FW (the per-feature index
    /// of the contributing row) so the BW can scatter into just that
    /// row — different plan shape; deferred.
    EmbeddingBagMax = 6,
    /// Gradient of `embedding_bag` (Max-mode) — reserved.
    EmbeddingBagMaxBackward = 7,
}

/// Quantization op discriminant — Category P from the comprehensive plan.
///
/// Stored as `u16` in [`crate::KernelSku::op`] when
/// `category == OpCategory::Quantization`. Phase 8 Milestone 8.1 wires the
/// trailblazer set: per-tensor + per-channel quantize / dequantize plus
/// fake_quantize (round-trip in FP space). All entries support FW + BW
/// where applicable (FW-only for kinds that have no meaningful gradient).
///
/// **Trailblazer dtype scope.** Input FP × output int:
/// - Input FP: `f32, f64, f16, bf16`.
/// - Output int: `s8, u8`. Sub-byte packed types (`s4`, `u4`) are deferred.
/// - `scale` matches the input FP dtype; `zero_point` is always `i32`
///   (wide enough for any int output qmin/qmax range).
///
/// **Backward convention (Straight-Through Estimator).** The BW of
/// `quantize` and `fake_quantize` uses STE — the gradient passes through
/// (with a `1/scale` factor for `quantize`, no factor for `fake_quantize`)
/// where the rounded result was in-range `[qmin, qmax]`, zero elsewhere.
/// The "in-range mask" is **recomputed in BW from the saved `input`
/// tensor** rather than saved as a separate FW output — this matches
/// PyTorch's internal FakeQuantize and keeps the FW signature clean.
/// Callers must therefore retain the original input tensor for the BW
/// pass (which they would do anyway for autograd).
///
/// Future milestones extend this enum with `PerToken` / `PerGroup` /
/// `DynamicRange` variants — discriminant gaps are intentionally left
/// for those.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum QuantizeKind {
    /// `quantize_per_tensor(x, scale, zero_point)` —
    /// `q = clamp(round(x / scale) + zero_point, qmin, qmax)`.
    /// One scalar `scale` (FP) and `zero_point` (i32) for the whole
    /// tensor. PyTorch `torch.quantize_per_tensor`.
    PerTensor = 0,
    /// Gradient of [`Self::PerTensor`] via STE:
    /// `dx = (dy / scale) * in_range_mask`, where the mask is
    /// `qmin <= round(x/scale) + zp <= qmax`.
    PerTensorBackward = 1,
    /// `dequantize_per_tensor(q, scale, zero_point)` —
    /// `x = scale * (q - zero_point)`. Linear; exactly invertible up to
    /// rounding. PyTorch `torch.Tensor.dequantize`.
    DequantizePerTensor = 2,
    /// Gradient of [`Self::DequantizePerTensor`]: `dq = dy * scale`
    /// (linear identity scaled).
    DequantizePerTensorBackward = 3,
    /// `quantize_per_channel(x, scale[C], zero_point[C], axis)` — same
    /// math as [`Self::PerTensor`] but with one `scale[c]` /
    /// `zero_point[c]` pair per slice along `axis`. PyTorch
    /// `torch.quantize_per_channel`.
    PerChannel = 4,
    /// Gradient of [`Self::PerChannel`] via STE:
    /// `dx = (dy / scale[c]) * in_range_mask[c]`.
    PerChannelBackward = 5,
    /// `dequantize_per_channel(q, scale[C], zero_point[C], axis)` —
    /// `x = scale[c] * (q - zero_point[c])`.
    DequantizePerChannel = 6,
    /// Gradient of [`Self::DequantizePerChannel`]:
    /// `dq = dy * scale[c]`.
    DequantizePerChannelBackward = 7,
    /// `fake_quantize_per_tensor(x, scale, zero_point)` —
    /// `y = scale * (clamp(round(x/scale)+zp, qmin, qmax) - zp)`. The
    /// roundtrip quantize-then-dequantize in FP space; produces a lossy
    /// FP output. PyTorch
    /// `torch.fake_quantize_per_tensor_affine`.
    FakeQuantize = 8,
    /// Gradient of [`Self::FakeQuantize`] via STE:
    /// `dx = dy * in_range_mask`. **No `1/scale` factor** — the
    /// dequant-side multiplication by `scale` in FW cancels the
    /// `1/scale` from STE.
    FakeQuantizeBackward = 9,
    /// Reserved — `quantize_per_token` (per-row dynamic-range
    /// quantization used by activation quantization).
    PerToken = 16,
    /// Reserved — gradient of [`Self::PerToken`].
    PerTokenBackward = 17,
    /// Reserved — `quantize_per_group` (block-wise quantization used by
    /// GPTQ / AWQ / GGML).
    PerGroup = 18,
    /// Reserved — gradient of [`Self::PerGroup`].
    PerGroupBackward = 19,
    /// Reserved — `dynamic_range_quantize` (post-training dynamic
    /// quantization).
    DynamicRange = 20,
    // ---- Milestone 8.2 completion — per-token / per-group dequant
    //      + backwards (FW PerToken / PerGroup discriminants were
    //      reserved above at 16-19) ----
    /// `dequantize_per_token(q, scale[N], zero_point[N])` —
    /// `y[n, d] = scale[n] * (q[n, d] - zp[n])`. Per-row inverse of
    /// [`Self::PerToken`].
    DequantizePerToken = 21,
    /// Gradient of [`Self::DequantizePerToken`]:
    /// `dq = dy * scale[n]` (straight-through).
    DequantizePerTokenBackward = 22,
    /// `dequantize_per_group(q, scale[outer, num_groups],
    /// zero_point[outer, num_groups])` — per-group inverse of
    /// [`Self::PerGroup`].
    DequantizePerGroup = 23,
    /// Gradient of [`Self::DequantizePerGroup`]:
    /// `dq[i, j] = dy[i, j] * scale[i, j/g]` (straight-through).
    DequantizePerGroupBackward = 24,
    // ---- Milestone 8.3 — composing quantization ops ----
    /// `quantized_linear(activation_fp, weight_q_s8, weight_scale,
    /// bias?)` — W8A8 fused quantized matmul. Pipeline: dynamic-range
    /// per-token quantize the activation → int8 GEMM with int32
    /// accumulator → dequantize via per-row `scale_a` and per-channel
    /// `scale_w`. The canonical inference-time LLM matmul recipe
    /// (e.g. SmoothQuant, AWQ-runtime); FP activation in, FP output out,
    /// int8 storage only on the GEMM. Backward isn't shipped — this op
    /// is inference-only by convention.
    QuantizedLinear = 25,
    // ---- Milestone 8.4 — GGUF block-format quant family ----
    /// `gguf_dequantize(packed_bytes) -> fp_tensor` — unpack a
    /// GGUF-packed weight buffer (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 +
    /// Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K) into a dense FP
    /// tensor. The block format is carried out-of-band on the plan
    /// descriptor (see [`GgufBlockFormat`]); the kernel surface
    /// fans out across block formats but the enum value is the same.
    /// Inference-only by convention (BW not shipped).
    GgufDequantize = 26,
    /// `gguf_mmvq(packed_weight, fp_activation) -> fp_output` —
    /// fused dequant + matrix-vector multiply: the inference-time
    /// "decode-step" matmul used by llama.cpp on GGUF weights.
    /// FP activation in (f32 today), FP output out. Inference-only
    /// (BW not shipped).
    GgufMmvq = 27,
}

/// GGUF block-format selector for [`QuantizeKind::GgufDequantize`] /
/// [`QuantizeKind::GgufMmvq`] plans. Mirrors the discriminants used by
/// llama.cpp / `ggml` so a descriptor can be round-tripped to a GGUF
/// file header without translation.
///
/// Block sizes:
///   * Type-0/1 variants (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`)
///     pack 32 quantized values per block plus a shared FP scale
///     (+ min for the `_1` variants).
///   * k-quants variants (`Q2_K` ... `Q8_K`) pack 256 values per
///     super-block with a multi-level scale hierarchy
///     (quantized sub-block scales + FP super-block scale).
///
/// Discriminant values match the `GGML_TYPE_*` enum in upstream
/// `ggml.h`, ensuring binary compatibility with GGUF file headers.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum GgufBlockFormat {
    /// 4-bit, 32-element block, single FP scale. `block_q4_0`.
    Q4_0 = 2,
    /// 4-bit, 32-element block, FP scale + FP min. `block_q4_1`.
    Q4_1 = 3,
    /// 5-bit, 32-element block, single FP scale. `block_q5_0`.
    Q5_0 = 6,
    /// 5-bit, 32-element block, FP scale + FP min. `block_q5_1`.
    Q5_1 = 7,
    /// 8-bit, 32-element block, single FP scale. `block_q8_0`.
    Q8_0 = 8,
    /// 2.5-bit (effective), 256-element super-block. `block_q2_K`.
    Q2K = 10,
    /// 3.4-bit (effective), 256-element super-block. `block_q3_K`.
    Q3K = 11,
    /// 4.5-bit (effective), 256-element super-block. `block_q4_K`.
    Q4K = 12,
    /// 5.5-bit (effective), 256-element super-block. `block_q5_K`.
    Q5K = 13,
    /// 6.6-bit (effective), 256-element super-block. `block_q6_K`.
    Q6K = 14,
    /// 8-bit, 256-element super-block (CPU-side intermediate).
    /// `block_q8_K`. Dequant supported; MMVQ NOT supported (matches
    /// llama.cpp — no upstream MMVQ specialization).
    Q8K = 15,
}

impl GgufBlockFormat {
    /// Number of FP elements per packed block.
    #[inline]
    pub const fn block_size(self) -> usize {
        match self {
            GgufBlockFormat::Q4_0
            | GgufBlockFormat::Q4_1
            | GgufBlockFormat::Q5_0
            | GgufBlockFormat::Q5_1
            | GgufBlockFormat::Q8_0 => 32,
            _ => 256,
        }
    }

    /// Number of bytes per packed block. Matches `sizeof(block_q*)`
    /// from `ggml.h`. Used by the Rust plan layer to size the input
    /// weight buffer.
    #[inline]
    pub const fn type_size(self) -> usize {
        match self {
            // 2 (fp16 d) + 16 (qs[16])
            GgufBlockFormat::Q4_0 => 18,
            // 2*2 (half2 dm) + 16 (qs[16])
            GgufBlockFormat::Q4_1 => 20,
            // 2 (fp16 d) + 4 (qh) + 16 (qs[16])
            GgufBlockFormat::Q5_0 => 22,
            // 2*2 (half2 dm) + 4 (qh) + 16 (qs[16])
            GgufBlockFormat::Q5_1 => 24,
            // 2 (fp16 d) + 32 (qs[32])
            GgufBlockFormat::Q8_0 => 34,
            // 2*2 (half2 dm) + QK_K/16 (16 scales) + QK_K/4 (64 qs) = 4+16+64
            GgufBlockFormat::Q2K => 84,
            // hmask(32) + qs(64) + scales(12) + d(2)
            GgufBlockFormat::Q3K => 110,
            // dm(4) + scales(12) + qs(128)
            GgufBlockFormat::Q4K => 144,
            // dm(4) + scales(12) + qh(32) + qs(128)
            GgufBlockFormat::Q5K => 176,
            // ql(128) + qh(64) + scales(16) + d(2)
            GgufBlockFormat::Q6K => 210,
            // d(4) + qs(256) + bsums(32)
            GgufBlockFormat::Q8K => 292,
        }
    }

    /// `true` for the type-0/1 family (32-element blocks); `false`
    /// for the k-quants family (256-element super-blocks).
    #[inline]
    pub const fn is_type_01(self) -> bool {
        matches!(
            self,
            GgufBlockFormat::Q4_0
                | GgufBlockFormat::Q4_1
                | GgufBlockFormat::Q5_0
                | GgufBlockFormat::Q5_1
                | GgufBlockFormat::Q8_0
        )
    }

    /// `true` if MMVQ (fused dequant + matvec) is supported for this
    /// block format. `false` for [`GgufBlockFormat::Q8K`] only —
    /// llama.cpp / Fuel reserve Q8_K as a CPU-side intermediate.
    #[inline]
    pub const fn has_mmvq(self) -> bool {
        !matches!(self, GgufBlockFormat::Q8K)
    }
}


/// Mixture-of-Experts (MoE) variant selector — used as the `op`
/// discriminant for kernel SKUs whose [`crate::OpCategory`] is
/// [`crate::OpCategory::Moe`]. Phase 8 Milestone 8.5 wires the three
/// fused per-token-dispatch + expert-matmul + accumulate kernels.
///
/// MoE forward pass shape:
///   * Input activations `[T, D_model]`.
///   * Per-token top-k expert indices `[T, top_k]` (i32).
///   * Per-token top-k expert weights `[T, top_k]` (FP).
///   * Per-expert weight matrices `[num_experts, D_model, D_expert]`
///     (dtype depends on the variant: FP for `Wmma`, GGUF-packed bytes
///     for `ScalarGguf` / `WmmaGguf`).
///   * Output `[T, D_model]` (after expert mixing).
///
/// All three variants are inference-only by convention; backward
/// passes are not shipped (MoE training uses higher-level autograd
/// surfaces that compose the per-expert FFN ops manually).
///
/// Lineage: vendored from `attention.rs` via `fuel-cuda-kernels`. See
/// `crates/baracuda-kernels-sys/LICENSE-thirdparty.md` for the full
/// attribution chain.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum MoeKind {
    /// Scalar dispatch path operating on GGUF-quantized expert weights
    /// staged through a q8_1 intermediate (FP32 activations in, FP32
    /// output out). No tensor cores. Used as a portability fallback
    /// and as the slower-but-simpler reference for the WMMA + GGUF
    /// hot path. Block formats: `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`,
    /// `Q5_K`, `Q6_K` (matches Fuel's `moe_gemm_gguf` switch).
    ScalarGguf = 0,
    /// WMMA tensor-core path operating on dense FP expert weights
    /// (f16 / bf16). The FP MoE hot path used when full-precision
    /// expert weights are available — typically training-time or
    /// FP-deployment inference. sm_70+ required.
    Wmma = 1,
    /// Combined WMMA tensor-core + GGUF-quantized weight path. The
    /// dispatcher dequantizes one GGUF block per N-row into shared
    /// memory, then issues a 16×16×16 WMMA mma.sync against the
    /// dense activation tile. The production hot path for quantized
    /// LLM inference. Activation dtype: f16 / bf16. Weight block
    /// formats: same set as [`Self::ScalarGguf`]. sm_70+ required.
    WmmaGguf = 2,
}
