//! Structure-class keying for AOT kernel specialization.
//!
//! A [`StructureKey`] is the canonical identity of an *input/output layout
//! class* — the join token shared across three consumers (per the Baracuda↔Fuel
//! boundary contract in `docs/design/kernel-specialization.md`):
//!
//! 1. **runtime dispatch** — pick the specialized kernel registered for a key;
//! 2. **FKC predicate generation** — a generated kernel contract's admissibility
//!    predicate *is* its structure key, so the planner's miss signal is honest;
//! 3. **telemetry tagging** — Fuel tags each dispatch/miss record with the
//!    key's [`StructureKey::to_token`] string.
//!
//! The key is computed by [`structure_key`] from a slice of [`OperandDesc`] —
//! the **minimal operand-description projection** the key reads. Fuel constructs
//! each `OperandDesc` from its `FdxOperandDesc`; Baracuda callers use
//! [`OperandDesc::from_tensor_ref`]. Neither side reimplements the key — both
//! call this one function, so the build matrix and the runtime lookup speak the
//! same language by construction.
//!
//! # Scope (v1)
//!
//! This first cut targets the elementwise specialization pilot. It derives the
//! per-operand and whole-key predicate axes (contiguity, broadcast, flip,
//! vector width, inner-extent divisibility, index width, effective rank, work
//! class) from raw shape/stride/alignment. The following are deliberately left
//! for follow-ups and are called out at their use sites:
//!
//! - **Reduction keying** ([`StructureKey::reduce_axes`] is always empty here).
//! - **Quant-aware keying** ([`OperandDesc::quant`] is carried so Fuel can bind
//!   the interface, but v1 does not fold it into the key — quant operands are
//!   out of scope until the quant pilot).
//! - **Full canonicalization** (size-1 squeeze is applied; adjacent-contiguous
//!   merge feeds [`StructureKey::eff_rank`]; legality-aware axis *reordering*
//!   to maximize cell-merging is a follow-up).

use crate::{ArchSku, ElementKind, KernelDtype, OpCategory, TensorRef};
use baracuda_types::DeviceRepr;

/// Maximum tensor rank the structure key supports — matches the rank ceiling
/// of every strided baracuda kernel (`baracuda::coord` `MAX_RANK`).
pub const MAX_RANK: usize = 8;

/// Maximum number of operands (inputs + output) a single key describes.
pub const MAX_OPERANDS: usize = 8;

/// Structure-key schema version. Bumped when a predicate axis is added or
/// altered; old-version tokens stay distinguishable by this field.
pub const STRUCTURE_KEY_VERSION: u16 = 1;

// ===========================================================================
// Predicate axes
// ===========================================================================

/// Width of the integer arithmetic used for element offsets.
///
/// `int32` offset math is materially cheaper (fewer registers, tighter loops);
/// the boundary is 2³¹ elements and is architecture-independent.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum IdxWidth {
    /// All element offsets fit in `i32` (`< 2³¹`).
    Idx32,
    /// At least one offset needs `i64`.
    Idx64,
}

/// Per-operand memory-layout class — the single most codegen-relevant axis.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum Contiguity {
    /// Row-major packed: linear addressing, no coordinate unravel.
    #[default]
    Contig,
    /// Innermost (fastest-varying) axis has stride 1, outer axes strided:
    /// the inner loop vectorizes even though the outer walk is strided.
    InnerContig,
    /// Arbitrary strides — full coordinate unravel per element.
    Strided,
    /// At least one axis has stride 0 — the load can be hoisted out of the loop.
    Broadcast,
}

/// Achievable vectorized access width, derived from base-pointer alignment,
/// innermost stride/extent, and dtype size. `ld.128`/`st.128` (V4 for f32, V8
/// for f16) versus scalar is 2–4× on bandwidth-bound ops.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum VecWidth {
    /// One element per access.
    #[default]
    Scalar,
    /// Two elements per access.
    V2,
    /// Four elements per access.
    V4,
    /// Eight elements per access.
    V8,
}

/// Divisibility bucket of an operand's innermost extent — drives remainder-loop
/// elimination and full unrolling. The ladder is `Div16 ⊐ Div8 ⊐ Div4 ⊐ Div2 ⊐ Any`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum DivBucket {
    /// Inner extent divisible by 16.
    Div16,
    /// Divisible by 8 (but not 16).
    Div8,
    /// Divisible by 4 (but not 8).
    Div4,
    /// Divisible by 2 (but not 4).
    Div2,
    /// No useful power-of-two divisor.
    #[default]
    Any,
}

/// Total-work size class — replaces a "stepped max-dim" axis. Tiny work wants a
/// single-warp or single-block kernel (no grid-stride, no millions of idle
/// threads); everything larger is one grid-stride kernel.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum WorkClass {
    /// Fits in one warp (≤ 32 elements).
    OneWarp,
    /// Fits in one block (≤ 1024 elements).
    OneBlock,
    /// Larger — a grid-stride loop.
    GridStride,
}

/// A bitmask over canonical axes (bit `i` ⇒ axis `i`). Used for the broadcast
/// axis set and the reduction axis set; rank is capped at [`MAX_RANK`] so a
/// single byte suffices.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct AxisMask(pub u8);

impl AxisMask {
    /// The empty mask (no axes set).
    pub const EMPTY: AxisMask = AxisMask(0);

    /// `true` if `axis` is set.
    #[inline]
    #[must_use]
    pub const fn is_set(self, axis: u8) -> bool {
        axis < 8 && (self.0 >> axis) & 1 == 1
    }

    /// Set `axis` (no-op if `axis >= 8`).
    #[inline]
    pub fn set(&mut self, axis: u8) {
        if axis < 8 {
            self.0 |= 1 << axis;
        }
    }

    /// `true` if no axes are set.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

// ===========================================================================
// The key
// ===========================================================================

/// Per-operand predicate sub-key. One of these is carried for every input and
/// the output.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct OperandKey {
    /// Memory-layout class.
    pub contig: Contiguity,
    /// Which axes broadcast (stride 0).
    pub bcast: AxisMask,
    /// Achievable vectorized access width.
    pub vec_width: VecWidth,
    /// Innermost-extent divisibility bucket.
    pub inner_div: DivBucket,
    /// `true` if any axis has a negative stride (a flipped / reversed view).
    pub flipped: bool,
}

/// The canonical identity of an input/output layout class.
///
/// Construct via [`structure_key`]. Two layouts that canonicalize to the same
/// `StructureKey` are served by the same specialized kernel. Heap-free and
/// `Copy` so it can be hashed into a dispatch table or an autotuner cache
/// directly; [`StructureKey::to_token`] gives the stable string form used on
/// the telemetry wire.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct StructureKey {
    /// Schema version ([`STRUCTURE_KEY_VERSION`]).
    pub version: u16,
    /// The op taxonomy this key was computed for (drives canonicalization
    /// legality).
    pub op: OpCategory,
    /// Primary dtype (operand 0). Mixed-dtype ops fold per-operand dtype in a
    /// follow-up; v1 assumes a uniform operand dtype.
    pub dtype: ElementKind,
    /// Compute capability the specialized kernel targets.
    pub arch: ArchSku,
    /// Offset-arithmetic width.
    pub idx: IdxWidth,
    /// Total-work size class.
    pub work: WorkClass,
    /// Raw iteration rank — the maximum operand rank (for elementwise ops the
    /// operands are rank-aligned, broadcasting via stride 0, so this is the
    /// shared logical rank the strided schedules unravel over). Size-1 squeeze
    /// and contiguous-axis collapse are deferred optimizations.
    pub rank: u8,
    /// Number of valid entries in [`StructureKey::operands`].
    pub n_operands: u8,
    /// Per-operand sub-keys; only `operands[0..n_operands]` are meaningful, the
    /// tail is [`OperandKey::default`] so equal keys hash equal.
    pub operands: [OperandKey; MAX_OPERANDS],
    /// Reduced-axis set for reduction-class ops; [`AxisMask::EMPTY`] otherwise
    /// (always empty in v1).
    pub reduce_axes: AxisMask,
}

// ===========================================================================
// Operand description (the minimal projection the key reads)
// ===========================================================================

/// Quant family, mirroring the FDX `FDXQuant.family` codes (FDX is the
/// normative owner; this is the Baracuda-side projection).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum QuantFamily {
    /// GGUF block layout, scale baked inline.
    Ggml,
    /// OCP microscaling (per-block F8E8M0 scale).
    Mx,
    /// Dynamic per-tensor/token/channel affine integer.
    AffineInt,
    /// Dynamic per-tensor/token/channel affine float.
    AffineFloat,
    /// NF4/QLoRA — low-bit data plus a separate per-block absmax scale.
    AffineBlock,
}

/// Where a quant scale lives, mirroring FDX `FDXScalePlacement`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ScalePlacement {
    /// Scale baked inline with the data block.
    Inline,
    /// Scale in a separate buffer.
    SeparateBuffer,
    /// Scale broadcast per axis.
    BroadcastPerAxis,
}

/// Quantization facts for a quant operand. Carried so Fuel can bind the
/// interface; **v1 [`structure_key`] does not yet key on these** (quant
/// operands are out of scope until the quant pilot).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct QuantFacts {
    /// Quant family.
    pub family: QuantFamily,
    /// Sub-byte bit width (e.g. 4 for Q4), or 0 if not sub-byte.
    pub sub_byte_bits: u8,
    /// Block extent in logical elements, or 0 if not block-quantized.
    pub block_elems: u16,
    /// Scale placement.
    pub scale: ScalePlacement,
}

/// Kind of a symbolic (live-vs-capacity) extent, mirroring FDX `FDXExtent.kind`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SymKind {
    /// A single dynamic scalar bound.
    Scalar,
    /// A `[min, capacity]` range.
    Range,
    /// An affine form `c0 + Σ cᵢ·symᵢ` (e.g. `k_len = cached + new`).
    Affine,
}

/// A symbolic extent on one axis. The axis's *capacity* is its
/// [`OperandDesc::shape`] entry (which keys strides and index width); this flags
/// that the live length is dynamic, which is itself a specialization axis for
/// attention-class ops (static `k_len == capacity` vs dynamic).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SymExtent {
    /// Which axis is symbolic.
    pub axis: u8,
    /// The kind of symbolic bound.
    pub kind: SymKind,
}

/// The minimal per-operand description [`structure_key`] reads.
///
/// Owning and `Copy` (inline `[i64; MAX_RANK]` arrays, no lifetimes) so both
/// Fuel (from `FdxOperandDesc`) and Baracuda (from [`TensorRef`]) construct it
/// by value. Only `shape[0..rank]` / `strides[0..rank]` are meaningful.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct OperandDesc {
    /// Tensor rank (`≤ MAX_RANK`).
    pub rank: u8,
    /// Per-axis extents (capacity for symbolic axes).
    pub shape: [i64; MAX_RANK],
    /// Per-axis signed element strides. `0` = broadcast, `< 0` = flipped.
    pub strides: [i64; MAX_RANK],
    /// Base/logical operand dtype.
    pub dtype: ElementKind,
    /// Base-pointer alignment in bytes (drives vector width).
    pub align_bytes: u32,
    /// Quantization facts, if this is a quant operand.
    pub quant: Option<QuantFacts>,
    /// Symbolic-extent facts, if any axis is live-vs-capacity.
    pub symbolic: Option<SymExtent>,
}

impl OperandDesc {
    /// Build a plain (non-quant, non-symbolic) operand description from `rank`
    /// extents, strides, dtype, and base-pointer alignment.
    ///
    /// # Panics
    /// Panics if `rank > MAX_RANK`.
    #[must_use]
    pub fn new(
        rank: usize,
        shape: &[i64],
        strides: &[i64],
        dtype: ElementKind,
        align_bytes: u32,
    ) -> Self {
        assert!(rank <= MAX_RANK, "rank {rank} exceeds MAX_RANK {MAX_RANK}");
        let mut s = [0i64; MAX_RANK];
        let mut st = [0i64; MAX_RANK];
        s[..rank].copy_from_slice(&shape[..rank]);
        st[..rank].copy_from_slice(&strides[..rank]);
        Self {
            rank: rank as u8,
            shape: s,
            strides: st,
            dtype,
            align_bytes,
            quant: None,
            symbolic: None,
        }
    }

    /// Build an operand description from a borrowed device tensor view.
    ///
    /// `align_bytes` is supplied by the caller (it knows its allocation /
    /// view alignment — a base `cudaMalloc` is 256-byte aligned, but a sub-view
    /// may be less). dtype is taken from `T` via [`KernelDtype::KIND`].
    ///
    /// # Panics
    /// Panics if `N > MAX_RANK`.
    #[must_use]
    pub fn from_tensor_ref<T, const N: usize>(
        view: &TensorRef<'_, T, N>,
        align_bytes: u32,
    ) -> Self
    where
        T: KernelDtype + DeviceRepr + Copy + 'static,
    {
        assert!(N <= MAX_RANK, "rank {N} exceeds MAX_RANK {MAX_RANK}");
        let mut shape = [0i64; MAX_RANK];
        let mut strides = [0i64; MAX_RANK];
        for d in 0..N {
            shape[d] = i64::from(view.shape[d]);
            strides[d] = view.stride[d];
        }
        Self {
            rank: N as u8,
            shape,
            strides,
            dtype: T::KIND,
            align_bytes,
            quant: None,
            symbolic: None,
        }
    }
}

// ===========================================================================
// Derivation
// ===========================================================================

/// Compute the [`StructureKey`] for an op over its operands on a target arch.
///
/// `operands` is inputs followed by the output; the first operand is treated as
/// the primary (it sets [`StructureKey::dtype`], the work class, and the
/// effective rank). An empty slice yields a rank-0 scalar key.
///
/// This is the single canonical key function — Fuel calls it rather than
/// reimplementing the derivation, so telemetry and the build matrix join on the
/// same token.
#[must_use]
pub fn structure_key(op: OpCategory, operands: &[OperandDesc], arch: ArchSku) -> StructureKey {
    let n = operands.len().min(MAX_OPERANDS);
    let mut keys = [OperandKey::default(); MAX_OPERANDS];
    let mut max_off: i64 = 0;
    for (slot, od) in keys.iter_mut().zip(operands.iter()).take(n) {
        *slot = derive_operand_key(od);
        max_off = max_off.max(max_touched_offset(od));
    }

    let idx = if max_off >= (1i64 << 31) {
        IdxWidth::Idx64
    } else {
        IdxWidth::Idx32
    };

    let (dtype, work) = match operands.first() {
        Some(p) => (p.dtype, work_class(p)),
        None => (ElementKind::F32, WorkClass::OneWarp),
    };
    // Raw iteration rank = the widest operand rank (output rank for elementwise).
    let rank = operands.iter().map(|o| o.rank).max().unwrap_or(0);

    StructureKey {
        version: STRUCTURE_KEY_VERSION,
        op,
        dtype,
        arch,
        idx,
        work,
        rank,
        n_operands: n as u8,
        operands: keys,
        reduce_axes: AxisMask::EMPTY, // reduction keying: follow-up
    }
}

/// Innermost non-unit axis of an operand, or `None` if the operand is all
/// size-≤1 axes (a scalar).
fn inner_axis(od: &OperandDesc) -> Option<usize> {
    (0..od.rank as usize).rev().find(|&d| od.shape[d] > 1)
}

fn derive_operand_key(od: &OperandDesc) -> OperandKey {
    let rank = od.rank as usize;

    // Broadcast mask: extent-> 1 axes with stride 0.
    let mut bcast = AxisMask::EMPTY;
    let mut flipped = false;
    for d in 0..rank {
        if od.shape[d] > 1 && od.strides[d] == 0 {
            bcast.set(d as u8);
        }
        if od.strides[d] < 0 {
            flipped = true;
        }
    }

    let inner = inner_axis(od);
    let contig = classify_contiguity(od, bcast, inner);
    let vec_width = classify_vec_width(od, inner, bcast);
    let inner_div = match inner {
        Some(d) => div_bucket(od.shape[d]),
        None => DivBucket::Any,
    };

    OperandKey {
        contig,
        bcast,
        vec_width,
        inner_div,
        flipped,
    }
}

fn classify_contiguity(od: &OperandDesc, bcast: AxisMask, inner: Option<usize>) -> Contiguity {
    if !bcast.is_empty() {
        return Contiguity::Broadcast;
    }
    let Some(inner) = inner else {
        return Contiguity::Contig; // scalar
    };
    let rank = od.rank as usize;

    // Expected row-major contiguous |stride| per axis (over non-unit axes).
    let mut acc: i64 = 1;
    let mut all_match = true;
    for d in (0..rank).rev() {
        if od.shape[d] <= 1 {
            continue;
        }
        if od.strides[d].abs() != acc {
            all_match = false;
        }
        acc = acc.saturating_mul(od.shape[d]);
    }
    if all_match {
        Contiguity::Contig
    } else if od.strides[inner].abs() == 1 {
        Contiguity::InnerContig
    } else {
        Contiguity::Strided
    }
}

fn classify_vec_width(od: &OperandDesc, inner: Option<usize>, bcast: AxisMask) -> VecWidth {
    let (Some(inner), Some(dsz)) = (inner, dtype_size_bytes(od.dtype)) else {
        return VecWidth::Scalar;
    };
    // Only forward unit-stride contiguous inner runs vectorize in v1.
    if od.strides[inner] != 1 || !bcast.is_empty() {
        return VecWidth::Scalar;
    }
    let ext = od.shape[inner].max(0) as u64;
    let align = u64::from(od.align_bytes);
    let dsz = u64::from(dsz);
    for &v in &[8u64, 4, 2] {
        let vbytes = v * dsz;
        if vbytes <= 16 && align % vbytes == 0 && ext % v == 0 {
            return match v {
                8 => VecWidth::V8,
                4 => VecWidth::V4,
                _ => VecWidth::V2,
            };
        }
    }
    VecWidth::Scalar
}

fn div_bucket(extent: i64) -> DivBucket {
    let e = extent.max(0);
    if e % 16 == 0 {
        DivBucket::Div16
    } else if e % 8 == 0 {
        DivBucket::Div8
    } else if e % 4 == 0 {
        DivBucket::Div4
    } else if e % 2 == 0 {
        DivBucket::Div2
    } else {
        DivBucket::Any
    }
}

/// Largest linear element offset the operand can touch (`Σ |strideₐ|·(extₐ−1)`),
/// used to pick the index width.
fn max_touched_offset(od: &OperandDesc) -> i64 {
    let mut off: i64 = 0;
    for d in 0..od.rank as usize {
        let span = od.strides[d]
            .saturating_abs()
            .saturating_mul((od.shape[d] - 1).max(0));
        off = off.saturating_add(span);
    }
    off
}

fn work_class(od: &OperandDesc) -> WorkClass {
    let mut numel: i64 = 1;
    for d in 0..od.rank as usize {
        numel = numel.saturating_mul(od.shape[d].max(0));
    }
    if numel <= 32 {
        WorkClass::OneWarp
    } else if numel <= 1024 {
        WorkClass::OneBlock
    } else {
        WorkClass::GridStride
    }
}

/// Byte size of a byte-addressable dtype, or `None` for sub-byte dtypes (which
/// are treated as non-vectorizable in v1).
fn dtype_size_bytes(dt: ElementKind) -> Option<u32> {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    Some(match dt {
        S8 | U8 | Bool | Fp8E4M3 | Fp8E5M2 => 1,
        F16 | Bf16 => 2,
        F32 | F32Strict | I32 => 4,
        F64 | I64 | Complex32 => 8,
        Complex64 => 16,
        S4 | U4 | Bin => return None,
    })
}

// ===========================================================================
// Token codec
// ===========================================================================

impl StructureKey {
    /// Encode as the stable, lossless string token carried on the telemetry
    /// wire. Round-trips through [`StructureKey::from_token`].
    ///
    /// Form: `sk<ver>|<op>|<dtype>|<arch>|<idx>|<work>|r<rank>|<op0>;…|<reduce>`
    /// where each operand is `<contig>/<bcasthex>/<vec>/<div>/<flip>`.
    #[must_use]
    pub fn to_token(&self) -> String {
        let mut ops = String::new();
        for (i, o) in self.operands.iter().take(self.n_operands as usize).enumerate() {
            if i > 0 {
                ops.push(';');
            }
            ops.push_str(&format!(
                "{}/{:02x}/{}/{}/{}",
                contig_code(o.contig),
                o.bcast.0,
                vec_code(o.vec_width),
                div_code(o.inner_div),
                if o.flipped { 'r' } else { 'f' },
            ));
        }
        let reduce = if self.reduce_axes.is_empty() {
            "-".to_string()
        } else {
            format!("x{:02x}", self.reduce_axes.0)
        };
        format!(
            "sk{}|{}|{}|{}|{}|{}|r{}|{}|{}",
            self.version,
            op_code(self.op),
            dtype_code(self.dtype),
            arch_code(self.arch),
            idx_code(self.idx),
            work_code(self.work),
            self.rank,
            ops,
            reduce,
        )
    }

    /// Parse a token produced by [`StructureKey::to_token`]. Returns `None` on
    /// any malformed field or an unknown op short-code (a future op category
    /// with no token code assigned).
    #[must_use]
    pub fn from_token(token: &str) -> Option<StructureKey> {
        let parts: Vec<&str> = token.split('|').collect();
        if parts.len() != 9 {
            return None;
        }
        let version: u16 = parts[0].strip_prefix("sk")?.parse().ok()?;
        let op = op_from_code(parts[1])?;
        let dtype = dtype_from_code(parts[2])?;
        let arch = arch_from_code(parts[3])?;
        let idx = match parts[4] {
            "i32" => IdxWidth::Idx32,
            "i64" => IdxWidth::Idx64,
            _ => return None,
        };
        let work = match parts[5] {
            "warp" => WorkClass::OneWarp,
            "block" => WorkClass::OneBlock,
            "grid" => WorkClass::GridStride,
            _ => return None,
        };
        let rank: u8 = parts[6].strip_prefix('r')?.parse().ok()?;

        let mut operands = [OperandKey::default(); MAX_OPERANDS];
        let mut n_operands = 0u8;
        if !parts[7].is_empty() {
            for (slot, field) in operands.iter_mut().zip(parts[7].split(';')) {
                *slot = parse_operand(field)?;
                n_operands += 1;
            }
        }

        let reduce_axes = match parts[8] {
            "-" => AxisMask::EMPTY,
            s => AxisMask(u8::from_str_radix(s.strip_prefix('x')?, 16).ok()?),
        };

        Some(StructureKey {
            version,
            op,
            dtype,
            arch,
            idx,
            work,
            rank,
            n_operands,
            operands,
            reduce_axes,
        })
    }
}

fn parse_operand(field: &str) -> Option<OperandKey> {
    let f: Vec<&str> = field.split('/').collect();
    if f.len() != 5 {
        return None;
    }
    let contig = match f[0] {
        "co" => Contiguity::Contig,
        "ic" => Contiguity::InnerContig,
        "st" => Contiguity::Strided,
        "br" => Contiguity::Broadcast,
        _ => return None,
    };
    let bcast = AxisMask(u8::from_str_radix(f[1], 16).ok()?);
    let vec_width = match f[2] {
        "v1" => VecWidth::Scalar,
        "v2" => VecWidth::V2,
        "v4" => VecWidth::V4,
        "v8" => VecWidth::V8,
        _ => return None,
    };
    let inner_div = match f[3] {
        "d16" => DivBucket::Div16,
        "d8" => DivBucket::Div8,
        "d4" => DivBucket::Div4,
        "d2" => DivBucket::Div2,
        "da" => DivBucket::Any,
        _ => return None,
    };
    let flipped = match f[4] {
        "f" => false,
        "r" => true,
        _ => return None,
    };
    Some(OperandKey {
        contig,
        bcast,
        vec_width,
        inner_div,
        flipped,
    })
}

const fn idx_code(v: IdxWidth) -> &'static str {
    match v {
        IdxWidth::Idx32 => "i32",
        IdxWidth::Idx64 => "i64",
    }
}

const fn work_code(v: WorkClass) -> &'static str {
    match v {
        WorkClass::OneWarp => "warp",
        WorkClass::OneBlock => "block",
        WorkClass::GridStride => "grid",
    }
}

const fn contig_code(v: Contiguity) -> &'static str {
    match v {
        Contiguity::Contig => "co",
        Contiguity::InnerContig => "ic",
        Contiguity::Strided => "st",
        Contiguity::Broadcast => "br",
    }
}

const fn vec_code(v: VecWidth) -> &'static str {
    match v {
        VecWidth::Scalar => "v1",
        VecWidth::V2 => "v2",
        VecWidth::V4 => "v4",
        VecWidth::V8 => "v8",
    }
}

const fn div_code(v: DivBucket) -> &'static str {
    match v {
        DivBucket::Div16 => "d16",
        DivBucket::Div8 => "d8",
        DivBucket::Div4 => "d4",
        DivBucket::Div2 => "d2",
        DivBucket::Any => "da",
    }
}

const fn arch_code(v: ArchSku) -> &'static str {
    match v {
        ArchSku::Sm80 => "sm80",
        ArchSku::Sm89 => "sm89",
        ArchSku::Sm90a => "sm90a",
    }
}

fn arch_from_code(s: &str) -> Option<ArchSku> {
    Some(match s {
        "sm80" => ArchSku::Sm80,
        "sm89" => ArchSku::Sm89,
        "sm90a" => ArchSku::Sm90a,
        _ => return None,
    })
}

const fn dtype_code(v: ElementKind) -> &'static str {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    match v {
        F16 => "f16",
        Bf16 => "bf16",
        F32 => "f32",
        F32Strict => "f32s",
        F64 => "f64",
        S8 => "s8",
        U8 => "u8",
        I32 => "i32",
        I64 => "i64",
        Bool => "bool",
        Fp8E4M3 => "e4m3",
        Fp8E5M2 => "e5m2",
        S4 => "s4",
        U4 => "u4",
        Bin => "b1",
        Complex32 => "c32",
        Complex64 => "c64",
    }
}

fn dtype_from_code(s: &str) -> Option<ElementKind> {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    Some(match s {
        "f16" => F16,
        "bf16" => Bf16,
        "f32" => F32,
        "f32s" => F32Strict,
        "f64" => F64,
        "s8" => S8,
        "u8" => U8,
        "i32" => I32,
        "i64" => I64,
        "bool" => Bool,
        "e4m3" => Fp8E4M3,
        "e5m2" => Fp8E5M2,
        "s4" => S4,
        "u4" => U4,
        "b1" => Bin,
        "c32" => Complex32,
        "c64" => Complex64,
        _ => return None,
    })
}

fn op_code(v: OpCategory) -> &'static str {
    match v {
        OpCategory::Gemm => "gem",
        OpCategory::UnaryElementwise => "une",
        OpCategory::BinaryElementwise => "bin",
        OpCategory::TernaryElementwise => "ter",
        OpCategory::GatedActivation => "gat",
        OpCategory::Reduction => "red",
        OpCategory::Scan => "scn",
        OpCategory::Normalization => "nrm",
        OpCategory::Softmax => "sft",
        OpCategory::Convolution => "cnv",
        OpCategory::Pooling => "pol",
        OpCategory::Attention => "att",
        OpCategory::Indexing => "idx",
        OpCategory::Embedding => "emb",
        OpCategory::ShapeLayout => "shp",
        OpCategory::Sorting => "srt",
        OpCategory::Quantization => "qnt",
        OpCategory::Random => "rnd",
        OpCategory::Loss => "los",
        OpCategory::SegmentOps => "seg",
        OpCategory::Image => "img",
        OpCategory::Fft => "fft",
        OpCategory::Linalg => "lin",
        OpCategory::Moe => "moe",
        // Deliberately exhaustive (no `_` arm): `OpCategory` is the defining
        // crate's own enum here, so a newly added category surfaces as a build
        // break and forces a token code rather than silently encoding as "unk".
    }
}

fn op_from_code(s: &str) -> Option<OpCategory> {
    Some(match s {
        "gem" => OpCategory::Gemm,
        "une" => OpCategory::UnaryElementwise,
        "bin" => OpCategory::BinaryElementwise,
        "ter" => OpCategory::TernaryElementwise,
        "gat" => OpCategory::GatedActivation,
        "red" => OpCategory::Reduction,
        "scn" => OpCategory::Scan,
        "nrm" => OpCategory::Normalization,
        "sft" => OpCategory::Softmax,
        "cnv" => OpCategory::Convolution,
        "pol" => OpCategory::Pooling,
        "att" => OpCategory::Attention,
        "idx" => OpCategory::Indexing,
        "emb" => OpCategory::Embedding,
        "shp" => OpCategory::ShapeLayout,
        "srt" => OpCategory::Sorting,
        "qnt" => OpCategory::Quantization,
        "rnd" => OpCategory::Random,
        "los" => OpCategory::Loss,
        "seg" => OpCategory::SegmentOps,
        "img" => OpCategory::Image,
        "fft" => OpCategory::Fft,
        "lin" => OpCategory::Linalg,
        "moe" => OpCategory::Moe,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn od(shape: &[i64], strides: &[i64], dtype: ElementKind, align: u32) -> OperandDesc {
        OperandDesc::new(shape.len(), shape, strides, dtype, align)
    }

    #[test]
    fn contiguous_f32_vectorizes_to_v4() {
        // [128, 256] row-major f32, 256-byte aligned: inner extent 256 (%16),
        // f32 caps at V4 (float4 = 16 bytes).
        let a = od(&[128, 256], &[256, 1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        assert_eq!(k.n_operands, 3);
        assert_eq!(k.operands[0].contig, Contiguity::Contig);
        assert_eq!(k.operands[0].vec_width, VecWidth::V4);
        assert_eq!(k.operands[0].inner_div, DivBucket::Div16);
        assert_eq!(k.idx, IdxWidth::Idx32);
        assert_eq!(k.work, WorkClass::GridStride);
        assert_eq!(k.rank, 2); // raw iteration rank (collapse is deferred)
        assert!(!k.operands[0].flipped);
    }

    #[test]
    fn f16_contiguous_vectorizes_to_v8() {
        let a = od(&[64, 128], &[128, 1], ElementKind::F16, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        assert_eq!(k.operands[0].vec_width, VecWidth::V8); // f16 V8 = 16 bytes
    }

    #[test]
    fn broadcast_axis_detected() {
        // Second operand broadcasts axis 0 (stride 0 over extent 128).
        let a = od(&[128, 256], &[256, 1], ElementKind::F32, 256);
        let b = od(&[128, 256], &[0, 1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[a, b, a], ArchSku::Sm89);
        assert_eq!(k.operands[1].contig, Contiguity::Broadcast);
        assert!(k.operands[1].bcast.is_set(0));
        assert!(!k.operands[1].bcast.is_set(1));
    }

    #[test]
    fn negative_stride_is_flipped() {
        // Reversed innermost axis.
        let a = od(&[128, 256], &[256, -1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        assert!(k.operands[0].flipped);
        assert_eq!(k.operands[0].vec_width, VecWidth::Scalar); // reversed ⇒ no vec in v1
    }

    #[test]
    fn transposed_view_is_strided() {
        // [128, 256] transposed: strides [1, 128] — inner axis stride 128 ≠ 1.
        let a = od(&[128, 256], &[1, 128], ElementKind::F32, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        assert_eq!(k.operands[0].contig, Contiguity::Strided);
        assert_eq!(k.operands[0].vec_width, VecWidth::Scalar);
    }

    #[test]
    fn large_tensor_needs_idx64() {
        // 2^31 elements ⇒ max offset ≥ 2^31.
        let big: i64 = 1 << 16;
        let a = od(&[big, big], &[big, 1], ElementKind::F16, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm90a);
        assert_eq!(k.idx, IdxWidth::Idx64);
    }

    #[test]
    fn token_round_trips() {
        let a = od(&[128, 256], &[256, 1], ElementKind::F32, 256);
        let b = od(&[128, 256], &[0, 1], ElementKind::F32, 256);
        let c = od(&[128, 256], &[256, -1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[a, b, c], ArchSku::Sm89);
        let token = k.to_token();
        let parsed = StructureKey::from_token(&token).expect("round-trip parse");
        assert_eq!(k, parsed);
        // Token is human-greppable.
        assert!(token.starts_with("sk1|bin|f32|sm89|"));
    }

    #[test]
    fn scalar_operand_is_contiguous() {
        let s = od(&[], &[], ElementKind::F32, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[s, s], ArchSku::Sm80);
        assert_eq!(k.rank, 0);
        assert_eq!(k.operands[0].contig, Contiguity::Contig);
        assert_eq!(k.work, WorkClass::OneWarp);
    }

    #[test]
    fn from_tensor_ref_projects_dtype_and_shape() {
        // Build a TensorRef-shaped desc through the adapter path is exercised in
        // integration tests with a real DeviceSlice; here we validate the plain
        // constructor parity the adapter relies on.
        let a = OperandDesc::new(2, &[8, 16], &[16, 1], ElementKind::Bf16, 256);
        assert_eq!(a.rank, 2);
        assert_eq!(a.dtype, ElementKind::Bf16);
        assert_eq!(a.shape[1], 16);
    }
}
