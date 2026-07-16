//! IR-DAG cross-backend fuzzer (B11) — random elementwise IR driven through the
//! emitter + lifter and every backend, via a seeded LCG (no `proptest`/`rand`
//! dependency; fully deterministic, so any failure reproduces from the printed
//! seed/body).
//!
//! Four properties are checked here (CPU-only, no compiler required):
//!
//! - **`cuda_emit_lift_structural_round_trip`** — random IR over the *liftable* op
//!   subset (ops whose CUDA emit spelling the hand-written `lift.rs` recognizes
//!   1:1) → `generate(Cuda)` → `lift_elementwise` must recover a STRUCTURALLY
//!   EQUAL `ScalarExpr`. Generalizes the single-case `lift::tests` round-trips to
//!   thousands of random trees — a divergence is a real emitter/lifter asymmetry.
//! - **`cuda_full_op_emit_never_panics`** — random IR over the *full* float op
//!   surface (all 39 `UnaryOp`, the 16 float-valid `BinaryOp`, `Select`, `Coord`,
//!   `Param`, finite/NaN/Inf `Const`) → `build_plan` + `generate(Cuda)` must be
//!   total (no panic) and emit non-empty source. Pure emitter robustness.
//! - **`tri_backend_scalar_emit_never_panics`** — the same random body emitted
//!   through ALL THREE backends (Cuda/CpuC/Slang) over their three-way-common
//!   Scalar-schedule coverage; a panic is a real lowering-seam gap. (The sweep
//!   itself surfaced Slang's intentional `Copysign`/`Nextafter` decline.)
//! - **`oracle_elementwise_matches_independent_evaluator`** — `oracle::evaluate`
//!   (plan interpreter) vs an INDEPENDENT f64 tree-walk over a numerically-safe
//!   palette + random inputs, compared at tolerance — a numeric differential of
//!   the correctness oracle's elementwise plumbing.
//!
//! The Slang cross-lift (needs the `convert` feature's cc-built tree-sitter
//! grammars) and a CpuC compile-and-run numeric differential (needs a host C
//! compiler) are deferred follow-ups — neither the `convert` feature nor a C
//! compiler is available in the default build here.

use crate::ir::BinaryOp;
use crate::{
    CpuC, Cuda, Fidelity, OpDef, ScalarExpr, Slang, TypedBuffer, UnaryOp, build_plan, compare,
    evaluate, generate, input, konst, lift_elementwise, param,
};
use baracuda_kernel_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
};

// ---------------------------------------------------------------------------
// Seeded LCG (Knuth/MMIX constants) — the same generator used by the reader
// fuzzers; no external dependency, deterministic per seed.
// ---------------------------------------------------------------------------
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Perturb so distinct small seeds diverge immediately.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform in `0..n` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

// ---------------------------------------------------------------------------
// Op palettes.
// ---------------------------------------------------------------------------

/// Constants that survive `const_lit` (`{v:?}`) → CUDA source → the lifter's
/// `parse::<f64>` and round-trip *structurally exactly*. Deliberately
/// NON-NEGATIVE: the emitter folds `konst(-c)` into the literal `-c`, which the
/// lifter faithfully parses as `Neg(Const(c))` — semantically identical but a
/// different tree. Negative *values* are covered here via `UnaryOp::Neg`
/// (`Neg → "(-x)" → Neg`, exact both ways) instead, keeping the structural
/// invariant crisp. (The folded-negative-literal normalization is a known,
/// semantics-preserving lifter behavior, not a defect.)
const LIFTABLE_CONSTS: &[f64] = &[0.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.25];

/// Unary ops the CUDA emitter spells as a single intrinsic (or `(-x)`) the lifter
/// maps back 1:1 (`expf`/`logf`/`sqrtf`/…, `Neg`). Verified against cuda.rs and
/// lift.rs::unary_fn.
const LIFTABLE_UNARY: &[UnaryOp] = &[
    UnaryOp::Neg,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Sqrt,
    UnaryOp::Rsqrt,
    UnaryOp::Abs,
    UnaryOp::Tanh,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Floor,
    UnaryOp::Ceil,
    UnaryOp::Trunc,
    UnaryOp::Exp2,
];

/// Binary ops emitted as a recognized intrinsic CALL (`powf`/`fmaxf`/…).
const LIFTABLE_BINARY: &[BinaryOp] = &[
    BinaryOp::Pow,
    BinaryOp::Atan2,
    BinaryOp::Copysign,
    BinaryOp::FmaxIeee,
    BinaryOp::FminIeee,
];

/// Every `UnaryOp` (all are float-valid).
const ALL_UNARY: &[UnaryOp] = &[
    UnaryOp::Neg,
    UnaryOp::Abs,
    UnaryOp::Sqr,
    UnaryOp::Sqrt,
    UnaryOp::Rsqrt,
    UnaryOp::Recip,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Tanh,
    UnaryOp::Sigmoid,
    UnaryOp::Relu,
    UnaryOp::Erf,
    UnaryOp::Gelu,
    UnaryOp::Silu,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Floor,
    UnaryOp::Ceil,
    UnaryOp::Round,
    UnaryOp::Sign,
    UnaryOp::Step,
    UnaryOp::Erfc,
    UnaryOp::Trunc,
    UnaryOp::Exp2,
    UnaryOp::Expm1,
    UnaryOp::Log2,
    UnaryOp::Log10,
    UnaryOp::Log1p,
    UnaryOp::Sinh,
    UnaryOp::Cosh,
    UnaryOp::Tan,
    UnaryOp::Asin,
    UnaryOp::Acos,
    UnaryOp::Atan,
    UnaryOp::Asinh,
    UnaryOp::Acosh,
    UnaryOp::Atanh,
    UnaryOp::Cbrt,
    UnaryOp::Lgamma,
];

/// The float-valid `BinaryOp`s (excludes the int-only bitwise/shift and the
/// logical family, which the CUDA emitter rightly rejects on an f32 op).
const FLOAT_BINARY: &[BinaryOp] = &[
    BinaryOp::Max,
    BinaryOp::Min,
    BinaryOp::Pow,
    BinaryOp::Rem,
    BinaryOp::Atan2,
    BinaryOp::Copysign,
    BinaryOp::Nextafter,
    BinaryOp::FmaxIeee,
    BinaryOp::FminIeee,
    BinaryOp::RemTrunc,
    BinaryOp::CmpEq,
    BinaryOp::CmpNe,
    BinaryOp::CmpLt,
    BinaryOp::CmpLe,
    BinaryOp::CmpGt,
    BinaryOp::CmpGe,
];

/// Constants for the robustness sweep — includes the non-finite classes the
/// emitter must spell (`NAN`/`INFINITY`) without panicking.
const FULL_CONSTS: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -3.5,
    1e-7,
    1e30,
    f64::MIN_POSITIVE,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
];

/// A generator configuration: which leaves and internal nodes are legal.
struct Palette {
    consts: &'static [f64],
    unary: &'static [UnaryOp],
    binary: &'static [BinaryOp],
    select: bool,
    coord: bool,
    param: bool,
}

const LIFTABLE: Palette = Palette {
    consts: LIFTABLE_CONSTS,
    unary: LIFTABLE_UNARY,
    binary: LIFTABLE_BINARY,
    select: false,
    coord: false,
    param: false,
};

const FULL: Palette = Palette {
    consts: FULL_CONSTS,
    unary: ALL_UNARY,
    binary: FLOAT_BINARY,
    select: true,
    coord: true,
    param: true,
};

/// Binary ops in the THREE-WAY-common coverage of Cuda/CpuC/Slang: the liftable
/// set minus `Copysign`. Slang v1 panic-declines `Copysign`/`Nextafter` and 8
/// transcendentals (`Erf`/`Erfc`/`Gelu`/`Lgamma`/`Cbrt`/`Asinh`/`Acosh`/`Atanh`)
/// as a documented follow-up (slang.rs) — the same honest panic-boundary
/// convention CUDA/CpuC use for unsupported dtypes. Of those, `Copysign` is the
/// only one in the liftable palette; the transcendentals + `Nextafter` aren't
/// liftable anyway. `LIFTABLE_UNARY` is already entirely within Slang's coverage.
const TRI_BINARY: &[BinaryOp] = &[
    BinaryOp::Pow,
    BinaryOp::Atan2,
    BinaryOp::FmaxIeee,
    BinaryOp::FminIeee,
];

const TRI_BACKEND: Palette = Palette {
    consts: LIFTABLE_CONSTS,
    unary: LIFTABLE_UNARY,
    binary: TRI_BINARY,
    select: false,
    coord: false,
    param: false,
};

// ---------------------------------------------------------------------------
// The random-expression generator.
// ---------------------------------------------------------------------------

fn gen_leaf(rng: &mut Rng, n_inputs: usize, pal: &Palette) -> crate::Expr {
    // Leaf kinds: 0 input, 1 const, 2 coord, 3 param.
    let mut kinds: [u8; 4] = [0, 1, 0, 0];
    let mut n = 2;
    if pal.coord {
        kinds[n] = 2;
        n += 1;
    }
    if pal.param {
        kinds[n] = 3;
        n += 1;
    }
    match kinds[rng.below(n)] {
        0 => input(rng.below(n_inputs) as u8),
        1 => konst(*rng.pick(pal.consts)),
        2 => crate::coord(0), // axis 0 — valid for the rank-1 cell
        _ => param(rng.below(4) as u8),
    }
}

fn gen_expr(rng: &mut Rng, depth: usize, n_inputs: usize, pal: &Palette) -> crate::Expr {
    // Bias toward leaves as depth runs out, and 1-in-3 early stop for variety.
    if depth == 0 || rng.below(3) == 0 {
        return gen_leaf(rng, n_inputs, pal);
    }
    // Internal-node kinds: 0 arith, 1 unary, 2 binary, 3 select.
    let mut kinds: [u8; 4] = [0, 1, 2, 0];
    let mut n = 3;
    if pal.select {
        kinds[n] = 3;
        n += 1;
    }
    match kinds[rng.below(n)] {
        0 => {
            let a = gen_expr(rng, depth - 1, n_inputs, pal);
            let b = gen_expr(rng, depth - 1, n_inputs, pal);
            match rng.below(4) {
                0 => a + b,
                1 => a - b,
                2 => a * b,
                _ => a / b,
            }
        }
        1 => gen_expr(rng, depth - 1, n_inputs, pal).unary(*rng.pick(pal.unary)),
        2 => {
            let a = gen_expr(rng, depth - 1, n_inputs, pal);
            let b = gen_expr(rng, depth - 1, n_inputs, pal);
            a.binary(*rng.pick(pal.binary), b)
        }
        _ => {
            let cond = gen_expr(rng, depth - 1, n_inputs, pal);
            let a = gen_expr(rng, depth - 1, n_inputs, pal);
            let b = gen_expr(rng, depth - 1, n_inputs, pal);
            cond.select(a, b)
        }
    }
}

/// A contiguous rank-1 f32 cell with **align 4**, which forces the Scalar
/// (non-vectorized) schedule — the only CUDA form the `out[i] = <expr>;` lifter
/// can parse. Mirrors `lift::tests::round_trip_reemits_to_cuda_and_cpuc`.
fn scalar_key(n_inputs: usize) -> StructureKey {
    let a = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 4);
    let cat = match n_inputs {
        1 => OpCategory::UnaryElementwise,
        2 => OpCategory::BinaryElementwise,
        _ => OpCategory::TernaryElementwise,
    };
    let descs = vec![a; n_inputs + 1];
    structure_key(cat, &descs, ArchSku::Sm89)
}

/// `true` if the body contains a structurally-duplicated NON-leaf subtree — the
/// exact condition (`consumers > 1` on a non-leaf) under which the emitter's DAG
/// CSE hoists a shared value to a `float tmpN = …;` prelude (ir.rs / `lower_dag`).
/// The single-statement pilot lifter parses only `out[i] = <expr>;`, so a hoisted
/// `tmp` reads as an unknown identifier — a known lifter limitation, not a defect.
/// The structural round-trip is therefore scoped to single-use (non-CSE) trees,
/// which this filters. Detected the same way the interner decides: bottom-up,
/// leaves excluded, first structural repeat wins.
fn has_cse_hoist(e: &crate::ScalarExpr, seen: &mut std::collections::HashSet<String>) -> bool {
    use crate::ScalarExpr::{
        Add, Binary, Const, Coord, Div, Input, Mul, Param, Reduced, Select, Sub, Unary,
    };
    let kids: Vec<&crate::ScalarExpr> = match e {
        Input(_) | Const(_) | Param(_) | Reduced(_) | Coord(_) => return false, // leaf
        Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Binary(_, a, b) => vec![a, b],
        Unary(_, a) => vec![a],
        Select(a, b, c) => vec![a, b, c],
    };
    for k in kids {
        if has_cse_hoist(k, seen) {
            return true;
        }
    }
    // Non-leaf: a second structurally-equal occurrence ⇒ the emitter hoists it.
    !seen.insert(format!("{e:?}"))
}

// ---------------------------------------------------------------------------
// Properties.
// ---------------------------------------------------------------------------

#[test]
fn cuda_emit_lift_structural_round_trip() {
    let mut rng = Rng::new(0xB11_0001);
    let mut lifted_ok = 0usize;
    let mut skipped_cse = 0usize;
    for i in 0..6000 {
        let n_inputs = 1 + rng.below(3); // 1..=3
        let op = OpDef::elementwise(
            "fuzz",
            n_inputs as u8,
            &[ElementKind::F32],
            gen_expr(&mut rng, 4, n_inputs, &LIFTABLE),
        );
        // Single-use trees only — a shared non-leaf emits a hoisted `tmp` the
        // pilot lifter can't parse (see `has_cse_hoist`).
        if has_cse_hoist(&op.body, &mut std::collections::HashSet::new()) {
            skipped_cse += 1;
            continue;
        }
        let key = scalar_key(n_inputs);
        let k = generate(&op, &key, &Cuda);
        let lifted = match lift_elementwise(&k.source, "fuzz", &[ElementKind::F32]) {
            Ok(l) => l,
            Err(e) => panic!(
                "case {i}: liftable single-use IR did not lift back ({e:?})\nbody: {:?}\nsource:\n{}",
                op.body, k.source
            ),
        };
        assert_eq!(
            lifted.op.body, op.body,
            "case {i}: emit→lift changed the IR\norig: {:?}\ngot:  {:?}\nsource:\n{}",
            op.body, lifted.op.body, k.source
        );
        lifted_ok += 1;
    }
    // Guard against a vacuous pass: the vast majority of random trees are
    // single-use and MUST round-trip; only a minority hit CSE and are skipped.
    assert!(
        lifted_ok > 4000,
        "too few round-trip cases exercised ({lifted_ok} ok, {skipped_cse} CSE-skipped) — \
         the generator or filter is degenerate"
    );
}

#[test]
fn cuda_full_op_emit_never_panics() {
    let mut rng = Rng::new(0xB11_0002);
    for _ in 0..5000 {
        let n_inputs = 1 + rng.below(3);
        let op = OpDef::elementwise(
            "fuzz",
            n_inputs as u8,
            &[ElementKind::F32],
            gen_expr(&mut rng, 5, n_inputs, &FULL),
        );
        let key = scalar_key(n_inputs);
        // build_plan + emit must be total (no panic) and yield real source.
        let _plan = build_plan(&op, &key);
        let k = generate(&op, &key, &Cuda);
        assert!(
            !k.source.is_empty(),
            "empty CUDA source for body: {:?}",
            op.body
        );
    }
}

#[test]
fn tri_backend_scalar_emit_never_panics() {
    // The genuinely CROSS-backend robustness check: the same random body emitted
    // through all THREE backends (CUDA / CpuC / Slang), which share the neutral
    // `Lowering` seam. Scoped to the liftable palette at F32 with the Scalar
    // schedule (align 4) — the common ground every backend serves: single-output
    // contiguous elementwise, finite consts, no Coord/Param/Select (CpuC/Slang v1
    // decline those). A panic here is a real lowering-seam gap in one backend.
    let mut rng = Rng::new(0xB11_0003);
    for _ in 0..4000 {
        let n_inputs = 1 + rng.below(3);
        let op = OpDef::elementwise(
            "fuzz",
            n_inputs as u8,
            &[ElementKind::F32],
            gen_expr(&mut rng, 4, n_inputs, &TRI_BACKEND),
        );
        let key = scalar_key(n_inputs);
        for (name, src) in [
            ("cuda", generate(&op, &key, &Cuda).source),
            ("cpu_c", generate(&op, &key, &CpuC).source),
            ("slang", generate(&op, &key, &Slang).source),
        ] {
            assert!(
                !src.is_empty(),
                "{name} emitted empty source for body: {:?}",
                op.body
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Numeric differential: the plan interpreter (`oracle::evaluate`) vs an
// INDEPENDENT tree-walk, over a numerically-safe palette.
// ---------------------------------------------------------------------------

/// Small finite constants for the numeric sweep (negatives allowed — this is a
/// value check, not the structural round-trip).
const SAFE_CONSTS: &[f64] = &[-2.0, -0.5, 0.0, 0.5, 1.5, 2.0];

/// A random body over `{+, -, *, Neg, Abs, Sqr}` — total on finite inputs (no
/// `Div`/`Max`/`Min`, so no NaN source and no NaN-semantics ambiguity between the
/// two evaluators), keeping the differential about the PLUMBING (multi-input
/// indexing, output store, f32 narrowing), not scalar-op edge cases.
fn gen_safe(rng: &mut Rng, depth: usize, n_inputs: usize) -> crate::Expr {
    if depth == 0 || rng.below(3) == 0 {
        return if rng.below(4) == 0 {
            konst(*rng.pick(SAFE_CONSTS))
        } else {
            input(rng.below(n_inputs) as u8)
        };
    }
    match rng.below(6) {
        0 => gen_safe(rng, depth - 1, n_inputs) + gen_safe(rng, depth - 1, n_inputs),
        1 => gen_safe(rng, depth - 1, n_inputs) - gen_safe(rng, depth - 1, n_inputs),
        2 => gen_safe(rng, depth - 1, n_inputs) * gen_safe(rng, depth - 1, n_inputs),
        3 => gen_safe(rng, depth - 1, n_inputs).unary(UnaryOp::Neg),
        4 => gen_safe(rng, depth - 1, n_inputs).unary(UnaryOp::Abs),
        _ => gen_safe(rng, depth - 1, n_inputs).unary(UnaryOp::Sqr),
    }
}

/// Independent f64 tree-walk — shares NO code with `oracle::evaluate` (which
/// routes through `build_plan` → the `Val` domain → store rounding). `ins[i]` is
/// input `i`'s value at the current element.
fn eval_ref(e: &ScalarExpr, ins: &[f64]) -> f64 {
    match e {
        ScalarExpr::Input(i) => ins[*i as usize],
        ScalarExpr::Const(c) => *c,
        ScalarExpr::Add(a, b) => eval_ref(a, ins) + eval_ref(b, ins),
        ScalarExpr::Sub(a, b) => eval_ref(a, ins) - eval_ref(b, ins),
        ScalarExpr::Mul(a, b) => eval_ref(a, ins) * eval_ref(b, ins),
        ScalarExpr::Unary(UnaryOp::Neg, a) => -eval_ref(a, ins),
        ScalarExpr::Unary(UnaryOp::Abs, a) => eval_ref(a, ins).abs(),
        ScalarExpr::Unary(UnaryOp::Sqr, a) => {
            let x = eval_ref(a, ins);
            x * x
        }
        other => unreachable!("safe palette produced an unexpected node: {other:?}"),
    }
}

#[test]
fn oracle_elementwise_matches_independent_evaluator() {
    let mut rng = Rng::new(0xB11_0004);
    let len: i64 = 48;
    for _ in 0..2500 {
        let n_inputs = 1 + rng.below(3);
        let op = OpDef::elementwise(
            "fuzz",
            n_inputs as u8,
            &[ElementKind::F32],
            gen_safe(&mut rng, 4, n_inputs),
        );
        // Small contiguous rank-1 f32 cell (align 4 → Scalar) — descs shared by
        // the plan and the evaluate() call (inputs THEN output).
        let a = OperandDesc::new(1, &[len], &[1], ElementKind::F32, 4);
        let cat = match n_inputs {
            1 => OpCategory::UnaryElementwise,
            2 => OpCategory::BinaryElementwise,
            _ => OpCategory::TernaryElementwise,
        };
        let descs = vec![a; n_inputs + 1];
        let key = structure_key(cat, &descs, ArchSku::Sm89);
        let plan = build_plan(&op, &key);

        // Random finite f32 inputs in [-3, 3).
        let cols: Vec<Vec<f32>> = (0..n_inputs)
            .map(|_| {
                (0..len)
                    .map(|_| {
                        let u = (rng.next() >> 40) as f64 / (1u64 << 24) as f64; // [0,1)
                        (u * 6.0 - 3.0) as f32
                    })
                    .collect()
            })
            .collect();
        let inbufs: Vec<TypedBuffer> = cols
            .iter()
            .map(|c| TypedBuffer::from_f32(&[len], c))
            .collect();

        let actual = &evaluate(&plan, &descs, &inbufs, &[])[0];

        let exp: Vec<f32> = (0..len as usize)
            .map(|idx| {
                let ins: Vec<f64> = cols.iter().map(|c| c[idx] as f64).collect();
                eval_ref(&op.body, &ins) as f32
            })
            .collect();
        let expected = TypedBuffer::from_f32(&[len], &exp);

        compare(
            &expected,
            actual,
            Fidelity::Tolerant {
                rel: 1e-6,
                abs: 1e-6,
            },
        )
        .unwrap_or_else(|e| {
            panic!(
                "oracle vs independent evaluator diverged: {e}\nbody: {:?}",
                op.body
            )
        });
    }
}
