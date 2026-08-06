//! CUDA → IR frontend (Phase 4 spike) — the source-language side of the
//! translation hub (`docs/design/ir-translation-hub.md`).
//!
//! Recognizes a **grid-stride elementwise** CUDA kernel and lifts its per-element
//! body expression into the neutral IR (an [`OpDef`]), so an existing
//! hand-written kernel can be re-emitted to every backend the generator has an
//! emitter for (CUDA, CpuC, …). It **refuses** anything it cannot express
//! faithfully — a non-`[i]` index, an unknown call, shared memory, an atomic,
//! a library call — returning a typed [`LiftError`] refusal rather than silently
//! mis-lifting. That is the "convert what you can, leave the rest in the source
//! language" contract: portability scales with the lift fraction, honestly.
//!
//! This is a deliberately small recognizer + expression parser — enough to prove
//! source → IR → every-backend end-to-end. A production frontend would parse the
//! full CUDA AST (libclang); here we target one idiom and validate the round-trip
//! against the oracle.

use crate::ir::{BinaryOp, Expr, OpDef, ScalarExpr, UnaryOp};
use baracuda_kernel_vocab::ElementKind;

/// Why a source kernel could not be fully lifted — the KISS-Consume refusal
/// taxonomy. Each variant is one machine-actionable category (see
/// [`LiftError::refusal`]); the string on the two residue variants names the
/// construct so the un-lifted remainder stays honest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiftError {
    /// No `__global__` kernel in the source. (not-a-kernel)
    NotAKernel,
    /// A kernel, but not the `out[i] = <expr>;` elementwise op class this
    /// recognizer targets. (wrong-op-class — the recognized-op-class miss.)
    NotElementwise,
    /// A construct KISS-Ops could express that this small recognizer doesn't
    /// handle yet (an unknown call/token, a non-`[i]` read, …). A recognizer gap,
    /// not an IR limit. (unrecognized-but-expressible)
    Unrecognized(String),
    /// A construct with no neutral-IR representation — it MUST stay in the source
    /// language (shared memory, atomics, sync, library calls, inline asm, warp
    /// shuffles). (inexpressible-residue)
    Inexpressible(String),
}

/// The KISS-Consume refusal category of a [`LiftError`] — machine-actionable, so a
/// consumer can choose its next move (retry, request the op be added, fall back to
/// another provider, or leave the fragment in the source language) rather than
/// parse a free-text reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsumeRefusal {
    /// The input is not a kernel.
    NotAKernel,
    /// A kernel of a different op class than this recognizer targets.
    WrongOpClass,
    /// KISS-Ops could express it; this recognizer doesn't yet.
    UnrecognizedButExpressible,
    /// No neutral-IR representation; leave it in the source language.
    InexpressibleResidue,
}

impl LiftError {
    /// This refusal's KISS-Consume category.
    #[must_use]
    pub fn refusal(&self) -> ConsumeRefusal {
        match self {
            LiftError::NotAKernel => ConsumeRefusal::NotAKernel,
            LiftError::NotElementwise => ConsumeRefusal::WrongOpClass,
            LiftError::Unrecognized(_) => ConsumeRefusal::UnrecognizedButExpressible,
            LiftError::Inexpressible(_) => ConsumeRefusal::InexpressibleResidue,
        }
    }
}

/// A successfully lifted elementwise op.
#[derive(Debug)]
pub struct Lifted {
    /// The neutral-IR op, ready for `generate(&op, &key, &backend)`.
    pub op: OpDef,
    /// Number of input operands the body references (`in0..in{n-1}`).
    pub n_inputs: u8,
}

/// Lift a grid-stride elementwise CUDA kernel's body into an [`OpDef`] named
/// `name`, accepting `dtypes`. See the module docs for the recognized idiom.
pub fn lift_elementwise(
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    if !src.contains("__global__") {
        return Err(LiftError::NotAKernel);
    }
    // Constructs that place a kernel outside the neutral elementwise IR: they
    // belong in the source language (residue), never mis-lifted.
    for marker in [
        "__shared__",
        "atomicAdd",
        "atomicCAS",
        "__syncthreads",
        "cublas",
        "cudnn",
        "printf",
        "asm",
        "cp.async",
        "__shfl",
    ] {
        if src.contains(marker) {
            return Err(LiftError::Inexpressible(marker.to_string()));
        }
    }
    let (rhs, idx_var) = extract_store(src).ok_or(LiftError::NotElementwise)?;
    let mut p = Parser::new(&rhs, &idx_var)?;
    let body = p.parse_expr()?;
    p.expect_end()?;
    let n_inputs = p.max_input.map_or(0, |m| m + 1);
    let op = OpDef::elementwise(name, n_inputs, dtypes, Expr(body));
    Ok(Lifted { op, n_inputs })
}

/// Find the first `out[<idx>] = <rhs>;` store and return `(rhs, idx_var)`.
fn extract_store(src: &str) -> Option<(String, String)> {
    let start = src.find("out[")?;
    let after_br = &src[start + 4..];
    let close = after_br.find(']')?;
    let idx_var = after_br[..close].trim().to_string();
    // After `]`, the next non-space must be a single `=` (assignment, not `==`).
    let rest = after_br[close + 1..].trim_start();
    let mut chars = rest.chars();
    if chars.next()? != '=' {
        return None;
    }
    if chars.clone().next() == Some('=') {
        return None; // a comparison, not a store
    }
    let rhs_all = &rest[1..];
    let semi = rhs_all.find(';')?;
    Some((rhs_all[..semi].trim().to_string(), idx_var))
}

// ---- tokens ----

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    LBrack,
    RBrack,
    Comma,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, LiftError> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit()
            || (c == '.' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit())
        {
            let st = i;
            while i < b.len() {
                let d = b[i] as char;
                if d.is_ascii_digit()
                    || d == '.'
                    || d == 'e'
                    || d == 'E'
                    || ((d == '+' || d == '-') && i > st && matches!(b[i - 1] as char, 'e' | 'E'))
                {
                    i += 1;
                } else {
                    break;
                }
            }
            let lit = &s[st..i];
            // Consume a trailing float suffix (`f`/`F`).
            if i < b.len() && matches!(b[i] as char, 'f' | 'F') {
                i += 1;
            }
            let v: f64 = lit
                .parse()
                .map_err(|_| LiftError::Unrecognized(format!("literal {lit}")))?;
            out.push(Tok::Num(v));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] as char == '_') {
                i += 1;
            }
            out.push(Tok::Ident(s[st..i].to_string()));
        } else {
            let t = match c {
                '+' => Tok::Plus,
                '-' => Tok::Minus,
                '*' => Tok::Star,
                '/' => Tok::Slash,
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                '[' => Tok::LBrack,
                ']' => Tok::RBrack,
                ',' => Tok::Comma,
                other => return Err(LiftError::Unrecognized(format!("token '{other}'"))),
            };
            out.push(t);
            i += 1;
        }
    }
    Ok(out)
}

// ---- parser (precedence-climbing recursive descent) ----

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    idx_var: String,
    max_input: Option<u8>,
}

impl Parser {
    fn new(rhs: &str, idx_var: &str) -> Result<Self, LiftError> {
        Ok(Self {
            toks: tokenize(rhs)?,
            pos: 0,
            idx_var: idx_var.to_string(),
            max_input: None,
        })
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect_end(&self) -> Result<(), LiftError> {
        if self.pos == self.toks.len() {
            Ok(())
        } else {
            Err(LiftError::Unrecognized(format!(
                "trailing tokens {:?}",
                &self.toks[self.pos..]
            )))
        }
    }

    // expr := add
    fn parse_expr(&mut self) -> Result<ScalarExpr, LiftError> {
        self.parse_add()
    }

    // add := mul (('+'|'-') mul)*
    fn parse_add(&mut self) -> Result<ScalarExpr, LiftError> {
        let mut lhs = self.parse_mul()?;
        loop {
            if self.eat(&Tok::Plus) {
                let rhs = self.parse_mul()?;
                lhs = ScalarExpr::Add(Box::new(lhs), Box::new(rhs));
            } else if self.eat(&Tok::Minus) {
                let rhs = self.parse_mul()?;
                lhs = ScalarExpr::Sub(Box::new(lhs), Box::new(rhs));
            } else {
                return Ok(lhs);
            }
        }
    }

    // mul := unary (('*'|'/') unary)*
    fn parse_mul(&mut self) -> Result<ScalarExpr, LiftError> {
        let mut lhs = self.parse_unary()?;
        loop {
            if self.eat(&Tok::Star) {
                let rhs = self.parse_unary()?;
                lhs = ScalarExpr::Mul(Box::new(lhs), Box::new(rhs));
            } else if self.eat(&Tok::Slash) {
                let rhs = self.parse_unary()?;
                lhs = ScalarExpr::Div(Box::new(lhs), Box::new(rhs));
            } else {
                return Ok(lhs);
            }
        }
    }

    // unary := '-' unary | primary
    fn parse_unary(&mut self) -> Result<ScalarExpr, LiftError> {
        if self.eat(&Tok::Minus) {
            let e = self.parse_unary()?;
            return Ok(ScalarExpr::Unary(UnaryOp::Neg, Box::new(e)));
        }
        self.parse_primary()
    }

    // primary := num | '(' expr ')' | inK '[' idx ']' | func '(' args ')'
    fn parse_primary(&mut self) -> Result<ScalarExpr, LiftError> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(ScalarExpr::Const(v)),
            Some(Tok::LParen) => {
                let e = self.parse_expr()?;
                if !self.eat(&Tok::RParen) {
                    return Err(LiftError::Unrecognized("unmatched '('".into()));
                }
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                if self.peek() == Some(&Tok::LParen) {
                    self.parse_call(&name)
                } else if self.peek() == Some(&Tok::LBrack) {
                    self.parse_input_ref(&name)
                } else {
                    Err(LiftError::Unrecognized(format!("identifier '{name}'")))
                }
            }
            other => Err(LiftError::Unrecognized(format!("unexpected {other:?}"))),
        }
    }

    // inK '[' idx ']'  (idx must be the elementwise index variable)
    fn parse_input_ref(&mut self, name: &str) -> Result<ScalarExpr, LiftError> {
        let k: u8 = name
            .strip_prefix("in")
            .and_then(|d| d.parse().ok())
            .ok_or_else(|| LiftError::Unrecognized(format!("read '{name}[..]' (not inK)")))?;
        self.eat(&Tok::LBrack);
        match self.next() {
            Some(Tok::Ident(v)) if v == self.idx_var => {}
            other => {
                return Err(LiftError::Unrecognized(format!(
                    "non-elementwise index in {name}[{other:?}] (expected [{}])",
                    self.idx_var
                )));
            }
        }
        if !self.eat(&Tok::RBrack) {
            return Err(LiftError::Unrecognized("unmatched '['".into()));
        }
        self.max_input = Some(self.max_input.map_or(k, |m| m.max(k)));
        Ok(ScalarExpr::Input(k))
    }

    // func '(' expr (',' expr)? ')'
    fn parse_call(&mut self, name: &str) -> Result<ScalarExpr, LiftError> {
        self.eat(&Tok::LParen);
        let a = self.parse_expr()?;
        if self.eat(&Tok::Comma) {
            let b = self.parse_expr()?;
            if !self.eat(&Tok::RParen) {
                return Err(LiftError::Unrecognized("unmatched '(' in call".into()));
            }
            let op = binary_fn(name)
                .ok_or_else(|| LiftError::Unrecognized(format!("call '{name}(_,_)'")))?;
            Ok(ScalarExpr::Binary(op, Box::new(a), Box::new(b)))
        } else {
            if !self.eat(&Tok::RParen) {
                return Err(LiftError::Unrecognized("unmatched '(' in call".into()));
            }
            let op = unary_fn(name)
                .ok_or_else(|| LiftError::Unrecognized(format!("call '{name}(_)'")))?;
            Ok(ScalarExpr::Unary(op, Box::new(a)))
        }
    }
}

/// CUDA unary math intrinsic → [`UnaryOp`], or `None` (unrecognized → residue).
pub(crate) fn unary_fn(name: &str) -> Option<UnaryOp> {
    Some(match name {
        "expf" | "__expf" => UnaryOp::Exp,
        "logf" | "__logf" => UnaryOp::Log,
        "sqrtf" | "__fsqrt_rn" => UnaryOp::Sqrt,
        "rsqrtf" => UnaryOp::Rsqrt,
        "fabsf" | "fabs" | "abs" => UnaryOp::Abs,
        "tanhf" => UnaryOp::Tanh,
        "sinf" => UnaryOp::Sin,
        "cosf" => UnaryOp::Cos,
        "floorf" => UnaryOp::Floor,
        "ceilf" => UnaryOp::Ceil,
        "truncf" => UnaryOp::Trunc,
        "exp2f" => UnaryOp::Exp2,
        _ => return None,
    })
}

/// CUDA binary math intrinsic → [`BinaryOp`], or `None`. Note `fmaxf`/`fminf`
/// are IEEE NaN-**suppressing** ([`BinaryOp::FmaxIeee`]/`FminIeee`), distinct
/// from the NaN-propagating `Max`/`Min`.
pub(crate) fn binary_fn(name: &str) -> Option<BinaryOp> {
    Some(match name {
        "fmaxf" | "fmax" => BinaryOp::FmaxIeee,
        "fminf" | "fmin" => BinaryOp::FminIeee,
        "powf" => BinaryOp::Pow,
        "atan2f" => BinaryOp::Atan2,
        "copysignf" => BinaryOp::Copysign,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: &[ElementKind] = &[ElementKind::F32];

    fn body(src: &str) -> ScalarExpr {
        lift_elementwise(src, "lifted", F32).unwrap().op.body
    }

    #[test]
    fn lifts_fused_multiply_add() {
        let src = "__global__ void k(const float* in0, const float* in1, const float* in2, float* out, long long n) {\n\
                   long long i = blockIdx.x*blockDim.x+threadIdx.x;\n\
                   for (; i < n; i += gridDim.x*blockDim.x) { out[i] = in0[i] * in1[i] + in2[i]; }\n}";
        let lifted = lift_elementwise(src, "fma", F32).unwrap();
        assert_eq!(lifted.n_inputs, 3);
        // Add(Mul(Input0,Input1), Input2)
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Add(
                Box::new(ScalarExpr::Mul(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Input(1)),
                )),
                Box::new(ScalarExpr::Input(2)),
            )
        );
    }

    #[test]
    fn lifts_unary_intrinsic() {
        let src = "__global__ void k(const float* in0, float* out, long long n){ out[i] = __expf(in0[i]); }";
        assert_eq!(
            body(src),
            ScalarExpr::Unary(UnaryOp::Exp, Box::new(ScalarExpr::Input(0)))
        );
    }

    #[test]
    fn lifts_relu_as_ieee_fmax_not_torch_max() {
        // CUDA fmaxf is IEEE NaN-suppressing — must map to FmaxIeee, not Max.
        let src = "__global__ void k(const float* in0, float* out, long long n){ out[i] = fmaxf(in0[i], 0.0f); }";
        assert_eq!(
            body(src),
            ScalarExpr::Binary(
                BinaryOp::FmaxIeee,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Const(0.0)),
            )
        );
    }

    #[test]
    fn precedence_and_parens() {
        let src = "__global__ void k(const float* in0, const float* in1, float* out, long long n){ out[i] = (in0[i] + in1[i]) * 2.0f; }";
        assert_eq!(
            body(src),
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Add(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Input(1)),
                )),
                Box::new(ScalarExpr::Const(2.0)),
            )
        );
    }

    #[test]
    fn refuses_non_elementwise_index() {
        // A neighbor read `in0[i+1]` is not elementwise — residue, not mis-lift.
        let src =
            "__global__ void k(const float* in0, float* out, long long n){ out[i] = in0[i+1]; }";
        assert!(matches!(
            lift_elementwise(src, "x", F32),
            Err(LiftError::Unrecognized(_))
        ));
    }

    #[test]
    fn shared_memory_is_inexpressible_residue() {
        // A construct with no neutral-IR representation → inexpressible-residue,
        // "leave it in the source language".
        let smem = "__global__ void k(float* out){ __shared__ float s[32]; out[i] = s[0]; }";
        let e = lift_elementwise(smem, "x", F32).unwrap_err();
        assert_eq!(e, LiftError::Inexpressible("__shared__".into()));
        assert_eq!(e.refusal(), ConsumeRefusal::InexpressibleResidue);
    }

    #[test]
    fn unknown_call_is_unrecognized_but_expressible() {
        // KISS-Ops could express `mystery` if it were a known op; the recognizer
        // just doesn't map it → unrecognized-but-expressible (a recognizer gap).
        let unk = "__global__ void k(const float* in0, float* out){ out[i] = mystery(in0[i]); }";
        let e = lift_elementwise(unk, "x", F32).unwrap_err();
        assert!(matches!(e, LiftError::Unrecognized(_)));
        assert_eq!(e.refusal(), ConsumeRefusal::UnrecognizedButExpressible);
    }

    #[test]
    fn refusal_maps_the_kiss_consume_categories() {
        assert_eq!(LiftError::NotAKernel.refusal(), ConsumeRefusal::NotAKernel);
        assert_eq!(
            LiftError::NotElementwise.refusal(),
            ConsumeRefusal::WrongOpClass
        );
    }

    #[test]
    fn round_trip_reemits_to_cuda_and_cpuc() {
        use crate::{CpuC, generate};
        use baracuda_cuda_emit::Cuda;
        use baracuda_kernel_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};
        // A hand-written CUDA elementwise kernel a "copy-paster" might have.
        let src = "__global__ void mul(const float* in0, const float* in1, float* out, long long n) {\n\
                   long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
                   long long step = gridDim.x*blockDim.x;\n\
                   for (; i < n; i += step) { out[i] = in0[i] * in1[i]; }\n}";
        let lifted = lift_elementwise(src, "lifted_mul", F32).unwrap();
        assert_eq!(lifted.n_inputs, 2);
        // Re-emit the SAME lifted IR to two different backends. `align = 4`
        // yields the Scalar (non-vectorized) schedule the CpuC v1 backend serves.
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let cuda = generate(&lifted.op, &key, &Cuda);
        let cpuc = generate(&lifted.op, &key, &CpuC);
        // Both backends read both inputs — the op survived source → IR → re-emit.
        for s in [&cuda.source, &cpuc.source] {
            assert!(
                s.contains("in0[") && s.contains("in1["),
                "missing an input:\n{s}"
            );
        }
        assert!(
            cuda.source.contains("__global__"),
            "not a CUDA kernel:\n{}",
            cuda.source
        );
        assert!(
            cpuc.source.contains("for (long long i"),
            "not a CpuC host loop:\n{}",
            cpuc.source
        );
        // Visible payoff (run with --nocapture).
        println!("=== re-emitted CUDA ===\n{}", cuda.source);
        println!("=== re-emitted CpuC ===\n{}", cpuc.source);
    }

    #[test]
    fn rejects_non_kernel_and_non_elementwise() {
        assert_eq!(
            lift_elementwise("int f(){return 0;}", "x", F32).unwrap_err(),
            LiftError::NotAKernel
        );
        assert_eq!(
            lift_elementwise("__global__ void k(){ int z = 3; }", "x", F32).unwrap_err(),
            LiftError::NotElementwise
        );
    }

    /// Dump the lift round-trip PAIR for the on-device differential validator
    /// (`ondevice/lift_roundtrip_validate.cu`): the re-emitted CUDA kernel and a
    /// hand-authored ORIGINAL with the SAME body. On device the two must produce
    /// BIT-IDENTICAL output — the numerical proof that CUDA → IR → re-emit is
    /// faithful. Mirrors `cuda::tests::dump_coord_unravel_helper` (env var +
    /// `fs::write`). Run with:
    ///   `LIFT_OUT=<outdir> cargo test -p baracuda-kernelgen --lib lift::tests::dump_lift_roundtrip -- --ignored --nocapture`
    #[test]
    #[ignore = "writes the lift round-trip pair for the on-device differential validator"]
    fn dump_lift_roundtrip() {
        use crate::generate;
        use baracuda_cuda_emit::Cuda;
        use baracuda_kernel_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};

        let out = std::env::var("LIFT_OUT").unwrap_or_else(|_| ".".to_string());

        // A hand-written CUDA elementwise kernel with a couple of fused ops: a
        // per-element expression referencing TWO inputs, `in0` twice. The lifted
        // IR is re-emitted below; the same body is re-authored as `lift_orig`.
        let src = "__global__ void lift_src(const float* in0, const float* in1, float* out, long long n) {\n\
                   long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
                   long long step = gridDim.x*blockDim.x;\n\
                   for (; i < n; i += step) { out[i] = in0[i] * in1[i] + in0[i]; }\n}";
        let lifted = lift_elementwise(src, "lift_rt", F32).unwrap();
        assert_eq!(lifted.n_inputs, 2);

        // Contiguous SCALAR cell: align = 4 forces Schedule::Scalar, so the
        // re-emitted signature is `(const float* in0, const float* in1, float*
        // out, long long n)` — the SAME shape as the hand-written original, so
        // the on-device harness launches both with identical args.
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let k = generate(&lifted.op, &key, &Cuda);

        // (a) the re-emitted CUDA kernel (its `.name` carries a cell suffix).
        let gen_path = format!("{out}/lift_gen.cu");
        std::fs::write(&gen_path, &k.source).unwrap();

        // (b) the ORIGINAL kernel, compilable, with the IDENTICAL body under the
        // fixed name `lift_orig` (the harness `#include`s both).
        let orig = "__global__ void lift_orig(const float* in0, const float* in1, float* out, long long n) {\n\
                    long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
                    long long step = gridDim.x*blockDim.x;\n\
                    for (; i < n; i += step) { out[i] = in0[i] * in1[i] + in0[i]; }\n}\n";
        let orig_path = format!("{out}/lift_orig.cu");
        std::fs::write(&orig_path, orig).unwrap();

        println!("wrote {gen_path}");
        println!("wrote {orig_path}");
        // The harness must launch the generated kernel by THIS name.
        println!("GENERATED KERNEL NAME: {}", k.name);
        // Visible payoff (run with --nocapture): confirm the FP arithmetic matches.
        println!("=== re-emitted CUDA (lift_gen.cu) ===\n{}", k.source);
    }
}
