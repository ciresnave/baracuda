//! CUDA-idiom golden vectors for [`baracuda_cuda_parse::CUDA`].
//!
//! Every test here reproduces, at full fidelity (same source string, same
//! equality assertion), one of the CUDA-only tests that lived in
//! `unpopped::convert`'s private `mod tests` before the donation. Checked, not
//! assumed, that full fidelity is possible: every type/field these assertions
//! touch (`Lifted{op,n_inputs}`, `OpDef{body,access}`,
//! `Access::{Reduction,Scan}`'s fields) is `pub`, reachable through
//! `unpopped::convert`/`unpopped::ir` from outside the crate -- proven not
//! merely by inspection but by this file compiling and passing.
//!
//! `grammars_load_and_parse`'s CUDA half moves here too (`grammars_load_and_parse`
//! itself asserted both `parse_cuda` and `parse_slang`, so it splits rather
//! than moves whole -- the `parse_slang` half stays in Unpopped).
//!
//! **Not a correctness proof for CUDA parsing in general** -- see the crate's
//! module doc. These are CUDA-idiom golden vectors.

use unpopped::convert::{lift_elementwise, lift_reduction, lift_scan};
use unpopped::ir::{Access, BinaryOp, ReduceOp, ScalarExpr, UnaryOp};
use unpopped::lift::{Lifted, LiftError};
use unpopped_vocab::ElementKind;

use baracuda_cuda_parse::{CUDA, parse_cuda};

const F32: &[ElementKind] = &[ElementKind::F32];

fn cuda_body(src: &str) -> ScalarExpr {
    lift_elementwise(&CUDA, src, "x", F32).unwrap().op.body
}

fn reduce_op(l: &Lifted) -> Option<ReduceOp> {
    if let Access::Reduction { op, .. } = &l.op.access {
        Some(*op)
    } else {
        None
    }
}

fn scan_info(l: &Lifted) -> Option<(ReduceOp, &ScalarExpr, bool, bool)> {
    if let Access::Scan {
        op,
        pre,
        reverse,
        exclusive,
        ..
    } = &l.op.access
    {
        Some((*op, pre, *reverse, *exclusive))
    } else {
        None
    }
}

/// CUDA half of `unpopped::convert`'s former `grammars_load_and_parse` --
/// the `parse_slang` half stayed in Unpopped.
#[test]
fn cuda_grammar_loads_and_parses() {
    assert!(parse_cuda("__global__ void k(float* out){ out[i]=0.0f; }").is_some());
}

#[test]
fn cuda_lifts_fused_multiply_add() {
    let src = "__global__ void mul(const float* in0, const float* in1, const float* in2, float* out, long long n) {\n\
        long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
        for (; i < n; i += gridDim.x*blockDim.x) { out[i] = in0[i] * in1[i] + in2[i]; }\n}";
    let lifted = lift_elementwise(&CUDA, src, "fma", F32).unwrap();
    assert_eq!(lifted.n_inputs, 3);
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
fn cuda_lifts_unary_intrinsic() {
    let src = "__global__ void k(const float* in0, float* out, long long n){ out[i] = __expf(in0[i]); }";
    assert_eq!(
        cuda_body(src),
        ScalarExpr::Unary(UnaryOp::Exp, Box::new(ScalarExpr::Input(0)))
    );
}

#[test]
fn cuda_fmaxf_is_ieee_not_torch_max() {
    let src = "__global__ void k(const float* in0, float* out, long long n){ out[i] = fmaxf(in0[i], 0.0f); }";
    assert_eq!(
        cuda_body(src),
        ScalarExpr::Binary(
            BinaryOp::FmaxIeee,
            Box::new(ScalarExpr::Input(0)),
            Box::new(ScalarExpr::Const(0.0)),
        )
    );
}

#[test]
fn cuda_precedence_from_the_grammar() {
    let src = "__global__ void k(const float* in0, const float* in1, const float* in2, float* out){ out[i] = in0[i] + in1[i]*in2[i]; }";
    assert_eq!(
        cuda_body(src),
        ScalarExpr::Add(
            Box::new(ScalarExpr::Input(0)),
            Box::new(ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(1)),
                Box::new(ScalarExpr::Input(2)),
            )),
        )
    );
}

/// Both of these must be refused, but for *different* reasons -- `Unrecognized`
/// ("ask for this op to be added") vs `Inexpressible` ("leave this fragment in
/// the source language"). Collapsing them to "some error" would let a genuine
/// recognizer gap be mistaken for a hard IR limit.
#[test]
fn cuda_refuses_residue() {
    let neigh = "__global__ void k(const float* in0, float* out){ out[i] = in0[i+1]; }";
    assert!(matches!(
        lift_elementwise(&CUDA, neigh, "x", F32),
        Err(LiftError::Unrecognized(_))
    ));
    let smem = "__global__ void k(float* out){ __shared__ float s[32]; out[i] = s[0]; }";
    assert!(
        matches!(
            lift_elementwise(&CUDA, smem, "x", F32),
            Err(LiftError::Inexpressible(ref w)) if w == "__shared__"
        ),
        "got {:?}",
        lift_elementwise(&CUDA, smem, "x", F32)
    );
}

#[test]
fn cuda_lifts_sum_reduction() {
    let src = "__global__ void sum(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc += in0[i]; out[0] = acc; }";
    let lifted = lift_reduction(&CUDA, src, "sum", F32).unwrap();
    assert_eq!(lifted.n_inputs, 1);
    assert_eq!(lifted.op.body, ScalarExpr::Input(0));
    assert_eq!(reduce_op(&lifted), Some(ReduceOp::Sum));
}

#[test]
fn cuda_lifts_sum_of_squares() {
    let src = "__global__ void ss(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc += in0[i]*in0[i]; out[0] = acc; }";
    let lifted = lift_reduction(&CUDA, src, "ss", F32).unwrap();
    assert_eq!(
        lifted.op.body,
        ScalarExpr::Mul(
            Box::new(ScalarExpr::Input(0)),
            Box::new(ScalarExpr::Input(0))
        )
    );
    assert_eq!(reduce_op(&lifted), Some(ReduceOp::Sum));
}

#[test]
fn cuda_lifts_prod_and_acc_plus_forms() {
    let prod = "__global__ void pr(const float* in0, float* out, long long n){ float acc = 1.0f; for (long long i=0;i<n;i++) acc *= in0[i]; out[0] = acc; }";
    assert_eq!(
        reduce_op(&lift_reduction(&CUDA, prod, "pr", F32).unwrap()),
        Some(ReduceOp::Prod)
    );
    let sum2 = "__global__ void s2(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc = acc + in0[i]; out[0] = acc; }";
    assert_eq!(
        reduce_op(&lift_reduction(&CUDA, sum2, "s2", F32).unwrap()),
        Some(ReduceOp::Sum)
    );
}

#[test]
fn cuda_lifts_max_via_fmaxf() {
    let src = "__global__ void mx(const float* in0, float* out, long long n){ float acc = in0[0]; for (long long i=1;i<n;i++) acc = fmaxf(acc, in0[i]); out[0] = acc; }";
    let lifted = lift_reduction(&CUDA, src, "mx", F32).unwrap();
    assert_eq!(lifted.op.body, ScalarExpr::Input(0));
    assert_eq!(reduce_op(&lifted), Some(ReduceOp::Max));
}

#[test]
fn cuda_lifts_cumsum() {
    let src = "__global__ void cs(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++){ acc += in0[i]; out[i] = acc; } }";
    let lifted = lift_scan(&CUDA, src, "cs", F32).unwrap();
    assert_eq!(lifted.n_inputs, 1);
    let (op, pre, rev, exc) = scan_info(&lifted).unwrap();
    assert_eq!(op, ReduceOp::Sum);
    assert_eq!(pre, &ScalarExpr::Input(0));
    assert!(!rev && !exc);
}

#[test]
fn cuda_lifts_cumprod_and_cummax() {
    let cp = "__global__ void cp(const float* in0, float* out, long long n){ float acc = 1.0f; for (long long i=0;i<n;i++){ acc *= in0[i]; out[i] = acc; } }";
    assert_eq!(
        scan_info(&lift_scan(&CUDA, cp, "cp", F32).unwrap())
            .unwrap()
            .0,
        ReduceOp::Prod
    );
    let cm = "__global__ void cm(const float* in0, float* out, long long n){ float acc = in0[0]; for (long long i=0;i<n;i++){ acc = fmaxf(acc, in0[i]); out[i] = acc; } }";
    assert_eq!(
        scan_info(&lift_scan(&CUDA, cm, "cm", F32).unwrap())
            .unwrap()
            .0,
        ReduceOp::Max
    );
}
