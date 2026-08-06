//! Portable-C CPU lowering — the second backend (v1), proving the neutral
//! [`crate::plan::KernelPlan`] IR is genuinely backend-agnostic.
//!
//! [`CpuC`] emits **portable C99** (compiles + runs GPU-free) for the scalar
//! contiguous Elementwise path — a plain `void` function over a serial
//! `for (long long i = 0; i < n; ++i)` loop, `#include <math.h>` for the transcendental
//! atoms, and NONE of CUDA's `extern "C" __global__` / `blockIdx` / `threadIdx`
//! launch harness. It is a third independent leg of the correctness triangle
//! (CUDA emitter ↔ CPU-C emitter ↔ Rust oracle).
//!
//! ## What is reused vs. new (per `docs/backend-agnostic-emission-design.md`)
//!
//! The expression seam is already the correct factoring: the body math lowers
//! through the SAME language-neutral [`lower_dag`] the CUDA scalar path uses,
//! against a CpuC [`Lowering`]. The per-op spellers CUDA injects are ~95%
//! portable C99 (`powf`/`fmaxf`/`fmodf`/`atan2f`/the `Cmp*` operators/the ternary
//! select), so this backend REUSES them verbatim — [`crate::cfamily::binary_f32`] /
//! [`crate::cfamily::binary_f64`] / [`crate::cfamily::binary_int`] /
//! [`crate::cfamily::select_f32`] / [`crate::cfamily::select_f64`], plus
//! [`crate::backend::const_lit`] (`NAN`/`INFINITY`/decimal — already valid C, and
//! `<math.h>` supplies the two macros). The ONLY CUDA-specific unary atom is
//! `rsqrt` (a CUDA intrinsic, `rsqrtf`/`rsqrt`); the CpuC unary twin
//! ([`unary_f32_cpu`]/[`unary_f64_cpu`]) reuses [`crate::cfamily::unary_f32`] /
//! [`crate::cfamily::unary_f64`] for EVERY other op and overrides only `Rsqrt` to
//! `1.0f/sqrtf(x)` (f64: `1.0/sqrt(x)`). Because the reused CUDA fns are promoted
//! `pub(crate)` with their bodies untouched, every CUDA golden stays
//! byte-identical.
//!
//! ## v1 scope (honest boundary)
//!
//! - Dtypes: `F32`/`F32Strict`/`F64` + the integer compute dtypes
//!   (`I32`/`I64`/`S8`/`U8`). `F16`/`Bf16` are DECLINED (no CPU half codec yet —
//!   a documented limit; the oracle has one for a follow-up), as is the
//!   `U32` index/address dtype (mirroring [`Backend::supports_dtype`]).
//! - Schedules: only [`Schedule::Scalar`] (the scalar contiguous Elementwise
//!   path). Every other schedule (Vectorized/Strided/Reduction/…) panics clearly
//!   — AOT authoring is trusted, so a panic is the honest v1 boundary (the same
//!   convention as the CUDA emitter's unsupported arms). Because the scalar
//!   schedule is only ever chosen for a contiguous, single-output, view/gather/
//!   scatter/offset/coord-free Elementwise cell, accepting only it inherently
//!   excludes every complex case.

use crate::backend::{Backend, GeneratedKernel, Lowering, lower_dag};
use crate::cfamily::{
    assert_no_int_div_or_const, binary_f32, binary_f64, binary_int, dtype_tag, out_ctype_of,
    param_args, param_ctype, scalar_ctype, select_f32, select_f64, store_expr_of, unary_f32,
    unary_f64,
};
use crate::ir::{BinaryOp, ExprDag, UnaryOp};
use crate::plan::{KernelPlan, Schedule};
use baracuda_kernel_vocab::ElementKind;

/// The portable-C CPU backend. Lowers a [`KernelPlan`] to `.c` source.
#[derive(Copy, Clone, Debug, Default)]
pub struct CpuC;

impl Backend for CpuC {
    fn name(&self) -> &str {
        "cpu_c"
    }
    fn provider(&self) -> &str {
        // The in-tree portable-C reference backend is provided by the generator
        // itself, not by any vendor — so it reports the generator's name.
        "unpopped"
    }

    fn supports_dtype(&self, dtype: ElementKind) -> bool {
        // Portable C has a real scalar type for every compute dtype the CUDA
        // backend spells EXCEPT the halves: `F16`/`Bf16` are declined in v1 (no
        // CPU half codec yet). `U32` is an index/address dtype only (never a
        // value/key dtype — mirror the `Cuda` exclusion so a U32 elementwise add
        // cannot silently lower to an `unsigned int` kernel). Everything else
        // (`F32`/`F32Strict`/`F64`/`I32`/`I64`/`S8`/`U8`) has a plain C scalar
        // type and lowers through the portable spellers.
        !matches!(
            dtype,
            ElementKind::U32 | ElementKind::F16 | ElementKind::Bf16
        ) && scalar_ctype(dtype).is_some()
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> GeneratedKernel {
        // Dtype guard — the AOT emitter backstop beside the JIT `supports_dtype`
        // trust boundary (a declined dtype is a typed decline there; this catches
        // a direct AOT `lower` on an unlowerable dtype). F16/Bf16/U32 miss here.
        assert!(
            self.supports_dtype(plan.dtype),
            "cpu_c backend v1: unsupported dtype {:?} — f16/bf16 are declined (no CPU half \
             codec yet); f32/f64 + integer compute dtypes only",
            plan.dtype
        );
        let ctype =
            scalar_ctype(plan.dtype).expect("supports_dtype gate guarantees a scalar ctype");
        // v1 mirrors the single-store `emit_scalar`. A MULTI-output elementwise op
        // can also key `Schedule::Scalar` (the CUDA `emit_scalar_multi` path); the
        // N-store CPU emitter is a follow-up, so reject it honestly here.
        assert!(
            plan.n_outputs == 1,
            "cpu_c backend v1: single-output only (the N-store multi-output emitter is a \
             follow-up); op '{}' has {} outputs",
            plan.op_name,
            plan.n_outputs
        );
        // Independent int Div/Const backstop, REUSED from the CUDA emitter (same
        // dtype-blind hazard: a `Const` is spelled as an f64 literal, infix `Div`
        // is `/` — both device/host dangerous at an integer dtype). The plan gate
        // (`assert_int_op_admissibility`) rejects these upstream; this is the
        // gate-every-layer backstop, identical coverage to `Cuda::lower`.
        if crate::plan::is_int_dtype(plan.dtype) {
            // v1 is Elementwise/Scalar-only (panics below on any other
            // schedule), so the reduction-predicate exemption never applies
            // here — always `false` for both flags (inert; `in_reduction`
            // false already excludes the exemption regardless of
            // `at_reduction_root`), same coverage as before Task 3b.
            assert_no_int_div_or_const(plan.body, plan.dtype, false, false);
        }
        match plan.schedule {
            Schedule::Scalar => emit_scalar_cpu(plan, ctype),
            other => panic!(
                "cpu_c backend v1: Elementwise (the scalar contiguous path) ONLY — got \
                 schedule {other:?}. Vectorized / Strided / Reduction / RowReduce / \
                 Contraction / Scan / Window / RowSort / Im2Col are follow-ups; the scalar \
                 schedule is chosen precisely for the contiguous, single-output, non-strided \
                 Elementwise cell this v1 serves."
            ),
        }
    }
}

/// The scalar contiguous Elementwise emitter — the CPU twin of
/// `baracuda_cuda_emit::cuda`'s `emit_scalar`. Same body math ([`lower_dag`] over the shared
/// seam), but a plain `void` signature + `#include <math.h>` + a SERIAL
/// `for (long long i = 0; i < n; ++i)` loop instead of the `extern "C" __global__`
/// header and the grid-stride prologue.
fn emit_scalar_cpu(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let name = format!("unpopped_cpu_{}_{}", plan.op_name, dtype_tag(plan.dtype));
    let n = plan.n_inputs;
    let octype = out_ctype_of(plan, 0, ctype);
    let mut s = String::new();
    s.push_str("// Generated by baracuda-kernelgen (cpu_c backend) — do not edit.\n");
    s.push_str(&format!(
        "// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    ));
    // Portable C99: `<math.h>` supplies the transcendental atoms (expf/sqrtf/…),
    // the `Cmp`/`Rem` helpers, and the `NAN`/`INFINITY` macros `const_lit` emits.
    s.push_str("#include <math.h>\n\n");
    s.push_str(&format!("void {name}(\n"));
    for i in 0..n {
        s.push_str(&format!("    const {ctype}* in{i},\n"));
    }
    s.push_str(&format!("    {octype}* out,\n"));
    s.push_str(&format!(
        "    long long n{})\n{{\n",
        param_args(plan.body, param_ctype(plan))
    ));
    let acc = |idx: u8| format!("in{idx}[i]");
    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(plan.body),
        ctype,
        &Lowering {
            leaf: &acc,
            reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
            coord: &|d| {
                panic!(
                    "cpu_c backend: Coord({d}) reached the scalar emitter — Coord bodies \
                     lower via Strided only (the linear-index loop has no per-axis \
                     coordinates)"
                )
            },
            unary: &|op, x| cpu_unary(op, x, plan.dtype),
            binary: &|op, a, b| cpu_binary(op, a, b, plan.dtype),
            select: &|c, a, b| cpu_select(c, a, b, plan.dtype),
        },
    );
    let store = store_expr_of(plan, 0, root);
    if prelude.is_empty() {
        s.push_str(&format!(
            "    for (long long i = 0; i < n; ++i) out[i] = {store};\n"
        ));
    } else {
        // Shared interiors: hoist the `tmp` block inside the loop (its RHS reads
        // the per-`i` inputs), so a shared value is computed once per element —
        // the same placement as the CUDA scalar path.
        s.push_str("    for (long long i = 0; i < n; ++i) {\n");
        for decl in &prelude {
            s.push_str(&format!("        {decl}\n"));
        }
        s.push_str(&format!("        out[i] = {store};\n    }}\n"));
    }
    s.push_str("}\n");
    GeneratedKernel { name, source: s }
}

/// Lower a unary op for `dtype` on the CPU — the twin of `baracuda_cuda_emit::cuda`'s
/// `cuda_unary`, minus the f16/bf16 promote arms (declined in v1). f32/f64 route
/// to the CpuC unary spellers ([`unary_f32_cpu`]/[`unary_f64_cpu`]); an integer
/// dtype has no unary math (the ir admissibility table rejects it), so the panic
/// is the emitter backstop.
fn cpu_unary(op: UnaryOp, x: String, dtype: ElementKind) -> String {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => unary_f32_cpu(op, x),
        ElementKind::F64 => unary_f64_cpu(op, x),
        other => panic!(
            "cpu_c backend: no unary math for dtype {other:?} — v1 lowers unary for f32/f64 \
             only (integer dtypes have no unary math; f16/bf16 are declined)"
        ),
    }
}

/// The f32 unary twin: identical to [`crate::cfamily::unary_f32`] for EVERY op
/// except `Rsqrt`, which the CUDA path spells as the intrinsic `rsqrtf(x)` — not
/// portable C — so it is respelled `1.0f/sqrtf(x)` (the one genuinely new atom).
/// Every other arm (`expf`/`sqrtf`/`fabsf`/`erff`/…) is a C99 `<math.h>` function
/// reused verbatim.
fn unary_f32_cpu(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Rsqrt => format!("(1.0f/sqrtf({x}))"),
        _ => unary_f32(op, x),
    }
}

/// The f64 unary twin: [`crate::cfamily::unary_f64`] for every op except `Rsqrt`,
/// respelled `1.0/sqrt(x)` (the CUDA intrinsic `rsqrt(x)` is not portable C).
fn unary_f64_cpu(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Rsqrt => format!("(1.0/sqrt({x}))"),
        _ => unary_f64(op, x),
    }
}

/// Lower a non-infix binary op for `dtype` on the CPU — REUSES the CUDA spellers
/// verbatim (`powf`/`atan2f`/`copysignf`/`fmaxf`/`fmodf`/the `Cmp*` operators are
/// all C99, and the int-op speller is raw C operators). Mirrors the shape of
/// `cuda_binary` minus the f16/bf16 promote arms (declined in v1).
fn cpu_binary(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => binary_f32(op, a, b),
        ElementKind::F64 => binary_f64(op, a, b),
        ElementKind::I32 | ElementKind::I64 | ElementKind::S8 | ElementKind::U8 => {
            binary_int(op, a, b, dtype)
        }
        other => panic!(
            "cpu_c backend: no binary math for dtype {other:?} — f16/bf16/u32 are declined in v1"
        ),
    }
}

/// Lower a ternary select for `dtype` on the CPU — REUSES the CUDA
/// identity-cast-pinned C ternary spellers ([`select_f32`]/[`select_f64`]), which
/// are portable C as-is. v1 select is float-only (an int select raises the
/// unresolved cond-observer question), so an integer dtype backstop-panics.
fn cpu_select(c: String, a: String, b: String, dtype: ElementKind) -> String {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => select_f32(c, a, b),
        ElementKind::F64 => select_f64(c, a, b),
        other => panic!(
            "cpu_c backend: Select has no {other:?} lowering — v1 select is float-only (f32/f64)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate;
    use crate::ir::{OpDef, input};
    use baracuda_kernel_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};

    /// A binary Elementwise cell whose small `align` defeats vectorization, so
    /// `build_plan` derives [`Schedule::Scalar`] (the path CpuC v1 serves) — the
    /// exact `binary_scalar_key` device-test precedent.
    fn binary_scalar_key(dt: ElementKind, align: u32) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    /// A unary Elementwise scalar cell (align defeats vectorization).
    fn unary_scalar_key(dt: ElementKind, align: u32) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    #[test]
    fn f32_elementwise_emits_portable_serial_c_no_cuda_tokens() {
        // The brief's canonical cell: input(0)*input(1) + input(0).
        let op = OpDef::elementwise(
            "affine",
            2,
            &[ElementKind::F32],
            input(0) * input(1) + input(0),
        );
        let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &CpuC);
        let src = &k.source;
        // Exported symbol distinct from the CUDA `baracuda_gen_*`.
        assert_eq!(k.name, "unpopped_cpu_affine_f32");
        // Portable C structure.
        assert!(
            src.contains("#include <math.h>"),
            "missing math.h in:\n{src}"
        );
        assert!(
            src.contains("void unpopped_cpu_affine_f32("),
            "missing plain void signature in:\n{src}"
        );
        assert!(
            src.contains("const float* in0,"),
            "missing plain (non-__restrict__) input pointer in:\n{src}"
        );
        assert!(
            src.contains("float* out,"),
            "missing output pointer in:\n{src}"
        );
        assert!(src.contains("long long n)"), "missing n arg in:\n{src}");
        // The SERIAL loop + store (not a grid-stride kernel).
        assert!(
            src.contains("for (long long i = 0; i < n; ++i)"),
            "missing serial loop in:\n{src}"
        );
        assert!(src.contains("out[i] ="), "missing store in:\n{src}");
        // The infix body — byte-identical to the neutral driver's output.
        assert!(
            src.contains("out[i] = ((in0[i] * in1[i]) + in0[i]);"),
            "missing infix body in:\n{src}"
        );
        // NONE of the CUDA launch harness tokens.
        for tok in [
            "blockIdx",
            "threadIdx",
            "__global__",
            "__restrict__",
            "gridDim",
        ] {
            assert!(!src.contains(tok), "CUDA token `{tok}` leaked into:\n{src}");
        }
    }

    #[test]
    fn rsqrt_spells_portable_one_over_sqrt_not_the_cuda_intrinsic() {
        use crate::ir::UnaryOp;
        // f32: 1.0f/sqrtf, never rsqrtf.
        let op32 = OpDef::elementwise("rs", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Rsqrt));
        let k32 = generate(&op32, &unary_scalar_key(ElementKind::F32, 4), &CpuC);
        assert!(
            k32.source.contains("out[i] = (1.0f/sqrtf(in0[i]));"),
            "f32 rsqrt not respelled 1.0f/sqrtf in:\n{}",
            k32.source
        );
        assert!(
            !k32.source.contains("rsqrtf"),
            "CUDA rsqrtf intrinsic leaked in:\n{}",
            k32.source
        );
        // f64: 1.0/sqrt, never rsqrt.
        let op64 = OpDef::elementwise("rs", 1, &[ElementKind::F64], input(0).unary(UnaryOp::Rsqrt));
        let k64 = generate(&op64, &unary_scalar_key(ElementKind::F64, 8), &CpuC);
        assert!(
            k64.source.contains("out[i] = (1.0/sqrt(in0[i]));"),
            "f64 rsqrt not respelled 1.0/sqrt in:\n{}",
            k64.source
        );
        assert!(
            !k64.source.contains("rsqrt("),
            "CUDA rsqrt intrinsic leaked in:\n{}",
            k64.source
        );
    }

    #[test]
    fn reused_math_fn_stays_portable_c99() {
        use crate::ir::BinaryOp;
        // A reused speller (powf) rides straight through — proving the CUDA
        // spellers are portable C and are shared, not re-implemented.
        let op = OpDef::elementwise(
            "p",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::Pow, input(1)),
        );
        let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &CpuC);
        assert!(
            k.source.contains("out[i] = powf(in0[i], in1[i]);"),
            "reused powf speller missing in:\n{}",
            k.source
        );
    }

    #[test]
    fn declines_f16_via_supports_dtype() {
        // The documented v1 decline: no CPU half codec yet.
        assert!(!CpuC.supports_dtype(ElementKind::F16));
        assert!(!CpuC.supports_dtype(ElementKind::Bf16));
        // U32 is an index/address dtype only (mirror Cuda).
        assert!(!CpuC.supports_dtype(ElementKind::U32));
        // The real compute dtypes are supported.
        for dt in [
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::I32,
            ElementKind::I64,
            ElementKind::S8,
            ElementKind::U8,
        ] {
            assert!(CpuC.supports_dtype(dt), "{dt:?} should be supported");
        }
    }

    #[test]
    #[should_panic(expected = "scalar contiguous path")]
    fn declines_non_scalar_schedule() {
        // A V4-aligned contiguous binary f32 op keys Schedule::Vectorized, which
        // CpuC v1 does not serve — it must panic clearly, not mis-emit.
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        let _ = generate(&op, &binary_scalar_key(ElementKind::F32, 256), &CpuC);
    }
}
