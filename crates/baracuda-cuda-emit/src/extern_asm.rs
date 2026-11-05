//! Spells one `cuda-ptx` inline-asm body of an Unpopped [`ExternDecl`] as a CUDA statement (ROADMAP U11 step 3;
//! `docs/u11-ptx-leg-design.md`). Behind `--features extern_asm_cuda`, default off; it changes no other emitted output.
//!
//! **Unexercised on a device until `tests/extern_ptx.rs` has run on the runner.** Only the operand spelling is unit-tested.

use std::fmt;

use unpopped::kernel::{BodyText, ExternDecl};

/// The `TargetBody::target` key this module reads.
pub const CUDA_PTX_TARGET: &str = "cuda-ptx";

/// Why an [`ExternDecl`] could not be spelled. Every variant names the declaration; none falls back to the `reference`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExternError {
    /// The declaration has no `cuda-ptx` body.
    NoBody {
        /// The declaration's name.
        decl: String,
    },
    /// The `cuda-ptx` body is a `Symbol`; only inline asm is spelled in v1.
    NotAsm {
        /// The declaration's name.
        decl: String,
    },
    /// The declaration uses an effect v1 cannot spell (memory spaces, a rendezvous).
    UnsupportedEffect {
        /// The declaration's name.
        decl: String,
    },
    /// The constraint string does not match the signature (`=`-prefixed outputs first, then one input per parameter).
    BadConstraints {
        /// The declaration's name.
        decl: String,
        /// What was wrong.
        why: String,
    },
    /// The caller passed the wrong number of argument or result expressions.
    BadOperands {
        /// The declaration's name.
        decl: String,
        /// What was wrong.
        why: String,
    },
    /// The template names an operand `{N}` that does not exist.
    BadTemplate {
        /// The declaration's name.
        decl: String,
        /// What was wrong.
        why: String,
    },
}

impl fmt::Display for ExternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBody { decl } => write!(f, "extern `{decl}`: no `{CUDA_PTX_TARGET}` body"),
            Self::NotAsm { decl } => write!(
                f,
                "extern `{decl}`: the `{CUDA_PTX_TARGET}` body is not inline asm"
            ),
            Self::UnsupportedEffect { decl } => write!(
                f,
                "extern `{decl}`: memory or sync effects are not spelled in v1"
            ),
            Self::BadConstraints { decl, why } => write!(f, "extern `{decl}`: constraints: {why}"),
            Self::BadOperands { decl, why } => write!(f, "extern `{decl}`: operands: {why}"),
            Self::BadTemplate { decl, why } => write!(f, "extern `{decl}`: template: {why}"),
        }
    }
}

impl std::error::Error for ExternError {}

/// Splits `constraints` into (output constraints without `=`, input constraints), outputs first.
fn split_constraints(
    decl: &ExternDecl,
    constraints: &str,
) -> Result<(Vec<String>, Vec<String>), ExternError> {
    let bad = |why: String| ExternError::BadConstraints {
        decl: decl.name.clone(),
        why,
    };
    let parts: Vec<&str> = if constraints.trim().is_empty() {
        Vec::new()
    } else {
        constraints.split(',').map(str::trim).collect()
    };
    let outs = parts.iter().take_while(|p| p.starts_with('=')).count();
    if let Some(p) = parts[outs..]
        .iter()
        .find(|p| p.starts_with('=') || p.starts_with('+') || p.is_empty())
    {
        return Err(bad(format!(
            "`{p}` is not an input constraint (outputs `=x` must come first)"
        )));
    }
    if outs != decl.results.len() || parts.len() - outs != decl.params.len() {
        return Err(bad(format!(
            "{outs} outputs and {} inputs, but the declaration has {} results and {} params",
            parts.len() - outs,
            decl.results.len(),
            decl.params.len()
        )));
    }
    Ok((
        parts[..outs].iter().map(|p| p[1..].to_string()).collect(),
        parts[outs..].iter().map(|p| (*p).to_string()).collect(),
    ))
}

/// The `{N}` operand at the start of `rest`: its index and the byte length it spans.
fn leading_operand(rest: &str) -> Option<(usize, usize)> {
    let end = rest.strip_prefix('{').and(rest.find('}'))?;
    let inner = &rest[1..end];
    if inner.is_empty() || !inner.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    inner.parse().ok().map(|n| (n, end + 1))
}

/// CUDA's operand index for Unpopped's `{n}` (params first, then results) when outputs come first.
fn cuda_operand(decl: &ExternDecl, n: usize) -> Result<usize, ExternError> {
    let (params, results) = (decl.params.len(), decl.results.len());
    if n < params {
        Ok(results + n)
    } else if n < params + results {
        Ok(n - params)
    } else {
        Err(ExternError::BadTemplate {
            decl: decl.name.clone(),
            why: format!(
                "operand `{{{n}}}` but the declaration has {params} params and {results} results"
            ),
        })
    }
}

/// Rewrites Unpopped's operand numbering (`{i}` params first, then results) to CUDA's (`%j` outputs first), and escapes
/// the text for a C string literal. `%` doubles; `{` not followed by digits and `}` is left as it is (PTX scope braces).
fn renumber(decl: &ExternDecl, template: &str) -> Result<String, ExternError> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(c) = rest.chars().next() {
        if let Some((n, len)) = leading_operand(rest) {
            out.push_str(&format!("%{}", cuda_operand(decl, n)?));
            rest = &rest[len..];
            continue;
        }
        match c {
            '%' => out.push_str("%%"),
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n\\t"),
            '\r' => {}
            _ => out.push(c),
        }
        rest = &rest[c.len_utf8()..];
    }
    Ok(out)
}

/// Spells the `cuda-ptx` asm body of `decl` as one CUDA statement. `args` are the C expressions of the parameters and
/// `results` the C lvalues of the results. Plain `asm` for a `pure` declaration that is not `volatile`, else `asm volatile`.
///
/// Example: `popc.b32` (params `[U32]`, results `[U32]`, template `popc.b32 {1}, {0};`, constraints `=r,r`) with
/// `args = ["x"]`, `results = ["r"]` gives `asm volatile("popc.b32 %0, %1;" : "=r"(r) : "r"(x));`.
///
/// Refuses, naming the declaration, when there is no `cuda-ptx` body, the body is a symbol, the effects use memory or a
/// rendezvous, or the constraints, operands or template do not match the signature. It never uses the `reference`.
///
/// The asm body is unexercised on a device until `tests/extern_ptx.rs` has run on the runner.
///
/// # Errors
/// An [`ExternError`] per the above.
pub fn extern_asm_cuda(
    decl: &ExternDecl,
    args: &[String],
    results: &[String],
) -> Result<String, ExternError> {
    let body = decl
        .bodies
        .iter()
        .find(|b| b.target == CUDA_PTX_TARGET)
        .ok_or_else(|| ExternError::NoBody {
            decl: decl.name.clone(),
        })?;
    let BodyText::Asm {
        template,
        constraints,
    } = &body.body
    else {
        return Err(ExternError::NotAsm {
            decl: decl.name.clone(),
        });
    };
    if !decl.effects.reads.is_empty()
        || !decl.effects.writes.is_empty()
        || decl.effects.sync.is_some()
    {
        return Err(ExternError::UnsupportedEffect {
            decl: decl.name.clone(),
        });
    }
    if args.len() != decl.params.len() || results.len() != decl.results.len() {
        return Err(ExternError::BadOperands {
            decl: decl.name.clone(),
            why: format!(
                "{} args and {} results given, {} params and {} results declared",
                args.len(),
                results.len(),
                decl.params.len(),
                decl.results.len()
            ),
        });
    }
    let (outs, ins) = split_constraints(decl, constraints)?;
    let text = renumber(decl, template)?;
    let list = |cs: &[String], xs: &[String]| {
        cs.iter()
            .zip(xs)
            .map(|(c, x)| format!("\"{c}\"({x})"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let kw = if decl.effects.volatile || !decl.effects.pure {
        "asm volatile"
    } else {
        "asm"
    };
    let outs = list(
        &outs.iter().map(|c| format!("={c}")).collect::<Vec<_>>(),
        results,
    );
    Ok(format!("{kw}(\"{text}\" : {outs} : {});", list(&ins, args)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use unpopped::kernel::{Effects, TargetBody};
    use unpopped_vocab::ElementKind;

    fn popc() -> ExternDecl {
        ExternDecl::new(
            "popc",
            vec![ElementKind::U32],
            vec![ElementKind::U32],
            Effects::opaque(),
        )
        .with_body(TargetBody::new(
            "cuda-ptx",
            BodyText::Asm {
                template: "popc.b32 {1}, {0};".into(),
                constraints: "=r,r".into(),
            },
        ))
    }
    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn popc_renumbers_params_after_results() {
        assert_eq!(
            extern_asm_cuda(&popc(), &s(&["x"]), &s(&["r"])).unwrap(),
            r#"asm volatile("popc.b32 %0, %1;" : "=r"(r) : "r"(x));"#
        );
    }

    #[test]
    fn fma_three_params_one_result() {
        let d = ExternDecl::new(
            "fma",
            vec![ElementKind::F32; 3],
            vec![ElementKind::F32],
            Effects::pure(),
        )
        .with_body(TargetBody::new(
            "cuda-ptx",
            BodyText::Asm {
                template: "fma.rn.f32 {3}, {0}, {1}, {2};".into(),
                constraints: "=f,f,f,f".into(),
            },
        ));
        assert_eq!(
            extern_asm_cuda(&d, &s(&["a", "b", "c"]), &s(&["r"])).unwrap(),
            r#"asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));"#
        );
    }

    #[test]
    fn two_results_keep_their_order() {
        let d = ExternDecl::new(
            "d",
            vec![ElementKind::U32],
            vec![ElementKind::U32; 2],
            Effects::opaque(),
        )
        .with_body(TargetBody::new(
            "cuda-ptx",
            BodyText::Asm {
                template: "op {1}, {2}, {0};".into(),
                constraints: "=r,=r,r".into(),
            },
        ));
        assert_eq!(
            extern_asm_cuda(&d, &s(&["x"]), &s(&["a", "b"])).unwrap(),
            r#"asm volatile("op %0, %1, %2;" : "=r"(a), "=r"(b) : "r"(x));"#
        );
    }

    #[test]
    fn percent_quote_and_scope_braces_survive() {
        let d = ExternDecl::new("m", vec![], vec![ElementKind::U32], Effects::opaque()).with_body(
            TargetBody::new(
                "cuda-ptx",
                BodyText::Asm {
                    template: "{ mov.u32 {0}, %tid.x; }".into(),
                    constraints: "=r".into(),
                },
            ),
        );
        assert_eq!(
            extern_asm_cuda(&d, &[], &s(&["r"])).unwrap(),
            r#"asm volatile("{ mov.u32 %0, %%tid.x; }" : "=r"(r) : );"#
        );
    }

    #[test]
    fn pure_decl_is_plain_asm_volatile_decl_is_volatile() {
        let mut d = popc();
        d.effects = Effects::pure();
        assert!(
            extern_asm_cuda(&d, &s(&["x"]), &s(&["r"]))
                .unwrap()
                .starts_with("asm(")
        );
        d.effects.volatile = true;
        assert!(
            extern_asm_cuda(&d, &s(&["x"]), &s(&["r"]))
                .unwrap()
                .starts_with("asm volatile(")
        );
    }

    #[test]
    fn refusals_name_the_declaration() {
        let (a, r) = (s(&["x"]), s(&["r"]));
        let mut d = popc();
        d.bodies[0].target = "spirv".into();
        assert!(
            matches!(extern_asm_cuda(&d, &a, &r), Err(ExternError::NoBody { decl }) if decl == "popc")
        );
        let mut d = popc();
        d.bodies[0].body = BodyText::Symbol {
            library: "l".into(),
            symbol: "s".into(),
        };
        assert!(matches!(
            extern_asm_cuda(&d, &a, &r),
            Err(ExternError::NotAsm { .. })
        ));
        let mut d = popc();
        d.effects = Effects::opaque().with_sync(unpopped::exec::Scope::Block);
        assert!(matches!(
            extern_asm_cuda(&d, &a, &r),
            Err(ExternError::UnsupportedEffect { .. })
        ));
        assert!(matches!(
            extern_asm_cuda(&popc(), &[], &r),
            Err(ExternError::BadOperands { .. })
        ));
        assert!(matches!(
            extern_asm_cuda(&popc(), &a, &[]),
            Err(ExternError::BadOperands { .. })
        ));
        for bad in ["=r", "r,=r", "=r,r,r", "=r,+r", "=r,"] {
            let mut d = popc();
            d.bodies[0].body = BodyText::Asm {
                template: "popc.b32 {1}, {0};".into(),
                constraints: bad.into(),
            };
            assert!(
                matches!(
                    extern_asm_cuda(&d, &a, &r),
                    Err(ExternError::BadConstraints { .. })
                ),
                "{bad}"
            );
        }
        let mut d = popc();
        d.bodies[0].body = BodyText::Asm {
            template: "popc.b32 {2}, {0};".into(),
            constraints: "=r,r".into(),
        };
        let e = extern_asm_cuda(&d, &a, &r).unwrap_err();
        assert!(matches!(e, ExternError::BadTemplate { .. }));
        assert!(e.to_string().contains("popc"));
    }
}
