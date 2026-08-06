//! Baracuda's live JIT synthesizer for the Fuel kernel seam — the
//! `fuel_kernel_seam::Synthesizer` implementation (`BaracudaSynthesizer`) Fuel
//! calls (§5). Carved out of `baracuda-kernelgen`'s `jit.rs` (carve step 2): it
//! wires the CUDA backend ([`crate::Cuda`]) + the NVRTC compiler into the neutral
//! generator's backend-agnostic seam front-end
//! (`baracuda_kernelgen::jit::seam::synthesize`). Behind `--features seam`.

#[cfg(feature = "nvrtc")]
use crate::nvrtc::NvrtcCompiler;
use baracuda_kernelgen::ArtifactKind;
use baracuda_kernel_vocab::{OpCategory, OperandDesc};
use fuel_kernel_seam_types::PatternNode as SeamNode;

// ===== The live §5 call — the `fuel_kernel_seam::Synthesizer` Fuel invokes =====

use crate::Cuda;
// Alias the envelope types: their bare `JitRequest`/`JitResponse` names would
// shadow our own internal `jit::{JitRequest, JitResponse}` (glob-imported via
// `super::*`) across this whole module and silently retype the native core.
use fuel_kernel_seam::{
    ArtifactKind as SeamArtifactKind, JitRequest as SeamRequest, JitResponse as SeamResponse,
    LinkEntry as SeamLinkEntry, SynthArtifact as SeamArtifact, Synthesizer,
};
use std::collections::HashMap;
use std::sync::Mutex;

// The retained artifact IS the envelope [`SeamArtifact`]
// (`fuel_kernel_seam::SynthArtifact`) — Fuel stays type-decoupled (its own Q1
// invariant), so `take_kernel` hands back Fuel's type, not a Baracuda one. Built
// at the trait boundary in `synthesize` from the native response; the debug-only
// `.cu` source and the `recipe` are dropped (the re-fuse `pattern:` rides in the
// `contract`, and `decompose` is the `JitRequest.region` Fuel holds — both
// reconstructable, confirmed frozen 2026-07-04).

/// Baracuda's live JIT synthesizer — the [`Synthesizer`] Fuel calls (§5). Each
/// [`Synthesizer::synthesize`] adapts the envelope [`JitRequest`] to the native
/// [`synthesize`] core, returns a [`JitResponse`], and retains the compiled
/// artifact under its `entry_point` for the seam-call site to fetch
/// ([`Self::take_kernel`]) and bind (load PTX → `KernelRef` → Fuel's
/// `adopt_runtime_fused`). **Never panics** (the trait's contract): an
/// unbuildable / out-of-vocabulary / over-budget region is a typed
/// [`JitResponse::Declined`], not an error or a crash across the boundary.
#[derive(Debug)]
pub struct BaracudaSynthesizer {
    /// A hard ceiling on the per-request compile budget (ms). Each request also
    /// carries its own `budget.max_compile_ms`; the effective budget is the min.
    max_compile_ms: u32,
    registry: Mutex<HashMap<String, SeamArtifact>>,
}

impl BaracudaSynthesizer {
    /// A synthesizer with the given on-demand compile budget ceiling (ms, `> 0`).
    #[must_use]
    pub fn new(max_compile_ms: u32) -> Self {
        Self {
            max_compile_ms,
            registry: Mutex::new(HashMap::new()),
        }
    }
}

impl Synthesizer for BaracudaSynthesizer {
    fn synthesize(&self, req: &SeamRequest) -> SeamResponse {
        // The on-demand compiler: real nvrtc when compiled in, else a stub (a
        // Stub artifact a loader must refuse — keeps the endpoint callable for
        // wiring/tests without a CUDA toolchain).
        #[cfg(feature = "nvrtc")]
        let compiler = NvrtcCompiler::new(req.arch);
        #[cfg(not(feature = "nvrtc"))]
        let compiler = baracuda_kernelgen::StubCompiler;

        let n_inputs = req.operands.len().saturating_sub(1);
        // The envelope carries no op category; elementwise schedule is layout-
        // driven, so the category is a key tag — derive it from operand arity
        // (clamped at ternary for 4+-input fused regions).
        let op_category = match n_inputs {
            0 | 1 => OpCategory::UnaryElementwise,
            2 => OpCategory::BinaryElementwise,
            _ => OpCategory::TernaryElementwise,
        };
        let fused_op_id = region_op_id(&req.region, &req.operands);
        let _ = n_inputs; // arity feeds op_category above; no longer a cost input

        // The request budget is authoritative (§5.2 moved it onto JitRequest); the
        // synthesizer's own ceiling caps it.
        let budget = req.budget.max_compile_ms.min(self.max_compile_ms);

        match baracuda_kernelgen::jit::seam::synthesize(
            &req.region,
            &req.operands,
            op_category,
            req.arch,
            &fused_op_id,
            budget,
            &Cuda,
            &compiler,
        ) {
            Ok(resp) => {
                // Map the native artifact provenance to the envelope's loadable-only
                // ArtifactKind. A non-loadable stub NEVER crosses the seam — decline
                // honestly (frozen 2026-07-04: no unloadable placeholder is ever
                // `Synthesized`, so a Fuel loader never has to refuse one). This is
                // the no-CUDA-toolchain path (StubCompiler when `nvrtc` is off).
                let kind = match resp.kernel.kind {
                    ArtifactKind::Ptx => SeamArtifactKind::Ptx,
                    ArtifactKind::Cubin => SeamArtifactKind::Cubin,
                    ArtifactKind::Stub => {
                        return SeamResponse::Declined {
                            reason: format!(
                                "{fused_op_id}: no loadable artifact (no CUDA toolchain / stub compiler)"
                            ),
                        };
                    }
                };
                let entry_point = resp.kernel.entry_point.clone();
                // Convert to the envelope SynthArtifact at the boundary so Fuel
                // depends on none of our types. The `recipe` + `.cu` source are
                // dropped (re-fuse `pattern:` rides in `contract`; `decompose` is
                // the region Fuel holds). Baracuda's LinkEntry has no `symbol` — an
                // extern "C" kernel's symbol IS its entry_point.
                let link = SeamLinkEntry {
                    entry_point: resp.link.entry_point.clone(),
                    symbol: resp.link.entry_point.clone(),
                    structure_key: resp.link.structure_key,
                    revision_hash: resp.link.revision_hash,
                };
                self.registry
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(
                        entry_point.clone(),
                        SeamArtifact {
                            artifact: resp.kernel.artifact,
                            kind,
                            link,
                            contract: resp.contract,
                        },
                    );
                // Light handle — the heavy artifact is fetched only if Fuel's
                // cost-gate adopts (via `take_kernel`). Cost now rides the contract's
                // cost-expr, not the wire response.
                SeamResponse::Synthesized { entry_point }
            }
            // Honest decline (the trait forbids panicking) — region op/dtype out
            // of vocabulary, malformed request, over budget, or compile failure.
            Err(e) => SeamResponse::Declined {
                reason: format!("{e:?}"),
            },
        }
    }

    /// Hand over + remove the retained [`SeamArtifact`] for `entry_point` — the
    /// §5.2 two-step handover's second step (Fuel calls it once its cost-gate
    /// adopts). On the trait (Fuel invokes it via `&dyn Synthesizer`). `None` if
    /// never synthesized or already taken (single adopt).
    fn take_kernel(&self, entry_point: &str) -> Option<SeamArtifact> {
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(entry_point)
    }
}

/// A stable, readable kernel id for a region: `jit_<root-op>_<hash>`, where the
/// full 64-bit hash covers the region structure **and** the operand projection,
/// so distinct regions (or the same region at different layouts/dtypes) are
/// collision-resistant on one `entry_point`. (The id keys the artifact
/// `registry`, so a collision would silently overwrite a not-yet-taken kernel —
/// hence the full u64, not a truncated prefix.)
fn region_op_id(region: &SeamNode, operands: &[OperandDesc]) -> String {
    use std::hash::{Hash, Hasher};
    let root = match region {
        SeamNode::Op { op, .. } => format!("{op:?}").to_lowercase(),
        _ => "fused".to_string(),
    };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{region:?}|{operands:?}").hash(&mut h);
    format!("jit_{root}_{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use baracuda_kernelgen::{JitError, StubCompiler};
    use baracuda_kernelgen::jit::seam::synthesize;
    use baracuda_kernel_vocab::{ArchSku, ElementKind};
    use fuel_kernel_seam_types::OpTag;
    use fuel_kernel_seam::JitBudget;
    use fuel_kernel_seam_types::OpAttrs;

    fn op(op: OpTag, operands: Vec<SeamNode>) -> SeamNode {
        SeamNode::Op {
            op,
            operands,
            attrs: OpAttrs::default(),
        }
    }

    fn operands(dt: ElementKind, n: usize) -> Vec<OperandDesc> {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
        std::iter::repeat_n(a, n).collect()
    }

    #[test]
    fn synthesize_fuel_region() {
        // relu(a + b) in Fuel's grammar -> a synthesized fused kernel.
        let region = op(
            OpTag::Relu,
            vec![op(
                OpTag::Add,
                vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
            )],
        );
        let resp = synthesize(
            &region,
            &operands(ElementKind::F32, 3),
            OpCategory::BinaryElementwise,
            ArchSku::Sm89,
            "jit_relu_add",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap();
        assert!(resp.contract.contains("fused_op: jit_relu_add"));
        assert!(resp.recipe.pattern.contains("op: Relu"));
        assert!(resp.kernel.source.contains("__global__"));
    }

    #[test]
    fn iota_region_declines_typed() {
        // OpTag::Iota EXISTS in fuel-kernel-seam-types 0.10.2 (a "value
        // source"), so a Fuel region CAN name it — but the seam converter
        // drops OpAttrs (where Iota's axis lives), and an axis-less
        // coordinate is the wrong kernel. The region must decline TYPED
        // (UnsupportedOp("Iota")), never panic and never synthesize an
        // axis-guessed `ScalarExpr::Coord` — both bare and nested under
        // supported ops.
        let bare = op(OpTag::Iota, vec![]);
        let err = synthesize(
            &bare,
            &operands(ElementKind::F32, 1),
            OpCategory::UnaryElementwise,
            ArchSku::Sm89,
            "jit_iota",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert!(
            matches!(err, JitError::UnsupportedOp(ref o) if o == "Iota"),
            "got {err:?}"
        );
        // Nested: mul(x, iota) — the triu-mask-ish shape a matcher could
        // plausibly hand us once Iota appears in user graphs.
        let nested = op(
            OpTag::Mul,
            vec![SeamNode::Bind { index: 0 }, op(OpTag::Iota, vec![])],
        );
        let err = synthesize(
            &nested,
            &operands(ElementKind::F32, 2),
            OpCategory::BinaryElementwise,
            ArchSku::Sm89,
            "jit_mul_iota",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert!(
            matches!(err, JitError::UnsupportedOp(ref o) if o == "Iota"),
            "got {err:?}"
        );
        // And through the live Synthesizer envelope: a typed Declined,
        // never a panic across the §5 boundary.
        let synth = BaracudaSynthesizer::new(1000);
        let req = SeamRequest {
            region: op(
                OpTag::Mul,
                vec![SeamNode::Bind { index: 0 }, op(OpTag::Iota, vec![])],
            ),
            operands: operands(ElementKind::F32, 2),
            arch: ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 1000,
            },
        };
        let SeamResponse::Declined { reason } = synth.synthesize(&req) else {
            panic!("expected Declined");
        };
        assert!(
            reason.contains("Iota"),
            "reason should name the tag: {reason}"
        );
    }

    #[test]
    fn tanh_gelu_optag_is_unsupported() {
        // Op::Gelu (tanh-approx) is a distinct tag we don't synthesize.
        let region = op(OpTag::Gelu, vec![SeamNode::Bind { index: 0 }]);
        let err = synthesize(
            &region,
            &operands(ElementKind::F32, 2),
            OpCategory::UnaryElementwise,
            ArchSku::Sm89,
            "x",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert!(matches!(err, JitError::UnsupportedOp(_)));
    }

    #[cfg(not(feature = "nvrtc"))]
    #[test]
    fn synthesizer_without_a_toolchain_declines_no_stub_crosses_the_seam() {
        // Frozen 2026-07-04: the envelope `ArtifactKind` is loadable-only
        // (Ptx|Cubin), so a non-loadable/stub synth returns `Declined` — no
        // unloadable placeholder is ever `Synthesized`, and a Fuel loader never has
        // to refuse one. Without the `nvrtc` feature there is no real compiler, so
        // a perfectly buildable region still declines honestly. The Synthesized +
        // retain + `take_kernel` handover is exercised with a real artifact under
        // `nvrtc` in `synthesizer_produces_real_ptx_on_device`.
        let region = op(
            OpTag::Relu,
            vec![op(
                OpTag::Add,
                vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
            )],
        );
        let synth = BaracudaSynthesizer::new(1000);
        let req = SeamRequest {
            region,
            operands: operands(ElementKind::F32, 3),
            arch: ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 1000,
            },
        };
        let SeamResponse::Declined { reason } = synth.synthesize(&req) else {
            panic!("expected Declined (no loadable artifact without nvrtc)");
        };
        assert!(
            reason.contains("no loadable artifact"),
            "reason should explain the stub decline: {reason}"
        );
    }

    #[test]
    fn seam_nested_cmp_region_declines_awaiting_cast_vocabulary() {
        // relu-backward via the seam grammar: Mul(dy, Gt(x, z)). The
        // kernel WOULD be correct, but the response contract's pattern
        // block would encode Gt directly under Mul — an edge no real Fuel
        // graph has (their compare builders pin U8 output; real graphs
        // interpose Cast, which the pattern grammar can't express). Fuel's
        // matcher can't produce this region from a real graph either, so
        // the typed decline costs zero live coverage. Revisit with Cast.
        let region = op(
            OpTag::Mul,
            vec![
                SeamNode::Bind { index: 0 },
                op(
                    OpTag::Gt,
                    vec![SeamNode::Bind { index: 1 }, SeamNode::Bind { index: 2 }],
                ),
            ],
        );
        let err = synthesize(
            &region,
            &operands(ElementKind::F32, 4),
            OpCategory::TernaryElementwise,
            ArchSku::Sm89,
            "jit_relu_bw",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert!(
            matches!(&err, JitError::UnsupportedOp(m) if m.contains("interior comparison")),
            "expected the interior-cmp typed decline, got {err:?}"
        );
    }

    #[test]
    fn seam_root_cmp_region_declines_typed_under_both_projections() {
        let region = op(
            OpTag::Gt,
            vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
        );
        // Uniform-dtype projection: the root-cmp gate declines (a cmp's
        // output is a U8 mask, not the key dtype).
        let err = synthesize(
            &region,
            &operands(ElementKind::F32, 3),
            OpCategory::BinaryElementwise,
            ArchSku::Sm89,
            "x",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert!(
            matches!(&err, JitError::UnsupportedOp(m) if m.contains("comparison at region root")),
            "got {err:?}"
        );
        // HONEST projection (f32 inputs, U8 output OperandDesc — the seam
        // request CAN express it): the increment-1 uniform-dtype gate
        // declines as MixedDtype before synthesis.
        let mut ops = operands(ElementKind::F32, 3);
        ops[2] = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::U8, 256);
        let err = synthesize(
            &region,
            &ops,
            OpCategory::BinaryElementwise,
            ArchSku::Sm89,
            "x",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert_eq!(err, JitError::MixedDtype);
        // And the live envelope path is a typed Declined, never a panic.
        let synth = BaracudaSynthesizer::new(1000);
        let req = SeamRequest {
            region,
            operands: ops,
            arch: ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 1000,
            },
        };
        assert!(matches!(
            synth.synthesize(&req),
            SeamResponse::Declined { .. }
        ));
    }

    #[test]
    fn seam_where_region_declines_at_the_typed_pattern_miss() {
        // Fuel's fused-select shape through the seam grammar:
        // Where(Gt(a, b), x, y), all-float binds — genuinely uniform-dtype.
        // The interior-cmp carve-out lets it past the Cast decline (Where
        // consumes the mask edge DIRECTLY); the v1 decline is the typed
        // pattern miss (the withheld Where advert), and the envelope path
        // is a typed Declined, never a panic.
        let region = op(
            OpTag::Where,
            vec![
                op(
                    OpTag::Gt,
                    vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
                ),
                SeamNode::Bind { index: 2 },
                SeamNode::Bind { index: 3 },
            ],
        );
        let err = synthesize(
            &region,
            &operands(ElementKind::F32, 5),
            OpCategory::TernaryElementwise,
            ArchSku::Sm89,
            "jit_where",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert_eq!(
            err,
            JitError::Pattern(baracuda_kernelgen::pattern::PatternError::SelectUnsupported),
            "the decline must name the withheld Where advert"
        );
        let synth = BaracudaSynthesizer::new(1000);
        let req = SeamRequest {
            region,
            operands: operands(ElementKind::F32, 5),
            arch: ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 1000,
            },
        };
        let SeamResponse::Declined { reason } = synth.synthesize(&req) else {
            panic!("expected Declined");
        };
        assert!(
            reason.contains("SelectUnsupported"),
            "reason should name the typed miss: {reason}"
        );
    }

    #[test]
    fn seam_bound_cond_where_region_declines_typed_under_both_projections() {
        // M11's pin, mirroring seam_root_cmp_region_declines_typed_under_
        // both_projections: a BOUND cond is a U8 tensor → the [U8,T,T]
        // operand tuple — inexpressible under uniform keying.
        let region = op(
            OpTag::Where,
            vec![
                SeamNode::Bind { index: 0 },
                SeamNode::Bind { index: 1 },
                SeamNode::Bind { index: 2 },
            ],
        );
        // Uniform-dtype projection (all T incl. the cond): the bound-cond
        // gate declines typed — synthesizing would misdescribe Fuel's
        // U8-cond op as a key-dtype `!= 0` kernel.
        let err = synthesize(
            &region,
            &operands(ElementKind::F32, 4),
            OpCategory::TernaryElementwise,
            ArchSku::Sm89,
            "x",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert!(
            matches!(&err, JitError::UnsupportedOp(m) if m.contains("bound cond")),
            "got {err:?}"
        );
        // HONEST projection (U8 cond OperandDesc, float arms/out — the
        // shape Fuel would really send): the uniform-dtype gate declines
        // as MixedDtype before synthesis.
        let mut ops = operands(ElementKind::F32, 4);
        ops[0] = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::U8, 256);
        let err = synthesize(
            &region,
            &ops,
            OpCategory::TernaryElementwise,
            ArchSku::Sm89,
            "x",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert_eq!(err, JitError::MixedDtype);
        // And the live envelope path is a typed Declined, never a panic.
        let synth = BaracudaSynthesizer::new(1000);
        let req = SeamRequest {
            region,
            operands: ops,
            arch: ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 1000,
            },
        };
        assert!(matches!(
            synth.synthesize(&req),
            SeamResponse::Declined { .. }
        ));
    }

    #[test]
    fn synthesizer_declines_never_panics() {
        // An out-of-vocabulary region (Op::Gelu tanh) is an honest Declined, never
        // an error or a panic across the trait boundary.
        let synth = BaracudaSynthesizer::new(1000);
        let req = SeamRequest {
            region: op(OpTag::Gelu, vec![SeamNode::Bind { index: 0 }]),
            operands: operands(ElementKind::F32, 2),
            arch: ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 1000,
            },
        };
        assert!(matches!(
            synth.synthesize(&req),
            SeamResponse::Declined { .. }
        ));
    }

    #[test]
    fn synthesizer_declines_unlowerable_dtype_never_panics() {
        // Regression (adversarial review): a PURE-INFIX region (Add over binds —
        // no unary/binary-fn, no Param) at a dtype the CUDA backend can't spell as
        // a scalar (Bool / Complex64) used to PANIC in scalar_ctype during
        // `generate`, because dtype_compatible lets pure-infix bodies through for
        // ANY dtype. The backend dtype-lowerability gate must Decline, never unwind
        // across the trait boundary (which would crash the host). (S8 left this
        // list in increment 0c — it is now an audited compute dtype and
        // synthesizes; see seam_uniform_int_add_synthesizes.)
        let synth = BaracudaSynthesizer::new(1000);
        let region = op(
            OpTag::Add,
            vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
        );
        for dt in [ElementKind::Bool, ElementKind::Complex64] {
            let req = SeamRequest {
                region: region.clone(),
                operands: operands(dt, 3),
                arch: ArchSku::Sm89,
                budget: JitBudget {
                    max_compile_ms: 1000,
                },
            };
            assert!(
                matches!(synth.synthesize(&req), SeamResponse::Declined { .. }),
                "{dt:?} must Decline, not panic",
            );
        }
    }

    #[test]
    fn seam_uniform_int_add_synthesizes_and_div_declines_typed() {
        // Increment 0c: uniform-U8/S8 COMPUTE regions are audited. There is
        // no OpTag for the bitwise ops (BitAnd would be the natural probe —
        // it does not exist in 0.10.2), so the LEGAL-op probe is infix Add
        // at U8/S8: it must synthesize (wrapping semantics, scalar path,
        // real contract carrying the U8/I8 dtype)…
        for (dt, fkc) in [(ElementKind::U8, "[U8]"), (ElementKind::S8, "[I8]")] {
            let region = op(
                OpTag::Add,
                vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
            );
            let resp = synthesize(
                &region,
                &operands(dt, 3),
                OpCategory::BinaryElementwise,
                ArchSku::Sm89,
                "jit_int_add",
                1000,
                &Cuda,
                &StubCompiler,
            )
            .unwrap_or_else(|e| panic!("{dt:?} Add must synthesize, got {e:?}"));
            assert!(resp.kernel.source.contains("__global__"));
            assert!(
                resp.contract.contains(&format!("dtypes: {fkc}")),
                "{dt:?}: {}",
                resp.contract
            );
        }
        // …while an ILLEGAL op at the SAME dtype still declines typed:
        // Div-for-int is rejected (no bespoke int div; /0 is device-UB), so
        // a uniform-U8 Div region must not ride in on the dtype flip.
        let div = op(
            OpTag::Div,
            vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
        );
        let err = synthesize(
            &div,
            &operands(ElementKind::U8, 3),
            OpCategory::BinaryElementwise,
            ArchSku::Sm89,
            "jit_u8_div",
            1000,
            &Cuda,
            &StubCompiler,
        )
        .unwrap_err();
        assert_eq!(err, JitError::UnsupportedDtype, "U8 Div must decline typed");
        // And the live envelope path stays a typed Declined, never a panic.
        let synth = BaracudaSynthesizer::new(1000);
        let req = SeamRequest {
            region: op(
                OpTag::Div,
                vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
            ),
            operands: operands(ElementKind::U8, 3),
            arch: ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 1000,
            },
        };
        assert!(matches!(
            synth.synthesize(&req),
            SeamResponse::Declined { .. }
        ));
    }

    /// End-to-end live path on-device: a Fuel `JitRequest` through the
    /// `Synthesizer` impl yields a real nvrtc PTX artifact (retrievable for the
    /// seam-call site to bind). Ignored (needs nvrtc + CUDA).
    #[cfg(feature = "nvrtc")]
    #[test]
    #[ignore = "requires nvrtc runtime + CUDA install"]
    fn synthesizer_produces_real_ptx_on_device() {
        let region = op(
            OpTag::Relu,
            vec![op(
                OpTag::Add,
                vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
            )],
        );
        let synth = BaracudaSynthesizer::new(5000);
        let req = SeamRequest {
            region,
            operands: operands(ElementKind::F32, 3),
            arch: ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 5000,
            },
        };
        // Light handle — entry_point only.
        let SeamResponse::Synthesized { entry_point } = synth.synthesize(&req) else {
            panic!("expected Synthesized");
        };
        assert!(entry_point.starts_with("baracuda_gen_jit_relu_"));
        // The two-step handover: the envelope SynthArtifact carries the loadable PTX
        // + the FKC contract + the binding row (no .cu source, no recipe — dropped
        // at the boundary). Single adopt: the second take_kernel is None.
        let art = synth.take_kernel(&entry_point).expect("artifact retained");
        assert_eq!(art.kind, SeamArtifactKind::Ptx);
        assert!(
            String::from_utf8(art.artifact.clone())
                .unwrap()
                .contains(".entry")
        );
        assert!(art.contract.contains("fused_op:"));
        assert_eq!(art.link.entry_point, entry_point);
        assert_eq!(art.link.symbol, entry_point); // extern "C": symbol == entry_point
        assert!(synth.take_kernel(&entry_point).is_none());
    }
}
