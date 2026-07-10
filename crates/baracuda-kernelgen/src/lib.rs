//! # baracuda-kernelgen
//!
//! Build-time generator that turns an op's **abstract IR** (the algorithm) plus
//! a [`baracuda_kernels_types::StructureKey`] cell (the schedule) into a
//! specialized kernel — and its FKC contract.
//!
//! ## Stability posture (published as of alpha.76)
//!
//! This crate is published so Fuel can construct the live JIT synthesizer from
//! crates.io. The **supported surface is the `seam` feature** — the
//! `fuel_kernel_seam::Synthesizer` impl (`BaracudaSynthesizer`) with its
//! two-step `synthesize`/`take_kernel` handover, frozen with Fuel (2026-07-04).
//! Everything else (the IR, plan, emitters, contracts, dispatch artifacts) is
//! **alpha-fluid generator internals**: the `0.0.1-alpha.N` lockstep implies no
//! cross-version API stability anywhere — pin exact versions.
//!
//! The crate is **language-agnostic except for the lowering backend**:
//!
//! - [`ir`] — the op IR (a [`ScalarExpr`] DAG). Backend-neutral.
//! - [`plan`] — the schedule decision (`StructureKey` → [`KernelPlan`]). Neutral.
//! - [`backend`] — the [`Backend`] trait + the neutral [`backend::lower_expr`].
//! - [`cuda`] — the **only** backend-specific module today. Slang / SPIR-V /
//!   Metal / CPU backends are additional [`Backend`] impls, no core changes.
//!
//! Op logic is described as IR rather than opaque CUDA precisely so the emitter
//! can *see the dataflow* and transform it (vectorize, hoist, fuse) — and so the
//! same op can be lowered to any backend. A hand-written escape hatch is
//! reserved for bespoke ops the IR can't yet express.
//!
//! This is a dev/build tool (`publish = false`); the artifacts it emits are
//! committed and ship inside `baracuda-kernels-sys`.
//!
//! ## Status (v1)
//!
//! Pilot scope: f32 elementwise ops, contiguous operands, emitted `float4`-
//! vectorized when the cell says V4 and scalar otherwise, lowered to CUDA.
//! Other dtypes, strided / broadcast / reduction schedules, additional
//! backends, FKC emission, and the algebraic optimizer are the growth path.

pub mod backend;
pub mod contract;
pub mod cuda;
pub mod dispatch_artifact;
pub mod ir;
pub mod jit;
pub mod link;
pub mod optimize;
pub mod oracle;
pub mod pattern;
pub mod plan;
pub mod telemetry;

pub use backend::{Backend, GeneratedKernel, Variant, VariantFidelity};
pub use contract::{bundle, contract, front_matter};
pub use cuda::Cuda;
pub use dispatch_artifact::{emit_dispatch_table, parse_dispatch_table};
pub use ir::{
    Access, AccumSpec, AxisRole, ContractionAxes, DagNode, Expr, ExprDag, NodeId, OpDef, ReduceOp,
    ReduceStage, ScalarExpr, SortOrder, SortOut, UnaryOp, coord, input, konst, param, reduced,
};
#[cfg(feature = "nvrtc")]
pub use jit::NvrtcCompiler;
pub use jit::{
    ArtifactKind, Compiler, JitBudget, JitError, JitRequest, JitResponse, Recipe, StubCompiler,
    SynthKernel, synthesize,
};
pub use link::{LinkEntry, emit_link_registry, link_entry};
pub use optimize::{optimize, optimize_top_k};
pub use oracle::{Fidelity, TypedBuffer, compare, evaluate};
pub use pattern::{PatternError, PatternNode, derive_pattern, to_fkc};
pub use plan::{KernelPlan, Schedule, build_plan};
pub use telemetry::{
    Candidate, DispatchRecord, HwFingerprint, ImplId, Ingest, MissRecord, RankedCell,
    TELEMETRY_SCHEMA_MAX, VariantVote, arch_sku_of, ingest_jsonl, merge_reports, rank_matrix,
    resolve_impl, variant_votes,
};

use baracuda_kernels_types::StructureKey;

/// Generate a specialized kernel for `op` at structure cell `key`, lowered by
/// `backend`. Convenience over [`build_plan`] followed by [`Backend::lower`].
#[must_use]
pub fn generate(op: &OpDef, key: &StructureKey, backend: &dyn Backend) -> GeneratedKernel {
    backend.lower(&build_plan(op, key))
}

/// Generate the full **variant set** for a cell: the default lowering (tag
/// `"base"`, bit-identical by definition) plus every backend schedule variant.
/// Ship-top-K policy (variants backlog doc): all validated variants ship, each
/// with its own contract; the item-07 bench gate ranks them per arch and the
/// dispatch table records Baracuda's default — Fuel stays the runtime selector.
#[must_use]
pub fn generate_variants(op: &OpDef, key: &StructureKey, backend: &dyn Backend) -> Vec<Variant> {
    let plan = build_plan(op, key);
    // BASE_OFFSET SLICE: an offsetted kernel's OOB guarantee is a CALLER
    // PRECONDITION (the k/n_out trust model, the gext/sext + RowSort-k<=1024
    // class) — no per-element bounds branch is emitted, so the base variant's
    // launch_note carries the contract. Offset-free plans keep the empty note
    // (byte-stable for every pre-increment op).
    let has_offset =
        plan.base_offsets.iter().any(|b| !b.is_zero()) || !plan.out_base_offset.is_zero();
    let launch_note = if has_offset {
        "offset launch contract: each `long long off*` argument is a runtime base \
         ELEMENT offset added to its operand's base pointer at kernel entry. Caller \
         precondition (no per-element bounds check is emitted): for every offsetted \
         operand, off + <maximal element address reachable via the declared \
         shape/strides (and gext/sext window, if composed)> must lie within the \
         allocated buffer; only off >= 0 is validated."
            .to_string()
    } else {
        String::new()
    };
    let mut vs = vec![Variant {
        tag: "base",
        kernels: vec![backend.lower(&plan)],
        fidelity: VariantFidelity::BitIdentical,
        launch_note,
    }];
    vs.extend(backend.lower_variants(&plan));
    vs
}
