//! Emit a complete Fuel **FKC kernel contract** from an op + its structure cell.
//!
//! Where [`crate::pattern`] derives only the `pattern:` block (for a fused op),
//! this assembles the *whole* importable contract Fuel reads to bind a kernel:
//! the bundle front-matter plus the per-kernel `accept` / `return` / `op_params`
//! / `caps` / `cost` / `precision` / `determinism` blocks (and `pattern:` when
//! the op is a recognized fusion).
//!
//! The block field set is pinned to the Profile-v1 conformance matrix
//! (*Kernel-Seam Interop* §4.3) and the `ImplId` five-field tuple (FKC §4.11:
//! `backend`, op, `dtypes`, `kernel_source`, `kernel_revision_hash` — five
//! separable wire fields, never a hash). The admissibility predicate **is** the
//! structure key (per `docs/design/kernel-specialization.md`), so each contract
//! carries its [`StructureKey::to_token`] verbatim under `accept` — the planner's
//! miss signal stays honest by construction.
//!
//! # Scope (v1) / reconciliation
//!
//! Elementwise ops. The block *field names* follow §4.3 and the `ImplId` tuple;
//! a few leaf *spellings* (the dtype tokens — review item E5 — and the precision
//! / layout enums) are reconstructed from the FKC annex and reconciled when the
//! full `kernel-contract-format.md` is wired. They are isolated in the small
//! helpers below so reconciliation is a localized change.
//!
//! The single-op **primitive `op_kind` spelling** is likewise reconciled — the
//! internal pattern roots (`Add`, `Relu`, `GeluErf`, …) are mapped to the exact
//! importer spellings (`AddElementwise`, `ReluElementwise`, …) by
//! [`fuel_primitive_op_kind`], verbatim against fuel-dispatch `fkc/lower.rs`
//! `lower_op_kind`; a root with no importable Fuel `OpKind` is an honest miss
//! (the contract is withheld) rather than a bundle-poisoning unknown name.

use crate::backend::GeneratedKernel;
use crate::ir::{BinaryOp, ExprDag, NodeId, OobPolicy, OpDef, ScalarExpr, UnaryOp};
use crate::pattern::{PatternNode, derive_pattern, to_fkc};
use baracuda_kernel_vocab::{Contiguity, ElementKind, StructureKey, VecWidth};

/// Canonicalize a caller's backend token to the exact capitalized spelling
/// Fuel's FKC importer accepts (fuel-dispatch `fkc/lower.rs` `lower_backend`,
/// verified 2026-07-08: the explicit table maps `Cpu`/`Cuda`/`Vulkan`/`Metal`
/// and errors [`UnknownBackend`] on anything else). Baracuda's callers pass the
/// lowercase provider token (`backend.name()` → `"cuda"`); the FKC wire form is
/// `Cuda`. Applied at every emission site ([`front_matter`] + [`contract`]) so
/// callers stay unchanged. An unrecognized token passes through verbatim — the
/// emitter never fabricates a backend name; an unknown one is a caller bug that
/// surfaces honestly as Fuel's `UnknownBackend` at import rather than being
/// silently rewritten.
fn fkc_backend_token(backend: &str) -> &str {
    match backend {
        "cpu" | "Cpu" => "Cpu",
        "cuda" | "Cuda" => "Cuda",
        "vulkan" | "Vulkan" => "Vulkan",
        "metal" | "Metal" => "Metal",
        other => other,
    }
}

/// Provider-wide FKC bundle front-matter (FKC §0/§3.1) — emitted once per bundle
/// file, above the per-kernel [`contract`] blocks. `revision_base` is the
/// provider source-tree revision the kernels were built from (the
/// `ImplId.kernel_revision_hash` base). Carries `seam_profiles: [1]` so an
/// importer can reject a contract outside the negotiated seam profile (§3.5).
/// The `backend` token is canonicalized to Fuel's capitalized spelling via
/// [`fkc_backend_token`] (`"cuda"` → `Cuda`).
#[must_use]
pub fn front_matter(backend_name: &str, revision_base: &str) -> String {
    let backend = fkc_backend_token(backend_name);
    format!(
        "---\n\
         fkc_version: 1\n\
         provider:\n  \
         name: baracuda\n  \
         backend: {backend}\n  \
         kernel_source: baracuda\n  \
         link_registry: baracuda_link_registry\n  \
         revision_base: \"{revision_base}\"\n\
         seam_profiles: [1]\n\
         ---\n"
    )
}

/// Assemble a complete importable FKC bundle: the provider [`front_matter`]
/// followed by every `contract` under its own `## ` heading.
///
/// **Why the `## ` framing is load-bearing (not cosmetic).** Fuel's FKC parser
/// (fuel-dispatch `fkc/parse.rs` `split_sections`, verified 2026-07-08) collects
/// a fenced ```` ```fkc ```` block ONLY when it sits under a `## ` heading and
/// SILENTLY DROPS a headingless block (prose before the first `## ` is ignored).
/// A bundle that merely concatenates [`contract`] outputs therefore imports
/// `Ok`-but-EMPTY — a no-op adopt that looks like success. This assembler is the
/// single source of the framing every emission site (bin, examples, tests,
/// proofs) shares, so no caller can reintroduce that hazard by hand.
///
/// The heading title is the contract's own `kernel:` name (matching Fuel's
/// corpus, docs/kernel-contracts/cpu/elementwise-binary.fkc.md, where each
/// `## <section>` names its kernel). The title is diagnostic only — the true
/// kernel identity is the `kernel:`/`entry_point:` fields inside the block — so
/// the exact heading text is not parsed; it need only exist and start `## `.
///
/// NOTE the JIT seam (`crate::jit`) deliberately does NOT use this: Fuel wraps
/// `art.contract` with the provider identity + heading at adopt time (their
/// 2026-07-08 answer (a)), so the seam ships the bare [`contract`] block.
#[must_use]
pub fn bundle(backend_name: &str, revision_base: &str, contracts: &[String]) -> String {
    let mut s = front_matter(backend_name, revision_base);
    for c in contracts {
        // Withhold a fused contract whose `fused_op:` name is NOT one of Fuel's
        // FusedOps SCREAMING_SNAKE constants — an unknown fused_op is BUNDLE-FATAL
        // (fuel-dispatch: `lower_fused_op` → `UnknownFusedOp`; `validate_file`
        // runs it for every registrable section and propagates with `?`;
        // `import_bundle_str` runs `validate_file(&file)?` over the WHOLE file), so
        // one free-form fused name poisons every correct primitive beside it. None
        // of Baracuda's elementwise fusions (`relu_add`, `affine_silu`, a bare
        // `copy`, …) matches a Fuel FusedOp (those are big structured kernels —
        // softmax / norms / attention / conv / matmul), so they are all withheld
        // HERE, from the importable bundle. The kernel still generates + runs AOT
        // and rides the JIT seam as a BARE block (`crate::jit`; Fuel's adopt stores
        // that contract text UNPARSED — 2026-07-08 answer (a) — so it never reaches
        // this whole-file import path). A fusion that genuinely maps to a Fuel
        // FusedOp is emitted by `contract` with the SCREAMING_SNAKE constant as its
        // `fused_op:` value (see `fuel_fused_op_name`), which this filter admits.
        // Verified 2026-07-08 against Fuel's real `import_bundle_str`.
        if !contract_admissible(c, false) {
            continue;
        }
        // The heading title = the contract's `kernel:` name (diagnostic; the
        // block's own fields carry the real identity). A malformed contract
        // with no `kernel:` line falls back to a generic title rather than
        // panicking — it would fail import on its own merits, not the framing.
        let title = kernel_name_of(c).unwrap_or("kernel");
        s.push_str(&format!("\n## {title}\n\n"));
        s.push_str(c);
    }
    s
}

/// Assemble a **KISC-framed** importable bundle: the provider [`front_matter`]
/// followed by each admitted [`contract`] as its own self-delimiting KISC
/// document (KISS-Contract §6.11), in order. Replaces [`bundle`]'s `## `-heading
/// framing — each kernel is one KISC document (magic + `len` + `crc32`,
/// hard-reject), so a malformed contract can no longer import as a silent empty
/// and a bad document declines alone instead of poisoning the file. Emitted
/// behind the negotiated `SEAM_CAP_KISC_FRAMING` cutover; [`bundle`] stays for
/// pre-KISC peers.
///
/// Admission is capability-gated on `recipe_import` (see [`contract_admissible`]):
/// against a pre-recipe-import peer, a fused contract with no Fuel FusedOp is
/// withheld; once Baracuda emits the recipe AND the peer advertises recipe-import
/// (`SEAM_CAP_RECIPE_IMPORT`), that withhold retires — no code change here, only
/// [`contract_carries_recipe`] becoming real. The KISC header-line and bundle
/// structure are PROVISIONAL (see [`crate::kisc`]).
#[must_use]
pub fn bundle_kisc(
    backend_name: &str,
    revision_base: &str,
    contracts: &[String],
    recipe_import: bool,
) -> String {
    let mut s = front_matter(backend_name, revision_base);
    for c in contracts {
        if !contract_admissible(c, recipe_import) {
            continue;
        }
        s.push_str(&crate::kisc::kisc_frame(c));
        // Documents are self-delimiting (the header declares `len`); a single
        // `\n` between them keeps the file readable and line-tool-friendly.
        s.push('\n');
    }
    s
}

/// The `fused_op:` name declared inside a [`contract`] block, or `None` for a
/// primitive (`op_kind:`) contract. Used by [`bundle`] to withhold a fused
/// contract Fuel cannot import.
fn fused_op_of(contract: &str) -> Option<&str> {
    contract
        .lines()
        .find_map(|l| l.trim().strip_prefix("fused_op: "))
        .map(str::trim)
}

/// Whether `contract` is admissible into an importable bundle for a peer whose
/// importer supports `recipe_import` (recipe-verify-and-register).
///
/// - A primitive (`op_kind:`) contract is always importable — Fuel knows the base ops.
/// - A fused contract whose `fused_op:` IS a Fuel FusedOp is importable by the
///   closed-vocabulary importer.
/// - A fused contract whose `fused_op:` is NOT a Fuel FusedOp is importable ONLY by
///   a `recipe_import` peer, and ONLY if the contract carries a recipe (the KISS-Ops
///   Semantics op-DAG to the base floor) the importer can verify + register. Baracuda
///   does not emit that recipe yet ([`contract_carries_recipe`]), so today this stays
///   withheld even against a recipe-import peer — this arm is the seam where retiring
///   the withhold lands (in lockstep with recipe emission + Fuel's recipe importer).
///
/// Shared by [`bundle`] (which passes `recipe_import = false`) and [`bundle_kisc`].
#[doc(hidden)]
pub fn contract_admissible(contract: &str, recipe_import: bool) -> bool {
    match fused_op_of(contract) {
        None => true,
        Some(name) if FUEL_FUSED_OPS.contains(&name) => true,
        Some(_) => recipe_import && contract_carries_recipe(contract),
    }
}

/// Whether `contract` carries a **recipe** — the KISS-Ops Semantics op-DAG
/// decomposed to the base floor, which a recipe-verify importer runs to validate
/// and register an op it does not already know.
///
/// Detected by the `semantics:` line [`contract`] emits when the op re-bases onto
/// confirmed KISS-Ops tokens ([`crate::recipe::semantics_dag`]). An op with no
/// confirmed re-basing carries no recipe, so an otherwise-unknown fused op stays
/// withheld even from a recipe-import peer until its recipe can be emitted.
fn contract_carries_recipe(contract: &str) -> bool {
    contract
        .lines()
        .any(|l| l.trim_start().starts_with("semantics: "))
}

/// The exhaustive `FusedOps::*` SCREAMING_SNAKE constant names Fuel's
/// `fkc/lower.rs` `fused_op_id_for_const_name` table accepts (verified verbatim
/// 2026-07-08). A `fused_op:` value NOT in this set is bundle-fatal, so [`bundle`]
/// withholds it. Mirrors the [`fuel_primitive_op_kind`] honest-miss posture for
/// the fused surface.
const FUEL_FUSED_OPS: &[&str] = &[
    "SOFTMAX_LAST_DIM",
    "FUSED_LINEAR",
    "RMS_NORM_LAST_DIM",
    "LAYER_NORM_LAST_DIM",
    "ROPE",
    "CONV2D",
    "SOFTMAX_LAST_DIM_BACKWARD",
    "LAYER_NORM_LAST_DIM_BACKWARD",
    "RMS_NORM_LAST_DIM_BACKWARD",
    "REDUCE_MAX_TO_BACKWARD",
    "CONV_TRANSPOSE2D",
    "FLASH_ATTN",
    "PAGED_ATTN",
    "QMATMUL",
    "POWI_BACKWARD",
    "INPLACE_AFFINE",
    "FUSED_SOFTMAX_CROSS_ENTROPY",
    "CAUSAL_CONV1D",
    "SELECTIVE_SCAN",
    "SSD_CHUNK_SCAN",
    "NF4_MATMUL",
    "FLASH_ATTN_BACKWARD_Q",
    "FLASH_ATTN_BACKWARD_K",
    "FLASH_ATTN_BACKWARD_V",
];

/// Map a Baracuda fusion's stable `op.name` to the Fuel FusedOps SCREAMING_SNAKE
/// constant it genuinely IS (name-AND-semantics match only), or `None` for a
/// fusion with no Fuel FusedOp identity. NONE of Baracuda's current elementwise
/// fusions map (Fuel's FusedOps are big structured kernels — softmax / norms /
/// attention / conv / matmul — not generic elementwise compositions), so this is
/// `None` for everything today; it is the conservative extension point (mirroring
/// [`fuel_primitive_op_kind`]) so a genuinely-matching fusion emits the exact Fuel
/// constant as its `fused_op:` value (which [`bundle`] then admits) instead of a
/// bundle-fatal free-form name.
fn fuel_fused_op_name(_baracuda_name: &str) -> Option<&'static str> {
    // No Baracuda fusion maps to a Fuel FusedOps constant yet. Add an arm here
    // (returning a `FUEL_FUSED_OPS` member) only for a genuine name+semantics
    // match, documented against Fuel's FusedOps table.
    None
}

/// The `kernel:` name declared inside a [`contract`] block (the `## ` heading
/// title [`bundle`] frames it under), or `None` for a block with no `kernel:`
/// line.
fn kernel_name_of(contract: &str) -> Option<&str> {
    contract
        .lines()
        .find_map(|l| l.trim().strip_prefix("kernel: "))
        .map(str::trim)
}

/// Emit the per-kernel FKC contract block (a ```` ```fkc ```` fenced section) for
/// `kernel`, generated for `op` at structure cell `key` and lowered by
/// `backend`.
///
/// A single graph-`Op` body advertises as a primitive (`op_kind`); a multi-op
/// body advertises as a recognized fusion (`fused_op` + a `pattern:` block).
#[must_use]
pub fn contract(
    op: &OpDef,
    key: &StructureKey,
    kernel: &GeneratedKernel,
    backend: &dyn crate::backend::Backend,
) -> Option<String> {
    // Skip a cell whose dtype has no FKC §5 base-dtype spelling — an unbindable
    // contract would corrupt the planner's honest miss signal (§4.3).
    let dtype = fkc_dtype(key.dtype)?;

    // Increment-1 MULTI-OUTPUT honest miss (typed, no contract; the kernel still
    // generates + works AOT — the cmp/bitwise/Coord precedent). Read against
    // Fuel's actual sources, a multi-output elementwise op has no advertisable
    // identity in the current seam envelope:
    //   1. `op_kind`: no Fuel `OpKind` names an elementwise multi-output dual
    //      (fuel-dispatch fkc/lower.rs `lower_op_kind` has no MulElementwiseBackward
    //      etc.; Fuel splits multi-output backward into per-output OpKinds — e.g.
    //      FlashAttnBackwardQ/K/V — one single-output kernel each).
    //   2. `fused_op`: needs a `pattern:` block, but a `PatternNode`
    //      (fuel-kernel-seam-types) is single-ROOTED — a single-output subgraph;
    //      a forest of N distinct output roots is not expressible, and
    //      `derive_pattern` reads only `op.body` (output 0), so emitting a
    //      contract here would advertise a WRONG single-output pattern that
    //      ignores the other outputs and mis-describes the N-operand key.
    //   3. Fuel's ONLY multi-output ABI (kernel-contract-format.md §5.5, "Option
    //      C") is a `return.bundle` — ONE packed output buffer mapped to a
    //      `FusedOp.output_views` — semantically different from our N DISTINCT
    //      output buffers, and it still needs the fused-op pattern identity (2)
    //      that we cannot provide.
    // So: no contract, until Fuel's seam grows a multi-output region envelope
    // (a forest pattern, or a per-output split). Guarded up front so the wrong
    // single-output pattern is never emitted.
    if op.n_outputs() > 1 {
        return None;
    }

    // Item-01 LAYOUT-VIEW honest miss (typed, no contract; the kernel still
    // generates + runs AOT — the Coord/multi-output/nested-cmp precedent). An
    // address-affecting view (a `Permute`/transpose or `Broadcast` read-through)
    // means the kernel computes `body(transpose(input))`, but our EMITTED pattern
    // grammar cannot say so — verified against Fuel's actual sources:
    //   1. Baracuda's `pattern::PatternNode` is `Op` + `Bind` ONLY — no layout
    //      node and no `OpAttrs` channel. `derive_pattern` walks `op.body`
    //      alone (views live on `OpDef::views`, outside the value walk), so the
    //      pattern it derives describes reading `Input(i)` AT the iteration
    //      coordinate — silently dropping the transpose. Advertising it (a
    //      primitive `op_kind: Relu`, or a `fused_op`) would bind where Fuel's
    //      graph has `relu(transpose(x))`, not `relu(x)`: a wrong bind held off
    //      only by the structure key.
    //   2. Fuel CAN express it honestly, but only in a shape our grammar lacks:
    //      fuel-kernel-seam-types `PatternNode::Op { op: OpTag::Permute, attrs:
    //      OpAttrs { perm } }` with a `perm` guard. Its §4.3 spec
    //      (fkc-fusion-patterns.md) is explicit that a layout op whose ATTRIBUTES
    //      are load-bearing (a transpose's `perm` IS the correctness check) MUST
    //      be matched with `op:` + a `guard:`, NOT via `see_through` — and Fuel's
    //      `see_through` skip is a no-op STUB today anyway (fuel-graph
    //      jit.rs `match_node`'s `SeeThrough` arm). We have no `OpTag`/attrs
    //      vocabulary to author that guard.
    //   3. The concrete-region (decompose) direction rejects layout re-emit
    //      outright (fuel-graph runtime_fused.rs: `Transpose`/`Permute`/`Reshape`
    //      are `UnRepresentable`), so there is no bidirectional identity to bind.
    // So: no contract until Baracuda's pattern grammar grows a layout node + an
    // attrs channel (the `perm` guard). A same-rank `Reshape` / `Identity` view
    // is NOT address-affecting (identity linear map) and still advertises
    // normally — the derived `body`-over-inputs pattern is exactly correct there.
    if crate::plan::op_has_addressing_view(op) {
        return None;
    }

    // BASE_OFFSET honest miss (typed, no contract; the kernel still generates +
    // runs AOT — the views/gather/scatter precedent). An offsetted kernel's
    // entry point carries the `_off…` suffix and its ABI REQUIRES the trailing
    // `long long off{i}` launch scalars, but the FKC accept block has no channel
    // to say so (`layout_spec`'s `start_offset` stays a truthful always-
    // `rejected`, and the frozen JIT envelope has no `off` dispatch slot) — any
    // emitted contract would advertise an ABI Fuel launches WITHOUT the off
    // args: the kernel bumps its base pointers by garbage and reads OOB under a
    // contract that promised no start offset. `derive_pattern`'s
    // `OffsetUnsupported` already misses the plain-elementwise path, but the
    // Model-A gather advert below derives `op_kind` STRUCTURALLY (it never
    // consults the pattern), so an offsetted U32-index gather would sail past
    // that guard — this up-front check is the contract-level gate for EVERY
    // offsetted op (dual-gated with the pattern miss, like the other classes).
    if crate::plan::op_has_offset(op) {
        return None;
    }

    // Increment-4 GATHER honest miss (typed, no contract; the kernel still
    // generates + runs AOT — the Contraction-node precedent). A data-dependent
    // `ReadIndex::Indexed` read cannot be advertised honestly against Fuel's
    // ACTUAL sources:
    //   1. The index operand's dtype is UNKEYABLE. Baracuda's `StructureKey` has
    //      NO per-operand dtype FIELD — a single operand-0 dtype (structure_key.rs:
    //      "v1 assumes a uniform operand dtype"), so nothing in the token names the
    //      index operand as i32 vs i64. (The dtype's byte size DOES leak
    //      incidentally into that operand's `vec_width` — for a full-shape index an
    //      i32 vectorizes wider than an i64 — but that side-channel is unreliable:
    //      it collapses to equal for the 1-D/broadcast index of index_select /
    //      embedding, where both dtypes are `Scalar`. So the token neither reliably
    //      distinguishes nor is meant to distinguish index dtype.) Fuel's gather
    //      admissibility is instead an explicit per-operand dtype TUPLE —
    //      fuel-dispatch `fkc/cpu_link.rs` keys gather/index_select as `[T, U32, T]`
    //      (the `indices` operand is a FIXED U32 slot; `out: passthrough(source)`).
    //      A Baracuda contract keyed only on T would advertise a kernel Fuel could
    //      bind to the wrong index dtype (i32-kernel ↔ i64/U32-call) — no keyed
    //      field guards it.
    //   2. Even setting dtype aside, the emitted `PatternNode` grammar (`Op` +
    //      `Bind`, no `OpAttrs` channel) cannot carry the gather `axis` or the OOB
    //      policy. Fuel DOES name `OpTag::Gather`/`IndexSelect`
    //      (fuel-kernel-seam-types) — but their identity rides `OpAttrs.axis` and,
    //      for residency gathers, a `fdx.gather.kind` admissibility enum
    //      (fuel-dispatch `fkc/validate.rs`) we have no vocabulary to author.
    //   3. Bespoke gather's OOB semantics (silently skip; embedding zero-fills)
    //      also differ from torch/Fuel gather's in-bounds contract — a third
    //      reason the advertised op would mis-describe the kernel.
    // Increment-6 MODEL-A gather contract wiring: the gather honest-miss guard is
    // now SELECTIVE (was unconditional through ramp #5). A gather is honestly
    // advertisable — verified against Fuel's ACTUAL sources — IFF its index dtype
    // is U32:
    //   1. **Index dtype (resolved, Fuel reply 2026-07-04 `docs/fuel-reply-mixed-
    //      dtype-key`).** Fuel keys off the FKC per-operand dtype TUPLE assembled
    //      from `accept.inputs[i].dtype` (`fkc/lower.rs` `assemble_dtype_variants`
    //      → `kernel.rs` binding map), NOT a coarse token — so filling the accept
    //      block honestly (data `T`, index `U32`) makes wrong-bind structurally
    //      impossible with NO `STRUCTURE_KEY_VERSION` bump. Fuel is **U32-index
    //      everywhere** (`fkc/cpu_link.rs` gather/index_select key `[T, U32, T]`);
    //      an i32/i64 index gather is UNREACHABLE from a Fuel graph node, so it
    //      stays an honest miss (`gather_advert` returns `None` → no contract).
    //   2. **op_kind (verified spelling).** Every gather structurally maps to a
    //      REAL Fuel `OpKind` — a full-shape index → `Gather`, a 1-D/broadcast
    //      index → `IndexSelect` (both in `fkc/lower.rs` `lower_op_kind`'s exact
    //      string table; `fuel-ir/dispatch.rs` `OpKind::{Gather,IndexSelect}`).
    //      Reason 2 of the old miss (the `Op`+`Bind` PATTERN grammar can't carry
    //      the gather axis/OOB) is MOOT for a primitive `op_kind:` advert — a
    //      primitive carries no `pattern:` block; the axis rides Fuel's graph node
    //      + `OpParams`, and the OOB rides the new `oob_policy` field (below).
    //   3. **OOB semantics made explicit (Fuel Q3).** Fuel's gather is in-bounds/
    //      `error`; ours skips (gather/index_select) or zero-fills (embedding) —
    //      a genuine mismatch, so the contract advertises it in an `oob_policy`
    //      field (additive/`#[serde(default)]` on Fuel's side; their parser is
    //      permissive — `deny_unknown_fields` deliberately unset — so emitting it
    //      is safe today and load-bearing when Fuel wires the slot in lockstep).
    // A non-u32 gather no longer dies here (increment-7 recipe wiring): Fuel's pinned
    // `gather` RECIPE schema admits index_dtype ∈ {u32,i32,i64}, so an i32/i64 index
    // gather — unreachable as a Fuel graph PRIMITIVE (`op_kind: Gather` keys a fixed
    // U32 slot) — falls through to the recipe-carrying advert (`fused_op:` + a
    // `gather[…]` `semantics:` node, admitted only to a recipe-import peer). The u32
    // gather still takes the op_kind primitive path below; a `None` here is the
    // recipe-path signal, NOT an honest miss. (An un-fkc-spellable DATA dtype already
    // returned `None` at the top via `fkc_dtype(key.dtype)`.)
    let gather_advert = if crate::plan::op_has_gather(op) {
        gather_advert(op, key)
    } else {
        None
    };

    // Increment-6 SCATTER: STILL an unconditional honest miss (the WRITE-side
    // guard is NOT lifted, unlike the read-side gather guard above). Re-verified
    // against Fuel's ACTUAL sources this round; the u32 keying is resolved but
    // insufficient — three independent, source-grounded blockers remain, EACH
    // fatal on its own:
    //   1. **scatter (pure Assign) / bincount / AtomicMax/Min → NO Fuel op_kind.**
    //      `fuel-ir/dispatch.rs` names ONLY `OpKind::{IndexSelect, Gather, IndexAdd,
    //      ScatterAdd}` — there is NO bare `Scatter`, NO `Bincount`/`Histogram`, and
    //      NO scatter-reduce (amax/amin/prod) kind. Net-new vocabulary, a separate
    //      future negotiation.
    //   2. **scatter_add / index_add → OPERAND-ARITY mismatch (the decisive new
    //      finding).** Fuel's `ScatterAdd`/`IndexAdd` key is a FOUR-operand tuple
    //      `[T, U32, T, T]` — (`base`, U32 `indices`, `src`, `passthrough(base)`
    //      out); `fuel-ir/dispatch.rs:375` "Inputs (base, indices, src)",
    //      `fkc/cpu_link.rs` key `[T, U32, T, T]`. Baracuda's scatter_add is
    //      IN-PLACE accumulation (`out += updates`): 2 inputs (updates=src, index)
    //      + an in-place output that DOUBLES as `base` — its honest per-operand
    //      accept tuple is `[T, U32, T]` (3 slots). Fabricating a separate `base`
    //      INPUT to reach 4 slots would misdescribe the kernel ABI (a dishonest
    //      accept block). The assembled key can NOT equal Fuel's — a structural
    //      mismatch, not a spelling gap.
    //   3. **Determinism (unchanged, still an independent blocker for the FP path).**
    //      Fuel wires `ScatterAdd`/`IndexAdd` for FLOAT dtypes only (`cpu_link.rs`:
    //      f32/f64/bf16/f16); a float atomic-add scatter is run-to-run
    //      nondeterministic, and an honest advert would set `determinism:
    //      nondeterministic`, which by `fkc/validate.rs` Rule 9 obligates a coupled
    //      `precision.bit_stable_on_same_hardware: false` + `audited: true` block
    //      Baracuda does not yet author. (Baracuda's deterministic base for the FP
    //      cell is the gather-sum reformulation, whose entry_point does compute
    //      scatter_add semantics — but reasons 2's arity mismatch blocks it anyway.)
    // So NO contract for ANY scatter/scatter_add/index_add/bincount — the honest
    // miss the ramp-#5 conclusion reached, now on firmer (arity + op_kind) ground.
    if crate::plan::op_has_scatter(op) {
        return None;
    }

    // Item-01 / Item-03 LAYOUT-HONESTY honest miss (typed, no contract; the
    // kernel still generates + runs AOT — the multi-output / view / gather
    // precedent). Two per-operand memory-layout facts the emitted five-flag
    // `LayoutSpec` (Fuel `fkc/schema.rs`) CANNOT state truthfully, so the cell is
    // WITHHELD rather than advertised with a layout LIE (the one-directional
    // safety rule: understating is safe, overstating is not):
    //
    //   1. FLIPPED (reverse-stride) operand. The Elementwise schedule has NO
    //      flipped gate (`crate::plan`, unlike the RowReduce/Scan/Window/RowSort
    //      emitters which assert `!flipped`), so the generated kernel reads the
    //      axis FORWARD (`in{k}[i]`) — it does NOT implement the reversed cell its
    //      structure key names. `reverse_strides` is a single accepted|rejected
    //      flag with no "reads-reversed" spelling, and the kernel does not read
    //      reversed anyway, so the only honest posture is to withhold. (Every
    //      EMITTED contract therefore truthfully carries `reverse_strides:
    //      rejected`, and the `accept.structure_key` token no longer contradicts
    //      the layout block.)
    //
    //   2. BAKED-BROADCAST operand. A `Contiguity::Broadcast` operand's stride-0
    //      mask is BAKED into the kernel (fully-broadcast → hoisted `in{k}[0]`,
    //      cuda.rs; partial → its bcast-axis stride terms are compile-time dropped
    //      by `offset_expr`), so the kernel CANNOT walk a contiguous/strided
    //      tensor in that slot. Fuel's tri-state has no `broadcast_stride0:
    //      required` spelling, so the baked-broadcast fact is UNSPEAKABLE — an
    //      honest miss (e.g. a broadcast bias-add cell yields `None`). EXCEPTION:
    //      the ONE Broadcast-class operand we DO advertise (Model-A's deliverable)
    //      is the U32 gather / index_select INDEX operand — its layout is emitted
    //      truthfully by `layout_spec` (`contiguous: required`), not withheld,
    //      because the "broadcast" over the non-gathered axes is the DEFINITION of
    //      index_select, not a caller-varying tensor layout.
    //
    // Verified against the real emitter (cuda.rs `offset_expr` / the fully-
    // broadcast hoist) + `classify_contiguity` (structure_key.rs), 2026-07-08.
    {
        // The gather's index operand is exempt from the broadcast-withhold rule (a
        // 1-D index's stride-0 broadcast over the non-gathered axes is the DEFINITION
        // of index_select/embedding, not a caller-varying layout). Read the operand
        // off the read-index roles — NOT `gather_advert` — so it exempts BOTH a u32
        // op_kind gather AND a non-u32 recipe-carrying index_select.
        let gather_index = crate::plan::gather_of(&op.read_index)
            .map(|(_, index_operand, ..)| index_operand as usize);
        for i in 0..key.n_operands as usize {
            let o = key.operands[i];
            if o.flipped {
                return None;
            }
            if o.contig == Contiguity::Broadcast && Some(i) != gather_index {
                return None;
            }
        }
    }

    // The derived fusion pattern and the neutral KISS-Ops recipe, computed up front
    // so the honesty gates below can consult them.
    //   - `pattern` is `None` when `derive_pattern` REJECTS the body (NoFkcName /
    //     CoordUnsupported / …) — a pattern-DERIVATION miss, distinct from a `Some(p)`
    //     root that merely has no importable Fuel op_kind spelling (AddScalar etc.).
    //   - `recipe` is the KISS-Ops op-DAG ([`crate::recipe::semantics_dag`]), `None`
    //     when any node has no confirmed token. So `recipe.is_some()` ALREADY IS the
    //     "every primitive is in Fuel's resolvable floor" gate — the exhaustive
    //     `unary/binary_kiss_name` + the confirmed source-op leaves never fabricate a
    //     token, and Fuel resolves any named floor op (grammar reply Q6). No separate
    //     floor check is needed; an unconfirmed token → `None` → withheld honest miss.
    let pattern = derive_pattern(op).ok();
    let recipe = crate::recipe::semantics_dag(op);
    // Brief 4 — a PLAIN elementwise op (NOT an indexed gather; that rides the
    // `recipe_carrying` branch below) whose pattern derivation FAILED yet which
    // carries a valid, Fuel-resolvable recipe: e.g. `BitAnd`→`bit_and(in0,in1)`
    // (NoFkcName), `Erfc`→`erfc(in0)` (NoFkcName), `triu`→`mul(in0, cmp_ge(iota(1),
    // …))` (CoordUnsupported). Its importable identity is the RECIPE, so it advertises
    // `fused_op:` + a `semantics:` line — retiring the pattern-miss withhold that
    // `derive_pattern` alone left. UNLIKE the non-elementwise `recipe_carrying`
    // branch, its output shape+dtype = the input's, so it KEEPS the true elementwise
    // return block (`same_as(in0)` + `passthrough(in0)`/`fixed(U8)`).
    let elementwise_recipe_miss = matches!(op.access, crate::ir::Access::Elementwise)
        && !crate::plan::op_has_gather(op)
        && pattern.is_none()
        && recipe.is_some();

    // Increment 0b honesty gate (tightened by the adversarial review): a body
    // containing a Cmp* ANYWHERE emits a contract only as the u8-out
    // single-op primitive.
    //
    // - Float-mask top-level cmp (out_dtype None): a key-dtype-store kernel is
    //   not Fuel's "comparison → U8 mask" op — advertising `op_kind: Lt` would
    //   bind where Fuel expects a 1-byte mask. No contract.
    // - NESTED cmp (e.g. relu-bw `dy * (x > z)`): the emitted pattern would
    //   encode the interior Cmp* as a DIRECT operand of a float op, but no
    //   constructible Fuel graph has that edge — Fuel's compare builders pin
    //   the output to U8 and its binary ops assert operand-dtype equality, so
    //   every real mask-multiply graph interposes `Cast(U8→float)`, which is
    //   outside the §4.1 pattern grammar and NOT in the §4.3 see-through set.
    //   The advertised matcher could therefore never fire on the graphs it
    //   means (silent coverage loss), while the only structurally-matching
    //   graphs would be all-U8 mask arithmetic — a wrong bind held off only by
    //   the structure key. Withheld until `Cast` joins the pattern
    //   vocabulary; the kernel itself still generates (AOT + seam lowering
    //   are unaffected).
    // - EXEMPTION (Brief 4): an `elementwise_recipe_miss` is exempt. The Cast gap
    //   is a PATTERN-grammar limit, but this op's identity is the RECIPE, which
    //   expresses the cmp honestly (`cmp_ge` is a floor op Fuel resolves + verifies
    //   numerically; the recipe is dtype-agnostic and realized at the accept dtypes),
    //   so a Coord/cmp body like `triu` rides the recipe path. A cmp body whose
    //   pattern DERIVES (gt_mask, relu_bw) is NOT a pattern miss, so it stays withheld.
    let out_u8 = op.out_dtype == Some(ElementKind::U8);
    if expr_contains_cmp(&op.body) && !out_u8 && !elementwise_recipe_miss {
        return None;
    }

    // WHERE/SELECT honesty gate (its own guard, per the house rule that advert
    // paths need independent gating — the pattern-layer `SelectUnsupported`
    // miss does NOT substitute): a body containing a Select ANYWHERE has its
    // contract withheld wholesale. The single-op `Where` advert needs the
    // Model-A per-operand dtype tuple (Fuel binds `[U8, T, T, T]` — the cond
    // is strictly a U8 tensor, inexpressible under uniform-dtype keying), and
    // the fused-cmp form needs fuel-side matcher validation first
    // (propose-first channel). Load-bearing beyond belt-and-suspenders: the
    // Model-A gather advert below derives `op_kind` STRUCTURALLY (it never
    // consults the pattern), so a select body inside a u32-index gather would
    // otherwise sail past the pattern miss and advertise. The kernel still
    // generates + runs AOT — only the contract is withheld.
    if expr_contains_select(&op.body) {
        return None;
    }

    // Recipe-carrying = the op's IMPORTABLE identity is the recipe, not a pattern /
    // op_kind. Fires for (a) a NON-elementwise op (contraction/reduction/scan/
    // rowreduce — its output shape/dtype ≠ any input's, so the FKC same_as/passthrough
    // rules can't state them, and the realized recipe is the authority — Fuel's shape
    // answer, 2026-07-17), and (b) a data-dependent GATHER whose index is not a Fuel
    // graph primitive (a non-u32 index — `gather_advert` is `None`, so it does not ride
    // the op_kind path): its out shape = the index shape ≠ same_as(in0), so it too
    // defers shape to the recipe. Both OMIT `shape_rule`. EXCLUDES a u32 gather
    // (`gather_advert` Some → op_kind primitive). A PLAIN elementwise pattern-miss is
    // handled separately by `elementwise_recipe_miss` (Brief 4) — same `fused_op:` +
    // `semantics:` advert, but it KEEPS the elementwise `same_as(in0)` return block
    // (out shape/dtype = the input's). Scatter never reaches here (withheld above).
    let recipe_carrying = recipe.is_some()
        && gather_advert.is_none()
        && (!matches!(op.access, crate::ir::Access::Elementwise) || crate::plan::op_has_gather(op));
    let n_ops = pattern.as_ref().map_or(0, count_ops);
    // A u8-out predicate advertises ONLY as the single-op primitive; a FUSED
    // u8-out body would hit the same missing-Cast pattern problem above. This is a
    // PATTERN-grammar limit, so a recipe-carrying op (whose identity is the recipe,
    // not a pattern) is exempt — a non-elementwise `recipe_carrying` op's u8 output
    // rides the recipe dtype (e.g. a hetero-out `any` reduction), and an elementwise
    // `elementwise_recipe_miss` op emits the true `fixed(U8)` return rule (out dtype =
    // U8, out shape = in0).
    if out_u8 && n_ops != 1 && !recipe_carrying && !elementwise_recipe_miss {
        return None;
    }
    let is_fusion = n_ops > 1;
    let op_line = if let Some(g) = &gather_advert {
        // A u32-index gather advertises the verified Fuel primitive OpKind
        // (`Gather` / `IndexSelect`) — `derive_pattern` intentionally rejects a
        // gather (its `Op`+`Bind` grammar can't carry the axis), so the gather
        // op_kind is derived structurally, not from a pattern. A primitive advert
        // carries NO `pattern:` block (the axis rides Fuel's graph node), so the
        // pattern-grammar gap is moot here.
        format!("op_kind: {}", g.op_kind)
    } else {
        match &pattern {
            // exactly one graph op → a primitive identity. BOTH the comparison
            // predicates (via `cmp_dispatch_op_kind`) and the arithmetic
            // primitives (via `fuel_primitive_op_kind`) emit the DISPATCH OpKind
            // spellings Fuel's importer accepts (`AddElementwise`,
            // `LessElementwise`, …) — NOT the internal pattern spellings
            // (`Add`, `Lt`, …) `root_op_name` yields. This is load-bearing for
            // the WHOLE bundle: Fuel's FKC importer (fuel-dispatch
            // `fkc/lower.rs` `lower_op_kind`) is an exhaustive string table that
            // typed-rejects unknown names with `UnknownOpKind`; `fkc/validate.rs`
            // `validate_file` propagates it with `?`; and `fkc/register.rs`
            // `import_bundle_str` runs `validate_file(&file)?` over the whole
            // bundle — so ONE unknown op_kind fails the ENTIRE import, poisoning
            // every correctly-spelled contract beside it. An unmapped primitive
            // is therefore WITHHELD (honest miss), never emitted with a raw
            // internal spelling (see `fuel_primitive_op_kind`).
            Some(_) if n_ops == 1 && out_u8 => {
                format!("op_kind: {}", cmp_dispatch_op_kind(&op.body))
            }
            Some(p) if n_ops == 1 => {
                // A single-op root with no importable Fuel `OpKind` — e.g.
                // `AddScalar`/`MulScalar`: Fuel has no scalar-param OpKind (it
                // maps `Op::AddScalar`/`MulScalar` onto the `Affine` kernel
                // `y = a*x + b`, whose op_params + `extract:` scalar routing
                // live only inside a `pattern:`/`fused_op` block, not a bare
                // primitive contract). Withheld (`?` → `None`) rather than emit
                // an unimportable name that would poison the bundle; the kernel
                // still generates + lowers AOT.
                let spelling = fuel_primitive_op_kind(&root_op_name(p))?;
                format!("op_kind: {spelling}")
            }
            // ≥2 graph ops (or a bare n_ops==0 copy) → a fused identity. A fusion
            // that genuinely maps to a Fuel FusedOp emits the exact SCREAMING_SNAKE
            // constant (so it imports through `bundle()`); otherwise it carries the
            // op's free-form stable name, which `bundle()` WITHHOLDS from an
            // importable bundle (an unknown `fused_op:` is bundle-fatal) while the
            // JIT seam still ships it as a bare, unparsed block. See `bundle` /
            // `fuel_fused_op_name`.
            Some(_) => match fuel_fused_op_name(&op.name) {
                Some(fuel_const) => format!("fused_op: {fuel_const}"),
                None => format!("fused_op: {}", op.name),
            },
            // body not expressible as a pattern (Const / non-elementwise / bind
            // mismatch, or an elementwise NoFkcName/CoordUnsupported miss). If the op
            // carries a neutral KISS-Ops recipe — a NON-elementwise op / non-u32 gather
            // (`recipe_carrying`) OR a plain elementwise pattern-miss
            // (`elementwise_recipe_miss`, Brief 4) — advertise it as a recipe-carrying
            // fused op: bundle() still WITHHOLDS it (recipe_import=false ⇒
            // contract_admissible false), and bundle_kisc admits it ONLY to a
            // recipe-import peer via the `semantics:` line below (contract_carries_recipe).
            // Otherwise skip — an honest miss rather than a faked op_kind from the
            // free-form name.
            None if recipe_carrying || elementwise_recipe_miss => {
                format!("fused_op: {}", op.name)
            }
            None => return None,
        }
    };

    let out_idx = key.n_operands.saturating_sub(1) as usize;
    let params = params_used(&op.body);
    let (prec_mode, prec_ulp) = precision_of(&op.body);

    let mut s = String::from("```fkc\n");
    s.push_str(&format!("kernel: {}_{}\n", op.name, cell_suffix(key)));
    s.push_str(&op_line);
    s.push('\n');
    s.push_str(&format!(
        "blurb: \"{}\"\n",
        blurb(op, key, dtype, is_fusion)
    ));
    // ImplId tuple (FKC §4.11), kept as five separable fields. The backend
    // token is canonicalized to Fuel's capitalized spelling (`cuda` → `Cuda`;
    // see `fkc_backend_token`) so the block imports through `lower_backend`.
    s.push_str(&format!("backend: {}\n", fkc_backend_token(backend.name())));
    // `kernel_source:` is the PROVIDER identity (provenance), supplied by the
    // backend — a neutral generator cannot hardcode who provides it. Baracuda's
    // CUDA backend reports `baracuda` (byte-identical to the pre-abstraction
    // literal); the in-tree reference backends report the generator's name.
    s.push_str(&format!("kernel_source: {}\n", backend.provider()));
    s.push_str(&format!("dtypes: [{dtype}]\n"));
    s.push_str(&format!("entry_point: {}\n", kernel.name));
    s.push_str(&format!(
        "kernel_revision_hash: \"{:016x}\"\n",
        revision_hash(&kernel.source)
    ));

    // oob_policy (Model-A gather, Fuel Q3): advertise the out-of-bounds semantics
    // EXPLICITLY so the skip/zero_fill vs Fuel's `error` mismatch is contract-
    // visible (Fuel wires the schema slot + import validation in lockstep). Only a
    // gather carries it; a uniform op omits it (byte-identical, no new field).
    if let Some(g) = &gather_advert {
        s.push_str(&format!("oob_policy: {}\n", g.oob_policy));
    }

    // accept — the admissibility predicate IS the structure key (the honesty
    // invariant); the per-input dtype/layout lines are a human-readable gloss.
    s.push_str("accept:\n");
    s.push_str(&format!("  structure_key: \"{}\"\n", key.to_token()));
    s.push_str("  inputs:\n");
    // The gather's index operand + its REAL FKC dtype token — read off the read-index
    // roles so it is honest on BOTH the u32 op_kind path (U32, Fuel assembles the
    // mixed-dtype key `[T, U32, T]`) AND the non-u32 recipe path (I32/I64, never the
    // data dtype). `None` for a non-gather op ⇒ every input emits the uniform key
    // `dtype` (byte-identical to the pre-Model-A emission).
    let gather_index_slot: Option<(usize, &str)> =
        crate::plan::gather_of(&op.read_index).and_then(|(_, index_operand, _, _, index_dtype)| {
            fkc_dtype(index_dtype).map(|tok| (index_operand as usize, tok))
        });
    for i in 0..op.n_inputs as usize {
        let in_dtype = match gather_index_slot {
            Some((io, tok)) if io == i => tok,
            _ => dtype,
        };
        // Fuel's FKC `TensorDesc` carries the operand dtype as the PLURAL
        // `dtypes: [..]` list (fuel-dispatch `fkc/schema.rs` `TensorDesc.dtypes:
        // Vec<String>`), NOT a singular `dtype:` — and `deny_unknown_fields` is
        // OFF, so a singular `dtype:` line is silently dropped and the operand
        // resolves to an EMPTY dtype set → `BadScalarType` at import. Emit the
        // one-element plural list so Fuel's `resolve_operand_dtypes` /
        // `assemble_dtype_variants` actually key on it (the Model-A per-operand
        // dtype is only conveyed this way). Review-confirmed against Fuel's real
        // `import_bundle_str` (singular → BadScalarType; plural → Ok).
        //
        // `name: in{i}` — a stable, index-based operand role (Fuel's
        // `TensorDesc.name`; `assemble_dtype_variants` reads it, and the output
        // `passthrough(in0)` rule references it by name). Index-based, not
        // lhs/rhs: the emitter does not invent operand semantics. `layout:` is
        // the five-flag `LayoutSpec` inline map (see `layout_spec`), the form
        // Fuel's `TensorDesc.layout: Option<LayoutSpec>` deserializes — a bare
        // `layout: contiguous` string is a serde type error at import.
        s.push_str(&format!(
            "    - name: in{i}\n      dtypes: [{in_dtype}]\n      layout: {}\n",
            layout_spec(key, i)
        ));
    }

    if !params.is_empty() {
        // A non-empty `params` set ONLY ever occurs on a scalar-param body, which
        // is either a standalone `AddScalar`/`MulScalar` (n_ops==1, an unmapped
        // honest miss → no contract) or a fusion carrying `param(i)` (n_ops≥2 →
        // `fused_op:`, which `bundle()` WITHHOLDS from importable bundles, as no
        // Baracuda fusion maps to a Fuel FusedOp). So this top-level `op_params:`
        // sequence — which is NOT Fuel's schema shape (Fuel nests op params under
        // `accept.op_params: { variant, fields }`) — never reaches Fuel's parser
        // through a bundle; it rides only the JIT seam, where the contract text is
        // stored UNPARSED. It is retained as human-readable documentation of the
        // fusion's scalar `extract:` carriers.
        s.push_str("op_params:\n");
        for p in &params {
            // Scalar params are launch arguments in the op's SCALAR COMPUTE dtype
            // (the `extract:` carrier): `F32` for an f32 op (byte-identical to the
            // pre-F64 hardcode), `F64` for an f64 op (the F64-param increment). This
            // reuses the same FKC `dtype` token the accept block spells at `dtypes:`.
            s.push_str(&format!("  - name: param{p}\n    dtype: {dtype}\n"));
        }
    }

    s.push_str("return:\n  outputs:\n");
    // layout_guarantee (§5.3): a contiguous output cell guarantees a packed
    // contiguous buffer, else the always-true `preallocated` (the executor
    // pre-allocates it). The five-flag `LayoutSpec` map is the ACCEPT-input
    // `TensorDesc.layout` form only; on an OUTPUT it's a silently-dropped unknown
    // key. Verified 2026-07-08 against Fuel's parser.
    let layout_guarantee = match key.operands[out_idx].contig {
        Contiguity::Contig => "contiguous",
        _ => "preallocated",
    };
    // The FKC `OutputDesc` rule fields (§5.1/§5.2), all omittable (serde default).
    // Fuel confirmed the exact grammar 2026-07-17: `dtype_rule: passthrough(<role>)
    // | fixed(<DType>)`; `shape_rule: same_as(<role>) | from_params(<field>, …)` —
    // there is NO `from_recipe` form.
    //
    // RECIPE-CARRYING non-elementwise op (contraction/reduction/scan): the realized
    // recipe is Fuel's single shape+dtype authority (`primitive_shape → (shape,
    // dtype)`). SHAPE — no FKC `shape_rule` form states a non-elementwise output
    // shape (a matmul's out `[M,N]` ≠ any input), and `shape_rule` is a claim Fuel
    // VERIFIES against the recipe, not a shape authority. Fuel's `eval_shape_rule`
    // (fuel-dispatch return_check.rs, commit b1c33f91) now EVALUATES `same_as(role)`
    // and `ShapeRuleMismatch`es a claim that disagrees with the realized shape
    // (`from_params` stays a stub) — so a `same_as(in0)` here would be a FALSE claim;
    // it is OMITTED and the recipe carries the shape. DTYPE — `dtype_rule` IS
    // interpreted (it builds the binding-key output slot), so it is emitted: a
    // hetero output declares `fixed(<dtype>)`, a uniform output `passthrough(in0)`.
    //
    // ELEMENTWISE op: a basis whose shape rule IS `same_as(in0)` (`passthrough(role)`
    // → `DtypeRule::Passthrough`, then `resolve_output_slot_dtype` appends the output
    // slot; `fixed(U8)` for a u8-predicate). Byte-identical to the pre-recipe emit.
    if recipe_carrying {
        let dtype_rule = match op.out_dtype.and_then(fkc_dtype) {
            Some(d) => format!("fixed({d})"),
            None => "passthrough(in0)".to_string(),
        };
        s.push_str(&format!(
            "    - dtype_rule: {dtype_rule}\n      \
             layout_guarantee: {layout_guarantee}\n      \
             aliasing: none\n"
        ));
    } else {
        let dtype_rule = if out_u8 {
            "fixed(U8)"
        } else {
            "passthrough(in0)"
        };
        // A u32-index gather/index_select/embedding rides the op_kind path
        // (`gather_advert` Some ⇒ NOT recipe_carrying), so it lands here — but its
        // output shape is the INDEX shape, not `same_as(in0=data)` (the SAME reason
        // the i32/i64 gather recipe path omits shape_rule; its own `semantics:
        // gather[…]` node is the shape authority). Emitting `same_as(in0)` is a false
        // claim Fuel's `eval_shape_rule` (return_check.rs) can `ShapeRuleMismatch` on
        // a resized-axis gather — so OMIT it. `dtype_rule` (the gathered data dtype)
        // stays correct. Every true-elementwise cell (out shape == in0) keeps
        // `shape_rule: same_as(in0)` byte-identically.
        let shape_rule_line = if gather_advert.is_some() {
            String::new()
        } else {
            "shape_rule: same_as(in0)\n      ".to_string()
        };
        s.push_str(&format!(
            "    - dtype_rule: {dtype_rule}\n      {shape_rule_line}\
             layout_guarantee: {layout_guarantee}\n      \
             aliasing: none\n"
        ));
    }

    s.push_str("caps:\n");
    // in_place: ALWAYS `false`. Two Fuel-side facts, both verified 2026-07-08:
    //   (1) SCHEMA. `CapsBlock.in_place` is `Option<bool>` (fuel-dispatch
    //       `fkc/schema.rs`, "Booleans are literal true/false") — the
    //       pre-reconcile `allowed`/`forbidden` STRINGS were a serde type error
    //       that failed the whole bundle.
    //   (2) SEMANTICS (the inversion the review caught). Fuel spec §4.6:
    //       `caps.in_place: true` declares the kernel WRITES ITS OUTPUT INTO AN
    //       INPUT BUFFER (output aliases input N), and the planner then treats it
    //       as consuming-and-producing the SAME Storage — no separate output
    //       alloc. Baracuda's generated elementwise kernels are OUT-OF-PLACE:
    //       every input AND the output pointer carry `__restrict__` (cuda.rs), so
    //       an actual out==in invocation — exactly what §4.6's planner behavior
    //       produces — is formally UB, NOT merely tolerated. The same contract
    //       emits `aliasing: none` ("fresh output buffer", §5.4), which §5.4 pairs
    //       with `in_place: false`. So `true` LIED about a kernel fact (and would
    //       clobber a fan-out>1 input once Fuel wires the §4.6 consumer). FKC has
    //       no "aliasing merely tolerated" flag, so nothing is lost. This is
    //       exactly Fuel's corpus-wide `in_place: false` posture for out-of-place
    //       kernels (the only corpus `true` is the genuinely destructive
    //       Clamp/PowI-inplace / linear-quant surface).
    s.push_str("  in_place: false\n");
    s.push_str(&format!("  alignment_bytes: {}\n", required_align(key)));
    // The awkward-layout strategy. Fuel's schema field is
    // `caps.awkward_layout_strategy` (`fkc/schema.rs` `CapsBlock`); the
    // pre-reconcile `awkward_layout:` key was an UNKNOWN field silently dropped
    // (`deny_unknown_fields` is off), so Baracuda's strategy never reached Fuel.
    // The value strings (`requires_contiguous`/`handles_strided`) are already
    // Fuel-exact — only the key name was wrong.
    s.push_str(&format!(
        "  awkward_layout_strategy: {}\n",
        awkward_layout(key)
    ));
    // The unit of the kernel's `n` launch argument: a vectorized/packed cell
    // counts w-element VECTORS (n/width), everything else elements. Pinned on
    // the wire per the 2026-07-03 Fuel exchange (documentation-only for Fuel
    // today — launches are provider-internal — but load-bearing for their
    // declared-cost trampoline, and an 8x launch/cost hazard if ever assumed).
    let plan = crate::plan::build_plan(op, key);
    let cw = backend.effective_count_width(&plan);
    if cw > 1 {
        s.push_str(&format!("  count_unit: vectors_x{cw}\n"));
    } else {
        s.push_str("  count_unit: elements\n");
    }

    s.push_str("cost:\n");
    s.push_str("  provenance: declared\n");
    s.push_str("  class: elementwise\n");
    // Fuel's `CostBlock` (fuel-dispatch `fkc/schema.rs`) carries `flops` /
    // `bytes_moved` as EXPRESSION STRINGS over the §4.4 symbol vocabulary — NOT
    // the pre-reconcile `flops_per_elem` / `bytes_per_elem` scalars, which are
    // unknown keys silently dropped (both → `CompiledCostExpr::Unknown`, the
    // declared cost never reaches dispatch). `n` is the §4.4 output-element count
    // (`fkc/cost_expr.rs` `bind_cost_symbols` binds it); an elementwise cell does
    // `flops_per_elem` flops and moves `bytes_per_elem` bytes per element, so the
    // shape-parameterized expression is `<coeff> * n`. Parse-validated by Fuel's
    // `compile_cost` (verified 2026-07-08: `"1 * n"` / `"12 * n"` → Expr, imports
    // Ok). `class: elementwise` keeps the block above the §8a placeholder gate.
    s.push_str(&format!("  flops: \"{} * n\"\n", count_flops(&op.body)));
    s.push_str(&format!(
        "  bytes_moved: \"{} * n\"\n",
        bytes_per_elem(op, key)
    ));

    // Precision → Fuel's `PrecisionBlock` vocabulary ONLY (fuel-dispatch
    // `fkc/schema.rs`: bit_stable_on_same_hardware / max_ulp / max_relative /
    // max_absolute / audited / notes). The pre-reconcile `mode:` key was NON-
    // SCHEMA — silently dropped, and (for the unbounded floored-`Rem` case, whose
    // block was `mode:`-ONLY) a whole-bundle-FATAL `PlaceholderPrecision` (an
    // all-null block has nothing to lower; `fkc/precision.rs`). Re-expressed:
    //   - bit_stable_on_same_hardware: DERIVED from the `determinism: bitwise`
    //     fact below — deterministic vendor math is, a fortiori, bit-stable on the
    //     SAME hardware (the weaker of the two claims). True for every cell here.
    //   - max_ulp: the declared ULP upper bound when finite (0 ⇒ correctly
    //     rounded / bit-reproducible); OMITTED for the unbounded floored-`Rem`
    //     case (no finite result-ULP number bounds a boundary flip).
    //   - audited: true — the bounds are declared against CUDA vendor ULP tiers.
    //   - notes: carries the old `mode:` semantics (and, for the unbounded case,
    //     the reason the bound is absent — Fuel lowers audited+bit_stable+no-bound
    //     to a populated guarantee, not the fatal placeholder).
    s.push_str("precision:\n");
    s.push_str("  bit_stable_on_same_hardware: true\n");
    if let Some(u) = prec_ulp {
        s.push_str(&format!("  max_ulp: {u}\n"));
    }
    // KISS-Contract §6.8 `accumulation_type` (sk3 RFC §4.2): a contraction/
    // reduction-bearing cell declares the dtype its fold accumulates in,
    // spelled from the SAME closed dtype set as the key's `<acc>` coordinate
    // (`baracuda_kernel_vocab::dtype_token` — one dtype, two surfaces). A pure
    // elementwise/permutation cell has no fold, so the field is absent.
    let acc_dtype = match plan.access {
        // Gem cells: the key's `<acc>` IS the declaration — emitting from the
        // key makes the §4.2 contract↔key consistency pin hold by
        // construction (and `emit_contraction` accumulates in exactly this
        // dtype: float for f16/bf16/f32/f32-strict, double for f64).
        crate::ir::Access::Contraction { .. } => key.contraction.map(|c| c.acc),
        // Reduction-class cells (no key coordinate — the §6.8 declaration is
        // the only surface): the generated fold's accumulator rule, mirroring
        // the cuda.rs emitters — F64/F32Strict fold in double, integers fold
        // native (wrapping), every other float folds up-converted in float.
        crate::ir::Access::Reduction { .. }
        | crate::ir::Access::RowReduce { .. }
        | crate::ir::Access::Scan { .. }
        | crate::ir::Access::Window { .. } => Some(match plan.dtype {
            ElementKind::F64 | ElementKind::F32Strict => ElementKind::F64,
            dt if crate::plan::is_int_dtype(dt) => dt,
            _ => ElementKind::F32,
        }),
        _ => None,
    };
    if let Some(a) = acc_dtype {
        s.push_str(&format!(
            "  accumulation_type: {}\n",
            baracuda_kernel_vocab::dtype_token(a)
        ));
    }
    s.push_str("  audited: true\n");
    s.push_str(&format!(
        "  notes: \"{}\"\n",
        precision_notes(prec_mode, prec_ulp)
    ));
    s.push_str("determinism: bitwise\n");

    if let (Some(p), true) = (&pattern, is_fusion) {
        s.push_str(&to_fkc(p));
    }

    // Neutral KISS-Ops Semantics recipe (KISS-Contract §2.3), emitted when the op
    // re-bases onto confirmed KISS-Ops tokens — the decomposition a recipe-verify
    // importer runs, and what [`contract_carries_recipe`] detects to admit an
    // otherwise-unknown fused op to a recipe-import peer. Format PROVISIONAL —
    // see [`crate::recipe`].
    if let Some(r) = &recipe {
        s.push_str(&format!("semantics: {r}\n"));
    }

    s.push_str("```\n");
    Some(s)
}

// ---------------------------------------------------------------------------
// Model-A gather advertisement (increment 6)
// ---------------------------------------------------------------------------

/// The Fuel-facing advert facts for a u32-index gather, or `None` for a gather
/// that stays an honest miss (a non-u32 index → the recipe path instead). Computed
/// once at the top of [`contract`] and threaded into the op_kind line and the
/// `oob_policy` field. (The index operand + its accept dtype are read separately off
/// the read-index roles at the accept block — honest for the recipe path too.)
struct GatherAdvert {
    /// The verified Fuel primitive `OpKind` spelling — `"Gather"` (full-shape
    /// index) or `"IndexSelect"` (1-D / broadcast index). Both are exact strings
    /// from `fuel-dispatch fkc/lower.rs` `lower_op_kind`.
    op_kind: &'static str,
    /// The `oob_policy` value — `skip` (gather / index_select) / `zero_fill`
    /// (embedding) / `clamp` (generator-only). Fuel's own gather is `error`.
    oob_policy: &'static str,
}

/// Decide whether a gather `op` at cell `key` is honestly advertisable, and with
/// what op_kind / oob_policy. `None` ⇒ honest miss (a non-U32 index — Fuel is
/// U32-index everywhere, so i32/i64 can't bind — or an un-fkc-spellable index).
///
/// The op_kind is derived STRUCTURALLY from the index operand's broadcast mask
/// (matching the [`crate::ir::ReadIndex::Indexed`] doc: a full-shape index →
/// torch-`Gather`; a 1-D/broadcast index → `IndexSelect` / embedding), NOT from
/// the op's free-form `name` (which is not a dispatch key).
fn gather_advert(op: &OpDef, key: &StructureKey) -> Option<GatherAdvert> {
    let (_g, index_operand, axis, oob, index_dtype) = crate::plan::gather_of(&op.read_index)?;
    // Fuel keys the index operand as a FIXED U32 slot (`fkc/cpu_link.rs` `[T, U32,
    // T]`); an i32/i64 index gather is unreachable from a Fuel graph node, so it
    // stays an honest miss (AOT-only — the kernel still generates + runs).
    if index_dtype != ElementKind::U32 {
        return None;
    }
    let op_kind = if index_is_1d(key, index_operand as usize, axis) {
        "IndexSelect"
    } else {
        "Gather"
    };
    let oob_policy = match oob {
        OobPolicy::Skip => "skip",
        OobPolicy::ZeroFill => "zero_fill",
        OobPolicy::Clamp => "clamp",
    };
    Some(GatherAdvert {
        op_kind,
        oob_policy,
    })
}

/// `true` if the index operand is a **1-D / broadcast** index (index_select /
/// embedding), `false` for a **full-shape** index (torch-gather). A 1-D index
/// broadcasts over every iteration axis EXCEPT the gathered `axis`; a full-shape
/// index broadcasts none. Reads the index operand's broadcast mask off the
/// structure key (the same fact the emitter's index-offset degeneration uses).
fn index_is_1d(key: &StructureKey, index_operand: usize, axis: u8) -> bool {
    let m = key.operands[index_operand].bcast;
    (0..key.rank).all(|d| if d == axis { !m.is_set(d) } else { m.is_set(d) })
}

// ---------------------------------------------------------------------------
// Pattern shape helpers
// ---------------------------------------------------------------------------

fn count_ops(node: &PatternNode) -> usize {
    match node {
        PatternNode::Bind(_) => 0,
        PatternNode::Op { operands, .. } => 1 + operands.iter().map(count_ops).sum::<usize>(),
    }
}

#[doc(hidden)]
pub fn root_op_name(node: &PatternNode) -> String {
    match node {
        PatternNode::Op { op, .. } => op.clone(),
        PatternNode::Bind(_) => "Identity".to_string(),
    }
}

/// Map an internal single-op pattern-root spelling ([`root_op_name`], e.g.
/// `"Add"`, `"Relu"`, `"GeluErf"`) to the EXACT `op_kind` string Fuel's FKC
/// importer accepts, or `None` for a root with no importable Fuel `OpKind` (an
/// honest miss — the contract is then withheld).
///
/// # Source anchor
/// Every right-hand side below is a verbatim arm of fuel-dispatch
/// `src/fkc/lower.rs` `lower_op_kind` — the exhaustive `&str -> OpKind` table
/// (read 2026-07-08). The `fuel_primitive_op_kind_outputs_are_accepted_fuel_strings`
/// test cross-checks every output against an independently-copied verbatim
/// snapshot of that table, so a spelling drift fails the build rather than
/// poisoning an import at runtime.
///
/// # Blast radius (why an unmapped root is WITHHELD, not force-emitted)
/// `lower_op_kind` returns `Err(UnknownOpKind)` on any unknown name;
/// `fkc/validate.rs` `validate_file` propagates it with `?`; `fkc/register.rs`
/// `import_bundle_str` runs `validate_file(&file)?` over the WHOLE bundle — so a
/// single bad `op_kind:` line fails the entire import, taking down every
/// correctly-spelled contract sharing the bundle. Emitting the raw internal
/// spelling (the pre-fix bug) made every bundle containing an arithmetic
/// primitive unimportable. Hence `Some` → emit, `None` → withhold.
///
/// # Semantic reconciliations (confirmed against Fuel's sources, not assumed)
/// - `Rem` → `RemElementwise`: Baracuda pins FLOORED remainder (the 0.10.2
///   reconcile) and Fuel's `RemElementwise` is the PyTorch convention
///   `a - floor(a/b)*b` (sign of the divisor) — fuel-dispatch
///   `tests/cuda_dispatch_live.rs` ("PyTorch convention", `-5 % 3 = 1`,
///   `5 % -3 = -1`). Same semantics.
/// - `Step` → `StepElementwise`: both are the `x > 0` Heaviside step (1.0 where
///   `x > 0`, else 0.0) — fuel-dispatch `tests/{cuda,vulkan}_dispatch_live.rs`
///   ("Heaviside step (1.0 where x > 0, else 0.0)"). Same semantics.
/// - `GeluErf` → `GeluErfElementwise` (exact-erf GELU), NOT `GeluElementwise`
///   (the tanh approximation). Baracuda's `UnaryOp::Gelu` lowers to exact erf
///   (`cuda.rs`, aec3bf7 provenance) and `pattern.rs` `unary_name` emits the
///   `GeluErf` root, so the exact flavor is carried end to end.
/// - `Relu` → `ReluElementwise`: mapping RESTORED per Fuel's 2026-07-08 DECISION
///   (`ReluElementwise` = NaN-PROPAGATING, torch parity). Baracuda's synthesized
///   relu is NaN-propagating; Fuel's CPU + CUDA rebind to the propagating
///   convention has now LANDED (2026-07-09, Fuel `main`: `772e27a0` CPU
///   relu/max/min, `00b25dc0` CUDA `ReluElementwise`, `5d52ee82` CUDA
///   `ReluInplace` — all rebound to our alpha.76 `unary_relu_propagating_*`
///   family, verified additive-only FFI). Both slots now agree on NaN, so a JIT
///   adopt is behaviorally identical — the convention is fully reconciled
///   (forward + in-place, CPU + CUDA); no transient divergence remains.
///
/// `AddScalar`/`MulScalar` are deliberately absent: Fuel has no scalar-param
/// primitive `OpKind` (see the branch note in [`contract`]) — they are honest
/// misses as a standalone single-op advert, and ride the `pattern:` block
/// (with `extract:` scalar routing) when they appear inside a larger fusion.
#[doc(hidden)]
pub fn fuel_primitive_op_kind(root: &str) -> Option<&'static str> {
    Some(match root {
        // Infix binary (tensor ⊗ tensor) primitives.
        "Add" => "AddElementwise",
        "Sub" => "SubElementwise",
        "Mul" => "MulElementwise",
        "Div" => "DivElementwise",
        // Non-infix binary primitives.
        "Maximum" => "MaximumElementwise",
        "Minimum" => "MinimumElementwise",
        "Pow" => "PowElementwise",
        "Rem" => "RemElementwise",
        // Unary primitives.
        "Neg" => "NegElementwise",
        "Abs" => "AbsElementwise",
        "Sqr" => "SqrElementwise",
        "Sqrt" => "SqrtElementwise",
        "Rsqrt" => "RsqrtElementwise",
        "Recip" => "RecipElementwise",
        "Exp" => "ExpElementwise",
        "Log" => "LogElementwise",
        "Tanh" => "TanhElementwise",
        "Sigmoid" => "SigmoidElementwise",
        // `Relu` → `ReluElementwise`: the withhold is LIFTED (Fuel's 2026-07-08
        // consolidated answer) and the reconciliation is now FULLY CLOSED. The
        // earlier hold was a real semantic divergence — our synthesized relu is
        // NaN-PROPAGATING (`x < 0 ? 0 : x`, torch.relu; cuda.rs pins it) while
        // Fuel's `ReluElementwise` slot then NaN-SCRUBBED in all three of its
        // authorities (CPU `x.max(0.0)`, the FKC doc "NaN-as-missing", the
        // incumbent CUDA `fmaxf` kernel). Fuel DECIDED `ReluElementwise` =
        // NaN-propagating (torch parity; "external convention over internal
        // consistency") and its CPU-core + FKC-doc + CUDA rebind to our bespoke
        // NaN-propagating kernel (crates/baracuda-kernels-sys
        // `unary_relu_propagating_fp.cu`) have now LANDED on Fuel `main`
        // (2026-07-09: `772e27a0` CPU relu/max/min, `00b25dc0` CUDA
        // `ReluElementwise`, `5d52ee82` CUDA `ReluInplace`; the incumbent `fmaxf`
        // one stays as the separate Fmax family). Both slots now agree on NaN, so
        // a JIT adopt is behaviorally identical — no transient scrub divergence
        // remains (verified Fuel-side by dispatch-level pins born red against a
        // live-sabotaged binding: cuda_relu_propagates_nan_*, live suite 173/173).
        "Relu" => "ReluElementwise",
        "Erf" => "ErfElementwise",
        "GeluErf" => "GeluErfElementwise",
        "Silu" => "SiluElementwise",
        "Sin" => "SinElementwise",
        "Cos" => "CosElementwise",
        "Floor" => "FloorElementwise",
        "Ceil" => "CeilElementwise",
        "Round" => "RoundElementwise",
        "Sign" => "SignElementwise",
        "Step" => "StepElementwise",
        // `AddScalar`/`MulScalar` and any other spelling have no importable
        // primitive `OpKind` — honest miss (see the fn-level docs).
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Body scans
// ---------------------------------------------------------------------------

/// Sorted, unique runtime-parameter indices referenced by the body.
fn params_used(body: &ScalarExpr) -> Vec<u8> {
    let mut out = Vec::new();
    scan_params(body, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

fn scan_params(e: &ScalarExpr, out: &mut Vec<u8>) {
    match e {
        ScalarExpr::Param(i) => out.push(*i),
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => {}
        ScalarExpr::Unary(_, x) => scan_params(x, out),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => {
            scan_params(a, out);
            scan_params(b, out);
        }
        ScalarExpr::Select(c, a, b) => {
            scan_params(c, out);
            scan_params(a, out);
            scan_params(b, out);
        }
    }
}

/// Declared flop count per output element: **one per distinct arithmetic / unary
/// node in the value DAG**.
///
/// Counting the [`ExprDag`] (not the authored tree) charges a shared interior
/// *once* — matching the item-02 emitter, which hoists it to a `tmp` computed
/// once. A body with no sharing counts identically to the old tree walk, so this
/// is a pure fix: `flops_per_elem` for a body with a duplicated subtree **drops**
/// (to the honest, actually-computed count) and never rises.
fn count_flops(e: &ScalarExpr) -> u32 {
    let dag = ExprDag::from_expr(e);
    (0..dag.len() as NodeId)
        .filter(|&id| !dag.node(id).is_leaf())
        .count() as u32
}

/// Conservative max-ULP bound over the lowered body — a *declared upper bound*,
/// the sum of the **vendor-approximate** ops' errors (transcendentals ~2 ulp,
/// `powf` 4, `logf` 1, the hand-rolled `Sigmoid`/`Silu`/`Gelu` composites a bit
/// higher). Correctly-rounded ops (the infix arithmetic, `Neg`/`Abs`/`Sqr`/`Sqrt`/
/// `Recip`/`Relu`/`Floor`/`Ceil`/`Round`/`Sign`/`Step`, `Max`/`Min`/`Rem`)
/// contribute 0, so a body with none is `correctly_rounded`. Over-stating is safe
/// (the planner won't admit into a too-tight slot); under-stating is not.
fn ulp_bound(e: &ScalarExpr) -> f64 {
    match e {
        // Coord rates 0 like the other leaves: the long-long → float/double
        // cast is exact under the documented caller precondition (axis extent
        // within the dtype's exact-integer range — see `ScalarExpr::Coord`).
        // Defensive today: no contract is ever emitted for a Coord body
        // (derive_pattern rejects CoordUnsupported first), but the table
        // stays exhaustive on purpose so the rating is decided here, not
        // silently defaulted, when the Iota bridge lands.
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => 0.0,
        ScalarExpr::Unary(op, x) => ulp_bound(x) + unary_ulp(*op),
        ScalarExpr::Binary(op, a, b) => ulp_bound(a) + ulp_bound(b) + binary_ulp(*op),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b) => ulp_bound(a) + ulp_bound(b),
        // Select contributes 0 ulp of its OWN: the pick never rounds (the arms'
        // bits move untouched, the cond compare is exact) — the same modeling
        // call as the Cmp* predicates in `binary_ulp`. Subexpression tiers sum
        // as usual. (Defensive today — the select withhold in `contract` keeps
        // any select body contract-less — but the table stays exhaustive on
        // purpose so the rating is decided here, not silently defaulted, when
        // the Where advert lands.)
        ScalarExpr::Select(c, a, b) => ulp_bound(c) + ulp_bound(a) + ulp_bound(b),
    }
}

/// Per-op CUDA f32 ULP error. Correctly-rounded / exact ops (`Neg`/`Abs`/`Sqr`/
/// `Sqrt`/`Recip`/`Relu`/`Floor`/`Ceil`/`Round`/`Sign`/`Step`/`Trunc`) are 0.
/// Deliberately **exhaustive** (no wildcard): a future op must be rated here on
/// purpose, or the compiler objects — a silent 0.0 default would under-state,
/// and under-stating is the one unsafe direction.
fn unary_ulp(op: UnaryOp) -> f64 {
    match op {
        // exact / correctly rounded
        UnaryOp::Neg
        | UnaryOp::Abs
        | UnaryOp::Sqr
        | UnaryOp::Sqrt
        | UnaryOp::Recip
        | UnaryOp::Relu
        | UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round
        | UnaryOp::Sign
        | UnaryOp::Step
        | UnaryOp::Trunc => 0.0,
        UnaryOp::Log | UnaryOp::Expm1 | UnaryOp::Log1p | UnaryOp::Cbrt | UnaryOp::Log2 => 1.0,
        UnaryOp::Exp
        | UnaryOp::Tanh
        | UnaryOp::Erf
        | UnaryOp::Sin
        | UnaryOp::Cos
        | UnaryOp::Rsqrt
        | UnaryOp::Exp2
        | UnaryOp::Log10
        | UnaryOp::Cosh
        | UnaryOp::Atan => 2.0,
        // composites of expf/erff + rounding — conservatively a bit higher —
        // and the vendor-3-ulp inverse/hyperbolic tier.
        UnaryOp::Sigmoid
        | UnaryOp::Silu
        | UnaryOp::Gelu
        | UnaryOp::Sinh
        | UnaryOp::Acos
        | UnaryOp::Asinh
        | UnaryOp::Atanh => 3.0,
        // domain-edge-sensitive fns at the loosest shared tier (vendor: tanf 4,
        // asinf 4, acoshf 4, erfcf 4).
        UnaryOp::Tan | UnaryOp::Asin | UnaryOp::Acosh | UnaryOp::Erfc => 4.0,
        // lgammaf: vendor declares 6 ulp OUTSIDE the interval (-10.001, -2.264)
        // and larger inside it (near the negative-real poles). 6 is the honest
        // vendor headline bound — rating it 4 (the previous loosest tier) would
        // under-state, so this op introduces the 6 tier; the near-pole caveat is
        // inherent to lgamma at any finite rating.
        UnaryOp::Lgamma => 6.0,
    }
}

/// Per-op CUDA f32 ULP error for the binary fns, mirroring [`unary_ulp`]
/// (exhaustive, no wildcard). `Max`/`Min` (compare-selects) and the bit-level
/// increment-0a ops (`copysignf`/`nextafterf`/`fmaxf`/`fminf`/`fmodf` — all
/// vendor 0-ulp) contribute 0. Floored `Rem` is rated UNBOUNDED (infinity ⇒
/// the contract emits `approximate` with no `max_ulp` claim): it lowers as
/// the composite `a - floor(a/b)*b`, and a 0.5-ulp quotient perturbation that
/// crosses an integer boundary flips the floor — the result is then off by
/// |b|, which no finite result-ULP number bounds. (Inherent to the floored-mod
/// formula; torch.remainder behaves identically. `RemTrunc` = `fmodf` is a
/// genuinely exact vendor primitive and keeps 0.)
fn binary_ulp(op: BinaryOp) -> f64 {
    match op {
        BinaryOp::Max
        | BinaryOp::Min
        | BinaryOp::Copysign
        | BinaryOp::Nextafter
        | BinaryOp::FmaxIeee
        | BinaryOp::FminIeee
        | BinaryOp::RemTrunc => 0.0,
        // The Cmp* predicates are exact C-operator compares producing exactly
        // 1.0/0.0 — 0 ulp. A compare of *approximate* subexpressions is still
        // an exact compare: this table rates each op's own rounding, and the
        // subexpressions' tiers are summed by `ulp_bound` as usual (the
        // decision-boundary sensitivity of a predicate is not a result-ULP
        // quantity — same modeling call as rating Sign/Step 0 in `unary_ulp`).
        BinaryOp::CmpEq
        | BinaryOp::CmpNe
        | BinaryOp::CmpLt
        | BinaryOp::CmpLe
        | BinaryOp::CmpGt
        | BinaryOp::CmpGe => 0.0,
        BinaryOp::Rem => f64::INFINITY,
        BinaryOp::Atan2 => 3.0, // vendor atan2f: 3 ulp
        BinaryOp::Pow => 4.0,
        // The increment-0c integer ops are BIT-EXACT: "exact" here means exact
        // WRAPPING two's-complement semantics (and, for the logical ops, the
        // exact 0/1 normalization) — there is no rounding step at all, so 0 is
        // the honest rating, not an under-statement. (Today these ops emit NO
        // contract — no OpTag/lower_op_kind name exists, so derive_pattern
        // rejects first — but this table stays exhaustive on purpose: when
        // Fuel names them, the rating is already decided here, not silently
        // defaulted.)
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => 0.0,
    }
}

/// Precision contract: `correctly_rounded` (0 ulp, bit-reproducible) when the
/// body is all correctly-rounded primitives, else `approximate` with the
/// conservative [`ulp_bound`]. (`F32Strict` would force `correctly_rounded` — it
/// is a precision mode, not a wire dtype.)
/// Whether `e` contains any comparison predicate anywhere (not just the root)
/// — the contract-withholding walk for the missing-`Cast` pattern gap. Shared
/// with the JIT's interior-cmp decline (`crate::jit`).
pub(crate) fn expr_contains_cmp(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b) => expr_contains_cmp(a) || expr_contains_cmp(b),
        ScalarExpr::Unary(_, a) => expr_contains_cmp(a),
        ScalarExpr::Binary(bop, a, b) => {
            bop.is_cmp() || expr_contains_cmp(a) || expr_contains_cmp(b)
        }
        // NO cond carve-out contract-side (unlike the JIT's interior-cmp
        // decline, which permits a cmp as a Select's cond child): select
        // bodies are withheld wholesale by `expr_contains_select` anyway, and
        // the cmp honesty gate keeps its full reach independently — the arm
        // exists for exhaustiveness, not policy.
        ScalarExpr::Select(c, a, b) => {
            expr_contains_cmp(c) || expr_contains_cmp(a) || expr_contains_cmp(b)
        }
    }
}

/// Whether `e` contains a [`ScalarExpr::Select`] anywhere — the WHERE/SELECT
/// contract-withholding walk. Any select-containing body has its contract
/// withheld WHOLESALE (see the guard in [`contract`]): the single-op `Where`
/// advert needs the Model-A per-operand dtype tuple (Fuel's cond is a U8
/// tensor — `[U8, T, T, T]`), and the fused-cmp form needs fuel-side matcher
/// validation (propose-first) — neither exists in v1. The pattern layer's
/// `PatternError::SelectUnsupported` miss does NOT substitute for this guard
/// (0b house rule: the contract gets its own layer — concretely, the Model-A
/// gather advert derives its `op_kind` STRUCTURALLY without consulting the
/// pattern, so a select body inside a u32-index gather would advertise
/// without this walk).
fn expr_contains_select(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Select(..) => true,
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => expr_contains_select(a) || expr_contains_select(b),
        ScalarExpr::Unary(_, a) => expr_contains_select(a),
    }
}

/// The dispatch `OpKind` spelling for a validated u8-out predicate body (root
/// IS a Cmp* — `assert_valid_out_dtype` guarantees it before any contract is
/// emitted). These are the exact strings Fuel's `lower_op_kind` table accepts.
fn cmp_dispatch_op_kind(body: &ScalarExpr) -> &'static str {
    let ScalarExpr::Binary(bop, _, _) = body else {
        unreachable!("u8-out body root is a validated Cmp*");
    };
    match bop {
        BinaryOp::CmpEq => "EqualElementwise",
        BinaryOp::CmpNe => "NotEqualElementwise",
        BinaryOp::CmpLt => "LessElementwise",
        BinaryOp::CmpLe => "LessEqualElementwise",
        BinaryOp::CmpGt => "GreaterElementwise",
        BinaryOp::CmpGe => "GreaterEqualElementwise",
        _ => unreachable!("u8-out body root is a validated Cmp*"),
    }
}

#[doc(hidden)]
pub fn precision_of(body: &ScalarExpr) -> (&'static str, Option<u32>) {
    let u = ulp_bound(body);
    if u <= 0.0 {
        ("correctly_rounded", Some(0))
    } else if u.is_infinite() {
        // A body containing an op with no finite result-ULP bound (floored
        // Rem's quotient-boundary flip): honest contract = approximate with
        // NO max_ulp claim, never an under-stated finite number.
        ("approximate", None)
    } else {
        ("approximate", Some(u.ceil() as u32))
    }
}

/// Free-text precision `notes` (Fuel `PrecisionBlock.notes`) carrying the old
/// `mode:` semantics in Fuel's schema vocabulary. For the unbounded (`prec_ulp ==
/// None`, floored `Rem`) case it names the reason the ULP bound is absent, so the
/// block lowers to a populated audited/bit-stable guarantee rather than the fatal
/// placeholder.
fn precision_notes(prec_mode: &str, prec_ulp: Option<u32>) -> &'static str {
    match (prec_mode, prec_ulp) {
        ("correctly_rounded", _) => "correctly_rounded; bit-reproducible",
        ("approximate", None) => {
            "approximate; floored-mod boundary flip — no finite result-ULP bound"
        }
        // finite-ULP approximate (transcendentals): max_ulp is a declared bound.
        _ => "approximate; max_ulp is a declared vendor upper bound",
    }
}

// ---------------------------------------------------------------------------
// Structure-cell projections
// ---------------------------------------------------------------------------

fn layout_token(key: &StructureKey, i: usize) -> &'static str {
    match key.operands[i].contig {
        Contiguity::Contig => "contiguous",
        Contiguity::InnerContig => "inner_contiguous",
        Contiguity::Strided => "strided",
        Contiguity::Broadcast => "broadcast",
    }
}

/// The five-flag `LayoutSpec` (Fuel `fkc/schema.rs` `LayoutSpec`, §4.1) for
/// operand `i`, rendered as the inline-map form Fuel's corpus uses (verified
/// against docs/kernel-contracts/cpu/elementwise-binary.fkc.md line 90). Fuel's
/// `TensorDesc.layout` is `Option<LayoutSpec>` (a struct of optional
/// `required`/`accepted`/`rejected` string tri-states), so the pre-reconcile
/// bare `layout: contiguous` string was a serde type error at import.
///
/// Driven by the operand's structure-key contiguity:
///
/// - **Contiguous** cell: compiled for the packed layout it REQUIRES (contiguous
///   required, everything else rejected — byte-for-byte Fuel's own
///   contiguous-only corpus kernels).
/// - **InnerContig / Strided** cell: the emitter walks this operand with FULL
///   runtime strides — `offset_expr` (cuda.rs) keeps every iteration-axis term
///   `c{d}·stride{d}` with the operand's runtime stride, and (verified) an
///   InnerContig/Strided operand always has an EMPTY broadcast mask
///   (`classify_contiguity` routes any stride-0 axis to `Broadcast` first,
///   structure_key.rs), so NO axis term is compile-time dropped. A runtime
///   stride-0 axis is therefore handled correctly by that same walk (its term
///   contributes 0), which is exactly what `broadcast_stride0: accepted` means —
///   so it is emitted `accepted`, truthfully. This is how Fuel's own strided
///   corpus spells it (strided + broadcast accepted → projects
///   `KernelCaps.strided_input = true` and `is_generic_contract = true`;
///   fuel-dispatch `fkc/caps_map.rs` §6, verified 2026-07-08); the previous
///   `broadcast_stride0: rejected` was an UNDER-claim that silently collapsed
///   every strided cell to `strided_input = false` (identical caps to a
///   contiguous cell) and corrupted the miss-telemetry genericity bit.
/// - **Broadcast** operand: reachable here ONLY for the U32 gather / index_select
///   INDEX operand — every OTHER Broadcast-class operand causes the whole
///   contract to be WITHHELD up front (`contract`, item-01 layout-honesty guard):
///   its stride-0 mask is BAKED into the kernel (a fully-broadcast operand is
///   hoisted `in[k][0]`, a partial one has its bcast-axis terms compile-time
///   dropped by `offset_expr`), so the kernel cannot walk a contiguous/strided
///   tensor in that slot and Fuel's tri-state has no `broadcast_stride0:
///   required` spelling to say so. The gather index is a 1-D index read along the
///   gathered axis; the honest, safe posture for its PHYSICAL index buffer is
///   `contiguous: required` (the conservative, Fuel-corpus `[T, U32, T]` shape —
///   a safe understatement; the kernel also handles a strided index via its
///   runtime stride, but `contiguous: required` never overstates).
///
/// `start_offset` / `reverse_strides` (negative strides) are rejected everywhere
/// — no EMITTED cell reads a non-zero base offset or a reversed axis (a flipped
/// operand is withheld up front, so `reverse_strides: rejected` stays truthful by
/// construction). Understating a capability is safe (the planner just contiguizes
/// more than strictly needed, `caps.awkward_layout`); overstating is not — the
/// same one-directional-safety rule as the ULP bound.
fn layout_spec(key: &StructureKey, i: usize) -> String {
    let (contiguous, strided, broadcast_stride0) = match key.operands[i].contig {
        Contiguity::Contig => ("required", "rejected", "rejected"),
        Contiguity::InnerContig | Contiguity::Strided => ("accepted", "accepted", "accepted"),
        // Gather/index_select U32 index only (all other Broadcast operands are
        // withheld up front). Conservative, truthful understatement.
        Contiguity::Broadcast => ("required", "rejected", "rejected"),
    };
    format!(
        "{{ contiguous: {contiguous}, strided: {strided}, \
         broadcast_stride0: {broadcast_stride0}, start_offset: rejected, \
         reverse_strides: rejected }}"
    )
}

/// A kernel specialized for a strided/broadcast cell handles awkward layouts;
/// a contiguous/vectorized cell requires the packed layout it was built for.
fn awkward_layout(key: &StructureKey) -> &'static str {
    match key.operands[0].contig {
        Contiguity::Strided | Contiguity::Broadcast => "handles_strided",
        Contiguity::Contig | Contiguity::InnerContig => "requires_contiguous",
    }
}

/// Required base-pointer alignment (bytes): a vectorized cell needs its vector
/// width; a scalar cell needs the dtype's natural alignment.
///
/// SEAM NOTE (H2, adversarial pass 2026-07-02 — RESOLVED 2026-07-03): the
/// kernel's `n` argument is a VECTOR count (`n / width`) on vectorized/packed
/// cells and an ELEMENT count otherwise — an 8x hazard for any consumer that
/// assumed elements. The caps block now states it explicitly (`count_unit:`,
/// via `effective_count_width`, which mirrors the emitter's dispatch including
/// its scalar fallbacks); Fuel confirmed they never derive launches from
/// contracts and will consume the field when their declared-cost trampoline
/// compiles cost expressions over `n`.
fn required_align(key: &StructureKey) -> u32 {
    let dsz = dtype_size(key.dtype);
    match key.operands[0].vec_width {
        VecWidth::V8 => (8 * dsz).min(16),
        VecWidth::V4 => (4 * dsz).min(16),
        VecWidth::V2 => 2 * dsz,
        VecWidth::Scalar => dsz,
    }
}

fn bytes_per_elem(op: &OpDef, key: &StructureKey) -> u32 {
    // inputs at the key dtype + one output at the (possibly narrower, u8-mask)
    // output dtype (broadcast operands touch fewer; this is a declared upper
    // estimate). Uniform ops reduce to the old (n_inputs + 1) * size exactly.
    u32::from(op.n_inputs) * dtype_size(key.dtype) + dtype_size(op.out_dtype.unwrap_or(key.dtype))
}

/// `<op>_<dtype>_<contig0>_<vec0>` cell discriminator for the readable `kernel`
/// name (the linkable symbol is `entry_point`).
#[doc(hidden)]
pub fn cell_suffix(key: &StructureKey) -> String {
    let o = &key.operands[0];
    format!(
        "{}_{}_{}",
        dtype_short(key.dtype),
        contig_short(o.contig),
        vec_short(o.vec_width)
    )
}

fn blurb(op: &OpDef, key: &StructureKey, dtype: &str, is_fusion: bool) -> String {
    let kind = if is_fusion { "fused" } else { "elementwise" };
    format!(
        "{} {} ({}, {} layout).",
        kind,
        op.name,
        dtype,
        layout_token(key, 0)
    )
}

// ---------------------------------------------------------------------------
// Leaf spellings (reconciled against the FKC annex; review item E5 for dtypes)
// ---------------------------------------------------------------------------

/// FKC §5 logical-DType token, or `None` for a dtype with no §5 *base-dtype*
/// slot (so the caller skips the cell rather than emit an unbindable contract).
///
/// Reconciled to FKC rev-4 §5 (review item E5): `Bool` → `U8` (Fuel has no Bool
/// dtype — masks are U8), signed-8 → `I8`, `F32Strict` rides as `F32` (a
/// precision mode, not a wire dtype). Packed sub-byte / quant payloads
/// (`S4`/`U4`/`Bin`) ride the **FDX sidecar**, not a base dtype, so carry no
/// token here; `Fp8E5M2` and complex have no §5 slot yet — all return `None`.
fn fkc_dtype(dt: ElementKind) -> Option<&'static str> {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, F16, F32, F32Strict, F64, Fp8E4M3, Fp8E5M2, I32,
        I64, S4, S8, U4, U8, U32,
    };
    Some(match dt {
        F32 | F32Strict => "F32",
        F16 => "F16",
        Bf16 => "BF16",
        F64 => "F64",
        I32 => "I32",
        I64 => "I64",
        // U32 is the gather/scatter INDEX operand's FKC §5 spelling, verified
        // against Fuel's `fkc/lower.rs` `lower_dtype` table (`"U32" =>
        // DType::U32`). Emitted on the index slot of a gather/index_select
        // accept block so Fuel assembles the key `[T, U32, T]`.
        U32 => "U32",
        S8 => "I8",        // §5: signed-8 spells I8
        U8 | Bool => "U8", // §5 (B5/E5): Fuel has no Bool — masks are U8
        Fp8E4M3 => "F8E4M3",
        // No §5 base-dtype slot: FDX-sidecar payloads + unlisted fp8 / complex.
        Fp8E5M2 | S4 | U4 | Bin | Complex32 | Complex64 => return None,
    })
}

fn dtype_short(dt: ElementKind) -> &'static str {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, F16, F32, F32Strict, F64, Fp8E4M3, Fp8E5M2, I32,
        I64, S4, S8, U4, U8, U32,
    };
    match dt {
        F32 | F32Strict => "f32",
        F16 => "f16",
        Bf16 => "bf16",
        F64 => "f64",
        I32 => "i32",
        I64 => "i64",
        U32 => "u32",
        S8 => "s8",
        U8 => "u8",
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

fn contig_short(c: Contiguity) -> &'static str {
    match c {
        Contiguity::Contig => "co",
        Contiguity::InnerContig => "ic",
        Contiguity::Strided => "st",
        Contiguity::Broadcast => "br",
    }
}

fn vec_short(v: VecWidth) -> &'static str {
    match v {
        VecWidth::Scalar => "v1",
        VecWidth::V2 => "v2",
        VecWidth::V4 => "v4",
        VecWidth::V8 => "v8",
    }
}

fn dtype_size(dt: ElementKind) -> u32 {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, F16, F32, F32Strict, F64, Fp8E4M3, Fp8E5M2, I32,
        I64, S4, S8, U4, U8, U32,
    };
    match dt {
        S4 | U4 | Bin => 1, // sub-byte: round up to a byte for the declared estimate
        S8 | U8 | Bool | Fp8E4M3 | Fp8E5M2 => 1,
        F16 | Bf16 => 2,
        F32 | F32Strict | I32 | U32 => 4, // U32: 4-byte index dtype
        F64 | I64 | Complex32 => 8,
        Complex64 => 16,
    }
}

/// 64-bit FNV-1a over the kernel source — the `ImplId.kernel_revision_hash` base.
/// Stable and dependency-free (unlike `DefaultHasher`, which is unspecified).
pub(crate) fn revision_hash(src: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in src.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{input, param};

    /// A VERBATIM snapshot of every `op_kind` string fuel-dispatch
    /// `src/fkc/lower.rs` `lower_op_kind` accepts — copied literal-for-literal
    /// from that table (read 2026-07-08). This is the INDEPENDENT ground truth:
    /// it is NOT derived from `fuel_primitive_op_kind`, so a typo that maps a
    /// Baracuda root to a non-existent Fuel spelling is caught here rather than
    /// poisoning a real bundle import.
    const FUEL_LOWER_OP_KIND_ACCEPTED: &[&str] = &[
        "MatMul",
        "AddElementwise",
        "SubElementwise",
        "MulElementwise",
        "DivElementwise",
        "ReluElementwise",
        "NegElementwise",
        "SqrElementwise",
        "SqrtElementwise",
        "RecipElementwise",
        "AbsElementwise",
        "TanhElementwise",
        "ExpElementwise",
        "LogElementwise",
        "SinElementwise",
        "CosElementwise",
        "SigmoidElementwise",
        "SiluElementwise",
        "GeluElementwise",
        "StepElementwise",
        "SumReduce",
        "MaxReduce",
        "MinReduce",
        "MeanReduce",
        "Cast",
        "Conv2D",
        "ConvTranspose2D",
        "ReduceSumTo",
        "ReduceMaxTo",
        "FusedLinear",
        "FlashAttn",
        "FlashAttnBackwardQ",
        "FlashAttnBackwardK",
        "FlashAttnBackwardV",
        "PagedAttn",
        "Affine",
        "ClampElementwise",
        "PowIElementwise",
        "PowIElementwiseBackward",
        "MaximumElementwise",
        "MinimumElementwise",
        "EqualElementwise",
        "NotEqualElementwise",
        "LessElementwise",
        "LessEqualElementwise",
        "GreaterElementwise",
        "GreaterEqualElementwise",
        "Where",
        "FloorElementwise",
        "CeilElementwise",
        "RoundElementwise",
        "SignElementwise",
        "ErfElementwise",
        "GeluErfElementwise",
        "PowElementwise",
        "RsqrtElementwise",
        "RemElementwise",
        "Flip",
        "Roll",
        "CumSum",
        "Pad",
        "Triu",
        "Tril",
        "LogSoftmaxLastDim",
        "LogSoftmaxLastDimBackward",
        "MaskedFill",
        "PadBackward",
        "Concat",
        "SoftmaxLastDim",
        "SoftmaxLastDimBackward",
        "RmsNormLastDim",
        "RmsNormLastDimBackward",
        "LayerNormLastDim",
        "LayerNormLastDimBackward",
        "ReduceMaxToBackward",
        "IndexSelect",
        "Gather",
        "Rope",
        "IndexAdd",
        "ScatterAdd",
        "ArgMaxDim",
        "ArgMinDim",
        "QMatMul",
        "WriteSlice",
        "WriteSliceRotating",
        "Copy",
        "ReluInplace",
        "SiluInplace",
        "GeluInplace",
        "TanhInplace",
        "SigmoidInplace",
        "NegInplace",
        "AbsInplace",
        "SqrInplace",
        "SqrtInplace",
        "RsqrtInplace",
        "RecipInplace",
        "ExpInplace",
        "LogInplace",
        "SinInplace",
        "CosInplace",
        "SignInplace",
        "FloorInplace",
        "CeilInplace",
        "RoundInplace",
        "ErfInplace",
        "GeluErfInplace",
        "ClampInplace",
        "PowIInplace",
        "InplaceAffine",
        "FusedSoftmaxCrossEntropy",
        "CausalConv1d",
        "SelectiveScan",
        "SsdChunkScan",
        "Nf4Matmul",
    ];

    /// The 29 internal single-op roots (`root_op_name` outputs) that MUST map to
    /// an importable Fuel spelling, with the exact expected spelling. Mirrors
    /// `pattern.rs` `binary_name`/`unary_name` + the infix `Add`/`Sub`/`Mul`/`Div`.
    /// `Relu` is MAPPED again (2026-07-08): Fuel pinned `ReluElementwise` =
    /// NaN-propagating (torch parity) and rebinds it to the bespoke propagating
    /// CUDA kernel, so the semantic divergence that held it out is resolved — see
    /// `fuel_primitive_op_kind` and `relu_maps_to_relu_elementwise` below.
    const MAPPED_ROOTS: &[(&str, &str)] = &[
        ("Add", "AddElementwise"),
        ("Sub", "SubElementwise"),
        ("Mul", "MulElementwise"),
        ("Div", "DivElementwise"),
        ("Maximum", "MaximumElementwise"),
        ("Minimum", "MinimumElementwise"),
        ("Pow", "PowElementwise"),
        ("Rem", "RemElementwise"),
        ("Neg", "NegElementwise"),
        ("Abs", "AbsElementwise"),
        ("Sqr", "SqrElementwise"),
        ("Sqrt", "SqrtElementwise"),
        ("Rsqrt", "RsqrtElementwise"),
        ("Recip", "RecipElementwise"),
        ("Exp", "ExpElementwise"),
        ("Log", "LogElementwise"),
        ("Tanh", "TanhElementwise"),
        ("Sigmoid", "SigmoidElementwise"),
        ("Relu", "ReluElementwise"),
        ("Erf", "ErfElementwise"),
        ("GeluErf", "GeluErfElementwise"),
        ("Silu", "SiluElementwise"),
        ("Sin", "SinElementwise"),
        ("Cos", "CosElementwise"),
        ("Floor", "FloorElementwise"),
        ("Ceil", "CeilElementwise"),
        ("Round", "RoundElementwise"),
        ("Sign", "SignElementwise"),
        ("Step", "StepElementwise"),
    ];

    #[test]
    fn select_bookkeeping_is_one_flop_zero_ulp() {
        // Bookkeeping arms land even while the contract is withheld (so the
        // ratings are decided, not defaulted, when the Where advert lands):
        // the DAG-driven flop count charges a select ONE flop, and the ULP
        // table rates a select 0 (an exact pick — the Cmp* modeling call).
        let body = input(0).select(input(1), input(2)).0;
        assert_eq!(count_flops(&body), 1, "select = 1 flop, deliberately");
        assert_eq!(ulp_bound(&body), 0.0, "select never rounds (0 ulp)");
        let (mode, ulp) = precision_of(&body);
        assert_eq!(mode, "correctly_rounded");
        assert_eq!(ulp, Some(0));
        // Params thread through all three children.
        let with_params = param(0).select(input(0) + param(1), param(2)).0;
        assert_eq!(params_used(&with_params), vec![0, 1, 2]);
    }

    #[test]
    fn flops_count_dedups_shared_subtree() {
        use crate::ir::konst;
        // out = g / (g + 1), g = a*b: the product appears on both sides but is
        // computed once by the emitter, so it is charged once. Distinct DAG
        // non-leaf nodes: Mul, Add, Div = 3 (a tree walk would count Mul twice → 4).
        let g = input(0) * input(1);
        assert_eq!(count_flops(&(g.clone() / (g + konst(1.0))).0), 3);
        // Structurally-identical products authored separately still hash-cons to one.
        let (m1, m2) = (input(0) * input(1), input(0) * input(1));
        assert_eq!(count_flops(&(m1 / (m2 + konst(1.0))).0), 3);
        // A genuinely different second product does NOT merge → 4 ops.
        let distinct = (input(0) * input(1)) / ((input(0) * input(2)) + konst(1.0));
        assert_eq!(count_flops(&distinct.0), 4);
    }

    #[test]
    fn bundle_kisc_frames_each_admitted_contract_and_drops_the_heading() {
        let c1 = "kernel: relu\nop_kind: ReluElementwise\naccept: sk3|une|f32\n".to_string();
        let c2 = "kernel: add\nop_kind: AddElementwise\naccept: sk3|bin|f32\n".to_string();
        let b = bundle_kisc("cuda", "rev0", &[c1.clone(), c2.clone()], false);
        // Provider front-matter still leads the file.
        assert!(b.starts_with("---\n"), "front matter leads: {b}");
        // Each contract is its own KISC document; NO `## ` heading framing.
        assert!(
            b.contains(&crate::kisc::kisc_frame(&c1)),
            "c1 framed as a KISC document: {b}"
        );
        assert!(
            b.contains(&crate::kisc::kisc_frame(&c2)),
            "c2 framed as a KISC document: {b}"
        );
        assert!(
            !b.contains("\n## "),
            "the markdown heading framing is gone: {b}"
        );
    }

    #[test]
    fn bundle_kisc_withholds_a_non_fuel_fused_op_from_a_pre_recipe_peer() {
        // Pre-recipe-import peer (recipe_import = false): a Fuel FusedOp is framed;
        // a free-form fused name is withheld (non-importable by the closed vocab).
        let ok = "kernel: softmax\nfused_op: SOFTMAX_LAST_DIM\n".to_string();
        let bad = "kernel: relu_add\nfused_op: RELU_ADD\n".to_string();
        let b = bundle_kisc("cuda", "rev0", &[ok.clone(), bad.clone()], false);
        assert!(
            b.contains(&crate::kisc::kisc_frame(&ok)),
            "an admitted Fuel FusedOp is framed: {b}"
        );
        assert!(
            !b.contains(&crate::kisc::kisc_frame(&bad)),
            "a non-Fuel fused_op stays withheld from a pre-recipe peer: {b}"
        );
    }

    #[test]
    fn bundle_kisc_recipe_import_peer_still_withholds_a_recipeless_fusion() {
        // The gate exists but is inert TODAY: even a recipe-import peer can't
        // verify a generic fusion until Baracuda emits the recipe (the neutral
        // Semantics op-DAG), so `contract_carries_recipe` is false and the free-form
        // fused op stays withheld. This pins the "retire in lockstep" contract.
        let bad = "kernel: relu_add\nfused_op: RELU_ADD\n".to_string();
        let b = bundle_kisc("cuda", "rev0", std::slice::from_ref(&bad), true);
        assert!(
            !b.contains(&crate::kisc::kisc_frame(&bad)),
            "recipe-import peer + no recipe emitted yet ⇒ still withheld: {b}"
        );
    }

    #[test]
    fn contract_carries_recipe_detects_the_semantics_line() {
        assert!(contract_carries_recipe(
            "kernel: x\nsemantics: add(in0, in1)\n"
        ));
        assert!(!contract_carries_recipe(
            "kernel: x\nop_kind: AddElementwise\n"
        ));
    }

    #[test]
    fn bundle_kisc_admits_a_recipe_carrying_fusion_for_a_recipe_import_peer() {
        // A fused op Fuel doesn't know, but CARRYING a recipe: a recipe-import peer
        // can verify + register it, so it is now framed — the withhold seam flips
        // live. The same contract stays withheld from a pre-recipe peer.
        let recipe_fusion =
            "kernel: relu_add\nfused_op: RELU_ADD\nsemantics: add(relu(in0), in1)\n".to_string();
        let live = bundle_kisc("cuda", "rev0", std::slice::from_ref(&recipe_fusion), true);
        assert!(
            live.contains(&crate::kisc::kisc_frame(&recipe_fusion)),
            "recipe-carrying fusion is admitted for a recipe-import peer: {live}"
        );
        let old = bundle_kisc("cuda", "rev0", std::slice::from_ref(&recipe_fusion), false);
        assert!(
            !old.contains(&crate::kisc::kisc_frame(&recipe_fusion)),
            "still withheld from a pre-recipe peer: {old}"
        );
    }

    #[test]
    fn revision_hash_is_source_sensitive() {
        assert_ne!(revision_hash("kernel a"), revision_hash("kernel b"));
        assert_eq!(revision_hash("stable"), revision_hash("stable"));
    }

    #[test]
    fn vocab_ulp_tiers_rate_the_new_fns() {
        use crate::ir::{BinaryOp, UnaryOp};
        // Op-sensitive precision for the increment-0a vocabulary: the declared
        // max_ulp is the fn's vendor tier, and the exact/bit-level ops stay
        // correctly_rounded. (Under-stating is the unsafe direction — these pins
        // hold the table honest.)
        let u = |op: UnaryOp| precision_of(&input(0).unary(op).0);
        assert_eq!(u(UnaryOp::Trunc), ("correctly_rounded", Some(0)));
        assert_eq!(u(UnaryOp::Log1p), ("approximate", Some(1)));
        assert_eq!(u(UnaryOp::Expm1), ("approximate", Some(1)));
        assert_eq!(u(UnaryOp::Cbrt), ("approximate", Some(1)));
        assert_eq!(u(UnaryOp::Exp2), ("approximate", Some(2)));
        assert_eq!(u(UnaryOp::Sinh), ("approximate", Some(3)));
        // the domain-edge-sensitive fns carry the loose tiers.
        assert_eq!(u(UnaryOp::Tan), ("approximate", Some(4)));
        assert_eq!(u(UnaryOp::Asin), ("approximate", Some(4)));
        assert_eq!(u(UnaryOp::Acosh), ("approximate", Some(4)));
        assert_eq!(u(UnaryOp::Erfc), ("approximate", Some(4)));
        assert_eq!(u(UnaryOp::Lgamma), ("approximate", Some(6)));
        let b = |op: BinaryOp| precision_of(&input(0).binary(op, input(1)).0);
        assert_eq!(b(BinaryOp::Atan2), ("approximate", Some(3)));
        // bit-level / exact binaries are correctly rounded.
        assert_eq!(b(BinaryOp::Copysign), ("correctly_rounded", Some(0)));
        assert_eq!(b(BinaryOp::Nextafter), ("correctly_rounded", Some(0)));
        assert_eq!(b(BinaryOp::FmaxIeee), ("correctly_rounded", Some(0)));
        assert_eq!(b(BinaryOp::FminIeee), ("correctly_rounded", Some(0)));
        assert_eq!(b(BinaryOp::RemTrunc), ("correctly_rounded", Some(0)));
    }

    #[test]
    fn cmp_predicates_rate_zero_ulp() {
        use crate::ir::BinaryOp;
        // The compare itself is exact (0 ulp): a pure-cmp body is
        // correctly_rounded…
        for op in [
            BinaryOp::CmpEq,
            BinaryOp::CmpNe,
            BinaryOp::CmpLt,
            BinaryOp::CmpLe,
            BinaryOp::CmpGt,
            BinaryOp::CmpGe,
        ] {
            assert_eq!(
                precision_of(&input(0).binary(op, input(1)).0),
                ("correctly_rounded", Some(0)),
                "{op:?}"
            );
        }
        // …and a compare of an APPROXIMATE subexpression adds 0 of its own —
        // the body ulp is exactly the subexpression's tier (exp -> 2), the
        // pinned increment-0b modeling decision.
        let e = input(0).exp().binary(BinaryOp::CmpGt, input(1));
        assert_eq!(precision_of(&e.0), ("approximate", Some(2)));
    }

    #[test]
    fn relu_maps_to_relu_elementwise() {
        // Withhold LIFTED (Fuel's 2026-07-08 consolidated answer), reconciliation
        // now FULLY CLOSED. The earlier hold was a genuine semantic divergence —
        // our synthesized relu NaN-propagates (torch.relu) while Fuel's
        // ReluElementwise slot then NaN-scrubbed (CPU `x.max(0.0)`, the FKC doc's
        // "NaN-as-missing", the incumbent CUDA `fmaxf` kernel). Fuel DECIDED
        // ReluElementwise = NaN-propagating (torch parity) and its CPU + CUDA
        // rebind to our bespoke propagating kernel has now LANDED on Fuel `main`
        // (2026-07-09: 772e27a0 CPU, 00b25dc0 CUDA ReluElementwise, 5d52ee82 CUDA
        // ReluInplace, vs our alpha.76 unary_relu_propagating_* family). Both
        // slots agree on NaN — a JIT adopt is behaviorally identical, no divergence.
        assert_eq!(fuel_primitive_op_kind("Relu"), Some("ReluElementwise"));
        // And the spelling is one Fuel's importer accepts (verbatim cross-check).
        assert!(FUEL_LOWER_OP_KIND_ACCEPTED.contains(&"ReluElementwise"));
    }

    #[test]
    fn fuel_primitive_op_kind_outputs_are_accepted_fuel_strings() {
        // The VERBATIM cross-check: every mapping output must be a real string
        // Fuel's `lower_op_kind` accepts (else the whole bundle fails import),
        // AND every declared root must actually map (no silent gap).
        for (root, want) in MAPPED_ROOTS {
            let got = fuel_primitive_op_kind(root)
                .unwrap_or_else(|| panic!("root {root:?} must map to a Fuel spelling"));
            assert_eq!(got, *want, "root {root:?} mapped to the wrong spelling");
            assert!(
                FUEL_LOWER_OP_KIND_ACCEPTED.contains(&got),
                "mapping output {got:?} (root {root:?}) is NOT in Fuel's lower_op_kind table"
            );
        }
        // Sanity: the semantic-critical pairs are exactly as reconciled.
        // Rem = floored / torch.remainder; Step = x>0 heaviside; GeluErf = exact
        // erf (NOT the tanh-approx GeluElementwise).
        assert_eq!(fuel_primitive_op_kind("Rem"), Some("RemElementwise"));
        assert_eq!(fuel_primitive_op_kind("Step"), Some("StepElementwise"));
        assert_eq!(
            fuel_primitive_op_kind("GeluErf"),
            Some("GeluErfElementwise")
        );
        assert_ne!(fuel_primitive_op_kind("GeluErf"), Some("GeluElementwise"));
    }

    #[test]
    fn unmapped_primitive_root_is_an_honest_miss() {
        // The honest-miss policy, exercised build-DIRECTLY (not via the emitter):
        // AddScalar/MulScalar (Fuel has no scalar-param primitive OpKind) and any
        // exotic/hand-built spelling map to None → the contract is withheld.
        assert_eq!(fuel_primitive_op_kind("AddScalar"), None);
        assert_eq!(fuel_primitive_op_kind("MulScalar"), None);
        assert_eq!(fuel_primitive_op_kind("Identity"), None); // Bind root (never single-op anyway)
        assert_eq!(fuel_primitive_op_kind("Gelu"), None); // bare tanh-Gelu root is not one we emit
        assert_eq!(fuel_primitive_op_kind("TotallyBogusOp"), None);
        // No mapping output may EVER be a raw internal spelling that Fuel rejects.
        for (root, _) in MAPPED_ROOTS {
            assert_ne!(
                fuel_primitive_op_kind(root),
                Some(*root),
                "root {root:?} must be reconciled, not passed through raw"
            );
        }
    }
}
