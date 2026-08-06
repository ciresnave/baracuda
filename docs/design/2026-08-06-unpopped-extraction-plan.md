# Unpopped extraction plan — carve the kernel generator out of Baracuda

**Status:** ready to execute, gated on Baracuda `main` @ `419e4f22` (post-PR#12). Coordinated with the Unpopped peer (repo `C:\Projects\Unpopped`).

## Decisions (Eric)

- **Rename under new identity:** `baracuda-kernelgen → unpopped` (the flagship generator crate); `baracuda-kernel-vocab → unpopped-vocab`. Fresh `0.1.0`, new crates.io slots (`unpopped` is unclaimed — Eric reserves it). The alpha.78 line ends at the seam; Fuel re-pins.
- **License:** dual `MIT OR Apache-2.0` (Unpopped adds `LICENSE-APACHE` beside its MIT).
- **Dependency direction:** immediate crates.io pinning — no path-dep bridge. Baracuda consumes published `unpopped`/`unpopped-vocab` from day one. Makes **standalone-green** (does the carved crate build+test *outside* the workspace) the load-bearing coupling audit.
- **History:** `git filter-repo` the two crate paths; the rename lands as commits on top of preserved history (keeps the IR-expansion ramp + adversarial-review trail).

## Target architecture (settled)

- **`unpopped`** — the language-agnostic pipeline: source/op-spec → IR → transforms → the `Backend` **trait** + the `Compiler` **trait** + the CPU oracle + a reusable `run_cli` driver. Plus the neutral in-tree impls that make it testable alone: `CpuC` (portable-C99 reference `Backend`) and `StubCompiler`. **Zero backend/vendor-specific deps.**
- **Backends** (external, vendor-named, each links `unpopped`, implements `unpopped::Backend`): Baracuda provides the CUDA backend (`baracuda-*`); the Slang successor provides the Slang backend; Vulkane later provides Vulkan (`vulkane-*`). Not `unpopped-*` — the `unpopped-*` namespace is reserved for Unpopped's own crates.
- **Distribution repo** (dedicated, small): depends on `unpopped` + feature-gated backends (`--features cuda,slang`), maps `--target` → backend, calls `run_cli`. Single binary, multi-emitter, lean-or-fat by features. The only place that names specific backends.
- **Consumers** (Fuel, Baracuda-on-behalf-of-consumers): call the executable. No crate link, no version lockstep.
- **Neutral-ABI freeze** waits for Vulkane to exercise it (second independent backend).

---

## THE key finding — the CUDA carve is not a file move (the shared-speller refactor)

`cuda.rs` is **not** purely the CUDA backend. It is the sole home of ~14 **C-family scalar-op spellers** that `cpu_c.rs`, `slang.rs`, and `contract.rs` import directly and verbatim:

- `cpu_c.rs:44-49` imports 14: `assert_no_int_div_or_const, binary_f32, binary_f64, binary_int, dtype_tag, out_ctype_of, param_args, param_ctype, scalar_ctype, select_f32, select_f64, store_expr_of, unary_f32, unary_f64`.
- `slang.rs:54` imports 2: `assert_no_int_div_or_const, dtype_tag`.
- `contract.rs:895` uses `crate::cuda::effective_count_width`.

Relocating `cuda.rs` as-is strands `cpu_c.rs` (which STAYS in `unpopped`), `slang.rs`, and `contract.rs`.

**Prerequisite refactor (the enabler task):** carve the shared spellers out of `cuda.rs` into a NEW neutral module in `unpopped` (working name `unpopped::cfamily` / `spell`). The C-family spellers (`unary_f32/f64`, `binary_f32/f64/int`, `select_f32/f64`, `dtype_tag`, `out_ctype_of`, `scalar_ctype`, `param_ctype/args`, `store_expr_of`, `assert_no_int_div_or_const` — cuda.rs:393,573,778,801,6894,6944,7022,7087,7169,7267,7272,7381,7567,7579) are ~95% portable C99 already (per `cpu_c.rs:17-27`), so they're genuinely neutral. `cuda.rs` (now external) depends on the core speller module; `cpu_c.rs` (core) and `slang.rs` (external Slang) do too. `effective_count_width` is the exception — it's backend-aware (see below) and stays with the CUDA backend, exposed to core via a trait method.

This is why the CUDA-backend carve is the one **large** item — it's a refactor (split the shared spellers first), not a `git mv`.

---

## Module classification (24 files, from the survey)

**UNPOPPED-CORE (clean, backend-agnostic):**
`ir.rs`, `plan.rs`, `backend.rs` (Backend trait + Lowering seam — all `pub`), `oracle.rs`, `pattern.rs`, `shape.rs`, `recipe.rs`, `optimize.rs`, `text.rs`, `telemetry.rs`, `link.rs`, `dispatch_artifact.rs`, `kisc.rs`, `lib.rs` (shell — re-export edits needed).

**UNPOPPED-CORE (neutral in-tree impls — the standalone-green enablers):**
`cpu_c.rs` (portable-C99 reference `Backend` — STAYS; without it `unpopped` ships zero working backends), the `Compiler` trait + `StubCompiler` (in `jit.rs`).

**RELOCATE — CUDA backend crate (Baracuda-side):**
`cuda.rs` (minus the extracted shared spellers), `NvrtcCompiler` (jit.rs:242-292, the only `baracuda-nvrtc` pull, at jit.rs:261), `BaracudaSynthesizer` (jit.rs:1046-1165, the Fuel `Synthesizer` impl — hardcodes `Cuda`).

**RELOCATE — Slang backend crate:**
`slang.rs` (real second `Backend` impl, non-C-family emitter, no vendor SDK dep — needs the core speller module for its 2 shared fns).

**STRADDLES (split at the cut):**
- `contract.rs` — 99% core (all FKC gating reads OpDef/plan predicates); ONE backend-aware reach: `count_unit` via `crate::cuda::effective_count_width` (contract.rs:895). Fix: add a `Backend` trait method `effective_count_width(&self, plan) -> u32` so `contract()` is backend-generic. (`effective_count_width` is `pub(crate)` at cuda.rs:440 — the trait-method promotion is a clean *widening* of a currently-non-public fn, not a breaking change.) Rest → core.
- `convert.rs` (`convert` feature) — CUDA + Slang tree-sitter lifters share generic CST helpers; production code has no `Cuda`/`Slang` struct dep, only tests (convert.rs:758,850). Core keeps the neutral lifters; the CUDA-lift test arm relocates.
- `jit.rs` — Compiler trait + StubCompiler + generic `synthesize()` → core; `NvrtcCompiler` + `BaracudaSynthesizer` → CUDA backend; the generic `seam::synthesize` conversion → core ONLY if core takes a `fuel-kernel-seam-types` dep (judgment call, below).
- `lift.rs` — CUDA-syntax-specific hand-written lifter, but zero `cuda.rs` dep. Frontend; core-portable as a CUDA *frontend* (a converter, not the emitter).
- `bin/kernelgen.rs` — the CLI; hardcodes `&Cuda` at ~15 sites (bin/kernelgen.rs:17,50,…). Becomes the `run_cli(registry, args)` driver (core, backend-free) + the reference bundler (distribution repo) supplies the backend registry.

**DEV-TEST (stay with core or the harness):**
`fuzz.rs` (`#[cfg(test)]`, exercises all 3 backends — needs the neutral speller module + at least CpuC), `kiss_ref_diff.rs` (`#[cfg(test)]`, kiss-ref differential — dev-deps `kiss-ref-core/ops/classify`).

---

## The three neutrality seams

1. **nvrtc — clean.** `baracuda_nvrtc` referenced exactly once (jit.rs:261, inside `#[cfg(feature="nvrtc")] impl Compiler for NvrtcCompiler`). Core keeps the `Compiler` trait + `StubCompiler`; `NvrtcCompiler` → CUDA backend. `synthesize()` already injects `&dyn Backend, &dyn Compiler` (jit.rs:332) — backend-agnostic by construction.
2. **Slang — a real second backend.** `impl Backend for Slang` (slang.rs:75), pure-Rust string emitter, no SDK dep. Relocates to the Slang provider; depends on the core speller module (2 fns).
3. **seam (Fuel) — NOT a clean neutral shim.** `seam::synthesize` (jit.rs:868) is backend-generic (injects backend+compiler). But `BaracudaSynthesizer` (jit.rs:1065, the `fuel_kernel_seam::Synthesizer` Fuel calls) **hardcodes `Cuda`** (jit.rs:1019,1091) — Fuel's `Synthesizer::synthesize` carries no backend param, so the impl picks internally. **Disposition:** `BaracudaSynthesizer` travels with the CUDA backend (or a Baracuda seam-integration crate). The generic `seam::synthesize` could live in core only if core accepts a `fuel-kernel-seam-types` dep (a Fuel-protocol dep, not a vendor dep) — **judgment: keep the Fuel seam entirely on the Baracuda side** for now (Fuel is the external consumer; the neutral core shouldn't carry a consumer-specific protocol dep pre-freeze). Revisit if a second consumer wants the same envelope.

## The oracle (conformance contract)

The oracle is **independent by construction** — it never calls `crate::cuda` (a Rust plan-interpreter); 41 emitter mentions are *independence assertions* ("shares zero lowering code," "a bug cannot hide in both"). The only coupling is ONE doc line-ref: `oracle.rs:1174` ("Mirror the emitter's reduction scope (cuda.rs:2192-2209)"). **Task:** restate that one line's semantics (K-ascending accumulation, WideFloat accumulator, reduction scope, NaN propagation) as an explicit backend-agnostic **conformance contract** in `unpopped` — the oracle becomes the reference every backend must satisfy (what Vulkane needs). Enforcement per-backend: the CUDA backend runs the oracle differential in its own tests, and `tools/kiss-ref-diff` runs it cross-repo, so semantic drift fails loudly. Small mechanical (one comment) / medium design (write the spec).

## The vocab carve (step one — the small first gate)

`unpopped-vocab`'s only `baracuda-types` coupling is **one symbol**: `DeviceRepr` (element.rs:62), a marker trait (device_repr.rs:26). Driver-free **confirmed** via `cargo tree` (no `cuda-sys` transitively). **DECISION (Eric-delegated): neutralize via re-export, not vendor/fork** — `unpopped-vocab` owns the canonical `DeviceRepr`, `baracuda-types` re-exports it. This avoids duplicate `unsafe impl` blocks + drift-risk and keeps `unpopped-vocab` free of any `baracuda-*` back-pointer.

**Mechanism — SPLIT `device_repr.rs` at the orphan-rule line (do NOT move it whole — that cycles: `:10 use crate::numeric::{Half,BFloat16,Complex32,Complex64}` are `baracuda-types` types):**
- → `unpopped-vocab` (a new `device_repr.rs`): the `DeviceRepr` trait (`:26`), `impl_device_repr_primitive!` + its std-primitive impls (`:28-39`), the `[T;N]` blanket (`:49`), the tuple macro + impls (`:51-73`), and impls for the FOREIGN primitives vocab's `KernelDtype` bound needs (`half::f16`/`bf16`/`float8` — orphan rule requires these live with the trait owner). Neutral tests move too.
- → stays in `baracuda-types`: the 4 impls for its OWN `numeric::{Half,BFloat16,Complex32,Complex64}` (`:42-45`), rewritten against the re-exported trait (local type + foreign trait = legal). `baracuda-types` adds `pub use baracuda_kernel_vocab::DeviceRepr;` and drops its own definition. Layering inverts correctly: `unpopped-vocab` (deps only `half`/`float8`) is the foundation, `baracuda-types` builds on it. No cycle.
- **Validator: whole-workspace `cargo build` green** — if any `KernelDtype: DeviceRepr` bound stops resolving for a foreign primitive, move that impl to the trait-owner (vocab).
- **Follow-up (not blocking):** `unpopped-vocab::element::Complex32/64` shadows `baracuda-types::numeric::Complex32/64` — after the split they straddle two crates; add a boundary doc note + decide to canonicalize on the neutral `unpopped-vocab` definition later.
- **Known tradeoff — `no_std`:** `baracuda-types` is `#![no_std]`; it now takes a non-optional dep on the *std* `baracuda-kernel-vocab`, so it is no longer strictly `no_std`-*linkable* for a bare-metal target (its own code stays `core`-only; the weakening is transitive). In-tree this is inert — the workspace is host/CUDA only and both the targeted and full builds are green. **Decision (Eric-delegated): accept it.** If a `no_std` consumer of either `baracuda-types` or `unpopped-vocab` ever materializes, the fix is a tiny `no_std` leaf owning just the trait + primitive/array/tuple/half/f8 impls (deps `half`+`float8` only), which both crates dep — but that adds a 4th neutral crate for a guarantee nothing currently exercises, so it is deliberately deferred, not adopted.

## ★ Deferred theme — BEHAVIORAL neutrality (post-carve, post-publish)

Dependency-neutrality (the core carries no `baracuda-*`/driver dep) is what the carve achieves. It is NOT the same as **behavioral** neutrality — the neutral core still *encodes CUDA-specific spelling and semantics* in shared code. These are invisible today for one structural reason: **CUDA is the only backend on the path**, and the neutral backends either decline the dtype (`CpuC` declines f16/bf16) or never call the code (`Slang` never calls `scalar_ctype`). **A Vulkan/SPIR-V backend (Vulkane) trips all of them on day one** — it is the first backend that both supports f16 and needs the conformance semantics. Track as ONE follow-up ("complete the `Backend` abstraction — the neutral core must encode no single backend's spelling or semantics"), so Vulkane onboarding surfaces them together, not three times. The three known instances:

1. **f16/bf16 ctype + half load/store intrinsics** — `cfamily::scalar_ctype`'s F16/Bf16 arms return `__half`/`__nv_bfloat16`, and `half_load_intrinsic`/`half_store_intrinsic`/`promote_load_f32`/`demote_store_f32`/`cast_scalar`'s half arms emit `__half2float`-class CUDA intrinsics. Silent-wrong-output hazard: a neutral-looking API silently spells `__half` into a non-CUDA backend's output, uncatchable by any neutral test. Fix: f16 ctype + half load/store become `Backend` methods; the neutral core declines f16 rather than mis-spelling it.
2. **`effective_count_width`** — the reduction/count width decision, reached by `contract.rs`. If CUDA-specific, becomes a `Backend` method (its dependency-severing minimal fix, item 3, is the first installment of this theme).
3. **oracle conformance semantics** — K-ascending / `WideFloat` accumulation / reduction-scope / NaN-propagation stated by reference to one emitter; the deferred half of item 4 is writing them as backend-agnostic invariants.

Each has a **dependency-severing minimal fix that is prep-NOW** (items 3/4/5 sever the `crate::cuda::` reach so the carve builds under `deny`), distinct from the **behavioral depth deferred here**. Owner: Baracuda-side (I hand Vulkane the conformance story). *(Credit: theme synthesized with the Unpopped peer, 2026-08-06.)*

### Vulkane-gate expansion (2026-08-06, peer-relayed Vulkane review of the trait — the "freeze after Vulkane" input the plan always waited for)

**4th behavioral leak — `const_lit` (backend.rs:198), a TRANSFORM-LAYER SOUNDNESS issue (worse than spelling).** A free `pub fn` (not a `Lowering` seam), called from `lower_expr` on the neutral path every backend traverses. Emits `"NAN"`/`"INFINITY"` (C `<math.h>` macros — absent in GLSL/HLSL/SPIR-V) + needs a suffix seam (unsuffixed `1.0` is 32-bit in GLSL). The soundness part: its doc says the optimizer's bit-preservation proofs are grounded in *double-promoted, correctly-rounded* (C) semantics — GLSL has no implicit float→double promotion, so **every optimizer rewrite whose proof invokes double promotion must be re-proven per backend** or it silently emits wrong bits. Item: *audit every optimizer rewrite for dependence on C arithmetic semantics; re-prove or backend-gate each — BEFORE any non-CUDA backend runs the optimizer.* Needs a `const_lit` Lowering seam.

**f16 RE-SCOPED — IR-shaped debt, not emitter-shaped (Vulkane).** SPIR-V has no `__half`/half intrinsics: f16 = `OpTypeFloat 16` + ordinary `OpFAdd`/`OpLoad`, packed = `OpTypeVector %half 2` + ordinary vector ops. So the half-intrinsic node **shouldn't exist in the neutral IR at all** — carry typed f16 + typed loads/stores; the promote/demote/pack (`__half2float`/`__hadd2`) live only in the CUDA emitter. Supersedes the earlier "abstract behind a Backend method" framing.

**Six Backend-trait gaps (Vulkane, priority order) — two are publish-boundary-expensive:** #1 `GeneratedKernel.source: String` can't hold a SPIR-V `[u32]` word stream (folds into #2); **#2 no ABI/binding manifest** (the big one — SPIR-V needs structured descriptor-set/binding-per-operand/push-constant/workgroup; `Variant.launch_note: String` is prose a host can't act on); **#3 no `&Target` param / per-device** (`effective_count_width` varies by device — lean *per-device-explicit* + document it); #4 `supports_dtype(dtype)->bool` too coarse (Vulkan support is per-storage-class — f16 math vs storage-buffer vs push-constant are separate bits); #5 `lower`→`GeneratedKernel` not `Result` (Vulkan decline reasons richer than dtype); #6 no spec-constant representation (Vulkan pipeline-creation binding time; `lower_variants` is the wrong binding time). Only #1/#2 get expensive after publish (they're *shapes of published data*).

**warp-32 — the sharpest finding.** A fixed warp of 32 is probably the most load-bearing CUDA assumption in the reduction schedules; subgroup width (32 NVIDIA / 64 AMD / 8-32 Intel) changes reduction-tree shape → bits. The conformance contract must make it *explicit*.

**Root cause (deepens the protocol answer): CUDA resolves the device at compile time, Vulkan at pipeline-creation time.** So a Vulkan provider genuinely CANNOT populate §6.8 `determinism_class`/`bit_stability`/accumulator at provision time — the answer doesn't exist yet. Third answer to identity-vs-request: some guarantees are neither structure_key identity NOR a provision-request field; they emerge POST-specialization in the *returned* contract. "Requirements are identity" holds only for guarantees knowable pre-specialization, and Vulkan proves that set is smaller than it looks.

**KISS Q1 ruling (2026-08-06):** `determinism_class` NOT a gap (derived from op+operand classes; provider can't offer a stricter variant — CONTRACT-6.8-0003). `bit_stability` NOT a gap (per-kernel advertised fact, correctly absent from identity). NO provider numeric-posture handshake anywhere (announce u64 carries none; availability records MUST NOT carry guarantees) — guarantees live ONLY per-kernel in the returned contract. **The accumulator IS the gap = KISS-CLASSIFY-6.7-0012's parked forward-requirement → sk4** (schema-affecting, 3-way structure_key regen; KISS scheduling sk4 timing with Eric). Interim rule: absent an accumulator coordinate, a non-contraction reduce MUST accumulate in compute dtype. **Confirmed provider-side evidence: my emitter WIDENS — contract.rs:1828-1831 builds an F32Strict Sum reduction and asserts `accumulation_type: f64`; so Baracuda ships f64-widened non-contraction reductions, contract-advertised but sk3-un-discriminable — the parked case in production.** (AccumSpec/WideFloat is contraction-only, ir.rs:2232/:2266; `Access::Reduction{op,axes,keepdim,epilogue}` has no accumulator field — the f64-widen is a hardcoded emit-time policy.) **sk4 now has THREE coordinates: (a) this non-contraction accumulator, (b) math-precision, (c) per-operand dtype** — Fuel's PagedAttn probe declined dense-SDPA on `MixedDtype` because `OperandKey` carries no dtype, so per-operand dtype (float data + integer indices in one indexed region) is discarded at key derivation and the region is undescribable; the uniform-dtype gate declines HONESTLY, not as a limitation to route around.

### PRE-PUBLISH GATES for `unpopped` 0.1.0 (publish does NOT auto-follow the carve — peer-owned gate, I confirm resolution in the carve-ready ping)

1. **`#[non_exhaustive]` batch (Baracuda-side, before the cut):** `VariantFidelity` (free — core matches, backends produce; the 3-way accumulator/order/compensated split needs it). `GeneratedKernel` / `Variant` / `Lowering` (+ a constructor each — the cost that lets #1/#2 land in 0.2 without a major bump; `Lowering` needs the `const_lit` seam now). **Plus `JitError`** (the JIT decline taxonomy — 6 variants, currently plain `Clone,Debug,PartialEq,Eq`): the STRONGEST case — a decline enum always grows, and Fuel's PagedAttn probe proved decline-specificity is load-bearing (`Declined{MixedDtype}` was "the single most useful byte in the investigation", one conclusive run vs an open hunt across region-size/op-support/budget); free (a wildcard arm wherever matched). **Keep `MixedDtype` a DISTINCT reason** — do NOT fold it into a generic decline even when sk4 reshapes per-operand dtype; the generic JIT surface (`Compiler`/`synthesize`/`JitError`/`StubCompiler`) stays in neutral kernelgen→`unpopped`, matching KISS-Synth §6.6's never-panic typed-decline taxonomy. Validated against Cuda/CpuC/`baracuda-cuda-emit`(heaviest constructor)/Slang/goldens.
2. **`Access::Reduction` accumulator field (sk4 seam), before the cut** — defaulted to the current emitter policy (f64-widen where it widens, else compute-dtype) so it's behavior-preserving and sk4's coordinate slots in without a breaking variant-field change. (The enum's `#[non_exhaustive]` does NOT protect struct-variant fields.)
3. **The alpha.78 all-zero regression UNDERSTOOD** (Fuel bisect: alpha.76/77 PASS, alpha.78 FAIL; `relu(add(a,b))` → all zeros, output buffer never written, no error; symbol byte-identical → in the emitted body or launch ABI, NOT identity). "Understood" not necessarily "fixed on my side" — if it's Fuel's marshalling vs a changed signature, that's a valid resolution. **d5be1ad8 RULED OUT** (contraction-only). Leading hypotheses: the elementwise output-store generalization (`store_expr_of`/`out_ctype_of`) or a launch-ABI/signature change (F64-param/count work). Discriminate via repro (generate `relu(add)` scalar at alpha.77 vs current — signature-changed⇒ABI, body/store-changed⇒emitter, source-identical⇒jit/launch path) + launch on the 4070. Post-carve (needs settled tree + build).
4. **On-device output-CORRECTNESS test (the standing gap-closer):** goldens prove *bytes*, on-device proves *compilation*, but nothing proved *the launched kernel writes the right answer* — the exact gap that let all-zero-no-error ship. Step-5 validation upgrades to launch + output-compare of `relu(add)` (+ a small op matrix) on the 4070. The neutral generator should own the consumer-shaped version (peer makes it an `unpopped` carve requirement); I own the `baracuda-cuda-emit` mirror.

**kiss-ref bump (dev-dep, post-carve):** 0.1.0→0.2.3 crosses the 0.2.0 major boundary but the only compile break is ONE match arm — `Node::ConstBits(_)` — IF `kiss_ref_diff` matches `Node` exhaustively (else zero; construct-only = no migration). `eval_recipe` sig unchanged, `ScalarFloat` no-op (non-impl), rest additive; `dets[cmp]` Ulp→OIN is a free value-level correctness pickup. Verify actual `Node` usage + real pin at point of use.

Done Baracuda-side PRE-CUT (on `feat/unpopped-extraction-prep`), so the peer filter-repos an already-neutral vocab. **Consumers to re-point** after publish: `baracuda-kernels-types` (stays Baracuda-side, workspace dep → published `unpopped-vocab`) and `tools/kiss-ref-diff`.

## Consumers (verified — re-point these)

- `baracuda-kernels-bench` (Cargo dep + 2 test files) → published `unpopped`.
- `tools/kiss-ref-diff` (own `[workspace]`, path-deps kernelgen + vocab + `baracuda-driver`) → published `unpopped` + `unpopped-vocab`, keeps `baracuda-driver`. **Stays Baracuda-side as the cross-repo on-device differential/integration harness** (NVRTC JIT + GPU launch — can't live in a driver-free core).
- `baracuda-kernels-types` → published `unpopped-vocab` (the vocab-carve re-point).
- `baracuda-seam` — doc-comment reference only (lib.rs:183), NOT a dep. Grep-cleanup only.

## "Don't move" / caveats

- **Goldens + `.gitattributes eol=lf`** (PR#12 `d5be1ad8`) — CORRECTED 2026-08-06: the byte-identity goldens travel with the **CUDA emitter → `baracuda-cuda-emit`** (NOT `unpopped`), so the earlier "must travel with unpopped" note was WRONG and its path-scoped `.gitattributes` rule silently matched nothing (the Unpopped peer's carve caught it after 32 `ondevice/*.cu` harnesses landed as CRLF). The `.gitattributes eol=lf` rule must match `*.cu` **workspace-wide**, not a single `.../goldens/*.cu` path, so a relocation can't drop the guarantee. `unpopped` itself ships NO goldens.
- **kiss-ref dev-deps** (`kiss-ref-core/ops/classify`, `[dev-dependencies]`) — travel with the `kiss_ref_diff` test module.
- **Workspace inheritance to materialize** in Unpopped: `edition`, `rust-version`, `license` (→ `MIT OR Apache-2.0`), `authors`, `repository`, `homepage`, `[lints] workspace`.
- **Rewrite 12 rustdoc intra-doc links (BUILD-BLOCKING) + grep stale prose (cosmetic):** 12 `[crate::cuda::…]` rustdoc links live in core-clean files (backend.rs 1, ir.rs 3, plan.rs 9 — e.g. plan.rs:233 `[crate::cuda::Cuda::lower]`, ir.rs:1887 `[crate::cuda::assert_offsets_lowerable]`, backend.rs:6 `[crate::cuda::Cuda]`). These are NOT code deps but rustdoc-RESOLVED links — they become broken intra-doc links when `cuda` leaves the crate, and the workspace's post-docs-sweep `deny` posture (`missing_docs` denied, `broken_intra_doc_links` hard-fails under `-D warnings`) turns that into a build failure. Core can't dep on the relocated backend to keep them resolving → rewrite each to plain text or the trait-level concept (`Backend::lower`). SEPARATELY, `baracuda_kernelgen::`/`baracuda_kernel_vocab::` prose references (e.g. `baracuda-seam:183`) go stale on rename — plain grep-and-fix, cosmetic.
- **Ignore `.claude/worktrees/agent-*`** in any scan — stale worktree copies shadow the real crates.

## Sequencing + effort profile

1. **Vocab carve** — *small* (first publish gate): carve `unpopped-vocab`, standalone-green, publish → `kernels-types` + `kiss-ref-diff` re-point. De-risks the big publish.
2. **Oracle contract-promotion** — *small mechanical / medium design*: one line-ref restated + write the conformance spec.
3. **C-family speller module extraction** — *medium refactor* (the enabler): carve the ~14 shared spellers from `cuda.rs` into a neutral core module; `cpu_c.rs`/`slang.rs`/`contract.rs`/`cuda.rs` re-point to it.
4. **`Backend::effective_count_width` trait method** — *small*: makes `contract.rs` backend-generic (removes its last `cuda::` reach).
5. **CUDA-backend carve** — *large, mechanical* (~17k lines): move `cuda.rs` (post-speller-split) + `NvrtcCompiler` + `BaracudaSynthesizer` to the Baracuda CUDA-backend crate; relocate the CUDA-lift test arm (convert.rs:758); wire the `run_cli` driver + distribution bundler in place of `bin/kernelgen.rs`'s hardcoded `&Cuda`; **rewrite the 12 `[crate::cuda::…]` rustdoc links in core** to plain-text/trait-level (build-blocking under the workspace `deny` lints once `cuda` leaves).
6. **Slang-backend carve** — move `slang.rs` to the Slang provider (depends on the core speller module).
7. **standalone-green** the carved crates outside the workspace → **publish `unpopped`** → Baracuda re-points (`kernels-bench`, `kiss-ref-diff`) + the CUDA backend crate consumes published `unpopped`.
8. **Fuel migration handoff** (Baracuda-owned): tell Fuel the rename + the new `Synthesizer` location; leave-and-deprecate the old `baracuda-kernelgen`/`-vocab` alpha.78 slots (don't yank — Fuel is pinned).

**Neutral-ABI freeze** (the IR + the request/response envelope + the oracle conformance contract) waits for Vulkane's backend to exercise it.
