# Baracuda ask — JIT-on-request protocol (§5): proposed JitRequest/JitResponse schema (2026-06-21)

To Fuel, on the **JIT-on-request** seam (Kernel-Seam Interop §5). You're building
the strategist side; we've built the **synthesizer** side and validated it on
real hardware. This proposes the wire contract between them — concretely, so the
two halves converge **before** either freezes a surface (the same propose-first
discipline that made FDX/FKC land clean). Companion to the ratified Profile v1.

TL;DR — our synthesizer half is **built and on-device-validated** (§2), so this is
an implementation-true proposal, not paper. The headline: **reuse the FKC §3
pattern-node grammar as the `JitRequest.region` form** (the same grammar you
already defined for `pattern:` matching, read in the "input" direction) — so
there's almost no new grammar to design. We propose concrete `JitRequest` /
`JitResponse` schemas (§3/§4), flag the eight decisions that need your agreement
(§5), and ask for your wire types so we reconcile the marshalling (§8). When both
halves are aligned and built, we flip `SeamCapJitOnRequest` on.

---

## 1. The division (agreed — restated for grounding)

Per §5.1, unchanged: **Fuel is the strategist** — it chooses *which*
primitive-subgraph region to fuse, *when*, and whether to *adopt* the result
(cost-gated). **Baracuda is the synthesizer** — it builds the best kernel for the
**Fuel-chosen** region and returns it. No backend-side opportunity-finding; the
inward e-graph §5.1 permits us is pointed only at the region you hand us. This
proposal is purely the *interface* across that line.

## 2. What we've built (so this is implementation-true)

In `baracuda-kernelgen::jit`, on branch `feat/kernel-specialization`:

- **`synthesize(request, backend, compiler) -> JitResponse`** — the full loop:
  region → op IR → schedule (your `structure_key`) → on-demand compile → FKC
  contract + recipe + link row. It reuses the AOT generator wholesale; the only
  new step is `region_to_op` (the inverse of `derive_pattern`).
- **On-device validated on sm_89:** a synthesized `relu(a+b)` region lowered to
  `.cu` and compiled by **nvrtc to real PTX**. That run surfaced and fixed a real
  AOT/JIT portability bug (our emitted source `#include <cstdint>` /
  `<cuda_runtime.h>` — nvcc-fine but nvrtc is headerless), so the generator now
  emits header-light source that compiles under **both** nvcc and nvrtc.
- **Adversarially audited** (a 20-finding pass): trust-boundary guards, backend
  injection, recipe-both-halves-from-one-node, artifact provenance + link row.

So the schema below is the contract our working code already implements (in
Rust-native form); we want to reconcile it with yours, not invent from scratch.

## 3. Proposed `JitRequest`

```text
JitRequest {
  region:       PatternNode      # the primitive subgraph to fuse — FKC §3 grammar (§5.1 below)
  n_inputs:     u8               # region input count; bind indices are exactly [0, n_inputs)
  op_category:  OpCategory       # your taxonomy; keys schedule legality (you choose it — strategist)
  operands:     [FdxOperandDesc] # inputs then output — the SAME minimal projection structure_key reads
  arch:         ArchSku          # target compute capability (keys the schedule)
  fused_op_id:  String           # the stable identity to register the synthesized op under
  budget:       { max_compile_ms: u32 }   # compile/resource ceiling (you set it)
}
```

## 4. Proposed `JitResponse`

```text
JitResponse {
  kernel:   { entry_point: String, source: String, artifact: bytes, kind: {Ptx|Cubin|Stub} }
  contract: String     # a full FKC contract block (accept/return/op_params/cost/precision/determinism)
  recipe:   { pattern: String, decompose: String }   # both halves (rev-4 §1)
  link:     LinkEntry   # (entry_point, structure_key, revision_hash) — resolves entry_point at load (FKC §12.6)
}
```

---

## 5. The eight decisions that need your agreement

### 5.1 Region = the FKC §3 pattern-node grammar (the key one)

We propose `JitRequest.region` is **exactly an FKC §3 `PatternNode` tree** — `Op`
nodes over the §4.1 graph-`Op` vocabulary, with `bind: i` leaves for the region's
external inputs — i.e. *the same grammar you already defined for `pattern:`
matching, in the input direction*. Our `region_to_op` is literally the inverse of
`derive_pattern`'s walk. **Benefit: no new grammar to design** — the JIT region
and the FKC pattern share one node form, and a synthesized op's `pattern:` is just
that region re-emitted.

**What we need confirmed:** that your "partial base map (primitive subgraph)"
serializes cleanly to that node form. One caveat we've scoped: the §3 grammar is a
**tree** and our op IR has no shared interiors, so a region with a shared interior
(a true diamond) tree-ifies (the kernel recomputes it). That's fine for the
elementwise-epilogue regions of increment 1 and matches your own §9 deferral of
interior node-identity — but if your base-map regions are DAGs you want preserved,
that's the first thing to coordinate.

### 5.2 Operands = the `FdxOperandDesc` projection (confirm)

`JitRequest.operands` is the inputs-then-output list of the *same* minimal
`FdxOperandDesc → OperandDesc` projection you already pass to `structure_key`. We
key the schedule cell from it verbatim (never re-derived). This just reuses the
ratified telemetry/structure-key contract — confirm it carries over to JIT.

### 5.3 Scalar params → `op_params` via `extract` (confirm round-trip)

A region `AddScalar`/`MulScalar` node → we lower its scalar to a **runtime
`Param`** (a launch arg), exactly as the AOT path does, and the emitted contract's
`extract:` pulls each scalar back from its graph path into `op_params`. So the
region's scalar attributes round-trip to op-params with no extra channel — confirm
that's how you want the synthesized op's params bound.

### 5.4 Recipe: `decompose` = the region (interim)

The synthesized fused op is, by construction, equivalent to exactly the region you
sent — so its `decompose` **is** that region. We derive both recipe halves from
one canonical node, so `pattern:` and `decompose:` are structurally identical and
both carry the scalar `extract:` routing. Since the *declarative decompose format*
is your §9-deferred item, we emit `decompose:` as the region's pattern-node
subgraph with a `decompose:` header (provisional). Confirm that interim, or hand
us the format when it lands.

### 5.5 Budget semantics

We treat `budget.max_compile_ms` as a **soft** ceiling: nvrtc has no compile-
deadline API, so it gates optimization depth / the inward e-graph's iteration
count at a coarse grain rather than a hard abort. We reject a zero budget. Confirm
soft-ceiling is acceptable, or tell us if you need a hard wall-clock abort (we'd
wrap compilation in a watchdog).

### 5.6 Target: backend, device vs. arch, dtypes

§5.2's `target` is `(backend, device, shapes, dtypes)`. We map: **backend** =
injected `&dyn Backend` (today CUDA; the request needs a backend selector when we
add Slang/Metal/CPU — propose a small enum tag you set); **shapes + dtypes** =
inside `operands`; **device** = folded into `arch` (`ArchSku`) for the schedule
key, with the *finer* device identity (ordinal / exact SM / driver) refined by the
compiler only when the artifact must be SM-specific. Flag: if you need to pin a
specific device ordinal in the request, we'll add a `DeviceTarget`.

### 5.7 Honest-miss error taxonomy

The synthesizer returns a typed error rather than a wrong kernel; you treat it as a
**miss** (keep the primitive subgraph, no fused kernel — exactly the honesty
invariant). Proposed set: `UnsupportedOp` (outside our IR vocabulary),
`UnsupportedDtype` / `MixedDtype` (no §5 base spelling / non-uniform operands),
`OperandArity` (`operands ≠ n_inputs + 1`), `Budget`, `Compile` (toolchain
failure, carrying the nvrtc log). Confirm a miss is non-fatal on your side.

### 5.8 Artifact: PTX + provenance + link row

`kernel.kind` tags the artifact (`Ptx` from nvrtc / `Cubin` / `Stub`) so your
loader **must refuse a `Stub`** rather than feed non-loadable bytes to the driver.
And `JitResponse.link` is the `link_registry` row that makes `entry_point`
resolvable to a `KernelRef` at load (§12.6) — without it an adopted JIT kernel
can't bind. Confirm you'll consume `link` (or that your adoption path resolves the
entry_point another way).

---

## 6. Transport — Rust dep vs. C-ABI trampoline

Two Rust projects can integrate via these types directly (you construct the
`PatternNode` region + the `FdxOperandDesc` projection and call `synthesize`),
mirroring how `structure_key` is called. If instead the seam expects JIT over the
C ABI like `baracuda_seam_hello`, we'll add a trampoline — but that needs the
region + operands marshalled to a C form, so tell us which before we build it.

## 7. Scope (increment 1) + deferred

**In:** elementwise-epilogue regions over the IR vocabulary (`Add/Sub/Mul/Div`,
`AddScalar/MulScalar`, the unary math/activations incl. `GeluErf` = exact erf),
**uniform dtype**, sole-consumer trees, CUDA backend. **Deferred (with you):**
mixed-operand dtypes, DAG/diamond regions (interior node-identity), the full §4.1
op vocabulary (`Maximum`/`Pow`/reductions/…), the inward e-graph optimizer, and a
hard compile-deadline. None blocks the protocol shape.

## 8. What we need from Fuel + the capability flip

1. Your **`JitRequest` / `JitResponse` wire types** (so we reconcile marshalling —
   like FDX, the byte form is yours; we map ours onto it).
2. Confirmation of **§5.1 region = FKC §3 grammar** (5.1) and the operand
   projection (5.2).
3. Answers on **budget** (5.5), **device pinning** (5.6), **miss-is-non-fatal**
   (5.7), and **you consume `link`** (5.8).
4. **Transport** choice (§6).

Once aligned and both halves are built, we advertise **`SeamCapJitOnRequest`** at
the handshake and the seam goes live. Our half is implementation-true today — send
the wire types and we'll reconcile fast.
