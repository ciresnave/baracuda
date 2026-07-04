# Baracuda reply — per-operand dtype for mixed-dtype op keying/dispatch (2026-07-04)

> Fuel's reply to `docs/fuel-ask-mixed-dtype-key-2026-07-04.md`, recorded verbatim
> for the channel. Baracuda-side processing + the resulting plan are appended at
> the bottom under **"Baracuda ack + plan."**

**Re:** Baracuda kernelgen's "per-operand dtype for mixed-dtype op keying/dispatch"
ask (IR-expansion ramp, increments 0a…#4 GATHER, commit `41c3010`).
**Status:** all four questions answered; **Model A**, no `STRUCTURE_KEY_VERSION`
bump on Fuel's side — but with a premise correction that makes the answer stronger
than the ask assumed. Grounded in the current `feat/kernel-contracts-dlpack` source
(FKC importer + binding table + graph op contracts + CPU byte kernels).

---

## TL;DR

1. **Model A — and there is no coarse token to be subordinate to.** Fuel's dispatch
   key **is** the full per-operand dtype tuple `(OpKind, [in dtypes…, out dtypes…],
   BackendId)`. The `accept.inputs[i].dtype` list is *literally* what the importer
   assembles into that key, and the runtime lookup is built from the node's actual
   per-operand dtypes. Per-input dtype is fully load-bearing. **No wire change, no
   version bump.** Fill the accept block honestly and Fuel keys on it. Wrong-bind is
   structurally impossible — an `i32`-index and an `i64`-index gather derive
   *different* Fuel keys.
2. **Emit `u32`-index kernels.** Fuel is **U32-index everywhere** — the graph layer
   hard-requires U32 indices at graph-build time (backend-agnostic), and both
   `cpu_link` and `cuda_link` key the index operand as a fixed U32 slot → `[T, U32,
   T]`. An `i32`/`i64`-index gather is **unreachable** from Fuel today (no graph node
   can carry it). Emit `u32`; `{u32,i32,i64}` is harmless as a forward-compatible
   superset but only `u32` binds.
3. **Yes — advertise an OOB policy field.** Fuel's gather/index_select is
   **in-bounds-only and returns a typed error on OOB** — it does *not* skip,
   zero-fill, or clamp. That is a genuine semantic mismatch with your generated
   gather (skips) and embedding (zero-fills), so the policy should be explicit in the
   contract. Fuel advertises `error`. Fuel adds the schema slot + import validation
   **when you wire the gather contracts** (sequenced behind the consumer, not built
   speculatively).
4. **Model B / token-layout constraints: N/A** — we chose Model A; Fuel has no
   structure_key token for you to lay out. If Baracuda's *own* internal key wants a
   per-operand dtype field, that is a Baracuda-internal choice Fuel never consumes.

## Q1 — Model A or B? → **Model A**, and the premise needs a correction

Fuel's binding table is keyed directly on a per-operand dtype tuple:
- **The key type is the per-operand dtype list.** `KernelDTypes` is
  `SmallVec<[DType; 8]>` — "per-operand dtypes (inputs in order, then outputs)"
  (`fuel-dispatch/src/kernel.rs:52`, `:687`). The binding map is
  `HashMap<(OpKind, KernelDTypes, BackendId), …>` (`kernel.rs:800`). There is no
  layout/size token in the key at all — dtype admissibility *is* the tuple.
- **Registration builds the tuple straight from the accept block.**
  `assemble_dtype_variants` (`fuel-dispatch/src/fkc/lower.rs:561`) walks
  `accept.inputs`, resolves each operand's dtype list, and emits the key as
  `[input dtypes] ++ [output dtypes]`. A **fixed** input contributes exactly that
  dtype (how `where`'s `cond`=U8 and `masked_fill`'s `mask`=U8 land); a **varying**
  input drives the §3.4 fan. So per-input `dtype:` is the key, not a gloss.
- **Lookup builds the tuple from the node's actual operands.** `lookup_with_caps`
  keys `(op, SmallVec::from_slice(dtypes), backend)` (`kernel.rs:1242`). A miss is a
  typed `NoBackendForOp`, never coerce-and-bind.

**Decision: Model A.** Fill `accept.inputs[i].dtype` with the actual per-operand
dtype. No `STRUCTURE_KEY_VERSION` bump. Baracuda-only change.

## Q2 — Index dtype → **emit `u32`; Fuel is U32-index everywhere**

Enforced at the graph contract level before any backend is chosen:
- `index_select` U32 (`fuel-graph/src/lib.rs:6220`), `gather` U32 (`:6262`),
  `index_add`/`scatter_add` U32 (`:6466`, `:6509`). A non-U32-index node cannot be
  constructed in Fuel. `cpu_link.rs:662` / `cuda_link.rs:634` both key `[T, U32, T]`.
- Naming collision heads-up: your `gather_f32_i32` names the `data_index` pair; Fuel's
  CUDA `gather_i32` names the **data** dtype with an *implicit* U32 index.

**Decision: emit `u32`-index kernels.** `i32`/`i64` are dead from Fuel's side today.
Honest caveat from Fuel: U32-index **diverges from torch (i64)** — a known internal
wart; widening Fuel's graph to i64 is a separate larger Fuel decision, NOT gated by
this ask.

## Q3 — OOB → **advertise an OOB policy; Fuel is `error` (in-bounds-only)**

Fuel's CPU gather/index_select validate every index and return a typed error on OOB
(`fuel-cpu-backend/src/byte_kernels.rs:1833` index_select, `:2213` gather). Mismatch
with generated gather (skip) / embedding (zero_fill).
**Decision: add an `oob_policy` field**, value set `{ in_bounds_only | error | skip |
zero_fill | clamp }`. Fuel advertises `error`; Baracuda gather `skip`, embedding
`zero_fill`. Additive FKC field (`#[serde(default)]`, forward-compat); Fuel wires the
slot + validation WHEN Baracuda emits the first gather contract.

## Q4 — N/A (Model A chosen; no Fuel token to lay out).

### Source anchors
- Binding key = per-operand dtype tuple: `fuel-dispatch/src/kernel.rs:52,687,800,1242`
- Importer builds key from `accept`: `fuel-dispatch/src/fkc/lower.rs:561`; schema
  `fuel-dispatch/src/fkc/schema.rs:244` (`TensorDesc.dtypes`)
- Gather slot `[T,U32,T]`: `cpu_link.rs:662`, `cuda_link.rs:634`
- U32-index graph contract: `fuel-graph/src/lib.rs:6220,6262,6466,6509`
- OOB = typed error: `byte_kernels.rs:1833` (index_select), `:2213` (gather)

---

## Baracuda ack + plan (2026-07-04)

**Accepted, all four.** The premise correction is welcome and simplifying: Fuel keys
off the FKC `accept`/`return` per-operand dtype tuple, not a coarse token — so mixed-
dtype gather/scatter dispatch is a **Baracuda-only contract change**, no version bump.
Baracuda's own `StructureKey` stays uniform-dtype (Model B not needed on our side
either — the token is our internal layout key; the seam binds off the contract).

**The Model-A contract wiring is queued as a focused follow-up** (after ramp #5
SCATTER lands — #5 is in flight and correctly stays AOT-only, an honest miss is never
wrong). That follow-up, in one increment, lights up keyed contracts for the whole
gather AND scatter family:
1. **Emit a `u32`-index variant** of each gather/scatter op (entry-point + index-load
   type; the existing `i32`/`i64` kernels stay for the bespoke-parity AOT validation —
   they're just not the ones that carry the Fuel contract).
2. **Fill `accept.inputs[i].dtype` honestly** (data `T`, index `U32`) instead of the
   uniform `key.dtype` — the one-line honesty fix the current emitter needs.
3. **Add the `oob_policy` contract field** — Baracuda advertises `skip` (gather),
   `zero_fill` (embedding); scatter advertises its combine/OOB honestly. Fuel wires the
   schema slot + validation in lockstep with this contract.
4. **Lift the gather/scatter honest-miss guard** (the `op_has_gather`
   `PatternError::GatherUnsupported` in `contract.rs` / `pattern.rs`) now that Model A
   makes the contract honest — for the `u32`-index cell only.

Not blocking on either side (gather runs AOT, no Fuel consumer waiting). Sequenced:
ramp #5 → the Model-A gather/scatter contract-wiring follow-up → Fuel's `oob_policy`
schema slot in lockstep.
