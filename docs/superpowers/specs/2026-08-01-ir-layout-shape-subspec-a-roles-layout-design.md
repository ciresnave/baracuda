# Sub-spec A — General contraction operand roles + layout (detailed)

**Parent:** `docs/superpowers/specs/2026-08-01-ir-layout-shape-parent-design.md`.
**Status:** design approved (brainstorm, 2026-08-01). The keystone sub-spec — B/C/D extend
the machinery introduced here. Merges frontier items 1 (View×Contraction) + 4 (general
einsum roles).

---

## A.0 The unblock

Fold a `Permute` / `Broadcast` / rank-3 role reorder on an lhs/rhs contraction operand into
the M/N/K address math instead of materializing a `contiguize` copy. Directly enables
transposed / column-major / mixed-layout / batched matmul and the **SDPA QKᵀ** path (Q read
canonical, K read transposed, contracted over `head_dim`).

## A.1 Representation (recap of parent §2)

Layout is a **discrete class in `ContractionKey`** (re-keys → distinct cells), lowered
through **one stride-binding emitter**. Semantics come from `ContractionAxes.roles`
(`ir.rs:1375`), physical storage order from each operand's `View::Permute` (`ir.rs:1436`).
The emitter derives every stride as an **extent-product** of the existing `m/n/k/B` launch
args — **no new runtime stride args** (those are sub-spec D).

## A.2 Role-vector legality (open-question Q4)

Replace the `plan.rs:366` pin (which whitelists only `ContractionAxes::matmul()` /
`batched_matmul()`) with a general predicate `validate_contraction_roles(&ContractionAxes,
lhs_rank, rhs_rank) -> Result<()>`:

- **Well-formedness:** `axes.lhs.len() == lhs_rank`, `axes.rhs.len() == rhs_rank`; every
  axis carries exactly one role.
- **lhs:** exactly one `FreeM`; **exactly one `ContractedK`** (v1 single-K, §A.6); zero or
  more `Batch`; **no `FreeN`**.
- **rhs:** exactly one `FreeN`; **exactly one `ContractedK`**; zero or more `Batch`; **no
  `FreeM`**.
- **Batch correspondence:** the `Batch` axes on lhs and rhs must be equal in count (their
  extents match by caller precondition, not plan-validated — §parent 4). Batch axes flow to
  the output frame.
- **K pairing:** lhs's single `ContractedK` pairs with rhs's single `ContractedK` (matched
  extent = caller precondition).

`matmul()` and `batched_matmul()` **pass** this predicate (they become instances, not the
whitelist). Placement is unconstrained: `[M,K]`, `[K,M]`, `[B,M,K]`, `[M,B,K]`, … all legal.

## A.3 `ContractionKey` layout class

`ContractionKey` (`structure_key.rs:239`) gains a **per-operand storage-order class** for
the two core operands: `lhs_order`, `rhs_order`. Each is a **Copy, heap-free** encoding of
the operand's storage permutation (the `View::Permute.perm` projected to the key; e.g. a
Lehmer/index code in a `u8`, since v1 ranks ≤ 3). **Identity default** ⇒ existing row-major
cells serialize **byte-identically** (additive codec, no `STRUCTURE_KEY_VERSION` bump for
the identity case — the same additive discipline `batch: Option<SizeClass>` already uses; a
non-identity order adds a token component only when present).

- **Derivation:** generalize `derive_contraction`'s `dense()` predicate (`structure_key.rs:616`,
  and `dense3()` `:657`) into `classify_mat_layout(&OperandDesc, &[AxisRole]) -> LayoutOrder`.
  It **stops rejecting** non-row-major operands; instead it reads the storage order from the
  strides (which axis is unit-stride, then the outer order) exactly as `classify_contiguity`
  (`:784`) already inspects strides, and returns the order class. A layout it cannot classify
  as a **packed** permutation (i.e. a genuinely non-packed / runtime-strided operand) still
  declines in v1 → sub-spec D.
- **Broadcast** rides the existing `OperandKey.bcast` (`structure_key.rs:279`) — already in
  the key; no new field. A broadcast axis contributes a stride-0 (dropped) term (§A.4).
- **Output stays Canonical row-major in sub-spec A** (`out_order` = identity). Output views
  are sub-spec B; A does not touch the store side.

## A.4 The stride-binding emitter

Replace the three hardcoded offset expressions in `emit_contraction` (`cuda.rs:3269`) — lhs
`in0[{lb}mm*k + kk]`, rhs `in1[{rb}kk*n + col]`, out `out[{ob}mm*n + col]` — with **one
binding** computed per operand from its role→axis map (`ContractionAxes`) and storage order
(`View::Permute`):

> `stride(role R on operand O)` = ∏ extents of O's axes **storage-inner** to R's axis
> (a product of the `m/n/k/B` launch args). Operand address =
> `Σ_roles coord(role)·stride(role)`, where `coord(FreeM)=mm`, `coord(FreeN)=col`,
> `coord(ContractedK)=kk`, `coord(Batch)=b`. Broadcast axes (`OperandKey.bcast`) contribute
> **no term** (stride 0), exactly as `offset_expr` (`cuda.rs:6349`) already drops them for
> Elementwise.

**Canonical is the identity special case** — the emitted string is byte-identical to today
for every existing row-major cell. The batched prefixes (`lb`/`rb`/`ob`, `cuda.rs:3386`)
become the `Batch`-role term of the same binding.

**Worked example — SDPA QKᵀ (transposed rhs):**

| rhs layout | storage order | `stride(K)` | `stride(N)` | address | coalesced? |
|---|---|---|---|---|---|
| Canonical `[K,N]` | K outer, N inner | `n` | `1` | `kk*n + col` | yes (adjacent `col`) |
| Transposed `[N,K]` | N outer, K inner | `1` | `k` | `col*k + kk` | no (adjacent `col` strides by `k`) |

Both are **correct** and oracle-validated; the transposed read is non-coalesced → its
coalescing-restoring schedule is a deferred variant (§A.6).

## A.5 Lifted gates & validation wiring

1. **`plan.rs:1043`** (Elementwise-only view gate): lift so a non-Identity `View` is
   admitted on `Access::Contraction` operands. Keep the gate for the other non-Elementwise
   classes (they are not in this sub-spec). Retain the multi-output guard (`plan.rs:1051`).
2. **`plan.rs:366`** (role pin): replace with `validate_contraction_roles` (§A.2).
3. **View legality** stays `assert_valid_views` (`plan.rs:1027`) — a `Permute` must be a
   true permutation; agreement with runtime strides remains a caller precondition.
4. **`cuda.rs:1757`** backstop (`assert_views_lowerable`): extend to accept the contraction
   binding path.

## A.6 Scoping (ratified 2026-08-01)

- **Correct-first / optimize-as-variant.** v1 emits *correct* address math for every
  admitted layout, CPU-oracle bit-validated, even when non-coalesced. Coalescing-restoring
  schedules (smem-staged transposed reads) are **bench-gated variants**, added exactly as
  `contraction_splitk_variant` (`cuda.rs:3448`) was. Correctness is the gate; speed is a
  variant.
- **Single K-group in v1.** Exactly one `ContractedK` axis per operand (§A.2). General role
  *order* is fully supported; multi-group einsum (nested K-folds) defers (parent §8).

## A.7 Oracle matrix & on-device

Validate every admitted cell bit-exact against `oracle::evaluate` (`oracle.rs:1053`):

`{ lhs_order ∈ (Canonical, Transposed) } × { rhs_order ∈ (Canonical, Transposed) } ×
{ rank-2, rank-3 batched } × { broadcast on/off (batch-broadcast KV) }`

Then confirm the headline cells on the **RTX 4070** through the `gpu-run` lock: transposed-rhs
QKᵀ (rank-3 batched), and GQA broadcast-KV (rhs batch stride 0). Every on-device run:
`pwsh scripts/gpu-run.ps1 -Project baracuda -- <cmd>`.

## A.8 Files touched

| what | file | change |
|---|---|---|
| layout class + derivation | `crates/baracuda-kernel-vocab/src/structure_key.rs` | add `lhs_order`/`rhs_order` to `ContractionKey`; `dense()`/`dense3()` → `classify_mat_layout`; codec additive (identity default) |
| wire codec | `crates/baracuda-kernel-vocab/src/shape_expr.rs` | order-class token component (additive) |
| gates + role legality | `crates/baracuda-kernelgen/src/plan.rs` | lift `:1043` for Contraction; replace `:366` with `validate_contraction_roles` |
| emitter binding | `crates/baracuda-kernelgen/src/cuda.rs` | `emit_contraction` (`:3269`) stride-binding; backstop `:1757` |
| shape frame | `crates/baracuda-kernelgen/src/shape.rs` | reuse `output_shape` (`:150`) — no change expected |
| oracle cases | `crates/baracuda-kernelgen/src/oracle.rs` | add the §A.7 matrix |

## A.9 Out of scope (→ elsewhere)

- Output-side views / sliced writes → **sub-spec B**.
- Rank-change reshape → **sub-spec C**.
- Arbitrary / negative / stride-2 (runtime stride args) → **sub-spec D**.
- Multi-K-group einsum, TF32/tensor-core `AccumSpec` → **roadmap** (parent §8).
