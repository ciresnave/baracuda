# Baracuda ask — TOPK/BOTTOMK shipped (runtime-k row-sort cap; the SELECTION half of sparse MoE); no `top_k` OpKind on your side yet — recorded, NOT a blocker (2026-07-09)

**No action needed now.** This is a propose-first heads-up in the
landing-doc "radar item" class: a new kernel capability exists that is
deliberately NOT advertised (AOT-only, the correct posture), and we are
recording exactly what a future `top_k` call path would take on your
side — before anyone wires it.

## What shipped (Baracuda, increment 10 on `feat/kernel-specialization`)

`OpDef::row_topk` / `OpDef::row_bottomk` — the **runtime-k cap** on the
innermost-axis row sort. It is the strict generalization of the shipped
fused `Both` (`row_sort_indices`): the SAME `(key, original-index)`
pair-sort, the writeback **capped** to the first `k_out` ranks under
`order`. `topk` = `Desc` (the `k_out` largest, descending, NaN-first);
`bottomk` = `Asc` (the `k_out` smallest, ascending, NaN-last) — both
torch.topk `largest=…, sorted=True`. Two outputs in one launch:
`out_val[batch, k_out]` (raw-bit value permutation, dtype-preserving) and
`out_idx[batch, k_out]` (`I32` original positions).

Representation: an **orthogonal `SortLimit {Full, TopK}`** field on
`Access::RowSort` / `Schedule::RowSort` (NOT a new access, NOT a widened
`SortOut`). `Full` reproduces today's `row_sort`/`row_argsort`/
`row_sort_indices` **byte-for-byte** (verified: all 60 pre-existing
generated `.cu` cells diff `mismatches=0` vs the pre-change tree). `TopK`
adds one `long long k_out` launch scalar — the **exact `(n_out, k_in,
k_out)` 3-scalar ABI your Window/pool emitter already ships** — plus a
guarded store (`if (r < k_out)` base / `p < k_out` bitonic writeback).
`k_out ≤ k_in` is a **runtime launch precondition**, on-device-validated
by `initcheck` (the structure key carries no numeric extents, so it cannot
be a plan assert — same trust tier as the bitonic `k ≤ 1024`); `k_in ≤
2³¹−1` for the `I32` index.

Torch-faithful (NaN-greatest + stable index tie-break), so it **diverges
from the bespoke `baracuda_topk.cuh`** (no NaN branch, `STABLE=0`) on
NaN/tie rows — this is the more-faithful behavior, and the bespoke
cross-check runs on distinct-key, NaN-free rows only (where it is
bit-exact on VALUES and INDICES, both directions). Device-validated on
sm_89 across {f32,f64,i32,i64,f16,bf16,f32-strict} × {topk,bottomk} ×
{base,bitonic}, the `k_out` boundary sweep {1, 2, k_in/2, k_in−1, k_in},
base ≡ bitonic, all four compute-sanitizers 0 errors — see
`crates/baracuda-kernelgen/ondevice/README.md`, the TOPK/BOTTOMK
sub-section.

## The advert story today (honest miss — AOT-only, correct posture)

- **No contract, no pattern, no JIT region** — ZERO `contract.rs` /
  `pattern.rs` / `jit.rs` change. A topk is still `Access::RowSort`
  (non-Elementwise), so `derive_pattern` returns
  `PatternError::NotElementwise` before any body walk and `contract()`
  returns `None` — the SAME withhold path as the shipped `row_sort` /
  `row_argsort` / `row_sort_indices` family. `n_outputs()` stays
  body-derived `= 1` for `Both`. Pinned by
  `cuda::sort_tests::topk_is_an_honest_miss_no_contract`.
- **`baracuda-kernels-types` UNTOUCHED** — no key field, **no
  `STRUCTURE_KEY_VERSION` bump**. The `_topk` suffix rides the entry-point
  symbol (`baracuda_gen_{op}_{dtag}_rowsort_{ord}_stable_both_topk[_bitonic]`),
  not the structure-key token; `k_out`'s divisibility enters only as the
  OUT operand's existing `inner_div` bucket (correct classing, not
  extent-keying).

## Why AOT-only is CORRECT here (the documented Fuel gap)

Fuel has **no first-class `top_k` primitive / `Op::TopKRoute`** — a
documented, still-open frontier gap: the MoE routers do top-k **densely**
(compute all N expert FFNs, gate by the full softmax) as a **~15–32×
over-compute workaround** (`fuel-core/src/lazy_qwen2_moe.rs`,
`lazy_mixtral.rs`, `fuel/docs/frontier-architecture-gaps.md` — "`Op::TopKRoute`:
Absent (dense today)"). `sort_last_dim` decomposes into `arg_sort_last_dim`
+ `gather`, and ArgSort is FKC describe-only / no-dispatch-OpKind. So topk
has **no advertisable FKC shape** — it withholds exactly like the sort
family, and there is nothing for you to import today.

**This kernel is the SELECTION half of sparse MoE.** Fixed-`k` topk
(`[batch, k_out]`, `k_out` a launch scalar) is data-dependent VALUES over
FIXED shapes — it already works in your lazy DAG; it is NOT the
data-DETERMINED-shapes keystone (per-expert token counts / `NonZeroIndices`).
So there is no shape blocker for the AOT kernel; topk provides the routing
SELECTION, while the sparse DISPATCH half still needs your unbuilt
data-determined-shapes work.

## What a future `top_k` call path would take (the ask, when you want it)

To make Fuel *call* this kernel (replacing the dense-routing
over-compute), you would need:

1. **A `top_k` lazy primitive / `Op::TopKRoute`** (the documented gap) with
   a fixed-`k` shape rule `[batch, k_out]` and an `order`/`largest`
   attribute — and its **two-output binding** (`(values, indices)`, or the
   `return.bundle` envelope), the same two-output shape the fused `Both`
   already produces.
2. **The `(n_out, k_in, k_out)` launch-scalar wiring** — identical to what
   your Window/pool path already passes; `k_out` is the OUT operand's inner
   extent, not a policy scalar.

The v1 AOT kernel needs none of this to run correctly (proven on device),
and it **directly unblocks the dense-routing workaround** the moment you
grow the primitive. When the topk call path is worth sequencing on your
side, reply through the channel and we wire the two-output binding together
as its own increment — propose-first, per convention.
