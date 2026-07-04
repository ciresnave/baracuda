# Baracuda ask — per-operand dtype for mixed-dtype op keying/dispatch

**From:** Baracuda kernelgen (the IR-expansion ramp — increments 0a…#4 GATHER
landed AOT; commit `41c3010`).
**Status:** propose-first. A keying/dispatch question that has accumulated across
three increments and now blocks **keyed dispatch of the whole gather family**.
Nothing here is urgent for Fuel today (the affected kernels run AOT); the ask is
to pin the dispatch model before Baracuda wires the contracts so we don't ship a
version bump you'd rather not take.

## The problem in one sentence

Baracuda's `StructureKey` carries a **single top-level dtype** (operand 0;
`structure_key.rs`: *"v1 assumes a uniform operand dtype"*) — there is **no
per-operand dtype field** — so a genuinely **mixed-dtype** op cannot be keyed
honestly, and we now have three of them.

## The three mixed-dtype op classes (shipped or shipping)

| Increment | Op class | Mixed-dtype shape | How the non-primary dtype is carried TODAY |
|---|---|---|---|
| 0b | comparison → mask | data `T` → **`U8`** output | `out_dtype` on the op + the FKC `return.dtype_rule: fixed(U8)` (your own §5.1 spelling — you accepted this) |
| 0e | hetero-output reduce | data `T` → **`U8`/`I64`** output | same — `return.dtype_rule` |
| **#4** | **gather / index_select / embedding** | data `T` + **`i32`/`i64` INDEX input** + `T` out | **nothing keyed** — the index dtype rides the op + the `entry_point` symbol (`gather_f32_i32` vs `_i64`) only |

The output-dtype cases (0b/0e) were solvable **without touching the key** because
the `return.dtype_rule` block already carries the output dtype, and you dispatch
on it. **Gather is the new gap: the differing dtype is an INPUT** (the index
operand), and the `accept:` block's per-input `dtype:` is the natural home — but
today Baracuda fills every input's `dtype:` with the uniform `key.dtype`, and it
is unclear whether your matcher treats that per-input `dtype:` as **load-bearing
admissibility** or as a human-readable gloss subordinate to the `structure_key`
token.

Concretely: an `i32`-index and an `i64`-index gather **can derive the same
`structure_key` token** (the index dtype's byte size only *incidentally* leaks
into that operand's `vec_width` — and even that collapses to equal for the 1-D
index of `index_select`/`embedding`). So the token is not a reliable index-dtype
discriminator, and a Baracuda contract that let you bind an `i32`-index kernel to
an `i64`-index call would be the wrong-bind our honest-miss rule forbids. Hence
**#4 emits no contract at all today** — gather is AOT-only until this is settled.

## What we already know from your side

- `fuel-dispatch/src/fkc/cpu_link.rs` keys gather/index_select as a **per-operand
  dtype tuple `[T, U32, T]`** — the `indices` operand is a **fixed U32 slot**,
  `out: passthrough(source)`. So Fuel *already* thinks in per-operand dtypes for
  gather admissibility, and **expects U32 indices**.
- (Correcting an earlier note of ours: the `UnsupportedGatherKind` in
  `dlpack-extension.md` V18 is the **paged-residency** validator, not the dense
  gather index operand — the U32 pinning is the `cpu_link.rs` `[T,U32,T]` fact
  above, which is the one that matters here.)

## The two models — which do you want?

**Model A — per-input dtype in the FKC `accept` block (no key/version change).**
Baracuda fills the `accept.inputs[i].dtype` with the **actual per-operand dtype**
(data `T`, index `i32`/`i64`), and your matcher treats that list as
**load-bearing admissibility** alongside the `structure_key` token (the token
stays the coarse layout/size predicate; the accept block is the fine dtype
predicate). No `STRUCTURE_KEY_VERSION` bump; reuses a schema slot that already
exists; extends exactly the mechanism 0b/0e used for the *output* dtype to
*input* dtypes. **This is our recommendation** if your dispatcher can key on the
accept block's per-input dtype.

**Model B — per-operand dtype in the `structure_key` token itself.** Add a
per-operand dtype field to the token (a `STRUCTURE_KEY_VERSION` bump, wire-
visible, both sides re-derive). Cleaner in that the token becomes a total
admissibility predicate again, but it's a coordinated schema evolution and it
changes every existing token's byte layout.

## Questions

1. **Model A or B?** Does your FKC matcher already treat `accept.inputs[i].dtype`
   as load-bearing (Model A works with no wire change), or is the
   `structure_key` token the sole admissibility predicate (⇒ Model B, a version
   bump we'd coordinate)?
2. **Index dtype reconciliation.** Your gather key is `[T, U32, T]` (U32
   indices); Baracuda currently emits `i32`/`i64` (matching the bespoke
   `baracuda-kernels-sys` gather's templated `IndexT`, which is i32/i64). Do you
   want Baracuda to emit **`u32`-index** gather kernels to match your slot, keep
   `i32`/`i64` and have you widen the slot, or support the set
   `{u32, i32, i64}`? (Baracuda can emit any of them — it's an entry-point +
   index-load-type choice.)
3. **OOB contract.** Bespoke/generated gather **skips** OOB indices (leaves the
   cell unwritten) and embedding **zero-fills**; torch/your gather assumes
   in-bounds. Do you want the FKC contract to advertise an **OOB policy** field
   (`skip`/`zero_fill`/`clamp`) so the caller's expectation is explicit, or is
   in-bounds-only the contract and OOB is caller UB?
4. If Model B: any constraints on **where** per-operand dtype lands in the token
   (a new field vs. widening each operand's sub-key) so telemetry/back-compat
   stay sane on your side?

## Not blocking

Gather + the whole family run AOT today; this only gates **keyed dispatch/seam**
for them. We'll wire the contracts the moment you pick a model. If it's Model A,
that's a Baracuda-only change (fill the accept block honestly); if Model B, we
coordinate the version bump.
