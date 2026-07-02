# 09 — f16/bf16 half2 packed-SIMD vectorization — implementation brief

> Scope owner: baracuda-kernelgen. Status as of 2026-06-30 (branch
> `feat/kernel-specialization`, crates.io alpha.72). Read the four grounding files
> before starting: `crates/baracuda-kernelgen/src/cuda.rs`,
> `crates/baracuda-kernelgen/src/plan.rs`,
> `crates/baracuda-kernels-types/src/structure_key.rs`,
> `crates/baracuda-kernelgen/src/ir.rs`.

---

## 1. Objective

Give the CUDA emitter a **packed-SIMD path** for f16/bf16 elementwise (and,
later, reduction) kernels so that two half-precision lanes compute in one
`half2` / `nv_bfloat162` op instead of the current scalar-per-element
convert→f32→convert lowering. Today the structure key already classifies a
contiguous 16-byte-aligned f16/bf16 operand as `VecWidth::V8`
(`structure_key.rs:522-544`), and `build_plan` faithfully produces
`Schedule::Vectorized { width: 8 }` (`plan.rs:100-116`) — but the CUDA backend's
`vector_type` returns `None` for those dtypes (`cuda.rs:78-85`), so every f16/bf16
"vectorized" cell silently **falls back to `emit_scalar`** (`cuda.rs:37-44`),
computing one element at a time in float. This is the single largest unclaimed
memory-bound throughput win in the generator: half-precision workloads (LLM
activations, norms, attention epilogues — Fuel's primary traffic) are exactly
the memory-bound regime where packing two FMA-class lanes into one instruction
lifts arithmetic intensity without changing the byte budget. It is foundational
because it closes the last hole in the schedule×dtype coverage table
(`docs/design/kernel-specialization.md:405-411`) and it makes the V8 axis the
structure key already spends bits on actually *pay off* rather than being a
no-op discriminator.

---

## 2. Status & blockers

**Baracuda-unblocked.** This is a pure emitter refinement inside
`baracuda-kernelgen`. No Fuel answer is required, no cross-repo contract change is
required, and the seam (`jit.rs`) reaches it for free because JIT-synthesized
elementwise ops already flow through `build_plan` → `Schedule::Vectorized`. The
win is claimable end-to-end (AOT catalog + live §5 seam) the moment the backend
grows a packed lowering.

**Nothing is Fuel-blocked here.** The FKC contract's `dtypes: [F16]` / `[BF16]`,
`accept.structure_key`, `precision`, and `determinism` blocks are all already
emitted correctly for f16/bf16 today (`contract.rs`); this change alters only the
*generated CUDA source* behind a fixed `entry_point`, which changes the
`kernel_revision_hash` (FNV-1a over source, `contract.rs:409`) but nothing else on
the wire. The contract stays honest by construction (§5 below).

**Design-open (one decision, self-contained).** The *packing width*: the key
buckets f16/bf16 at V8 (eight halves = one 128-bit `ld.128`), but the packed math
unit is `half2` (two halves). The kernel must load 128-bit *and* compute in four
`half2` ops per 8-lane vector. The two viable emit shapes (a `struct` of four
`half2` fields vs. a raw 128-bit load reinterpreted as `half2[4]`) are both
nvrtc-headerless-safe; §5 picks the struct form and §10 records the trade. This is
an internal decision, not a blocker.

---

## 3. Dependencies & sequencing

**Must land before this: nothing hard-required.** The elementwise packed path is
independent — it only touches `vector_type` + a new `emit_vectorized_packed`
sibling in `cuda.rs`, plus the plan-side width mapping. It can ship against
today's IR and today's structure key.

**Coordinate / sequence with:**
- **01 (layout/shape nodes)** and **03 (strided/multi-axis reductions)** — the
  house note flags these as the emit paths this change "touches." Concretely:
  today only the *contiguous forward-unit-stride* case keys as V8
  (`classify_vec_width` bails to `Scalar` on any broadcast or non-unit inner
  stride, `structure_key.rs:527`). So the packed path is confined to
  `emit_vectorized` and never fires inside `emit_strided`
  (`cuda.rs:187-249`) — **no conflict today**. When 01/03 later widen
  vectorization to inner-contiguous strided runs, the packed lane-splat helper
  built here must be reused there rather than re-derived. Build the lane
  accessor as a *shared helper* (§5) so 01/03 inherit it.
- **04 (integer accumulation)** — orthogonal; the packed path is f16/bf16-only.
- **06 (fused residual-add LayerNorm)** and the RowReduce family — the reduction
  and row-reduce emitters (`emit_reduction` `cuda.rs:265`, `emit_row_reduce`
  `cuda.rs:397`) deliberately accumulate in **float/double** and only pack the
  *load/store* narrowing. A packed *reduction* fold (`__hadd2` tree, packed
  epilogue) is a **follow-up stage** of this item (§5, Stage 2), explicitly
  sequenced *after* the elementwise stage lands and *after* 06's catalog work, to
  avoid perturbing the load-bearing determinism guarantees of the row-reduce
  block tree (one-block-per-row, no atomicAdd).

**This enables downstream:**
- **07 (per-arch dispatch + bench-gate)** — gives 07 a real half-precision
  bandwidth delta to gate on (the current f16 "vectorized" cell is
  indistinguishable from scalar, so 07 has nothing to measure there).
- **08 (telemetry variant-selection)** — a genuine packed vs. scalar-fallback f16
  variant pair becomes a selectable top-K candidate.
- **09 itself is terminal** in the dependency graph (an emitter refinement, not a
  keystone), so nothing else *blocks on* it landing.

---

## 4. Current code — what exists today

### 4.1 The structure key already asks for V8 (the width is present, unused)

`crates/baracuda-kernels-types/src/structure_key.rs:522-544` — `classify_vec_width`
returns `V8` for a contiguous, 16-byte-aligned, unit-inner-stride f16/bf16
operand whose inner extent is `% 8 == 0`:

```rust
for &v in &[8u64, 4, 2] {
    let vbytes = v * dsz;                 // dsz = 2 for F16/Bf16
    if vbytes <= 16 && align % vbytes == 0 && ext % v == 0 {
        return match v { 8 => VecWidth::V8, 4 => VecWidth::V4, _ => VecWidth::V2 };
    }
}
```

The `f16_contiguous_vectorizes_to_v8` test (`structure_key.rs:947-951`) pins this:
a `[64,128]` f16 operand keys `VecWidth::V8`. Reversed / strided / broadcast
operands correctly downgrade to `Scalar` (`structure_key.rs:527`,
`negative_stride_is_flipped` / `transposed_view_is_strided` tests).

### 4.2 `build_plan` propagates V8 as `width: 8`

`crates/baracuda-kernelgen/src/plan.rs:100-116` — `vec_width_elems(V8) == 8`
(`plan.rs:296-303`); for an all-contiguous elementwise op the plan is
`Schedule::Vectorized { width: min_width }` where `min_width` is the narrowest
operand width. For a uniform-f16 op that is `8`.

### 4.3 The CUDA backend has no packed vector type → silent scalar fallback

`crates/baracuda-kernelgen/src/cuda.rs:37-50`:

```rust
Schedule::Vectorized { width } => match vector_type(plan.dtype, width) {
    Some((vty, lanes)) => emit_vectorized(plan, vty, lanes),
    // dtype has no packed vector path yet (e.g. f16 V8): fall back to
    // scalar — still correct, and still gets the narrower-dtype
    // bandwidth win.
    None => emit_scalar(plan, ctype),
},
```

`vector_type` (`cuda.rs:78-85`) only knows `float4`/`float2`/`double2`:

```rust
fn vector_type(dt: ElementKind, width: u32) -> Option<(&'static str, &'static [&'static str])> {
    match (dt, width) {
        (ElementKind::F32 | ElementKind::F32Strict, 4) => Some(("float4", &["x", "y", "z", "w"])),
        (ElementKind::F32 | ElementKind::F32Strict, 2) => Some(("float2", &["x", "y"])),
        (ElementKind::F64, 2) => Some(("double2", &["x", "y"])),
        _ => None,                 // <-- f16/bf16 V8 lands here → scalar fallback
    }
}
```

The `f16_falls_back_to_scalar_with_fp16_header` test (`cuda.rs:852-860`) *asserts*
the current fallback: `baracuda_gen_add_f16_scalar`, `out[i] = (in0[i] + in1[i])`.
That test must be updated (§7).

### 4.4 `emit_vectorized` — the shape to mirror

`cuda.rs:122-157`. For each lane name it splats the accessor
`v{idx}.{lane}` into `lower_expr`, computing `vo.{lane} = <body>` per lane, then
stores `out[i] = vo`. The math is delegated to the shared `cuda_unary` /
`cuda_binary` closures. **This is the template** — the packed path is the same
loop with a `half2`-typed vector, `half2` lane pairs, and packed intrinsics for
the ops that have them.

### 4.5 The f16/bf16 compute-in-float lowering (what packing replaces)

`cuda.rs:699-714` (`cuda_unary`) and `cuda.rs:747-765` (`cuda_binary`) wrap the
f32 spelling in `__half2float(...)` / `__float2half(...)` (bf16 analogues). Every
op is *correct* today; it is just scalar and float-round-tripped. The `f16` header
`#include <cuda_fp16.h>` / bf16 `#include <cuda_bf16.h>` is already emitted by
`extra_include` (`cuda.rs:67-73`) — these headers bundle the `half2` types and
intrinsics under nvrtc, so no new include is needed.

### 4.6 The seam reaches the vectorized path

`jit.rs:406-425` `region_to_op` hardcodes `Access::Elementwise`; a synthesized
op's plan therefore routes through the same `build_plan` → `Schedule::Vectorized`
as an AOT op. So a JIT'd f16 elementwise fusion **also** benefits with zero seam
changes — but that also means the packed lowering must be as trustworthy as the
scalar one (it crosses the JIT trust boundary; §8).

---

## 5. Design / delta

### 5.1 The packing model (elementwise, Stage 1)

A V8 f16 vector is **one 128-bit load = eight halves = four `half2` lanes**. The
emitted kernel:

1. Loads 8 halves per operand per iteration via a 128-bit vector type.
2. Views them as four `half2` (bf16: `nv_bfloat162`) lane-pairs.
3. Computes the body **once per half2 lane-pair** using packed intrinsics where
   they exist, falling back to a per-half `__half2float` scalarization of the
   pair for ops with no packed intrinsic.
4. Stores 8 halves per output via the same 128-bit vector type.

**Vector type choice (design decision, §2/§10):** emit a small generated struct
that aliases a 128-bit load, avoiding reliance on a specific CUDA vector typedef
that may differ nvcc vs. nvrtc:

```cpp
// f16 example, generated inline (bf16 swaps __half2->__nv_bfloat162)
struct __align__(16) h2x4 { __half2 a, b, c, d; };
```

`struct __align__(16)` is valid under both nvcc and nvrtc (it is core C++ + the
`__align__` built-in, no header). A single `h2x4 v = *reinterpret_cast<const
h2x4*>(&in0[i8])` is a coalesced 128-bit load. The four fields `a,b,c,d` are the
four `half2` lanes — the direct analogue of `float4`'s `x,y,z,w`. (Alternative:
`int4` load + `__halves2half2` unpack — rejected as more error-prone and no
faster; recorded in §10.)

### 5.2 Two lowering tiers per op

Define, alongside `cuda_unary`/`cuda_binary`, a **packed** spelling keyed on
`(op, dtype)`:

- **Tier A — native packed intrinsic** (fast path). CUDA provides packed forms
  for a subset:
  - infix: `__hadd2`, `__hsub2`, `__hmul2`, `__h2div` (f16); `nv_bfloat162`
    analogues.
  - unary with packed intrinsics: `__habs2`, `h2exp`, `h2log`, `h2sqrt`,
    `h2rsqrt`, `h2sin`, `h2cos`, `__hneg2`, and (via compare-select) relu.
- **Tier B — scalarize the pair** (fallback, always correct). For any op with no
  packed intrinsic (`erff`/`Gelu`/`Silu`/`Sigmoid`/`Pow`/`Rem`/`Tanh` composites,
  etc.), decompose the `half2` into its `.x`/`.y` halves with `__low2half` /
  `__high2half`, run the **existing** `cuda_unary`/`cuda_binary` float lowering on
  each, and repack with `__halves2half2`. This reuses the verified float path
  verbatim, so Tier B is numerically **identical to today's scalar kernel** — the
  safety floor.

Crucially: **the choice of tier per op is a codegen decision, invisible to the
contract.** A body that is all Tier-A ops gets the full 2× packing; a body with
one Tier-B op still packs its loads/stores (bandwidth win) and only scalarizes the
compute. Both are correct.

> Determinism note: `__hadd2` etc. are IEEE-correctly-rounded on the packed lanes,
> bit-identical to the scalar `__hadd`. The transcendental `h2*` intrinsics are
> *approximate* and **may differ in the last bit** from the scalar `__float2half(expf(__half2float(x)))`
> path. This matters for the precision contract (§8) and for the differential
> oracle (§7): Tier-A transcendentals must diff against the **scalar-fallback f16
> kernel**, not against an f32 oracle, or against a widened tolerance. To keep the
> `determinism: bitwise` guarantee honest for transcendentals, **Stage 1 uses
> Tier-B (scalarized) for every approximate op** and reserves Tier-A packed
> transcendentals for a measured, separately-validated follow-up. Tier-A is used
> in Stage 1 only for the correctly-rounded ops (`__hadd2/__hsub2/__hmul2/__h2div`,
> `__habs2`, `__hneg2`, relu-select, sqr). This keeps `determinism: bitwise`
> literally true.

### 5.3 Code sketch — the emitter

New function `emit_vectorized_packed` in `cuda.rs`, selected when the dtype is
f16/bf16 and width ≥ 2:

```rust
// cuda.rs — new arm in `vector_type` OR a dedicated dispatch in `lower`.
// Preferred: a dedicated packed dispatch so the struct + intrinsics stay local.
match plan.schedule {
    Schedule::Vectorized { width } => {
        if let Some((vty, lanes)) = vector_type(plan.dtype, width) {
            emit_vectorized(plan, vty, lanes)                 // f32/f64 (unchanged)
        } else if let Some(pk) = packed_kind(plan.dtype, width) {
            emit_vectorized_packed(plan, pk)                  // NEW: f16/bf16 V8/V4/V2
        } else {
            emit_scalar(plan, ctype)                          // last resort
        }
    }
    ...
}
```

`packed_kind(dtype, width)` yields the pair-type name (`__half2` /
`__nv_bfloat162`), the number of `half2` lanes (`width/2` — 4 for V8, 2 for V4, 1
for V2), and the convert-in/out spellings. `emit_vectorized_packed` mirrors
`emit_vectorized` (`cuda.rs:122-157`) but:

- signature uses the 128-bit struct pointer (`const h2x4* __restrict__ in{i}`),
- per lane-pair `l` it emits `vo.<field_l> = <packed-or-scalarized body>`,
- the leaf accessor returns `v{idx}.<field_l>` (a `half2`), and the
  `unary`/`binary` closures dispatch Tier A vs. Tier B on `(op, dtype)`.

The **lane accessor closure is the shared helper** 01/03 will reuse: a
`fn pack_lane(field) -> String` plus the Tier-A/Tier-B `half2` unary/binary
spellers, factored so a future strided-inner-contiguous packer calls the same
spellers over a different load path.

### 5.4 Plan-side width mapping (small, may be a no-op)

`build_plan` already emits `width: 8` for f16 V8. The only plan question is
whether to *also* let f16 pick V4/V2 when V8 alignment fails but V4 does — the key
already handles this (`classify_vec_width` tries 8→4→2). So `packed_kind` must
accept `width ∈ {2,4,8}`. **No `plan.rs` change is required** beyond confirming
`vec_width_elems` (`plan.rs:296-303`) maps V4→4, V2→2 (it does). Add a plan-level
unit test that a uniform-f16 contiguous op yields `Vectorized{8}` (guards against
a future regression that would re-route f16 to Scalar).

### 5.5 StructureKey facts consumed

- `key.operands[k].vec_width` (V8/V4/V2) — already the width source via
  `build_plan`.
- `key.operands[k].contig == Contig` and empty `bcast` — guaranteed by the fact
  that `classify_vec_width` only returns a packed width for the forward-unit-stride
  contiguous case (`structure_key.rs:527`). The packed emitter may therefore assume
  a dense 128-bit-loadable operand — **the same precondition `emit_vectorized`
  already relies on**. No new StructureKey axis is needed; the alignment fact
  (`align_bytes`, folded into `vec_width`) already gates the 128-bit load's
  legality.

### 5.6 FKC / contract implications

- `entry_point` symbol name changes: today the fallback emits
  `baracuda_gen_add_f16_scalar`; the packed path should emit a distinct name
  (`baracuda_gen_add_f16_co_v8` — mirroring the f32 `_co_v4` convention,
  `cuda.rs:122-128`). This is a **new symbol**, so link registry + AOT catalog
  entries change (§6).
- `kernel_revision_hash` changes (new source) — expected and correct.
- `required_align` (`contract.rs:289-297`) already computes `(8*dsz).min(16) = 16`
  for a V8 f16 cell — so the packed kernel's 16-byte-alignment requirement is
  *already advertised* in `caps.alignment_bytes`. Good: the contract was honest
  about needing 16-byte alignment even while the kernel secretly ran scalar. No
  `contract.rs` change required for alignment.
- `precision` / `determinism`: unchanged **iff** Stage 1 uses Tier-B for all
  approximate ops (§5.2). The `precision_of` bound (`contract.rs:256`) is dtype-
  agnostic (counts ULP over the *body*), so it stays correct. Verify no drift (§8).

---

## 6. Implementation steps

Ordered; each step names the file it edits.

1. **IR — none.** No new `ScalarExpr`/`UnaryOp`/`BinaryOp` variants; packing is a
   pure lowering concern. (`ir.rs` untouched — confirm and note in the PR.)
2. **`cuda.rs` — packed vector type.** Add `packed_kind(dtype, width) ->
   Option<PackedKind>` returning the pair-type name, lane count (`width/2`), field
   names (`a,b,c,d` capped to lane count), and the f16-vs-bf16 convert spellings.
3. **`cuda.rs` — the emitter.** Add `emit_vectorized_packed(plan, pk)` modelled on
   `emit_vectorized` (`cuda.rs:122-157`): 128-bit struct load per operand, per
   lane-pair body, 128-bit store. Emit the `struct __align__(16) h2x4 {…};`
   typedef in the kernel preamble (after the include, before `extern "C"`).
4. **`cuda.rs` — the two-tier spellers.** Add `cuda_unary_packed` /
   `cuda_binary_packed` that return Tier-A packed intrinsics for the
   correctly-rounded op set and **delegate to a scalarize-the-pair helper** (built
   on the existing `cuda_unary`/`cuda_binary`) for everything else. Wire them into
   the `Lowering` closures inside `emit_vectorized_packed`.
5. **`cuda.rs` — dispatch.** Extend the `Schedule::Vectorized` arm (`cuda.rs:37-44`)
   to try `packed_kind` before the scalar fallback (§5.3).
6. **`plan.rs` — guard test only.** Add a unit test asserting a uniform-f16
   contiguous elementwise op plans `Vectorized{8}` (no code change expected;
   `plan.rs`).
7. **`pattern.rs` / `jit.rs` — none.** Pattern derivation and the seam are dtype-
   agnostic; the packed path is reached transparently. Confirm with a seam test
   (§7) — no source edits.
8. **`contract.rs` — none (verify).** Confirm `required_align`, `precision_of`,
   and `determinism` are unchanged for a f16 V8 cell; add an assertion test that
   the packed cell's contract still says `alignment_bytes: 16` and
   `determinism: bitwise` (`contract.rs`).
9. **FFI / build wiring.** Regenerate the AOT catalog so the new
   `baracuda_gen_*_f16_co_v8` / `*_bf16_co_v8` symbols are compiled and registered
   in the link registry (`link.rs` / the `bin/kernelgen.rs` catalog driver). Verify
   the emitted `.cu` compiles under the AOT nvcc build.
10. **Catalog / docs.** Update the schedule×dtype coverage table
    (`docs/design/kernel-specialization.md:405-411`) to record f16/bf16 as
    packed-V8 (drop the "fall back to Scalar — follow-up" note). Update
    `OP-MATRIX.md` if it tracks kernelgen coverage. Update the stale status note
    per MEMORY (the doc still lists shipped features as not-emittable).

---

## 7. Test & on-device validation plan

House discipline: **nvrtc headerless compile + nvcc numeric on sm_89 (RTX 4070)**
is mandatory for this kernel change.

### 7.1 Unit tests (kernelgen, host-side, `cargo test -p baracuda-kernelgen`)

- **Update** `f16_falls_back_to_scalar_with_fp16_header` (`cuda.rs:852-860`): it
  currently asserts scalar fallback; retarget it to assert the packed emit
  (`baracuda_gen_add_f16_co_v8`, a `__half2` load, `__hadd2` in the body, a 128-bit
  store). Add a bf16 twin (`baracuda_gen_add_bf16_co_v8`, `nv_bfloat162`).
- **New** `f16_v8_packs_four_half2_lanes`: assert four field assignments
  (`vo.a = … ; vo.b = … ; vo.c = … ; vo.d = …`) for a V8 cell, two for V4, one
  for V2.
- **New** `f16_packed_tierB_scalarizes_transcendental`: `silu(a+b)` in f16 emits
  the scalarize-repack form (`__low2half`/`__high2half` + the existing float silu
  + `__halves2half2`), *not* a nonexistent `h2silu`.
- **New** `f16_packed_relu_is_nan_propagating`: the packed relu uses a
  compare-select (or `__hmax2` only if verified NaN-propagating), matching the
  scalar NaN-propagating relu convention (`cuda.rs:654`).
- **New** (plan) `f16_contiguous_plans_vectorized_8` (`plan.rs`).
- **New** (contract) `f16_v8_contract_align16_bitwise` (`contract.rs`).

### 7.2 nvrtc headerless compile (the portability gate)

For each of `{add (Tier-A infix), silu(a+b) (Tier-B), relu(a+b), a*b+c fma,
max(a,b)}` × `{f16, bf16}` × `{V8, V4, V2}`: feed the generated source to
`NvrtcCompiler` (`jit.rs`) with the **same header-light options the seam uses** and
assert it compiles. This is the load-bearing check that the `struct __align__(16)`
typedef + `__half2`/`h2*` intrinsics are available under nvrtc with only
`#include <cuda_fp16.h>` / `<cuda_bf16.h>` (already emitted by `extra_include`).

### 7.3 nvcc numeric on sm_89 (the correctness gate)

Compile each generated kernel with `nvcc -arch=sm_89` and run against a numeric
oracle on the RTX 4070:

- **Oracle:** the **scalar-fallback f16/bf16 kernel** (today's
  `emit_scalar` output) is the differential reference, *not* an f32 kernel. For
  Tier-B ops the packed kernel must match the scalar kernel **bit-for-bit** (it
  runs the identical float lowering). For Tier-A correctly-rounded ops
  (`__hadd2`/`__hmul2`/…) it must **also** match bit-for-bit (IEEE packed add ==
  IEEE scalar add). This makes the §10 "specialized == generic bit-for-bit"
  safety net (`docs/design/kernel-specialization.md:340-343`) literally testable.
- **Inputs:** include denormals, ±0, ±inf, NaN, and the max-finite f16 (65504) to
  probe overflow on the narrowing store; include an **odd inner extent that keys as
  V4/V2 not V8** to exercise the narrower packed widths; include an inner extent
  that is `% 8 != 0` and confirm it correctly keys to Scalar (never reaches the
  packed emitter).
- **Alignment:** confirm a 16-byte-aligned base (the `caps.alignment_bytes: 16`
  the contract promises) and add a deliberately mis-aligned view test asserting the
  key downgrades it to Scalar (so the 128-bit load is never emitted for an
  under-aligned buffer).

### 7.4 compute-sanitizer

The elementwise packed kernel has **no shared memory and no cross-thread
communication** (pure grid-stride map), so `racecheck`/`synccheck` are not
required for Stage 1. Run **`initcheck` + `memcheck`** on one V8 and one V4 case to
catch an out-of-bounds 128-bit load (the classic tail bug when inner extent isn't
a multiple of 8 — which the key *should* prevent, so this is the belt-and-braces
check). If Stage 2 (packed reduction) is built, `racecheck`/`synccheck` become
mandatory per house discipline.

---

## 8. Adversarial-verify checklist

Run the multi-agent find → dedup → skeptic-refute pass after the change. Probe
specifically for:

1. **Tail out-of-bounds 128-bit load.** Does any path emit the packed struct load
   for an inner extent not divisible by the lane count? Trace
   `classify_vec_width` (`structure_key.rs:522`) → `build_plan` → the packed
   emitter and confirm the `% v == 0` guard is the *only* gate and it is honored;
   confirm no `emit_vectorized_packed` grid-stride remainder reads past `nv`.
2. **Silent scalar fallback masquerading as packed.** After the change, does a
   f16 V8 cell that *should* pack ever still hit `emit_scalar`? (A stray `_ =>
   None` left in `vector_type` catching f16 before `packed_kind` runs.) Assert the
   emitted symbol name is `_co_v8`, not `_scalar`.
3. **bf16 intrinsic-name skew.** `nv_bfloat162` intrinsics are named `h2*` in some
   CUDA versions and `hbf16*`/`__hadd2` (overloaded) in others. Verify the exact
   spelling compiles under the *pinned* nvrtc used by the seam, not just nvcc.
4. **Determinism regression via Tier-A transcendentals.** Did any approximate op
   sneak into Tier-A, silently breaking `determinism: bitwise`? Grep the packed
   speller for `h2exp`/`h2sin`/`h2cos`/`h2log` in the Stage-1 set and confirm they
   are **not** used (Stage 1 scalarizes them). Diff the emitted transcendental
   against the scalar kernel and require bit-equality.
5. **NaN handling in packed max/relu.** `__hmax2` is IEEE `maxNum` (NaN-
   *suppressing*) — using it for `BinaryOp::Max` would violate the house
   NaN-propagating convention (`cuda.rs:726`, the deliberate compare-select, not
   `fmaxf`). Confirm the packed Max/relu is a compare-select over the pair, not
   `__hmax2`.
6. **Lane-splat recompute of a compound inner (Tier-B).** Tier-B scalarization
   references each half of a pair; a compound Tuner-B inner (e.g. relu's `x`
   appearing twice) could recompute the whole subexpression per half **and** per
   lane. Confirm this is no worse than the existing scalar path's known
   recompute-TODO (`cuda.rs:638`, 653) and does not, say, re-load global memory
   per half.
7. **Alignment assumption on a sub-view.** The struct load assumes 16-byte
   alignment; confirm a legitimately V8-keyed operand always has `align_bytes % 16
   == 0` (it must, since `classify_vec_width` gated on `align % vbytes == 0` with
   `vbytes = 16`). Confirm no seam path can hand a V8 width with a <16-byte-aligned
   base.
8. **Reintroduced UAF / double-free — N/A here** (no allocation), but confirm the
   catalog/link wiring change (§6) doesn't drop the old `_scalar` symbol that some
   other consumer still links.

---

## 9. Definition of done

- [ ] `emit_vectorized_packed` + `packed_kind` + two-tier spellers land in
      `cuda.rs`; f16/bf16 V8/V4/V2 contiguous elementwise cells emit packed source,
      and the `Schedule::Vectorized` dispatch prefers packed over scalar fallback.
- [ ] All new/updated kernelgen unit tests green (`cargo test -p
      baracuda-kernelgen`), including the retargeted `f16_falls_back_to_scalar_*`
      test and the plan/contract guard tests.
- [ ] **On-device validated on sm_89 (RTX 4070):** every generated packed kernel
      nvrtc-headerless-compiles *and* nvcc-compiles, and diffs **bit-for-bit**
      against the scalar-fallback f16/bf16 oracle across the numeric corpus
      (denormals, ±0/±inf/NaN, max-finite, V8/V4/V2 widths). `initcheck`+`memcheck`
      clean on a V8 and a V4 case.
- [ ] **FKC honest-miss preserved:** the contract for a f16 V8 cell still emits
      `dtypes: [F16]`, `caps.alignment_bytes: 16`, `precision.mode` unchanged, and
      `determinism: bitwise` (verified by test); the `structure_key` in `accept`
      is unchanged; only `entry_point` + `kernel_revision_hash` move.
- [ ] Under-aligned / non-`%8` / strided / broadcast / reversed f16 operands still
      route to Scalar (never the packed load) — asserted.
- [ ] AOT catalog + link registry regenerated with the new `_co_v8` symbols; AOT
      build compiles them.
- [ ] `docs/design/kernel-specialization.md:405-411` coverage table updated (f16/bf16
      = packed V8, no longer "follow-up"); `OP-MATRIX.md` updated if it tracks this;
      stale status notes corrected.
- [ ] Adversarial-verify pass run and its findings resolved (§8), with the
      determinism-of-transcendentals check explicitly recorded.

---

## 10. Open questions / Fuel asks

**No Fuel asks.** This change does not touch the seam contract, the FKC wire
format, or any cross-repo asset. Fuel consumes the same `entry_point` +
`kernel_revision_hash` mechanism it already does.

**Internal decisions to record in the PR (not blockers):**

1. **Vector-type spelling** — `struct __align__(16) { __half2 a,b,c,d; }` (chosen)
   vs. `int4`-load + `__halves2half2`-unpack (rejected: more intrinsics, no faster,
   more nvrtc-version surface). Confirm the struct form's `reinterpret_cast` from
   `const __half*` is a warning-clean coalesced load under the pinned nvcc/nvrtc.
2. **Tier-A transcendental packing** — deferred out of Stage 1 to keep
   `determinism: bitwise` literally true. Open question for a follow-up: is the
   ~2× on `h2exp`/`h2sin` worth relaxing a *subset* of f16 softmax/gelu kernels to
   `determinism: run_to_run` (or a widened ULP)? Needs a bench + a Fuel policy
   check on whether any consumer relies on bitwise f16 transcendental determinism —
   **that** would be the one future Fuel ask, explicitly out of scope here.
3. **Packed reduction (Stage 2)** — `__hadd2` tree fold + packed epilogue for
   `emit_reduction`/`emit_row_reduce`. Deferred and sequenced after 06; must
   preserve the one-block-per-row / no-atomicAdd determinism invariants
   (`cuda.rs:379-396`) and would require `racecheck`/`synccheck`. Flagged here so
   the elementwise lane helpers are built reusable.
4. **Should V4/V2 f16 even pack, or only V8?** V8 is the clear win (one `ld.128`);
   V4 (`ld.64`, two half2) and V2 (`ld.32`, one half2) are smaller wins. Decision:
   support all three via `packed_kind(width)` since the key already distinguishes
   them and the emitter cost is marginal — but bench V4/V2 in 07 to confirm they
   beat scalar-fallback before committing catalog rows.
