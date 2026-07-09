# Baracuda ask — HETERO multi-output shipped (fused dropout: F32 value + U8 keep-mask, bit-exact); the `return.bundle` + `Op::BernoulliMask` prerequisites, on radar (2026-07-09)

**No action needed now.** This is a propose-first heads-up in the alpha.76
landing-doc "radar item" class: a new kernel capability exists that is
deliberately NOT advertised to Fuel, and we are recording exactly what an honest
"Fuel calls the fused dropout kernel" path would take — before anyone wires it.
The v1 AOT kernel needs none of it and ships bit-exact today.

## What shipped (Baracuda, post-ramp increment on `feat/kernel-specialization`)

The **first hetero (mixed-dtype) multi-output** kernel: one fused pass writes an
**F32 value** output AND a **U8 keep-mask** output, each stored through its own
dtype, from a shared body-DAG. It rides an additive per-output dtype channel
(`OpDef::extra_out_dtypes` + `OpDef::elementwise_multi_hetero`) — no
`StructureKey` change, **no `STRUCTURE_KEY_VERSION` bump, no kernels-types
change**. The v1 vehicle `dropout_fw` reuses the just-shipped `Select`/`Cmp*`:

- output 0 (value, F32): `y = x * (rand < keep_prob ? scale : 0)` — a genuine
  multiply of `x` by a **selected multiplier** (`Select(cond, scale, 0.0)`),
  bit-identical to the bespoke `dropout_fw` single `x*(m ? scale : 0)` multiply;
- output 1 (mask, U8): `mask = (rand < keep_prob)` — the **same** `Cmp*` node,
  hoisted ONCE by cross-body CSE and consumed by both outputs (output 0 as the
  `Select` cond in the compute dtype, output 1 as the `(unsigned char)` store).

The per-output store conversion (`Cmp*` `0.0/1.0 -> U8`, exact) is applied at the
STORE SITE, never baked into the shared DAG node — so output 0's value is never
corrupted by the mask's cast. Noise is a **host-filled uniform-F32 `rand`
operand** (one sample per cell — matching your own host-sampled-mask convention in
`lazy_nn_dropout.rs`), and `keep_prob = 1-p` / `scale = 1/(1-p)` are host-computed
F32 params passed by value: so dropout is a pure elementwise map over `(x, rand)`
with two scalar params, no in-kernel RNG. Whole-buffer memcmp vs the bespoke
`baracuda_kernels_dropout_f32_run` is `bit_diff(y) == 0` AND `bit_diff(mask) == 0`
across the p x shape matrix on sm_89 (probe-seeded, shared `rand`) — see
`crates/baracuda-kernelgen/ondevice/README.md`, `dropout_validate.cu` section.

## The advert story today (honest miss — AOT-ONLY, no code change)

`contract()` returns `None` for **every** `op.n_outputs() > 1` op structurally
(`contract.rs:245`), BEFORE `derive_pattern` — so hetero rides the existing
uniform-multi-output honest miss verbatim, with **zero** `contract.rs` /
`pattern.rs` / `jit.rs` change. The three documented reasons stand unchanged: no
Fuel dual `OpKind` for an elementwise multi-output; `PatternNode` is
single-rooted (a forest of N distinct output roots is inexpressible, and
`derive_pattern` reads only `op.body`); Fuel's only multi-output ABI is a single
packed `return.bundle`, not N distinct buffers. Fuel additionally has no
`Dropout`/`BernoulliMask` `OpTag`, so `synth_op` can never synthesize a dropout
region — an honest miss by absence. **Hetero multi-output is AOT-only forever
given current Fuel types**, the same posture the uniform multi-output increment
and SelectiveScan/SsdChunkScan already occupy.

## What "Fuel calls the fused dropout kernel" would take (the ask, when you want it)

Two Fuel-side prerequisites — recorded so the increment that wires this starts
from here. Neither is a Baracuda blocker:

1. **A graph-level dropout op** — an `Op::BernoulliMask { p, seed }`, or a
   value+keep-mask dropout op, exists only as a NAMED FUTURE in your
   `lazy_nn_dropout.rs` (~36, 47-52), not as a shipped op. Without it there is no
   graph node whose lowering could dispatch the fused kernel; today dropout is a
   host-sampled mask + an elementwise multiply, which is why the AOT kernel is the
   right and only home for the fusion.
2. **Register Baracuda's FKC as a hetero `return.bundle` FusedOp** — a fused
   dropout would map its two distinct outputs into one `FusedOp.output_views`
   bundle. Baracuda would be Fuel's **first genuinely mixed-dtype bundle
   producer** (both shipped bundle ops are uniform, per `selective_scan.rs`
   ~167-183). The good news: the per-slot hetero dtypes live **off-key** in
   `output_views` — only the primary slot's dtype reaches the dispatch key
   (`fuel-dispatch/src/fkc/lower.rs` ~671-679) — so even that future advert needs
   **no Fuel key-version bump**, exactly mirroring how Baracuda's per-output dtype
   rides `OpDef` and the entry-point symbol off-key on our side.

## Baracuda-side follow-ups recorded (not this increment)

- **F64 dropout** — bespoke's `Scale` is `double`, but kernelgen `Param` is
  F32-only (`plan.rs`), and Fuel dropout is F32-only anyway
  (`lazy_nn_dropout.rs` ~34). An f64-param channel is a later increment.
- **Sort/reduction hetero** (fused argsort values+indices, max+argmax) — a
  materially larger, orthogonal lift (`Access::RowSort`/`Reduction`, not
  `Elementwise`), deferred with rejection gates.
- **Non-U8 side-output** (an I32 count beside a float, or any wider-than-compute
  integer output) — the compute-dtype -> out exact-store invariant only holds for
  `Cmp* -> U8`, so v1 rejects it (G1). A widened-integer side-output is a later
  increment.

Nothing above blocks you; nothing above is wired speculatively. When a graph-level
dropout op + a hetero `return.bundle` FusedOp registration become worth it on your
side, reply through the channel and we sequence the fused-dropout call path as its
own increment.
