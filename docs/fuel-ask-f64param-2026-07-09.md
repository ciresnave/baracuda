# Baracuda ask — F64 scalar-param channel shipped (dtype-aware runtime launch params; F64 dropout bit-exact, AOT-only); the `double` launch-arg marshalling prerequisite, on radar (2026-07-09)

**No action needed now.** This is a propose-first heads-up in the alpha.76
landing-doc "radar item" class (the sibling of the hetero-multi dropout note): a
new kernel capability exists that is deliberately NOT advertised to Fuel, and we
are recording exactly what an honest "Fuel dispatches an f64-param kernel" path
would take — before anyone wires it. The v1 AOT kernel needs none of it and ships
bit-exact today, and Fuel dropout is F32-only (no f64 consumer exists).

## What shipped (Baracuda, increment 12 on `feat/kernel-specialization`)

The **runtime scalar-param launch channel goes dtype-aware.** Until now every
`param(i)` launch argument was hardwired `float p{i}` in the kernel signature
(the two helpers `param_args`/`param_args_multi`). It now follows the SCALAR
COMPUTE dtype `scalar_ctype(plan.dtype)`:

- `"float"` for F32/F32Strict — **byte-for-byte identical** to before, by
  construction (`scalar_ctype(F32) == "float"`), so every shipped f32 param
  kernel signature, launch arg, and contract line is unchanged; and
- `"double"` for F64 — the new capability: `double p{i}` in the signature, the
  surrounding math already double (`binary_f64`/`select_f64`/`const_lit`).

A launch param is **always a scalar arg**, never vectorized: even when operands
are `double2`/`float4` the param stays `double p0` (not `double2 p0`). The
emitter param assert relaxes its allowlist by exactly one variant — `{F32,
F32Strict}` → `{F32, F32Strict, F64}` — staying an explicit allowlist (NOT an open
`is_float`), so `f16`/`bf16` params STILL reject (their correct param ctype would
be `"float"`, not `scalar_ctype(F16)="__half"` — a deferred numerics decision with
no bespoke oracle). **No `StructureKey` change, no `STRUCTURE_KEY_VERSION` bump, no
kernels-types change** — the param width rides the kernel `source` signature
(`double p0`) and is reflected in `revision_hash`; the compute dtype is already a
`StructureKey` token.

The vehicle is **F64 dropout**: a two-char dtype flip on the `dropout_fw` op
(`&[F32]` → `&[F64]`) reuses the entire hetero-multi U8-mask machinery — the value
output is uniform F64, only the mask stays U8 — generating
`baracuda_gen_dropout_f64_mo2_scalar` with `, double p0, double p1)`.

Acceptance (sm_89, CUDA 13.3, `ondevice/dropout_f64_validate.cu`): whole-buffer
`bit_diff(y) == 0 AND bit_diff(mask) == 0` vs a **CPU f64 closed-form oracle**
(PRIMARY — exact IEEE double, the lone `x*mult` multiply is correctly rounded
host↔device, NaN payloads preserved for f64) across the probe-class × keep_prob
matrix, plus the **bespoke `baracuda_kernels_dropout_f64_run`** under a widening
protocol (SECONDARY, see below), plus a byte-identical-f32-dropout regression cell.
compute-sanitizer memcheck/racecheck/synccheck/initcheck: 0 errors.

## The advert story today (honest miss — AOT-ONLY, no Fuel dependency)

- **Dropout is a multi-output honest miss at ANY dtype** — `contract()` returns
  `None` for every `n_outputs > 1` op (`contract.rs`, the `derive_pattern` guard),
  so f64 does not change the dropout Fuel story at all (it rides the same
  uniform/hetero multi-output honest miss).
- **The scalar-param JIT seam stays `c.f32_only`, UNTOUCHED.** An f64 param region
  is an honest miss (`UnsupportedDtype`), never a panic — AOT-first, exactly the
  hetero-multi posture. (For a *standalone* `AddScalar`/`MulScalar` region the miss
  is over-determined: the region also has no contract, so it misses at the contract
  gate regardless of the param-dtype gate.)
- **One honesty-only contract change.** For a *single-output* f64-param op (e.g. an
  f64 affine `a*x+b`) a contract IS emitted; its `op_params:` block previously
  hardcoded `dtype: F32`. It now emits the real FKC dtype token (`F64` for f64,
  `F32` byte-identical for f32) — reusing the same token the `accept.dtypes:` line
  already spells. This block "rides only the JIT seam, stored UNPARSED" (its own
  comment), so it never reaches Fuel's parser through a bundle — pure honesty.

## What an honest "Fuel dispatches an f64-param kernel" path would take (radar, not now)

The frozen envelope needs **no structural change** for f64 params: `LinkEntry` =
`(entry_point, structure_key, revision_hash)` carries no param slot; `structure_key`
already keys the f64 compute dtype distinctly, and `revision_hash` (over
`kernel.source`) distinguishes a `double p0` signature from `float p0`. The FKC
`AddScalar`/`MulScalar` `extract:` routing records only `(name, path)` — no dtype —
so it is dtype-agnostic.

The ONE new requirement, whenever Fuel later dispatches an f64-param kernel: **its
launch-arg marshalling must pass a `double`** (8 bytes, by value) for the f64 param
slot, not a `float`. This is the only Fuel-side dependency, and it is a read-only
ground-truth question to be filed propose-first *if/when* that call path is pursued
(e.g. an f64 dropout consumer, or an f64 affine/eps-as-Param reduction). The v1 AOT
kernel is device-validated against the CPU oracle now and needs none of it.

## The bespoke f64 widening protocol (recorded honestly — the topk-carve-out parallel)

The bespoke `dropout_fw_kernel<double,double>` (`baracuda_random.cuh`) hardwires
`rand` as `const float*` and `keep_prob` as `float` (`= 1.0f - p`) EVEN at
T=double, while the generator loads `rand` as `double` and compares double<double.
So the bespoke is a valid SECONDARY cross-check only under an exact widening
protocol: fill the generator's f64 `rand` with the exact widening of the bespoke's
float rand (float→double is exact and order-preserving), and pass the generator
keep_prob as `(double)(1.0f - p)` (widen the FLOAT keep_prob, NOT `1.0 -
(double)p`). `scale` is a pure double passthrough on both sides. Then masks and
values match bit-for-bit. This divergence is documented in the ondevice README and
is the direct parallel to topk's bespoke NaN/tie carve-out — the CPU f64 oracle is
therefore the PRIMARY gate.
