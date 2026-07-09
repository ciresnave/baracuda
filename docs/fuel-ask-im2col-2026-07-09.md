# Baracuda ask — 2-D im2col / unfold shipped (`Access::Im2Col`, byte-identical to the bespoke on the FULL probe space); AOT-only, and the ONLY advertisable path is the im2col→GEMM→reshape `Conv2D` fusion (2026-07-09)

**No action needed now.** This is a propose-first heads-up in the alpha.76
landing-doc "radar item" class: a new kernel capability exists that is
deliberately NOT advertised to Fuel, and we record exactly what an honest "Fuel
calls the im2col kernel" path would take — before anyone wires it. The v1 AOT
kernel needs none of it and ships bit-exact today. **This is NOT a blocker.**

## What shipped (Baracuda, increment 11 on `feat/kernel-specialization`)

The **2-D im2col / unfold forward** kernel — the conv-lowering workhorse
(`Conv2d ≡ im2col → GEMM → reshape`). A NEW `Access::Im2Col { kernel, stride, pad,
dilation }` + `Schedule::Im2Col` + `emit_im2col` + `OpDef::im2col_2d`, sized right
at the shipped Window increment and touching **IR / plan / cuda only** — **ZERO
`contract.rs` / `pattern.rs` / `jit.rs` change, ZERO `baracuda-kernels-types`
change, no `STRUCTURE_KEY_VERSION` bump, byte-identical existing emission** (purely
additive on the `#[non_exhaustive]` `Access` enum + new match arms).

It is a pure EXPANDING structured gather: each of the `kh·kw` window taps over a
rank-4 `[N,C,H_in,W_in]` NCHW input becomes its OWN output cell, producing the
column matrix `[N, C·kh·kw, oH·oW]` (**Layout A** — channel-major then tap, spatial
row-major: `y[n, c·kh·kw + ki·kw + kj, oh·oW + ow]`, the exact bespoke `im2col_2d` +
PyTorch `F.unfold` order). The extent-INVERSE of `Access::Window` (which folds taps
into one downsampled output; this expands them). One thread per output cell
(grid-stride) computes the closed-form source coord `in_h = oh·stride_h − pad_h +
ki·dilation_h` / `in_w` symmetric, bounds-checks both spatial axes, and RAW-BIT
copies the in-bounds NCHW element or stores the typed zero (`zero_of<T>()`) for an
out-of-bounds (zero-pad) tap — no fold, no arithmetic, no NaN/tie convention. So
every dtype is a bit-exact move.

Conv geometry `(kh,kw,stride,pad,dilation)` rides the `Access`/`Schedule` node as
compile-time literals; the six runtime extents `(N, C, H_in, W_in, oH, oW)` ride
`long long` launch args (the Window 3-scalar ABI generalized to 6). The
`(H_in,W_in)→(oH,oW)` conv arithmetic is a **runtime-launch-arg caller
precondition** (the structure key carries no numeric extents — the same trust level
as Window's `k_in→k_out`), on-device-validated via `initcheck`.

**Acceptance (the strongest in the crate):** whole-buffer `memcmp`, `bit_diff == 0`,
vs BOTH the shipped bespoke `im2col_2d_fw_kernel` (`baracuda_im2col.cuh`) AND an
independent CPU closed-form byte oracle, across `{f32,f64,f16,bf16,i32,i64} ×` the
geometry matrix (1×1, 3×3+pad, stride>kernel, kh≠kw, dilation>1, non-square H≠W /
oH≠oW, batched N>1), probe-seeded with NaN payloads / ±inf / ±0 / subnormals and
**NO carve-out** — because the gather is raw-bit with the identical index map and
Layout A column order, byte-identity holds EVERYWHERE (unlike topk, which was
torch-faithful and had to exclude NaN/tie rows from its bespoke cross-check). All 41
cells + 4 sanitizers (memcheck/racecheck/synccheck/initcheck, 0 errors) PASS on
sm_89 / CUDA 13.3. See `crates/baracuda-kernelgen/ondevice/README.md`,
`im2col_validate.cu` section.

## The advert story today (honest miss — AOT-ONLY, no code change)

An im2col emits **no FKC contract**. `derive_pattern` rejects any non-`Elementwise`
access at the single gate `pattern.rs:229` — `Access::Im2Col` included — BEFORE any
body walk, so `pattern = None` and `contract()` withholds. `body == Input(0)` keeps
`n_outputs() == 1`, so the multi-output gate never fires; `NotElementwise` withholds
one step earlier regardless. This is the SAME wall that withholds shipped
Window/Scan/RowSort, with **zero** `contract.rs` / `pattern.rs` / `jit.rs` change (a
single honest-miss regression test, `im2col_is_an_honest_miss_no_contract`, is the
only addition).

**Why the AOT-only posture is CORRECT, not a limitation:** Fuel treats convolution
as a first-class **primitive** — the FKC-importable op-kind whitelist has
`Conv2D` / `ConvTranspose2D` and NO `Im2Col` / `Unfold` / `Pool` / `Window`
(`fuel-dispatch/src/fkc/lower.rs:167-196`, verified 2026-07-09: `Conv2D` at :192,
`ConvTranspose2D` at :193); the CPU conv contract states
"convolution is not a fused op … no im2col/Winograd fast path (naive direct loop)".
Fuel's three im2col spellings (the `fuel-conv::im2col` parity oracle, the
candle-style `Im2Col` / `Im2Col1D`, and the bespoke `conv2d_im2col.slang`) are ALL
internal lowering helpers, never a first-class/advertised OpKind. So im2col has **no
importable FKC shape** — there is nothing for a `derive_pattern` to bind to, and no
`OpTag` for `synth_op` to synthesize. AOT-only forever given current Fuel types, the
posture Window/Scan/RowSort already occupy.

## The ONLY path that could flip im2col from AOT-only to contract-carrying

im2col could ride a contract **only** as the producer half of a full
**im2col→GEMM→reshape `Conv2D` FUSION** — i.e. advertised under an existing Fuel
`Conv2D` OpKind rather than as a standalone `Im2Col`/`Unfold` op. That is a much
larger fused-op story (im2col producer → Contraction → reshape epilogue), out of
scope for this single-op increment and recorded here as the strategic follow-on. If
that call path is ever pursued, Fuel would need EITHER an `Unfold`/`Im2Col` lazy
primitive OR (more likely) to recognize a Baracuda FKC bundle as a `Conv2D`
lowering; a propose-first would be filed at that point. The v1 AOT kernel needs none
of it and is already device-validated bit-exact against the bespoke.

## Keying / ABI (no Fuel-facing change)

- **`baracuda-kernels-types` UNTOUCHED** — the conv params ride the `OpDef` +
  `Access`/`Schedule` (kernelgen-internal); the expanded output extent
  `[N, C·kh·kw, oH·oW]` rides launch args; the extent never touches
  `StructureKey` / `OperandKey`. **No new key field, no `STRUCTURE_KEY_VERSION`
  bump, no propose-first for the key.**
- The generated entry point is `baracuda_gen_<op>_<dtag>_im2col` with the
  6-`long long`-extent signature `(const T* in0, T* out, N, C, H_in, W_in, oH, oW)`.
  Distinct geometries of one op emit distinct symbols (the op_name encodes the
  geometry); the dtype rides the `<dtag>` infix.

## De-scoped family follow-ons (each a named future increment)

col2im / im2col-backward (`F.fold` — scatter-ADD with `atomicAdd`, non-deterministic
so it breaks the bit-exact memcmp acceptance; the bespoke sets
`bit_stable_on_same_hardware = false`), 1-D / 3-D im2col (additive rank variants),
grouped conv (`groups > 1` — needs the grouped bespoke as oracle), a per-tap `pre`
map (the Window `pre` bridge, additive), non-zero / reflect / replicate padding (no
bespoke parity target), Layout B `[N,oH,oW,C,kh,kw]` (candle order — no bespoke
byte-parity target), and the `Conv2D` fusion above (the only advertisable path).
