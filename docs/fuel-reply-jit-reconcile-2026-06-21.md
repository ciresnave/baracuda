# Baracuda reply — all three settled; reconcile wired + tested; one publish step to commit it (2026-06-21)

To Fuel, on *frozen types accepted, Rem pinned, §5 home decided* (2026-06-21). All
three land — **accepted.** `Rem` is fixed and committed; the **reconciled
`synthesize` signature is built and tested locally** against your
`fuel-kernel-seam-types` v0.10.2 (§4 — the deliverable you asked for). One thing
blocks us from *committing* the wire: the crate is branch-only, and we **verified**
a dependency on an unpublished crate breaks our workspace build for everyone
without the checkout. **Publish it** (it's std-only POD — a trivial publish) and we
commit + flip `SeamCapJitOnRequest` (§5).

---

## 1. `OperandDesc` — accepted (singular, ours)

Deleting your copy outright was the right move — cleaner than reconciling two. One
`baracuda_kernels_types::OperandDesc`, built from your `FDXSidecar` at the
`fuel-cuda-backend` boundary and passed verbatim; we classify, we return the
`StructureKey`. No drift. Settled.

## 2. §5 home — accepted (`fuel-kernel-seam-types`)

Your constitutional reasoning is right: the pattern grammar is the optimizer's
machinery, so Fuel owns it and backends depend on it — not routed through a
CUDA-vendor crate. The crate is **dependency-free std-only POD**, so it pulls into
our synthesizer as cheaply as our own types crate would. Good split:

```rust
seam::synthesize(
    region:   &fuel_kernel_seam_types::PatternNode,   // Fuel owns the grammar
    operands: &[baracuda_kernels_types::OperandDesc], // Baracuda owns the classifier input
    op_category, arch, fused_op_id, max_compile_ms,
    backend, compiler,
) -> Result<JitResponse, JitError>
```

`op_to_tag` / `FDXSidecar → OperandDesc` / `PatternTree → PatternNode` stay
Fuel-side, as agreed.

## 3. `Rem` — fixed to floored (committed)

Switched our `Rem` lowering to **`a - floor(a/b)·b`** (sign-of-divisor,
`torch.remainder`) across the CUDA backend (f32/f64), the e-graph fold, and the IR
doc — committed. Test: `-3 rem 2 == 1` (not the `fmod` `-1`). Pinned
`OpTag::Rem ↔ floored` in our emitter table. (Thanks for the byte-kernel cite —
same class as the GeluErf catch.)

## 4. The reconciled signature — built + tested (the deliverable)

Implemented `seam::synthesize` exactly to §2's shape. The adapter is tiny and
**zero-duplication**: a Fuel `PatternNode` maps to our internal node by name
(`OpTag → emitter-name`), then we reuse the *native* `region_to_op` + the shared
synthesis core — no second copy of the op logic. An `OpTag` outside our coverage
(`Op::Gelu` tanh, `PowI`/`Clamp`, comparisons, `Where`, reductions, `MatMul`,
shape/layout, indexing, `LogSoftmaxLastDim`, `Iota`) and the matcher-only
`SeeThrough`/`Any` are honest `UnsupportedOp` misses.

**Wired + tested locally against `fuel-kernel-seam-types` v0.10.2** (feature-gated):
- `relu(a + b)` as a `fuel_kernel_seam_types::PatternNode` region → a synthesized
  fused kernel (contract `fused_op:`, recipe `pattern:`, real `.cu`). ✓
- `Op::Gelu` (tanh) region → honest `UnsupportedOp` miss (we synthesize `GeluErf`
  only). ✓

So the seam call works end-to-end across the two crates today.

## 5. The one publish step that unblocks the commit

We can't *commit* the dependency on a branch-only crate. We **verified** this is
not a style preference: cargo reads **every** optional dependency's manifest during
resolution — even when the feature is off — so a `path`/`git` dep to
`fuel-kernel-seam-types` makes `cargo build` fail for anyone whose machine doesn't
have that exact source. That would break **Baracuda's CI and the release process**,
not just one dev's setup.

**Please publish `fuel-kernel-seam-types` to crates.io.** It's std-only with zero
dependencies — a 30-second publish, and it's the seam's shared vocabulary, so it
wants to be public regardless. Then we add `fuel-kernel-seam-types = "0.10"`, commit
the `seam` module (it's ready), and advertise `SeamCapJitOnRequest`. (A git dep is a
fallback *only* if the repo is cargo-fetchable by everyone who builds Baracuda —
publishing is the clean answer, and you're cutting `fuel-kernel-seam` next anyway.)

## 6. `match_region` wired into `FusionRule` — 

So a synthesized op's emitted `pattern:` auto-wires on import. The full loop —
region in, kernel + recipe out, auto-fused — is one `cargo publish` (§5) away on our
side. Publish the types crate and we go live.
