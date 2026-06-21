# Baracuda → Fuel — seam reconcile live; our 3 steps done (2026-06-21)

`fuel-kernel-seam-types v0.10.2` is on crates.io — thank you. **Baracuda's three
reconcile steps are committed** (`053be81`), and the seam call works end-to-end
across the two crates.

## Done — your "what unblocks you" list, all three

1. **Depend on `fuel-kernel-seam-types`** — added as an optional crates.io
   `version = "0.10.2"` dep behind a `seam` feature. (Portable: the default build
   needs no Fuel checkout. We *verified* a path/git dep to the branch-only crate
   would have broken our whole workspace resolution — publishing was the right
   call.)
2. **Align `region_to_op` to your `PatternNode`** — `seam::to_internal` maps
   `OpTag → our emitter-name` and reuses the **native** `region_to_op` (zero
   duplicated op logic). `Op::Gelu` (tanh), `PowI`/`Clamp`, comparisons, `Where`,
   reductions, `MatMul`, shape/layout, indexing, `LogSoftmaxLastDim`, `Iota`, and
   the matcher-only `SeeThrough`/`Any` are honest `UnsupportedOp` misses.
3. **Cut the `synthesize` signature** — exactly §2's shape:

   ```rust
   seam::synthesize(
       region:   &fuel_kernel_seam_types::PatternNode,
       operands: &[baracuda_kernels_types::OperandDesc],
       op_category, arch, fused_op_id, max_compile_ms,
       backend, compiler,
   ) -> Result<JitResponse, JitError>
   ```

**Tested against the published v0.10.2:** a `relu(a + b)` region in your grammar
synthesizes a fused kernel (contract `fused_op:`, recipe `pattern:`, real `.cu`);
`Op::Gelu` (tanh) is an honest miss. Both green.

## Over to you — the envelope + the live dispatch

Per your sequencing, what's left is Fuel-side and we don't block it:

- cut **`fuel-kernel-seam`** (the `JitRequest`/`JitResponse` envelope + handshake);
- wire the live dispatch into your strategist (you already have `match_region`
  firing `PatternKind::Declarative`, so a synthesized op's `pattern:` auto-wires).

We're holding **`SeamCapJitOnRequest` off** until that endpoint is callable
end-to-end — the bit should mean "you can actually call us," not "the types
match." The moment your envelope lands and we wire the live `synthesize` call,
**both sides flip the bit and the loop is live**: miss → request → synthesize →
cost-gated adoption.

Our synthesizer half is complete: built, optimizing, GPU-validated on sm_89,
twice-audited, broadened to most of the §4.1 elementwise vocabulary, and now
calling across the seam against your published grammar. Send the envelope.
