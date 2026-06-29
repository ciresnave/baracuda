# Baracuda → Fuel — the §5 Synthesizer is live; here's your seam-call-site handoff (2026-06-24)

`fuel-kernel-seam` 0.10.2 is on crates.io and **Baracuda implements it.** The
`Synthesizer` trait is wired, on-device validated, and `SeamCapJitOnRequest` is now
**ON** in `BARACUDA_CAPABILITIES`. The loop is live the moment your seam-call site
calls us.

## What we built (commit `3b8adbf`)

`baracuda_kernelgen::jit::seam::BaracudaSynthesizer` impls
`fuel_kernel_seam::Synthesizer`:

```rust
let synth = BaracudaSynthesizer::new(/* max_compile_ms */ 5000);
match synth.synthesize(&req) {            // req: fuel_kernel_seam::JitRequest
    JitResponse::Synthesized(k) => { /* k.entry_point, k.pattern (= region), k.cost */ }
    JitResponse::Declined { reason } => { /* leave the region on primitives */ }
}
```

- Adapts your `JitRequest` (region + raw `OperandDesc` projection + arch) to our
  native synthesizer, classifies via `structure_key` (we compute the key, you never
  pre-classify — the ratified division), and emits the specialized kernel.
- `cost` = `"n * (n_inputs+1)"` — the fused op's single-pass memory traffic. Your
  `cost_expr` core parses it (binding `n` = out elem count); cost-gate it against the
  multi-pass primitive path.
- **Never panics** (your trait's contract): an out-of-vocabulary op, a
  backend-unlowerable dtype, an over-budget or malformed request is a typed
  `Declined`, never an unwind across the boundary.

## Your seam-call site — the one piece left, and it's yours

The envelope's `SynthesizedKernel` is light (carries `entry_point`, not the PTX), so
after a `Synthesized` reply your site:

1. cost-gates `k.cost`; if adopting,
2. `let art = synth.take_kernel(&k.entry_point)` → the retained `SynthArtifact`
   (`source`, `artifact` = the PTX bytes, `kind` = `Ptx`, `contract`, `recipe`,
   `link`);
3. load `art.artifact` into a CUDA module, bind `entry_point` → a fuel-dispatch
   `KernelRef`;
4. `adopt_runtime_fused(name, k.pattern, kernel_ref, dtypes, BackendId::Cuda)` — which
   you already built in `fuel-dispatch/runtime_fused_kernels.rs`.

`take_kernel` consumes the entry (returns `None` if already taken / never
synthesized). We kept the PTX→`KernelRef` wrapping on your side deliberately: the
`KernelRef` signature + the launch live in `fuel-dispatch`, and the envelope stays
free of any CUDA-runtime dep.

## One detail you'll want to know

We added a workspace `[patch.crates-io] baracuda-kernels-types → our local copy` so
the envelope's `OperandDesc`/`ArchSku` are the **same** compilation unit as ours
(otherwise the registry copy and our path copy are distinct types and the impl won't
typecheck). If you build a harness that links both our synthesizer and the registry
`baracuda-kernels-types`, you'll want the same patch.

## Validation

Default 59 + `--features seam` 64 tests green; clippy clean. On sm_89 the full live
path — your `JitRequest` → `Synthesizer::synthesize` → real nvrtc PTX, retrievable
via `take_kernel` — passes (`synthesizer_produces_real_ptx_on_device`). An
adversarial 3-agent review hardened the boundary (a pure-infix region at a
backend-unlowerable dtype used to panic in lowering; now a clean `Declined`, with a
region-depth stack-guard and a collision-resistant entry-point id).

Send the seam-call site and we watch the first real region fuse.
