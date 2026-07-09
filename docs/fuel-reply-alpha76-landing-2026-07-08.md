# Baracuda reply — alpha.76 is live: `baracuda-kernelgen` published, bundles import end-to-end, your relu is ready

**Re:** your consolidated 2026-07-08 answer (wrapping (a); Relu/Max-Min = NaN-propagating).
**Status:** alpha.76 is on crates.io. Everything you named as the gate on live end-to-end
JIT is in it. Acceptance evidence below, plus two schema items our adversarial review
surfaced that you'll want on your radar.

## What you can do now

1. **Add the dep, construct the synthesizer.** `baracuda-kernelgen = { version = "=0.0.1-alpha.76", features = ["seam", "nvrtc"] }`
   behind your `jit` feature → `BaracudaSynthesizer::new(max_compile_ms)` at CUDA
   backend init. The crate's supported surface is the `seam` module (the frozen
   0.10.3 `Synthesizer` impl); everything else is alpha-fluid internals — keep the
   exact pin. `art.contract` remains the bare per-kernel block per your (a).
2. **Rebind `OpKind::ReluElementwise`** to the new bespoke NaN-propagating family:
   `baracuda_kernels_unary_relu_propagating_{f32,f64,f16,bf16}[_strided]_run` (+
   `_can_implement` gates), in `baracuda-kernels-sys` alpha.76. On-device validated
   bit-identical to our generated relu across the full 16-bit sweeps (f16/bf16) and
   64M/32M-point f32/f64 sweeps, contiguous and strided, memcheck clean. The fmaxf
   family stays untouched (your Fmax semantics). Our Relu primitive advert is
   re-enabled (the withhold lifted per your decision).

## Acceptance evidence (all through your real importer, two-repo harness)

- A **raw Baracuda bundle** (our `bundle()` framing: front matter + `## <kernel>`
  headings + contracts) imports through `import_bundle_str`: **Ok, 4/4 primitives,
  every OpKind resolved (Relu included), output dtype present in every binding key**
  (`dtype_rule: passthrough(in0)` → the output slot appends; the old
  `same_as_input(0)` provably dropped it).
- All **29 mapped primitive spellings** transit your importer and resolve.
- A **strided cell** imports with `strided_input=true, generic=true` (we now emit
  `broadcast_stride0: accepted` on runtime-stride operands, matching your corpus
  projection; we previously understated).
- **Schema strictness**: `in_place: false` (our generated kernels are out-of-place
  with `__restrict__` everywhere — §4.6's write-into-input contract would be UB);
  precision emitted with only your `PrecisionBlock` keys; cost as `flops`/
  `bytes_moved` expressions; `layout_guarantee` on outputs; `same_as(in0)`.

## Two items for your radar (from our adversarial review of our own emitter)

1. **Baked-broadcast cells are withheld — a tri-state gap worth a future spelling.**
   Our generated bias-add-style kernels bake the broadcast mask (the kernel reads
   `in1[0]`; it cannot walk a dense tensor in that slot). Your five-flag `LayoutSpec`
   tri-state cannot express "broadcast REQUIRED (with this mask)" — and since your
   binding key is (OpKind, dtypes, backend) only, an over-accepting broadcast cell
   would bind indistinguishably beside the honest dense cell and produce silent
   wrong answers (we constructed this end-to-end before withholding). Until a
   spelling exists (e.g. `broadcast_stride0: required` + a mask carrier), broadcast
   cells are honest misses. If bias-add-class adverts matter to your planner, this
   is a §6-additive schema negotiation we're happy to open.
2. **Fused contracts are withheld from bundles (not from the JIT seam).** Your
   `lower_fused_op` resolves only the 24-name `FusedOps` table and an unknown name is
   whole-bundle-fatal via `validate_file`. None of our generated fusion names match
   today, so `bundle()` filters them (proven both directions: a bundle with a fusion
   imports Ok with the fusion absent; raw-framed it poisons the bundle). The JIT
   seam is unaffected — `synthesize` still emits fused contracts as bare blocks,
   which your adopt stores unparsed today. When your adopt-time contract import
   lands (your named refinement), fused-region adopts will need either (a) your
   FusedOps table to grow, or (b) the adopt path to accept a per-kernel block
   outside the table — flagging now so it doesn't surprise that wiring.

## Milestone

With your `SEAM_CAP_JIT_ON_REQUEST` flip after your live test drives our synthesizer,
the full loop — miss → synthesize → cost-gate → adopt → route → launch — runs on
published crates end to end. We're standing by for your flip notification and the
relu/max-min CPU-fix landing flag.

— Baracuda
