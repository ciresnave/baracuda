# Baracuda reply — JIT synthesizer seam integration (§5.2): binding, link_registry, concurrency, budget

**Re:** Fuel's four §5.2 integration questions (direct-Rust surface, runtime
link_registry, sync/async, budget actionability).
**Status:** all four answered from the current `baracuda-kernelgen::jit`
source (the `seam` module + the `Compiler` trait + `link.rs`). Short version:
**Baracuda impls your `Synthesizer` trait** (dep points Baracuda → `fuel-kernel-seam`);
runtime-added synthesized kernels are handed over via a per-instance registry +
`take_kernel`; the trait method is **sync** but the impl is `Send + Sync` +
interior-mutable so **you own the concurrency** (call it on an idle-time thread
per G7); and `max_compile_ms` is honored as a validated budget + a typed decline
today, but **hard wall-clock enforcement is a coarse/future item** we can tighten
with a watchdog if you want a real-time ceiling.

## Q1 — Direct-Rust surface: **Baracuda impls your `Synthesizer` trait**

`impl Synthesizer for BaracudaSynthesizer` (`jit.rs:904`), where `Synthesizer` is
**your** trait from the `fuel_kernel_seam` envelope crate. So:

- **Trait ownership: Fuel.** `fuel_kernel_seam::Synthesizer` is yours; Baracuda
  provides the impl, not a `fn` you wrap.
- **Dependency direction: Baracuda → `fuel-kernel-seam`** (+ `fuel-kernel-seam-types`
  for the region grammar), both crates.io deps behind Baracuda's `seam` feature.
  Baracuda depends on Fuel's envelope; Fuel depends on nothing of Baracuda's at
  the type level — you hold a `&dyn Synthesizer` / `Box<dyn Synthesizer>` that
  *is* a `BaracudaSynthesizer` and call `.synthesize(&req)`.
- The method signature is exactly your envelope's:
  `fn synthesize(&self, req: &JitRequest) -> JitResponse` (`jit.rs:905`), never
  panics (an unbuildable / out-of-vocabulary / over-budget region is a typed
  `JitResponse::Declined`, not an error or a cross-boundary crash).

Construction: `BaracudaSynthesizer::new(max_compile_ms)` (`jit.rs:887`). You own
the instance and its lifetime; Baracuda owns none of Fuel's types.

## Q2 — Runtime link_registry: **yes, runtime-added entries via a per-instance registry + `take_kernel`**

Two distinct surfaces — don't conflate them:

1. **AOT static roster** — `emit_link_registry` produces
   `pub static BARACUDA_LINK_REGISTRY: &[(&str, &str, u64)]` (`link.rs:48`,
   `(entry_point, structure_key_token, revision_hash)`), the compile-time catalog
   named in the bundle front-matter as `link_registry: baracuda_link_registry`.
   This is for pre-built kernels, resolved at load.

2. **JIT runtime-added** — a synthesized kernel is NOT in the static roster (it
   didn't exist at build time). `BaracudaSynthesizer` holds
   `registry: Mutex<HashMap<String, SynthArtifact>>` (`jit.rs:881`); every
   `synthesize` **inserts** the compiled artifact under its `entry_point`
   (`jit.rs:942`) — this is the runtime-add. The wire response
   (`JitResponse::Synthesized`) stays light (carries only the `entry_point`); the
   PTX/source/contract/recipe **and the `link: LinkEntry` row** ride in the
   retained `SynthArtifact`.

**How you call it at adopt time** (`jit.rs:894-901` + module doc `:872-877`):

```
let resp = synth.synthesize(&req);          // returns JitResponse::Synthesized{entry_point} | Declined
// ... your cost-gate decides to adopt ...
let art  = synth.take_kernel(&entry_point)  // Option<SynthArtifact>: removes + returns the artifact
    .expect("adopt a kernel this synthesizer produced");
// art.artifact = PTX bytes (art.kind = artifact provenance); art.link = LinkEntry
//   (entry_point → KernelRef row, the FKC §12.6 runtime binding);
//   art.contract / art.recipe = the FKC contract + re-fuse recipe.
// → load art.artifact as a module, resolve entry_point, wrap KernelRef, adopt_runtime_fused.
```

So the `link_registry` for a synthesized kernel is delivered **per kernel, at
adopt time**, as `art.link` — the runtime-added row you register into your
dispatch alongside the static roster. `take_kernel` removes the entry (single
adopt), returns `None` if never synthesized / already taken. The registry is
`Mutex`-guarded, so concurrent synthesize + take are safe (see Q3).

## Q3 — Concurrency: **sync trait, but you own async — the impl is `Send + Sync`**

The trait method is synchronous request/response per §5.2 — a blocking
`&self -> JitResponse`. But `BaracudaSynthesizer` is **`Send + Sync`** (auto: its
fields are `u32` + `Mutex<HashMap<String, SynthArtifact>>`, and `SynthArtifact` is
all owned `String`/`Vec<u8>`/enums), and it's **interior-mutable** (the registry
is behind a `Mutex`, `synthesize` takes `&self`). Consequences for your G7
idle-time model:

- **Call it on a background / idle-time thread.** Nothing forces the compile onto
  the realize path — hold the `Arc<BaracudaSynthesizer>`, spawn the synthesize on
  a worker, don't block the realize; adopt the result (via `take_kernel`) when it
  lands. That matches the constitution's "JIT fusion is a background
  re-optimization trigger" framing.
- **Concurrent calls are safe.** `&self` + the `Mutex` registry means multiple
  threads may `synthesize`/`take_kernel` at once; the registry serializes the
  inserts/removes. (The compile itself is CPU/nvrtc work, not internally
  parallelized — one region per call.)

So: **the method is sync; the concurrency model is yours.** Baracuda gives you a
thread-safe, interior-mutable synthesizer you can drive synchronously OR from a
background thread, exactly as G7 wants — no async trait needed, and no blocking
requirement imposed on your realize. If you'd prefer an explicitly `async`
entry point (a `fn synthesize_async(...) -> impl Future`), we can add one that
just offloads to a blocking pool — but the `Send + Sync` sync fn already lets you
own the threading without it.

## Q4 — Budget: **`max_compile_ms` honored as a validated budget + typed decline; hard wall-clock enforcement is a coarse/future item**

Honest, in three parts:

- **The decline mechanism exists and is wired.** `max_compile_ms == 0` is a typed
  `JitError::Budget` → `JitResponse::Declined` (`jit.rs:306-308`), never a panic.
  So "decline when it can't honor the budget" is already the contract — a region
  that can't be built in budget returns `Declined`, and your ranker falls back to
  the primitive path.
- **BUT wall-clock enforcement is coarse today, not a hard deadline.** The
  `Compiler::compile(source, entry, max_compile_ms)` seam (`jit.rs:179`) passes
  the budget through, and the doc is explicit (`jit.rs:222-224`): *"nvrtc has no
  compile-deadline API; `max_compile_ms` gates optimization depth / the inward
  e-graph's iteration count at a coarser grain (future)."* So today the budget
  bounds the **optimizer's** work (region depth + e-graph saturation), not a
  hard interrupt of a runaway nvrtc compile. A region that's cheap to optimize
  but slow to nvrtc-compile could overrun the wall-clock ceiling without an abort.
- **A hard real-time ceiling is implementable — confirm if you want it.** nvrtc
  compilation isn't interruptible, so a true `max_compile_ms` wall-clock ceiling
  needs a watchdog: compile on a worker thread, join with a timeout, and return
  `Declined` (abandoning the worker) on overrun. We can add that behind the seam
  — the question is whether you need a **hard** ceiling (real-time realize path)
  or the current **coarse** budget (bound optimizer effort, decline on
  structural over-budget) is enough for a background/idle-time trigger.

**Other budget axes already enforced** (so you know what's covered): a
`MAX_REGION_DEPTH` stack guard (deep-region rejection), the `MAX_OPERANDS` arity
cap, and the dtype-support gate — all typed declines. **Candidate axes we could
add if you'd want them:** (a) a **register/shared-memory budget** (decline a
synthesized kernel whose resource use would hurt occupancy on the target arch),
(b) a **region op-count / flops cap** (bound the fused body size independent of
compile time). Tell us which budget axes your ranker would actually gate on and
we'll surface them on `JitRequest.budget` alongside `max_compile_ms`.

## Summary

| Question | Answer |
|---|---|
| Direct-Rust surface | Baracuda **impls** your `Synthesizer` trait; dep = Baracuda → `fuel-kernel-seam`; you own the trait + the instance. |
| Runtime link_registry | Per-instance `Mutex<HashMap>` registry accepts runtime-added synthesized kernels; fetch at adopt via `take_kernel(entry_point)` → `SynthArtifact{artifact, link, contract, recipe}`; `art.link` is the runtime binding row. |
| Sync vs async | Sync trait method; impl is `Send + Sync` + interior-mutable, so **you** own the concurrency — call it on a background/idle-time thread per G7. Can add an `async` wrapper if you want one. |
| Budget | `max_compile_ms` validated + typed `Declined` today; hard wall-clock enforcement is coarse (bounds optimizer effort, not a compile interrupt) — a watchdog gives a hard ceiling if you need real-time; other axes (regs/smem, op-count) available on request. |
