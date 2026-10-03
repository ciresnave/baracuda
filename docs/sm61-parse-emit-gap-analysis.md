# sm_61-through-latest gap analysis: `baracuda-cuda-parse` / `baracuda-cuda-emit`

Research-only, no code changes. Answers the gap list CireSnave's 2026-10-02
ruling asked for (quoted in full in `README.md`'s "Hardware support" section
and in the addendum at the top of `docs/llama-cpp-technique-catalogue.md`):
what does the IR parser/emitter pair need, per architecture from sm_61
through the latest, and who owns closing each gap — baracuda or Unpopped.

**Method**: static code read of `origin/main` at `c4be3829` (this repo),
`unpopped-vocab` 0.11.0 (the version this repo is actually pinned to) and
0.14.0 (the latest published, to check whether anything changed), via the
local cargo registry cache. **No compilation, no device testing** — every
claim below is grounded in a specific file:line read, not inferred from
prose or assumed from the architecture's public reputation. Flagged
explicitly wherever that static-only method leaves a real question open.

## Headline finding (read this first)

**The blocker is not where the earlier Phase A note assumed.** Three facts,
each independently verified below, combine into a much narrower gap than
"baracuda needs a non-tensor-core kernel family":

1. Unpopped's own neutral layer **already supports an open, arbitrary
   `cuda:sm61`-style target token** — this was a deliberate migration away
   from the closed `ArchSku` enum, completed before this ruling, for
   unrelated reasons (supporting `vulkan:`/`rocm:`/`metal:` targets). It is
   not something Unpopped needs to build; it already exists.
2. `baracuda-cuda-emit`'s actual code-generation templates (`cuda.rs`) use
   **zero tensor-core, async-copy, or dp4a intrinsics anywhere** in their
   real (non-test) code, for any op family. The generated `.cu` source is
   plain scalar/pointer-arithmetic CUDA C — architecturally portable to
   Pascal already, as far as a static read can tell.
3. The actual concrete wiring gap is one function:
   `NvrtcCompiler::new(arch: ArchSku)` in
   `crates/baracuda-cuda-emit/src/nvrtc.rs`, which hardcodes the closed,
   4-variant `ArchSku` enum (`Sm80`/`Sm89`/`Sm90`/`Sm90a`) to pick the
   `--gpu-architecture=` NVRTC flag — so it cannot be asked to compile for
   `sm_61` today, even though the open target model underneath it already
   could express that target.

**This is baracuda's gap to close, not Unpopped's**, and it's small in
code-surface terms (one function's parameter type and one match arm's worth
of logic). The much bigger remaining unknown is **on-device verification**:
nobody has compiled or run an emitted kernel on real sm_61 hardware. See
"What this analysis cannot tell you" at the end.

## Architecture roster (what actually exists to target)

Read from `unpopped-vocab::ArchSku` (`src/layout.rs`), both the pinned
0.11.0 and the latest 0.14.0 — **identical in both**, so this isn't a
version-lag issue:

```
ArchSku::Sm80   // Ampere baseline (also the Ada/Hopper JIT-forward target)
ArchSku::Sm89   // Ada Lovelace (FP8 tensor cores)
ArchSku::Sm90   // Hopper portable baseline (-arch=sm_90)
ArchSku::Sm90a  // Hopper specialized (-arch=sm_90a)
```

There is no Sm61/Sm70/Sm75/Sm100 variant in either version. **This enum is
intentionally closed** (`#[non_exhaustive]` is explicitly NOT applied — its
own doc comment says new variants are "a deliberate breaking-change event"
so every match site surfaces a build break). That closure is real, but as
of the `TargetId` migration (below), `ArchSku` is a *convenience
constructor* for the 4 historically-common CUDA cases, not the actual
representation `StructureKey` stores — so its closure no longer blocks
expressing a 5th CUDA target.

### The `TargetId` open-target model (already shipped, not a gap)

`unpopped-vocab::structure_key::StructureKey::target` is typed `TargetId`
(`src/target.rs`), not `ArchSku`. Its own module doc states the reason
plainly:

> "`ArchSku` is a CUDA-only enum: four `sm*` variants and no way to spell
> `vulkan:`, `rocm:` or `metal:` at all. That is a vocabulary owned by one
> vendor sitting inside a crate whose entire claim is neutrality... A
> conforming `vulkan:` reference vector could not be represented, so it was
> *excluded* from the cross-project byte-match rather than matched."

`TargetId` is a `u16` handle into a process-wide intern table, validated
against KISS-CLASSIFY §6.8's grammar (`<namespace>:<capability-set>`,
exactly one `:`, non-empty both sides, ASCII, no `structure_key` separator
bytes) — **and it validates the grammar only, never the vocabulary**
(`rocm:gfx942` is accepted without the crate knowing what a gfx942 is, by
design, per §6.8-0004: each namespace's capability-set belongs to its
maintainer). `TargetId::parse("cuda:sm61")` would succeed today — nothing
in this validation path singles out the 4 `ArchSku` capability strings as
special; they're just the four tokens that happen to already be registered
at process start (`RESERVED` in `target.rs`) and have an infallible
`From<ArchSku>` convenience conversion. A test in that same file
(`the_cuda_tokens_are_byte_identical_to_the_closed_enums`) confirms the
round trip: `TargetId::parse("cuda:sm89") == TargetId::from(ArchSku::Sm89)`.

**Conclusion: Unpopped does not need to add an `ArchSku::Sm61` variant, or
anything else, to support a `cuda:sm61` target token.** The open model
already does. (It's possible a *future* Unpopped change adds a dedicated
`ArchSku::Sm61` purely for ergonomics — infallible `From` instead of a
fallible `parse()` call — but that would be a convenience, not a
prerequisite.)

## `baracuda-cuda-emit`: what actually reads the target today

Grepped every `ArchSku`/`arch_sku`/target reference in
`crates/baracuda-cuda-emit/src/cuda.rs` (the `Cuda` `Backend` — the actual
IR→`.cu` emitter): **every single hit is inside the `#[cfg(test)]` module**
(starting ~line 7928). Test fixtures hardcode `ArchSku::Sm89` as an
arbitrary placeholder when constructing a `StructureKey` for a test — not
because the choice of arch changes anything about the kernel source those
tests assert against. Zero hits in the non-test code above that module.

Corroborating check: grepped `cuda.rs` for `mma.sync`, `cp.async`,
`ldmatrix`, `wmma`, `__dp4a`, `tensor core` — **zero hits, anywhere in the
file**, including the Gemm/contraction-epilogue path (which uses a plain
scalar accumulator loop over `k`, not a CUTLASS/WMMA fragment scheme; see
the `contraction_bias_emits_bias_param_and_column_load` test and its
surrounding real emitter code for a concrete example).

**Conclusion: the real emitter generates architecture-portable scalar CUDA
C for every op family it currently handles, and does not branch on target
at all.** As far as a static code read can establish, there is no
per-architecture codegen TODAY for *any* target, Ampere included — the
"intent + hint" split this project is moving toward doesn't exist yet in
this crate; everything currently shipped is "intent, no hint."

This is genuinely good news for the sm_61 goal specifically (nothing to
retrofit for portability — it was already portable) but also means: don't
expect any existing performance tuning to carry over per-arch either, since
none exists yet to carry. The Phase A catalogue (`llama-cpp-technique-
catalogue.md`) is the reference for what arch-specific hints *could* be
added later; none of that is wired into `cuda.rs` yet.

## `baracuda-cuda-emit`: the actual gap — `NvrtcCompiler`

`crates/baracuda-cuda-emit/src/nvrtc.rs`:

```rust
pub struct NvrtcCompiler { arch: ArchSku }
impl NvrtcCompiler {
    pub fn new(arch: ArchSku) -> Self { Self { arch } }
    // ... uses format!("--gpu-architecture={}", arch_flag(self.arch))
}
fn arch_flag(arch: ArchSku) -> &'static str {
    match arch {
        ArchSku::Sm80 => "sm_80",
        ArchSku::Sm89 => "sm_89",
        ArchSku::Sm90 => "sm_90",
        ArchSku::Sm90a => "sm_90a",
    }
}
```

This is the **one real, production (non-test) place in baracuda-cuda-emit
that reads an architecture value to make a decision** — and it takes the
closed `ArchSku` by value, so it structurally cannot be asked to compile
for `sm_61` (or any 5th target) without a code change here, regardless of
what `TargetId`/`StructureKey` can already express upstream.

**Fix shape (not implemented in this research pass — Phase B work)**:
`TargetId::capability_set()` already gives the bare string a target needs
(`"sm89"` for `"cuda:sm89"`, confirmed from its own doc example). The NVRTC
flag format needs an underscore baracuda's current `arch_flag` hardcodes
(`"sm_89"`, not `"sm89"`) — so the fix isn't a bare string substitution,
it's: accept a `TargetId` (or something that yields one), extract its
`capability_set()`, and reformat it into the `sm_NN` shape NVRTC expects
(validating the namespace is `"cuda"` first, since a `vulkan:`/`rocm:`
token has no business reaching an NVRTC flag at all). This is a small,
well-scoped change entirely inside this one file/struct.

## `baracuda-cuda-parse`: no arch-awareness, by design — nothing to do

Grepped `crates/baracuda-cuda-parse/src/*.rs` for `ArchSku`/`arch_sku`/
`TargetId`/`sm_6`/`sm61`/`Pascal`: **zero hits.** Consistent with its job
(lifting existing CUDA *source* idioms to arch-neutral IR — parsing
doesn't need to know what architecture the *resulting* kernel will later
target; that's an emit-time concern). **No gap here, and none expected** —
flagging the absence with its own positive-control-style check (the grep
pattern itself clearly works, since it found real hits in `cuda-emit`'s
equivalent search) rather than assuming silence means a broken search.

## Ownership summary

| Component | sm_61-through-latest gap | Owner |
| --- | --- | --- |
| `unpopped-vocab::StructureKey`/`TargetId` | None — open target model already supports arbitrary `cuda:smNN` tokens | N/A (already shipped) |
| `unpopped-vocab::ArchSku` | Optional ergonomic convenience (infallible `From` for a 5th CUDA token); not a prerequisite | Unpopped's call, not required |
| `baracuda-cuda-emit::cuda::Cuda` (codegen templates) | None found — already architecture-portable scalar C, for every op family checked | N/A (already shipped, pending on-device confirmation — see below) |
| `baracuda-cuda-emit::nvrtc::NvrtcCompiler` | **Real gap**: hardcoded to the closed 4-variant `ArchSku`; needs to accept an open target and derive the `--gpu-architecture=` flag from it | **baracuda** |
| `baracuda-cuda-parse` | None — architecture-neutral by design, confirmed no arch references exist | N/A |
| New "intent op" vocabulary | Not needed for sm_61 specifically — this is a backend-wiring gap, not a missing neutral-IR capability | N/A for this ruling |

## What this analysis cannot tell you (explicitly unverified — follow-up)

- **No kernel has actually been compiled for `sm_61` or run on real Pascal
  hardware** as part of this research pass. The claim "the generated
  source has no tensor-core/async-copy intrinsics" is a grep-based absence
  claim over the op families `cuda.rs` currently covers — it does not rule
  out a PTX ISA feature that's absent on Pascal for some *other* reason
  (certain atomic operations, specific intrinsic availability, `__activemask`-
  class warp primitives introduced after Pascal, etc.) that a grep for the
  three named intrinsics wouldn't catch. Actually invoking `nvcc`/NVRTC with
  `-arch=sm_61` against a representative sample of the emitted kernels (or,
  short of real Pascal hardware, at least a successful PTX/SASS compile) is
  the real verification step this static read cannot substitute for.
- **This dev box is an RTX 4070 (sm_89)** per existing project memory — no
  Pascal/P40 device is available locally for the on-device half of that
  verification. Whether CI or another machine in the portfolio has sm_61
  hardware wasn't checked in this pass.
- **sm_70/sm_75 (Volta/Turing)** weren't investigated in this pass at all —
  `ArchSku` doesn't have a variant for them either, and the same open
  `TargetId` argument presumably applies, but that wasn't separately
  confirmed; flagging as a follow-up rather than assuming it transfers
  without checking.
- **"The latest" architecture** is `Sm90a` (Hopper-specialized) as of both
  unpopped-vocab 0.11.0 and 0.14.0 — no Blackwell/`sm_100` variant exists
  yet in either. If "latest" needs to mean Blackwell specifically, that's
  an upstream Unpopped gap (a new `ArchSku` variant or at minimum a
  registered `cuda:sm100` token), not investigated further here.
- **Whether `unpopped::generate()`/the dispatch layer** (as opposed to
  `baracuda-cuda-emit`'s own code) has any target-conditioned gating that
  would reject an unrecognized `cuda:sm61` token before it even reaches
  `cuda.rs` was not checked — this pass only confirmed `cuda.rs` itself
  doesn't branch on target, not the full call path from `generate()` down
  into it.
