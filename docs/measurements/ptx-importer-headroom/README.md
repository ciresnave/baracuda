# Does a PTX importer have headroom over the hand-tuned kernels? (static lever survey)

## The question

A proposed "PTX importer" would read the PTX nvcc emits from the hand-tuned CUDA
kernels, apply optimizations the kernels don't have, and emit back out — so the
hand-tuned kernels become the *starting point*, not the ceiling. The hypothesis
behind it: **CUDA C may not expose all the levers PTX does, so the hand-tuned kernels
may be leaving performance reachable only one level down.** If true, the "a naive
emitter loses badly to 426 hand-tuned kernels" objection dissolves, because you start
from them instead of competing with them.

This survey checks that hypothesis against the emitted code before anyone writes an
importer. It is a **static lever inventory, not a benchmark.**

## TL;DR — the answer leans NULL

For the sampled shapes, **the PTX-level levers are already in the PTX, and the ones
that are not, nvcc chose not to emit.** The hand-tuned kernels have largely captured
what a PTX importer could see. Residual headroom is narrow, and it is one of three
kinds, none of which is "free win a rewriter picks up":

1. **Already captured** (read-only cache): present in 8/8 sampled kernels, *in the PTX*.
2. **nvcc declined it** (load/store vectorization): absent in 7/8, but nvcc emitted
   scalar having seen the C++ types, alignment, and access pattern. The importer reads
   lowered scalar PTX with *less* information than nvcc had, and sits downstream of the
   very decision it would reverse.
3. **Algorithmic, not peephole** (cp.async pipelining): genuinely absent in the int8
   GEMM — real headroom — but it is a pipeline rewrite (commit/wait barriers +
   double-buffering), the category "a naive emitter loses badly," not a rewrite pass.

And the occupancy signal that usually decides — register pressure — is **ptxas-owned**:
ptxas re-runs register allocation and scheduling on any imported PTX, so a PTX importer
does not control it.

This **refutes the premise (the levers CUDA C can't reach), not the possibility** of
specific per-kernel wins. The int8 GEMM's cp.async gap is a real per-kernel
hand-optimization — it is just not systematic free-headroom capture.

## The instrument error this survey survived — record it, don't hide it

The first pass reported **"read-only cache: 0 of 8 — headroom."** That was wrong, and
the wrong version pointed the design at *build the importer*.

The grep looked for the SASS token `.CI` (the read-only/non-coherent cache modifier on
some architectures). On this toolchain's SASS the modifier is spelled **`.CONSTANT`**
(and in PTX it is `ld.global.nc`). A correct grep for the wrong spelling returned a
false absence. The corrected count is the **opposite**: read-only cache is used in
**8 of 8** sampled kernels — 7 via `.CONSTANT` loads, and flash via the superior
cp.async path that needs no read-only hint.

This is a false absence produced by a correct command answering an adjacent question —
the sixth such instance in a single working session, and the only one that would have
cost weeks rather than minutes. It was caught mid-survey by verifying the actual load
notation in the SASS (`cuobjdump -sass | grep -oE 'LDG[.A-Z0-9]*' | sort | uniq -c`)
before trusting the count. **A survey that records the instrument error it survived is
worth more than one that presents only the corrected table.**

## Method — ship-exact, and validated as such

- **Source of truth:** the compiled objects in `target/.../baracuda-kernels-sys-*/out`
  (already built with the shipping flags), read with `cuobjdump -sass` and
  `cuobjdump -res-usage`; plus recompiles to `-ptx` and `-cubin -Xptxas -v` to read the
  PTX substrate and the authoritative register/spill report.
- **Ship-exact flags validated, not assumed:** the shipping build's default compute
  capability on this host is `sm_80` (forge's fallback cap; the build did not target
  the host's own `sm_89`). A recompile of `binary_add_fp.cu` with the reconstructed
  flags (`-gencode arch=compute_80,code=sm_80 --default-stream per-thread -std=c++20
  -D_USE_MATH_DEFINES -Ikernels/include`) reproduced the shipped object
  **bit-for-resource** (strided-`double` kernel: 38 registers, 224-byte stack frame),
  confirming the reconstructed invocation is the one that ships. Without this check
  every number would be about a configuration nobody ships.
- **Sample:** 9 kernels spanning shapes known to differ — elementwise, reduction
  (softmax), int8 tiled GEMM, dense GEMV, flash-attention backward, arbitrary-mask
  attention, dequantize, topk. This is **9 of 426**; absences are reported as
  "N of 9 sampled," never "the kernels don't use it."

## Per-kernel lever inventory (sm_80, ship-exact)

Counts are occurrences in the object's SASS unless marked *(PTX)*. `readonly` =
`.CONSTANT` loads (SASS) / `ld.global.nc` (PTX). `worst REG` / `frame` / `shared` from
`cuobjdump -res-usage` (worst function in the object).

| kernel (shape) | worst REG | frame B | shared B | readonly loads | vector loads (.128/.64) | vector stores (.128) | cp.async (LDGSTS) | tensor core |
|----------------|-----------|---------|----------|----------------|-------------------------|----------------------|-------------------|-------------|
| binary_add (elementwise) | 42 | 224–288 | 0 | 216 / 216 | 0 / 16 | 0 | 0 | — |
| gumbel_softmax (reduction) | 46 | 288 | 0 | 542 / 542 | 0 / 306 | 0 | 0 | — |
| gemm_s8_rrr (int8 tiled GEMM) | 96 | 0 | 4096 | 128 / 128 | 0 / 0 | 0 | **0** | IMMA `m16n8k32.s8` (16) |
| gemv_dense (memory-bound GEMV) | 32 | 0 | 0 | 186 / 192 | 0 / 0 | 0 | 0 | — |
| flash_bwd hdim128 bf16 (attention) | **255** | 176 | dyn | 0 | 32 / 16 | **528** | **672** | HMMA (5760) |
| attn_arbmask (attention) | 64 | 0 | 256 | 88 / 88 | 0 / 7 | 0 | 0 | — |
| dequantize (quant) | 29 | 0 | 0 | 92 / 92 | 0 / 0 | 0 | 0 | — |
| topk (sort) | 32 | 0 | 0 | 32 / 32 | 0 / 11 | 0 | 0 | — |

PTX-substrate spot-checks (what a PTX importer would actually read):

- `binary_add.ptx`: **120 `ld.global.nc`, 0 plain `ld.global`, 0 `ld.global.v2/v4`,
  0 `cp.async`, 0 `mma`.** The read-only hint is in the PTX; vectorization is not.
- `gemm_s8.ptx`: **16 `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32`,
  0 `cp.async`, 128 `ld.global.nc`, 0 `ld.global.v2/v4`, 40 `ld.shared` / 128
  `st.shared`.** Tensor core + read-only cache in the PTX; the global→shared staging is
  synchronous (`ld.global.nc`→`st.shared`), not `cp.async`.

Read across the rows:

- **Read-only cache:** 8/8. Captured, and captured *in the PTX*.
- **Vectorization:** present only in flash (`LDG/STG/LDGSTS .128`); scalar in the other
  7, including the memory-bound GEMV (all `.CONSTANT` scalar `u16`/`f32`) and the
  elementwise/reduction kernels. This is the one importer-addressable gap — see the crux.
- **cp.async:** present only in flash (672). The int8 GEMM stages synchronously.
- **Tensor cores:** shape-appropriate — present in the two matmul kernels, correctly
  absent in the six non-matmul kernels.

## Occupancy / `ptxas -v` — and a frame that is not a spill

`ptxas -v` on the recompiled elementwise kernels reports, for every strided variant,
**"224 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads."** The stack frame
is the by-value `Dims<32>`/`Dims<64>` stride structs passed in local memory for the
general strided-addressing path — **not register spilling.** The `LDL`/`STL` traffic in
the SASS is the addressing code reading those structs, not spill traffic. `ptxas -v`
showing a frame is not the same fact as spilling, and the two are one glance apart; the
first read nearly mislabeled it.

The one genuine register-pressure case is **flash_bwd, capped at 255 registers** — the
classic flash-backward occupancy limiter. That is real headroom, but it is **ptxas's**
(register allocation), and a PTX importer re-runs ptxas; it cannot control the
allocation. The fix for flash-bwd occupancy is algorithmic or launch-bounds, not a PTX
rewrite. (flash's shared memory is dynamic `extern __shared__`, so it reads as 0 in the
static `res-usage` field despite heavy `LDS`/`STS` — noted so the 0 is not misread.)

## The crux — why the answer is null-leaning

The importer reads nvcc's PTX, which is **post-optimization output.** So:

- For the levers nvcc **applied** (read-only cache), they are already in the PTX. There
  is nothing for the importer to add.
- For the lever nvcc **declined** (vectorization), nvcc emitted scalar loads having seen
  the C++ types, `__restrict__`, alignment, and the access pattern — and chose not to
  widen. The importer sees only the lowered scalar PTX, where types are `.u16`/`.f32`
  and pointer alignment hints are mostly gone. It therefore has **strictly less basis to
  safely vectorize than nvcc had**, and it sits downstream of the very decision it would
  reverse. "nvcc declined" is evidence *against* the transform being free, not for it.

The reason is the same in every row, which is why a wider sample would not move the
conclusion: more kernels already emitting `ld.global.nc` does not change it, and more
kernels lacking `.v4` does not either — the cause is representational, not a count.

## What survives as real headroom

**The int8 GEMM's `cp.async` gap.** It stages global→shared synchronously
(`ld.global.nc`→`st.shared`) where flash uses `cp.async` (`LDGSTS`) to overlap the load
with compute. That is genuinely absent and genuinely headroom — flash proves the pattern
is available when hand-authored. But converting it is an **algorithmic pipeline rewrite**
(commit/wait barriers, double-buffering), not a peephole — exactly the category a naive
emitter handles badly. It is a per-kernel hand-optimization, not systematic capture.

## Limits (unsoftened)

- **9 of 426 sampled.** Absences above are "N of 9," not "the kernels don't."
- **sm_80, not sm_89.** The shipping build capped at `sm_80`; the host is `sm_89`. The
  `sm_89`-specific mma/fp8 shapes are **unmeasured**, not measured-and-absent. This is the
  one extension that could matter, and only if someone builds on those shapes.
- **Static survey, not a benchmark.** A scalar load or a sync-staged GEMM is a
  *candidate*, not a proven speedup.
- **Vectorization safety not verified.** The survey establishes that nvcc *declined* to
  vectorize, not that widening would be *safe* (alignment/aliasing) for any kernel.

## Reproduce

```bash
# 1. Locate a populated ship-exact object dir (built with the shipping flags).
DIR=$(for d in target/debug/build/baracuda-kernels-sys-*/out; do \
        echo "$(find "$d" -maxdepth 1 -name '*.o' | wc -l) $d"; done | sort -rn | head -1 | awk '{print $2}')

# 2. Ship-exact SASS + resource usage per kernel (no recompile).
cuobjdump -sass       "$DIR"/binary_add_fp-*.o
cuobjdump -res-usage  "$DIR"/binary_add_fp-*.o

# 3. Read the ACTUAL load notation before counting a lever (the .CI/.CONSTANT lesson).
cuobjdump -sass "$DIR"/gemv_dense-*.o | grep -oE '\bLDG[.A-Z0-9]*' | sort | uniq -c | sort -rn

# 4. PTX substrate + authoritative spill report (validate flags reproduce the ship object).
cd crates/baracuda-kernels-sys
nvcc -ccbin "<VS-MSVC>" -gencode arch=compute_80,code=sm_80 --default-stream per-thread \
     -std=c++20 -D_USE_MATH_DEFINES -Ikernels/include \
     -ptx        kernels/elementwise/binary_add_fp.cu -o binary_add.ptx
nvcc -ccbin "<VS-MSVC>" -gencode arch=compute_80,code=sm_80 --default-stream per-thread \
     -std=c++20 -D_USE_MATH_DEFINES -Ikernels/include \
     -cubin -Xptxas -v kernels/elementwise/binary_add_fp.cu -o binary_add.cubin
```

`lever_sweep.sh` in this directory runs the full per-kernel sweep.

## Files

- `README.md` — this survey.
- `lever_sweep.sh` — the SASS lever + occupancy sweep over the ship objects (uses the
  corrected `.CONSTANT`/`.CI` read-only pattern).
