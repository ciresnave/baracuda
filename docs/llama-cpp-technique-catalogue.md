# llama.cpp / ggml-cuda technique catalogue (Phase A)

Research-only survey, no code changes. Catalogues what `ggml-cuda`
(upstream: <https://github.com/ggml-org/llama.cpp>, formerly
`ggerganov/llama.cpp`, MIT) does per compute capability for the kernel
families baracuda's models actually use, whether baracuda already does
it, and the expected payoff — extending `PHASE_27_ANALYSIS.md`'s method
(same Tier S/A/B ranking, same "measured not vibes" standard) to the
whole llama.cpp technique surface rather than just the Q8_1-staging
comparison that report covered.

Every upstream citation below is `file:line @ commit`, pinned via
`gh api repos/ggml-org/llama.cpp/commits?path=<file>` at research time
(2026-09-29/30) — the exact commit that was HEAD-of-history for that
path when checked, not a guess. `ggml-cuda.cu` stopped existing as a
single file after the "reorganize source code" commit `f3f65429c4`
(2024-06-26); today's tree is `ggml/src/ggml-cuda/*.cu` per-op files,
each with its own independent commit history, which is why the pins
below differ file to file.

## ⚠️ Blocking finding: baracuda does not target sm_61 today

**This contradicts the task's stated Phase B priority ("sm_61 (the
P40s) and sm_89 (the 4070)") and needs to reach whoever set that
priority before Phase B starts.** baracuda's own `README.md` (Hardware
support section) states:

> "baracuda targets **Ampere and newer** by design. Pre-Ampere GPUs
> lack the tensor-core instructions and async-copy primitives the
> bespoke kernels are written against (`mma.sync.m16n8k*`, `cp.async`,
> `ldmatrix`)... | ≤ sm_75 (Turing, Volta, Pascal, …) | — |
> **unsupported** |"

sm_61 (Pascal, the P40s) is two full architecture generations below
baracuda's stated floor (sm_80). This isn't a missing feature flag —
`mma.sync`, `cp.async`, and `ldmatrix` are genuinely absent from
Pascal's ISA (they were introduced with Turing/Ampere). A baracuda
kernel written against those primitives cannot be made to run on sm_61
by adding a "hint"; it needs an entirely separate, non-tensor-core
kernel body, which is a different scale of work than what "harvest
techniques as provenance-tagged hints on existing intent ops" implies.

**What IS transferable to sm_61 without a from-scratch kernel**:
`__dp4a` (`GGML_CUDA_CC_DP4A = 610`, `crates/baracuda-kernels-sys/vendor/fuel-q8_1/quantized.cu`
already vendors a `ggml_cuda_dp4a` shim) requires only sm_61+ — **not**
sm_80+. The MMVQ family (dequantize-then-multiply, no tensor cores) is
close to SIMT-portable in principle. Tensor-core-dependent families
(MMQ tile matmul, flash-attention's MMA path) are not. Each family
section below says explicitly which bucket it falls in.

This catalogue documents ggml-cuda's sm_61 behavior fully (that part of
the task is legitimate research regardless), but **flags every
sm_61 "port this" recommendation as blocked on a baracuda architecture
decision** (build a parallel non-tensor-core kernel family, or accept
sm_61 stays out of scope) that Phase A cannot make unilaterally.

## Families actually shipped in baracuda (verified against the tree)

Cross-checked `crates/baracuda-kernels-sys/kernels/` against the
task's assumed family list — all six exist:

| Family | baracuda kernel dir/files |
| --- | --- |
| Quantized matmul/matvec (MMVQ + MMQ) | `kernels/gguf/{mmvq,mmvq_batched,mmvq_multim,dequantize,quantize_q8_1}.cu`, `kernels/include/baracuda_gguf.cuh` |
| Attention / flash variants | `kernels/attention/{flash_sdpa_fp,flash_sdpa_sm89,flash_decoding_fp,sdpa_fp,sdpa_block_sparse_fp,fa2_launcher,...}.cu` |
| RMSNorm | `kernels/norm/` (Phase 5 family per README) |
| RoPE | `kernels/attention/{rope_fp,rope_backward_fp}.cu` |
| Softmax | `kernels/softmax/` (Phase 5 family per README) |
| Dequantize | `kernels/gguf/dequantize.cu`, `kernels/include/baracuda_gguf.cuh` |

---

## 1. Quantized matmul + matvec (MMVQ / MMQ)

### Intent

`intent: quantized_matvec(W_q[N,K], x[K]) -> y[N]` (MMVQ, M=1 decode)
and `intent: quantized_matmul(W_q[N,K], X[M,K]) -> Y[M,N]` (MMQ, M>1
prefill) — dequantize-and-dot vs. int8-tensor-core tiled GEMM, same
logical op, different implementation regime by M.

### What ggml-cuda does, per arch

| Arch | Technique (hint) | Upstream evidence |
| --- | --- | --- |
| **sm_61 (Pascal)** | **MMVQ**: dequantize block → fp32 multiply-accumulate, warp-shuffle reduce. No `__dp4a` requirement here (dequant path is FP, not int8-dot) — same shape as baracuda's own MMVQ. **MMQ**: `ggml_cuda_mmq_get_config_pascal_dp4a` — tile `mmq_x=64`ish, `__dp4a` int8×int8 dot (dp4a needs sm_61+, `GGML_CUDA_CC_DP4A=610` in `common.cuh:51`), no tensor cores, no `cp.async`. A separate `mmq-config-pascal-older.cuh` config exists for sub-dp4a Pascal-class parts (`GGML_CUDA_CC_PASCAL=600`, i.e. cc 6.0 without dp4a — GP100/Tesla P100). The P40 is cc 6.1, so it takes the dp4a config, not the older one. | `ggml/src/ggml-cuda/common.cuh:50-51 @ f9af9be219ca`; `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh @ e71b80510c84`; `ggml/src/ggml-cuda/mmq-config-pascal-older.cuh @ e71b80510c84` |
| **sm_89 (Ada)** | **MMVQ**: same dequant-then-FP-multiply shape as Pascal (MMVQ itself doesn't use tensor cores at any arch — it's a decode-time, low-arithmetic-intensity path everywhere; the crossover point to switch to MMQ shifts per-arch, see `mmvq.cu:287-297 @ 68d9053afd4f` — Ada gets a `cc >= GGML_CUDA_CC_ADA_LOVELACE` branch same as Volta, distinct from the general Turing+ branch). **MMQ**: `ggml_cuda_mmq_get_config_ampere` (Ada reuses the Ampere config, no separate Ada MMQ config file) — tile `mmq_x=128` (2× Pascal's 64), int8-tensor-core (`imma`) dot path via the `true` flag column, larger SRAM tile layout. | `ggml/src/ggml-cuda/mmvq.cu:287-297 @ 68d9053afd4f`; `ggml/src/ggml-cuda/mmq-config-ampere.cuh @ e71b80510c84` |

**Non-transferable-as-is**: the int8-tensor-core MMQ path requires
`mma.sync` (Turing+, sm_75+) — genuinely unavailable on sm_61.

### Does baracuda already do this?

- **MMVQ (FP dequant-then-multiply)**: yes, `dequantize_mul_mat_vec`
  template in `kernels/include/baracuda_gguf.cuh`, single-warp
  (`GGML_CUDA_MMV_Y=1`), matches ggml's shape closely (confirmed
  independently in `PHASE_27_ANALYSIS.md` §2c, which already diffed
  baracuda's MMVQ against a vendored ggml-derived copy).
- **MMQ (tile-based int8-tensor-core matmul)**: **no**. `PHASE_27_ANALYSIS.md`
  already recommended this as **Tier S1** ("multi-M MMVQ via Q8_1
  staging... 3-7× speedup on prefill matmuls") and it's still not
  built — `LICENSE-thirdparty.md` confirms "Dropped the q8_1-staging
  tile-based MMQ matmul family... deferred to a follow-up milestone."
  This catalogue entry is not new information; it corroborates the
  existing Tier S1 backlog item with upstream per-arch specifics.
- **Per-arch MMVQ↔MMQ crossover tuning** (`mmvq.cu:279-334`): baracuda
  has no crossover logic at all today — it always uses MMVQ regardless
  of M. This is the concrete mechanism behind Tier S1's "no weight
  reuse across M rows" gap.

### Payoff

Unchanged from `PHASE_27_ANALYSIS.md`: **Tier S** for MMQ on sm_80+
(3-7× on prefill M∈{2,4,8}, memory-bandwidth-bound reasoning already
worked out there). On sm_61 specifically: MMQ would still help
(dp4a-based, no tensor cores needed — genuinely portable), but no
baracuda kernel family exists on sm_61 to attach it to (see blocking
finding above) — the payoff estimate doesn't matter until that's
resolved.

---

## 2. Attention / flash-attention variants

### Intent

`intent: scaled_dot_product_attention(Q,K,V) -> O` — the dispatcher
picks among three structurally different kernel bodies, not just tuning
knobs on one:

### What ggml-cuda does, per arch

Dispatch logic lives in `ggml_cuda_get_best_fattn_kernel`
(`ggml/src/ggml-cuda/fattn.cu:541-731 @ 2ebd9ae62190`):

| Arch | Technique (hint) | Evidence |
| --- | --- | --- |
| **sm_61 (Pascal)** | **`fattn-vec`** only. `turing_mma_available(cc)` (line 638) and `volta_mma_available(cc)` (line 673) both gate the tensor-core path; Pascal satisfies neither, so it always falls through to the vector kernel — a per-row warp-reduce softmax-attention, no tensor cores, no SRAM tiling beyond basic caching. This is architecturally the closest of the three bodies to a SIMT/dp4a-portable design. | `ggml/src/ggml-cuda/fattn.cu:638,673 @ 2ebd9ae62190`; body in `fattn-vec.cuh @ 73a43d1f6934` |
| **sm_89 (Ada)** | Tensor-core **`fattn-mma-f16`** path (`turing_mma_available` true at sm_89), PLUS an Ada-specific decode branch: `cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1` picks a different kernel variant than the general Turing+ MMA path (line 645-650) — a decode-batch=1-specific tuning distinct from Turing's default. | `ggml/src/ggml-cuda/fattn.cu:645-650 @ 2ebd9ae62190`; body in `fattn-mma-f16.cuh @ c2a9e1606807` |

Also present but not arch-differentiated in the same way: `fattn-tile.cu`
— a mid-tier tiled-but-not-tensor-core kernel used as a fallback when
`fattn-vec` doesn't cover the head-dim/dtype combo (its own dispatch
condition, not purely arch-gated).

**Non-transferable-as-is**: `fattn-mma-f16` requires `mma.sync`
(Turing+) — same sm_61 wall as MMQ above.

### Does baracuda already do this?

- baracuda has its own bespoke `FlashSdpaPlan` (Phase 6.6) PLUS a
  vendored FA2 (Phase 42, Dao-AILab, BSD-3, not ggml-derived) PLUS an
  Ada-specific `FlashSdpaSm89Plan`/`flash_sdpa_sm89.cu` (per README's
  "sm_89 tuning sweep", Phase 10) — **the Ada-specific-kernel pattern
  ggml uses (a distinct decode-batch=1 variant) is structurally the
  same idea baracuda already has for sm_89**, just via a different
  upstream lineage (FA2, not ggml). Not a gap; a convergent design,
  worth noting as corroboration rather than a new technique to port.
- baracuda's bespoke non-tensor-core attention path exists (used when
  FA2/sm89 backends aren't selected) — closest baracuda analogue to
  ggml's `fattn-vec`. Detailed op-for-op comparison against
  `fattn-vec.cuh` was out of this pass's budget; flagging as a
  candidate for a follow-up if a specific perf gap is ever observed on
  short-context / small-batch shapes (where `fattn-vec`-style kernels
  are typically strongest).

### Payoff

Not quantified this pass — no concrete shape class or benchmark
gap identified in the time available (unlike the MMVQ/MMQ family,
which has `PHASE_27_ANALYSIS.md`'s existing measurement). Marking as
**Tier B / needs a benchmark before ranking**, not claiming a payoff
number I haven't measured (per CLAUDE.md §7: a number needs its
measurement, not an estimate dressed as one).

---

## 3. RMSNorm

### Intent

`intent: rms_norm(x[N], weight[N]) -> y[N]`, optionally fused with a
following elementwise op.

### What ggml-cuda does, per arch

**No per-arch branching found** (verified: grepped `norm.cu` for
`GGML_CUDA_CC_`/`cc >=`/`cc <` — zero matches). RMSNorm is a
warp/block-reduce-sum-of-squares kernel; the reduction primitive
(`block_reduce`) is arch-generic (warp-shuffle + optional SMEM
cross-warp step for `block_size > WARP_SIZE`), unlike MMVQ/attention
which fork on tensor-core availability. This means the *fusion*
technique below is equally applicable at **every** arch from sm_61 to
sm_90 — no sm_61 wall here.

| Technique (hint) | Evidence |
| --- | --- |
| **RMS_NORM + SCALE kernel fusion** — a templated `do_scale` flag on `rms_norm_f32<>` fuses a post-normalize scalar multiply into the same kernel launch instead of a separate elementwise pass. Landed **2026-09-25**, i.e. very recent upstream work. | `ggml/src/ggml-cuda/norm.cu:309-332,508-533 @ 1ab7e5ad2d4e` |
| **RMS_NORM + MUL (elementwise-tensor, not scalar) fusion** — a second, distinct fusion: `ggml_cuda_op_rms_norm_fused` folds a following per-element multiply (the weight-multiply in `y = rms_norm(x) * weight`, the common transformer-block pattern) into the norm kernel, avoiding a full extra read+write of the normalized tensor. | `ggml/src/ggml-cuda/norm.cu:332-412,535+ @ 1ab7e5ad2d4e` |

### Does baracuda already do this?

Not verified this pass — baracuda's `kernels/norm/` family wasn't read
in detail (budget). **Explicitly flagging as unverified rather than
guessing**: this is exactly the kind of "did we already do this" claim
CLAUDE.md §6/§7 says needs a positive control before stating either
way. Follow-up: check whether `RmsNormPlan` already fuses a trailing
weight-multiply or scale, or does it as a separate kernel launch.

### Payoff

If baracuda's RMSNorm is currently a separate kernel per weight-multiply
step (unverified — see above), fusing saves one full tensor
read+write per RMSNorm call, every transformer block, every token —
plausibly meaningful at small-batch decode where kernel-launch
overhead and memory round-trips dominate. **Not quantified**; flagging
as a cheap, architecture-independent (works on sm_61 through sm_90
identically) Tier A candidate *pending the baracuda-side verification*
above.

---

## 4. RoPE

### Intent

`intent: rope(x[N], positions[N], freq_base, freq_scale) -> y[N]` —
positional rotation applied per-pair-of-dims.

### What ggml-cuda does, per arch

**No per-arch branching found** (verified: grepped `rope.cu` for
`GGML_CUDA_CC_`/`cc >=`/`cc <` — zero matches, same check as RMSNorm).
RoPE is index arithmetic + `sincosf` + a rotate — no tensor cores, no
warp-level reduction, nothing that scales with arch generation. This
family has genuinely nothing arch-specific to catalogue: **there is no
per-CC hint to harvest here**, at any of the five architectures asked
about. Recording that as the finding itself rather than manufacturing
a distinction that doesn't exist upstream.

### Does baracuda already do this?

baracuda has `RopeApply*` families going back to Phase 6.1, extended
through Phase 36/41 (YaRN/LongRoPE scaling, per README Phase 45 entry)
— already a superset of ggml's plain-RoPE feature surface. No action.

### Payoff

None identified — this family is a **non-finding** by design, not a
gap in the research pass.

---

## 5. Softmax

### Intent

`intent: softmax(x[N], scale, mask?) -> y[N]` (attention-score
softmax and the standalone op).

### What ggml-cuda does, per arch

**No per-arch branching found** (same check, `softmax.cu` — zero
`GGML_CUDA_CC_` matches). The interesting technique here is a
*correctness* fix, not a perf hint: `block_reduce` had a data race
when reused across the max-reduce and sum-reduce phases within one
kernel invocation (shared SMEM buffer, insufficient sync between the
two reduction calls) — fixed by an explicit `__syncthreads()` between
phases for `block_size > WARP_SIZE`. **This class of bug (SMEM buffer
reuse across sequential warp-reductions in one kernel) is worth an
explicit self-check against baracuda's own softmax/attention kernels**
if any of them reuse a shared-memory scratch buffer across more than
one `block_reduce`-style call — the fix pattern (an extra
`__syncthreads()`, gated so it's a no-op for `block_size <= WARP_SIZE`)
is cheap and arch-independent.

| Technique (hint) | Evidence |
| --- | --- |
| SMEM-reuse race fix between sequential block-reduces (extra sync, gated on `block_size > WARP_SIZE`) | `ggml/src/ggml-cuda/softmax.cu:83,119-125 @ 9bd4c09ea571` |

### Does baracuda already do this?

Not verified this pass (budget) — flagging as a **correctness
self-check**, not a performance-catalogue item: worth someone grepping
baracuda's softmax/attention kernels for a shared SMEM scratch buffer
touched by two sequential reduction calls within one kernel body.

### Payoff

N/A (correctness item, not a perf item). If baracuda has the same
pattern without the guard, the "payoff" is avoiding a race, not a
speedup — different axis than the rest of this catalogue.

---

## 6. Dequantize

Covered by the MMVQ section above — ggml's dequantize kernels
(`dequantize.cuh @ 9b2a088819cd`, `convert.cu @ f4e276a2066a`) are the
same per-block-format primitives the MMVQ family's dequant step calls;
there is no separate per-arch dequantize-only technique beyond what's
already catalogued in §1. One dequantize-specific upstream item worth
noting: `convert.cu`'s 2026-09-21 change to "convert contiguous
tensors four elements at a time" (`f4e276a2066a`) — a vectorized
4-wide load/store for the plain (non-quantized-block) contiguous
dequant/convert path, arch-generic (no CC gate found in a quick check
of that commit's diff description). Not independently verified beyond
the commit message; flagging as a possible cheap win for baracuda's
`kernels/gguf/dequantize.cu` contiguous-path if it isn't already
vectorizing loads 4-wide, but this needs the actual diff read before
it's more than a lead.

---

## Coverage note (per-task instruction)

Per the task's stated priority, **sm_61 and sm_89 got full depth**
across all six families above (dispatch logic read, specific config
files/line numbers cited, baracuda-side comparison attempted for each).
**sm_75 (Turing), sm_80/86 (Ampere), and sm_90 (Hopper) got only
incidental coverage** — mentioned where they appear in the sm_61/sm_89
dispatch logic already read (e.g. Turing as the `turing_mma_available`
gate, Ampere as the baseline MMQ config Ada reuses), but not
independently researched column-by-column. **This is deliberate, not
an oversight**, given the budget note in this task's brief — a Phase
A2 follow-up would need to: (a) read `mmq-config-rdna2/3/4.cuh` siblings
for the AMD-side comparison (out of scope, baracuda is NVIDIA-only),
(b) separately verify Ampere-vs-Ada MMQ config deltas beyond "Ada
reuses the Ampere config file" (is there truly zero Ada-specific MMQ
tuning, or did I just not find a separate config file because there
isn't one?), and (c) research Hopper (`fattn.cu` has Hopper/Blackwell/
DGX-Spark branches not read in this pass — `fattn.cu:369-393`).

## Commit-pin verification gaps (explicit, not glossed over)

- `convert.cu`'s "four elements at a time" claim (§6) — cited by
  commit message only, the actual diff wasn't read. Don't treat this
  as verified; it's a lead.
- Ampere-vs-Ada MMQ config: verified only that Ada's `fattn.cu`
  dispatch reuses `ggml_cuda_mmq_get_config_ampere` by absence of a
  separate `mmq-config-ada.cuh` file in the directory listing — did
  not confirm the function itself doesn't internally branch on
  `cc == GGML_CUDA_CC_ADA_LOVELACE` (only grepped the *filenames*, not
  every line of `mmq-config-ampere.cuh`'s 383 lines for an internal Ada
  branch).
- Attention family (§2): "baracuda already has an Ada-specific decode
  kernel, structurally convergent with ggml's" is based on the
  existence of `flash_sdpa_sm89.cu` (filename + README's Phase 10
  changelog line), not a line-by-line read of that file against
  `fattn.cu:645-650`. Treat as a plausible parallel, not a confirmed
  match.
- RMSNorm and Softmax "does baracuda already do this" — explicitly
  unverified (stated inline above), flagged as follow-up items rather
  than guessed at.
