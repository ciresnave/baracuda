# Third-party provenance — audit of vendored and derived files

Audit of every third-party-derived file in this repository, per CireSnave's provenance rule
(`C:\Projects\CIRESNAVE-EXPECTATIONS.md` §6.4a, 2026-09-17): trace each file to its **origin**, verify
our copy against it, keep the origin's licence, notices and credit, and mark the file with the origin's
licence. Measured at `origin/main` `6b39f550` (alpha.100), 2026-10-14. PM-tasked; **no vendored file,
licence file or source header was changed by this PR** (those are listed in §4 for a PM decision).
Only baracuda-authored documentation (`VENDOR.md` files, `LICENSE-thirdparty.md`) was corrected.

**Method** (reusable by other lanes: [`docs/provenance/METHOD.md`](docs/provenance/METHOD.md)). Files enumerated with `git ls-files`. For each, our blob (`git show HEAD:<path>`, LF) was
compared with the upstream blob at the pinned commit in a `--filter=blob:none` clone: byte-identical /
modified (hunks, +/- lines) / hand-port (line-overlap test; positive control: a known-verbatim file
scored 104/104 shared lines, the hand-ports 0) / no upstream. Licence files were compared by sha256.
Per-file evidence: `docs/provenance/*.tsv` (4 files). **Limits:** line overlap is the method for
"derived-ness" of adapted kernels, with no per-hunk legal review; upstream `HEAD` was used where no
commit was recorded; llama.cpp's earliest `ggml-cuda.cu` ancestor is dated only approximately.

## 1. Inventory

| tree | origin | licence | pinned? | our files vs origin |
|---|---|---|---|---|
| `kernels-sys/vendor/flash-attention` (86) | Dao-AILab/flash-attention v2.8.3 `060c9188`; hd160/224/512 fwd from EricLBuehler/candle `132643dc` / huggingface/candle `5430d32c` | BSD-3-Clause (FA); candle MIT OR Apache-2.0 | FA yes, candle partly | 66 identical, 15 modified (12 candle `.cu` +5 lines, 3 headers), 5 baracuda-authored (VENDOR.md, shim) |
| `vendor/flashinfer` (30) | flashinfer-ai/flashinfer `eee0d75f` (a `main` commit; **not** tag v0.6.12) | Apache-2.0 + NOTICE | yes | 22 identical (LICENSE+NOTICE match), 7 modified with in-file change notices, 1 authored |
| `vendor/awq` (8) | mit-han-lab/llm-awq `d6e797a4` | MIT | yes | LICENSE + `dequantize.cuh` identical; `gemm_cuda_gen.cu` modified (-68/+20, stated in-file); shim authored |
| `vendor/marlin` (4) | IST-DASLab/marlin `1f25790b` | Apache-2.0 | yes | LICENSE + kernel identical |
| `vendor/mhc` (11) | AndreSlavescu/mHC.cu `a426939c` | MIT | yes | LICENSE identical; 8 bodies identical with a baracuda header prepended |
| `vendor/bitsandbytes` (5) | bitsandbytes-foundation | MIT | **no** | `nf4_kernel.cuh` hand-port (only the 16 QLoRA codebook constants match); `nf4_gemv.cuh` authored; LICENSE reworded |
| `vendor/causal-conv1d` (3), `vendor/mamba` (3), `vendor/xformers` (3) | Dao-AILab / state-spaces / facebookresearch | BSD-3 / Apache-2.0 / BSD-3 | **no** | attribution only; kernels (under `kernels/`) share 0 lines with upstream; LICENSE files reworded |
| `optim/vendor/apex` (7) | NVIDIA/apex | BSD-3 | **no** | 4 `.cuh` are PyTorch-free rewrites (<= 9 shared lines); LICENSE reformatted |
| `te-sys/vendor/transformer-engine` (1) | NVIDIA/TransformerEngine | Apache-2.0 | **no** | no TE source copied; `ATTRIBUTION.md` embeds a non-verbatim Apache text |
| `vendor/fuel-q8_1` (6) | Fuel `e1070acf9` = candle `fd8448d2` <- llama.cpp ggml-cuda (MIT) + guoqingbao/vllm.rs "xinfer" via candle#3196 | MIT; xinfer licence carried | no | `quantized.cu` differs from Fuel only in header comments; LICENSE-llama.cpp / LICENSE-xinfer identical |
| `kernels/` adapted (15 listed + 8 unlisted) | Fuel/candle (cast, fill, affine, moe, gguf), llama.cpp (gguf, sort, mmvq q8_1), attention.rs (MoE), Ootomo cutf, deepreinforce-ai/CUDA-L2 | MIT | partly | see §3 |
| `baracuda-forge` | guoqingbao/cudaforge `aa77db3` | MIT OR Apache-2.0 (Cargo.toml only) | yes | 1 file code-identical, 7 modified |

Chain of custody for the Fuel-routed kernels (verified by `git hash-object` equality): Fuel is a candle
fork (root `fe7a5f53` = candle "Initial commit"); the last Fuel blob of `affine/cast/fill` equals candle
`af5a69ee`, `quantized.cu`/`gguf.cuh` equal candle `fd8448d2`, `moe_gguf.cu`/`moe_utils.cuh`/
`moe_wmma_gguf.cu` equal candle `f2d5aabb`, `moe_wmma.cu` equals candle `a0dbd8b8`. Candle's own
headers name llama.cpp ggml-cuda and guoqingbao/attention.rs as the origins.

## 2. What holds (so it is not re-audited)

- Every vendored `LICENSE`/`NOTICE` that claims to be verbatim **is** byte-identical for FA2, FlashInfer
  (LICENSE+NOTICE, also equal to the tag), AWQ, Marlin, mHC, llama.cpp (`LICENSE-llama.cpp`), xinfer.
- Origin copyright headers are retained in every FA2, FlashInfer and AWQ file that had one.
- FlashInfer's seven edits and AWQ's edits carry in-file change comments (Apache-2.0 §4(b) met).
- Pins resolve upstream: FA `060c9188`, candle `5430d32c`/`2ce5f125`/`71cd6d55`, AWQ `d6e797a4`,
  Marlin `1f25790b`, mHC `a426939c`, FlashInfer `eee0d75f`, cudaforge `aa77db3`, CUDA-L2 `dbe0177`.
- `baracuda-cuda-parse` is first-party (moved from Unpopped, same author/licence); `baracuda_flce.cuh`
  and `loss/flce_fp.cu` credit Liger for the algorithm only and copy no source.

## 3. Gap register

Legend: **FIXED** = corrected in this PR (documentation only). **PM** = touches a vendored file, a
licence text or a source header, so it needs a PM decision (§4).

### Licence / notice text missing or not verbatim

| id | gap | status |
|---|---|---|
| G-FA-1 | Candle's licence (MIT OR Apache-2.0) and notice are not carried in `vendor/flash-attention`; the 12 candle-sourced `.cu` files carry Tri Dao's BSD-3 header only | PM |
| G-Q8-1 | `vendor/fuel-q8_1` credits llama.cpp and xinfer but **nowhere names candle** and carries no candle licence, though `quantized.cu` is a candle blob | PM |
| G-K-1 | No MIT/Apache text for candle, ggml (`Copyright (c) 2023-2026 The ggml authors`) or attention.rs outside `vendor/`; `LICENSE-llama.cpp` exists but the md does not reference it | PM |
| G-LIC-1 | `LICENSE` not byte-identical to upstream (VENDOR.md says verbatim): bitsandbytes (copyright line changed from Facebook to Dettmers), causal-conv1d (copyright line changed), mamba (Apache text reworded), xformers (preamble dropped, reformatted), apex (reformatted) | PM |
| G-LIC-2 | `AUTHORS` in causal-conv1d and mamba is rewritten prose, not upstream's | PM |
| G-TE-1 | `transformer-engine-sys/ATTRIBUTION.md` embeds an Apache-2.0 text that is not TE's `LICENSE` (190 vs 175 lines); the crate `LICENSE-APACHE` also differs | PM |
| G-BNB-1 | bitsandbytes `NOTICE.md` (BSD note re PyTorch portions) not carried | PM |
| G-FI-1 | `fp16.h` is MIT (Dukhan 2017-2024 / AMD 2025); only its SPDX line is carried and the NOTICE points at an upstream `licenses/` dir that is not vendored | PM |
| G-CUTF-1 | `kernels/include/baracuda_cp_async.cuh`, `baracuda_fp_bits.cuh` derive from Ootomo's `cutf` (MIT, "Copyright (c) 2018 mutsuki"): no SPDX header, no listing, no licence text anywhere | PM |
| G-L2-1 | `kernels-bench/external/cuda-l2-probes/*.cu` (5) are 57-78% line-overlapping with deepreinforce-ai/CUDA-L2 (MIT, (c) 2025 deepreinforce.ai); three have no licence mention; the README says `LICENSE.txt` lives in `external/cuda-l2/`, which is untracked | PM |
| G-FORGE-1 | `baracuda-forge/NOTICE` says licences are "reproduced verbatim", but root/forge `LICENSE-MIT` carry only the project line and no Guoqing Bao text; cudaforge itself has no LICENSE file (Cargo.toml only) | PM |

### Headers that misstate origin

| id | gap | status |
|---|---|---|
| G-H-1 | Six cast/fill/affine files say "Copyright 2024 Eric Holscher and the candle / fuel-cuda-kernels contributors": the name appears nowhere else and candle's licence names no holder | PM |
| G-H-2 | `baracuda_moe.cuh` and `moe_*.cu` label attention.rs "MIT OR Apache-2.0"; its `Cargo.toml` says `license = "MIT"` and it ships no LICENSE file | PM |
| G-APEX-2 | Apex `.cuh` files carry an NVIDIA BSD notice that upstream does not have on those (rewritten) files | PM |
| G-SORT-1 | `kernels/sort/sort.cu`, `baracuda_sort.cuh`: Fuel's `sort.cu` is "Adapted from llama.cpp ggml-cuda/argsort.cu" and equals candle HEAD; no SPDX header, not listed | PM (header); listed in §4 |
| G-MMVQ-1 | `mmvq_batched.cu`, `mmvq_multim.cu`, `quantize_q8_1.cu`, `baracuda_mmvq_{batched,multim}.cuh` carry SPDX naming llama.cpp but were absent from `LICENSE-thirdparty.md`, which also claimed the q8_1 family was dropped | **FIXED** (md corrected; listed below) |

### Unrecorded changes / unpinned origins / stale claims

| id | gap | status |
|---|---|---|
| G-FA-2 | VENDOR.md said "headers preserved verbatim" without listing the 15 modified files | **FIXED** (modification list added) |
| G-FA-3 | VENDOR.md hd512 launcher line reference stale (363-385; actual start 348); hd160/224 launchers are re-adapted, not copied; candle `main` was a moving ref | **FIXED** (resolved to `132643dc`, line corrected) |
| G-FI-2 | VENDOR.md "Tag v0.6.12" is wrong: `eee0d75f` is a `main` commit of 2026-05-28, tag is `d768c14e` | **FIXED** |
| G-AWQ-1 | `kernels/quantization/awq_launcher.cu` path wrong (is `kernels/quantize/`) | **FIXED** |
| G-MHC-1 | VENDOR.md said headers "preserved verbatim"; upstream has none, baracuda prepended them | **FIXED** |
| G-APEX-1 | VENDOR.md claimed LICENSE/AUTHORS verbatim, `.cu` files that are `.cuh`, functors "vendored unchanged" | **FIXED** (claims corrected); LICENSE text itself PM (G-LIC-1) |
| G-PIN-1 | No upstream commit recorded for bitsandbytes (codebook matches HEAD `83364904`), causal-conv1d (`cd81f041`), mamba (`e9594ce1`), xformers (`4f701fad`), apex (`575968bc`), TE (`53e241c5`); these are HEADs measured 2026-10-14, **not** vendor-time commits | PM (decide which ref to record) |
| G-3428 | `LICENSE-thirdparty.md` cited llama.cpp commit `25805fff`, which does not resolve in a full clone | **FIXED** (PR number kept, commit id flagged) |
| G-FQ-1 | `vendor/fuel-q8_1/README` "Vendored commit: see git log" — no pin; matching refs are Fuel `e1070acf9`, candle `fd8448d2`; the xinfer origin rests on an `@author` comment + candle#3196 co-author line | PM |
| G-FORGE-2 | NOTICE marks `compute_cap/parallel/hash/dependency.rs` "verbatim": they differ from `aa77db3` by 33/43/41/104 lines after stripping comments; `builder.rs` by 378; `error/toolkit/lib.rs` also modified and unnamed; no origin headers | PM |
| G-CUTLASS-1 | `baracuda-cutlass{,-kernels}-sys/NOTICE` and `baracuda-cutlass/NOTICE` SKU descriptions are stale (not a licence defect) | open (docs; separate) |
| G-FI-3 | `vendor/flashinfer/VENDOR.md` places `fp16.h` under `attention/`; the file is `include/flashinfer/fp16.h`. `fastdiv.cuh` has dead code (`numer`, `q` computed twice) — correctness not assessed | open |

## 3a. Licence-file bucket applied (PR alpha.102, 2026-10-14)

PM-approved (`TASK` 13:06Z). New files and replaced files are **upstream bytes**; `docs/provenance/licence-files-verified.tsv` compares each staged blob id with the upstream blob id (18/18 identical). No `.cu/.cuh/.h/.rs` file is in the diff. Pin policy for the previously unpinned origins: the last upstream commit **on or before the 2026-05-28 vendor date** (for each, the licence blob is also unchanged at upstream HEAD); it is a date-bounded pin, not a recovered vendor-time commit.

| gap | now |
|---|---|
| G-FA-1, G-Q8-1 | candle MIT + Apache texts added in `vendor/flash-attention/`, `vendor/fuel-q8_1/`, `LICENSES/` |
| G-K-1 | candle + llama.cpp texts in `LICENSES/`; attention.rs has **no** upstream licence file (open, [FINDING]) |
| G-LIC-1 | LICENSE replaced verbatim: bitsandbytes `5500cffe`, causal-conv1d `4f6ae4e2`, mamba `a14b1dff`, xformers `c04f47b6`, apex `becbb77c` |
| G-LIC-2 | causal-conv1d, mamba `AUTHORS` replaced verbatim; upstream-less AUTHORS (awq, bitsandbytes, marlin, mhc, xformers, apex) now say so on the first line |
| G-TE-1 | `ATTRIBUTION.md` embedded text replaced with TE `LICENSE` (`9e5a847c`, bea1b426) |
| G-BNB-1 | bitsandbytes `NOTICE.md` carried |
| G-FI-1 | `LICENSE-FP16` (Maratyszcza/FP16) added; AMD's 2025 line has no separate upstream text (SPDX line only) |
| G-CUTF-1 | `LICENSES/cutf-LICENSE` added (headers: next PR) |
| G-L2-1 | `LICENSE-CUDA-L2.txt` tracked next to the probes (headers: next PR) |

Still open from §3: G-H-1/2, G-APEX-2, G-SORT-1, G-CUTF-1 headers, G-PIN-1, G-FQ-1, G-FORGE-1/2 (next PRs), and attention.rs / cudaforge licence files that do not exist upstream.

## 3b. Header-comment bucket applied (PR alpha.103, 2026-10-14)

Comment-only edits; `docs/provenance/header-comment-changes-verified.tsv` shows, per code file, that the content is identical after stripping `//` and `/* */` comments and whitespace (only `//` lines were added/removed).

| gap | now |
|---|---|
| G-H-1 | "Eric Holscher" replaced by "The candle contributors (huggingface/candle ...) via the Fuel project": **upstream candle's LICENSE-MIT/APACHE carry no copyright line and its files no holder**, so no named holder exists to restore ([ASK] raised) |
| G-H-2 | attention.rs label corrected to MIT (its `Cargo.toml`; no LICENSE file upstream) |
| G-APEX-2 | invented NVIDIA copyright line replaced by an origin note; **Apex's LICENSE has no copyright line either** ([ASK] raised) |
| G-SORT-1 | SPDX + origin line added to `sort.cu`, `baracuda_sort.cuh` (llama.cpp, "Copyright (c) 2023 Georgi Gerganov" per `LICENSES/llama.cpp-LICENSE`) |
| G-CUTF-1 | SPDX + origin added to the two cutf-derived headers ("Copyright (c) 2018 mutsuki") |
| G-L2-1 | origin + licence pointer added to the five CUDA-L2 probes |
| G-FORGE-1/2 | forge NOTICE restated with measured per-file differences; `// Origin:` line on each module; upstream has no LICENSE file and no copyright year, so the earlier "Copyright (c) 2025 Guoqing Bao" is no longer asserted (only `authors = ["Guoqing Bao"]`) |

**PM ruling (2026-10-14, 13:15Z):** where upstream states no holder, the header says so ("The candle contributors", "Apex" origin note) and never carries a fabricated named holder.

**Open, upstream-only:** `guoqingbao/attention.rs` and `guoqingbao/cudaforge` ship no LICENSE file (only a `Cargo.toml` `license` field), so no verbatim licence text can be carried. Asking each upstream to add one would resolve it; **no upstream issue has been filed** (needs PM approval).

## 3c. Pins bucket applied (PR alpha.104, 2026-10-14)

Recorded in each directory's `VENDOR.md` / `README.md`. **Date-bounded, not vendor-time:** for the origins that recorded no commit, the reference is the last upstream commit on or before the 2026-05-28 vendor date; the true vendor-time commit is not recoverable, and each entry says so.

| origin | reference commit | date |
|---|---|---|
| bitsandbytes | `5500cffe91d0a2a8adfbb34cc0c195abb9cde6ab` | 2026-05-27 |
| causal-conv1d | `4f6ae4e26ae5fe8af9372f8d312ab25cc4595223` | 2026-05-09 |
| mamba | `a14b1dff0454a3bc27d9eb31355dc01e4b2490ec` | 2026-05-09 |
| xformers | `c04f47b69b53d60a53916fd61ddd32bdb4a6b927` | 2026-05-21 |
| NVIDIA/apex | `becbb77cea4cb54f2929f7c938a0a6f7dd1fdc39` | 2026-05-28 |
| NVIDIA/TransformerEngine | `9e5a847c90c436c9c9c5f3e756977cf568967ccc` | 2026-05-28 |
| fuel-q8_1 chain | Fuel `e1070acf9...` = candle `fd8448d2c7fe...`; xinfer `b88c1533...`; llama.cpp `0a5eebb45d...` | blob-equality, exact |
| FA2 hd160/224 | `EricLBuehler/candle` `132643dc466c...` | 2025-07-09 (exact blob match) |

G-PIN-1 and G-FQ-1 are closed by this. Remaining open: attention.rs / cudaforge ship no LICENSE (upstream-only).

## 4. Proposed fixes needing PM (no vendoring changes made)

1. **Add licence texts** (new files only, no edits to vendored sources): candle `LICENSE-MIT` +
   `LICENSE-APACHE` (HF) next to the 12 candle-sourced FA2 files and in `vendor/fuel-q8_1`; ggml MIT
   and attention.rs MIT referenced from `LICENSE-thirdparty.md`; cutf MIT and CUDA-L2 MIT
   (`LICENSE.txt` at commit `dbe0177`) tracked in-repo; cudaforge licence + Bao credit in forge.
   (G-FA-1, G-Q8-1, G-K-1, G-CUTF-1, G-L2-1, G-FORGE-1)
2. **Replace non-verbatim `LICENSE`/`AUTHORS` with upstream's** for bitsandbytes, causal-conv1d,
   mamba, xformers, apex, and TE's embedded Apache text. Where upstream has no AUTHORS, rename ours
   `BARACUDA-ATTRIBUTION.md` so "verbatim upstream" is not implied. (G-LIC-1/2, G-TE-1, G-APEX-1)
3. **Header corrections** (source files, comment-only): drop the invented "Eric Holscher" holder in
   the six cast/fill/affine files; fix the attention.rs licence label to MIT; remove the apex NVIDIA
   notice from rewritten files; add SPDX to sort + cutf-derived headers. (G-H-1/2, G-APEX-2, G-SORT-1,
   G-CUTF-1)
4. **Pin refs** for the six unpinned origins (HEAD-at-audit as above, or an explicit "unknown at
   vendor time"). (G-PIN-1, G-FQ-1)
5. **Forge NOTICE**: restate which files are modified and by how much; add origin headers.
   (G-FORGE-2)

Derived files that were absent from `LICENSE-thirdparty.md`, listed here until a PM-approved header
or licence change lands: `kernels/gguf/{mmvq_batched,mmvq_multim,quantize_q8_1}.cu`,
`kernels/include/baracuda_mmvq_{batched,multim}.cuh`, `kernels/sort/sort.cu`,
`kernels/include/baracuda_sort.cuh` (llama.cpp via Fuel/candle, MIT);
`kernels/include/baracuda_{cp_async,fp_bits}.cuh` (Ootomo cutf, MIT);
`crates/baracuda-kernels-bench/external/cuda-l2-probes/*.cu` (CUDA-L2, MIT). Paths are under
`crates/baracuda-kernels-sys/` except the probes.

## 6. Non-code assets

| file | maker / tool | date | licence |
|---|---|---|---|
| `assets/barracuda.png` (README hero image) | AI-generated: embedded C2PA manifest (OpenAI Media Service, signed OpenAI OpCo, LLC) records `c2pa.created`, softwareAgent `GPT-4o`, digitalSourceType `trainedAlgorithmicMedia`; added by Eric Evans (CireSnave) in the initial commit `e36466e6` | 2026-04-22 (manifest `when` and commit date) | **unknown: pending CireSnave** (not asserted by this repo; the output terms of the generating service were not checked here) |

Method: `git log --follow` for the first-add commit; read-only parse of the PNG chunks (`caBX` = C2PA JUMBF box) at `origin/main` `1dd782a2`. `assets/README.txt` ("Placeholder ... barracuda.jpg ... attached to your Claude chat") is a stale placeholder note that does not describe the current file.
