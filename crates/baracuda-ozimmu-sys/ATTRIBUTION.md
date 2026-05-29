# Attribution for `baracuda-ozimmu-sys`

The Ozaki-scheme FP64 GEMM algorithm — synthesize one FP64 GEMM out
of S² int8 tensor-core matmuls — is the work of:

- **Hiroyuki Ootomo, Katsuhisa Ozaki, Rio Yokota.** "DGEMM on Integer
  Matrix Multiplication Unit." *International Journal of High
  Performance Computing Applications* (IJHPCA) 2024.
  [arXiv:2306.11975](https://arxiv.org/abs/2306.11975).
- **Daichi Mukunoki, Hiroyuki Ootomo, Tomohide Imamura, Rio Yokota,
  Katsuhisa Ozaki.** "Efficient implementation of the Ozaki scheme
  on integer matrix multiplication unit." *Concurrency and
  Computation: Practice and Experience* (2024).

The reference implementation we forked from is:

- **`enp1s0/ozIMMU`** (Hiroyuki Ootomo, MIT-licensed) —
  <https://github.com/enp1s0/ozIMMU>.

The Phase 44c perf-enhancement variants (EF / RN / H + n-blocking)
are the work of:

- **Tomonori Uchino, Katsuhisa Ozaki, Toshiyuki Imamura.**
  "Performance enhancement of the Ozaki Scheme on integer matrix
  multiplication unit." *arXiv preprint*
  [arXiv:2409.13313](https://arxiv.org/abs/2409.13313) (2024). The
  paper introduces the three accuracy / perf variants (`EF` =
  error-free summation, `RN` = nearest-rounding split, `H` = EF + RN
  combined) and the n-blocking transform on the int8 GEMM call to
  recover scaling at large N.
- The reference implementation we ported the patches from is
  **`RIKEN-RCCS/accelerator_for_ozIMMU`** (MIT-licensed) —
  <https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU>. The repo
  ships full source-replacement directories (one per variant)
  rather than diff patches; baracuda Phase 44c reads them, lifts
  the algorithmic deltas, and applies them to baracuda's already-
  modified `cuda/gemm.cu` + `cuda/split.cu`.

## Phase 44 → 44b history

- **Phase 44 (baracuda alpha.56)** vendored `enp1s0/ozIMMU` under
  `vendor/ozimmu/` at upstream commit
  `08eea9231729d54dbfd92955f2cbfc21ec236856`, with a `cutf` git
  submodule (Hiroyuki Ootomo's CUDA-utility header library,
  formerly at `gitlab.momo86.net/mutsuki/cutf`) pinned at commit
  `c28c2025a5f3419661ce9cc632e3139a71b6f382`. Two small patches were
  applied: disable the LD_PRELOAD interception (we statically link
  cuBLAS) and drop the `cublas.cu` / `culip.cu` translation units
  from the build (Linux-only). Linux-only because the upstream code
  used the GCC/Clang `__uint128_t` extension that MSVC's nvcc host
  compiler doesn't provide.
- **Phase 44b (baracuda alpha.57)** clean-forks the whole stack:
  - The `cutf` submodule has been retired — its upstream went offline
    during the Phase 44 → 44b transition (`gitlab.momo86.net` returns
    nothing) and we already owned the code through transitive
    inclusion. The ~360 LOC of useful FP-bit-twiddle + cp_async
    utilities are now baracuda-native, living under
    `crates/baracuda-kernels-sys/kernels/include/baracuda_fp_bits.cuh`
    and `baracuda_cp_async.cuh`. The remaining ~2,200 LOC of cutf
    library wrappers (cublas.hpp, cusolver.hpp, etc.) duplicated
    infrastructure baracuda already had through its own `-sys` crates;
    those headers have been deleted outright.
  - The ozIMMU sources have moved from `vendor/ozimmu/src/` to
    `cuda/`. They're no longer "vendored" in any meaningful sense —
    we own the implementation now, have restyled it to baracuda
    conventions, and will continue diverging as needed.
  - The LD_PRELOAD path has been removed entirely (it never made
    sense for the baracuda integration; Phase 44 had gated it
    behind a `OZIMMU_BARACUDA_DIRECT_LINK` macro and Phase 44b
    just cuts the dead path).
  - The `__uint128_t` Windows blocker has been resolved via a new
    portable `baracuda::Uint128` (in
    `baracuda-kernels-sys/kernels/include/baracuda_fp_bits.cuh`).
    On Linux this is a typedef alias for the GCC builtin (bit-for-bit
    behavior preserved); on Windows it's a small struct that
    implements the operations the Ozaki splitter actually exercises.
  - The build path now works on Linux + Windows.
- **Phase 44c (baracuda alpha.57, no version bump)** folds in the
  RIKEN-RCCS perf-enhancement variants:
  - `split_int8_nearest` (nearest-rounding splitter — H-variant
    flavour, no per-slice scale array) added to `cuda/split.cu`.
  - `accumulate_in_f64_2` + `axby_2` (RN/H reconstruction helpers)
    added to `cuda/gemm.cu`.
  - `matmul_core` extended with an optional `beta_i` parameter (for
    EF/H group-wise accumulation) and n-blocking (n>12288 splits
    into 8192-wide chunks).
  - `gemm_int8_double_variant` dispatches Base / EF / RN / H based
    on a `variant` integer flag.
  - New C-ABI entry `baracuda_ozimmu_dgemm_with_variant` in
    `cuda/baracuda_shim.cu`. The Phase 44b
    `baracuda_ozimmu_dgemm` entry is unchanged (still routes to
    `mtk::ozimmu::gemm` → base path).
  - Public-API extension: `BackendKind::Ozaki { slices: u8 }`
    discriminant gains a high-3-bits variant field (0 = Base, 1 =
    EF, 2 = RN, 3 = H). New `OzakiVariant` enum + `OzakiSlices`
    helper constructors (`ozaki_slices::ef(8)`, etc.) in
    `baracuda-kernels-types::sku`. Source-compatible with Phase
    44b callers (`slices: 8` still decodes as Base/S8).

## License

The original ozIMMU code is MIT-licensed. The full text follows:

```
MIT License

Copyright (c) 2023 enp1s0

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The cutf utility code is also MIT-licensed (same author — Hiroyuki
Ootomo). Same text applies.

baracuda's own contributions on top — the C-ABI shim, the portable
`baracuda::Uint128`, the file-layout restructuring, the restyle pass —
are dual-licensed under MIT OR Apache-2.0, matching the rest of the
baracuda workspace.

## What baracuda does not claim

- The algorithm is not novel to baracuda; the IJHPCA paper above is
  the canonical reference. Our implementation is one of several;
  expect divergence from the reference at the perf / accuracy
  micro-tradeoff level once baracuda layers in its own tuning.
- The bit-twiddle utilities (`baracuda_fp_bits.cuh`) are derived from
  Hiroyuki Ootomo's `cutf` work. They're folded into baracuda only
  because the upstream went offline; we'd link to cutf-as-a-dep
  instead if it were still alive.
