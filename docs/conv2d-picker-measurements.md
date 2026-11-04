# Conv2d forward picker vs pinned `IMPLICIT_GEMM` (SA-49 items 4 and 5)

Measured 2026-11-04 on `main` 5b867259 (alpha.89) plus the extended `tests/conv2d_picker_measure.rs`
in this change. Device: NVIDIA GeForce RTX 4070 Laptop GPU (sm_89), driver 617.42, cuDNN 9.23 /
CUDA 13.3, under `scripts/gpu-run.ps1`. Command: the test exe built with
`cargo test --locked -p baracuda-kernels --features cudnn,device-tests --test conv2d_picker_measure --no-run`,
run with `--nocapture`: 1 ran, 1 passed, 0 failed (111.6 s). NCHW, batch 8, mean of 20 stream-synchronised
runs per sample, 5 samples per (shape, dtype, mode); times are min / median over the 5. Speedup =
pinned median / picker median (above 1.00 means the picker is faster). Picker output vs pinned output
is bit-identical (rel diff 0) in every f16 / bf16 case and at most 5.1e-4 in f32.
One device, one run of 5 samples: this is variance within a run, not across runs or machines.

## f32

| shape | pinned ms | picker algo / ms | speedup |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 0.401 / 0.402 | 6 / 0.196 / 0.198 | 2.03x |
| 3x3 s1 128ch 28x28 | 0.401 / 0.403 | 1 / 0.185 / 0.187 | 2.16x |
| 3x3 s1 256ch 14x14 | 0.435 / 0.438 | 1 / 0.217 / 0.218 | 2.01x |
| 1x1 s1 256->64 56x56 | 0.185 / 0.194 | 1 / 0.756 / 0.953 | **0.20x** |
| 7x7 s2 3->64 224x224 | 0.428 / 1.795 | 1 / 0.286 / 1.309 | 1.37x (noisy) |
| 1x1 s1 64->256 56x56 | 2.916 / 2.922 | 1 / 1.707 / 1.750 | 1.67x |
| 1x1 s1 512->128 28x28 | 1.368 / 1.369 | 1 / 0.735 / 0.739 | 1.85x |
| 1x1 s1 1024->256 14x14 | 1.450 / 1.454 | 1 / 0.706 / 0.711 | 2.04x |
| 1x1 s2 256->512 56x56 | 2.311 / 2.456 | 1 / 1.603 / 1.626 | 1.51x |
| 3x3 s1 groups 32, 256ch 14x14 | 0.869 / 0.960 | 1 / 0.926 / 1.046 | 0.92x |
| 3x3 s1 depthwise 128ch 28x28 | 0.229 / 0.242 | 1 / 0.229 / 0.244 | 0.99x |
| 3x3 s1 dilation 2, 128ch 28x28 | 2.465 / 2.466 | 1 / 1.065 / 1.069 | 2.31x |
| 3x3 s1 dilation 4, 64ch 56x56 | 2.201 / 2.205 | 1 / 1.058 / 1.059 | 2.08x |

## f16

| shape | pinned ms | picker ms | speedup |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 3.782 / 3.822 | 3.213 / 3.540 | 1.08x |
| 3x3 s1 128ch 28x28 | 2.119 / 2.134 | 2.107 / 2.120 | 1.01x |
| 3x3 s1 256ch 14x14 | 2.000 / 2.340 | 1.499 / 1.725 | 1.36x |
| 1x1 s1 256->64 56x56 | 1.152 / 1.972 | 1.153 / 1.681 | 1.17x |
| 7x7 s2 3->64 224x224 | 3.244 / 3.248 | 2.139 / 2.141 | 1.52x |
| 1x1 s1 64->256 56x56 | 1.548 / 1.573 | 1.068 / 1.088 | 1.45x |
| 1x1 s1 512->128 28x28 | 1.196 / 1.199 | 1.007 / 1.016 | 1.18x |
| 1x1 s1 1024->256 14x14 | 1.583 / 1.600 | 1.307 / 1.335 | 1.20x |
| 1x1 s2 256->512 56x56 | 2.218 / 2.223 | 1.756 / 1.760 | 1.26x |
| 3x3 s1 groups 32, 256ch 14x14 | 0.944 / 0.988 | 0.853 / 1.022 | 0.97x |
| 3x3 s1 depthwise 128ch 28x28 | 0.246 / 0.248 | 0.247 / 0.247 | 1.00x |
| 3x3 s1 dilation 2, 128ch 28x28 | 1.805 / 1.806 | 1.879 / 1.892 | 0.95x |
| 3x3 s1 dilation 4, 64ch 56x56 | 3.025 / 3.557 | 2.945 / 3.440 | 1.03x |

## bf16

| shape | pinned ms | picker ms | speedup |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 3.196 / 3.569 | 3.121 / 3.440 | 1.04x |
| 3x3 s1 128ch 28x28 | 1.723 / 1.728 | 1.761 / 1.791 | 0.96x |
| 3x3 s1 256ch 14x14 | 1.827 / 2.232 | 1.860 / 2.235 | 1.00x |
| 1x1 s1 256->64 56x56 | 1.034 / 1.978 | 0.985 / 1.880 | 1.05x |
| 7x7 s2 3->64 224x224 | 3.041 / 3.260 | 3.096 / 3.303 | 0.99x |
| 1x1 s1 64->256 56x56 | 1.495 / 1.571 | 1.524 / 1.549 | 1.01x |
| 1x1 s1 512->128 28x28 | 1.201 / 1.203 | 1.141 / 1.155 | 1.04x |
| 1x1 s1 1024->256 14x14 | 1.586 / 1.611 | 1.502 / 1.547 | 1.04x |
| 1x1 s2 256->512 56x56 | 2.228 / 2.251 | 2.131 / 2.158 | 1.04x |
| 3x3 s1 groups 32, 256ch 14x14 | 0.914 / 0.930 | 0.875 / 0.971 | 0.96x |
| 3x3 s1 depthwise 128ch 28x28 | 0.260 / 0.261 | 0.259 / 0.262 | 1.00x |
| 3x3 s1 dilation 2, 128ch 28x28 | 1.729 / 2.128 | 1.800 / 2.191 | 0.97x |
| 3x3 s1 dilation 4, 64ch 56x56 | 2.894 / 3.376 | 2.955 / 3.450 | 0.98x |

## Reading

- **f32 reproduces alpha.62**: 3x3 about 2.0x, and 1x1 256->64 at 56x56 about 5x slower (0.20x here, 0.17x then).
- **The 1x1 regression is not "1x1 in general".** The four other 1x1 shapes measured here (64->256,
  512->128, 1024->256, stride 2 256->512) win 1.5x to 2.0x in f32. The one loser is a narrow output
  (64 channels) at the largest spatial size. A guard on that shape class or a timed pick
  (`cudnnFindConvolutionForwardAlgorithm`) is the next experiment; this run does not test either.
- **Dilated 3x3 is the largest f32 win (2.1x to 2.3x)**; grouped and depthwise show no gain (0.92x to 0.99x).
- **Half precision does NOT reproduce alpha.64.** alpha.64 measured the picker winning 2 of 10 half
  cases; here it is at or above 1.0x on most and 0.95x to 0.98x on a few. Absolute half times are also
  4x to 7x higher than alpha.64's (e.g. f16 3x3 64ch pinned 3.8 ms here vs 0.59 ms), while f32 matches.
  The cause is not established (clocks, power state, a concurrent GPU user, or the new
  shapes preceding it are all unexamined), so neither half-precision table is trustworthy until it is repeated
  on a quiet device. Within this run the min/median spread is up to 1.9x in places (bf16 1x1 256->64: 1.034 / 1.978).
- **Decision: the picker stays opt-in.** The f32 data supports default-on except for the one regression
  and the half data is unresolved, so a default flip is not supported yet.

Still open under SA-49: backward (no picker), other GPUs, a quiet-device repeat of the half tables.
