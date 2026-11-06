# Conv2d forward picker vs pinned `IMPLICIT_GEMM` (SA-49 items 4 and 5)

Re-measured 2026-11-04 on `main` 45ed52b9 (alpha.90) plus the warm-up change to
`tests/conv2d_picker_measure.rs` in this change. **This replaces the tables from #323, which were taken with the GPU
at a low clock (see "Why the earlier tables were wrong").** Device: NVIDIA GeForce RTX 4070 Laptop GPU (sm_89),
driver 617.42, cuDNN 9.23 / CUDA 13.3, under `scripts/gpu-run.ps1`, no other compute process on the GPU. Command: the
test exe built with `cargo test --locked -p baracuda-kernels --features cudnn,device-tests --test conv2d_picker_measure --no-run`
(debug profile), run with `--nocapture`: 1 ran, 1 passed, 0 failed (180.5 s). NCHW, batch 8, mean of 20
stream-synchronised runs per sample after a 500 ms busy warm-up, 3 samples per (shape, dtype, mode); times are min / median
over the 3. Speedup = pinned median / picker median (above 1.00 means the picker is faster; bold = below 0.8x). The
test asserts picker output vs pinned output within a tolerance and passed; the per-row relative differences were not
recorded from this run (the #323 run had them bit-identical in f16 / bf16 and at most 5.1e-4 in f32).
One device, one run of 3 samples: this is variance within a run, not across runs or machines; min / median still differ by up to
1.6x on some rows (f16 1x1 512->128 pinned 0.198 / 0.312), so differences under about 1.2x are not resolved.

## f32

| shape | pinned ms (min / med) | picker algo / ms (min / med) | speedup (med) |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 0.256 / 0.263 | 6 / 0.141 / 0.142 | 1.85x |
| 3x3 s1 128ch 28x28 | 0.257 / 0.258 | 1 / 0.218 / 0.219 | 1.18x |
| 3x3 s1 256ch 14x14 | 0.279 / 0.280 | 1 / 0.169 / 0.195 | 1.43x |
| 1x1 s1 256->64 56x56 | 0.153 / 0.208 | 1 / 0.913 / 1.130 | **0.18x** |
| 7x7 s2 3->64 224x224 | 0.276 / 0.276 | 1 / 0.184 / 0.185 | 1.49x |
| 1x1 s1 64->256 56x56 | 0.178 / 0.184 | 1 / 0.197 / 0.230 | 0.80x |
| 1x1 s1 512->128 28x28 | 0.153 / 0.157 | 1 / 0.219 / 0.276 | **0.57x** |
| 1x1 s1 1024->256 14x14 | 0.160 / 0.163 | 1 / 0.244 / 0.249 | **0.65x** |
| 1x1 s2 256->512 56x56 | 0.245 / 0.249 | 1 / 0.157 / 0.181 | 1.38x |
| 3x3 s1 g32 256ch 14x14 | 0.856 / 0.986 | 1 / 0.856 / 0.920 | 1.07x |
| 3x3 s1 depthwise 128ch 28x28 | 0.092 / 0.134 | 1 / 0.115 / 0.123 | 1.09x |
| 3x3 s1 d2 128ch 28x28 | 0.258 / 0.262 | 1 / 0.175 / 0.192 | 1.36x |
| 3x3 s1 d4 64ch 56x56 | 0.253 / 0.253 | 1 / 0.170 / 0.191 | 1.32x |

## f16

| shape | pinned ms (min / med) | picker algo / ms (min / med) | speedup (med) |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 0.368 / 0.371 | 1 / 0.353 / 0.384 | 0.97x |
| 3x3 s1 128ch 28x28 | 0.225 / 0.227 | 1 / 0.374 / 0.435 | **0.52x** |
| 3x3 s1 256ch 14x14 | 0.305 / 0.310 | 1 / 0.408 / 0.411 | **0.75x** |
| 1x1 s1 256->64 56x56 | 0.187 / 0.194 | 1 / 0.359 / 0.397 | **0.49x** |
| 7x7 s2 3->64 224x224 | 0.276 / 0.280 | 1 / 0.202 / 0.287 | 0.98x |
| 1x1 s1 64->256 56x56 | 0.168 / 0.186 | 1 / 0.296 / 0.397 | **0.47x** |
| 1x1 s1 512->128 28x28 | 0.198 / 0.312 | 1 / 0.595 / 0.627 | **0.50x** |
| 1x1 s1 1024->256 14x14 | 0.183 / 0.201 | 1 / 0.452 / 0.565 | **0.36x** |
| 1x1 s2 256->512 56x56 | 0.202 / 0.321 | 1 / 0.197 / 0.323 | 0.99x |
| 3x3 s1 g32 256ch 14x14 | 1.090 / 1.142 | 1 / 0.888 / 1.154 | 0.99x |
| 3x3 s1 depthwise 128ch 28x28 | 0.126 / 0.177 | 1 / 0.144 / 0.165 | 1.07x |
| 3x3 s1 d2 128ch 28x28 | 0.224 / 0.227 | 1 / 0.362 / 0.377 | **0.60x** |
| 3x3 s1 d4 64ch 56x56 | 0.370 / 0.373 | 1 / 0.358 / 0.399 | 0.93x |

## bf16

| shape | pinned ms (min / med) | picker algo / ms (min / med) | speedup (med) |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 0.371 / 0.372 | 1 / 0.652 / 0.654 | **0.57x** |
| 3x3 s1 128ch 28x28 | 0.226 / 0.229 | 1 / 0.525 / 0.537 | **0.43x** |
| 3x3 s1 256ch 14x14 | 0.309 / 0.313 | 1 / 0.576 / 0.582 | **0.54x** |
| 1x1 s1 256->64 56x56 | 0.180 / 0.258 | 1 / 0.297 / 0.392 | **0.66x** |
| 7x7 s2 3->64 224x224 | 0.274 / 0.274 | 1 / 0.340 / 0.363 | **0.76x** |
| 1x1 s1 64->256 56x56 | 0.199 / 0.221 | 1 / 0.367 / 0.439 | **0.50x** |
| 1x1 s1 512->128 28x28 | 0.208 / 0.224 | 1 / 0.535 / 0.695 | **0.32x** |
| 1x1 s1 1024->256 14x14 | 0.156 / 0.206 | 1 / 0.622 / 0.627 | **0.33x** |
| 1x1 s2 256->512 56x56 | 0.198 / 0.201 | 1 / 0.539 / 0.562 | **0.36x** |
| 3x3 s1 g32 256ch 14x14 | 0.870 / 0.939 | 1 / 0.915 / 0.918 | 1.02x |
| 3x3 s1 depthwise 128ch 28x28 | 0.146 / 0.157 | 1 / 0.162 / 0.183 | 0.85x |
| 3x3 s1 d2 128ch 28x28 | 0.230 / 0.238 | 1 / 1.069 / 1.156 | **0.21x** |
| 3x3 s1 d4 64ch 56x56 | 0.372 / 0.372 | 1 / 1.093 / 1.131 | **0.33x** |

## Why the earlier tables were wrong

#323 reported half-precision times 4x to 7x higher than alpha.64 (baracuda#324). The cause is the GPU's idle power state, not
the shape order and not a concurrent user. The test's host side (debug-build data fill, upload, download) leaves the GPU idle
between measurements; it drops to P8 (210 to 390 MHz in a sampled run) and the timed loop (20 runs, about 12 ms of work)
finishes before the clocks ramp back. Evidence, all on this device:

- The 5 alpha.64 shapes run alone (9 s) reproduced alpha.64: f16 3x3 64ch pinned 0.599 ms (alpha.64: 0.591).
- The same 5 shapes with 5 repeats (50 s) ran slow: the same row at 3.377 ms median. Dropping the dilated shapes from the
  13-shape run did not help (two runs, slowdown starting at different shapes), so it is time in the process, not any shape.
- `nvidia-smi` sampled during the slow run: P8, 210 to 390 MHz, 0 to 47 % utilisation, no throttle reason set.
- With the 500 ms busy warm-up the GPU held P0 at 2550 MHz for 22 of 37 samples in the 13-shape run (the other 15: P0 at
  1170 to 2565 MHz, and P3 to P5 at 975 to 2385 MHz; samples every 5 s and not aligned to shapes) and the same row is 0.371 ms.

Absolute times here are therefore lower than alpha.62 and alpha.64 for the same shapes (f32 3x3 64ch pinned 0.263 ms vs
0.40 ms; f16 0.371 vs 0.591): those were taken at lower clocks, and their ratios are not carried over. A warm-up is a floor,
not proof of a steady clock: the sampling above shows the clock still moves.

## Reading

- **The picker is not a clear win in any dtype.** f32: wins on 9 of 13 shapes (3x3 1.18x to 1.85x, dilated 3x3 1.3x to
  1.4x, 7x7 s2 1.49x, 1x1 s2 1.38x) and loses on 4 (1x1 256->64 0.18x, 64->256 0.80x, 512->128 0.57x, 1024->256 0.65x).
  f16: at or above 1.0x on 1 of 13 (depthwise 1.07x), most between 0.36x and 0.99x. bf16: at or above 1.0x on 1 of 13
  (grouped 1.02x), most between 0.21x and 0.85x; dilated bf16 is the worst (0.21x, 0.33x).
- **This reverses two #323 claims.** #323 said the four other 1x1 f32 shapes win 1.5x to 2.0x and the one loser was a narrow
  output at large spatial size; with the clock held, three of those four 1x1 shapes lose (0.57x to 0.80x). And #323 said half
  data was unresolved while its rows showed the picker mostly at or above 1.0x; with the clock held half mostly loses, which
  is what alpha.64 found.
- **Dilated 3x3**: f32 wins (1.3x to 1.4x, not 2.1x to 2.3x), f16 mixed (0.60x, 0.93x), bf16 loses badly.
- **Decision: the picker stays opt-in, and nothing here supports a default flip.** A per-dtype or per-shape guard, or a timed
  pick (`cudnnFindConvolutionForwardAlgorithm`), is the next experiment; this run does not test either.

Still open under SA-49: backward (no picker) and other GPUs. The release-profile run asked for here was done on 2026-11-06 and is
the last section of this file: it does not settle whether the warm-up result is an artefact (the pinned times moved between profiles, cause not established).

## Release-profile re-run (SA-49 item 5, the last open sub-item of that row)

Measured 2026-11-06 on `main` 094932d1 (alpha.95), **release profile**, to check the warm-up result against a host side that does
not idle the GPU. Same device and driver (RTX 4070 Laptop, sm_89, driver 617.42, cuDNN 9.23 / CUDA 13.3), under
`scripts/gpu-run.ps1`. The exe was built with `cargo test --release --locked -p baracuda-kernels --features cudnn,device-tests --test conv2d_picker_measure --no-run`
and run twice back to back with `--nocapture`: 1 ran, 1 passed, 0 failed both times (134.0 s and 139.2 s). Each cell is `run 1 / run 2`;
times are the median over 3 samples of 20 runs after the 500 ms warm-up; speedup = pinned median / picker median (bold = below 0.8x).
The GPU clock was **not** sampled during these runs, so nothing below is attributed to clocks.

- **The conclusion does not change: the picker stays opt-in.** f16 is at or above 1.0x on 5 of 13 shapes in both
  runs, with the 1x1 shapes at 0.3x to 0.8x; bf16 is at or above 1.0x on 2 of 13 in both runs and loses 0.29x to 0.68x on the 1x1 and dilated shapes.
- **Run to run, 29 of 39 rows agree within 25%** (run 2 speedup / run 1 speedup between 0.8 and 1.25); the other 10 are mostly 1x1 shapes.
  The f32 3x3 and dilated wins (2.0x to 3.0x) and the f32 1x1 256->64 loss (0.17x) reproduce; the f32 1x1 64->256, 512->128 and 1024->256 rows do not
  (1.48x / 1.15x, 0.90x / 0.42x, 0.93x / 0.70x), so those three are unresolved here, not wins and not losses.
- **This run's pinned times differ from the debug-profile tables above** (for example f32 3x3 64ch pinned 0.49 ms here, 0.26 ms there, picker 0.14 to 0.20 ms in both), so its
  speedups are larger on the 3x3 rows and the two profiles' speedups must not be compared row by row. Why the pinned time moved is not established (profile, clock and
  process state were not separated), so this run does not show the warm-up result to be an artefact, and does not show it to be sound either.
- Still open under SA-49: backward (no picker) and other GPUs. A guard or timed-pick experiment is untested.

### f32 (release, run 1 / run 2)

| shape | pinned ms (med) | picker ms (med) | speedup (med) |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 0.494 / 0.484 | 0.199 / 0.200 | 2.48x / 2.42x |
| 3x3 s1 128ch 28x28 | 0.470 / 0.486 | 0.205 / 0.206 | 2.29x / 2.37x |
| 3x3 s1 256ch 14x14 | 0.515 / 0.514 | 0.208 / 0.173 | 2.48x / 2.97x |
| 1x1 s1 256->64 56x56 | 0.192 / 0.201 | 1.132 / 1.214 | **0.17x** / **0.17x** |
| 7x7 s2 3->64 224x224 | 0.618 / 0.555 | 0.411 / 0.381 | 1.50x / 1.46x |
| 1x1 s1 64->256 56x56 | 0.292 / 0.271 | 0.198 / 0.236 | 1.48x / 1.15x |
| 1x1 s1 512->128 28x28 | 0.222 / 0.203 | 0.246 / 0.485 | 0.90x / **0.42x** |
| 1x1 s1 1024->256 14x14 | 0.240 / 0.223 | 0.259 / 0.318 | 0.93x / **0.70x** |
| 1x1 s2 256->512 56x56 | 0.356 / 0.370 | 0.351 / 0.309 | 1.01x / 1.19x |
| 3x3 s1 g32 256ch 14x14 | 0.891 / 0.868 | 0.832 / 0.958 | 1.07x / 0.91x |
| 3x3 s1 depthwise 128ch 28x28 | 0.103 / 0.113 | 0.166 / 0.157 | **0.62x** / **0.72x** |
| 3x3 s1 d2 128ch 28x28 | 0.465 / 0.499 | 0.229 / 0.191 | 2.03x / 2.62x |
| 3x3 s1 d4 64ch 56x56 | 0.499 / 0.509 | 0.189 / 0.182 | 2.64x / 2.79x |

### f16 (release, run 1 / run 2)

| shape | pinned ms (med) | picker ms (med) | speedup (med) |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 0.602 / 0.662 | 0.478 / 0.460 | 1.26x / 1.44x |
| 3x3 s1 128ch 28x28 | 0.370 / 0.370 | 0.433 / 0.334 | 0.86x / 1.11x |
| 3x3 s1 256ch 14x14 | 0.433 / 0.421 | 0.406 / 0.574 | 1.06x / **0.73x** |
| 1x1 s1 256->64 56x56 | 0.260 / 0.258 | 0.349 / 0.449 | **0.74x** / **0.57x** |
| 7x7 s2 3->64 224x224 | 0.511 / 0.478 | 0.290 / 0.281 | 1.76x / 1.70x |
| 1x1 s1 64->256 56x56 | 0.218 / 0.299 | 0.330 / 0.384 | **0.66x** / **0.78x** |
| 1x1 s1 512->128 28x28 | 0.181 / 0.230 | 0.598 / 0.662 | **0.30x** / **0.35x** |
| 1x1 s1 1024->256 14x14 | 0.195 / 0.186 | 0.603 / 0.557 | **0.32x** / **0.33x** |
| 1x1 s2 256->512 56x56 | 0.322 / 0.313 | 0.256 / 0.250 | 1.26x / 1.25x |
| 3x3 s1 g32 256ch 14x14 | 1.051 / 1.011 | 1.076 / 1.202 | 0.98x / 0.84x |
| 3x3 s1 depthwise 128ch 28x28 | 0.135 / 0.151 | 0.183 / 0.164 | **0.74x** / 0.92x |
| 3x3 s1 d2 128ch 28x28 | 0.337 / 0.335 | 0.410 / 0.430 | 0.82x / **0.78x** |
| 3x3 s1 d4 64ch 56x56 | 0.616 / 0.657 | 0.506 / 0.473 | 1.22x / 1.39x |

### bf16 (release, run 1 / run 2)

| shape | pinned ms (med) | picker ms (med) | speedup (med) |
| --- | --- | --- | --- |
| 3x3 s1 64ch 56x56 | 0.629 / 0.638 | 0.613 / 0.609 | 1.03x / 1.05x |
| 3x3 s1 128ch 28x28 | 0.361 / 0.351 | 0.539 / 0.487 | **0.67x** / **0.72x** |
| 3x3 s1 256ch 14x14 | 0.427 / 0.455 | 0.477 / 0.493 | 0.90x / 0.92x |
| 1x1 s1 256->64 56x56 | 0.270 / 0.322 | 0.394 / 3.194 | **0.68x** / **0.10x** |
| 7x7 s2 3->64 224x224 | 0.487 / 0.458 | 0.531 / 0.543 | 0.92x / 0.84x |
| 1x1 s1 64->256 56x56 | 0.228 / 0.208 | 0.448 / 0.569 | **0.51x** / **0.36x** |
| 1x1 s1 512->128 28x28 | 0.211 / 0.218 | 0.689 / 0.929 | **0.31x** / **0.23x** |
| 1x1 s1 1024->256 14x14 | 0.196 / 0.382 | 0.666 / 1.049 | **0.29x** / **0.36x** |
| 1x1 s2 256->512 56x56 | 0.317 / 0.329 | 0.540 / 0.479 | **0.59x** / **0.69x** |
| 3x3 s1 g32 256ch 14x14 | 1.118 / 1.202 | 1.068 / 1.141 | 1.05x / 1.05x |
| 3x3 s1 depthwise 128ch 28x28 | 0.125 / 0.130 | 0.158 / 0.161 | **0.79x** / 0.81x |
| 3x3 s1 d2 128ch 28x28 | 0.339 / 0.332 | 1.158 / 1.109 | **0.29x** / **0.30x** |
| 3x3 s1 d4 64ch 56x56 | 0.633 / 0.618 | 1.188 / 1.185 | **0.53x** / **0.52x** |
