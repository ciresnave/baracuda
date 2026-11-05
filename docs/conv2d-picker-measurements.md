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

Still open under SA-49: backward (no picker), other GPUs, and a run whose host side does not idle the GPU (release profile or
pre-filled buffers) to check that the warm-up result is not itself an artefact of the warm-up.
