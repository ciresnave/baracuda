# mHC.cu backward: two defects

Found 2026-10-31 while adding the static-H backward to baracuda (ROADMAP SA-31). Upstream:
<https://github.com/AndreSlavescu/mHC.cu>, commit `a426939c2dbc11c443db041bcff12b65d1b6482a` (the vendored copy in
`crates/baracuda-kernels-sys/vendor/mhc/` is byte-identical apart from the SPDX header).
**Reported upstream: <https://github.com/AndreSlavescu/mHC.cu/issues/11>** (board 349; posted by CireSnave without the repro or
patch attached). A PR follows only if the author agrees.

Files, none of which uses any baracuda code:

- `docs/repro/mhc_upstream_backward_repro.cu`: pass/fail finite-difference repro (checks A to D) against the upstream headers.
  Build from an mHC.cu checkout (on Windows add `-Xcompiler /Zc:preprocessor`; adjust `-arch`):
  `nvcc -std=c++17 -arch=sm_89 -I src/csrc/include -I src/csrc/kernels repro.cu -o repro -lcublasLt -lcublas`. Exit 1 = a check failed.
- `docs/repro/sinkhorn_adjoint_check.py`: pure Python, no GPU, no dependencies.
- `docs/repro/mhc_upstream_0001-fix-sinkhorn-backward-adjoint-and-layer-dx.patch`: fix for both, `git apply`s to the commit above.

## Defect 1: `MHCLayer::backward` leaves dL/dx in two buffers

`x_expanded` reaches the loss through two paths. `stream_aggregate_backward` writes the aggregate-path gradient to
`grads.d_x_expanded`; `stream_distribute_mix_backward_fused` writes the mix-path gradient to `grads.d_x_mixed`. The C++
`MHCLayer::backward` returns without adding them, so `grads.d_x_expanded` alone is not dL/dx. (The Python binding,
`src/python/bindings.cu`, does add them, so Python callers are unaffected; it was not run here.) This is not a missing sum over
a dimension. Measured (RTX 4070 Laptop sm_89, CUDA 13.3, B=2, C=16, n=4, 20 iterations):

    x[ 0]: d_x_expanded=-2.89140  d_x_mixed=0.32486  sum=-2.56654  finite-diff=-2.56462
    x[ 5]: d_x_expanded= 1.17103  d_x_mixed=0.28820  sum= 1.45923  finite-diff= 1.46046

Fix: dL/dx = `d_x_expanded + d_x_mixed`; the patch leaves the sum in `d_x_expanded`. (If keeping them apart is intended for the
C++ API, a comment saying so would do.)

## Defect 2: `sinkhorn_knopp_backward` is not the adjoint of the forward

For `y = x / sum(x)` the exact adjoint is `dx = (dy - sum(dy * y)) / sum(x)`. The backward kernels (checkpointed and generic)
instead compute `d = d - w * sum(d * w)` per step and have no `1/sum`; the column step also takes its dot product from the tile
before the column normalisation. Measured through the layer, `grads.d_H_res` is about n times too large:

    h_res[ 0]: d_H_res=-0.63429  finite-diff=-0.16727  ratio=3.79
    h_res[12]: d_H_res= 0.65333  finite-diff= 0.16594  ratio=3.94

`sinkhorn_knopp_backward` alone, finite difference of the same GPU forward: N=4, 20 iterations gives ratios -6.07, 6.63, 104,
3.30 (checkpointed kernel); N=40, 5 iterations gives 26 to 28 (generic kernel). Pure-Python check (n=4, max error against a
central difference, gradient scale 0.17-0.35):

    iters= 1  exact 3.6e-11   mHC.cu style 7.5e-01
    iters= 2  exact 1.6e-10   mHC.cu style 6.2e-01
    iters=20  exact 2.0e-10   mHC.cu style 8.8e-01

baracuda's replacement kernel (`mhc_sinkhorn_backward_exact_kernel` in `kernels/attention/mhc_launcher.cu`) uses the exact
formula; with it all five gradients of the layer match the finite differences to <= 5e-7 of scale.

## Results of the repro (RTX 4070 Laptop, driver 617.42, CUDA 13.3, MSVC, float32)

| check | unmodified a426939c | with the patch |
|---|---|---|
| A: dL/dx in `grads.d_x_expanded` | FAIL (worst error 0.327 on scale 2.56) | PASS (0.0019) |
| B: `grads.d_H_res` | FAIL (ratio 2.9 to 4.2) | PASS (ratio 1.00) |
| C: `sinkhorn_knopp_backward`, N=4, 20 iterations | FAIL | PASS |
| D: `sinkhorn_knopp_backward`, N=40, 5 iterations | FAIL | PASS |

In the patched run the repro's printed `sum=` column is meaningless: the sum is already in `d_x_expanded`, so it is counted
twice; the verdict uses `d_x_expanded` only.

**Not verified:** other GPUs and architectures; other upstream commits; `sinkhorn_knopp_backward_batched_n4_kernel` (same formula
by reading, not run, not patched); the dynamic-H path; the Python binding path; bf16-only paths beyond what the repro touches;
upstream's `make test` with the patch applied.

## Derivation: the Sinkhorn-Knopp adjoint, and what `sinkhorn_knopp_backward` computes instead

## Forward

Input `K` (an `n x n` matrix, `K = exp(H_res)` in the layer). Each iteration normalises every row to sum 1, then
every column to sum 1:

    A = diag(r) K          r_i = 1 / sum_j K_ij        (row step)
    B = A diag(c)          c_j = 1 / sum_i A_ij        (column step)

and `B` is the next iteration's `K`. After `iters` iterations the output is the last `B`.

## One normalisation step

Take one vector `x` with sum `s = sum(x)` and `y = x / s`. Then `dy_k/dx_j = delta_kj / s - x_k / s^2`, so for an
upstream gradient `dy`:

    dx_j = sum_k dy_k * dy_k/dx_j
         = dy_j / s - (sum_k dy_k x_k) / s^2
         = ( dy_j - sum_k dy_k y_k ) / s            (using y = x / s)

That is: **subtract the scalar `<dy, y>` from every entry, then multiply by `1/s`.**

Applied to the matrix steps:

    column step  B = A diag(c):   dA_ij = c_j * ( dB_ij - sum_k dB_kj B_kj )
    row step     A = diag(r) K:   dK_ij = r_i * ( dA_ij - sum_l dA_il A_il )

Run the iterations in reverse order, and inside each one undo the column step first, then the row step.

## What is needed from the forward pass

For iteration `t`: `A_t` (the row-normalised tile, which `sinkhorn_knopp_backward_checkpointed_kernel` already saves),
`r_t` and `c_t` (the inverse sums). `B_t = A_t diag(c_t)` is recomputed from them. The patch saves `r_t` and `c_t` next to the
checkpoints (`2 * MAX_ITERS * N` floats). The generic kernel, which replays the forward pass for each iteration, takes
`r_t` from the row normalisation it already runs and computes `c_t` from `A_t`.

When the forward pass skips a normalisation (`sum <= eps`, it multiplies by 0 in the stored form), the step is the
identity, so the backward step leaves the gradient unchanged: the patch tests the stored inverse sum for 0.

## What the current kernels compute

Both backward kernels do, per step,

    d = d - w * sum(d * w)          (w = the saved row-normalised tile, for the column AND the row step)

with no `1/s` factor. Compared with the exact step:

1. The subtracted scalar is multiplied by `w` entry by entry (`d_ij - w_ij * <d, w>`) instead of being subtracted as a
   constant (`d_ij - <d, y>`). For `y = x / s` those are different maps.
2. The `1/s` factor is missing. Near uniform `K`, `1/s` is about `1/n`, so the result is about `n` times too large; the
   measured ratio for `n = 4` is 3.8 to 3.9 (see the results below).
3. For the column step the dot product uses `A_t` (before the column normalisation) where the exact step uses `B_t` (after).

`sinkhorn_adjoint_check.py` differentiates the forward numerically for `n = 4` and prints the max error of the exact step
and of the current form at 1, 2 and 20 iterations (about `1e-10` against about `0.6` to `0.9`, on gradients of size 0.2 to 0.35).

## `MHCLayer::backward` and dL/dx

`x_expanded` is used twice in the forward pass: it is aggregated into the layer input (`stream_aggregate`) and it is mixed
with the residual matrix (`stream_distribute_mix`). So dL/dx is the sum of two gradients. `backward()` writes the
aggregate-path gradient to `grads.d_x_expanded` and the mix-path gradient to `grads.d_x_mixed` and stops; the Python
binding (`src/python/bindings.cu`) adds them (`d_x_expanded = d_x_mix + d_x_from_agg`). The patch makes the C++ entry point
do the same, leaving the sum in `grads.d_x_expanded`.
