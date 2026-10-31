# mHC.cu backward: two defects (paste-ready report)

Found 2026-10-31 while adding the static-H backward to baracuda (ROADMAP SA-31). Upstream:
<https://github.com/AndreSlavescu/mHC.cu>, commit `a426939c2dbc11c443db041bcff12b65d1b6482a` (the vendored copy in
`crates/baracuda-kernels-sys/vendor/mhc/` is byte-identical apart from the SPDX header). Board 349 tracks the outward post.

Two files reproduce it, neither uses any baracuda code:

- `docs/repro/mhc_upstream_backward_repro.cu`: builds against the upstream headers, runs the static-H forward and
  `backward()` for B=2, C=16, n=4, 20 Sinkhorn iterations, and prints the gradients next to a central finite difference of
  the same GPU forward.
- `docs/repro/sinkhorn_adjoint_check.py`: pure Python, no GPU, no dependencies.

## Defect 1: `MHCLayer::backward` never combines the two paths of dL/dx

`stream_distribute_mix_backward_fused` writes the gradient through the mix path into `grads.d_x_mixed`;
`stream_aggregate_backward` writes the gradient through the aggregate path into `grads.d_x_expanded`. `backward()` returns
without adding them, so `grads.d_x_expanded` alone is not dL/dx. Measured (RTX 4070, CUDA 13.3):

    x[ 0]: d_x_expanded=-2.89140  d_x_mixed=0.32486  sum=-2.56654  finite-diff=-2.56462
    x[ 5]: d_x_expanded= 1.17103  d_x_mixed=0.28820  sum= 1.45923  finite-diff= 1.46046

Fix: dL/dx = `d_x_expanded + d_x_mixed`.

## Defect 2: `sinkhorn_knopp_backward` is not the adjoint of the forward

For `y = x / sum(x)` the exact adjoint is `dx = (dy - sum(dy * y)) / sum(x)`. The backward kernels (checkpointed
and generic) instead compute `d = d - y * sum(d * y)` per step and have no `1/sum`. Measured through the layer, `grads.d_H_res`
is about n times too large:

    h_res[ 0]: d_H_res=-0.63429  finite-diff=-0.16727  ratio=3.79
    h_res[12]: d_H_res= 0.65333  finite-diff= 0.16594  ratio=3.94

Pure-Python check of the Sinkhorn adjoint alone (n=4, fixed-seed K, max error against a central difference, gradient scale
0.17-0.35):

    iters= 1  exact 3.6e-11   mHC.cu style 7.5e-01
    iters= 2  exact 1.6e-10   mHC.cu style 6.2e-01
    iters=20  exact 2.0e-10   mHC.cu style 8.8e-01

baracuda's replacement kernel (`mhc_sinkhorn_backward_exact_kernel` in `kernels/attention/mhc_launcher.cu`) uses the exact
formula; with it all five gradients of the layer match the finite differences to <= 5e-7 of scale.
