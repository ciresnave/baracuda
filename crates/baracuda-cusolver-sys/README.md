# baracuda-cusolver-sys

Raw FFI bindings + dynamic loader for **NVIDIA cuSOLVER** — dense and
sparse linear-system solvers on the GPU. Covers `cuSolverDn` (dense),
`cuSolverSp` (sparse), `cuSolverRf` (refactor), and `cuSolverMg`
(multi-GPU dense).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcusolver.so` / `cusolver64_*.dll` /
`cusolverMg64_*.dll`.

**Most users want [`baracuda-cusolver`]** — that crate exposes typed
handles for each sub-API plus generic-over-scalar safe wrappers for the
common operations (LU, QR, Cholesky, SVD, eigensolvers).

## What's exposed

- **Dense (`cuSolverDn`)**: LU (`getrf` + `getrs`), QR (`geqrf` + `orgqr`
  + `ormqr`), Cholesky (`potrf` + `potrs` + `potri`), SVD, `syevd`,
  `syevj`, `gesvdj`, batched-`syevj` / batched-`gesvdj`, `gels`, plus
  the generic 64-bit `Xgetrf` / `Xgetrs` / `Xgeqrf` / `Xpotrf` /
  `Xpotrs` / `Xsyevd` family in `S` / `D` / `C` / `Z`.
- **Sparse (`cuSolverSp`)**: `csrlsvchol`, `csrlsvqr`.
- **Refactor (`cuSolverRf`)**: full Rf API.
- **Multi-GPU (`cuSolverMg`)**: `getrf`, `potrf`, `syevd`.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cusolver`]: https://docs.rs/baracuda-cusolver
