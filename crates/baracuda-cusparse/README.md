# baracuda-cusparse

Safe Rust wrappers for **NVIDIA cuSPARSE** — sparse linear algebra on
the GPU through the modern generic API.

Generic over scalar type (`f32` / `f64` / `Complex32` / `Complex64`)
where the math allows.

## Coverage

- **All sparse formats**: CSR, CSC, COO, BSR, with typed descriptor
  builders (`SparseMatrixCsr::new`, etc.).
- **Dense descriptors**: `DnVec`, `DnMat`.
- **Generic sparse-matrix ops** (modern, post cuSPARSE 11):
  - SpMV (sparse matrix × dense vector)
  - SpMM (sparse × dense)
  - SpGEMM (sparse × sparse) — three-phase: workspace estimate,
    workspace fill, compute.
  - SpSV / SpSM (triangular solve).
  - SDDMM (sampled dense-dense matmul).
- **Conversions**: sparse↔dense, CSR↔CSC.
- **Sparse-BLAS-1**: axpby, gather, scatter, rot.

## Workspace handling

The generic API requires the caller to provide scratch device memory
sized via a query call (e.g. `SpMV_bufferSize`). baracuda-cusparse
takes this through `&mut DeviceBuffer<u8>` arguments — explicit and
allocation-free at op time.

Pairs with [`baracuda-cusparse-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cusparse-sys`]: https://docs.rs/baracuda-cusparse-sys
