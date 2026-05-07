# baracuda-cusparse-sys

Raw FFI bindings + dynamic loader for **NVIDIA cuSPARSE** — sparse
linear algebra on the GPU.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcusparse.so` / `cusparse64_*.dll`.

**Most users want [`baracuda-cusparse`]** — that crate exposes typed
descriptors for sparse / dense matrices and vectors, the modern generic
API (SpMV, SpMM, SpGEMM 3-phase, SpSV, SpSM, SDDMM), and the
sparse↔dense / CSR↔CSC conversion helpers.

## What's exposed

- All sparse formats: CSR, CSC, COO, BSR.
- Generic API descriptors: `cusparseSpMatDescr_t`,
  `cusparseDnMatDescr_t`, `cusparseDnVecDescr_t`.
- SpMV / SpMM / SpGEMM (3-phase: workspace estimate → workspace fill →
  compute) / SpSV / SpSM / SDDMM.
- Sparse-BLAS-1 (axpby, gather, scatter, rot).
- Format conversions (`Sparse2Dense`, `Dense2Sparse`, `Csr2Csc`).

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cusparse`]: https://docs.rs/baracuda-cusparse
