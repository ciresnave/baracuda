// baracuda C-ABI version: the single source of truth for `baracuda_kernels_abi_version()`.
//
// An independent integer (SONAME-style), NOT the crate version. Bump it by hand ONLY when the
// C/FFI ABI of `baracuda-kernels-sys` or `baracuda-cutlass-kernels-sys` breaks: a signature,
// a `#[repr(C)]` layout, a status-code or enum value, or a documented contract
// (docs/planning/api-freeze/README.md section 4.2). Additive changes do not bump it.
// `scripts/check-abi-manifest.py` enforces this in CI.
//
// A C consumer includes this header and compares `BARACUDA_KERNELS_ABI_VERSION` with the value the
// loaded library returns, refusing a mismatched .dll/.so instead of crashing.
#ifndef BARACUDA_ABI_H
#define BARACUDA_ABI_H

#define BARACUDA_KERNELS_ABI_VERSION 1u

#ifdef __cplusplus
extern "C" {
#endif

unsigned int baracuda_kernels_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif  // BARACUDA_ABI_H
