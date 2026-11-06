//! Handle creation for the linalg / loss plans, funnelled through the
//! process-wide lock in `baracuda_kernels_sys::handle_init` (see its docs for
//! the race this serialises). Shared with the `kernels-sys` C facades, so
//! plans and facades exclude each other too.

use baracuda_cutlass::{Error, Result};
use baracuda_kernels_sys::{cublasHandle_t, cusolverDnHandle_t, handle_init};

/// Create a cuBLAS handle, serialised with every other handle creation.
pub fn create_cublas_handle() -> Result<cublasHandle_t> {
    handle_init::create_cublas().map_err(|status| Error::KernelInternal(-status))
}

/// Create a cuSOLVER dense handle, serialised with every other handle creation.
pub fn create_cusolver_dn_handle() -> Result<cusolverDnHandle_t> {
    handle_init::create_cusolver_dn().map_err(|status| Error::KernelInternal(-status))
}
