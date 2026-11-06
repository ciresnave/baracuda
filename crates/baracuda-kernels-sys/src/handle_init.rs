//! Process-wide serialisation of cuBLAS / cuSOLVER handle creation.
//!
//! `cublasCreate_v2` and `cusolverDnCreate` return `NOT_INITIALIZED`
//! (status 1) when threads create handles at the same time (cuSOLVER
//! initialises cuBLAS internally, so the two race each other too). Plans
//! built concurrently by Fuel, Lightbulb or `cargo test` threads therefore
//! failed intermittently with `KernelInternal(-1)`. Creation happens once
//! per plan, so every creation site in this workspace goes through this one
//! lock instead of retrying.

extern crate std;

use std::sync::{Mutex, MutexGuard};

use crate::{cublasCreate_v2, cublasHandle_t, cusolverDnCreate, cusolverDnHandle_t};

static CREATE_LOCK: Mutex<()> = Mutex::new(());

fn guard() -> MutexGuard<'static, ()> {
    // The lock guards no data, so a poisoned lock is still safe to use.
    CREATE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Create a cuBLAS handle, serialised with every other handle creation.
/// `Err` carries the raw cuBLAS status.
pub fn create_cublas() -> Result<cublasHandle_t, i32> {
    let _g = guard();
    let mut handle: cublasHandle_t = core::ptr::null_mut();
    let status = unsafe { cublasCreate_v2(&mut handle as *mut _) };
    if status != 0 || handle.is_null() {
        return Err(if status != 0 { status } else { 1 });
    }
    Ok(handle)
}

/// Create a cuSOLVER dense handle, serialised with every other handle
/// creation. `Err` carries the raw cuSOLVER status.
pub fn create_cusolver_dn() -> Result<cusolverDnHandle_t, i32> {
    let _g = guard();
    let mut handle: cusolverDnHandle_t = core::ptr::null_mut();
    let status = unsafe { cusolverDnCreate(&mut handle as *mut _) };
    if status != 0 || handle.is_null() {
        return Err(if status != 0 { status } else { 1 });
    }
    Ok(handle)
}
