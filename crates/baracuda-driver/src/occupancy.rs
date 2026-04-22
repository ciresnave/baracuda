//! Occupancy calculators — how many blocks can a kernel fit per SM, and
//! what block size maximizes utilization.
//!
//! These are essential for kernel tuning: before launching, ask the driver
//! "given this kernel and this dynamic-shared-memory budget, what's the
//! best grid/block shape?" The results depend on the target device's
//! register file, shared-memory size, and the kernel's own resource use.

use baracuda_cuda_sys::driver;

use crate::error::{check, Result};
use crate::module::Function;

/// How many blocks of `block_size` threads (using `dynamic_smem_bytes` of
/// dynamic shared memory per block) can run concurrently on each SM of the
/// current device.
pub fn max_active_blocks_per_multiprocessor(
    func: &Function,
    block_size: i32,
    dynamic_smem_bytes: usize,
) -> Result<i32> {
    let d = driver()?;
    let cu = d.cu_occupancy_max_active_blocks_per_multiprocessor()?;
    let mut n: core::ffi::c_int = 0;
    // SAFETY: `func.as_raw()` is a live kernel handle; `&mut n` is writable.
    check(unsafe { cu(&mut n, func.as_raw(), block_size, dynamic_smem_bytes) })?;
    Ok(n)
}

/// As above, but with an explicit flag bitmask (see
/// `CU_OCCUPANCY_*` in NVIDIA's headers). Passing `0` matches the
/// no-flags version.
pub fn max_active_blocks_per_multiprocessor_with_flags(
    func: &Function,
    block_size: i32,
    dynamic_smem_bytes: usize,
    flags: u32,
) -> Result<i32> {
    let d = driver()?;
    let cu = d.cu_occupancy_max_active_blocks_per_multiprocessor_with_flags()?;
    let mut n: core::ffi::c_int = 0;
    check(unsafe { cu(&mut n, func.as_raw(), block_size, dynamic_smem_bytes, flags) })?;
    Ok(n)
}

/// Block size that maximises occupancy for `func`, assuming the given
/// fixed `dynamic_smem_bytes`. Returns `(min_grid_size, optimal_block_size)`:
/// launch `min_grid_size` blocks of `optimal_block_size` threads to cover
/// the device with peak SM-utilization.
///
/// `block_size_limit` clamps the returned block size; pass `0` for the
/// device's documented maximum.
pub fn max_potential_block_size(
    func: &Function,
    dynamic_smem_bytes: usize,
    block_size_limit: i32,
) -> Result<(i32, i32)> {
    let d = driver()?;
    let cu = d.cu_occupancy_max_potential_block_size()?;
    let mut min_grid: core::ffi::c_int = 0;
    let mut block: core::ffi::c_int = 0;
    // SAFETY: both output pointers are writable; passing `None` as the
    // variable-dynamic-smem-size callback means dynamic_smem_bytes is taken
    // as a fixed value.
    check(unsafe {
        cu(
            &mut min_grid,
            &mut block,
            func.as_raw(),
            None,
            dynamic_smem_bytes,
            block_size_limit,
        )
    })?;
    Ok((min_grid, block))
}

/// Given `num_blocks` concurrent blocks per SM with `block_size` threads
/// each, how much dynamic shared memory (bytes) can each block still
/// allocate without losing occupancy.
///
/// Useful for tiling kernels that grow their shared-memory usage up to the
/// point occupancy drops.
pub fn available_dynamic_smem_per_block(
    func: &Function,
    num_blocks: i32,
    block_size: i32,
) -> Result<usize> {
    let d = driver()?;
    let cu = d.cu_occupancy_available_dynamic_smem_per_block()?;
    let mut bytes: usize = 0;
    check(unsafe { cu(&mut bytes, func.as_raw(), num_blocks, block_size) })?;
    Ok(bytes)
}
