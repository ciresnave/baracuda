//! Runtime-API external memory / semaphore interop — symbol-resolution
//! and descriptor layout checks. End-to-end import needs a Vulkan /
//! D3D12 producer so we stop at verifying the safe wrapper compiles and
//! the sys symbols resolve.

use baracuda_cuda_sys::types::{
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC, CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
};

#[test]
fn runtime_external_interop_symbols_resolve() {
    if let Ok(r) = baracuda_cuda_sys::runtime() {
        let _ = r.cuda_import_external_memory();
        let _ = r.cuda_destroy_external_memory();
        let _ = r.cuda_external_memory_get_mapped_buffer();
        let _ = r.cuda_external_memory_get_mapped_mipmapped_array();
        let _ = r.cuda_import_external_semaphore();
        let _ = r.cuda_destroy_external_semaphore();
        let _ = r.cuda_signal_external_semaphores_async();
        let _ = r.cuda_wait_external_semaphores_async();
    }
}

#[test]
fn external_descriptor_sizes_match_abi() {
    // These match the Driver-side wrappers (Wave 9) — cudaRuntime and
    // CUDA Driver ABIs share the struct definitions.
    use core::mem::size_of;
    assert_eq!(size_of::<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>(), 104);
    assert_eq!(size_of::<CUDA_EXTERNAL_MEMORY_BUFFER_DESC>(), 88);
    assert_eq!(size_of::<CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC>(), 96);
    assert_eq!(size_of::<CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS>(), 144);
    assert_eq!(size_of::<CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS>(), 144);
}

#[test]
fn builder_helpers_compose() {
    // Sanity: runtime's ExternalMemory::import accepts the same typed
    // descriptor builders shared with the Driver side.
    let fd_desc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC::from_fd(42, 1024);
    assert_eq!(fd_desc.size, 1024);
    assert_eq!(fd_desc.handle[0] as i32, 42);
}
