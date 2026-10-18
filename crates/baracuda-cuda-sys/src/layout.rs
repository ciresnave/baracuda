//! Compile-time layout checks for the `#[repr(C)]` mirrors of CUDA C structs.
//!
//! Each line asserts `size_of` / `align_of` of a Rust mirror against the value
//! `cl.exe` reports for the C struct of the same name from the CUDA 13.3 headers
//! (`cuda.h`, `cuda_runtime_api.h`; x86_64 Windows/MSVC, LP64-equivalent for these
//! types). A drift (wrong field type, missing padding, changed header) fails the build.
//!
//! Not asserted (no C counterpart verifiable here): `CUeglFrame` (needs `cudaEGL.h`),
//! `CUmemAllocationPropFlags` (baracuda-only helper), `cudaMemAllocationProp`
//! (not declared by the 13.3 runtime headers).

use core::mem::{align_of, size_of};

macro_rules! assert_layout {
    ($t:ty, $size:expr, $align:expr) => {
        const _: () = {
            assert!(
                size_of::<$t>() == $size,
                concat!("size_of ", stringify!($t))
            );
            assert!(
                align_of::<$t>() == $align,
                concat!("align_of ", stringify!($t))
            );
        };
    };
}

/// For structs ending in `long reserved[N]`: `long` is 4 bytes on Windows and 8 on
/// LP64 Linux, so the total is `base + N * size_of::<c_long>()`.
macro_rules! assert_layout_long {
    ($t:ty, $base:expr, $align:expr, $n:expr) => {
        const _: () = {
            assert!(
                size_of::<$t>() == $base + $n * size_of::<core::ffi::c_long>(),
                concat!("size_of ", stringify!($t))
            );
            assert!(
                align_of::<$t>() == $align,
                concat!("align_of ", stringify!($t))
            );
        };
    };
}

assert_layout!(crate::types::CUdevice, 4, 4);
assert_layout!(crate::types::CUdeviceptr, 8, 8);
assert_layout!(crate::types::CUarrayMapInfo, 96, 8);
assert_layout!(crate::types::CUDA_ARRAY_SPARSE_PROPERTIES, 48, 8);
assert_layout!(crate::types::CUDA_ARRAY_MEMORY_REQUIREMENTS, 32, 8);
assert_layout!(crate::types::CUlaunchAttributeValue, 64, 8);
assert_layout!(crate::types::CUaccessPolicyWindow, 32, 8);
assert_layout!(crate::types::CUlaunchAttribute, 72, 8);
assert_layout!(crate::types::CUlaunchConfig, 56, 8);
assert_layout!(crate::types::CUDA_MEMCPY2D, 128, 8);
assert_layout!(crate::types::CUDA_KERNEL_NODE_PARAMS, 72, 8);
assert_layout!(crate::types::CUDA_MEMSET_NODE_PARAMS, 40, 8);
assert_layout!(crate::types::CUDA_HOST_NODE_PARAMS, 16, 8);
assert_layout!(crate::types::CUipcEventHandle, 64, 1);
assert_layout!(crate::types::CUipcMemHandle, 64, 1);
assert_layout!(crate::types::CUDA_CONDITIONAL_NODE_PARAMS, 32, 8);
assert_layout!(crate::types::CUgraphNodeParams, 256, 8);
assert_layout!(crate::types::CUgraphEdgeData, 8, 1);
assert_layout!(crate::types::CUmulticastObjectProp, 32, 8);
assert_layout!(crate::types::CUdevSmResource, 16, 4);
assert_layout!(crate::types::CUdevResource, 144, 8);
assert_layout!(crate::types::CUDA_ARRAY_DESCRIPTOR, 24, 8);
assert_layout!(crate::types::CUDA_RESOURCE_DESC, 144, 8);
assert_layout!(crate::types::CUDA_TEXTURE_DESC, 104, 4);
assert_layout!(crate::types::CUmemLocation, 8, 4);
assert_layout!(crate::types::CUmemAllocationProp, 32, 8);
assert_layout!(crate::types::CUmemAccessDesc, 12, 4);
assert_layout!(crate::types::CUmemPoolProps, 88, 8);
assert_layout!(crate::types::CUmemPoolPtrExportData, 64, 1);
assert_layout!(crate::types::CUgraphExecUpdateResultInfo, 24, 8);
assert_layout!(crate::types::CUDA_MEM_ALLOC_NODE_PARAMS, 120, 8);
assert_layout!(crate::types::CUstreamBatchMemOpParams, 48, 8);
assert_layout!(crate::types::CUDA_BATCH_MEM_OP_NODE_PARAMS, 32, 8);
assert_layout!(crate::types::CUDA_EXTERNAL_MEMORY_HANDLE_DESC, 104, 8);
assert_layout!(crate::types::CUDA_EXTERNAL_MEMORY_BUFFER_DESC, 88, 8);
assert_layout!(crate::types::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC, 96, 8);
assert_layout!(crate::types::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, 144, 8);
assert_layout!(crate::types::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, 144, 8);
assert_layout!(crate::types::CUDA_ARRAY3D_DESCRIPTOR, 40, 8);
assert_layout!(crate::types::CUDA_MEMCPY3D, 200, 8);
assert_layout!(crate::runtime::types::cudaChannelFormatDesc, 20, 4);
assert_layout!(crate::runtime::types::cudaExtent, 24, 8);
assert_layout!(crate::runtime::types::cudaPos, 24, 8);
assert_layout!(crate::runtime::types::cudaPitchedPtr, 32, 8);
assert_layout!(crate::runtime::types::cudaMemcpy3DParms, 160, 8);
assert_layout!(crate::runtime::types::cudaTextureDesc, 72, 4);
assert_layout!(crate::runtime::types::cudaResourceViewDesc, 112, 8);
assert_layout!(crate::runtime::types::cudaLaunchAttribute, 72, 8);
assert_layout!(crate::runtime::types::cudaLaunchConfig_t, 56, 8);
assert_layout!(crate::runtime::types::dim3, 12, 4);
assert_layout!(crate::runtime::types::cudaMemLocation, 8, 4);
assert_layout!(crate::runtime::types::cudaMemAccessDesc, 12, 4);
assert_layout!(crate::runtime::types::cudaMemPoolProps, 88, 8);
assert_layout!(crate::runtime::types::cudaMemPoolPtrExportData, 64, 1);
assert_layout!(crate::runtime::types::cudaKernelNodeParams, 56, 8);
assert_layout!(crate::runtime::types::cudaMemsetParams, 40, 8);
assert_layout!(crate::runtime::types::cudaHostNodeParams, 16, 8);
assert_layout!(crate::runtime::types::cudaMemAllocNodeParams, 120, 8);
assert_layout_long!(crate::runtime::types::cudaPointerAttributes, 24, 8, 8);
assert_layout!(crate::runtime::types::cudaFuncAttributes, 144, 8);
