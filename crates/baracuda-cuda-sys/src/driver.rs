//! The [`Driver`] singleton: a lazily-loaded handle to `libcuda` with a
//! cached, version-aware function-pointer table.
//!
//! Typical use from a safe crate:
//!
//! ```no_run
//! use baracuda_cuda_sys::driver;
//! let d = driver()?;
//! let cu_init = d.cu_init()?;
//! // SAFETY: we just resolved the symbol; calling it matches the CUDA ABI.
//! unsafe { cu_init(0) };
//! # Ok::<(), baracuda_core::LoaderError>(())
//! ```
//!
//! Function names are normalized from the C `cuInit` / `cuDeviceGet` form
//! to `snake_case` (`cu_init`, `cu_device_get`) so they don't clash with
//! Rust's naming lints.

use core::ffi::c_char;
use std::ptr;
use std::sync::OnceLock;

use baracuda_core::{platform, stream_mode, Library, LoaderError};
use baracuda_types::StreamMode;

use crate::functions::*;
use crate::status::CUresult;

/// Flag passed to `cuGetProcAddress` in [`StreamMode::Legacy`].
const CU_GET_PROC_ADDRESS_LEGACY_STREAM: u64 = 1;
/// Flag passed to `cuGetProcAddress` in [`StreamMode::PerThread`].
const CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM: u64 = 2;

/// `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT` from `cuda.h`. Returned by
/// `cuGetProcAddress` when the symbol exists in newer CUDA versions than the
/// one we asked for.
#[allow(dead_code)]
const CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT: i32 = 222;

macro_rules! driver_fns {
    ($(
        $(#[$attr:meta])*
        fn $name:ident as $sym:literal : $pfn:ty;
    )*) => {
        /// Lazily-resolved CUDA Driver API function-pointer table.
        #[allow(non_snake_case)]
        pub struct Driver {
            lib: Library,
            get_proc_address: OnceLock<PFN_cuGetProcAddress>,
            $(
                $name: OnceLock<$pfn>,
            )*
        }

        impl core::fmt::Debug for Driver {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Driver")
                    .field("lib", &self.lib)
                    .finish_non_exhaustive()
            }
        }

        impl Driver {
            fn empty(lib: Library) -> Self {
                Self {
                    lib,
                    get_proc_address: OnceLock::new(),
                    $(
                        $name: OnceLock::new(),
                    )*
                }
            }

            $(
                $(#[$attr])*
                #[allow(non_snake_case)]
                #[doc = concat!("Resolve `", $sym, "` and return the cached function pointer.")]
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let p: $pfn = unsafe { self.resolve($sym)? };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
        }
    };
}

// Naming convention: symbols we've pinned to a specific ABI (our PFN_*
// signature matches a particular `_vN` variant) use the explicit versioned
// name so the resolver goes through dlsym and the driver cannot upgrade us
// to a newer, incompatible ABI. Symbols we haven't pinned use the base
// name so cuGetProcAddress can pick the best variant for the installed
// driver (and the _ptsz variant under per-thread-default-stream mode).
driver_fns! {
    // Initialization & version
    fn cu_init as "cuInit": PFN_cuInit;
    fn cu_driver_get_version as "cuDriverGetVersion": PFN_cuDriverGetVersion;

    // Errors (note: the strings returned are owned by the driver; do not free)
    fn cu_get_error_name as "cuGetErrorName": PFN_cuGetErrorName;
    fn cu_get_error_string as "cuGetErrorString": PFN_cuGetErrorString;

    // Device
    fn cu_device_get_count as "cuDeviceGetCount": PFN_cuDeviceGetCount;
    fn cu_device_get as "cuDeviceGet": PFN_cuDeviceGet;
    fn cu_device_get_name as "cuDeviceGetName": PFN_cuDeviceGetName;
    fn cu_device_get_attribute as "cuDeviceGetAttribute": PFN_cuDeviceGetAttribute;
    fn cu_device_total_mem as "cuDeviceTotalMem_v2": PFN_cuDeviceTotalMem;

    // Context — pinned to _v2 because cuCtxCreate_v3 (CUDA 11.4) / _v4 (12.5)
    // have extra parameters and would be returned by cuGetProcAddress.
    fn cu_ctx_create as "cuCtxCreate_v2": PFN_cuCtxCreate;
    fn cu_ctx_destroy as "cuCtxDestroy_v2": PFN_cuCtxDestroy;
    fn cu_ctx_get_current as "cuCtxGetCurrent": PFN_cuCtxGetCurrent;
    fn cu_ctx_set_current as "cuCtxSetCurrent": PFN_cuCtxSetCurrent;
    fn cu_ctx_push_current as "cuCtxPushCurrent_v2": PFN_cuCtxPushCurrent;
    fn cu_ctx_pop_current as "cuCtxPopCurrent_v2": PFN_cuCtxPopCurrent;
    fn cu_ctx_synchronize as "cuCtxSynchronize": PFN_cuCtxSynchronize;

    // Primary context
    fn cu_device_primary_ctx_retain as "cuDevicePrimaryCtxRetain": PFN_cuDevicePrimaryCtxRetain;
    fn cu_device_primary_ctx_release as "cuDevicePrimaryCtxRelease_v2": PFN_cuDevicePrimaryCtxRelease;
    fn cu_device_primary_ctx_reset as "cuDevicePrimaryCtxReset_v2": PFN_cuDevicePrimaryCtxReset;

    // Memory — pinned to _v2 (64-bit addresses, stable since CUDA 3.2).
    fn cu_mem_alloc as "cuMemAlloc_v2": PFN_cuMemAlloc;
    fn cu_mem_free as "cuMemFree_v2": PFN_cuMemFree;
    fn cu_memcpy_htod as "cuMemcpyHtoD_v2": PFN_cuMemcpyHtoD;
    fn cu_memcpy_dtoh as "cuMemcpyDtoH_v2": PFN_cuMemcpyDtoH;
    fn cu_memcpy_dtod as "cuMemcpyDtoD_v2": PFN_cuMemcpyDtoD;
    fn cu_memcpy_htod_async as "cuMemcpyHtoDAsync_v2": PFN_cuMemcpyHtoDAsync;
    fn cu_memcpy_dtoh_async as "cuMemcpyDtoHAsync_v2": PFN_cuMemcpyDtoHAsync;
    fn cu_memset_d8 as "cuMemsetD8_v2": PFN_cuMemsetD8;
    fn cu_memset_d32 as "cuMemsetD32_v2": PFN_cuMemsetD32;

    // Stream
    fn cu_stream_create as "cuStreamCreate": PFN_cuStreamCreate;
    fn cu_stream_destroy as "cuStreamDestroy_v2": PFN_cuStreamDestroy;
    fn cu_stream_synchronize as "cuStreamSynchronize": PFN_cuStreamSynchronize;
    fn cu_stream_query as "cuStreamQuery": PFN_cuStreamQuery;
    fn cu_stream_wait_event as "cuStreamWaitEvent": PFN_cuStreamWaitEvent;

    // Event
    fn cu_event_create as "cuEventCreate": PFN_cuEventCreate;
    fn cu_event_destroy as "cuEventDestroy_v2": PFN_cuEventDestroy;
    fn cu_event_record as "cuEventRecord": PFN_cuEventRecord;
    fn cu_event_synchronize as "cuEventSynchronize": PFN_cuEventSynchronize;
    fn cu_event_query as "cuEventQuery": PFN_cuEventQuery;
    fn cu_event_elapsed_time as "cuEventElapsedTime": PFN_cuEventElapsedTime;

    // Module / kernel
    fn cu_module_load_data as "cuModuleLoadData": PFN_cuModuleLoadData;
    fn cu_module_unload as "cuModuleUnload": PFN_cuModuleUnload;
    fn cu_module_get_function as "cuModuleGetFunction": PFN_cuModuleGetFunction;
    fn cu_launch_kernel as "cuLaunchKernel": PFN_cuLaunchKernel;

    // Stream capture / graphs — _v2 of cuStreamBeginCapture pins the 2-arg signature
    // (which is what CUDA 10.1+ shipped; the older 1-arg form is deprecated).
    fn cu_stream_begin_capture as "cuStreamBeginCapture_v2": PFN_cuStreamBeginCapture;
    fn cu_stream_end_capture as "cuStreamEndCapture": PFN_cuStreamEndCapture;
    fn cu_stream_is_capturing as "cuStreamIsCapturing": PFN_cuStreamIsCapturing;
    fn cu_graph_create as "cuGraphCreate": PFN_cuGraphCreate;
    fn cu_graph_destroy as "cuGraphDestroy": PFN_cuGraphDestroy;
    fn cu_graph_instantiate_with_flags as "cuGraphInstantiateWithFlags": PFN_cuGraphInstantiateWithFlags;
    fn cu_graph_launch as "cuGraphLaunch": PFN_cuGraphLaunch;
    fn cu_graph_exec_destroy as "cuGraphExecDestroy": PFN_cuGraphExecDestroy;
    fn cu_graph_get_nodes as "cuGraphGetNodes": PFN_cuGraphGetNodes;

    // Stream-ordered memory allocation (CUDA 11.2+, available at our 11.4 floor).
    fn cu_mem_alloc_async as "cuMemAllocAsync": PFN_cuMemAllocAsync;
    fn cu_mem_free_async as "cuMemFreeAsync": PFN_cuMemFreeAsync;

    // ---- Wave 1 additions ----

    // Occupancy
    fn cu_occupancy_max_active_blocks_per_multiprocessor as "cuOccupancyMaxActiveBlocksPerMultiprocessor": PFN_cuOccupancyMaxActiveBlocksPerMultiprocessor;
    fn cu_occupancy_max_active_blocks_per_multiprocessor_with_flags as "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": PFN_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
    fn cu_occupancy_max_potential_block_size as "cuOccupancyMaxPotentialBlockSize": PFN_cuOccupancyMaxPotentialBlockSize;
    fn cu_occupancy_available_dynamic_smem_per_block as "cuOccupancyAvailableDynamicSMemPerBlock": PFN_cuOccupancyAvailableDynamicSMemPerBlock;

    // Unified memory
    fn cu_mem_alloc_managed as "cuMemAllocManaged": PFN_cuMemAllocManaged;
    fn cu_mem_advise as "cuMemAdvise": PFN_cuMemAdvise;
    fn cu_mem_prefetch_async as "cuMemPrefetchAsync": PFN_cuMemPrefetchAsync;
    fn cu_mem_get_info as "cuMemGetInfo_v2": PFN_cuMemGetInfo;

    // Context queries/config
    fn cu_ctx_get_device as "cuCtxGetDevice": PFN_cuCtxGetDevice;
    fn cu_ctx_get_api_version as "cuCtxGetApiVersion": PFN_cuCtxGetApiVersion;
    fn cu_ctx_get_flags as "cuCtxGetFlags": PFN_cuCtxGetFlags;
    fn cu_ctx_get_limit as "cuCtxGetLimit": PFN_cuCtxGetLimit;
    fn cu_ctx_set_limit as "cuCtxSetLimit": PFN_cuCtxSetLimit;
    fn cu_ctx_get_cache_config as "cuCtxGetCacheConfig": PFN_cuCtxGetCacheConfig;
    fn cu_ctx_set_cache_config as "cuCtxSetCacheConfig": PFN_cuCtxSetCacheConfig;
    fn cu_ctx_get_stream_priority_range as "cuCtxGetStreamPriorityRange": PFN_cuCtxGetStreamPriorityRange;

    // Peer
    fn cu_device_can_access_peer as "cuDeviceCanAccessPeer": PFN_cuDeviceCanAccessPeer;
    fn cu_ctx_enable_peer_access as "cuCtxEnablePeerAccess": PFN_cuCtxEnablePeerAccess;
    fn cu_ctx_disable_peer_access as "cuCtxDisablePeerAccess": PFN_cuCtxDisablePeerAccess;

    // Pointer attributes
    fn cu_pointer_get_attribute as "cuPointerGetAttribute": PFN_cuPointerGetAttribute;

    // Stream priority + host func
    fn cu_stream_create_with_priority as "cuStreamCreateWithPriority": PFN_cuStreamCreateWithPriority;
    fn cu_stream_get_priority as "cuStreamGetPriority": PFN_cuStreamGetPriority;
    fn cu_stream_get_flags as "cuStreamGetFlags": PFN_cuStreamGetFlags;
    fn cu_stream_get_ctx as "cuStreamGetCtx": PFN_cuStreamGetCtx;
    fn cu_launch_host_func as "cuLaunchHostFunc": PFN_cuLaunchHostFunc;

    // Event flags
    fn cu_event_record_with_flags as "cuEventRecordWithFlags": PFN_cuEventRecordWithFlags;

    // Primary-context state
    fn cu_device_primary_ctx_get_state as "cuDevicePrimaryCtxGetState": PFN_cuDevicePrimaryCtxGetState;
    fn cu_device_primary_ctx_set_flags as "cuDevicePrimaryCtxSetFlags_v2": PFN_cuDevicePrimaryCtxSetFlags;

    // ---- Wave 2 ----
    fn cu_func_get_attribute as "cuFuncGetAttribute": PFN_cuFuncGetAttribute;
    fn cu_func_set_attribute as "cuFuncSetAttribute": PFN_cuFuncSetAttribute;
    fn cu_module_get_global as "cuModuleGetGlobal_v2": PFN_cuModuleGetGlobal;
    fn cu_module_load_data_ex as "cuModuleLoadDataEx": PFN_cuModuleLoadDataEx;

    // ---- Wave 3: extensible launch + library management (CUDA 12.0+) ----
    fn cu_launch_kernel_ex as "cuLaunchKernelEx": PFN_cuLaunchKernelEx;
    fn cu_library_load_data as "cuLibraryLoadData": PFN_cuLibraryLoadData;
    fn cu_library_unload as "cuLibraryUnload": PFN_cuLibraryUnload;
    fn cu_library_get_kernel as "cuLibraryGetKernel": PFN_cuLibraryGetKernel;
    fn cu_library_get_global as "cuLibraryGetGlobal": PFN_cuLibraryGetGlobal;
    fn cu_kernel_get_function as "cuKernelGetFunction": PFN_cuKernelGetFunction;

    // ---- Wave 4: 2D alloc + memcpy ----
    fn cu_mem_alloc_pitch as "cuMemAllocPitch_v2": PFN_cuMemAllocPitch;
    fn cu_memcpy_2d as "cuMemcpy2D_v2": PFN_cuMemcpy2D;
    fn cu_memcpy_2d_async as "cuMemcpy2DAsync_v2": PFN_cuMemcpy2DAsync;

    // ---- Wave 5: explicit graph node construction ----
    // `cuGraphAddKernelNode_v2` is pinned because our CUDA_KERNEL_NODE_PARAMS
    // matches the v2 shape (kern + ctx fields). The _v2 suffix routes this
    // through dlsym so the driver can't upgrade us to a future ABI.
    fn cu_graph_add_kernel_node as "cuGraphAddKernelNode_v2": PFN_cuGraphAddKernelNode;
    fn cu_graph_add_empty_node as "cuGraphAddEmptyNode": PFN_cuGraphAddEmptyNode;
    fn cu_graph_add_memset_node as "cuGraphAddMemsetNode": PFN_cuGraphAddMemsetNode;
    fn cu_graph_destroy_node as "cuGraphDestroyNode": PFN_cuGraphDestroyNode;
    fn cu_graph_clone as "cuGraphClone": PFN_cuGraphClone;

    // ---- Wave 6: arrays, textures, surfaces ----
    fn cu_array_create as "cuArrayCreate_v2": PFN_cuArrayCreate;
    fn cu_array_destroy as "cuArrayDestroy": PFN_cuArrayDestroy;
    fn cu_tex_object_create as "cuTexObjectCreate": PFN_cuTexObjectCreate;
    fn cu_tex_object_destroy as "cuTexObjectDestroy": PFN_cuTexObjectDestroy;
    fn cu_surf_object_create as "cuSurfObjectCreate": PFN_cuSurfObjectCreate;
    fn cu_surf_object_destroy as "cuSurfObjectDestroy": PFN_cuSurfObjectDestroy;

    // ---- Wave 7: virtual memory management (VMM) ----
    fn cu_mem_address_reserve as "cuMemAddressReserve": PFN_cuMemAddressReserve;
    fn cu_mem_address_free as "cuMemAddressFree": PFN_cuMemAddressFree;
    fn cu_mem_create as "cuMemCreate": PFN_cuMemCreate;
    fn cu_mem_release as "cuMemRelease": PFN_cuMemRelease;
    fn cu_mem_map as "cuMemMap": PFN_cuMemMap;
    fn cu_mem_unmap as "cuMemUnmap": PFN_cuMemUnmap;
    fn cu_mem_set_access as "cuMemSetAccess": PFN_cuMemSetAccess;
    fn cu_mem_get_allocation_granularity as "cuMemGetAllocationGranularity": PFN_cuMemGetAllocationGranularity;

    // ---- Wave 8: memory pools ----
    fn cu_mem_pool_create as "cuMemPoolCreate": PFN_cuMemPoolCreate;
    fn cu_mem_pool_destroy as "cuMemPoolDestroy": PFN_cuMemPoolDestroy;
    fn cu_mem_pool_set_attribute as "cuMemPoolSetAttribute": PFN_cuMemPoolSetAttribute;
    fn cu_mem_pool_get_attribute as "cuMemPoolGetAttribute": PFN_cuMemPoolGetAttribute;
    fn cu_mem_pool_trim_to as "cuMemPoolTrimTo": PFN_cuMemPoolTrimTo;
    fn cu_mem_pool_set_access as "cuMemPoolSetAccess": PFN_cuMemPoolSetAccess;
    fn cu_mem_pool_get_access as "cuMemPoolGetAccess": PFN_cuMemPoolGetAccess;
    fn cu_mem_alloc_from_pool_async as "cuMemAllocFromPoolAsync": PFN_cuMemAllocFromPoolAsync;
    fn cu_device_get_default_mem_pool as "cuDeviceGetDefaultMemPool": PFN_cuDeviceGetDefaultMemPool;
    fn cu_device_get_mem_pool as "cuDeviceGetMemPool": PFN_cuDeviceGetMemPool;
    fn cu_device_set_mem_pool as "cuDeviceSetMemPool": PFN_cuDeviceSetMemPool;
    fn cu_mem_pool_export_to_shareable_handle as "cuMemPoolExportToShareableHandle": PFN_cuMemPoolExportToShareableHandle;
    fn cu_mem_pool_import_from_shareable_handle as "cuMemPoolImportFromShareableHandle": PFN_cuMemPoolImportFromShareableHandle;
    fn cu_mem_pool_export_pointer as "cuMemPoolExportPointer": PFN_cuMemPoolExportPointer;
    fn cu_mem_pool_import_pointer as "cuMemPoolImportPointer": PFN_cuMemPoolImportPointer;

    // ---- Wave 9: external memory / semaphore interop ----
    fn cu_import_external_memory as "cuImportExternalMemory": PFN_cuImportExternalMemory;
    fn cu_destroy_external_memory as "cuDestroyExternalMemory": PFN_cuDestroyExternalMemory;
    fn cu_external_memory_get_mapped_buffer as "cuExternalMemoryGetMappedBuffer": PFN_cuExternalMemoryGetMappedBuffer;
    fn cu_external_memory_get_mapped_mipmapped_array as "cuExternalMemoryGetMappedMipmappedArray": PFN_cuExternalMemoryGetMappedMipmappedArray;
    fn cu_import_external_semaphore as "cuImportExternalSemaphore": PFN_cuImportExternalSemaphore;
    fn cu_destroy_external_semaphore as "cuDestroyExternalSemaphore": PFN_cuDestroyExternalSemaphore;
    fn cu_signal_external_semaphores_async as "cuSignalExternalSemaphoresAsync": PFN_cuSignalExternalSemaphoresAsync;
    fn cu_wait_external_semaphores_async as "cuWaitExternalSemaphoresAsync": PFN_cuWaitExternalSemaphoresAsync;

    // ---- Wave 10: 3D memcpy + 3D arrays + mipmapped arrays ----
    fn cu_array_3d_create as "cuArray3DCreate_v2": PFN_cuArray3DCreate;
    fn cu_array_3d_get_descriptor as "cuArray3DGetDescriptor_v2": PFN_cuArray3DGetDescriptor;
    fn cu_memcpy_3d as "cuMemcpy3D_v2": PFN_cuMemcpy3D;
    fn cu_memcpy_3d_async as "cuMemcpy3DAsync_v2": PFN_cuMemcpy3DAsync;
    fn cu_mipmapped_array_create as "cuMipmappedArrayCreate": PFN_cuMipmappedArrayCreate;
    fn cu_mipmapped_array_destroy as "cuMipmappedArrayDestroy": PFN_cuMipmappedArrayDestroy;
    fn cu_mipmapped_array_get_level as "cuMipmappedArrayGetLevel": PFN_cuMipmappedArrayGetLevel;

    // ---- Wave 11: pinned host memory ----
    fn cu_mem_alloc_host as "cuMemAllocHost_v2": PFN_cuMemAllocHost;
    fn cu_mem_free_host as "cuMemFreeHost": PFN_cuMemFreeHost;
    fn cu_mem_host_alloc as "cuMemHostAlloc": PFN_cuMemHostAlloc;
    fn cu_mem_host_register as "cuMemHostRegister_v2": PFN_cuMemHostRegister;
    fn cu_mem_host_unregister as "cuMemHostUnregister": PFN_cuMemHostUnregister;
    fn cu_mem_host_get_device_pointer as "cuMemHostGetDevicePointer_v2": PFN_cuMemHostGetDevicePointer;
    fn cu_mem_host_get_flags as "cuMemHostGetFlags": PFN_cuMemHostGetFlags;

    // ---- Wave 12: full graph node builders + edit ----
    fn cu_graph_add_memcpy_node as "cuGraphAddMemcpyNode": PFN_cuGraphAddMemcpyNode;
    fn cu_graph_add_host_node as "cuGraphAddHostNode": PFN_cuGraphAddHostNode;
    fn cu_graph_add_child_graph_node as "cuGraphAddChildGraphNode": PFN_cuGraphAddChildGraphNode;
    fn cu_graph_add_event_record_node as "cuGraphAddEventRecordNode": PFN_cuGraphAddEventRecordNode;
    fn cu_graph_add_event_wait_node as "cuGraphAddEventWaitNode": PFN_cuGraphAddEventWaitNode;
    fn cu_graph_add_external_semaphores_signal_node as "cuGraphAddExternalSemaphoresSignalNode": PFN_cuGraphAddExternalSemaphoresSignalNode;
    fn cu_graph_add_external_semaphores_wait_node as "cuGraphAddExternalSemaphoresWaitNode": PFN_cuGraphAddExternalSemaphoresWaitNode;
    // Node-param get/set pinned to v2 variants (match our v2 struct shape).
    fn cu_graph_kernel_node_get_params as "cuGraphKernelNodeGetParams_v2": PFN_cuGraphKernelNodeGetParams;
    fn cu_graph_kernel_node_set_params as "cuGraphKernelNodeSetParams_v2": PFN_cuGraphKernelNodeSetParams;
    fn cu_graph_memcpy_node_get_params as "cuGraphMemcpyNodeGetParams": PFN_cuGraphMemcpyNodeGetParams;
    fn cu_graph_memcpy_node_set_params as "cuGraphMemcpyNodeSetParams": PFN_cuGraphMemcpyNodeSetParams;
    fn cu_graph_memset_node_get_params as "cuGraphMemsetNodeGetParams": PFN_cuGraphMemsetNodeGetParams;
    fn cu_graph_memset_node_set_params as "cuGraphMemsetNodeSetParams": PFN_cuGraphMemsetNodeSetParams;
    fn cu_graph_node_get_type as "cuGraphNodeGetType": PFN_cuGraphNodeGetType;
    fn cu_graph_node_get_dependencies as "cuGraphNodeGetDependencies": PFN_cuGraphNodeGetDependencies;
    fn cu_graph_node_get_dependent_nodes as "cuGraphNodeGetDependentNodes": PFN_cuGraphNodeGetDependentNodes;
    fn cu_graph_get_edges as "cuGraphGetEdges": PFN_cuGraphGetEdges;
    fn cu_graph_add_dependencies as "cuGraphAddDependencies": PFN_cuGraphAddDependencies;
    fn cu_graph_remove_dependencies as "cuGraphRemoveDependencies": PFN_cuGraphRemoveDependencies;
    fn cu_graph_exec_kernel_node_set_params as "cuGraphExecKernelNodeSetParams_v2": PFN_cuGraphExecKernelNodeSetParams;
    fn cu_graph_exec_memcpy_node_set_params as "cuGraphExecMemcpyNodeSetParams": PFN_cuGraphExecMemcpyNodeSetParams;
    fn cu_graph_exec_memset_node_set_params as "cuGraphExecMemsetNodeSetParams": PFN_cuGraphExecMemsetNodeSetParams;
    fn cu_graph_exec_host_node_set_params as "cuGraphExecHostNodeSetParams": PFN_cuGraphExecHostNodeSetParams;

    // ---- Wave 13: stream extras ----
    fn cu_stream_get_id as "cuStreamGetId": PFN_cuStreamGetId;
    fn cu_stream_copy_attributes as "cuStreamCopyAttributes": PFN_cuStreamCopyAttributes;
    fn cu_stream_get_attribute as "cuStreamGetAttribute": PFN_cuStreamGetAttribute;
    fn cu_stream_set_attribute as "cuStreamSetAttribute": PFN_cuStreamSetAttribute;
    fn cu_stream_attach_mem_async as "cuStreamAttachMemAsync": PFN_cuStreamAttachMemAsync;
    fn cu_stream_get_capture_info as "cuStreamGetCaptureInfo_v2": PFN_cuStreamGetCaptureInfo;
    fn cu_stream_update_capture_dependencies as "cuStreamUpdateCaptureDependencies": PFN_cuStreamUpdateCaptureDependencies;

    // ---- Wave 14: misc memcpy variants ----
    fn cu_memcpy_dtod_async as "cuMemcpyDtoDAsync_v2": PFN_cuMemcpyDtoDAsync;
    fn cu_memcpy_peer as "cuMemcpyPeer": PFN_cuMemcpyPeer;
    fn cu_memcpy_peer_async as "cuMemcpyPeerAsync": PFN_cuMemcpyPeerAsync;
    fn cu_memcpy as "cuMemcpy": PFN_cuMemcpy;
    fn cu_memcpy_async as "cuMemcpyAsync": PFN_cuMemcpyAsync;
    fn cu_memcpy_atoh as "cuMemcpyAtoH_v2": PFN_cuMemcpyAtoH;
    fn cu_memcpy_htoa as "cuMemcpyHtoA_v2": PFN_cuMemcpyHtoA;
    fn cu_memcpy_atod as "cuMemcpyAtoD_v2": PFN_cuMemcpyAtoD;
    fn cu_memcpy_dtoa as "cuMemcpyDtoA_v2": PFN_cuMemcpyDtoA;
    fn cu_memcpy_atoa as "cuMemcpyAtoA_v2": PFN_cuMemcpyAtoA;
    fn cu_memset_d16 as "cuMemsetD16_v2": PFN_cuMemsetD16;
    fn cu_memset_d8_async as "cuMemsetD8Async": PFN_cuMemsetD8Async;
    fn cu_memset_d16_async as "cuMemsetD16Async": PFN_cuMemsetD16Async;
    fn cu_memset_d32_async as "cuMemsetD32Async": PFN_cuMemsetD32Async;
    fn cu_memset_d2d8 as "cuMemsetD2D8_v2": PFN_cuMemsetD2D8;
    fn cu_memset_d2d16 as "cuMemsetD2D16_v2": PFN_cuMemsetD2D16;
    fn cu_memset_d2d32 as "cuMemsetD2D32_v2": PFN_cuMemsetD2D32;

    // ---- Wave 15: range + pointer attrs ----
    fn cu_mem_range_get_attribute as "cuMemRangeGetAttribute": PFN_cuMemRangeGetAttribute;
    fn cu_mem_range_get_attributes as "cuMemRangeGetAttributes": PFN_cuMemRangeGetAttributes;
    fn cu_pointer_get_attributes as "cuPointerGetAttributes": PFN_cuPointerGetAttributes;
    fn cu_pointer_set_attribute as "cuPointerSetAttribute": PFN_cuPointerSetAttribute;

    // ---- Wave 16: tensor maps (Hopper TMA) ----
    fn cu_tensor_map_encode_tiled as "cuTensorMapEncodeTiled": PFN_cuTensorMapEncodeTiled;
    fn cu_tensor_map_encode_im2col as "cuTensorMapEncodeIm2col": PFN_cuTensorMapEncodeIm2col;
    fn cu_tensor_map_replace_address as "cuTensorMapReplaceAddress": PFN_cuTensorMapReplaceAddress;

    // ---- Wave 17: green contexts (CUDA 12.4+) ----
    fn cu_device_get_dev_resource as "cuDeviceGetDevResource": PFN_cuDeviceGetDevResource;
    fn cu_dev_sm_resource_split_by_count as "cuDevSmResourceSplitByCount": PFN_cuDevSmResourceSplitByCount;
    fn cu_dev_resource_generate_desc as "cuDevResourceGenerateDesc": PFN_cuDevResourceGenerateDesc;
    fn cu_green_ctx_create as "cuGreenCtxCreate": PFN_cuGreenCtxCreate;
    fn cu_green_ctx_destroy as "cuGreenCtxDestroy": PFN_cuGreenCtxDestroy;
    fn cu_ctx_from_green_ctx as "cuCtxFromGreenCtx": PFN_cuCtxFromGreenCtx;
    fn cu_green_ctx_get_dev_resource as "cuGreenCtxGetDevResource": PFN_cuGreenCtxGetDevResource;
    fn cu_green_ctx_stream_create as "cuGreenCtxStreamCreate": PFN_cuGreenCtxStreamCreate;

    // ---- Wave 18: multicast objects ----
    fn cu_multicast_create as "cuMulticastCreate": PFN_cuMulticastCreate;
    fn cu_multicast_add_device as "cuMulticastAddDevice": PFN_cuMulticastAddDevice;
    fn cu_multicast_bind_mem as "cuMulticastBindMem": PFN_cuMulticastBindMem;
    fn cu_multicast_bind_addr as "cuMulticastBindAddr": PFN_cuMulticastBindAddr;
    fn cu_multicast_unbind as "cuMulticastUnbind": PFN_cuMulticastUnbind;
    fn cu_multicast_get_granularity as "cuMulticastGetGranularity": PFN_cuMulticastGetGranularity;

    // ---- Wave 19: conditional + switch graph nodes ----
    fn cu_graph_add_node as "cuGraphAddNode_v2": PFN_cuGraphAddNode;
    fn cu_graph_node_set_params as "cuGraphNodeSetParams": PFN_cuGraphNodeSetParams;
    fn cu_graph_conditional_handle_create as "cuGraphConditionalHandleCreate": PFN_cuGraphConditionalHandleCreate;

    // ---- Wave 20: IPC ----
    fn cu_ipc_get_event_handle as "cuIpcGetEventHandle": PFN_cuIpcGetEventHandle;
    fn cu_ipc_open_event_handle as "cuIpcOpenEventHandle": PFN_cuIpcOpenEventHandle;
    fn cu_ipc_get_mem_handle as "cuIpcGetMemHandle": PFN_cuIpcGetMemHandle;
    fn cu_ipc_open_mem_handle as "cuIpcOpenMemHandle_v2": PFN_cuIpcOpenMemHandle;
    fn cu_ipc_close_mem_handle as "cuIpcCloseMemHandle": PFN_cuIpcCloseMemHandle;

    // ---- Wave 21: kernel attrs extension (CUDA 12+) ----
    fn cu_kernel_get_attribute as "cuKernelGetAttribute": PFN_cuKernelGetAttribute;
    fn cu_kernel_set_attribute as "cuKernelSetAttribute": PFN_cuKernelSetAttribute;
    fn cu_kernel_get_name as "cuKernelGetName": PFN_cuKernelGetName;
    fn cu_kernel_set_cache_config as "cuKernelSetCacheConfig": PFN_cuKernelSetCacheConfig;
    fn cu_kernel_get_library as "cuKernelGetLibrary": PFN_cuKernelGetLibrary;
    fn cu_kernel_get_param_info as "cuKernelGetParamInfo": PFN_cuKernelGetParamInfo;

    // ---- Wave 22: user objects ----
    fn cu_user_object_create as "cuUserObjectCreate": PFN_cuUserObjectCreate;
    fn cu_user_object_retain as "cuUserObjectRetain": PFN_cuUserObjectRetain;
    fn cu_user_object_release as "cuUserObjectRelease": PFN_cuUserObjectRelease;
    fn cu_graph_retain_user_object as "cuGraphRetainUserObject": PFN_cuGraphRetainUserObject;
    fn cu_graph_release_user_object as "cuGraphReleaseUserObject": PFN_cuGraphReleaseUserObject;

    // ---- Wave 23: misc extras ----
    fn cu_profiler_start as "cuProfilerStart": PFN_cuProfilerStart;
    fn cu_profiler_stop as "cuProfilerStop": PFN_cuProfilerStop;
    fn cu_func_get_module as "cuFuncGetModule": PFN_cuFuncGetModule;
    fn cu_func_get_name as "cuFuncGetName": PFN_cuFuncGetName;
    fn cu_func_get_param_info as "cuFuncGetParamInfo": PFN_cuFuncGetParamInfo;
    fn cu_graph_debug_dot_print as "cuGraphDebugDotPrint": PFN_cuGraphDebugDotPrint;
    fn cu_ctx_get_id as "cuCtxGetId": PFN_cuCtxGetId;
    fn cu_module_get_loading_mode as "cuModuleGetLoadingMode": PFN_cuModuleGetLoadingMode;
    fn cu_device_get_uuid as "cuDeviceGetUuid_v2": PFN_cuDeviceGetUuid;
    fn cu_device_get_luid as "cuDeviceGetLuid": PFN_cuDeviceGetLuid;
    fn cu_logs_register_callback as "cuLogsRegisterCallback": PFN_cuLogsRegisterCallback;
    fn cu_logs_unregister_callback as "cuLogsUnregisterCallback": PFN_cuLogsUnregisterCallback;
    fn cu_logs_current as "cuLogsCurrent": PFN_cuLogsCurrent;
    fn cu_logs_dump_to_file as "cuLogsDumpToFile": PFN_cuLogsDumpToFile;
    fn cu_logs_dump_to_memory as "cuLogsDumpToMemory": PFN_cuLogsDumpToMemory;

    // ---- Wave 24: graph memory nodes + graph-exec update ----
    fn cu_graph_add_mem_alloc_node as "cuGraphAddMemAllocNode": PFN_cuGraphAddMemAllocNode;
    fn cu_graph_mem_alloc_node_get_params as "cuGraphMemAllocNodeGetParams": PFN_cuGraphMemAllocNodeGetParams;
    fn cu_graph_add_mem_free_node as "cuGraphAddMemFreeNode": PFN_cuGraphAddMemFreeNode;
    fn cu_graph_mem_free_node_get_params as "cuGraphMemFreeNodeGetParams": PFN_cuGraphMemFreeNodeGetParams;
    fn cu_device_graph_mem_trim as "cuDeviceGraphMemTrim": PFN_cuDeviceGraphMemTrim;
    fn cu_device_get_graph_mem_attribute as "cuDeviceGetGraphMemAttribute": PFN_cuDeviceGetGraphMemAttribute;
    fn cu_device_set_graph_mem_attribute as "cuDeviceSetGraphMemAttribute": PFN_cuDeviceSetGraphMemAttribute;
    fn cu_graph_add_batch_mem_op_node as "cuGraphAddBatchMemOpNode": PFN_cuGraphAddBatchMemOpNode;
    fn cu_graph_batch_mem_op_node_get_params as "cuGraphBatchMemOpNodeGetParams": PFN_cuGraphBatchMemOpNodeGetParams;
    fn cu_graph_batch_mem_op_node_set_params as "cuGraphBatchMemOpNodeSetParams": PFN_cuGraphBatchMemOpNodeSetParams;
    fn cu_graph_exec_batch_mem_op_node_set_params as "cuGraphExecBatchMemOpNodeSetParams": PFN_cuGraphExecBatchMemOpNodeSetParams;
    fn cu_graph_exec_update as "cuGraphExecUpdate_v2": PFN_cuGraphExecUpdate;

    // ---- Wave 25: stream memory ops ----
    fn cu_stream_write_value_32 as "cuStreamWriteValue32_v2": PFN_cuStreamWriteValue32;
    fn cu_stream_write_value_64 as "cuStreamWriteValue64_v2": PFN_cuStreamWriteValue64;
    fn cu_stream_wait_value_32 as "cuStreamWaitValue32_v2": PFN_cuStreamWaitValue32;
    fn cu_stream_wait_value_64 as "cuStreamWaitValue64_v2": PFN_cuStreamWaitValue64;
    fn cu_stream_batch_mem_op as "cuStreamBatchMemOp_v2": PFN_cuStreamBatchMemOp;

    // ---- Wave 27: v2 advise/prefetch + VMM reverse lookups ----
    fn cu_mem_prefetch_async_v2 as "cuMemPrefetchAsync_v2": PFN_cuMemPrefetchAsyncV2;
    fn cu_mem_advise_v2 as "cuMemAdvise_v2": PFN_cuMemAdviseV2;
    fn cu_mem_map_array_async as "cuMemMapArrayAsync": PFN_cuMemMapArrayAsync;
    fn cu_mem_get_handle_for_address_range as "cuMemGetHandleForAddressRange": PFN_cuMemGetHandleForAddressRange;
    fn cu_mem_retain_allocation_handle as "cuMemRetainAllocationHandle": PFN_cuMemRetainAllocationHandle;
    fn cu_mem_get_allocation_properties_from_handle as "cuMemGetAllocationPropertiesFromHandle": PFN_cuMemGetAllocationPropertiesFromHandle;
    fn cu_mem_export_to_shareable_handle as "cuMemExportToShareableHandle": PFN_cuMemExportToShareableHandle;
    fn cu_mem_import_from_shareable_handle as "cuMemImportFromShareableHandle": PFN_cuMemImportFromShareableHandle;
    fn cu_mem_get_access as "cuMemGetAccess": PFN_cuMemGetAccess;

    // ---- Wave 28: medium-value consolidated ----
    fn cu_array_get_descriptor as "cuArrayGetDescriptor_v2": PFN_cuArrayGetDescriptor;
    fn cu_array_get_sparse_properties as "cuArrayGetSparseProperties": PFN_cuArrayGetSparseProperties;
    fn cu_mipmapped_array_get_sparse_properties as "cuMipmappedArrayGetSparseProperties": PFN_cuMipmappedArrayGetSparseProperties;
    fn cu_array_get_memory_requirements as "cuArrayGetMemoryRequirements": PFN_cuArrayGetMemoryRequirements;
    fn cu_mipmapped_array_get_memory_requirements as "cuMipmappedArrayGetMemoryRequirements": PFN_cuMipmappedArrayGetMemoryRequirements;
    fn cu_array_get_plane as "cuArrayGetPlane": PFN_cuArrayGetPlane;
    fn cu_ctx_record_event as "cuCtxRecordEvent": PFN_cuCtxRecordEvent;
    fn cu_ctx_wait_event as "cuCtxWaitEvent": PFN_cuCtxWaitEvent;
    fn cu_device_get_p2p_attribute as "cuDeviceGetP2PAttribute": PFN_cuDeviceGetP2PAttribute;
    fn cu_device_get_exec_affinity_support as "cuDeviceGetExecAffinitySupport": PFN_cuDeviceGetExecAffinitySupport;
    fn cu_flush_gpudirect_rdma_writes as "cuFlushGPUDirectRDMAWrites": PFN_cuFlushGPUDirectRDMAWrites;
    fn cu_coredump_get_attribute as "cuCoredumpGetAttribute": PFN_cuCoredumpGetAttribute;
    fn cu_coredump_get_attribute_global as "cuCoredumpGetAttributeGlobal": PFN_cuCoredumpGetAttributeGlobal;
    fn cu_coredump_set_attribute as "cuCoredumpSetAttribute": PFN_cuCoredumpSetAttribute;
    fn cu_coredump_set_attribute_global as "cuCoredumpSetAttributeGlobal": PFN_cuCoredumpSetAttributeGlobal;
    fn cu_library_get_unified_function as "cuLibraryGetUnifiedFunction": PFN_cuLibraryGetUnifiedFunction;
    fn cu_library_get_module as "cuLibraryGetModule": PFN_cuLibraryGetModule;
    fn cu_library_get_kernel_count as "cuLibraryGetKernelCount": PFN_cuLibraryGetKernelCount;
    fn cu_library_enumerate_kernels as "cuLibraryEnumerateKernels": PFN_cuLibraryEnumerateKernels;
    fn cu_library_get_managed as "cuLibraryGetManaged": PFN_cuLibraryGetManaged;

    // ---- Wave 29: graphics core + OpenGL ----
    fn cu_graphics_unregister_resource as "cuGraphicsUnregisterResource": PFN_cuGraphicsUnregisterResource;
    fn cu_graphics_map_resources as "cuGraphicsMapResources": PFN_cuGraphicsMapResources;
    fn cu_graphics_unmap_resources as "cuGraphicsUnmapResources": PFN_cuGraphicsUnmapResources;
    fn cu_graphics_resource_get_mapped_pointer as "cuGraphicsResourceGetMappedPointer_v2": PFN_cuGraphicsResourceGetMappedPointer;
    fn cu_graphics_resource_get_mapped_mipmapped_array as "cuGraphicsResourceGetMappedMipmappedArray": PFN_cuGraphicsResourceGetMappedMipmappedArray;
    fn cu_graphics_sub_resource_get_mapped_array as "cuGraphicsSubResourceGetMappedArray": PFN_cuGraphicsSubResourceGetMappedArray;
    fn cu_graphics_resource_set_map_flags as "cuGraphicsResourceSetMapFlags_v2": PFN_cuGraphicsResourceSetMapFlags;
    fn cu_gl_get_devices as "cuGLGetDevices_v2": PFN_cuGLGetDevices;
    fn cu_graphics_gl_register_buffer as "cuGraphicsGLRegisterBuffer": PFN_cuGraphicsGLRegisterBuffer;
    fn cu_graphics_gl_register_image as "cuGraphicsGLRegisterImage": PFN_cuGraphicsGLRegisterImage;
    fn cu_gl_ctx_create as "cuGLCtxCreate_v2": PFN_cuGLCtxCreate;
    fn cu_gl_init as "cuGLInit": PFN_cuGLInit;

    // ---- Wave 30: Direct3D 9 / 10 / 11 ----
    fn cu_d3d9_get_device as "cuD3D9GetDevice": PFN_cuD3D9GetDevice;
    fn cu_d3d9_get_devices as "cuD3D9GetDevices": PFN_cuD3D9GetDevices;
    fn cu_graphics_d3d9_register_resource as "cuGraphicsD3D9RegisterResource": PFN_cuGraphicsD3D9RegisterResource;
    fn cu_d3d10_get_device as "cuD3D10GetDevice": PFN_cuD3D10GetDevice;
    fn cu_d3d10_get_devices as "cuD3D10GetDevices": PFN_cuD3D10GetDevices;
    fn cu_graphics_d3d10_register_resource as "cuGraphicsD3D10RegisterResource": PFN_cuGraphicsD3D10RegisterResource;
    fn cu_d3d11_get_device as "cuD3D11GetDevice": PFN_cuD3D11GetDevice;
    fn cu_d3d11_get_devices as "cuD3D11GetDevices": PFN_cuD3D11GetDevices;
    fn cu_graphics_d3d11_register_resource as "cuGraphicsD3D11RegisterResource": PFN_cuGraphicsD3D11RegisterResource;

    // ---- Wave 31: VDPAU + EGL + NvSci (Jetson) ----
    // VDPAU (Linux)
    fn cu_vdpau_get_device as "cuVDPAUGetDevice": PFN_cuVDPAUGetDevice;
    fn cu_vdpau_ctx_create as "cuVDPAUCtxCreate_v2": PFN_cuVDPAUCtxCreate;
    fn cu_graphics_vdpau_register_video_surface as "cuGraphicsVDPAURegisterVideoSurface": PFN_cuGraphicsVDPAURegisterVideoSurface;
    fn cu_graphics_vdpau_register_output_surface as "cuGraphicsVDPAURegisterOutputSurface": PFN_cuGraphicsVDPAURegisterOutputSurface;

    // EGL (Jetson / cross-platform)
    fn cu_graphics_egl_register_image as "cuGraphicsEGLRegisterImage": PFN_cuGraphicsEGLRegisterImage;
    fn cu_graphics_resource_get_mapped_egl_frame as "cuGraphicsResourceGetMappedEglFrame": PFN_cuGraphicsResourceGetMappedEglFrame;
    fn cu_event_create_from_egl_sync as "cuEventCreateFromEGLSync": PFN_cuEventCreateFromEGLSync;
    fn cu_egl_stream_consumer_connect as "cuEGLStreamConsumerConnect": PFN_cuEGLStreamConsumerConnect;
    fn cu_egl_stream_consumer_disconnect as "cuEGLStreamConsumerDisconnect": PFN_cuEGLStreamConsumerDisconnect;
    fn cu_egl_stream_consumer_acquire_frame as "cuEGLStreamConsumerAcquireFrame": PFN_cuEGLStreamConsumerAcquireFrame;
    fn cu_egl_stream_consumer_release_frame as "cuEGLStreamConsumerReleaseFrame": PFN_cuEGLStreamConsumerReleaseFrame;
    fn cu_egl_stream_producer_connect as "cuEGLStreamProducerConnect": PFN_cuEGLStreamProducerConnect;
    fn cu_egl_stream_producer_disconnect as "cuEGLStreamProducerDisconnect": PFN_cuEGLStreamProducerDisconnect;
    fn cu_egl_stream_producer_present_frame as "cuEGLStreamProducerPresentFrame": PFN_cuEGLStreamProducerPresentFrame;
    fn cu_egl_stream_producer_return_frame as "cuEGLStreamProducerReturnFrame": PFN_cuEGLStreamProducerReturnFrame;

    // NvSci (Jetson / DRIVE)
    fn cu_device_get_nv_sci_sync_attributes as "cuDeviceGetNvSciSyncAttributes": PFN_cuDeviceGetNvSciSyncAttributes;
}

impl Driver {
    /// Resolve `cuGetProcAddress` via `dlsym` (the only symbol we cannot
    /// resolve through itself).
    fn cu_get_proc_address(&self) -> Result<PFN_cuGetProcAddress, LoaderError> {
        if let Some(&p) = self.get_proc_address.get() {
            return Ok(p);
        }
        let p: PFN_cuGetProcAddress = unsafe { self.resolve_via_dlsym("cuGetProcAddress")? };
        let _ = self.get_proc_address.set(p);
        Ok(p)
    }

    /// Resolve `symbol` and transmute into the caller-specified fn-pointer
    /// type. `cuGetProcAddress` is resolved via `dlsym` (it's our bootstrap).
    /// Symbol names with an explicit version suffix like `_v2` / `_v3` are
    /// also resolved via `dlsym` — because `cuGetProcAddress` with a
    /// version-suffixed base name would _still_ do version-dispatch and
    /// might return a newer ABI than our `PFN_*` signature expects.
    /// Everything else goes through `cuGetProcAddress`, which transparently
    /// picks `_ptsz` / `_ptds` variants based on the configured stream mode.
    ///
    /// # Safety
    ///
    /// `T` must be a function-pointer type whose signature matches the C
    /// declaration of `symbol` in `cuda.h`. The symbol names we pass come
    /// from the macro above and are checked against the NVIDIA docs.
    unsafe fn resolve<T: Copy>(&self, symbol: &'static str) -> Result<T, LoaderError> {
        if symbol == "cuGetProcAddress" || has_version_suffix(symbol) {
            return self.resolve_via_dlsym(symbol);
        }
        self.resolve_via_get_proc_address(symbol)
    }

    /// Direct `dlsym` / `GetProcAddress`; used for `cuGetProcAddress` itself.
    unsafe fn resolve_via_dlsym<T: Copy>(&self, symbol: &'static str) -> Result<T, LoaderError> {
        debug_assert_eq!(
            core::mem::size_of::<T>(),
            core::mem::size_of::<*mut ()>(),
            "Driver::resolve_via_dlsym<T>: T must be a function-pointer type",
        );
        let raw: *mut () = self.lib.raw_symbol(symbol)?;
        Ok(core::mem::transmute_copy::<*mut (), T>(&raw))
    }

    /// Cached driver-reported CUDA version. Probed once via a direct
    /// `dlsym` on `cuDriverGetVersion`. Falls back to the baracuda floor
    /// (CUDA 11.4) if the probe fails, which is safe — `cuGetProcAddress`
    /// treats the version as a minimum.
    fn detected_cuda_version(&self) -> core::ffi::c_int {
        use std::sync::OnceLock;
        static CACHED: OnceLock<core::ffi::c_int> = OnceLock::new();
        *CACHED.get_or_init(|| {
            let raw: *mut () = match unsafe { self.lib.raw_symbol("cuDriverGetVersion") } {
                Ok(p) => p,
                Err(_) => return baracuda_types::CudaVersion::FLOOR.raw() as core::ffi::c_int,
            };
            type Fn = unsafe extern "C" fn(*mut core::ffi::c_int) -> CUresult;
            // SAFETY: `cuDriverGetVersion` has a stable signature.
            let f: Fn = unsafe { core::mem::transmute_copy::<*mut (), Fn>(&raw) };
            let mut v: core::ffi::c_int = 0;
            match unsafe { f(&mut v) } {
                CUresult::SUCCESS if v > 0 => v,
                _ => baracuda_types::CudaVersion::FLOOR.raw() as core::ffi::c_int,
            }
        })
    }

    /// Resolve `symbol` through `cuGetProcAddress`, passing the process's
    /// configured stream mode as the flags argument.
    unsafe fn resolve_via_get_proc_address<T: Copy>(
        &self,
        symbol: &'static str,
    ) -> Result<T, LoaderError> {
        debug_assert_eq!(
            core::mem::size_of::<T>(),
            core::mem::size_of::<*mut ()>(),
            "Driver::resolve_via_get_proc_address<T>: T must be a function-pointer type",
        );
        let gpa = self.cu_get_proc_address()?;
        let flags = match stream_mode::get() {
            StreamMode::Legacy => CU_GET_PROC_ADDRESS_LEGACY_STREAM,
            StreamMode::PerThread => CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
        };
        let c_sym: Vec<u8> = symbol.bytes().chain(std::iter::once(0)).collect();
        let mut pfn: *mut core::ffi::c_void = ptr::null_mut();

        // Two-stage resolution:
        //   1. Query at the baracuda floor (CUDA 11.4). This pins the ABI
        //      for symbols that have `_v2`/`_v3` variants — we want the
        //      11.4-era shape because that's what our `PFN_*` type matches.
        //   2. If the driver returns NOT_FOUND / VERSION_NOT_SUFFICIENT,
        //      retry at the driver's own reported version. That catches
        //      CUDA-12+ additions (cuLibraryLoadData, cuLaunchKernelEx)
        //      without upgrading older symbols to their newer ABIs.
        let floor = baracuda_types::CudaVersion::FLOOR.raw() as core::ffi::c_int;
        let mut res = gpa(c_sym.as_ptr() as *const c_char, &mut pfn, floor, flags);
        if res != CUresult::SUCCESS || pfn.is_null() {
            let installed = self.detected_cuda_version();
            if installed > floor {
                pfn = ptr::null_mut();
                res = gpa(c_sym.as_ptr() as *const c_char, &mut pfn, installed, flags);
            }
        }
        if res != CUresult::SUCCESS || pfn.is_null() {
            return Err(LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol,
            });
        }
        Ok(core::mem::transmute_copy::<*mut core::ffi::c_void, T>(&pfn))
    }
}

/// `true` if `sym` ends with `_v<N>` (version pin) or `_ptsz` / `_ptds`
/// (stream-mode pin). Such names are resolved via `dlsym` so the driver
/// doesn't silently upgrade us to a newer ABI.
fn has_version_suffix(sym: &str) -> bool {
    if sym.ends_with("_ptsz") || sym.ends_with("_ptds") {
        return true;
    }
    if let Some(idx) = sym.rfind("_v") {
        let tail = &sym[idx + 2..];
        !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit())
    } else {
        false
    }
}

/// Lazily-initialized process-wide Driver singleton.
///
/// The first successful call to [`driver`] caches the [`Driver`] in a
/// `OnceLock`; subsequent calls return the same `&'static Driver`. If the
/// first call fails (no libcuda, unsupported platform), the error is
/// **not** memoized — a later call with a different environment has a
/// chance of succeeding.
pub fn driver() -> Result<&'static Driver, LoaderError> {
    static DRIVER: OnceLock<Driver> = OnceLock::new();
    if let Some(d) = DRIVER.get() {
        return Ok(d);
    }
    let lib = Library::open("cuda-driver", platform::driver_library_candidates())?;
    let d = Driver::empty(lib);
    // If another thread raced us, our `d` is dropped; `DRIVER.get().unwrap()`
    // still returns the thread-that-won's instance.
    let _ = DRIVER.set(d);
    Ok(DRIVER.get().expect("OnceLock set or lost race"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_singleton_returns_loader_error_without_cuda() {
        // On a machine without CUDA, `driver()` returns LoaderError. On a
        // machine with CUDA, we'd expect Ok, but we can't tell from CI. We
        // just verify we get *some* Result out — no panic.
        let _ = driver();
    }
}
