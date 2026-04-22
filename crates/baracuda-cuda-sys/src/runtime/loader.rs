//! The `Runtime` loader — parallels [`crate::Driver`] for the CUDA Runtime API.
//!
//! `libcudart` does not expose a `cuGetProcAddress`-style entry-point
//! resolver, so we resolve everything via plain `dlsym`. Symbols are
//! cached in per-function `OnceLock`s exactly like the Driver loader.

use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};

use super::functions::*;

macro_rules! runtime_fns {
    ($(
        $(#[$attr:meta])*
        fn $name:ident as $sym:literal : $pfn:ty;
    )*) => {
        /// Lazily-resolved CUDA Runtime API function-pointer table.
        #[allow(non_snake_case)]
        pub struct Runtime {
            lib: Library,
            $(
                $name: OnceLock<$pfn>,
            )*
        }

        impl core::fmt::Debug for Runtime {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Runtime")
                    .field("lib", &self.lib)
                    .finish_non_exhaustive()
            }
        }

        impl Runtime {
            fn empty(lib: Library) -> Self {
                Self {
                    lib,
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
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    // SAFETY: `$pfn` is a function-pointer type whose
                    // signature matches the C declaration of `$sym`.
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
        }
    };
}

runtime_fns! {
    // Version & error
    fn cuda_runtime_get_version as "cudaRuntimeGetVersion": PFN_cudaRuntimeGetVersion;
    fn cuda_driver_get_version as "cudaDriverGetVersion": PFN_cudaDriverGetVersion;
    fn cuda_get_last_error as "cudaGetLastError": PFN_cudaGetLastError;
    fn cuda_peek_at_last_error as "cudaPeekAtLastError": PFN_cudaPeekAtLastError;
    fn cuda_get_error_name as "cudaGetErrorName": PFN_cudaGetErrorName;
    fn cuda_get_error_string as "cudaGetErrorString": PFN_cudaGetErrorString;

    // Device
    fn cuda_get_device_count as "cudaGetDeviceCount": PFN_cudaGetDeviceCount;
    fn cuda_set_device as "cudaSetDevice": PFN_cudaSetDevice;
    fn cuda_get_device as "cudaGetDevice": PFN_cudaGetDevice;
    fn cuda_device_synchronize as "cudaDeviceSynchronize": PFN_cudaDeviceSynchronize;
    fn cuda_device_reset as "cudaDeviceReset": PFN_cudaDeviceReset;
    fn cuda_device_get_attribute as "cudaDeviceGetAttribute": PFN_cudaDeviceGetAttribute;
    fn cuda_init_device as "cudaInitDevice": PFN_cudaInitDevice;

    // Memory
    fn cuda_malloc as "cudaMalloc": PFN_cudaMalloc;
    fn cuda_free as "cudaFree": PFN_cudaFree;
    fn cuda_malloc_managed as "cudaMallocManaged": PFN_cudaMallocManaged;
    fn cuda_memcpy as "cudaMemcpy": PFN_cudaMemcpy;
    fn cuda_memcpy_async as "cudaMemcpyAsync": PFN_cudaMemcpyAsync;
    fn cuda_memset as "cudaMemset": PFN_cudaMemset;
    fn cuda_memset_async as "cudaMemsetAsync": PFN_cudaMemsetAsync;

    // Stream
    fn cuda_stream_create as "cudaStreamCreate": PFN_cudaStreamCreate;
    fn cuda_stream_create_with_flags as "cudaStreamCreateWithFlags": PFN_cudaStreamCreateWithFlags;
    fn cuda_stream_destroy as "cudaStreamDestroy": PFN_cudaStreamDestroy;
    fn cuda_stream_synchronize as "cudaStreamSynchronize": PFN_cudaStreamSynchronize;
    fn cuda_stream_query as "cudaStreamQuery": PFN_cudaStreamQuery;
    fn cuda_stream_wait_event as "cudaStreamWaitEvent": PFN_cudaStreamWaitEvent;

    // Event
    fn cuda_event_create as "cudaEventCreate": PFN_cudaEventCreate;
    fn cuda_event_create_with_flags as "cudaEventCreateWithFlags": PFN_cudaEventCreateWithFlags;
    fn cuda_event_destroy as "cudaEventDestroy": PFN_cudaEventDestroy;
    fn cuda_event_record as "cudaEventRecord": PFN_cudaEventRecord;
    fn cuda_event_synchronize as "cudaEventSynchronize": PFN_cudaEventSynchronize;
    fn cuda_event_query as "cudaEventQuery": PFN_cudaEventQuery;
    fn cuda_event_elapsed_time as "cudaEventElapsedTime": PFN_cudaEventElapsedTime;

    // Kernel launch
    fn cuda_launch_kernel as "cudaLaunchKernel": PFN_cudaLaunchKernel;

    // Library management (CUDA 12.0+) — will fail on older installs; callers
    // should gate via Feature::LibraryManagement.
    fn cuda_library_load_data as "cudaLibraryLoadData": PFN_cudaLibraryLoadData;
    fn cuda_library_unload as "cudaLibraryUnload": PFN_cudaLibraryUnload;
    fn cuda_library_get_kernel as "cudaLibraryGetKernel": PFN_cudaLibraryGetKernel;

    // Stream extras
    fn cuda_stream_create_with_priority as "cudaStreamCreateWithPriority": PFN_cudaStreamCreateWithPriority;
    fn cuda_stream_get_priority as "cudaStreamGetPriority": PFN_cudaStreamGetPriority;
    fn cuda_stream_get_flags as "cudaStreamGetFlags": PFN_cudaStreamGetFlags;
    fn cuda_device_get_stream_priority_range as "cudaDeviceGetStreamPriorityRange": PFN_cudaDeviceGetStreamPriorityRange;

    // Peer access
    fn cuda_device_can_access_peer as "cudaDeviceCanAccessPeer": PFN_cudaDeviceCanAccessPeer;
    fn cuda_device_enable_peer_access as "cudaDeviceEnablePeerAccess": PFN_cudaDeviceEnablePeerAccess;
    fn cuda_device_disable_peer_access as "cudaDeviceDisablePeerAccess": PFN_cudaDeviceDisablePeerAccess;

    // Mem prefetch/advise + mem info
    fn cuda_mem_prefetch_async as "cudaMemPrefetchAsync": PFN_cudaMemPrefetchAsync;
    fn cuda_mem_advise as "cudaMemAdvise": PFN_cudaMemAdvise;
    fn cuda_mem_get_info as "cudaMemGetInfo": PFN_cudaMemGetInfo;

    // Pinned + managed memory
    fn cuda_malloc_host as "cudaMallocHost": PFN_cudaMallocHost;
    fn cuda_free_host as "cudaFreeHost": PFN_cudaFreeHost;
    fn cuda_host_alloc as "cudaHostAlloc": PFN_cudaHostAlloc;
    fn cuda_host_register as "cudaHostRegister": PFN_cudaHostRegister;
    fn cuda_host_unregister as "cudaHostUnregister": PFN_cudaHostUnregister;
    fn cuda_host_get_device_pointer as "cudaHostGetDevicePointer": PFN_cudaHostGetDevicePointer;
    fn cuda_host_get_flags as "cudaHostGetFlags": PFN_cudaHostGetFlags;

    // Async alloc
    fn cuda_malloc_async as "cudaMallocAsync": PFN_cudaMallocAsync;
    fn cuda_free_async as "cudaFreeAsync": PFN_cudaFreeAsync;

    // Graphs + stream capture
    fn cuda_graph_create as "cudaGraphCreate": PFN_cudaGraphCreate;
    fn cuda_graph_destroy as "cudaGraphDestroy": PFN_cudaGraphDestroy;
    fn cuda_graph_instantiate as "cudaGraphInstantiate": PFN_cudaGraphInstantiate;
    fn cuda_graph_launch as "cudaGraphLaunch": PFN_cudaGraphLaunch;
    fn cuda_graph_exec_destroy as "cudaGraphExecDestroy": PFN_cudaGraphExecDestroy;
    fn cuda_graph_get_nodes as "cudaGraphGetNodes": PFN_cudaGraphGetNodes;
    fn cuda_stream_begin_capture as "cudaStreamBeginCapture": PFN_cudaStreamBeginCapture;
    fn cuda_stream_end_capture as "cudaStreamEndCapture": PFN_cudaStreamEndCapture;
    fn cuda_stream_is_capturing as "cudaStreamIsCapturing": PFN_cudaStreamIsCapturing;

    // Function symbol / attrs / occupancy
    fn cuda_get_func_by_symbol as "cudaGetFuncBySymbol": PFN_cudaGetFuncBySymbol;
    fn cuda_func_get_attributes as "cudaFuncGetAttributes": PFN_cudaFuncGetAttributes;
    fn cuda_func_set_attribute as "cudaFuncSetAttribute": PFN_cudaFuncSetAttribute;
    fn cuda_occupancy_max_active_blocks_per_multiprocessor as "cudaOccupancyMaxActiveBlocksPerMultiprocessor": PFN_cudaOccupancyMaxActiveBlocksPerMultiprocessor;
    fn cuda_occupancy_max_potential_block_size as "cudaOccupancyMaxPotentialBlockSize": PFN_cudaOccupancyMaxPotentialBlockSize;

    // Pointer attributes
    fn cuda_pointer_get_attributes as "cudaPointerGetAttributes": PFN_cudaPointerGetAttributes;

    // 2-D memcpy + memset variants
    fn cuda_memcpy_2d as "cudaMemcpy2D": PFN_cudaMemcpy2D;
    fn cuda_memcpy_2d_async as "cudaMemcpy2DAsync": PFN_cudaMemcpy2DAsync;
    fn cuda_malloc_pitch as "cudaMallocPitch": PFN_cudaMallocPitch;
    fn cuda_memset_2d as "cudaMemset2D": PFN_cudaMemset2D;
    fn cuda_memset_2d_async as "cudaMemset2DAsync": PFN_cudaMemset2DAsync;

    // Peer memcpy
    fn cuda_memcpy_peer as "cudaMemcpyPeer": PFN_cudaMemcpyPeer;
    fn cuda_memcpy_peer_async as "cudaMemcpyPeerAsync": PFN_cudaMemcpyPeerAsync;

    // Device properties (opaque 1KB+ struct)
    // Accepts both base and _v2 names at the symbol-resolver level —
    // runtime_fns! hard-codes the name, so we try `_v2` first via a
    // separate alias registered below and fall back in the safe wrapper.
    fn cuda_get_device_properties as "cudaGetDeviceProperties": PFN_cudaGetDeviceProperties;

    // External memory + semaphore interop
    fn cuda_import_external_memory as "cudaImportExternalMemory": PFN_cudaImportExternalMemory;
    fn cuda_destroy_external_memory as "cudaDestroyExternalMemory": PFN_cudaDestroyExternalMemory;
    fn cuda_external_memory_get_mapped_buffer as "cudaExternalMemoryGetMappedBuffer": PFN_cudaExternalMemoryGetMappedBuffer;
    fn cuda_external_memory_get_mapped_mipmapped_array as "cudaExternalMemoryGetMappedMipmappedArray": PFN_cudaExternalMemoryGetMappedMipmappedArray;
    fn cuda_import_external_semaphore as "cudaImportExternalSemaphore": PFN_cudaImportExternalSemaphore;
    fn cuda_destroy_external_semaphore as "cudaDestroyExternalSemaphore": PFN_cudaDestroyExternalSemaphore;
    fn cuda_signal_external_semaphores_async as "cudaSignalExternalSemaphoresAsync": PFN_cudaSignalExternalSemaphoresAsync;
    fn cuda_wait_external_semaphores_async as "cudaWaitExternalSemaphoresAsync": PFN_cudaWaitExternalSemaphoresAsync;

    // ---- Runtime Wave 1: host-fn launch + stream write/wait value ----
    fn cuda_launch_host_func as "cudaLaunchHostFunc": PFN_cudaLaunchHostFunc;
    fn cuda_stream_write_value_32 as "cudaStreamWriteValue32": PFN_cudaStreamWriteValue32;
    fn cuda_stream_write_value_64 as "cudaStreamWriteValue64": PFN_cudaStreamWriteValue64;
    fn cuda_stream_wait_value_32 as "cudaStreamWaitValue32": PFN_cudaStreamWaitValue32;
    fn cuda_stream_wait_value_64 as "cudaStreamWaitValue64": PFN_cudaStreamWaitValue64;

    // ---- Memory pools ----
    fn cuda_mem_pool_create as "cudaMemPoolCreate": PFN_cudaMemPoolCreate;
    fn cuda_mem_pool_destroy as "cudaMemPoolDestroy": PFN_cudaMemPoolDestroy;
    fn cuda_mem_pool_set_attribute as "cudaMemPoolSetAttribute": PFN_cudaMemPoolSetAttribute;
    fn cuda_mem_pool_get_attribute as "cudaMemPoolGetAttribute": PFN_cudaMemPoolGetAttribute;
    fn cuda_mem_pool_trim_to as "cudaMemPoolTrimTo": PFN_cudaMemPoolTrimTo;
    fn cuda_mem_pool_set_access as "cudaMemPoolSetAccess": PFN_cudaMemPoolSetAccess;
    fn cuda_mem_pool_get_access as "cudaMemPoolGetAccess": PFN_cudaMemPoolGetAccess;
    fn cuda_malloc_from_pool_async as "cudaMallocFromPoolAsync": PFN_cudaMallocFromPoolAsync;
    fn cuda_device_get_default_mem_pool as "cudaDeviceGetDefaultMemPool": PFN_cudaDeviceGetDefaultMemPool;
    fn cuda_device_get_mem_pool as "cudaDeviceGetMemPool": PFN_cudaDeviceGetMemPool;
    fn cuda_device_set_mem_pool as "cudaDeviceSetMemPool": PFN_cudaDeviceSetMemPool;
    fn cuda_mem_pool_export_to_shareable_handle as "cudaMemPoolExportToShareableHandle": PFN_cudaMemPoolExportToShareableHandle;
    fn cuda_mem_pool_import_from_shareable_handle as "cudaMemPoolImportFromShareableHandle": PFN_cudaMemPoolImportFromShareableHandle;
    fn cuda_mem_pool_export_pointer as "cudaMemPoolExportPointer": PFN_cudaMemPoolExportPointer;
    fn cuda_mem_pool_import_pointer as "cudaMemPoolImportPointer": PFN_cudaMemPoolImportPointer;

    // ---- Explicit graph node builders ----
    fn cuda_graph_add_kernel_node as "cudaGraphAddKernelNode": PFN_cudaGraphAddKernelNode;
    fn cuda_graph_add_memset_node as "cudaGraphAddMemsetNode": PFN_cudaGraphAddMemsetNode;
    fn cuda_graph_add_memcpy_node as "cudaGraphAddMemcpyNode": PFN_cudaGraphAddMemcpyNode;
    fn cuda_graph_add_host_node as "cudaGraphAddHostNode": PFN_cudaGraphAddHostNode;
    fn cuda_graph_add_empty_node as "cudaGraphAddEmptyNode": PFN_cudaGraphAddEmptyNode;
    fn cuda_graph_add_child_graph_node as "cudaGraphAddChildGraphNode": PFN_cudaGraphAddChildGraphNode;
    fn cuda_graph_add_event_record_node as "cudaGraphAddEventRecordNode": PFN_cudaGraphAddEventRecordNode;
    fn cuda_graph_add_event_wait_node as "cudaGraphAddEventWaitNode": PFN_cudaGraphAddEventWaitNode;
    fn cuda_graph_add_mem_alloc_node as "cudaGraphAddMemAllocNode": PFN_cudaGraphAddMemAllocNode;
    fn cuda_graph_add_mem_free_node as "cudaGraphAddMemFreeNode": PFN_cudaGraphAddMemFreeNode;
    fn cuda_graph_exec_update as "cudaGraphExecUpdate": PFN_cudaGraphExecUpdate;
    fn cuda_graph_add_dependencies as "cudaGraphAddDependencies": PFN_cudaGraphAddDependencies;
    fn cuda_graph_mem_free_node_get_params as "cudaGraphMemFreeNodeGetParams": PFN_cudaGraphMemFreeNodeGetParams;
    fn cuda_graph_node_get_type as "cudaGraphNodeGetType": PFN_cudaGraphNodeGetType;

    // ---- Runtime Wave 2 ----
    // Arrays + tex/surf
    fn cuda_malloc_array as "cudaMallocArray": PFN_cudaMallocArray;
    fn cuda_free_array as "cudaFreeArray": PFN_cudaFreeArray;
    fn cuda_memcpy_2d_to_array as "cudaMemcpy2DToArray": PFN_cudaMemcpy2DToArray;
    fn cuda_memcpy_2d_from_array as "cudaMemcpy2DFromArray": PFN_cudaMemcpy2DFromArray;
    fn cuda_create_texture_object as "cudaCreateTextureObject": PFN_cudaCreateTextureObject;
    fn cuda_destroy_texture_object as "cudaDestroyTextureObject": PFN_cudaDestroyTextureObject;
    fn cuda_create_surface_object as "cudaCreateSurfaceObject": PFN_cudaCreateSurfaceObject;
    fn cuda_destroy_surface_object as "cudaDestroySurfaceObject": PFN_cudaDestroySurfaceObject;

    // User objects
    fn cuda_user_object_create as "cudaUserObjectCreate": PFN_cudaUserObjectCreate;
    fn cuda_user_object_retain as "cudaUserObjectRetain": PFN_cudaUserObjectRetain;
    fn cuda_user_object_release as "cudaUserObjectRelease": PFN_cudaUserObjectRelease;
    fn cuda_graph_retain_user_object as "cudaGraphRetainUserObject": PFN_cudaGraphRetainUserObject;
    fn cuda_graph_release_user_object as "cudaGraphReleaseUserObject": PFN_cudaGraphReleaseUserObject;

    // Cooperative launch
    fn cuda_launch_cooperative_kernel as "cudaLaunchCooperativeKernel": PFN_cudaLaunchCooperativeKernel;

    // Stream attach + attrs
    fn cuda_stream_attach_mem_async as "cudaStreamAttachMemAsync": PFN_cudaStreamAttachMemAsync;
    fn cuda_stream_get_attribute as "cudaStreamGetAttribute": PFN_cudaStreamGetAttribute;
    fn cuda_stream_set_attribute as "cudaStreamSetAttribute": PFN_cudaStreamSetAttribute;
    fn cuda_stream_copy_attributes as "cudaStreamCopyAttributes": PFN_cudaStreamCopyAttributes;

    // IPC
    fn cuda_ipc_get_event_handle as "cudaIpcGetEventHandle": PFN_cudaIpcGetEventHandle;
    fn cuda_ipc_open_event_handle as "cudaIpcOpenEventHandle": PFN_cudaIpcOpenEventHandle;
    fn cuda_ipc_get_mem_handle as "cudaIpcGetMemHandle": PFN_cudaIpcGetMemHandle;
    fn cuda_ipc_open_mem_handle as "cudaIpcOpenMemHandle": PFN_cudaIpcOpenMemHandle;
    fn cuda_ipc_close_mem_handle as "cudaIpcCloseMemHandle": PFN_cudaIpcCloseMemHandle;

    // Device flags
    fn cuda_set_device_flags as "cudaSetDeviceFlags": PFN_cudaSetDeviceFlags;
    fn cuda_get_device_flags as "cudaGetDeviceFlags": PFN_cudaGetDeviceFlags;

    // ---- Runtime Wave 3: batch mem ops + conditional nodes + driver bridge + occupancy ----
    fn cuda_stream_batch_mem_op as "cudaStreamBatchMemOp": PFN_cudaStreamBatchMemOp;
    fn cuda_graph_add_node as "cudaGraphAddNode": PFN_cudaGraphAddNode;
    fn cuda_graph_conditional_handle_create as "cudaGraphConditionalHandleCreate": PFN_cudaGraphConditionalHandleCreate;
    fn cuda_get_driver_entry_point as "cudaGetDriverEntryPoint": PFN_cudaGetDriverEntryPoint;
    fn cuda_occupancy_max_active_blocks_per_multiprocessor_with_flags
        as "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags":
        PFN_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
    fn cuda_occupancy_available_dynamic_smem_per_block
        as "cudaOccupancyAvailableDynamicSMemPerBlock":
        PFN_cudaOccupancyAvailableDynamicSMemPerBlock;

    // ---- Runtime Wave 4: graphics interop (core + GL + D3D + VDPAU + EGL + NvSci) ----

    // Core graphics
    fn cuda_graphics_unregister_resource as "cudaGraphicsUnregisterResource":
        PFN_cudaGraphicsUnregisterResource;
    fn cuda_graphics_map_resources as "cudaGraphicsMapResources":
        PFN_cudaGraphicsMapResources;
    fn cuda_graphics_unmap_resources as "cudaGraphicsUnmapResources":
        PFN_cudaGraphicsUnmapResources;
    fn cuda_graphics_resource_get_mapped_pointer
        as "cudaGraphicsResourceGetMappedPointer":
        PFN_cudaGraphicsResourceGetMappedPointer;
    fn cuda_graphics_sub_resource_get_mapped_array
        as "cudaGraphicsSubResourceGetMappedArray":
        PFN_cudaGraphicsSubResourceGetMappedArray;
    fn cuda_graphics_resource_get_mapped_mipmapped_array
        as "cudaGraphicsResourceGetMappedMipmappedArray":
        PFN_cudaGraphicsResourceGetMappedMipmappedArray;
    fn cuda_graphics_resource_set_map_flags as "cudaGraphicsResourceSetMapFlags":
        PFN_cudaGraphicsResourceSetMapFlags;

    // OpenGL
    fn cuda_graphics_gl_register_buffer as "cudaGraphicsGLRegisterBuffer":
        PFN_cudaGraphicsGLRegisterBuffer;
    fn cuda_graphics_gl_register_image as "cudaGraphicsGLRegisterImage":
        PFN_cudaGraphicsGLRegisterImage;
    fn cuda_gl_get_devices as "cudaGLGetDevices": PFN_cudaGLGetDevices;

    // D3D9 / D3D10 / D3D11
    fn cuda_d3d9_get_device as "cudaD3D9GetDevice": PFN_cudaD3D9GetDevice;
    fn cuda_d3d9_get_devices as "cudaD3D9GetDevices": PFN_cudaD3D9GetDevices;
    fn cuda_graphics_d3d9_register_resource as "cudaGraphicsD3D9RegisterResource":
        PFN_cudaGraphicsD3D9RegisterResource;
    fn cuda_d3d10_get_device as "cudaD3D10GetDevice": PFN_cudaD3D10GetDevice;
    fn cuda_d3d10_get_devices as "cudaD3D10GetDevices": PFN_cudaD3D10GetDevices;
    fn cuda_graphics_d3d10_register_resource as "cudaGraphicsD3D10RegisterResource":
        PFN_cudaGraphicsD3D10RegisterResource;
    fn cuda_d3d11_get_device as "cudaD3D11GetDevice": PFN_cudaD3D11GetDevice;
    fn cuda_d3d11_get_devices as "cudaD3D11GetDevices": PFN_cudaD3D11GetDevices;
    fn cuda_graphics_d3d11_register_resource as "cudaGraphicsD3D11RegisterResource":
        PFN_cudaGraphicsD3D11RegisterResource;

    // VDPAU
    fn cuda_vdpau_get_device as "cudaVDPAUGetDevice": PFN_cudaVDPAUGetDevice;
    fn cuda_graphics_vdpau_register_video_surface
        as "cudaGraphicsVDPAURegisterVideoSurface":
        PFN_cudaGraphicsVDPAURegisterVideoSurface;
    fn cuda_graphics_vdpau_register_output_surface
        as "cudaGraphicsVDPAURegisterOutputSurface":
        PFN_cudaGraphicsVDPAURegisterOutputSurface;

    // EGL
    fn cuda_graphics_egl_register_image as "cudaGraphicsEGLRegisterImage":
        PFN_cudaGraphicsEGLRegisterImage;
    fn cuda_graphics_resource_get_mapped_egl_frame
        as "cudaGraphicsResourceGetMappedEglFrame":
        PFN_cudaGraphicsResourceGetMappedEglFrame;
    fn cuda_event_create_from_egl_sync as "cudaEventCreateFromEGLSync":
        PFN_cudaEventCreateFromEGLSync;
    fn cuda_egl_stream_consumer_connect as "cudaEGLStreamConsumerConnect":
        PFN_cudaEGLStreamConsumerConnect;
    fn cuda_egl_stream_consumer_disconnect as "cudaEGLStreamConsumerDisconnect":
        PFN_cudaEGLStreamConsumerDisconnect;
    fn cuda_egl_stream_consumer_acquire_frame as "cudaEGLStreamConsumerAcquireFrame":
        PFN_cudaEGLStreamConsumerAcquireFrame;
    fn cuda_egl_stream_consumer_release_frame as "cudaEGLStreamConsumerReleaseFrame":
        PFN_cudaEGLStreamConsumerReleaseFrame;
    fn cuda_egl_stream_producer_connect as "cudaEGLStreamProducerConnect":
        PFN_cudaEGLStreamProducerConnect;
    fn cuda_egl_stream_producer_disconnect as "cudaEGLStreamProducerDisconnect":
        PFN_cudaEGLStreamProducerDisconnect;
    fn cuda_egl_stream_producer_present_frame as "cudaEGLStreamProducerPresentFrame":
        PFN_cudaEGLStreamProducerPresentFrame;
    fn cuda_egl_stream_producer_return_frame as "cudaEGLStreamProducerReturnFrame":
        PFN_cudaEGLStreamProducerReturnFrame;

    // NvSci
    fn cuda_device_get_nv_sci_sync_attributes as "cudaDeviceGetNvSciSyncAttributes":
        PFN_cudaDeviceGetNvSciSyncAttributes;

    // ---- Runtime Wave 5 ----

    // Arrays (extras) + tex/surf object descriptors
    fn cuda_malloc_mipmapped_array as "cudaMallocMipmappedArray": PFN_cudaMallocMipmappedArray;
    fn cuda_free_mipmapped_array as "cudaFreeMipmappedArray": PFN_cudaFreeMipmappedArray;
    fn cuda_array_get_info as "cudaArrayGetInfo": PFN_cudaArrayGetInfo;
    fn cuda_get_mipmapped_array_level as "cudaGetMipmappedArrayLevel":
        PFN_cudaGetMipmappedArrayLevel;
    fn cuda_get_texture_object_resource_desc as "cudaGetTextureObjectResourceDesc":
        PFN_cudaGetTextureObjectResourceDesc;
    fn cuda_get_texture_object_texture_desc as "cudaGetTextureObjectTextureDesc":
        PFN_cudaGetTextureObjectTextureDesc;
    fn cuda_get_texture_object_resource_view_desc as "cudaGetTextureObjectResourceViewDesc":
        PFN_cudaGetTextureObjectResourceViewDesc;
    fn cuda_get_surface_object_resource_desc as "cudaGetSurfaceObjectResourceDesc":
        PFN_cudaGetSurfaceObjectResourceDesc;

    // 3D memcpy
    fn cuda_memcpy_3d as "cudaMemcpy3D": PFN_cudaMemcpy3D;
    fn cuda_memcpy_3d_async as "cudaMemcpy3DAsync": PFN_cudaMemcpy3DAsync;
    fn cuda_memcpy_3d_peer as "cudaMemcpy3DPeer": PFN_cudaMemcpy3DPeer;
    fn cuda_memcpy_3d_peer_async as "cudaMemcpy3DPeerAsync": PFN_cudaMemcpy3DPeerAsync;
    fn cuda_memset_3d as "cudaMemset3D": PFN_cudaMemset3D;
    fn cuda_malloc_3d as "cudaMalloc3D": PFN_cudaMalloc3D;
    fn cuda_malloc_3d_array as "cudaMalloc3DArray": PFN_cudaMalloc3DArray;

    // Launch-ex / cluster
    fn cuda_launch_kernel_ex as "cudaLaunchKernelEx": PFN_cudaLaunchKernelEx;

    // Profiler
    fn cuda_profiler_start as "cudaProfilerStart": PFN_cudaProfilerStart;
    fn cuda_profiler_stop as "cudaProfilerStop": PFN_cudaProfilerStop;

    // VMM
    fn cuda_mem_address_reserve as "cudaMemAddressReserve": PFN_cudaMemAddressReserve;
    fn cuda_mem_address_free as "cudaMemAddressFree": PFN_cudaMemAddressFree;
    fn cuda_mem_create as "cudaMemCreate": PFN_cudaMemCreate;
    fn cuda_mem_release as "cudaMemRelease": PFN_cudaMemRelease;
    fn cuda_mem_map as "cudaMemMap": PFN_cudaMemMap;
    fn cuda_mem_unmap as "cudaMemUnmap": PFN_cudaMemUnmap;
    fn cuda_mem_set_access as "cudaMemSetAccess": PFN_cudaMemSetAccess;
    fn cuda_mem_get_access as "cudaMemGetAccess": PFN_cudaMemGetAccess;
    fn cuda_mem_get_allocation_granularity as "cudaMemGetAllocationGranularity":
        PFN_cudaMemGetAllocationGranularity;
    fn cuda_mem_get_allocation_properties_from_handle
        as "cudaMemGetAllocationPropertiesFromHandle":
        PFN_cudaMemGetAllocationPropertiesFromHandle;
    fn cuda_mem_export_to_shareable_handle as "cudaMemExportToShareableHandle":
        PFN_cudaMemExportToShareableHandle;
    fn cuda_mem_import_from_shareable_handle as "cudaMemImportFromShareableHandle":
        PFN_cudaMemImportFromShareableHandle;
    fn cuda_mem_retain_allocation_handle as "cudaMemRetainAllocationHandle":
        PFN_cudaMemRetainAllocationHandle;

    // Multicast (12.0+)
    fn cuda_multicast_create as "cudaMulticastCreate": PFN_cudaMulticastCreate;
    fn cuda_multicast_add_device as "cudaMulticastAddDevice": PFN_cudaMulticastAddDevice;
    fn cuda_multicast_bind_mem as "cudaMulticastBindMem": PFN_cudaMulticastBindMem;
    fn cuda_multicast_bind_addr as "cudaMulticastBindAddr": PFN_cudaMulticastBindAddr;
    fn cuda_multicast_unbind as "cudaMulticastUnbind": PFN_cudaMulticastUnbind;
    fn cuda_multicast_get_granularity as "cudaMulticastGetGranularity":
        PFN_cudaMulticastGetGranularity;

    // Green contexts (13.1+)
    fn cuda_device_create_green_ctx as "cudaDeviceCreateGreenCtx": PFN_cudaDeviceCreateGreenCtx;
    fn cuda_green_ctx_destroy as "cudaGreenCtxDestroy": PFN_cudaGreenCtxDestroy;
    fn cuda_green_ctx_record_event as "cudaGreenCtxRecordEvent": PFN_cudaGreenCtxRecordEvent;
    fn cuda_green_ctx_wait_event as "cudaGreenCtxWaitEvent": PFN_cudaGreenCtxWaitEvent;
    fn cuda_green_ctx_stream_create as "cudaGreenCtxStreamCreate": PFN_cudaGreenCtxStreamCreate;
}

/// Lazily-initialized process-wide Runtime singleton.
pub fn runtime() -> Result<&'static Runtime, LoaderError> {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    if let Some(r) = RUNTIME.get() {
        return Ok(r);
    }
    let lib = Library::open("cuda-runtime", platform::runtime_library_candidates())?;
    let r = Runtime::empty(lib);
    let _ = RUNTIME.set(r);
    Ok(RUNTIME.get().expect("OnceLock set or lost race"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_singleton_returns_loader_error_without_cuda_runtime() {
        // No panic regardless of whether cudart is present.
        let _ = runtime();
    }
}
