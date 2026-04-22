//! C function-pointer aliases for the CUDA Runtime API.
//!
//! Runtime symbols are resolved via plain `dlsym` (unlike the Driver API,
//! `libcudart` does not expose `cuGetProcAddress`). Most runtime functions
//! are stable and do not have `_v2` / `_v3` variants you need to pin to.

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_uint, c_void};

use super::status::cudaError_t;
use super::types::{cudaEvent_t, cudaKernel_t, cudaLibrary_t, cudaMemcpyKind, cudaStream_t, dim3};

// ---- version & error -------------------------------------------------------

pub type PFN_cudaRuntimeGetVersion = unsafe extern "C" fn(version: *mut c_int) -> cudaError_t;
pub type PFN_cudaDriverGetVersion = unsafe extern "C" fn(version: *mut c_int) -> cudaError_t;
pub type PFN_cudaGetLastError = unsafe extern "C" fn() -> cudaError_t;
pub type PFN_cudaPeekAtLastError = unsafe extern "C" fn() -> cudaError_t;
pub type PFN_cudaGetErrorName = unsafe extern "C" fn(error: cudaError_t) -> *const c_char;
pub type PFN_cudaGetErrorString = unsafe extern "C" fn(error: cudaError_t) -> *const c_char;

// ---- device management -----------------------------------------------------

pub type PFN_cudaGetDeviceCount = unsafe extern "C" fn(count: *mut c_int) -> cudaError_t;
pub type PFN_cudaSetDevice = unsafe extern "C" fn(device: c_int) -> cudaError_t;
pub type PFN_cudaGetDevice = unsafe extern "C" fn(device: *mut c_int) -> cudaError_t;
pub type PFN_cudaDeviceSynchronize = unsafe extern "C" fn() -> cudaError_t;
pub type PFN_cudaDeviceReset = unsafe extern "C" fn() -> cudaError_t;
pub type PFN_cudaDeviceGetAttribute =
    unsafe extern "C" fn(value: *mut c_int, attr: c_int, device: c_int) -> cudaError_t;
pub type PFN_cudaInitDevice =
    unsafe extern "C" fn(device: c_int, device_flags: c_uint, flags: c_uint) -> cudaError_t;

// ---- memory ---------------------------------------------------------------

pub type PFN_cudaMalloc =
    unsafe extern "C" fn(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t;
pub type PFN_cudaFree = unsafe extern "C" fn(dev_ptr: *mut c_void) -> cudaError_t;
pub type PFN_cudaMallocManaged =
    unsafe extern "C" fn(dev_ptr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

pub type PFN_cudaMemcpy = unsafe extern "C" fn(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;
pub type PFN_cudaMemcpyAsync = unsafe extern "C" fn(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaMemset =
    unsafe extern "C" fn(dst: *mut c_void, value: c_int, count: usize) -> cudaError_t;
pub type PFN_cudaMemsetAsync = unsafe extern "C" fn(
    dst: *mut c_void,
    value: c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t;

// ---- stream ---------------------------------------------------------------

pub type PFN_cudaStreamCreate = unsafe extern "C" fn(stream: *mut cudaStream_t) -> cudaError_t;
pub type PFN_cudaStreamCreateWithFlags =
    unsafe extern "C" fn(stream: *mut cudaStream_t, flags: c_uint) -> cudaError_t;
pub type PFN_cudaStreamDestroy = unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t;
pub type PFN_cudaStreamSynchronize = unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t;
pub type PFN_cudaStreamQuery = unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t;
pub type PFN_cudaStreamWaitEvent =
    unsafe extern "C" fn(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> cudaError_t;

// ---- event ----------------------------------------------------------------

pub type PFN_cudaEventCreate = unsafe extern "C" fn(event: *mut cudaEvent_t) -> cudaError_t;
pub type PFN_cudaEventCreateWithFlags =
    unsafe extern "C" fn(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;
pub type PFN_cudaEventDestroy = unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t;
pub type PFN_cudaEventRecord =
    unsafe extern "C" fn(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
pub type PFN_cudaEventSynchronize = unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t;
pub type PFN_cudaEventQuery = unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t;
pub type PFN_cudaEventElapsedTime =
    unsafe extern "C" fn(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;

// ---- kernel launch --------------------------------------------------------

pub type PFN_cudaLaunchKernel = unsafe extern "C" fn(
    func: *const c_void,
    grid_dim: dim3,
    block_dim: dim3,
    args: *mut *mut c_void,
    shared_mem: usize,
    stream: cudaStream_t,
) -> cudaError_t;

// ---- library management (CUDA 12.0+) --------------------------------------

pub type PFN_cudaLibraryLoadData = unsafe extern "C" fn(
    library: *mut cudaLibrary_t,
    code: *const c_void,
    jit_options: *mut c_int,
    jit_option_values: *mut *mut c_void,
    num_jit_options: c_uint,
    library_options: *mut c_int,
    library_option_values: *mut *mut c_void,
    num_library_options: c_uint,
) -> cudaError_t;
pub type PFN_cudaLibraryUnload = unsafe extern "C" fn(library: cudaLibrary_t) -> cudaError_t;
pub type PFN_cudaLibraryGetKernel = unsafe extern "C" fn(
    kernel: *mut cudaKernel_t,
    library: cudaLibrary_t,
    name: *const c_char,
) -> cudaError_t;

// ---- Stream extras --------------------------------------------------------

pub type PFN_cudaStreamCreateWithPriority =
    unsafe extern "C" fn(stream: *mut cudaStream_t, flags: c_uint, priority: c_int) -> cudaError_t;

pub type PFN_cudaStreamGetPriority =
    unsafe extern "C" fn(stream: cudaStream_t, priority: *mut c_int) -> cudaError_t;

pub type PFN_cudaStreamGetFlags =
    unsafe extern "C" fn(stream: cudaStream_t, flags: *mut c_uint) -> cudaError_t;

pub type PFN_cudaDeviceGetStreamPriorityRange =
    unsafe extern "C" fn(least_priority: *mut c_int, greatest_priority: *mut c_int) -> cudaError_t;

// ---- Peer access ----------------------------------------------------------

pub type PFN_cudaDeviceCanAccessPeer =
    unsafe extern "C" fn(can_access: *mut c_int, device: c_int, peer_device: c_int) -> cudaError_t;

pub type PFN_cudaDeviceEnablePeerAccess =
    unsafe extern "C" fn(peer_device: c_int, flags: c_uint) -> cudaError_t;

pub type PFN_cudaDeviceDisablePeerAccess = unsafe extern "C" fn(peer_device: c_int) -> cudaError_t;

// ---- Mem prefetch / advise -----------------------------------------------

pub type PFN_cudaMemPrefetchAsync = unsafe extern "C" fn(
    dev_ptr: *const c_void,
    count: usize,
    dst_device: c_int,
    stream: cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaMemAdvise = unsafe extern "C" fn(
    dev_ptr: *const c_void,
    count: usize,
    advice: c_int,
    device: c_int,
) -> cudaError_t;

pub type PFN_cudaMemGetInfo =
    unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> cudaError_t;

// ---- Managed + pinned memory ---------------------------------------------

pub type PFN_cudaMallocHost =
    unsafe extern "C" fn(pp: *mut *mut c_void, size: usize) -> cudaError_t;

pub type PFN_cudaFreeHost = unsafe extern "C" fn(ptr: *mut c_void) -> cudaError_t;

pub type PFN_cudaHostAlloc =
    unsafe extern "C" fn(pp: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

pub type PFN_cudaHostRegister =
    unsafe extern "C" fn(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

pub type PFN_cudaHostUnregister = unsafe extern "C" fn(ptr: *mut c_void) -> cudaError_t;

pub type PFN_cudaHostGetDevicePointer = unsafe extern "C" fn(
    dev_ptr: *mut *mut c_void,
    host_ptr: *mut c_void,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaHostGetFlags =
    unsafe extern "C" fn(flags: *mut c_uint, host_ptr: *mut c_void) -> cudaError_t;

// ---- Async alloc / free (CUDA 11.2+) --------------------------------------

pub type PFN_cudaMallocAsync = unsafe extern "C" fn(
    dev_ptr: *mut *mut c_void,
    size: usize,
    stream: cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaFreeAsync =
    unsafe extern "C" fn(dev_ptr: *mut c_void, stream: cudaStream_t) -> cudaError_t;

// ---- Graphs + stream capture ---------------------------------------------

use super::types::{cudaGraphExec_t, cudaGraphNode_t, cudaGraph_t};

pub type PFN_cudaGraphCreate =
    unsafe extern "C" fn(graph: *mut cudaGraph_t, flags: c_uint) -> cudaError_t;

pub type PFN_cudaGraphDestroy = unsafe extern "C" fn(graph: cudaGraph_t) -> cudaError_t;

pub type PFN_cudaGraphInstantiate = unsafe extern "C" fn(
    graph_exec: *mut cudaGraphExec_t,
    graph: cudaGraph_t,
    flags: u64,
) -> cudaError_t;

pub type PFN_cudaGraphLaunch =
    unsafe extern "C" fn(graph_exec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;

pub type PFN_cudaGraphExecDestroy =
    unsafe extern "C" fn(graph_exec: cudaGraphExec_t) -> cudaError_t;

pub type PFN_cudaGraphGetNodes = unsafe extern "C" fn(
    graph: cudaGraph_t,
    nodes: *mut cudaGraphNode_t,
    num_nodes: *mut usize,
) -> cudaError_t;

pub type PFN_cudaStreamBeginCapture =
    unsafe extern "C" fn(stream: cudaStream_t, mode: c_int) -> cudaError_t;

pub type PFN_cudaStreamEndCapture =
    unsafe extern "C" fn(stream: cudaStream_t, graph: *mut cudaGraph_t) -> cudaError_t;

pub type PFN_cudaStreamIsCapturing =
    unsafe extern "C" fn(stream: cudaStream_t, status: *mut c_int) -> cudaError_t;

// ---- Function-symbol lookup + func attrs + occupancy ----------------------

pub type PFN_cudaGetFuncBySymbol =
    unsafe extern "C" fn(func: *mut *mut c_void, symbol: *const c_void) -> cudaError_t;

pub type PFN_cudaFuncGetAttributes =
    unsafe extern "C" fn(attr: *mut c_void, func: *const c_void) -> cudaError_t;

pub type PFN_cudaFuncSetAttribute =
    unsafe extern "C" fn(func: *const c_void, attr: c_int, value: c_int) -> cudaError_t;

pub type PFN_cudaOccupancyMaxActiveBlocksPerMultiprocessor = unsafe extern "C" fn(
    num_blocks: *mut c_int,
    func: *const c_void,
    block_size: c_int,
    dynamic_smem_bytes: usize,
) -> cudaError_t;

pub type PFN_cudaOccupancyMaxPotentialBlockSize = unsafe extern "C" fn(
    min_grid_size: *mut c_int,
    block_size: *mut c_int,
    func: *const c_void,
    dyn_smem_fn: *mut c_void,
    dyn_smem_bytes: usize,
    block_size_limit: c_int,
) -> cudaError_t;

// ---- Pointer attributes ---------------------------------------------------

pub type PFN_cudaPointerGetAttributes =
    unsafe extern "C" fn(attrs: *mut c_void, ptr: *const c_void) -> cudaError_t;

// ---- 2-D memcpy + memset variants + async memset -------------------------

pub type PFN_cudaMemcpy2D = unsafe extern "C" fn(
    dst: *mut c_void,
    dst_pitch: usize,
    src: *const c_void,
    src_pitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

pub type PFN_cudaMemcpy2DAsync = unsafe extern "C" fn(
    dst: *mut c_void,
    dst_pitch: usize,
    src: *const c_void,
    src_pitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaMallocPitch = unsafe extern "C" fn(
    dev_ptr: *mut *mut c_void,
    pitch: *mut usize,
    width: usize,
    height: usize,
) -> cudaError_t;

pub type PFN_cudaMemset2D = unsafe extern "C" fn(
    dev_ptr: *mut c_void,
    pitch: usize,
    value: c_int,
    width: usize,
    height: usize,
) -> cudaError_t;

pub type PFN_cudaMemset2DAsync = unsafe extern "C" fn(
    dev_ptr: *mut c_void,
    pitch: usize,
    value: c_int,
    width: usize,
    height: usize,
    stream: cudaStream_t,
) -> cudaError_t;

// ---- Peer memcpy ---------------------------------------------------------

pub type PFN_cudaMemcpyPeer = unsafe extern "C" fn(
    dst: *mut c_void,
    dst_device: c_int,
    src: *const c_void,
    src_device: c_int,
    count: usize,
) -> cudaError_t;

pub type PFN_cudaMemcpyPeerAsync = unsafe extern "C" fn(
    dst: *mut c_void,
    dst_device: c_int,
    src: *const c_void,
    src_device: c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t;

// ---- Device properties (big struct) --------------------------------------

/// `cudaGetDeviceProperties_v2` — the full 1KB+ device-properties struct.
/// We expose it as an opaque byte buffer; callers can cast to
/// `cudaDeviceProp` from a bindgen-generated header if they need typed
/// access to the individual fields.
pub type PFN_cudaGetDeviceProperties =
    unsafe extern "C" fn(prop: *mut c_void, device: c_int) -> cudaError_t;

// ---- External memory + semaphore interop ---------------------------------
//
// `cudaExternalMemoryHandleDesc` / `cudaExternalMemoryBufferDesc` /
// `cudaExternalSemaphoreHandleDesc` / `cudaExternalSemaphore*Params` are
// binary-compatible with their Driver-API counterparts in
// [`crate::types`]. We reuse those struct definitions here rather than
// duplicating them.

use super::types::{cudaExternalMemory_t, cudaExternalSemaphore_t};

pub type PFN_cudaImportExternalMemory = unsafe extern "C" fn(
    mem_out: *mut cudaExternalMemory_t,
    mem_handle_desc: *const crate::types::CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
) -> cudaError_t;

pub type PFN_cudaDestroyExternalMemory =
    unsafe extern "C" fn(mem: cudaExternalMemory_t) -> cudaError_t;

pub type PFN_cudaExternalMemoryGetMappedBuffer = unsafe extern "C" fn(
    dev_ptr: *mut *mut c_void,
    mem: cudaExternalMemory_t,
    buf_desc: *const crate::types::CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
) -> cudaError_t;

pub type PFN_cudaExternalMemoryGetMappedMipmappedArray = unsafe extern "C" fn(
    mipmap: *mut c_void,
    mem: cudaExternalMemory_t,
    mipmap_desc: *const c_void,
) -> cudaError_t;

pub type PFN_cudaImportExternalSemaphore = unsafe extern "C" fn(
    sem_out: *mut cudaExternalSemaphore_t,
    sem_handle_desc: *const crate::types::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
) -> cudaError_t;

pub type PFN_cudaDestroyExternalSemaphore =
    unsafe extern "C" fn(sem: cudaExternalSemaphore_t) -> cudaError_t;

pub type PFN_cudaSignalExternalSemaphoresAsync = unsafe extern "C" fn(
    ext_sem_array: *const cudaExternalSemaphore_t,
    params_array: *const crate::types::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    num_ext_sems: c_uint,
    stream: cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaWaitExternalSemaphoresAsync = unsafe extern "C" fn(
    ext_sem_array: *const cudaExternalSemaphore_t,
    params_array: *const crate::types::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    num_ext_sems: c_uint,
    stream: cudaStream_t,
) -> cudaError_t;

// ---- Runtime Wave 1: host-fn launch + stream write/wait value -----------

use super::types::{
    cudaHostFn_t, cudaHostNodeParams, cudaKernelNodeParams, cudaMemAccessDesc,
    cudaMemAllocNodeParams, cudaMemLocation, cudaMemPoolProps, cudaMemPoolPtrExportData,
    cudaMemPool_t, cudaMemsetParams,
};

pub type PFN_cudaLaunchHostFunc = unsafe extern "C" fn(
    stream: cudaStream_t,
    fn_ptr: cudaHostFn_t,
    user_data: *mut c_void,
) -> cudaError_t;

pub type PFN_cudaStreamWriteValue32 = unsafe extern "C" fn(
    stream: cudaStream_t,
    addr: *mut c_void,
    value: u32,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaStreamWriteValue64 = unsafe extern "C" fn(
    stream: cudaStream_t,
    addr: *mut c_void,
    value: u64,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaStreamWaitValue32 = unsafe extern "C" fn(
    stream: cudaStream_t,
    addr: *mut c_void,
    value: u32,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaStreamWaitValue64 = unsafe extern "C" fn(
    stream: cudaStream_t,
    addr: *mut c_void,
    value: u64,
    flags: c_uint,
) -> cudaError_t;

// ---- Memory pools --------------------------------------------------------

pub type PFN_cudaMemPoolCreate =
    unsafe extern "C" fn(pool: *mut cudaMemPool_t, props: *const cudaMemPoolProps) -> cudaError_t;

pub type PFN_cudaMemPoolDestroy = unsafe extern "C" fn(pool: cudaMemPool_t) -> cudaError_t;

pub type PFN_cudaMemPoolSetAttribute =
    unsafe extern "C" fn(pool: cudaMemPool_t, attr: c_int, value: *mut c_void) -> cudaError_t;

pub type PFN_cudaMemPoolGetAttribute =
    unsafe extern "C" fn(pool: cudaMemPool_t, attr: c_int, value: *mut c_void) -> cudaError_t;

pub type PFN_cudaMemPoolTrimTo =
    unsafe extern "C" fn(pool: cudaMemPool_t, min_bytes_to_keep: usize) -> cudaError_t;

pub type PFN_cudaMemPoolSetAccess = unsafe extern "C" fn(
    pool: cudaMemPool_t,
    desc_list: *const cudaMemAccessDesc,
    count: usize,
) -> cudaError_t;

pub type PFN_cudaMemPoolGetAccess = unsafe extern "C" fn(
    flags: *mut c_int,
    pool: cudaMemPool_t,
    location: *mut cudaMemLocation,
) -> cudaError_t;

pub type PFN_cudaMallocFromPoolAsync = unsafe extern "C" fn(
    dev_ptr: *mut *mut c_void,
    size: usize,
    pool: cudaMemPool_t,
    stream: cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaDeviceGetDefaultMemPool =
    unsafe extern "C" fn(pool: *mut cudaMemPool_t, device: c_int) -> cudaError_t;

pub type PFN_cudaDeviceGetMemPool =
    unsafe extern "C" fn(pool: *mut cudaMemPool_t, device: c_int) -> cudaError_t;

pub type PFN_cudaDeviceSetMemPool =
    unsafe extern "C" fn(device: c_int, pool: cudaMemPool_t) -> cudaError_t;

pub type PFN_cudaMemPoolExportToShareableHandle = unsafe extern "C" fn(
    shareable: *mut c_void,
    pool: cudaMemPool_t,
    handle_type: c_int,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaMemPoolImportFromShareableHandle = unsafe extern "C" fn(
    pool: *mut cudaMemPool_t,
    shareable: *mut c_void,
    handle_type: c_int,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaMemPoolExportPointer = unsafe extern "C" fn(
    export_data: *mut cudaMemPoolPtrExportData,
    ptr: *mut c_void,
) -> cudaError_t;

pub type PFN_cudaMemPoolImportPointer = unsafe extern "C" fn(
    ptr: *mut *mut c_void,
    pool: cudaMemPool_t,
    export_data: *mut cudaMemPoolPtrExportData,
) -> cudaError_t;

// ---- Explicit graph node builders ----------------------------------------

pub type PFN_cudaGraphAddKernelNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    node_params: *const cudaKernelNodeParams,
) -> cudaError_t;

pub type PFN_cudaGraphAddMemsetNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    node_params: *const cudaMemsetParams,
) -> cudaError_t;

pub type PFN_cudaGraphAddMemcpyNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    node_params: *const c_void, // cudaMemcpy3DParms — opaque here
) -> cudaError_t;

pub type PFN_cudaGraphAddHostNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    node_params: *const cudaHostNodeParams,
) -> cudaError_t;

pub type PFN_cudaGraphAddEmptyNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
) -> cudaError_t;

pub type PFN_cudaGraphAddChildGraphNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    child_graph: cudaGraph_t,
) -> cudaError_t;

pub type PFN_cudaGraphAddEventRecordNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t;

pub type PFN_cudaGraphAddEventWaitNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    event: cudaEvent_t,
) -> cudaError_t;

pub type PFN_cudaGraphAddMemAllocNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    node_params: *mut cudaMemAllocNodeParams,
) -> cudaError_t;

pub type PFN_cudaGraphAddMemFreeNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    dptr: *mut c_void,
) -> cudaError_t;

// ---- Graph exec update + dependency helpers -----------------------------

pub type PFN_cudaGraphExecUpdate = unsafe extern "C" fn(
    graph_exec: cudaGraphExec_t,
    graph: cudaGraph_t,
    error_node: *mut cudaGraphNode_t,
    result: *mut c_int,
) -> cudaError_t;

pub type PFN_cudaGraphAddDependencies = unsafe extern "C" fn(
    graph: cudaGraph_t,
    from: *const cudaGraphNode_t,
    to: *const cudaGraphNode_t,
    num_dependencies: usize,
) -> cudaError_t;

pub type PFN_cudaGraphMemFreeNodeGetParams =
    unsafe extern "C" fn(node: cudaGraphNode_t, dptr_out: *mut *mut c_void) -> cudaError_t;

pub type PFN_cudaGraphNodeGetType =
    unsafe extern "C" fn(node: cudaGraphNode_t, type_out: *mut c_int) -> cudaError_t;

// ---- Runtime Wave 2: arrays + tex/surf + user objects + coop + IPC ------

use super::types::{cudaArray_t, cudaUserObject_t};

pub type PFN_cudaMallocArray = unsafe extern "C" fn(
    array: *mut cudaArray_t,
    desc: *const c_void, // cudaChannelFormatDesc — 20 bytes
    width: usize,
    height: usize,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaFreeArray = unsafe extern "C" fn(array: cudaArray_t) -> cudaError_t;

pub type PFN_cudaMemcpy2DToArray = unsafe extern "C" fn(
    dst: cudaArray_t,
    w_offset: usize,
    h_offset: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

pub type PFN_cudaMemcpy2DFromArray = unsafe extern "C" fn(
    dst: *mut c_void,
    dpitch: usize,
    src: cudaArray_t,
    w_offset: usize,
    h_offset: usize,
    width: usize,
    height: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

pub type PFN_cudaCreateTextureObject = unsafe extern "C" fn(
    tex_out: *mut u64,
    resource_desc: *const c_void,
    tex_desc: *const c_void,
    resource_view_desc: *const c_void,
) -> cudaError_t;

pub type PFN_cudaDestroyTextureObject = unsafe extern "C" fn(tex: u64) -> cudaError_t;

pub type PFN_cudaCreateSurfaceObject =
    unsafe extern "C" fn(surf_out: *mut u64, resource_desc: *const c_void) -> cudaError_t;

pub type PFN_cudaDestroySurfaceObject = unsafe extern "C" fn(surf: u64) -> cudaError_t;

// User objects for graphs
pub type PFN_cudaUserObjectCreate = unsafe extern "C" fn(
    object_out: *mut cudaUserObject_t,
    ptr: *mut c_void,
    destroy: cudaHostFn_t,
    initial_refcount: c_uint,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaUserObjectRetain =
    unsafe extern "C" fn(object: cudaUserObject_t, count: c_uint) -> cudaError_t;

pub type PFN_cudaUserObjectRelease =
    unsafe extern "C" fn(object: cudaUserObject_t, count: c_uint) -> cudaError_t;

pub type PFN_cudaGraphRetainUserObject = unsafe extern "C" fn(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: c_uint,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphReleaseUserObject = unsafe extern "C" fn(
    graph: cudaGraph_t,
    object: cudaUserObject_t,
    count: c_uint,
) -> cudaError_t;

// Cooperative launch
pub type PFN_cudaLaunchCooperativeKernel = unsafe extern "C" fn(
    func: *const c_void,
    grid_dim: super::types::dim3,
    block_dim: super::types::dim3,
    args: *mut *mut c_void,
    shared_mem: usize,
    stream: cudaStream_t,
) -> cudaError_t;

// Stream attach
pub type PFN_cudaStreamAttachMemAsync = unsafe extern "C" fn(
    stream: cudaStream_t,
    dev_ptr: *mut c_void,
    length: usize,
    flags: c_uint,
) -> cudaError_t;

// Stream attrs
pub type PFN_cudaStreamGetAttribute =
    unsafe extern "C" fn(stream: cudaStream_t, attr: c_int, value_out: *mut c_void) -> cudaError_t;

pub type PFN_cudaStreamSetAttribute =
    unsafe extern "C" fn(stream: cudaStream_t, attr: c_int, value: *const c_void) -> cudaError_t;

pub type PFN_cudaStreamCopyAttributes =
    unsafe extern "C" fn(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t;

// IPC
pub type PFN_cudaIpcGetEventHandle = unsafe extern "C" fn(
    handle_out: *mut crate::types::CUipcEventHandle,
    event: cudaEvent_t,
) -> cudaError_t;

pub type PFN_cudaIpcOpenEventHandle = unsafe extern "C" fn(
    event_out: *mut cudaEvent_t,
    handle: crate::types::CUipcEventHandle,
) -> cudaError_t;

pub type PFN_cudaIpcGetMemHandle = unsafe extern "C" fn(
    handle_out: *mut crate::types::CUipcMemHandle,
    dev_ptr: *mut c_void,
) -> cudaError_t;

pub type PFN_cudaIpcOpenMemHandle = unsafe extern "C" fn(
    dev_ptr_out: *mut *mut c_void,
    handle: crate::types::CUipcMemHandle,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaIpcCloseMemHandle = unsafe extern "C" fn(dev_ptr: *mut c_void) -> cudaError_t;

// Device flags
pub type PFN_cudaSetDeviceFlags = unsafe extern "C" fn(flags: c_uint) -> cudaError_t;
pub type PFN_cudaGetDeviceFlags = unsafe extern "C" fn(flags_out: *mut c_uint) -> cudaError_t;

// ---- Runtime Wave 3: batch mem ops + conditional nodes + driver bridge + occupancy ----

pub type PFN_cudaStreamBatchMemOp = unsafe extern "C" fn(
    stream: cudaStream_t,
    count: c_uint,
    param_array: *mut crate::types::CUstreamBatchMemOpParams,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphAddNode = unsafe extern "C" fn(
    graph_node: *mut cudaGraphNode_t,
    graph: cudaGraph_t,
    dependencies: *const cudaGraphNode_t,
    num_dependencies: usize,
    node_params: *mut c_void,
) -> cudaError_t;

pub type PFN_cudaGraphConditionalHandleCreate = unsafe extern "C" fn(
    handle_out: *mut u64,
    graph: cudaGraph_t,
    default_launch_value: c_uint,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaGetDriverEntryPoint = unsafe extern "C" fn(
    symbol: *const c_char,
    fn_ptr_out: *mut *mut c_void,
    flags: u64,
    driver_status: *mut c_int,
) -> cudaError_t;

pub type PFN_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags =
    unsafe extern "C" fn(
        num_blocks: *mut c_int,
        func: *const c_void,
        block_size: c_int,
        dynamic_smem_bytes: usize,
        flags: c_uint,
    ) -> cudaError_t;

pub type PFN_cudaOccupancyAvailableDynamicSMemPerBlock = unsafe extern "C" fn(
    dynamic_smem_size: *mut usize,
    func: *const c_void,
    num_blocks: c_int,
    block_size: c_int,
) -> cudaError_t;

// ---- Runtime Wave 4: graphics interop (core + GL + D3D + VDPAU + EGL + NvSci) ----

// `cudaGraphicsResource_t` is an opaque pointer — typedef-compatible with
// the Driver API's `CUgraphicsResource`. We reuse the Driver alias so
// both loaders speak the same ABI.
use crate::CUgraphicsResource as cudaGraphicsResource_t;

// -- Core graphics (shared across GL / D3D / VDPAU / EGL) --

pub type PFN_cudaGraphicsUnregisterResource =
    unsafe extern "C" fn(resource: cudaGraphicsResource_t) -> cudaError_t;

pub type PFN_cudaGraphicsMapResources = unsafe extern "C" fn(
    count: c_int,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaGraphicsUnmapResources = unsafe extern "C" fn(
    count: c_int,
    resources: *mut cudaGraphicsResource_t,
    stream: cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaGraphicsResourceGetMappedPointer = unsafe extern "C" fn(
    dev_ptr_out: *mut *mut c_void,
    size_out: *mut usize,
    resource: cudaGraphicsResource_t,
) -> cudaError_t;

pub type PFN_cudaGraphicsSubResourceGetMappedArray = unsafe extern "C" fn(
    array_out: *mut *mut c_void, // cudaArray_t
    resource: cudaGraphicsResource_t,
    array_index: c_uint,
    mip_level: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphicsResourceGetMappedMipmappedArray = unsafe extern "C" fn(
    mipmap_out: *mut *mut c_void, // cudaMipmappedArray_t
    resource: cudaGraphicsResource_t,
) -> cudaError_t;

pub type PFN_cudaGraphicsResourceSetMapFlags =
    unsafe extern "C" fn(resource: cudaGraphicsResource_t, flags: c_uint) -> cudaError_t;

// -- OpenGL --

pub type PFN_cudaGraphicsGLRegisterBuffer = unsafe extern "C" fn(
    resource_out: *mut cudaGraphicsResource_t,
    buffer: c_uint, // GLuint
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphicsGLRegisterImage = unsafe extern "C" fn(
    resource_out: *mut cudaGraphicsResource_t,
    image: c_uint,  // GLuint
    target: c_uint, // GLenum
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaGLGetDevices = unsafe extern "C" fn(
    cuda_device_count_out: *mut c_uint,
    cuda_devices: *mut c_int,
    cuda_device_count_in: c_uint,
    device_list: c_uint, // cudaGLDeviceList
) -> cudaError_t;

// -- D3D9 / D3D10 / D3D11 --

pub type PFN_cudaD3D9GetDevice =
    unsafe extern "C" fn(device_out: *mut c_int, adapter_name: *const c_char) -> cudaError_t;

pub type PFN_cudaD3D9GetDevices = unsafe extern "C" fn(
    cuda_device_count_out: *mut c_uint,
    cuda_devices: *mut c_int,
    cuda_device_count_in: c_uint,
    d3d_device: *mut c_void,
    device_list: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphicsD3D9RegisterResource = unsafe extern "C" fn(
    resource_out: *mut cudaGraphicsResource_t,
    d3d_resource: *mut c_void,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaD3D10GetDevice =
    unsafe extern "C" fn(device_out: *mut c_int, adapter: *mut c_void) -> cudaError_t;

pub type PFN_cudaD3D10GetDevices = unsafe extern "C" fn(
    cuda_device_count_out: *mut c_uint,
    cuda_devices: *mut c_int,
    cuda_device_count_in: c_uint,
    d3d_device: *mut c_void,
    device_list: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphicsD3D10RegisterResource = unsafe extern "C" fn(
    resource_out: *mut cudaGraphicsResource_t,
    d3d_resource: *mut c_void,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaD3D11GetDevice =
    unsafe extern "C" fn(device_out: *mut c_int, adapter: *mut c_void) -> cudaError_t;

pub type PFN_cudaD3D11GetDevices = unsafe extern "C" fn(
    cuda_device_count_out: *mut c_uint,
    cuda_devices: *mut c_int,
    cuda_device_count_in: c_uint,
    d3d_device: *mut c_void,
    device_list: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphicsD3D11RegisterResource = unsafe extern "C" fn(
    resource_out: *mut cudaGraphicsResource_t,
    d3d_resource: *mut c_void,
    flags: c_uint,
) -> cudaError_t;

// -- VDPAU --

pub type PFN_cudaVDPAUGetDevice = unsafe extern "C" fn(
    device_out: *mut c_int,
    vdp_device: *mut c_void,
    vdp_get_proc_address: *mut c_void,
) -> cudaError_t;

pub type PFN_cudaGraphicsVDPAURegisterVideoSurface = unsafe extern "C" fn(
    resource_out: *mut cudaGraphicsResource_t,
    vdp_surface: *mut c_void,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphicsVDPAURegisterOutputSurface = unsafe extern "C" fn(
    resource_out: *mut cudaGraphicsResource_t,
    vdp_surface: *mut c_void,
    flags: c_uint,
) -> cudaError_t;

// -- EGL --

pub type PFN_cudaGraphicsEGLRegisterImage = unsafe extern "C" fn(
    resource_out: *mut cudaGraphicsResource_t,
    image: *mut c_void, // EGLImageKHR
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaGraphicsResourceGetMappedEglFrame = unsafe extern "C" fn(
    egl_frame_out: *mut c_void, // cudaEglFrame
    resource: cudaGraphicsResource_t,
    index: c_uint,
    mip_level: c_uint,
) -> cudaError_t;

pub type PFN_cudaEventCreateFromEGLSync = unsafe extern "C" fn(
    event_out: *mut cudaEvent_t,
    egl_sync: *mut c_void, // EGLSyncKHR
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaEGLStreamConsumerConnect = unsafe extern "C" fn(
    connection: *mut c_void, // cudaEglStreamConnection*
    egl_stream: *mut c_void, // EGLStreamKHR
) -> cudaError_t;

pub type PFN_cudaEGLStreamConsumerDisconnect =
    unsafe extern "C" fn(connection: *mut c_void) -> cudaError_t;

pub type PFN_cudaEGLStreamConsumerAcquireFrame = unsafe extern "C" fn(
    connection: *mut c_void,
    resource_out: *mut cudaGraphicsResource_t,
    stream_out: *mut cudaStream_t,
    timeout: c_uint,
) -> cudaError_t;

pub type PFN_cudaEGLStreamConsumerReleaseFrame = unsafe extern "C" fn(
    connection: *mut c_void,
    resource: cudaGraphicsResource_t,
    stream_inout: *mut cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaEGLStreamProducerConnect = unsafe extern "C" fn(
    connection: *mut c_void,
    egl_stream: *mut c_void,
    width: c_int,
    height: c_int,
) -> cudaError_t;

pub type PFN_cudaEGLStreamProducerDisconnect =
    unsafe extern "C" fn(connection: *mut c_void) -> cudaError_t;

pub type PFN_cudaEGLStreamProducerPresentFrame = unsafe extern "C" fn(
    connection: *mut c_void,
    egl_frame: *mut c_void,
    stream_inout: *mut cudaStream_t,
) -> cudaError_t;

pub type PFN_cudaEGLStreamProducerReturnFrame = unsafe extern "C" fn(
    connection: *mut c_void,
    egl_frame_out: *mut c_void,
    stream_inout: *mut cudaStream_t,
) -> cudaError_t;

// -- NvSci --

pub type PFN_cudaDeviceGetNvSciSyncAttributes = unsafe extern "C" fn(
    attr_list: *mut c_void, // NvSciSyncAttrList
    device: c_int,
    direction: c_int,
) -> cudaError_t;

// ---- Runtime Wave 5: arrays/tex/surf + 3D + launch-ex + profiler + VMM + multicast + green ctx + tensor maps ----

use super::types::{
    cudaGreenCtx_t, cudaLaunchConfig_t, cudaMemGenericAllocationHandle_t, cudaMipmappedArray_t,
    cudaSurfaceObject_t, cudaTextureObject_t,
};

// -- Arrays (extra) + tex/surf object descriptors --

pub type PFN_cudaMallocMipmappedArray = unsafe extern "C" fn(
    mipmap: *mut cudaMipmappedArray_t,
    desc: *const c_void,   // cudaChannelFormatDesc
    extent: *const c_void, // cudaExtent
    num_levels: c_uint,
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaFreeMipmappedArray =
    unsafe extern "C" fn(mipmap: cudaMipmappedArray_t) -> cudaError_t;

pub type PFN_cudaArrayGetInfo = unsafe extern "C" fn(
    desc_out: *mut c_void,   // cudaChannelFormatDesc
    extent_out: *mut c_void, // cudaExtent
    flags_out: *mut c_uint,
    array: cudaArray_t,
) -> cudaError_t;

pub type PFN_cudaGetMipmappedArrayLevel = unsafe extern "C" fn(
    level_out: *mut cudaArray_t,
    mipmap: cudaMipmappedArray_t,
    level: c_uint,
) -> cudaError_t;

pub type PFN_cudaGetTextureObjectResourceDesc = unsafe extern "C" fn(
    desc_out: *mut c_void, // cudaResourceDesc
    tex_object: cudaTextureObject_t,
) -> cudaError_t;

pub type PFN_cudaGetTextureObjectTextureDesc = unsafe extern "C" fn(
    desc_out: *mut c_void, // cudaTextureDesc
    tex_object: cudaTextureObject_t,
) -> cudaError_t;

pub type PFN_cudaGetTextureObjectResourceViewDesc = unsafe extern "C" fn(
    desc_out: *mut c_void, // cudaResourceViewDesc
    tex_object: cudaTextureObject_t,
) -> cudaError_t;

pub type PFN_cudaGetSurfaceObjectResourceDesc = unsafe extern "C" fn(
    desc_out: *mut c_void, // cudaResourceDesc
    surf_object: cudaSurfaceObject_t,
) -> cudaError_t;

// Signature note: the *existing* `PFN_cudaCreateTextureObject` already
// uses `*const c_void` for the descriptors, so no change there.

// -- 3D memcpy --

pub type PFN_cudaMemcpy3D = unsafe extern "C" fn(params: *const c_void) -> cudaError_t;
pub type PFN_cudaMemcpy3DAsync =
    unsafe extern "C" fn(params: *const c_void, stream: cudaStream_t) -> cudaError_t;
pub type PFN_cudaMemcpy3DPeer = unsafe extern "C" fn(params: *const c_void) -> cudaError_t;
pub type PFN_cudaMemcpy3DPeerAsync =
    unsafe extern "C" fn(params: *const c_void, stream: cudaStream_t) -> cudaError_t;

pub type PFN_cudaMemset3D = unsafe extern "C" fn(
    pitched_dev_ptr: *mut c_void, // cudaPitchedPtr (passed by value in C, but CUDA runtime's C ABI
    // copies via pointer on some platforms — we just pass the 32-byte
    // struct through as *mut for portability).
    value: c_int,
    extent: *const c_void, // cudaExtent
) -> cudaError_t;

pub type PFN_cudaMalloc3D = unsafe extern "C" fn(
    pitched_dev_ptr: *mut c_void, // out
    extent: *const c_void,        // cudaExtent
) -> cudaError_t;

pub type PFN_cudaMalloc3DArray = unsafe extern "C" fn(
    array: *mut cudaArray_t,
    desc: *const c_void,
    extent: *const c_void, // cudaExtent
    flags: c_uint,
) -> cudaError_t;

// -- Launch-ex / cluster --

pub type PFN_cudaLaunchKernelEx = unsafe extern "C" fn(
    config: *const cudaLaunchConfig_t,
    func: *const c_void,
    args: *mut *mut c_void,
) -> cudaError_t;

// -- Profiler --

pub type PFN_cudaProfilerStart = unsafe extern "C" fn() -> cudaError_t;
pub type PFN_cudaProfilerStop = unsafe extern "C" fn() -> cudaError_t;

// -- Tensor maps (Hopper, driver-hosted; runtime doesn't expose dedicated PFNs) --
// No runtime PFNs exist for tensor map encoding — the Hopper tensor-map
// API (`cuTensorMapEncodeTiled`, etc.) is Driver-API only. The Runtime
// accepts an already-built `CUtensorMap` via regular pointer-struct
// kernel-argument marshaling. So we skip the runtime tensor-map PFNs.

// -- VMM --

pub type PFN_cudaMemAddressReserve = unsafe extern "C" fn(
    dev_ptr_out: *mut *mut c_void,
    size: usize,
    alignment: usize,
    addr: *mut c_void,
    flags: u64,
) -> cudaError_t;

pub type PFN_cudaMemAddressFree =
    unsafe extern "C" fn(dev_ptr: *mut c_void, size: usize) -> cudaError_t;

pub type PFN_cudaMemCreate = unsafe extern "C" fn(
    handle_out: *mut cudaMemGenericAllocationHandle_t,
    size: usize,
    prop: *const c_void, // cudaMemAllocationProp
    flags: u64,
) -> cudaError_t;

pub type PFN_cudaMemRelease =
    unsafe extern "C" fn(handle: cudaMemGenericAllocationHandle_t) -> cudaError_t;

pub type PFN_cudaMemMap = unsafe extern "C" fn(
    ptr: *mut c_void,
    size: usize,
    offset: usize,
    handle: cudaMemGenericAllocationHandle_t,
    flags: u64,
) -> cudaError_t;

pub type PFN_cudaMemUnmap = unsafe extern "C" fn(ptr: *mut c_void, size: usize) -> cudaError_t;

pub type PFN_cudaMemSetAccess = unsafe extern "C" fn(
    ptr: *mut c_void,
    size: usize,
    desc: *const c_void, // cudaMemAccessDesc
    count: usize,
) -> cudaError_t;

pub type PFN_cudaMemGetAccess = unsafe extern "C" fn(
    flags_out: *mut u64,
    location: *const c_void, // cudaMemLocation
    ptr: *mut c_void,
) -> cudaError_t;

pub type PFN_cudaMemGetAllocationGranularity = unsafe extern "C" fn(
    granularity_out: *mut usize,
    prop: *const c_void, // cudaMemAllocationProp
    option: c_int,
) -> cudaError_t;

pub type PFN_cudaMemGetAllocationPropertiesFromHandle = unsafe extern "C" fn(
    prop_out: *mut c_void, // cudaMemAllocationProp
    handle: cudaMemGenericAllocationHandle_t,
) -> cudaError_t;

pub type PFN_cudaMemExportToShareableHandle = unsafe extern "C" fn(
    shareable_handle_out: *mut c_void,
    handle: cudaMemGenericAllocationHandle_t,
    handle_type: c_int,
    flags: u64,
) -> cudaError_t;

pub type PFN_cudaMemImportFromShareableHandle = unsafe extern "C" fn(
    handle_out: *mut cudaMemGenericAllocationHandle_t,
    os_handle: *mut c_void,
    shareable_handle_type: c_int,
) -> cudaError_t;

pub type PFN_cudaMemRetainAllocationHandle = unsafe extern "C" fn(
    handle_out: *mut cudaMemGenericAllocationHandle_t,
    addr: *mut c_void,
) -> cudaError_t;

// -- Multicast (12.0+) --

pub type PFN_cudaMulticastCreate = unsafe extern "C" fn(
    mc_handle_out: *mut cudaMemGenericAllocationHandle_t,
    prop: *const c_void, // cudaMulticastObjectProp
) -> cudaError_t;

pub type PFN_cudaMulticastAddDevice =
    unsafe extern "C" fn(mc_handle: cudaMemGenericAllocationHandle_t, device: c_int) -> cudaError_t;

pub type PFN_cudaMulticastBindMem = unsafe extern "C" fn(
    mc_handle: cudaMemGenericAllocationHandle_t,
    mc_offset: usize,
    mem_handle: cudaMemGenericAllocationHandle_t,
    mem_offset: usize,
    size: usize,
    flags: u64,
) -> cudaError_t;

pub type PFN_cudaMulticastBindAddr = unsafe extern "C" fn(
    mc_handle: cudaMemGenericAllocationHandle_t,
    mc_offset: usize,
    mem_ptr: *mut c_void,
    size: usize,
    flags: u64,
) -> cudaError_t;

pub type PFN_cudaMulticastUnbind = unsafe extern "C" fn(
    mc_handle: cudaMemGenericAllocationHandle_t,
    device: c_int,
    mc_offset: usize,
    size: usize,
) -> cudaError_t;

pub type PFN_cudaMulticastGetGranularity = unsafe extern "C" fn(
    granularity_out: *mut usize,
    prop: *const c_void, // cudaMulticastObjectProp
    option: c_int,
) -> cudaError_t;

// -- Green contexts (Runtime, 13.1+) --

pub type PFN_cudaDeviceCreateGreenCtx = unsafe extern "C" fn(
    green_ctx_out: *mut cudaGreenCtx_t,
    desc: *const c_void, // cudaDevResourceDesc
    flags: c_uint,
) -> cudaError_t;

pub type PFN_cudaGreenCtxDestroy = unsafe extern "C" fn(green_ctx: cudaGreenCtx_t) -> cudaError_t;

pub type PFN_cudaGreenCtxRecordEvent =
    unsafe extern "C" fn(green_ctx: cudaGreenCtx_t, event: cudaEvent_t) -> cudaError_t;

pub type PFN_cudaGreenCtxWaitEvent =
    unsafe extern "C" fn(green_ctx: cudaGreenCtx_t, event: cudaEvent_t) -> cudaError_t;

pub type PFN_cudaGreenCtxStreamCreate = unsafe extern "C" fn(
    stream_out: *mut cudaStream_t,
    green_ctx: cudaGreenCtx_t,
    flags: c_uint,
    priority: c_int,
) -> cudaError_t;
