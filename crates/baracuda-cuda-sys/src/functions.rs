//! C function-pointer aliases for the CUDA Driver API.
//!
//! Each `PFN_*` alias is the signature of the corresponding driver function
//! as it appears in `cuda.h`. The [`Driver`](crate::Driver) struct caches
//! a lazily-resolved function pointer per symbol.

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_uint, c_void};

use crate::status::CUresult;
use crate::types::{
    CUcontext, CUdevice, CUdeviceptr, CUevent, CUfunction, CUgraph, CUgraphExec, CUgraphNode,
    CUmodule, CUstream,
};

// ---- bootstrap -------------------------------------------------------------

/// CUDA 11.3+ driver entry-point resolver. We resolve every *other* symbol
/// through this function so we pick the correct `_ptsz` / `_v2` / `_v3`
/// variant for the installed driver and the process's stream-mode choice.
pub type PFN_cuGetProcAddress = unsafe extern "C" fn(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
) -> CUresult;

// ---- initialization & version ---------------------------------------------

pub type PFN_cuInit = unsafe extern "C" fn(flags: c_uint) -> CUresult;
pub type PFN_cuDriverGetVersion = unsafe extern "C" fn(version: *mut c_int) -> CUresult;

// ---- errors ---------------------------------------------------------------

pub type PFN_cuGetErrorName =
    unsafe extern "C" fn(error: CUresult, out: *mut *const c_char) -> CUresult;
pub type PFN_cuGetErrorString =
    unsafe extern "C" fn(error: CUresult, out: *mut *const c_char) -> CUresult;

// ---- device management ----------------------------------------------------

pub type PFN_cuDeviceGetCount = unsafe extern "C" fn(count: *mut c_int) -> CUresult;
pub type PFN_cuDeviceGet = unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult;
pub type PFN_cuDeviceGetName =
    unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
pub type PFN_cuDeviceGetAttribute =
    unsafe extern "C" fn(out: *mut c_int, attr: c_int, dev: CUdevice) -> CUresult;
pub type PFN_cuDeviceTotalMem = unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult;

// ---- context management ---------------------------------------------------

pub type PFN_cuCtxCreate =
    unsafe extern "C" fn(ctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
pub type PFN_cuCtxDestroy = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
pub type PFN_cuCtxGetCurrent = unsafe extern "C" fn(ctx: *mut CUcontext) -> CUresult;
pub type PFN_cuCtxSetCurrent = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
pub type PFN_cuCtxPushCurrent = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
pub type PFN_cuCtxPopCurrent = unsafe extern "C" fn(ctx: *mut CUcontext) -> CUresult;
pub type PFN_cuCtxSynchronize = unsafe extern "C" fn() -> CUresult;

// ---- primary context ------------------------------------------------------

pub type PFN_cuDevicePrimaryCtxRetain =
    unsafe extern "C" fn(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
pub type PFN_cuDevicePrimaryCtxRelease = unsafe extern "C" fn(dev: CUdevice) -> CUresult;
pub type PFN_cuDevicePrimaryCtxReset = unsafe extern "C" fn(dev: CUdevice) -> CUresult;

// ---- memory management ----------------------------------------------------

pub type PFN_cuMemAlloc = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytes: usize) -> CUresult;
pub type PFN_cuMemFree = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;

pub type PFN_cuMemcpyHtoD =
    unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
pub type PFN_cuMemcpyDtoH =
    unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
pub type PFN_cuMemcpyDtoD =
    unsafe extern "C" fn(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CUresult;

pub type PFN_cuMemcpyHtoDAsync = unsafe extern "C" fn(
    dst: CUdeviceptr,
    src: *const c_void,
    bytes: usize,
    stream: CUstream,
) -> CUresult;
pub type PFN_cuMemcpyDtoHAsync = unsafe extern "C" fn(
    dst: *mut c_void,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuMemsetD8 =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u8, count: usize) -> CUresult;
pub type PFN_cuMemsetD32 =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u32, count: usize) -> CUresult;

// ---- stream management ----------------------------------------------------

pub type PFN_cuStreamCreate =
    unsafe extern "C" fn(stream: *mut CUstream, flags: c_uint) -> CUresult;
pub type PFN_cuStreamDestroy = unsafe extern "C" fn(stream: CUstream) -> CUresult;
pub type PFN_cuStreamSynchronize = unsafe extern "C" fn(stream: CUstream) -> CUresult;
pub type PFN_cuStreamQuery = unsafe extern "C" fn(stream: CUstream) -> CUresult;
pub type PFN_cuStreamWaitEvent =
    unsafe extern "C" fn(stream: CUstream, event: CUevent, flags: c_uint) -> CUresult;

// ---- event management -----------------------------------------------------

pub type PFN_cuEventCreate = unsafe extern "C" fn(event: *mut CUevent, flags: c_uint) -> CUresult;
pub type PFN_cuEventDestroy = unsafe extern "C" fn(event: CUevent) -> CUresult;
pub type PFN_cuEventRecord = unsafe extern "C" fn(event: CUevent, stream: CUstream) -> CUresult;
pub type PFN_cuEventSynchronize = unsafe extern "C" fn(event: CUevent) -> CUresult;
pub type PFN_cuEventQuery = unsafe extern "C" fn(event: CUevent) -> CUresult;
pub type PFN_cuEventElapsedTime =
    unsafe extern "C" fn(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;

// ---- module loading & kernel launch --------------------------------------

pub type PFN_cuModuleLoadData =
    unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult;
pub type PFN_cuModuleUnload = unsafe extern "C" fn(module: CUmodule) -> CUresult;
pub type PFN_cuModuleGetFunction =
    unsafe extern "C" fn(func: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult;

pub type PFN_cuLaunchKernel = unsafe extern "C" fn(
    func: CUfunction,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: CUstream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult;

// ---- stream capture / graphs ---------------------------------------------

pub type PFN_cuStreamBeginCapture =
    unsafe extern "C" fn(stream: CUstream, mode: c_uint) -> CUresult;
pub type PFN_cuStreamEndCapture =
    unsafe extern "C" fn(stream: CUstream, graph: *mut CUgraph) -> CUresult;
pub type PFN_cuStreamIsCapturing =
    unsafe extern "C" fn(stream: CUstream, status: *mut c_uint) -> CUresult;
pub type PFN_cuGraphCreate = unsafe extern "C" fn(graph: *mut CUgraph, flags: c_uint) -> CUresult;
pub type PFN_cuGraphDestroy = unsafe extern "C" fn(graph: CUgraph) -> CUresult;
pub type PFN_cuGraphInstantiateWithFlags =
    unsafe extern "C" fn(graph_exec: *mut CUgraphExec, graph: CUgraph, flags: u64) -> CUresult;
pub type PFN_cuGraphLaunch =
    unsafe extern "C" fn(graph_exec: CUgraphExec, stream: CUstream) -> CUresult;
pub type PFN_cuGraphExecDestroy = unsafe extern "C" fn(graph_exec: CUgraphExec) -> CUresult;
pub type PFN_cuGraphGetNodes = unsafe extern "C" fn(
    graph: CUgraph,
    nodes: *mut *mut c_void,
    num_nodes: *mut usize,
) -> CUresult;

// ---- stream-ordered memory allocation (CUDA 11.2+) ----------------------

pub type PFN_cuMemAllocAsync =
    unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult;
pub type PFN_cuMemFreeAsync = unsafe extern "C" fn(dptr: CUdeviceptr, stream: CUstream) -> CUresult;

// ---- Wave 1: occupancy, unified memory, peer, pointer attrs --------------

pub type PFN_cuOccupancyMaxActiveBlocksPerMultiprocessor = unsafe extern "C" fn(
    num_blocks: *mut c_int,
    func: CUfunction,
    block_size: c_int,
    dynamic_smem_bytes: usize,
) -> CUresult;

pub type PFN_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags =
    unsafe extern "C" fn(
        num_blocks: *mut c_int,
        func: CUfunction,
        block_size: c_int,
        dynamic_smem_bytes: usize,
        flags: c_uint,
    ) -> CUresult;

/// Signature of the `block_size_to_dynamic_smem_size` callback.
pub type CUoccupancyB2DSize = Option<unsafe extern "C" fn(block_size: c_int) -> usize>;

pub type PFN_cuOccupancyMaxPotentialBlockSize = unsafe extern "C" fn(
    min_grid_size: *mut c_int,
    block_size: *mut c_int,
    func: CUfunction,
    block_size_to_dynamic_smem_size: CUoccupancyB2DSize,
    dynamic_smem_bytes: usize,
    block_size_limit: c_int,
) -> CUresult;

pub type PFN_cuOccupancyAvailableDynamicSMemPerBlock = unsafe extern "C" fn(
    dynamic_smem_size: *mut usize,
    func: CUfunction,
    num_blocks: c_int,
    block_size: c_int,
) -> CUresult;

pub type PFN_cuMemAllocManaged =
    unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytes: usize, flags: c_uint) -> CUresult;

pub type PFN_cuMemAdvise = unsafe extern "C" fn(
    devptr: CUdeviceptr,
    count: usize,
    advice: c_int,
    device: CUdevice,
) -> CUresult;

pub type PFN_cuMemPrefetchAsync = unsafe extern "C" fn(
    devptr: CUdeviceptr,
    count: usize,
    dst_device: CUdevice,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuMemGetInfo = unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> CUresult;

// ---- context queries -----------------------------------------------------

pub type PFN_cuCtxGetDevice = unsafe extern "C" fn(device: *mut CUdevice) -> CUresult;
pub type PFN_cuCtxGetApiVersion =
    unsafe extern "C" fn(ctx: CUcontext, version: *mut c_uint) -> CUresult;
pub type PFN_cuCtxGetFlags = unsafe extern "C" fn(flags: *mut c_uint) -> CUresult;
pub type PFN_cuCtxGetLimit = unsafe extern "C" fn(pvalue: *mut usize, limit: c_uint) -> CUresult;
pub type PFN_cuCtxSetLimit = unsafe extern "C" fn(limit: c_uint, value: usize) -> CUresult;
pub type PFN_cuCtxGetCacheConfig = unsafe extern "C" fn(pconfig: *mut c_uint) -> CUresult;
pub type PFN_cuCtxSetCacheConfig = unsafe extern "C" fn(config: c_uint) -> CUresult;
pub type PFN_cuCtxGetStreamPriorityRange =
    unsafe extern "C" fn(least_priority: *mut c_int, greatest_priority: *mut c_int) -> CUresult;

// ---- peer access ---------------------------------------------------------

pub type PFN_cuDeviceCanAccessPeer = unsafe extern "C" fn(
    can_access_peer: *mut c_int,
    dev: CUdevice,
    peer_dev: CUdevice,
) -> CUresult;
pub type PFN_cuCtxEnablePeerAccess =
    unsafe extern "C" fn(peer_context: CUcontext, flags: c_uint) -> CUresult;
pub type PFN_cuCtxDisablePeerAccess = unsafe extern "C" fn(peer_context: CUcontext) -> CUresult;

// ---- pointer attributes --------------------------------------------------

pub type PFN_cuPointerGetAttribute =
    unsafe extern "C" fn(data: *mut c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult;

// ---- stream priority/flags + host func -----------------------------------

pub type PFN_cuStreamCreateWithPriority =
    unsafe extern "C" fn(stream: *mut CUstream, flags: c_uint, priority: c_int) -> CUresult;
pub type PFN_cuStreamGetPriority =
    unsafe extern "C" fn(stream: CUstream, priority: *mut c_int) -> CUresult;
pub type PFN_cuStreamGetFlags =
    unsafe extern "C" fn(stream: CUstream, flags: *mut c_uint) -> CUresult;
pub type PFN_cuStreamGetCtx =
    unsafe extern "C" fn(stream: CUstream, pctx: *mut CUcontext) -> CUresult;
pub type CUhostFn = Option<unsafe extern "C" fn(user_data: *mut c_void)>;
pub type PFN_cuLaunchHostFunc =
    unsafe extern "C" fn(stream: CUstream, fn_ptr: CUhostFn, user_data: *mut c_void) -> CUresult;

// ---- event flags ---------------------------------------------------------

pub type PFN_cuEventRecordWithFlags =
    unsafe extern "C" fn(event: CUevent, stream: CUstream, flags: c_uint) -> CUresult;

// ---- primary context state ----------------------------------------------

pub type PFN_cuDevicePrimaryCtxGetState =
    unsafe extern "C" fn(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult;
pub type PFN_cuDevicePrimaryCtxSetFlags =
    unsafe extern "C" fn(dev: CUdevice, flags: c_uint) -> CUresult;

// ---- Wave 2: kernel attrs, module globals, module load with JIT options --

pub type PFN_cuFuncGetAttribute =
    unsafe extern "C" fn(pi: *mut c_int, attrib: c_int, hfunc: CUfunction) -> CUresult;
pub type PFN_cuFuncSetAttribute =
    unsafe extern "C" fn(hfunc: CUfunction, attrib: c_int, value: c_int) -> CUresult;

pub type PFN_cuModuleGetGlobal = unsafe extern "C" fn(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    module: CUmodule,
    name: *const c_char,
) -> CUresult;

/// `cuModuleLoadDataEx` — load a PTX/cubin image with JIT compiler options.
/// `options` is an array of `CUjit_option` (i32) values; `option_values` is
/// a parallel array of `void*` whose interpretation depends on the option.
pub type PFN_cuModuleLoadDataEx = unsafe extern "C" fn(
    module: *mut CUmodule,
    image: *const c_void,
    num_options: c_uint,
    options: *mut c_int,
    option_values: *mut *mut c_void,
) -> CUresult;

// ---- Wave 4: 2D allocation + memcpy --------------------------------------

pub type PFN_cuMemAllocPitch = unsafe extern "C" fn(
    dptr: *mut CUdeviceptr,
    pitch: *mut usize,
    width_in_bytes: usize,
    height: usize,
    element_size_bytes: c_uint,
) -> CUresult;

pub type PFN_cuMemcpy2D =
    unsafe extern "C" fn(p_copy: *const crate::types::CUDA_MEMCPY2D) -> CUresult;
pub type PFN_cuMemcpy2DAsync =
    unsafe extern "C" fn(p_copy: *const crate::types::CUDA_MEMCPY2D, stream: CUstream) -> CUresult;

// ---- Wave 3: cuLaunchKernelEx + Library Management -----------------------

use crate::types::CUlaunchConfig;
use crate::{CUkernel, CUlibrary};

pub type PFN_cuLaunchKernelEx = unsafe extern "C" fn(
    config: *const CUlaunchConfig,
    f: CUfunction,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult;

/// `cuLibraryLoadData` — context-independent module load (CUDA 12.0+).
///
/// The "JIT options" and "library options" are each a parallel pair of
/// `CUjit_option` / `CUlibraryOption` arrays with matching `void**` values.
/// For v0.1 we only expose the no-options call shape (pass `0` / `null` for
/// counts and arrays).
pub type PFN_cuLibraryLoadData = unsafe extern "C" fn(
    library: *mut CUlibrary,
    code: *const c_void,
    jit_options: *mut c_int,
    jit_option_values: *mut *mut c_void,
    num_jit_options: c_uint,
    library_options: *mut c_int,
    library_option_values: *mut *mut c_void,
    num_library_options: c_uint,
) -> CUresult;

pub type PFN_cuLibraryUnload = unsafe extern "C" fn(library: CUlibrary) -> CUresult;

pub type PFN_cuLibraryGetKernel = unsafe extern "C" fn(
    kernel: *mut CUkernel,
    library: CUlibrary,
    name: *const c_char,
) -> CUresult;

pub type PFN_cuLibraryGetGlobal = unsafe extern "C" fn(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    library: CUlibrary,
    name: *const c_char,
) -> CUresult;

pub type PFN_cuKernelGetFunction =
    unsafe extern "C" fn(pfunc: *mut CUfunction, kernel: CUkernel) -> CUresult;

// ---- Wave 5: explicit graph node construction ----------------------------

/// `cuGraphAddKernelNode_v2` — v2 shape, matches our
/// [`crate::types::CUDA_KERNEL_NODE_PARAMS`] layout (with `kern` + `ctx`
/// trailing fields). Pinned to `_v2` via [`has_version_suffix`] routing.
pub type PFN_cuGraphAddKernelNode = unsafe extern "C" fn(
    graph_node: *mut CUgraphNode,
    graph: CUgraph,
    dependencies: *const CUgraphNode,
    num_dependencies: usize,
    node_params: *const crate::types::CUDA_KERNEL_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphAddEmptyNode = unsafe extern "C" fn(
    graph_node: *mut CUgraphNode,
    graph: CUgraph,
    dependencies: *const CUgraphNode,
    num_dependencies: usize,
) -> CUresult;

/// `cuGraphAddMemsetNode` — always took a trailing `CUcontext` since CUDA
/// 10 (memsets can target any context, unlike kernel nodes).
pub type PFN_cuGraphAddMemsetNode = unsafe extern "C" fn(
    graph_node: *mut CUgraphNode,
    graph: CUgraph,
    dependencies: *const CUgraphNode,
    num_dependencies: usize,
    memset_params: *const crate::types::CUDA_MEMSET_NODE_PARAMS,
    ctx: CUcontext,
) -> CUresult;

pub type PFN_cuGraphDestroyNode = unsafe extern "C" fn(node: CUgraphNode) -> CUresult;

pub type PFN_cuGraphClone =
    unsafe extern "C" fn(clone: *mut CUgraph, original: CUgraph) -> CUresult;

// ---- Wave 6: arrays, textures, surfaces ----------------------------------

use crate::types::{
    CUarray, CUsurfObject, CUtexObject, CUDA_ARRAY_DESCRIPTOR, CUDA_RESOURCE_DESC,
    CUDA_TEXTURE_DESC,
};

pub type PFN_cuArrayCreate =
    unsafe extern "C" fn(array: *mut CUarray, desc: *const CUDA_ARRAY_DESCRIPTOR) -> CUresult;

pub type PFN_cuArrayDestroy = unsafe extern "C" fn(array: CUarray) -> CUresult;

pub type PFN_cuTexObjectCreate = unsafe extern "C" fn(
    tex: *mut CUtexObject,
    res_desc: *const CUDA_RESOURCE_DESC,
    tex_desc: *const CUDA_TEXTURE_DESC,
    res_view_desc: *const c_void,
) -> CUresult;

pub type PFN_cuTexObjectDestroy = unsafe extern "C" fn(tex: CUtexObject) -> CUresult;

pub type PFN_cuSurfObjectCreate =
    unsafe extern "C" fn(surf: *mut CUsurfObject, res_desc: *const CUDA_RESOURCE_DESC) -> CUresult;

pub type PFN_cuSurfObjectDestroy = unsafe extern "C" fn(surf: CUsurfObject) -> CUresult;

// ---- Wave 7: virtual memory management (VMM) ----------------------------

use crate::types::{CUmemAccessDesc, CUmemAllocationProp, CUmemGenericAllocationHandle};

pub type PFN_cuMemAddressReserve = unsafe extern "C" fn(
    ptr: *mut CUdeviceptr,
    size: usize,
    alignment: usize,
    addr: CUdeviceptr,
    flags: u64,
) -> CUresult;

pub type PFN_cuMemAddressFree = unsafe extern "C" fn(ptr: CUdeviceptr, size: usize) -> CUresult;

pub type PFN_cuMemCreate = unsafe extern "C" fn(
    handle: *mut CUmemGenericAllocationHandle,
    size: usize,
    prop: *const CUmemAllocationProp,
    flags: u64,
) -> CUresult;

pub type PFN_cuMemRelease = unsafe extern "C" fn(handle: CUmemGenericAllocationHandle) -> CUresult;

pub type PFN_cuMemMap = unsafe extern "C" fn(
    ptr: CUdeviceptr,
    size: usize,
    offset: usize,
    handle: CUmemGenericAllocationHandle,
    flags: u64,
) -> CUresult;

pub type PFN_cuMemUnmap = unsafe extern "C" fn(ptr: CUdeviceptr, size: usize) -> CUresult;

pub type PFN_cuMemSetAccess = unsafe extern "C" fn(
    ptr: CUdeviceptr,
    size: usize,
    desc: *const CUmemAccessDesc,
    count: usize,
) -> CUresult;

pub type PFN_cuMemGetAllocationGranularity = unsafe extern "C" fn(
    granularity: *mut usize,
    prop: *const CUmemAllocationProp,
    option: c_int,
) -> CUresult;

// ---- Wave 8: memory pools -----------------------------------------------

use crate::types::{CUmemPoolProps, CUmemPoolPtrExportData, CUmemoryPool};

pub type PFN_cuMemPoolCreate =
    unsafe extern "C" fn(pool: *mut CUmemoryPool, props: *const CUmemPoolProps) -> CUresult;

pub type PFN_cuMemPoolDestroy = unsafe extern "C" fn(pool: CUmemoryPool) -> CUresult;

pub type PFN_cuMemPoolSetAttribute =
    unsafe extern "C" fn(pool: CUmemoryPool, attr: c_int, value: *mut c_void) -> CUresult;

pub type PFN_cuMemPoolGetAttribute =
    unsafe extern "C" fn(pool: CUmemoryPool, attr: c_int, value: *mut c_void) -> CUresult;

pub type PFN_cuMemPoolTrimTo =
    unsafe extern "C" fn(pool: CUmemoryPool, min_bytes_to_keep: usize) -> CUresult;

pub type PFN_cuMemPoolSetAccess =
    unsafe extern "C" fn(pool: CUmemoryPool, map: *const CUmemAccessDesc, count: usize) -> CUresult;

pub type PFN_cuMemPoolGetAccess = unsafe extern "C" fn(
    flags: *mut c_int,
    pool: CUmemoryPool,
    location: *mut crate::types::CUmemLocation,
) -> CUresult;

pub type PFN_cuMemAllocFromPoolAsync = unsafe extern "C" fn(
    dptr: *mut CUdeviceptr,
    bytes: usize,
    pool: CUmemoryPool,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuDeviceGetDefaultMemPool =
    unsafe extern "C" fn(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult;

pub type PFN_cuDeviceGetMemPool =
    unsafe extern "C" fn(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult;

pub type PFN_cuDeviceSetMemPool =
    unsafe extern "C" fn(dev: CUdevice, pool: CUmemoryPool) -> CUresult;

pub type PFN_cuMemPoolExportToShareableHandle = unsafe extern "C" fn(
    handle_out: *mut c_void,
    pool: CUmemoryPool,
    handle_type: c_int,
    flags: u64,
) -> CUresult;

pub type PFN_cuMemPoolImportFromShareableHandle = unsafe extern "C" fn(
    pool_out: *mut CUmemoryPool,
    handle: *mut c_void,
    handle_type: c_int,
    flags: u64,
) -> CUresult;

pub type PFN_cuMemPoolExportPointer =
    unsafe extern "C" fn(share_data_out: *mut CUmemPoolPtrExportData, ptr: CUdeviceptr) -> CUresult;

pub type PFN_cuMemPoolImportPointer = unsafe extern "C" fn(
    ptr_out: *mut CUdeviceptr,
    pool: CUmemoryPool,
    share_data: *mut CUmemPoolPtrExportData,
) -> CUresult;

// ---- Wave 9: external memory / semaphore interop ------------------------

use crate::types::{
    CUexternalMemory, CUexternalSemaphore, CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC, CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
};

pub type PFN_cuImportExternalMemory = unsafe extern "C" fn(
    mem_out: *mut CUexternalMemory,
    mem_handle_desc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
) -> CUresult;

pub type PFN_cuDestroyExternalMemory = unsafe extern "C" fn(mem: CUexternalMemory) -> CUresult;

pub type PFN_cuExternalMemoryGetMappedBuffer = unsafe extern "C" fn(
    dptr: *mut CUdeviceptr,
    mem: CUexternalMemory,
    buf_desc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
) -> CUresult;

/// Mipmapped-array variant — descriptor points at `CUDA_ARRAY3D_DESCRIPTOR`,
/// which baracuda v0.1 does not yet expose. Provided at sys level for
/// forward compat; a safe wrapper lands with Wave 10 (3D arrays).
pub type PFN_cuExternalMemoryGetMappedMipmappedArray = unsafe extern "C" fn(
    mipmap: *mut c_void,
    mem: CUexternalMemory,
    mipmap_desc: *const c_void,
) -> CUresult;

pub type PFN_cuImportExternalSemaphore = unsafe extern "C" fn(
    sem_out: *mut CUexternalSemaphore,
    sem_handle_desc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
) -> CUresult;

pub type PFN_cuDestroyExternalSemaphore =
    unsafe extern "C" fn(sem: CUexternalSemaphore) -> CUresult;

pub type PFN_cuSignalExternalSemaphoresAsync = unsafe extern "C" fn(
    ext_sem_array: *const CUexternalSemaphore,
    params_array: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    num_ext_sems: c_uint,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuWaitExternalSemaphoresAsync = unsafe extern "C" fn(
    ext_sem_array: *const CUexternalSemaphore,
    params_array: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    num_ext_sems: c_uint,
    stream: CUstream,
) -> CUresult;

// ---- Wave 10: 3D memcpy + 3D arrays + mipmapped arrays ------------------

use crate::types::{CUmipmappedArray, CUDA_ARRAY3D_DESCRIPTOR, CUDA_MEMCPY3D};

pub type PFN_cuArray3DCreate = unsafe extern "C" fn(
    array: *mut crate::types::CUarray,
    desc: *const CUDA_ARRAY3D_DESCRIPTOR,
) -> CUresult;

pub type PFN_cuArray3DGetDescriptor = unsafe extern "C" fn(
    desc: *mut CUDA_ARRAY3D_DESCRIPTOR,
    array: crate::types::CUarray,
) -> CUresult;

pub type PFN_cuMemcpy3D = unsafe extern "C" fn(p_copy: *const CUDA_MEMCPY3D) -> CUresult;

pub type PFN_cuMemcpy3DAsync =
    unsafe extern "C" fn(p_copy: *const CUDA_MEMCPY3D, stream: CUstream) -> CUresult;

pub type PFN_cuMipmappedArrayCreate = unsafe extern "C" fn(
    mipmap: *mut CUmipmappedArray,
    desc: *const CUDA_ARRAY3D_DESCRIPTOR,
    num_mipmap_levels: c_uint,
) -> CUresult;

pub type PFN_cuMipmappedArrayDestroy = unsafe extern "C" fn(mipmap: CUmipmappedArray) -> CUresult;

pub type PFN_cuMipmappedArrayGetLevel = unsafe extern "C" fn(
    level_array: *mut crate::types::CUarray,
    mipmap: CUmipmappedArray,
    level: c_uint,
) -> CUresult;

// ---- Wave 11: pinned host memory ----------------------------------------

pub type PFN_cuMemAllocHost =
    unsafe extern "C" fn(pp: *mut *mut c_void, bytesize: usize) -> CUresult;

pub type PFN_cuMemFreeHost = unsafe extern "C" fn(p: *mut c_void) -> CUresult;

pub type PFN_cuMemHostAlloc =
    unsafe extern "C" fn(pp: *mut *mut c_void, bytesize: usize, flags: c_uint) -> CUresult;

pub type PFN_cuMemHostRegister =
    unsafe extern "C" fn(p: *mut c_void, bytesize: usize, flags: c_uint) -> CUresult;

pub type PFN_cuMemHostUnregister = unsafe extern "C" fn(p: *mut c_void) -> CUresult;

pub type PFN_cuMemHostGetDevicePointer =
    unsafe extern "C" fn(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: c_uint) -> CUresult;

pub type PFN_cuMemHostGetFlags =
    unsafe extern "C" fn(flags: *mut c_uint, p: *mut c_void) -> CUresult;

// ---- Wave 12: full graph node builders + edit ---------------------------

use crate::types::{CUDA_HOST_NODE_PARAMS, CUDA_KERNEL_NODE_PARAMS, CUDA_MEMSET_NODE_PARAMS};

pub type PFN_cuGraphAddMemcpyNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    copy_params: *const crate::types::CUDA_MEMCPY3D,
    ctx: CUcontext,
) -> CUresult;

pub type PFN_cuGraphAddHostNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    node_params: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphAddChildGraphNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    child_graph: CUgraph,
) -> CUresult;

pub type PFN_cuGraphAddEventRecordNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    event: CUevent,
) -> CUresult;

pub type PFN_cuGraphAddEventWaitNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    event: CUevent,
) -> CUresult;

pub type PFN_cuGraphAddExternalSemaphoresSignalNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    node_params: *const c_void,
) -> CUresult;

pub type PFN_cuGraphAddExternalSemaphoresWaitNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    node_params: *const c_void,
) -> CUresult;

pub type PFN_cuGraphKernelNodeGetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    params: *mut CUDA_KERNEL_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphKernelNodeSetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    params: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphMemcpyNodeGetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    params: *mut crate::types::CUDA_MEMCPY3D,
) -> CUresult;

pub type PFN_cuGraphMemcpyNodeSetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    params: *const crate::types::CUDA_MEMCPY3D,
) -> CUresult;

pub type PFN_cuGraphMemsetNodeGetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    params: *mut CUDA_MEMSET_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphMemsetNodeSetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    params: *const CUDA_MEMSET_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphNodeGetType =
    unsafe extern "C" fn(node: crate::types::CUgraphNode, type_: *mut c_int) -> CUresult;

pub type PFN_cuGraphNodeGetDependencies = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    dependencies: *mut crate::types::CUgraphNode,
    num_dependencies: *mut usize,
) -> CUresult;

pub type PFN_cuGraphNodeGetDependentNodes = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    dependent_nodes: *mut crate::types::CUgraphNode,
    num_dependent_nodes: *mut usize,
) -> CUresult;

pub type PFN_cuGraphGetEdges = unsafe extern "C" fn(
    graph: CUgraph,
    from: *mut crate::types::CUgraphNode,
    to: *mut crate::types::CUgraphNode,
    num_edges: *mut usize,
) -> CUresult;

pub type PFN_cuGraphAddDependencies = unsafe extern "C" fn(
    graph: CUgraph,
    from: *const crate::types::CUgraphNode,
    to: *const crate::types::CUgraphNode,
    num_dependencies: usize,
) -> CUresult;

pub type PFN_cuGraphRemoveDependencies = unsafe extern "C" fn(
    graph: CUgraph,
    from: *const crate::types::CUgraphNode,
    to: *const crate::types::CUgraphNode,
    num_dependencies: usize,
) -> CUresult;

pub type PFN_cuGraphExecKernelNodeSetParams = unsafe extern "C" fn(
    graph_exec: CUgraphExec,
    node: crate::types::CUgraphNode,
    params: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphExecMemcpyNodeSetParams = unsafe extern "C" fn(
    graph_exec: CUgraphExec,
    node: crate::types::CUgraphNode,
    params: *const crate::types::CUDA_MEMCPY3D,
    ctx: CUcontext,
) -> CUresult;

pub type PFN_cuGraphExecMemsetNodeSetParams = unsafe extern "C" fn(
    graph_exec: CUgraphExec,
    node: crate::types::CUgraphNode,
    params: *const CUDA_MEMSET_NODE_PARAMS,
    ctx: CUcontext,
) -> CUresult;

pub type PFN_cuGraphExecHostNodeSetParams = unsafe extern "C" fn(
    graph_exec: CUgraphExec,
    node: crate::types::CUgraphNode,
    params: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult;

// ---- Wave 13: stream extras ---------------------------------------------

pub type PFN_cuStreamGetId =
    unsafe extern "C" fn(stream: CUstream, stream_id: *mut u64) -> CUresult;

pub type PFN_cuStreamCopyAttributes =
    unsafe extern "C" fn(dst: CUstream, src: CUstream) -> CUresult;

/// `CUstreamAttrValue` is a union up to 48 bytes (access-policy window is
/// the largest member). Treat as opaque at this layer; the safe wrapper
/// exposes only the narrow cases that matter in practice.
pub type PFN_cuStreamGetAttribute =
    unsafe extern "C" fn(stream: CUstream, attr: c_int, value_out: *mut c_void) -> CUresult;

pub type PFN_cuStreamSetAttribute =
    unsafe extern "C" fn(stream: CUstream, attr: c_int, value: *const c_void) -> CUresult;

pub type PFN_cuStreamAttachMemAsync = unsafe extern "C" fn(
    stream: CUstream,
    dptr: CUdeviceptr,
    length: usize,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuStreamGetCaptureInfo = unsafe extern "C" fn(
    stream: CUstream,
    capture_status: *mut c_int,
    id: *mut u64,
    graph: *mut CUgraph,
    dependencies: *mut *const crate::types::CUgraphNode,
    num_dependencies: *mut usize,
) -> CUresult;

pub type PFN_cuStreamUpdateCaptureDependencies = unsafe extern "C" fn(
    stream: CUstream,
    dependencies: *mut crate::types::CUgraphNode,
    num_dependencies: usize,
    flags: c_uint,
) -> CUresult;

// ---- Wave 14: misc memcpy variants --------------------------------------

pub type PFN_cuMemcpyDtoDAsync = unsafe extern "C" fn(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuMemcpyPeer = unsafe extern "C" fn(
    dst: CUdeviceptr,
    dst_ctx: CUcontext,
    src: CUdeviceptr,
    src_ctx: CUcontext,
    bytes: usize,
) -> CUresult;

pub type PFN_cuMemcpyPeerAsync = unsafe extern "C" fn(
    dst: CUdeviceptr,
    dst_ctx: CUcontext,
    src: CUdeviceptr,
    src_ctx: CUcontext,
    bytes: usize,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuMemcpy =
    unsafe extern "C" fn(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CUresult;

pub type PFN_cuMemcpyAsync = unsafe extern "C" fn(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    bytes: usize,
    stream: CUstream,
) -> CUresult;

// Array 1-D copies (the index / offset is into the array's linear byte space).
pub type PFN_cuMemcpyAtoH = unsafe extern "C" fn(
    dst_host: *mut c_void,
    src_array: crate::types::CUarray,
    src_offset: usize,
    bytes: usize,
) -> CUresult;

pub type PFN_cuMemcpyHtoA = unsafe extern "C" fn(
    dst_array: crate::types::CUarray,
    dst_offset: usize,
    src_host: *const c_void,
    bytes: usize,
) -> CUresult;

pub type PFN_cuMemcpyAtoD = unsafe extern "C" fn(
    dst_device: CUdeviceptr,
    src_array: crate::types::CUarray,
    src_offset: usize,
    bytes: usize,
) -> CUresult;

pub type PFN_cuMemcpyDtoA = unsafe extern "C" fn(
    dst_array: crate::types::CUarray,
    dst_offset: usize,
    src_device: CUdeviceptr,
    bytes: usize,
) -> CUresult;

pub type PFN_cuMemcpyAtoA = unsafe extern "C" fn(
    dst_array: crate::types::CUarray,
    dst_offset: usize,
    src_array: crate::types::CUarray,
    src_offset: usize,
    bytes: usize,
) -> CUresult;

pub type PFN_cuMemsetD16 =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u16, count: usize) -> CUresult;

pub type PFN_cuMemsetD8Async =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u8, count: usize, stream: CUstream) -> CUresult;

pub type PFN_cuMemsetD16Async =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u16, count: usize, stream: CUstream) -> CUresult;

pub type PFN_cuMemsetD32Async =
    unsafe extern "C" fn(dst: CUdeviceptr, value: u32, count: usize, stream: CUstream) -> CUresult;

pub type PFN_cuMemsetD2D8 = unsafe extern "C" fn(
    dst: CUdeviceptr,
    pitch: usize,
    value: u8,
    width: usize,
    height: usize,
) -> CUresult;

pub type PFN_cuMemsetD2D16 = unsafe extern "C" fn(
    dst: CUdeviceptr,
    pitch: usize,
    value: u16,
    width: usize,
    height: usize,
) -> CUresult;

pub type PFN_cuMemsetD2D32 = unsafe extern "C" fn(
    dst: CUdeviceptr,
    pitch: usize,
    value: u32,
    width: usize,
    height: usize,
) -> CUresult;

// ---- Wave 15: range + pointer attrs batch -------------------------------

pub type PFN_cuMemRangeGetAttribute = unsafe extern "C" fn(
    data: *mut c_void,
    data_size: usize,
    attribute: c_int,
    devptr: CUdeviceptr,
    count: usize,
) -> CUresult;

pub type PFN_cuMemRangeGetAttributes = unsafe extern "C" fn(
    data: *mut *mut c_void,
    data_sizes: *mut usize,
    attributes: *mut c_int,
    num_attributes: usize,
    devptr: CUdeviceptr,
    count: usize,
) -> CUresult;

pub type PFN_cuPointerGetAttributes = unsafe extern "C" fn(
    num_attributes: c_uint,
    attributes: *mut c_int,
    data: *mut *mut c_void,
    ptr: CUdeviceptr,
) -> CUresult;

pub type PFN_cuPointerSetAttribute =
    unsafe extern "C" fn(value: *const c_void, attribute: c_int, ptr: CUdeviceptr) -> CUresult;

// ---- Wave 16: tensor maps (Hopper TMA) ----------------------------------

use crate::types::CUtensorMap;

pub type PFN_cuTensorMapEncodeTiled = unsafe extern "C" fn(
    tensor_map: *mut CUtensorMap,
    tensor_data_type: c_int,
    tensor_rank: c_uint,
    global_address: *mut c_void,
    global_dim: *const u64,
    global_strides: *const u64,
    box_dim: *const u32,
    element_strides: *const u32,
    interleave: c_int,
    swizzle: c_int,
    l2_promotion: c_int,
    oob_fill: c_int,
) -> CUresult;

pub type PFN_cuTensorMapEncodeIm2col = unsafe extern "C" fn(
    tensor_map: *mut CUtensorMap,
    tensor_data_type: c_int,
    tensor_rank: c_uint,
    global_address: *mut c_void,
    global_dim: *const u64,
    global_strides: *const u64,
    pixel_box_lower_corner: *const c_int,
    pixel_box_upper_corner: *const c_int,
    channels_per_pixel: c_uint,
    pixels_per_column: c_uint,
    element_strides: *const u32,
    interleave: c_int,
    swizzle: c_int,
    l2_promotion: c_int,
    oob_fill: c_int,
) -> CUresult;

pub type PFN_cuTensorMapReplaceAddress =
    unsafe extern "C" fn(tensor_map: *mut CUtensorMap, global_address: *mut c_void) -> CUresult;

// ---- Wave 17: green contexts (CUDA 12.4+) -------------------------------

use crate::types::{CUdevResource, CUdevResourceDesc, CUgreenCtx};

pub type PFN_cuDeviceGetDevResource =
    unsafe extern "C" fn(device: CUdevice, resource: *mut CUdevResource, type_: c_int) -> CUresult;

pub type PFN_cuDevSmResourceSplitByCount = unsafe extern "C" fn(
    result: *mut CUdevResource,
    nb_groups: *mut c_uint,
    input: *const CUdevResource,
    remaining: *mut CUdevResource,
    use_flags: c_uint,
    min_count: c_uint,
) -> CUresult;

pub type PFN_cuDevResourceGenerateDesc = unsafe extern "C" fn(
    desc_out: *mut CUdevResourceDesc,
    resources: *mut CUdevResource,
    nb_resources: c_uint,
) -> CUresult;

pub type PFN_cuGreenCtxCreate = unsafe extern "C" fn(
    green_ctx: *mut CUgreenCtx,
    desc: CUdevResourceDesc,
    dev: CUdevice,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuGreenCtxDestroy = unsafe extern "C" fn(green_ctx: CUgreenCtx) -> CUresult;

pub type PFN_cuCtxFromGreenCtx =
    unsafe extern "C" fn(out_ctx: *mut CUcontext, green_ctx: CUgreenCtx) -> CUresult;

pub type PFN_cuGreenCtxGetDevResource = unsafe extern "C" fn(
    green_ctx: CUgreenCtx,
    resource: *mut CUdevResource,
    type_: c_int,
) -> CUresult;

pub type PFN_cuGreenCtxStreamCreate = unsafe extern "C" fn(
    stream_out: *mut CUstream,
    green_ctx: CUgreenCtx,
    flags: c_uint,
    priority: c_int,
) -> CUresult;

// ---- Wave 18: multicast objects -----------------------------------------

use crate::types::CUmulticastObjectProp;

pub type PFN_cuMulticastCreate = unsafe extern "C" fn(
    mc_handle: *mut CUmemGenericAllocationHandle,
    prop: *const CUmulticastObjectProp,
) -> CUresult;

pub type PFN_cuMulticastAddDevice =
    unsafe extern "C" fn(mc_handle: CUmemGenericAllocationHandle, dev: CUdevice) -> CUresult;

pub type PFN_cuMulticastBindMem = unsafe extern "C" fn(
    mc_handle: CUmemGenericAllocationHandle,
    mc_offset: usize,
    mem_handle: CUmemGenericAllocationHandle,
    mem_offset: usize,
    size: usize,
    flags: u64,
) -> CUresult;

pub type PFN_cuMulticastBindAddr = unsafe extern "C" fn(
    mc_handle: CUmemGenericAllocationHandle,
    mc_offset: usize,
    memptr: CUdeviceptr,
    size: usize,
    flags: u64,
) -> CUresult;

pub type PFN_cuMulticastUnbind = unsafe extern "C" fn(
    mc_handle: CUmemGenericAllocationHandle,
    dev: CUdevice,
    mc_offset: usize,
    size: usize,
) -> CUresult;

pub type PFN_cuMulticastGetGranularity = unsafe extern "C" fn(
    granularity: *mut usize,
    prop: *const CUmulticastObjectProp,
    option: c_int,
) -> CUresult;

// ---- Wave 19: conditional + switch graph nodes --------------------------

use crate::types::{CUgraphConditionalHandle, CUgraphEdgeData, CUgraphNodeParams};

pub type PFN_cuGraphAddNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    dependency_data: *const CUgraphEdgeData,
    num_dependencies: usize,
    node_params: *mut CUgraphNodeParams,
) -> CUresult;

pub type PFN_cuGraphNodeSetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    node_params: *mut CUgraphNodeParams,
) -> CUresult;

pub type PFN_cuGraphConditionalHandleCreate = unsafe extern "C" fn(
    handle_out: *mut CUgraphConditionalHandle,
    graph: CUgraph,
    ctx: CUcontext,
    default_launch_value: c_uint,
    flags: c_uint,
) -> CUresult;

// ---- Wave 20: IPC -------------------------------------------------------

use crate::types::{CUipcEventHandle, CUipcMemHandle};

pub type PFN_cuIpcGetEventHandle =
    unsafe extern "C" fn(handle_out: *mut CUipcEventHandle, event: CUevent) -> CUresult;

pub type PFN_cuIpcOpenEventHandle =
    unsafe extern "C" fn(event_out: *mut CUevent, handle: CUipcEventHandle) -> CUresult;

pub type PFN_cuIpcGetMemHandle =
    unsafe extern "C" fn(handle_out: *mut CUipcMemHandle, dptr: CUdeviceptr) -> CUresult;

pub type PFN_cuIpcOpenMemHandle = unsafe extern "C" fn(
    dptr_out: *mut CUdeviceptr,
    handle: CUipcMemHandle,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuIpcCloseMemHandle = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;

// ---- Wave 21: kernel attrs extension (CUDA 12+) -------------------------

pub type PFN_cuKernelGetAttribute =
    unsafe extern "C" fn(pi: *mut c_int, attr: c_int, kernel: CUkernel, dev: CUdevice) -> CUresult;

pub type PFN_cuKernelSetAttribute =
    unsafe extern "C" fn(attr: c_int, val: c_int, kernel: CUkernel, dev: CUdevice) -> CUresult;

pub type PFN_cuKernelGetName =
    unsafe extern "C" fn(name: *mut *const c_char, kernel: CUkernel) -> CUresult;

pub type PFN_cuKernelSetCacheConfig =
    unsafe extern "C" fn(kernel: CUkernel, config: c_int, dev: CUdevice) -> CUresult;

pub type PFN_cuKernelGetLibrary =
    unsafe extern "C" fn(library_out: *mut CUlibrary, kernel: CUkernel) -> CUresult;

pub type PFN_cuKernelGetParamInfo = unsafe extern "C" fn(
    kernel: CUkernel,
    param_index: usize,
    param_offset: *mut usize,
    param_size: *mut usize,
) -> CUresult;

// ---- Wave 22: user objects ----------------------------------------------

use crate::types::CUuserObject;

pub type PFN_cuUserObjectCreate = unsafe extern "C" fn(
    object_out: *mut CUuserObject,
    ptr: *mut c_void,
    destroy: CUhostFn,
    initial_refcount: c_uint,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuUserObjectRetain =
    unsafe extern "C" fn(object: CUuserObject, count: c_uint) -> CUresult;

pub type PFN_cuUserObjectRelease =
    unsafe extern "C" fn(object: CUuserObject, count: c_uint) -> CUresult;

pub type PFN_cuGraphRetainUserObject = unsafe extern "C" fn(
    graph: CUgraph,
    object: CUuserObject,
    count: c_uint,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuGraphReleaseUserObject =
    unsafe extern "C" fn(graph: CUgraph, object: CUuserObject, count: c_uint) -> CUresult;

// ---- Wave 23: misc extras -----------------------------------------------

use crate::types::{CUlogIterator, CUlogsCallback, CUlogsCallbackHandle};

pub type PFN_cuProfilerStart = unsafe extern "C" fn() -> CUresult;
pub type PFN_cuProfilerStop = unsafe extern "C" fn() -> CUresult;

pub type PFN_cuFuncGetModule =
    unsafe extern "C" fn(module_out: *mut CUmodule, func: CUfunction) -> CUresult;

pub type PFN_cuFuncGetName =
    unsafe extern "C" fn(name: *mut *const c_char, func: CUfunction) -> CUresult;

pub type PFN_cuFuncGetParamInfo = unsafe extern "C" fn(
    func: CUfunction,
    param_index: usize,
    param_offset: *mut usize,
    param_size: *mut usize,
) -> CUresult;

pub type PFN_cuGraphDebugDotPrint =
    unsafe extern "C" fn(graph: CUgraph, path: *const c_char, flags: c_uint) -> CUresult;

pub type PFN_cuCtxGetId = unsafe extern "C" fn(ctx: CUcontext, ctx_id: *mut u64) -> CUresult;

pub type PFN_cuModuleGetLoadingMode = unsafe extern "C" fn(mode: *mut c_int) -> CUresult;

pub type PFN_cuDeviceGetUuid = unsafe extern "C" fn(uuid: *mut u8, dev: CUdevice) -> CUresult;

pub type PFN_cuDeviceGetLuid = unsafe extern "C" fn(
    luid: *mut c_char,
    device_node_mask: *mut c_uint,
    dev: CUdevice,
) -> CUresult;

pub type PFN_cuLogsRegisterCallback = unsafe extern "C" fn(
    callback: CUlogsCallback,
    user_data: *mut c_void,
    handle_out: *mut CUlogsCallbackHandle,
) -> CUresult;

pub type PFN_cuLogsUnregisterCallback =
    unsafe extern "C" fn(handle: CUlogsCallbackHandle) -> CUresult;

pub type PFN_cuLogsCurrent =
    unsafe extern "C" fn(iterator_out: *mut CUlogIterator, flags: c_uint) -> CUresult;

pub type PFN_cuLogsDumpToFile = unsafe extern "C" fn(
    iterator: *mut CUlogIterator,
    path: *const c_char,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuLogsDumpToMemory = unsafe extern "C" fn(
    iterator: *mut CUlogIterator,
    buffer: *mut c_char,
    size: *mut usize,
    flags: c_uint,
) -> CUresult;

// ---- Wave 24: graph memory nodes + graph-exec update --------------------

use crate::types::{
    CUgraphExecUpdateResultInfo, CUDA_BATCH_MEM_OP_NODE_PARAMS, CUDA_MEM_ALLOC_NODE_PARAMS,
};

pub type PFN_cuGraphAddMemAllocNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    node_params: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphMemAllocNodeGetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphAddMemFreeNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    dptr: CUdeviceptr,
) -> CUresult;

pub type PFN_cuGraphMemFreeNodeGetParams =
    unsafe extern "C" fn(node: crate::types::CUgraphNode, dptr_out: *mut CUdeviceptr) -> CUresult;

pub type PFN_cuDeviceGraphMemTrim = unsafe extern "C" fn(dev: CUdevice) -> CUresult;

pub type PFN_cuDeviceGetGraphMemAttribute =
    unsafe extern "C" fn(dev: CUdevice, attr: c_int, value: *mut c_void) -> CUresult;

pub type PFN_cuDeviceSetGraphMemAttribute =
    unsafe extern "C" fn(dev: CUdevice, attr: c_int, value: *mut c_void) -> CUresult;

pub type PFN_cuGraphAddBatchMemOpNode = unsafe extern "C" fn(
    graph_node: *mut crate::types::CUgraphNode,
    graph: CUgraph,
    dependencies: *const crate::types::CUgraphNode,
    num_dependencies: usize,
    node_params: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphBatchMemOpNodeGetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    node_params_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphBatchMemOpNodeSetParams = unsafe extern "C" fn(
    node: crate::types::CUgraphNode,
    node_params: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphExecBatchMemOpNodeSetParams = unsafe extern "C" fn(
    graph_exec: CUgraphExec,
    node: crate::types::CUgraphNode,
    node_params: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult;

pub type PFN_cuGraphExecUpdate = unsafe extern "C" fn(
    graph_exec: CUgraphExec,
    graph: CUgraph,
    result_info: *mut CUgraphExecUpdateResultInfo,
) -> CUresult;

// ---- Wave 25: stream memory ops -----------------------------------------

use crate::types::CUstreamBatchMemOpParams;

pub type PFN_cuStreamWriteValue32 = unsafe extern "C" fn(
    stream: CUstream,
    addr: CUdeviceptr,
    value: u32,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuStreamWriteValue64 = unsafe extern "C" fn(
    stream: CUstream,
    addr: CUdeviceptr,
    value: u64,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuStreamWaitValue32 = unsafe extern "C" fn(
    stream: CUstream,
    addr: CUdeviceptr,
    value: u32,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuStreamWaitValue64 = unsafe extern "C" fn(
    stream: CUstream,
    addr: CUdeviceptr,
    value: u64,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuStreamBatchMemOp = unsafe extern "C" fn(
    stream: CUstream,
    count: c_uint,
    param_array: *mut CUstreamBatchMemOpParams,
    flags: c_uint,
) -> CUresult;

// ---- Wave 27: v2 advise/prefetch + VMM reverse lookups ------------------

use crate::types::CUmemLocation;

/// `cuMemPrefetchAsync_v2` — takes a [`CUmemLocation`] (so you can
/// prefetch to any host / NUMA / device destination with one entry point).
pub type PFN_cuMemPrefetchAsyncV2 = unsafe extern "C" fn(
    dev_ptr: CUdeviceptr,
    count: usize,
    location: CUmemLocation,
    flags: c_uint,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuMemAdviseV2 = unsafe extern "C" fn(
    dev_ptr: CUdeviceptr,
    count: usize,
    advice: c_int,
    location: CUmemLocation,
) -> CUresult;

/// `cuMemMapArrayAsync` — bulk map / unmap sparse tiles of arrays or
/// mipmapped arrays into VMM-backed allocations, ordered on `stream`.
/// See [`crate::types::CUarrayMapInfo`] for the entry shape.
pub type PFN_cuMemMapArrayAsync = unsafe extern "C" fn(
    map_info_list: *mut crate::types::CUarrayMapInfo,
    count: c_uint,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuMemGetHandleForAddressRange = unsafe extern "C" fn(
    handle: *mut c_void,
    dptr: CUdeviceptr,
    size: usize,
    handle_type: c_int,
    flags: u64,
) -> CUresult;

pub type PFN_cuMemRetainAllocationHandle =
    unsafe extern "C" fn(handle: *mut CUmemGenericAllocationHandle, addr: *mut c_void) -> CUresult;

pub type PFN_cuMemGetAllocationPropertiesFromHandle = unsafe extern "C" fn(
    prop: *mut CUmemAllocationProp,
    handle: CUmemGenericAllocationHandle,
) -> CUresult;

pub type PFN_cuMemExportToShareableHandle = unsafe extern "C" fn(
    shareable_handle: *mut c_void,
    handle: CUmemGenericAllocationHandle,
    handle_type: c_int,
    flags: u64,
) -> CUresult;

pub type PFN_cuMemImportFromShareableHandle = unsafe extern "C" fn(
    handle: *mut CUmemGenericAllocationHandle,
    os_handle: *mut c_void,
    sh_handle_type: c_int,
) -> CUresult;

pub type PFN_cuMemGetAccess = unsafe extern "C" fn(
    flags: *mut u64,
    location: *const CUmemLocation,
    ptr: CUdeviceptr,
) -> CUresult;

// ---- Wave 28: medium-value consolidated ---------------------------------

use crate::types::{CUDA_ARRAY_MEMORY_REQUIREMENTS, CUDA_ARRAY_SPARSE_PROPERTIES};

// Array introspection
pub type PFN_cuArrayGetDescriptor = unsafe extern "C" fn(
    desc: *mut crate::types::CUDA_ARRAY_DESCRIPTOR,
    array: crate::types::CUarray,
) -> CUresult;

pub type PFN_cuArrayGetSparseProperties = unsafe extern "C" fn(
    sparse_properties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
    array: crate::types::CUarray,
) -> CUresult;

pub type PFN_cuMipmappedArrayGetSparseProperties = unsafe extern "C" fn(
    sparse_properties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
    mipmap: crate::types::CUmipmappedArray,
) -> CUresult;

pub type PFN_cuArrayGetMemoryRequirements = unsafe extern "C" fn(
    mem_requirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
    array: crate::types::CUarray,
    device: CUdevice,
) -> CUresult;

pub type PFN_cuMipmappedArrayGetMemoryRequirements = unsafe extern "C" fn(
    mem_requirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
    mipmap: crate::types::CUmipmappedArray,
    device: CUdevice,
) -> CUresult;

pub type PFN_cuArrayGetPlane = unsafe extern "C" fn(
    plane_array_out: *mut crate::types::CUarray,
    array: crate::types::CUarray,
    plane_idx: c_uint,
) -> CUresult;

// Context-level events
pub type PFN_cuCtxRecordEvent = unsafe extern "C" fn(ctx: CUcontext, event: CUevent) -> CUresult;
pub type PFN_cuCtxWaitEvent = unsafe extern "C" fn(ctx: CUcontext, event: CUevent) -> CUresult;

// P2P + exec affinity
pub type PFN_cuDeviceGetP2PAttribute = unsafe extern "C" fn(
    value: *mut c_int,
    attr: c_int,
    src_device: CUdevice,
    dst_device: CUdevice,
) -> CUresult;

pub type PFN_cuDeviceGetExecAffinitySupport =
    unsafe extern "C" fn(pi: *mut c_int, type_: c_int, dev: CUdevice) -> CUresult;

// RDMA flush
pub type PFN_cuFlushGPUDirectRDMAWrites =
    unsafe extern "C" fn(target: c_int, scope: c_int) -> CUresult;

// Core dump
pub type PFN_cuCoredumpGetAttribute =
    unsafe extern "C" fn(attr: c_int, value: *mut c_void, size: *mut usize) -> CUresult;

pub type PFN_cuCoredumpGetAttributeGlobal =
    unsafe extern "C" fn(attr: c_int, value: *mut c_void, size: *mut usize) -> CUresult;

pub type PFN_cuCoredumpSetAttribute =
    unsafe extern "C" fn(attr: c_int, value: *mut c_void, size: *mut usize) -> CUresult;

pub type PFN_cuCoredumpSetAttributeGlobal =
    unsafe extern "C" fn(attr: c_int, value: *mut c_void, size: *mut usize) -> CUresult;

// Library extras
pub type PFN_cuLibraryGetUnifiedFunction = unsafe extern "C" fn(
    fptr: *mut *mut c_void,
    library: CUlibrary,
    symbol: *const c_char,
) -> CUresult;

pub type PFN_cuLibraryGetModule =
    unsafe extern "C" fn(module_out: *mut CUmodule, library: CUlibrary) -> CUresult;

pub type PFN_cuLibraryGetKernelCount =
    unsafe extern "C" fn(count: *mut c_uint, lib: CUlibrary) -> CUresult;

pub type PFN_cuLibraryEnumerateKernels =
    unsafe extern "C" fn(kernels: *mut CUkernel, num_kernels: c_uint, lib: CUlibrary) -> CUresult;

pub type PFN_cuLibraryGetManaged = unsafe extern "C" fn(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    library: CUlibrary,
    name: *const c_char,
) -> CUresult;

// ---- Wave 29: graphics interop core + OpenGL ----------------------------

use crate::types::{CUgraphicsResource, GLenum, GLuint};

pub type PFN_cuGraphicsUnregisterResource =
    unsafe extern "C" fn(resource: CUgraphicsResource) -> CUresult;

pub type PFN_cuGraphicsMapResources = unsafe extern "C" fn(
    count: c_uint,
    resources: *mut CUgraphicsResource,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuGraphicsUnmapResources = unsafe extern "C" fn(
    count: c_uint,
    resources: *mut CUgraphicsResource,
    stream: CUstream,
) -> CUresult;

pub type PFN_cuGraphicsResourceGetMappedPointer = unsafe extern "C" fn(
    dev_ptr: *mut CUdeviceptr,
    size: *mut usize,
    resource: CUgraphicsResource,
) -> CUresult;

pub type PFN_cuGraphicsResourceGetMappedMipmappedArray = unsafe extern "C" fn(
    mipmap_array: *mut crate::types::CUmipmappedArray,
    resource: CUgraphicsResource,
) -> CUresult;

pub type PFN_cuGraphicsSubResourceGetMappedArray = unsafe extern "C" fn(
    array: *mut crate::types::CUarray,
    resource: CUgraphicsResource,
    array_index: c_uint,
    mip_level: c_uint,
) -> CUresult;

pub type PFN_cuGraphicsResourceSetMapFlags =
    unsafe extern "C" fn(resource: CUgraphicsResource, flags: c_uint) -> CUresult;

// OpenGL-specific

pub type PFN_cuGLGetDevices = unsafe extern "C" fn(
    cuda_device_count: *mut c_uint,
    cuda_devices: *mut CUdevice,
    cuda_device_count_in: c_uint,
    device_list: c_uint,
) -> CUresult;

pub type PFN_cuGraphicsGLRegisterBuffer = unsafe extern "C" fn(
    resource: *mut CUgraphicsResource,
    buffer: GLuint,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuGraphicsGLRegisterImage = unsafe extern "C" fn(
    resource: *mut CUgraphicsResource,
    image: GLuint,
    target: GLenum,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuGLCtxCreate =
    unsafe extern "C" fn(ctx: *mut CUcontext, flags: c_uint, device: CUdevice) -> CUresult;

pub type PFN_cuGLInit = unsafe extern "C" fn() -> CUresult;

// ---- Wave 30: Direct3D 9 / 10 / 11 interop -------------------------------

use crate::types::{ID3DDevice, ID3DResource};

// Per-API GetDevice / GetDevices / RegisterResource each share a shape.
// D3D9 uses `IDirect3DDevice9*`, D3D10 uses `ID3D10Device*`, D3D11 uses
// `ID3D11Device*` — all are opaque pointers.

pub type PFN_cuD3D9GetDevice =
    unsafe extern "C" fn(cuda_device: *mut CUdevice, adapter_name: *const c_char) -> CUresult;

pub type PFN_cuD3D9GetDevices = unsafe extern "C" fn(
    cuda_device_count: *mut c_uint,
    cuda_devices: *mut CUdevice,
    cuda_device_count_in: c_uint,
    d3d_device: ID3DDevice,
    device_list: c_uint,
) -> CUresult;

pub type PFN_cuGraphicsD3D9RegisterResource = unsafe extern "C" fn(
    resource: *mut CUgraphicsResource,
    d3d_resource: ID3DResource,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuD3D10GetDevice =
    unsafe extern "C" fn(cuda_device: *mut CUdevice, adapter: ID3DDevice) -> CUresult;

pub type PFN_cuD3D10GetDevices = unsafe extern "C" fn(
    cuda_device_count: *mut c_uint,
    cuda_devices: *mut CUdevice,
    cuda_device_count_in: c_uint,
    d3d_device: ID3DDevice,
    device_list: c_uint,
) -> CUresult;

pub type PFN_cuGraphicsD3D10RegisterResource = unsafe extern "C" fn(
    resource: *mut CUgraphicsResource,
    d3d_resource: ID3DResource,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuD3D11GetDevice =
    unsafe extern "C" fn(cuda_device: *mut CUdevice, adapter: ID3DDevice) -> CUresult;

pub type PFN_cuD3D11GetDevices = unsafe extern "C" fn(
    cuda_device_count: *mut c_uint,
    cuda_devices: *mut CUdevice,
    cuda_device_count_in: c_uint,
    d3d_device: ID3DDevice,
    device_list: c_uint,
) -> CUresult;

pub type PFN_cuGraphicsD3D11RegisterResource = unsafe extern "C" fn(
    resource: *mut CUgraphicsResource,
    d3d_resource: ID3DResource,
    flags: c_uint,
) -> CUresult;

// ---- Wave 31: VDPAU + EGL + NvSci (Jetson / automotive) -----------------

use crate::types::{
    CUeglFrame, EGLImageKHR, EGLStreamKHR, EGLSyncKHR, NvSciSyncAttrList, VdpDevice,
    VdpGetProcAddress, VdpOutputSurface, VdpVideoSurface,
};

// VDPAU
pub type PFN_cuVDPAUGetDevice = unsafe extern "C" fn(
    device: *mut CUdevice,
    vdp_device: VdpDevice,
    vdp_get_proc_address: VdpGetProcAddress,
) -> CUresult;

pub type PFN_cuVDPAUCtxCreate = unsafe extern "C" fn(
    ctx: *mut CUcontext,
    flags: c_uint,
    device: CUdevice,
    vdp_device: VdpDevice,
    vdp_get_proc_address: VdpGetProcAddress,
) -> CUresult;

pub type PFN_cuGraphicsVDPAURegisterVideoSurface = unsafe extern "C" fn(
    resource: *mut CUgraphicsResource,
    vdp_surface: VdpVideoSurface,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuGraphicsVDPAURegisterOutputSurface = unsafe extern "C" fn(
    resource: *mut CUgraphicsResource,
    vdp_surface: VdpOutputSurface,
    flags: c_uint,
) -> CUresult;

// EGL
pub type PFN_cuGraphicsEGLRegisterImage = unsafe extern "C" fn(
    resource: *mut CUgraphicsResource,
    image: EGLImageKHR,
    flags: c_uint,
) -> CUresult;

pub type PFN_cuGraphicsResourceGetMappedEglFrame = unsafe extern "C" fn(
    egl_frame: *mut CUeglFrame,
    resource: CUgraphicsResource,
    index: c_uint,
    mip_level: c_uint,
) -> CUresult;

pub type PFN_cuEventCreateFromEGLSync =
    unsafe extern "C" fn(event_out: *mut CUevent, egl_sync: EGLSyncKHR, flags: c_uint) -> CUresult;

pub type PFN_cuEGLStreamConsumerConnect =
    unsafe extern "C" fn(connection: *mut c_void, stream: EGLStreamKHR) -> CUresult;

pub type PFN_cuEGLStreamConsumerDisconnect =
    unsafe extern "C" fn(connection: *mut c_void) -> CUresult;

pub type PFN_cuEGLStreamConsumerAcquireFrame = unsafe extern "C" fn(
    connection: *mut c_void,
    resource: *mut CUgraphicsResource,
    stream: *mut CUstream,
    timeout: c_uint,
) -> CUresult;

pub type PFN_cuEGLStreamConsumerReleaseFrame = unsafe extern "C" fn(
    connection: *mut c_void,
    resource: CUgraphicsResource,
    stream: *mut CUstream,
) -> CUresult;

pub type PFN_cuEGLStreamProducerConnect = unsafe extern "C" fn(
    connection: *mut c_void,
    stream: EGLStreamKHR,
    width: core::ffi::c_int,
    height: core::ffi::c_int,
) -> CUresult;

pub type PFN_cuEGLStreamProducerDisconnect =
    unsafe extern "C" fn(connection: *mut c_void) -> CUresult;

pub type PFN_cuEGLStreamProducerPresentFrame = unsafe extern "C" fn(
    connection: *mut c_void,
    egl_frame: CUeglFrame,
    stream: *mut CUstream,
) -> CUresult;

pub type PFN_cuEGLStreamProducerReturnFrame = unsafe extern "C" fn(
    connection: *mut c_void,
    egl_frame: *mut CUeglFrame,
    stream: *mut CUstream,
) -> CUresult;

// NvSci (Jetson / DRIVE)
pub type PFN_cuDeviceGetNvSciSyncAttributes = unsafe extern "C" fn(
    nv_sci_sync_attr_list: NvSciSyncAttrList,
    dev: CUdevice,
    flags: c_int,
) -> CUresult;
