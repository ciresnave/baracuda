//! Raw FFI + dynamic loader for NVIDIA NCCL (multi-GPU collective communication).
//!
//! `baracuda-nccl` wraps this with a safe, typed API. Use this crate
//! directly only if you need a function that the safe layer hasn't
//! wrapped yet (in which case please file a bug).
//!
//! NCCL is primarily a Linux library; Windows support landed in later NCCL
//! versions but is uncommon. This crate compiles everywhere and defers the
//! "is NCCL actually installed?" question to runtime — [`nccl()`] returns
//! `LoaderError::LibraryNotFound` on hosts without NCCL.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

/// Opaque NCCL communicator.
pub type ncclComm_t = *mut c_void;

/// A 128-byte unique identifier for multi-process NCCL initialization.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ncclUniqueId {
    /// Internal field.
    pub internal: [i8; 128],
}

impl core::fmt::Debug for ncclUniqueId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ncclUniqueId").finish_non_exhaustive()
    }
}

impl Default for ncclUniqueId {
    fn default() -> Self {
        Self { internal: [0; 128] }
    }
}

/// NCCL element data type.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ncclDataType_t {
    /// 8-bit signed integer element.
    Int8 = 0,
    /// 8-bit unsigned integer element.
    Uint8 = 1,
    /// 32-bit signed integer element.
    Int32 = 2,
    /// 32-bit unsigned integer element.
    Uint32 = 3,
    /// 64-bit signed integer element.
    Int64 = 4,
    /// 64-bit unsigned integer element.
    Uint64 = 5,
    /// IEEE-754 binary16 (fp16) element.
    Float16 = 6,
    /// IEEE-754 binary32 (fp32) element.
    Float32 = 7,
    /// IEEE-754 binary64 (fp64) element.
    Float64 = 8,
    /// bfloat16 element.
    BFloat16 = 9,
}

/// NCCL reduction operation. Modeled as a transparent newtype rather
/// than a closed enum because [`PFN_ncclRedOpCreatePreMulSum`] returns
/// custom op IDs (≥ 5) that don't fit a closed Rust enum.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub struct ncclRedOp_t(pub i32);

#[allow(non_upper_case_globals)]
impl ncclRedOp_t {
    /// `ncclSum` — element-wise sum reduction.
    pub const Sum: Self = Self(0);
    /// `ncclProd` — element-wise product reduction.
    pub const Prod: Self = Self(1);
    /// `ncclMax` — element-wise max reduction.
    pub const Max: Self = Self(2);
    /// `ncclMin` — element-wise min reduction.
    pub const Min: Self = Self(3);
    /// `ncclAvg` — element-wise average reduction (NCCL 2.10+).
    pub const Avg: Self = Self(4);
}

// ---- status ---------------------------------------------------------------

/// Return code from an NCCL call.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct ncclResult_t(pub i32);

impl ncclResult_t {
    /// `ncclSuccess` — operation succeeded.
    pub const Success: Self = Self(0);
    /// `ncclUnhandledCudaError` — an underlying CUDA call failed.
    pub const UnhandledCudaError: Self = Self(1);
    /// `ncclSystemError` — a system-level error occurred (sockets, files, ...).
    pub const SystemError: Self = Self(2);
    /// `ncclInternalError` — an internal NCCL error occurred.
    pub const InternalError: Self = Self(3);
    /// `ncclInvalidArgument` — an argument was invalid.
    pub const InvalidArgument: Self = Self(4);
    /// `ncclInvalidUsage` — the call is invalid in the current state.
    pub const InvalidUsage: Self = Self(5);
    /// `ncclRemoteError` — another rank in the communicator failed.
    pub const RemoteError: Self = Self(6);
    /// `ncclInProgress` — non-blocking operation still in progress.
    pub const InProgress: Self = Self(7);

    /// Return `true` if the status code denotes success.
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for ncclResult_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "ncclSuccess",
            1 => "ncclUnhandledCudaError",
            2 => "ncclSystemError",
            3 => "ncclInternalError",
            4 => "ncclInvalidArgument",
            5 => "ncclInvalidUsage",
            6 => "ncclRemoteError",
            7 => "ncclInProgress",
            _ => "ncclUnrecognizedResult",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "unhandled CUDA error",
            2 => "system error",
            3 => "internal NCCL error",
            4 => "invalid argument",
            5 => "invalid usage",
            6 => "remote error (another rank failed)",
            7 => "operation in progress (non-blocking comm)",
            _ => "unrecognized NCCL status code",
        }
    }
    fn is_success(self) -> bool {
        ncclResult_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "nccl"
    }
}

// ---- function-pointer types ----------------------------------------------

/// Function-pointer type for `ncclGetVersion` (query NCCL library version). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclGetVersion = unsafe extern "C" fn(version: *mut c_int) -> ncclResult_t;
/// Function-pointer type for `ncclGetUniqueId` (generate a unique multi-rank initialization ID). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclGetUniqueId = unsafe extern "C" fn(id: *mut ncclUniqueId) -> ncclResult_t;
/// Function-pointer type for `ncclCommInitRank` (initialize a communicator rank). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommInitRank = unsafe extern "C" fn(
    comm: *mut ncclComm_t,
    nranks: c_int,
    comm_id: ncclUniqueId,
    rank: c_int,
) -> ncclResult_t;
/// Function-pointer type for `ncclCommInitAll` (initialize all-local-GPU communicators in one call). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommInitAll = unsafe extern "C" fn(
    comms: *mut ncclComm_t,
    ndev: c_int,
    dev_list: *const c_int,
) -> ncclResult_t;
/// Function-pointer type for `ncclCommDestroy` (destroy a communicator). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommDestroy = unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t;
/// Function-pointer type for `ncclCommCount` (query rank count on a communicator). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommCount =
    unsafe extern "C" fn(comm: ncclComm_t, count: *mut c_int) -> ncclResult_t;
/// Function-pointer type for `ncclCommUserRank` (query this rank's index on a communicator). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommUserRank =
    unsafe extern "C" fn(comm: ncclComm_t, rank: *mut c_int) -> ncclResult_t;

/// Function-pointer type for `ncclAllReduce` (all-reduce collective). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclAllReduce = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

/// Function-pointer type for `ncclBroadcast` (broadcast-from-root collective). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclBroadcast = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

/// Function-pointer type for `ncclGroupStart` (start grouped collective ops). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclGroupStart = unsafe extern "C" fn() -> ncclResult_t;
/// Function-pointer type for `ncclGroupEnd` (end grouped collective ops and commit them). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclGroupEnd = unsafe extern "C" fn() -> ncclResult_t;

// ---- Full collective surface ----

/// Function-pointer type for `ncclReduce` (reduce-to-root collective). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclReduce = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

/// Function-pointer type for `ncclAllGather` (all-gather collective). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclAllGather = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: usize,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

/// Function-pointer type for `ncclReduceScatter` (reduce-scatter collective). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclReduceScatter = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

/// Function-pointer type for `ncclSend` (point-to-point send). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclSend = unsafe extern "C" fn(
    sendbuff: *const c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

/// Function-pointer type for `ncclRecv` (point-to-point receive). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclRecv = unsafe extern "C" fn(
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

// ---- Communicator lifecycle extras ----

/// Function-pointer type for `ncclCommAbort` (abort outstanding ops on a communicator). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommAbort = unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t;
/// Function-pointer type for `ncclCommFinalize` (finalize a non-blocking communicator). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommFinalize = unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t;
/// Function-pointer type for `ncclCommGetAsyncError` (fetch a communicator's last async error). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommGetAsyncError =
    unsafe extern "C" fn(comm: ncclComm_t, async_error: *mut ncclResult_t) -> ncclResult_t;
/// Function-pointer type for `ncclCommCuDevice` (query CUDA device backing a communicator rank). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommCuDevice =
    unsafe extern "C" fn(comm: ncclComm_t, device: *mut c_int) -> ncclResult_t;
/// Function-pointer type for `ncclCommSplit` (split a communicator by color/key). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommSplit = unsafe extern "C" fn(
    comm: ncclComm_t,
    color: c_int,
    key: c_int,
    new_comm: *mut ncclComm_t,
    config: *mut c_void, // ncclConfig_t
) -> ncclResult_t;

/// Function-pointer type for `ncclCommInitRankConfig` (initialize a communicator rank with config). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommInitRankConfig = unsafe extern "C" fn(
    comm: *mut ncclComm_t,
    nranks: c_int,
    comm_id: ncclUniqueId,
    rank: c_int,
    config: *mut c_void, // ncclConfig_t
) -> ncclResult_t;

// ---- Memory helpers (NCCL 2.19+) ----

/// Function-pointer type for `ncclMemAlloc` (allocate NCCL-registered device memory). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclMemAlloc =
    unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> ncclResult_t;
/// Function-pointer type for `ncclMemFree` (free NCCL-registered device memory). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclMemFree = unsafe extern "C" fn(ptr: *mut c_void) -> ncclResult_t;

/// Function-pointer type for `ncclCommRegister` (register a user buffer with a communicator). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommRegister = unsafe extern "C" fn(
    comm: ncclComm_t,
    buff: *mut c_void,
    size: usize,
    handle: *mut *mut c_void,
) -> ncclResult_t;

/// Function-pointer type for `ncclCommDeregister` (deregister a user buffer from a communicator). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclCommDeregister =
    unsafe extern "C" fn(comm: ncclComm_t, handle: *mut c_void) -> ncclResult_t;

// ---- Custom reduction ops ----

/// Function-pointer type for `ncclRedOpCreatePreMulSum` (create a custom pre-multiplied-sum reduction op). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclRedOpCreatePreMulSum = unsafe extern "C" fn(
    op: *mut ncclRedOp_t,
    scalar: *mut c_void,
    datatype: ncclDataType_t,
    residence: i32, // ncclScalarResidence_t
    comm: ncclComm_t,
) -> ncclResult_t;

/// Function-pointer type for `ncclRedOpDestroy` (destroy a custom reduction op). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclRedOpDestroy =
    unsafe extern "C" fn(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;

// ---- Error strings ----

/// Function-pointer type for `ncclGetErrorString` (decode an ncclResult_t into a static C string). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclGetErrorString =
    unsafe extern "C" fn(result: ncclResult_t) -> *const core::ffi::c_char;
/// Function-pointer type for `ncclGetLastError` (fetch the last error string on a communicator). See <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html>.
pub type PFN_ncclGetLastError =
    unsafe extern "C" fn(comm: ncclComm_t) -> *const core::ffi::c_char;

// ---- loader --------------------------------------------------------------

fn nccl_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &["libnccl.so.2", "libnccl.so"]
    }
    #[cfg(target_os = "windows")]
    {
        &["nccl.dll", "libnccl.dll"]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

macro_rules! nccl_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        /// Lazily-resolved NCCL function-pointer table.
        pub struct Nccl {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Nccl {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Nccl").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Nccl {
            $(
                /// `func` (func).
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
        }
    };
}

nccl_fns! {
    nccl_get_version as "ncclGetVersion": PFN_ncclGetVersion;
    nccl_get_unique_id as "ncclGetUniqueId": PFN_ncclGetUniqueId;
    nccl_comm_init_rank as "ncclCommInitRank": PFN_ncclCommInitRank;
    nccl_comm_init_rank_config as "ncclCommInitRankConfig": PFN_ncclCommInitRankConfig;
    nccl_comm_init_all as "ncclCommInitAll": PFN_ncclCommInitAll;
    nccl_comm_destroy as "ncclCommDestroy": PFN_ncclCommDestroy;
    nccl_comm_abort as "ncclCommAbort": PFN_ncclCommAbort;
    nccl_comm_finalize as "ncclCommFinalize": PFN_ncclCommFinalize;
    nccl_comm_get_async_error as "ncclCommGetAsyncError": PFN_ncclCommGetAsyncError;
    nccl_comm_count as "ncclCommCount": PFN_ncclCommCount;
    nccl_comm_user_rank as "ncclCommUserRank": PFN_ncclCommUserRank;
    nccl_comm_cu_device as "ncclCommCuDevice": PFN_ncclCommCuDevice;
    nccl_comm_split as "ncclCommSplit": PFN_ncclCommSplit;
    nccl_all_reduce as "ncclAllReduce": PFN_ncclAllReduce;
    nccl_reduce as "ncclReduce": PFN_ncclReduce;
    nccl_broadcast as "ncclBroadcast": PFN_ncclBroadcast;
    nccl_all_gather as "ncclAllGather": PFN_ncclAllGather;
    nccl_reduce_scatter as "ncclReduceScatter": PFN_ncclReduceScatter;
    nccl_send as "ncclSend": PFN_ncclSend;
    nccl_recv as "ncclRecv": PFN_ncclRecv;
    nccl_group_start as "ncclGroupStart": PFN_ncclGroupStart;
    nccl_group_end as "ncclGroupEnd": PFN_ncclGroupEnd;
    nccl_mem_alloc as "ncclMemAlloc": PFN_ncclMemAlloc;
    nccl_mem_free as "ncclMemFree": PFN_ncclMemFree;
    nccl_comm_register as "ncclCommRegister": PFN_ncclCommRegister;
    nccl_comm_deregister as "ncclCommDeregister": PFN_ncclCommDeregister;
    nccl_red_op_create_pre_mul_sum as "ncclRedOpCreatePreMulSum": PFN_ncclRedOpCreatePreMulSum;
    nccl_red_op_destroy as "ncclRedOpDestroy": PFN_ncclRedOpDestroy;
    nccl_get_error_string as "ncclGetErrorString": PFN_ncclGetErrorString;
    nccl_get_last_error as "ncclGetLastError": PFN_ncclGetLastError;
}

/// Return the lazily-loaded NCCL library accessor.
pub fn nccl() -> Result<&'static Nccl, LoaderError> {
    static NCCL: OnceLock<Nccl> = OnceLock::new();
    if let Some(n) = NCCL.get() {
        return Ok(n);
    }
    let lib = Library::open("nccl", nccl_candidates())?;
    let n = Nccl::empty(lib);
    let _ = NCCL.set(n);
    Ok(NCCL.get().expect("OnceLock set or lost race"))
}
