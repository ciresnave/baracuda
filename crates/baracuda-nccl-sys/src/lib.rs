//! Raw FFI + dynamic loader for NVIDIA NCCL (multi-GPU collective communication).
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
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    BFloat16 = 9,
}

/// NCCL reduction operation.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ncclRedOp_t {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

// ---- status ---------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct ncclResult_t(pub i32);

impl ncclResult_t {
    pub const Success: Self = Self(0);
    pub const UnhandledCudaError: Self = Self(1);
    pub const SystemError: Self = Self(2);
    pub const InternalError: Self = Self(3);
    pub const InvalidArgument: Self = Self(4);
    pub const InvalidUsage: Self = Self(5);
    pub const RemoteError: Self = Self(6);
    pub const InProgress: Self = Self(7);

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

pub type PFN_ncclGetVersion = unsafe extern "C" fn(version: *mut c_int) -> ncclResult_t;
pub type PFN_ncclGetUniqueId = unsafe extern "C" fn(id: *mut ncclUniqueId) -> ncclResult_t;
pub type PFN_ncclCommInitRank = unsafe extern "C" fn(
    comm: *mut ncclComm_t,
    nranks: c_int,
    comm_id: ncclUniqueId,
    rank: c_int,
) -> ncclResult_t;
pub type PFN_ncclCommInitAll = unsafe extern "C" fn(
    comms: *mut ncclComm_t,
    ndev: c_int,
    dev_list: *const c_int,
) -> ncclResult_t;
pub type PFN_ncclCommDestroy = unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t;
pub type PFN_ncclCommCount =
    unsafe extern "C" fn(comm: ncclComm_t, count: *mut c_int) -> ncclResult_t;
pub type PFN_ncclCommUserRank =
    unsafe extern "C" fn(comm: ncclComm_t, rank: *mut c_int) -> ncclResult_t;

pub type PFN_ncclAllReduce = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

pub type PFN_ncclBroadcast = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

pub type PFN_ncclGroupStart = unsafe extern "C" fn() -> ncclResult_t;
pub type PFN_ncclGroupEnd = unsafe extern "C" fn() -> ncclResult_t;

// ---- Full collective surface ----

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

pub type PFN_ncclAllGather = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: usize,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

pub type PFN_ncclReduceScatter = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

pub type PFN_ncclSend = unsafe extern "C" fn(
    sendbuff: *const c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

pub type PFN_ncclRecv = unsafe extern "C" fn(
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: cudaStream_t,
) -> ncclResult_t;

// ---- Communicator lifecycle extras ----

pub type PFN_ncclCommAbort = unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t;
pub type PFN_ncclCommFinalize = unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t;
pub type PFN_ncclCommGetAsyncError =
    unsafe extern "C" fn(comm: ncclComm_t, async_error: *mut ncclResult_t) -> ncclResult_t;
pub type PFN_ncclCommCuDevice =
    unsafe extern "C" fn(comm: ncclComm_t, device: *mut c_int) -> ncclResult_t;
pub type PFN_ncclCommSplit = unsafe extern "C" fn(
    comm: ncclComm_t,
    color: c_int,
    key: c_int,
    new_comm: *mut ncclComm_t,
    config: *mut c_void, // ncclConfig_t
) -> ncclResult_t;

pub type PFN_ncclCommInitRankConfig = unsafe extern "C" fn(
    comm: *mut ncclComm_t,
    nranks: c_int,
    comm_id: ncclUniqueId,
    rank: c_int,
    config: *mut c_void, // ncclConfig_t
) -> ncclResult_t;

// ---- Memory helpers (NCCL 2.19+) ----

pub type PFN_ncclMemAlloc =
    unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> ncclResult_t;
pub type PFN_ncclMemFree = unsafe extern "C" fn(ptr: *mut c_void) -> ncclResult_t;

pub type PFN_ncclCommRegister = unsafe extern "C" fn(
    comm: ncclComm_t,
    buff: *mut c_void,
    size: usize,
    handle: *mut *mut c_void,
) -> ncclResult_t;

pub type PFN_ncclCommDeregister =
    unsafe extern "C" fn(comm: ncclComm_t, handle: *mut c_void) -> ncclResult_t;

// ---- Custom reduction ops ----

pub type PFN_ncclRedOpCreatePreMulSum = unsafe extern "C" fn(
    op: *mut ncclRedOp_t,
    scalar: *mut c_void,
    datatype: ncclDataType_t,
    residence: i32, // ncclScalarResidence_t
    comm: ncclComm_t,
) -> ncclResult_t;

pub type PFN_ncclRedOpDestroy =
    unsafe extern "C" fn(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;

// ---- Error strings ----

pub type PFN_ncclGetErrorString =
    unsafe extern "C" fn(result: ncclResult_t) -> *const core::ffi::c_char;
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
