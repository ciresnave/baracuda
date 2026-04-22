//! `CUresult` — the Driver API status enum — plus its `CudaStatus` impl.
//!
//! Modelled as `#[repr(transparent)] struct CUresult(pub i32)` rather than a
//! Rust enum: the CUDA driver is free to return a value we don't recognize
//! and we must not invoke UB by transmuting it into an exhaustive enum.

use baracuda_types::CudaStatus;

/// Return code from a CUDA Driver API call.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct CUresult(pub i32);

impl CUresult {
    pub const SUCCESS: Self = Self(0);
    pub const ERROR_INVALID_VALUE: Self = Self(1);
    pub const ERROR_OUT_OF_MEMORY: Self = Self(2);
    pub const ERROR_NOT_INITIALIZED: Self = Self(3);
    pub const ERROR_DEINITIALIZED: Self = Self(4);
    pub const ERROR_PROFILER_DISABLED: Self = Self(5);
    pub const ERROR_STUB_LIBRARY: Self = Self(34);
    pub const ERROR_DEVICE_UNAVAILABLE: Self = Self(46);
    pub const ERROR_NO_DEVICE: Self = Self(100);
    pub const ERROR_INVALID_DEVICE: Self = Self(101);
    pub const ERROR_DEVICE_NOT_LICENSED: Self = Self(102);
    pub const ERROR_INVALID_IMAGE: Self = Self(200);
    pub const ERROR_INVALID_CONTEXT: Self = Self(201);
    pub const ERROR_CONTEXT_ALREADY_CURRENT: Self = Self(202);
    pub const ERROR_MAP_FAILED: Self = Self(205);
    pub const ERROR_UNMAP_FAILED: Self = Self(206);
    pub const ERROR_ARRAY_IS_MAPPED: Self = Self(207);
    pub const ERROR_ALREADY_MAPPED: Self = Self(208);
    pub const ERROR_NO_BINARY_FOR_GPU: Self = Self(209);
    pub const ERROR_ALREADY_ACQUIRED: Self = Self(210);
    pub const ERROR_NOT_MAPPED: Self = Self(211);
    pub const ERROR_NOT_MAPPED_AS_ARRAY: Self = Self(212);
    pub const ERROR_NOT_MAPPED_AS_POINTER: Self = Self(213);
    pub const ERROR_ECC_UNCORRECTABLE: Self = Self(214);
    pub const ERROR_UNSUPPORTED_LIMIT: Self = Self(215);
    pub const ERROR_CONTEXT_ALREADY_IN_USE: Self = Self(216);
    pub const ERROR_PEER_ACCESS_UNSUPPORTED: Self = Self(217);
    pub const ERROR_INVALID_PTX: Self = Self(218);
    pub const ERROR_INVALID_GRAPHICS_CONTEXT: Self = Self(219);
    pub const ERROR_NVLINK_UNCORRECTABLE: Self = Self(220);
    pub const ERROR_JIT_COMPILER_NOT_FOUND: Self = Self(221);
    pub const ERROR_UNSUPPORTED_PTX_VERSION: Self = Self(222);
    pub const ERROR_JIT_COMPILATION_DISABLED: Self = Self(223);
    pub const ERROR_UNSUPPORTED_EXEC_AFFINITY: Self = Self(224);
    pub const ERROR_INVALID_SOURCE: Self = Self(300);
    pub const ERROR_FILE_NOT_FOUND: Self = Self(301);
    pub const ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: Self = Self(302);
    pub const ERROR_SHARED_OBJECT_INIT_FAILED: Self = Self(303);
    pub const ERROR_OPERATING_SYSTEM: Self = Self(304);
    pub const ERROR_INVALID_HANDLE: Self = Self(400);
    pub const ERROR_ILLEGAL_STATE: Self = Self(401);
    pub const ERROR_NOT_FOUND: Self = Self(500);
    pub const ERROR_NOT_READY: Self = Self(600);
    pub const ERROR_ILLEGAL_ADDRESS: Self = Self(700);
    pub const ERROR_LAUNCH_OUT_OF_RESOURCES: Self = Self(701);
    pub const ERROR_LAUNCH_TIMEOUT: Self = Self(702);
    pub const ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: Self = Self(703);
    pub const ERROR_PEER_ACCESS_ALREADY_ENABLED: Self = Self(704);
    pub const ERROR_PEER_ACCESS_NOT_ENABLED: Self = Self(705);
    pub const ERROR_PRIMARY_CONTEXT_ACTIVE: Self = Self(708);
    pub const ERROR_CONTEXT_IS_DESTROYED: Self = Self(709);
    pub const ERROR_ASSERT: Self = Self(710);
    pub const ERROR_TOO_MANY_PEERS: Self = Self(711);
    pub const ERROR_HOST_MEMORY_ALREADY_REGISTERED: Self = Self(712);
    pub const ERROR_HOST_MEMORY_NOT_REGISTERED: Self = Self(713);
    pub const ERROR_HARDWARE_STACK_ERROR: Self = Self(714);
    pub const ERROR_ILLEGAL_INSTRUCTION: Self = Self(715);
    pub const ERROR_MISALIGNED_ADDRESS: Self = Self(716);
    pub const ERROR_INVALID_ADDRESS_SPACE: Self = Self(717);
    pub const ERROR_INVALID_PC: Self = Self(718);
    pub const ERROR_LAUNCH_FAILED: Self = Self(719);
    pub const ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: Self = Self(720);
    pub const ERROR_NOT_PERMITTED: Self = Self(800);
    pub const ERROR_NOT_SUPPORTED: Self = Self(801);
    pub const ERROR_SYSTEM_NOT_READY: Self = Self(802);
    pub const ERROR_SYSTEM_DRIVER_MISMATCH: Self = Self(803);
    pub const ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: Self = Self(804);
    pub const ERROR_MPS_CONNECTION_FAILED: Self = Self(805);
    pub const ERROR_MPS_RPC_FAILURE: Self = Self(806);
    pub const ERROR_MPS_SERVER_NOT_READY: Self = Self(807);
    pub const ERROR_MPS_MAX_CLIENTS_REACHED: Self = Self(808);
    pub const ERROR_MPS_MAX_CONNECTIONS_REACHED: Self = Self(809);
    pub const ERROR_STREAM_CAPTURE_UNSUPPORTED: Self = Self(900);
    pub const ERROR_STREAM_CAPTURE_INVALIDATED: Self = Self(901);
    pub const ERROR_STREAM_CAPTURE_MERGE: Self = Self(902);
    pub const ERROR_STREAM_CAPTURE_UNMATCHED: Self = Self(903);
    pub const ERROR_STREAM_CAPTURE_UNJOINED: Self = Self(904);
    pub const ERROR_STREAM_CAPTURE_ISOLATION: Self = Self(905);
    pub const ERROR_STREAM_CAPTURE_IMPLICIT: Self = Self(906);
    pub const ERROR_CAPTURED_EVENT: Self = Self(907);
    pub const ERROR_STREAM_CAPTURE_WRONG_THREAD: Self = Self(908);
    pub const ERROR_TIMEOUT: Self = Self(909);
    pub const ERROR_GRAPH_EXEC_UPDATE_FAILURE: Self = Self(910);
    pub const ERROR_EXTERNAL_DEVICE: Self = Self(911);
    pub const ERROR_INVALID_CLUSTER_SIZE: Self = Self(912);
    pub const ERROR_UNKNOWN: Self = Self(999);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for CUresult {
    fn code(self) -> i32 {
        self.0
    }

    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUDA_SUCCESS",
            1 => "CUDA_ERROR_INVALID_VALUE",
            2 => "CUDA_ERROR_OUT_OF_MEMORY",
            3 => "CUDA_ERROR_NOT_INITIALIZED",
            4 => "CUDA_ERROR_DEINITIALIZED",
            5 => "CUDA_ERROR_PROFILER_DISABLED",
            34 => "CUDA_ERROR_STUB_LIBRARY",
            46 => "CUDA_ERROR_DEVICE_UNAVAILABLE",
            100 => "CUDA_ERROR_NO_DEVICE",
            101 => "CUDA_ERROR_INVALID_DEVICE",
            200 => "CUDA_ERROR_INVALID_IMAGE",
            201 => "CUDA_ERROR_INVALID_CONTEXT",
            205 => "CUDA_ERROR_MAP_FAILED",
            206 => "CUDA_ERROR_UNMAP_FAILED",
            209 => "CUDA_ERROR_NO_BINARY_FOR_GPU",
            214 => "CUDA_ERROR_ECC_UNCORRECTABLE",
            216 => "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
            218 => "CUDA_ERROR_INVALID_PTX",
            220 => "CUDA_ERROR_NVLINK_UNCORRECTABLE",
            300 => "CUDA_ERROR_INVALID_SOURCE",
            301 => "CUDA_ERROR_FILE_NOT_FOUND",
            400 => "CUDA_ERROR_INVALID_HANDLE",
            500 => "CUDA_ERROR_NOT_FOUND",
            600 => "CUDA_ERROR_NOT_READY",
            700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
            701 => "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            702 => "CUDA_ERROR_LAUNCH_TIMEOUT",
            708 => "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
            709 => "CUDA_ERROR_CONTEXT_IS_DESTROYED",
            710 => "CUDA_ERROR_ASSERT",
            716 => "CUDA_ERROR_MISALIGNED_ADDRESS",
            719 => "CUDA_ERROR_LAUNCH_FAILED",
            800 => "CUDA_ERROR_NOT_PERMITTED",
            801 => "CUDA_ERROR_NOT_SUPPORTED",
            999 => "CUDA_ERROR_UNKNOWN",
            _ => "CUDA_ERROR_UNRECOGNIZED",
        }
    }

    fn description(self) -> &'static str {
        match self.0 {
            0 => "no error",
            1 => "invalid argument",
            2 => "out of memory",
            3 => "driver not initialized",
            4 => "driver has been deinitialized",
            100 => "no CUDA-capable device is detected",
            101 => "invalid device ordinal",
            200 => "device kernel image is invalid",
            201 => "invalid device context",
            209 => "no kernel image is available for execution on the device",
            214 => "uncorrectable ECC error encountered",
            218 => "invalid PTX",
            219 => "invalid graphics context",
            300 => "invalid source",
            301 => "file not found",
            400 => "invalid resource handle",
            500 => "named symbol not found",
            600 => "operation not yet complete",
            700 => "an illegal memory access was encountered",
            701 => "launch requires resources the device cannot provide",
            702 => "launch timed out and was terminated",
            708 => "primary context is already active",
            709 => "context is destroyed",
            716 => "misaligned address",
            719 => "unspecified launch failure",
            800 => "operation not permitted",
            801 => "operation not supported",
            999 => "unknown error",
            _ => "unrecognized CUDA driver error code",
        }
    }

    fn is_success(self) -> bool {
        CUresult::is_success(self)
    }

    fn library(self) -> &'static str {
        "cuda-driver"
    }
}
