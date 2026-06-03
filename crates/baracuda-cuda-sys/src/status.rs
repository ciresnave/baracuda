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
    /// `SUCCESS` — success.
    pub const SUCCESS: Self = Self(0);
    /// `ERROR_INVALID_VALUE` — error invalid value.
    pub const ERROR_INVALID_VALUE: Self = Self(1);
    /// `ERROR_OUT_OF_MEMORY` — error out of memory.
    pub const ERROR_OUT_OF_MEMORY: Self = Self(2);
    /// `ERROR_NOT_INITIALIZED` — error not initialized.
    pub const ERROR_NOT_INITIALIZED: Self = Self(3);
    /// `ERROR_DEINITIALIZED` — error deinitialized.
    pub const ERROR_DEINITIALIZED: Self = Self(4);
    /// `ERROR_PROFILER_DISABLED` — error profiler disabled.
    pub const ERROR_PROFILER_DISABLED: Self = Self(5);
    /// `ERROR_STUB_LIBRARY` — error stub library.
    pub const ERROR_STUB_LIBRARY: Self = Self(34);
    /// `ERROR_DEVICE_UNAVAILABLE` — error device unavailable.
    pub const ERROR_DEVICE_UNAVAILABLE: Self = Self(46);
    /// `ERROR_NO_DEVICE` — error no device.
    pub const ERROR_NO_DEVICE: Self = Self(100);
    /// `ERROR_INVALID_DEVICE` — error invalid device.
    pub const ERROR_INVALID_DEVICE: Self = Self(101);
    /// `ERROR_DEVICE_NOT_LICENSED` — error device not licensed.
    pub const ERROR_DEVICE_NOT_LICENSED: Self = Self(102);
    /// `ERROR_INVALID_IMAGE` — error invalid image.
    pub const ERROR_INVALID_IMAGE: Self = Self(200);
    /// `ERROR_INVALID_CONTEXT` — error invalid context.
    pub const ERROR_INVALID_CONTEXT: Self = Self(201);
    /// `ERROR_CONTEXT_ALREADY_CURRENT` — error context already current.
    pub const ERROR_CONTEXT_ALREADY_CURRENT: Self = Self(202);
    /// `ERROR_MAP_FAILED` — error map failed.
    pub const ERROR_MAP_FAILED: Self = Self(205);
    /// `ERROR_UNMAP_FAILED` — error unmap failed.
    pub const ERROR_UNMAP_FAILED: Self = Self(206);
    /// `ERROR_ARRAY_IS_MAPPED` — error array is mapped.
    pub const ERROR_ARRAY_IS_MAPPED: Self = Self(207);
    /// `ERROR_ALREADY_MAPPED` — error already mapped.
    pub const ERROR_ALREADY_MAPPED: Self = Self(208);
    /// `ERROR_NO_BINARY_FOR_GPU` — error no binary for gpu.
    pub const ERROR_NO_BINARY_FOR_GPU: Self = Self(209);
    /// `ERROR_ALREADY_ACQUIRED` — error already acquired.
    pub const ERROR_ALREADY_ACQUIRED: Self = Self(210);
    /// `ERROR_NOT_MAPPED` — error not mapped.
    pub const ERROR_NOT_MAPPED: Self = Self(211);
    /// `ERROR_NOT_MAPPED_AS_ARRAY` — error not mapped as array.
    pub const ERROR_NOT_MAPPED_AS_ARRAY: Self = Self(212);
    /// `ERROR_NOT_MAPPED_AS_POINTER` — error not mapped as pointer.
    pub const ERROR_NOT_MAPPED_AS_POINTER: Self = Self(213);
    /// `ERROR_ECC_UNCORRECTABLE` — error ecc uncorrectable.
    pub const ERROR_ECC_UNCORRECTABLE: Self = Self(214);
    /// `ERROR_UNSUPPORTED_LIMIT` — error unsupported limit.
    pub const ERROR_UNSUPPORTED_LIMIT: Self = Self(215);
    /// `ERROR_CONTEXT_ALREADY_IN_USE` — error context already in use.
    pub const ERROR_CONTEXT_ALREADY_IN_USE: Self = Self(216);
    /// `ERROR_PEER_ACCESS_UNSUPPORTED` — error peer access unsupported.
    pub const ERROR_PEER_ACCESS_UNSUPPORTED: Self = Self(217);
    /// `ERROR_INVALID_PTX` — error invalid ptx.
    pub const ERROR_INVALID_PTX: Self = Self(218);
    /// `ERROR_INVALID_GRAPHICS_CONTEXT` — error invalid graphics context.
    pub const ERROR_INVALID_GRAPHICS_CONTEXT: Self = Self(219);
    /// `ERROR_NVLINK_UNCORRECTABLE` — error nvlink uncorrectable.
    pub const ERROR_NVLINK_UNCORRECTABLE: Self = Self(220);
    /// `ERROR_JIT_COMPILER_NOT_FOUND` — error jit compiler not found.
    pub const ERROR_JIT_COMPILER_NOT_FOUND: Self = Self(221);
    /// `ERROR_UNSUPPORTED_PTX_VERSION` — error unsupported ptx version.
    pub const ERROR_UNSUPPORTED_PTX_VERSION: Self = Self(222);
    /// `ERROR_JIT_COMPILATION_DISABLED` — error jit compilation disabled.
    pub const ERROR_JIT_COMPILATION_DISABLED: Self = Self(223);
    /// `ERROR_UNSUPPORTED_EXEC_AFFINITY` — error unsupported exec affinity.
    pub const ERROR_UNSUPPORTED_EXEC_AFFINITY: Self = Self(224);
    /// `ERROR_INVALID_SOURCE` — error invalid source.
    pub const ERROR_INVALID_SOURCE: Self = Self(300);
    /// `ERROR_FILE_NOT_FOUND` — error file not found.
    pub const ERROR_FILE_NOT_FOUND: Self = Self(301);
    /// `ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND` — error shared object symbol not found.
    pub const ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: Self = Self(302);
    /// `ERROR_SHARED_OBJECT_INIT_FAILED` — error shared object init failed.
    pub const ERROR_SHARED_OBJECT_INIT_FAILED: Self = Self(303);
    /// `ERROR_OPERATING_SYSTEM` — error operating system.
    pub const ERROR_OPERATING_SYSTEM: Self = Self(304);
    /// `ERROR_INVALID_HANDLE` — error invalid handle.
    pub const ERROR_INVALID_HANDLE: Self = Self(400);
    /// `ERROR_ILLEGAL_STATE` — error illegal state.
    pub const ERROR_ILLEGAL_STATE: Self = Self(401);
    /// `ERROR_NOT_FOUND` — error not found.
    pub const ERROR_NOT_FOUND: Self = Self(500);
    /// `ERROR_NOT_READY` — error not ready.
    pub const ERROR_NOT_READY: Self = Self(600);
    /// `ERROR_ILLEGAL_ADDRESS` — error illegal address.
    pub const ERROR_ILLEGAL_ADDRESS: Self = Self(700);
    /// `ERROR_LAUNCH_OUT_OF_RESOURCES` — error launch out of resources.
    pub const ERROR_LAUNCH_OUT_OF_RESOURCES: Self = Self(701);
    /// `ERROR_LAUNCH_TIMEOUT` — error launch timeout.
    pub const ERROR_LAUNCH_TIMEOUT: Self = Self(702);
    /// `ERROR_LAUNCH_INCOMPATIBLE_TEXTURING` — error launch incompatible texturing.
    pub const ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: Self = Self(703);
    /// `ERROR_PEER_ACCESS_ALREADY_ENABLED` — error peer access already enabled.
    pub const ERROR_PEER_ACCESS_ALREADY_ENABLED: Self = Self(704);
    /// `ERROR_PEER_ACCESS_NOT_ENABLED` — error peer access not enabled.
    pub const ERROR_PEER_ACCESS_NOT_ENABLED: Self = Self(705);
    /// `ERROR_PRIMARY_CONTEXT_ACTIVE` — error primary context active.
    pub const ERROR_PRIMARY_CONTEXT_ACTIVE: Self = Self(708);
    /// `ERROR_CONTEXT_IS_DESTROYED` — error context is destroyed.
    pub const ERROR_CONTEXT_IS_DESTROYED: Self = Self(709);
    /// `ERROR_ASSERT` — error assert.
    pub const ERROR_ASSERT: Self = Self(710);
    /// `ERROR_TOO_MANY_PEERS` — error too many peers.
    pub const ERROR_TOO_MANY_PEERS: Self = Self(711);
    /// `ERROR_HOST_MEMORY_ALREADY_REGISTERED` — error host memory already registered.
    pub const ERROR_HOST_MEMORY_ALREADY_REGISTERED: Self = Self(712);
    /// `ERROR_HOST_MEMORY_NOT_REGISTERED` — error host memory not registered.
    pub const ERROR_HOST_MEMORY_NOT_REGISTERED: Self = Self(713);
    /// `ERROR_HARDWARE_STACK_ERROR` — error hardware stack error.
    pub const ERROR_HARDWARE_STACK_ERROR: Self = Self(714);
    /// `ERROR_ILLEGAL_INSTRUCTION` — error illegal instruction.
    pub const ERROR_ILLEGAL_INSTRUCTION: Self = Self(715);
    /// `ERROR_MISALIGNED_ADDRESS` — error misaligned address.
    pub const ERROR_MISALIGNED_ADDRESS: Self = Self(716);
    /// `ERROR_INVALID_ADDRESS_SPACE` — error invalid address space.
    pub const ERROR_INVALID_ADDRESS_SPACE: Self = Self(717);
    /// `ERROR_INVALID_PC` — error invalid pc.
    pub const ERROR_INVALID_PC: Self = Self(718);
    /// `ERROR_LAUNCH_FAILED` — error launch failed.
    pub const ERROR_LAUNCH_FAILED: Self = Self(719);
    /// `ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` — error cooperative launch too large.
    pub const ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: Self = Self(720);
    /// `ERROR_NOT_PERMITTED` — error not permitted.
    pub const ERROR_NOT_PERMITTED: Self = Self(800);
    /// `ERROR_NOT_SUPPORTED` — error not supported.
    pub const ERROR_NOT_SUPPORTED: Self = Self(801);
    /// `ERROR_SYSTEM_NOT_READY` — error system not ready.
    pub const ERROR_SYSTEM_NOT_READY: Self = Self(802);
    /// `ERROR_SYSTEM_DRIVER_MISMATCH` — error system driver mismatch.
    pub const ERROR_SYSTEM_DRIVER_MISMATCH: Self = Self(803);
    /// `ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE` — error compat not supported on device.
    pub const ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: Self = Self(804);
    /// `ERROR_MPS_CONNECTION_FAILED` — error mps connection failed.
    pub const ERROR_MPS_CONNECTION_FAILED: Self = Self(805);
    /// `ERROR_MPS_RPC_FAILURE` — error mps rpc failure.
    pub const ERROR_MPS_RPC_FAILURE: Self = Self(806);
    /// `ERROR_MPS_SERVER_NOT_READY` — error mps server not ready.
    pub const ERROR_MPS_SERVER_NOT_READY: Self = Self(807);
    /// `ERROR_MPS_MAX_CLIENTS_REACHED` — error mps max clients reached.
    pub const ERROR_MPS_MAX_CLIENTS_REACHED: Self = Self(808);
    /// `ERROR_MPS_MAX_CONNECTIONS_REACHED` — error mps max connections reached.
    pub const ERROR_MPS_MAX_CONNECTIONS_REACHED: Self = Self(809);
    /// `ERROR_STREAM_CAPTURE_UNSUPPORTED` — error stream capture unsupported.
    pub const ERROR_STREAM_CAPTURE_UNSUPPORTED: Self = Self(900);
    /// `ERROR_STREAM_CAPTURE_INVALIDATED` — error stream capture invalidated.
    pub const ERROR_STREAM_CAPTURE_INVALIDATED: Self = Self(901);
    /// `ERROR_STREAM_CAPTURE_MERGE` — error stream capture merge.
    pub const ERROR_STREAM_CAPTURE_MERGE: Self = Self(902);
    /// `ERROR_STREAM_CAPTURE_UNMATCHED` — error stream capture unmatched.
    pub const ERROR_STREAM_CAPTURE_UNMATCHED: Self = Self(903);
    /// `ERROR_STREAM_CAPTURE_UNJOINED` — error stream capture unjoined.
    pub const ERROR_STREAM_CAPTURE_UNJOINED: Self = Self(904);
    /// `ERROR_STREAM_CAPTURE_ISOLATION` — error stream capture isolation.
    pub const ERROR_STREAM_CAPTURE_ISOLATION: Self = Self(905);
    /// `ERROR_STREAM_CAPTURE_IMPLICIT` — error stream capture implicit.
    pub const ERROR_STREAM_CAPTURE_IMPLICIT: Self = Self(906);
    /// `ERROR_CAPTURED_EVENT` — error captured event.
    pub const ERROR_CAPTURED_EVENT: Self = Self(907);
    /// `ERROR_STREAM_CAPTURE_WRONG_THREAD` — error stream capture wrong thread.
    pub const ERROR_STREAM_CAPTURE_WRONG_THREAD: Self = Self(908);
    /// `ERROR_TIMEOUT` — error timeout.
    pub const ERROR_TIMEOUT: Self = Self(909);
    /// `ERROR_GRAPH_EXEC_UPDATE_FAILURE` — error graph exec update failure.
    pub const ERROR_GRAPH_EXEC_UPDATE_FAILURE: Self = Self(910);
    /// `ERROR_EXTERNAL_DEVICE` — error external device.
    pub const ERROR_EXTERNAL_DEVICE: Self = Self(911);
    /// `ERROR_INVALID_CLUSTER_SIZE` — error invalid cluster size.
    pub const ERROR_INVALID_CLUSTER_SIZE: Self = Self(912);
    /// `ERROR_UNKNOWN` — error unknown.
    pub const ERROR_UNKNOWN: Self = Self(999);

    /// `is_success` — is success.
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
