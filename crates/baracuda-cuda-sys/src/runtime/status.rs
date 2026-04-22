//! `cudaError_t` — the Runtime API status enum — plus its `CudaStatus` impl.

use baracuda_types::CudaStatus;

/// Return code from a CUDA Runtime API call.
///
/// Modelled as `#[repr(transparent)] struct cudaError_t(pub i32)` — same
/// reasoning as [`crate::CUresult`]: the runtime may return codes we don't
/// yet recognize, and we must not transmute into an exhaustive enum.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cudaError_t(pub i32);

#[allow(non_upper_case_globals)]
impl cudaError_t {
    pub const Success: Self = Self(0);
    pub const InvalidValue: Self = Self(1);
    pub const MemoryAllocation: Self = Self(2);
    pub const InitializationError: Self = Self(3);
    pub const CudartUnloading: Self = Self(4);
    pub const ProfilerDisabled: Self = Self(5);
    pub const InvalidConfiguration: Self = Self(9);
    pub const InvalidPitchValue: Self = Self(12);
    pub const InvalidSymbol: Self = Self(13);
    pub const InvalidHostPointer: Self = Self(16);
    pub const InvalidDevicePointer: Self = Self(17);
    pub const InvalidTexture: Self = Self(18);
    pub const InvalidDeviceFunction: Self = Self(98);
    pub const NoDevice: Self = Self(100);
    pub const InvalidDevice: Self = Self(101);
    pub const DeviceNotLicensed: Self = Self(102);
    pub const SoftwareValidityNotEstablished: Self = Self(103);
    pub const StartupFailure: Self = Self(127);
    pub const InvalidKernelImage: Self = Self(200);
    pub const DeviceUninitialized: Self = Self(201);
    pub const MapBufferObjectFailed: Self = Self(205);
    pub const UnmapBufferObjectFailed: Self = Self(206);
    pub const ArrayIsMapped: Self = Self(207);
    pub const AlreadyMapped: Self = Self(208);
    pub const NoKernelImageForDevice: Self = Self(209);
    pub const AlreadyAcquired: Self = Self(210);
    pub const NotMapped: Self = Self(211);
    pub const ECCUncorrectable: Self = Self(214);
    pub const UnsupportedLimit: Self = Self(215);
    pub const DeviceAlreadyInUse: Self = Self(216);
    pub const PeerAccessUnsupported: Self = Self(217);
    pub const InvalidPtx: Self = Self(218);
    pub const InvalidGraphicsContext: Self = Self(219);
    pub const NvlinkUncorrectable: Self = Self(220);
    pub const JitCompilerNotFound: Self = Self(221);
    pub const UnsupportedPtxVersion: Self = Self(222);
    pub const InvalidSource: Self = Self(300);
    pub const FileNotFound: Self = Self(301);
    pub const SharedObjectSymbolNotFound: Self = Self(302);
    pub const SharedObjectInitFailed: Self = Self(303);
    pub const OperatingSystem: Self = Self(304);
    pub const InvalidResourceHandle: Self = Self(400);
    pub const IllegalState: Self = Self(401);
    pub const SymbolNotFound: Self = Self(500);
    pub const NotReady: Self = Self(600);
    pub const IllegalAddress: Self = Self(700);
    pub const LaunchOutOfResources: Self = Self(701);
    pub const LaunchTimeout: Self = Self(702);
    pub const PrimaryContextActive: Self = Self(708);
    pub const ContextIsDestroyed: Self = Self(709);
    pub const Assert: Self = Self(710);
    pub const MisalignedAddress: Self = Self(716);
    pub const LaunchFailure: Self = Self(719);
    pub const CooperativeLaunchTooLarge: Self = Self(720);
    pub const NotPermitted: Self = Self(800);
    pub const NotSupported: Self = Self(801);
    pub const SystemNotReady: Self = Self(802);
    pub const SystemDriverMismatch: Self = Self(803);
    pub const CompatNotSupportedOnDevice: Self = Self(804);
    pub const StreamCaptureUnsupported: Self = Self(900);
    pub const StreamCaptureInvalidated: Self = Self(901);
    pub const StreamCaptureMerge: Self = Self(902);
    pub const StreamCaptureUnmatched: Self = Self(903);
    pub const StreamCaptureUnjoined: Self = Self(904);
    pub const StreamCaptureIsolation: Self = Self(905);
    pub const StreamCaptureImplicit: Self = Self(906);
    pub const CapturedEvent: Self = Self(907);
    pub const StreamCaptureWrongThread: Self = Self(908);
    pub const Timeout: Self = Self(909);
    pub const GraphExecUpdateFailure: Self = Self(910);
    pub const Unknown: Self = Self(999);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for cudaError_t {
    fn code(self) -> i32 {
        self.0
    }

    fn name(self) -> &'static str {
        match self.0 {
            0 => "cudaSuccess",
            1 => "cudaErrorInvalidValue",
            2 => "cudaErrorMemoryAllocation",
            3 => "cudaErrorInitializationError",
            4 => "cudaErrorCudartUnloading",
            9 => "cudaErrorInvalidConfiguration",
            98 => "cudaErrorInvalidDeviceFunction",
            100 => "cudaErrorNoDevice",
            101 => "cudaErrorInvalidDevice",
            200 => "cudaErrorInvalidKernelImage",
            201 => "cudaErrorDeviceUninitialized",
            209 => "cudaErrorNoKernelImageForDevice",
            214 => "cudaErrorECCUncorrectable",
            218 => "cudaErrorInvalidPtx",
            220 => "cudaErrorNvlinkUncorrectable",
            400 => "cudaErrorInvalidResourceHandle",
            500 => "cudaErrorSymbolNotFound",
            600 => "cudaErrorNotReady",
            700 => "cudaErrorIllegalAddress",
            701 => "cudaErrorLaunchOutOfResources",
            709 => "cudaErrorContextIsDestroyed",
            716 => "cudaErrorMisalignedAddress",
            719 => "cudaErrorLaunchFailure",
            800 => "cudaErrorNotPermitted",
            801 => "cudaErrorNotSupported",
            999 => "cudaErrorUnknown",
            _ => "cudaErrorUnrecognized",
        }
    }

    fn description(self) -> &'static str {
        match self.0 {
            0 => "no error",
            1 => "invalid argument",
            2 => "out of memory",
            3 => "initialization error",
            4 => "CUDA runtime is shutting down",
            98 => "invalid device function",
            100 => "no CUDA-capable device detected",
            101 => "invalid device ordinal",
            200 => "invalid kernel image",
            201 => "CUDA device has not been initialized",
            209 => "no kernel image available for this device",
            214 => "uncorrectable ECC error",
            218 => "invalid PTX",
            400 => "invalid resource handle",
            500 => "named symbol not found",
            600 => "operation not yet complete",
            700 => "illegal memory access",
            701 => "launch requires more resources than the device can provide",
            716 => "misaligned address",
            719 => "unspecified launch failure",
            800 => "operation not permitted",
            801 => "operation not supported",
            _ => "unrecognized CUDA runtime error code",
        }
    }

    fn is_success(self) -> bool {
        cudaError_t::is_success(self)
    }

    fn library(self) -> &'static str {
        "cuda-runtime"
    }
}
