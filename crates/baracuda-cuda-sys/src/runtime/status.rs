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
    /// `Success` — success.
    pub const Success: Self = Self(0);
    /// `InvalidValue` — invalid value.
    pub const InvalidValue: Self = Self(1);
    /// `MemoryAllocation` — memory allocation.
    pub const MemoryAllocation: Self = Self(2);
    /// `InitializationError` — initialization error.
    pub const InitializationError: Self = Self(3);
    /// `CudartUnloading` — cudart unloading.
    pub const CudartUnloading: Self = Self(4);
    /// `ProfilerDisabled` — profiler disabled.
    pub const ProfilerDisabled: Self = Self(5);
    /// `InvalidConfiguration` — invalid configuration.
    pub const InvalidConfiguration: Self = Self(9);
    /// `InvalidPitchValue` — invalid pitch value.
    pub const InvalidPitchValue: Self = Self(12);
    /// `InvalidSymbol` — invalid symbol.
    pub const InvalidSymbol: Self = Self(13);
    /// `InvalidHostPointer` — invalid host pointer.
    pub const InvalidHostPointer: Self = Self(16);
    /// `InvalidDevicePointer` — invalid device pointer.
    pub const InvalidDevicePointer: Self = Self(17);
    /// `InvalidTexture` — invalid texture.
    pub const InvalidTexture: Self = Self(18);
    /// `InvalidDeviceFunction` — invalid device function.
    pub const InvalidDeviceFunction: Self = Self(98);
    /// `NoDevice` — no device.
    pub const NoDevice: Self = Self(100);
    /// `InvalidDevice` — invalid device.
    pub const InvalidDevice: Self = Self(101);
    /// `DeviceNotLicensed` — device not licensed.
    pub const DeviceNotLicensed: Self = Self(102);
    /// `SoftwareValidityNotEstablished` — software validity not established.
    pub const SoftwareValidityNotEstablished: Self = Self(103);
    /// `StartupFailure` — startup failure.
    pub const StartupFailure: Self = Self(127);
    /// `InvalidKernelImage` — invalid kernel image.
    pub const InvalidKernelImage: Self = Self(200);
    /// `DeviceUninitialized` — device uninitialized.
    pub const DeviceUninitialized: Self = Self(201);
    /// `MapBufferObjectFailed` — map buffer object failed.
    pub const MapBufferObjectFailed: Self = Self(205);
    /// `UnmapBufferObjectFailed` — unmap buffer object failed.
    pub const UnmapBufferObjectFailed: Self = Self(206);
    /// `ArrayIsMapped` — array is mapped.
    pub const ArrayIsMapped: Self = Self(207);
    /// `AlreadyMapped` — already mapped.
    pub const AlreadyMapped: Self = Self(208);
    /// `NoKernelImageForDevice` — no kernel image for device.
    pub const NoKernelImageForDevice: Self = Self(209);
    /// `AlreadyAcquired` — already acquired.
    pub const AlreadyAcquired: Self = Self(210);
    /// `NotMapped` — not mapped.
    pub const NotMapped: Self = Self(211);
    /// `ECCUncorrectable` — ecc uncorrectable.
    pub const ECCUncorrectable: Self = Self(214);
    /// `UnsupportedLimit` — unsupported limit.
    pub const UnsupportedLimit: Self = Self(215);
    /// `DeviceAlreadyInUse` — device already in use.
    pub const DeviceAlreadyInUse: Self = Self(216);
    /// `PeerAccessUnsupported` — peer access unsupported.
    pub const PeerAccessUnsupported: Self = Self(217);
    /// `InvalidPtx` — invalid ptx.
    pub const InvalidPtx: Self = Self(218);
    /// `InvalidGraphicsContext` — invalid graphics context.
    pub const InvalidGraphicsContext: Self = Self(219);
    /// `NvlinkUncorrectable` — nvlink uncorrectable.
    pub const NvlinkUncorrectable: Self = Self(220);
    /// `JitCompilerNotFound` — jit compiler not found.
    pub const JitCompilerNotFound: Self = Self(221);
    /// `UnsupportedPtxVersion` — unsupported ptx version.
    pub const UnsupportedPtxVersion: Self = Self(222);
    /// `InvalidSource` — invalid source.
    pub const InvalidSource: Self = Self(300);
    /// `FileNotFound` — file not found.
    pub const FileNotFound: Self = Self(301);
    /// `SharedObjectSymbolNotFound` — shared object symbol not found.
    pub const SharedObjectSymbolNotFound: Self = Self(302);
    /// `SharedObjectInitFailed` — shared object init failed.
    pub const SharedObjectInitFailed: Self = Self(303);
    /// `OperatingSystem` — operating system.
    pub const OperatingSystem: Self = Self(304);
    /// `InvalidResourceHandle` — invalid resource handle.
    pub const InvalidResourceHandle: Self = Self(400);
    /// `IllegalState` — illegal state.
    pub const IllegalState: Self = Self(401);
    /// `SymbolNotFound` — symbol not found.
    pub const SymbolNotFound: Self = Self(500);
    /// `NotReady` — not ready.
    pub const NotReady: Self = Self(600);
    /// `IllegalAddress` — illegal address.
    pub const IllegalAddress: Self = Self(700);
    /// `LaunchOutOfResources` — launch out of resources.
    pub const LaunchOutOfResources: Self = Self(701);
    /// `LaunchTimeout` — launch timeout.
    pub const LaunchTimeout: Self = Self(702);
    /// `PrimaryContextActive` — primary context active.
    pub const PrimaryContextActive: Self = Self(708);
    /// `ContextIsDestroyed` — context is destroyed.
    pub const ContextIsDestroyed: Self = Self(709);
    /// `Assert` — assert.
    pub const Assert: Self = Self(710);
    /// `MisalignedAddress` — misaligned address.
    pub const MisalignedAddress: Self = Self(716);
    /// `LaunchFailure` — launch failure.
    pub const LaunchFailure: Self = Self(719);
    /// `CooperativeLaunchTooLarge` — cooperative launch too large.
    pub const CooperativeLaunchTooLarge: Self = Self(720);
    /// `NotPermitted` — not permitted.
    pub const NotPermitted: Self = Self(800);
    /// `NotSupported` — not supported.
    pub const NotSupported: Self = Self(801);
    /// `SystemNotReady` — system not ready.
    pub const SystemNotReady: Self = Self(802);
    /// `SystemDriverMismatch` — system driver mismatch.
    pub const SystemDriverMismatch: Self = Self(803);
    /// `CompatNotSupportedOnDevice` — compat not supported on device.
    pub const CompatNotSupportedOnDevice: Self = Self(804);
    /// `StreamCaptureUnsupported` — stream capture unsupported.
    pub const StreamCaptureUnsupported: Self = Self(900);
    /// `StreamCaptureInvalidated` — stream capture invalidated.
    pub const StreamCaptureInvalidated: Self = Self(901);
    /// `StreamCaptureMerge` — stream capture merge.
    pub const StreamCaptureMerge: Self = Self(902);
    /// `StreamCaptureUnmatched` — stream capture unmatched.
    pub const StreamCaptureUnmatched: Self = Self(903);
    /// `StreamCaptureUnjoined` — stream capture unjoined.
    pub const StreamCaptureUnjoined: Self = Self(904);
    /// `StreamCaptureIsolation` — stream capture isolation.
    pub const StreamCaptureIsolation: Self = Self(905);
    /// `StreamCaptureImplicit` — stream capture implicit.
    pub const StreamCaptureImplicit: Self = Self(906);
    /// `CapturedEvent` — captured event.
    pub const CapturedEvent: Self = Self(907);
    /// `StreamCaptureWrongThread` — stream capture wrong thread.
    pub const StreamCaptureWrongThread: Self = Self(908);
    /// `Timeout` — timeout.
    pub const Timeout: Self = Self(909);
    /// `GraphExecUpdateFailure` — graph exec update failure.
    pub const GraphExecUpdateFailure: Self = Self(910);
    /// `Unknown` — unknown.
    pub const Unknown: Self = Self(999);

    /// `is_success` — is success.
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
