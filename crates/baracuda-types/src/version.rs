//! CUDA version identifier + feature-gating enum.
//!
//! `CudaVersion` uses the same packed encoding NVIDIA uses in
//! `cuDriverGetVersion` / `cudaRuntimeGetVersion`: `major * 1000 + minor * 10`
//! — i.e. CUDA 12.6.0 is encoded as `12060`.

use core::fmt;

/// A CUDA major.minor version.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct CudaVersion(u32);

impl CudaVersion {
    /// CUDA 11.4 — the baracuda floor.
    pub const CUDA_11_4: Self = Self::from_major_minor(11, 4);
    /// CUDA 11.8 — last 11.x release.
    pub const CUDA_11_8: Self = Self::from_major_minor(11, 8);
    /// CUDA 12.0 — introduced library management, green contexts, multicast, nvJitLink.
    pub const CUDA_12_0: Self = Self::from_major_minor(12, 0);
    /// CUDA 12.3 — introduced graph conditional nodes.
    pub const CUDA_12_3: Self = Self::from_major_minor(12, 3);
    /// CUDA 12.6.
    pub const CUDA_12_6: Self = Self::from_major_minor(12, 6);
    /// CUDA 12.8 — introduced graph SWITCH nodes.
    pub const CUDA_12_8: Self = Self::from_major_minor(12, 8);
    /// CUDA 13.0.
    pub const CUDA_13_0: Self = Self::from_major_minor(13, 0);

    /// The baracuda floor: the oldest CUDA driver/toolkit we officially support.
    pub const FLOOR: Self = Self::CUDA_11_4;

    /// Construct a version from major.minor parts (e.g. `12, 6`).
    #[inline]
    pub const fn from_major_minor(major: u32, minor: u32) -> Self {
        Self(major * 1000 + minor * 10)
    }

    /// Decode the raw integer returned by `cuDriverGetVersion` /
    /// `cudaRuntimeGetVersion`.
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// The packed integer form (major\*1000 + minor\*10).
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Major version number (e.g. 12 for CUDA 12.6).
    #[inline]
    pub const fn major(self) -> u32 {
        self.0 / 1000
    }

    /// Minor version number (e.g. 6 for CUDA 12.6).
    #[inline]
    pub const fn minor(self) -> u32 {
        (self.0 % 1000) / 10
    }

    /// `true` iff this version is ≥ `major.minor`.
    #[inline]
    pub const fn at_least(self, major: u32, minor: u32) -> bool {
        self.0 >= major * 1000 + minor * 10
    }
}

impl fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA {}.{}", self.major(), self.minor())
    }
}

/// Named CUDA capabilities, each gated by the minimum toolkit/driver version
/// in which they became available. Safe-API crates call [`supports`] before
/// invoking the underlying symbol and surface `Error::FeatureNotSupported`
/// when the installed driver is too old.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Feature {
    /// Stream-ordered memory allocator (`cuMemAllocAsync`, pools). CUDA 11.2+.
    StreamOrderedAllocator,
    /// Virtual memory management (`cuMemCreate`, `cuMemMap`, ...). CUDA 10.2+.
    VirtualMemoryManagement,
    /// Context-independent library loading (`cuLibraryLoadData`). CUDA 12.0+.
    LibraryManagement,
    /// Lightweight SM-partitioned contexts. CUDA 12.0+.
    GreenContexts,
    /// Multicast objects across multiple GPUs. CUDA 12.0+.
    MulticastObjects,
    /// Graph conditional (IF/WHILE) nodes. CUDA 12.3+.
    GraphConditionalNodes,
    /// Graph SWITCH nodes. CUDA 12.8+.
    GraphSwitchNodes,
    /// Extensible kernel launch (`cuLaunchKernelEx`, cluster dims). CUDA 12.0+.
    CudaLaunchKernelEx,
    /// nvJitLink modern JIT linker. CUDA 12.0+.
    NvJitLink,
    /// Hopper-only TMA tensor map descriptors. CUDA 11.8+ (HW 9.0+).
    TensorMapObjects,
    /// Runtime-API log buffer (`cudaLogs*`). CUDA 12.0+.
    RuntimeLogBuffer,
    /// `cudaInitDevice` (initialize primary context without making it current). CUDA 12.0+.
    CudaInitDevice,
    /// Green-context-aware execution context APIs on the runtime. CUDA 13.1+.
    RuntimeGreenContexts,
}

impl Feature {
    /// The minimum CUDA version at which the feature is available.
    pub const fn required_version(self) -> CudaVersion {
        match self {
            Feature::StreamOrderedAllocator => CudaVersion::from_major_minor(11, 2),
            Feature::VirtualMemoryManagement => CudaVersion::from_major_minor(10, 2),
            Feature::LibraryManagement => CudaVersion::CUDA_12_0,
            Feature::GreenContexts => CudaVersion::CUDA_12_0,
            Feature::MulticastObjects => CudaVersion::CUDA_12_0,
            Feature::GraphConditionalNodes => CudaVersion::CUDA_12_3,
            Feature::GraphSwitchNodes => CudaVersion::CUDA_12_8,
            Feature::CudaLaunchKernelEx => CudaVersion::CUDA_12_0,
            Feature::NvJitLink => CudaVersion::CUDA_12_0,
            Feature::TensorMapObjects => CudaVersion::from_major_minor(11, 8),
            Feature::RuntimeLogBuffer => CudaVersion::CUDA_12_0,
            Feature::CudaInitDevice => CudaVersion::CUDA_12_0,
            Feature::RuntimeGreenContexts => CudaVersion::from_major_minor(13, 1),
        }
    }
}

/// `true` if `feature` is callable on a CUDA installation of version `version`
/// or newer.
#[inline]
pub const fn supports(version: CudaVersion, feature: Feature) -> bool {
    version.raw() >= feature.required_version().raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode() {
        let v = CudaVersion::from_major_minor(12, 6);
        assert_eq!(v.major(), 12);
        assert_eq!(v.minor(), 6);
        assert_eq!(v.raw(), 12060);
    }

    #[test]
    fn ordering_is_by_version() {
        assert!(CudaVersion::CUDA_11_4 < CudaVersion::CUDA_12_0);
        assert!(CudaVersion::CUDA_12_0 < CudaVersion::CUDA_13_0);
    }

    #[test]
    fn at_least() {
        assert!(CudaVersion::CUDA_12_6.at_least(12, 0));
        assert!(!CudaVersion::CUDA_11_4.at_least(12, 0));
    }

    #[test]
    fn feature_gating() {
        assert!(supports(CudaVersion::CUDA_12_0, Feature::GreenContexts));
        assert!(!supports(CudaVersion::CUDA_11_8, Feature::GreenContexts));
        assert!(supports(CudaVersion::CUDA_12_8, Feature::GraphSwitchNodes));
        assert!(!supports(CudaVersion::CUDA_12_6, Feature::GraphSwitchNodes));
    }

    #[test]
    fn floor_is_consistent() {
        assert_eq!(CudaVersion::FLOOR, CudaVersion::CUDA_11_4);
    }
}
