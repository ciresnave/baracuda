//! Device enumeration + queries via the Runtime API.
//!
//! Unlike [`baracuda_driver::Device`], a [`Device`] in the Runtime API is
//! just an ordinal — there's no separate `CUdevice` handle. Contexts are
//! implicit (the "primary context" per device) and are shared with the
//! Driver API on the same device.

use baracuda_cuda_sys::runtime::{runtime, types::cudaDeviceAttr as Attr};

use crate::error::{check, Result};

/// A CUDA device (Runtime API view — a bare ordinal).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Device(pub(crate) i32);

impl Device {
    /// Number of CUDA devices visible to the process.
    pub fn count() -> Result<u32> {
        let r = runtime()?;
        let cu = r.cuda_get_device_count()?;
        let mut n: core::ffi::c_int = 0;
        check(unsafe { cu(&mut n) })?;
        Ok(n as u32)
    }

    /// Construct a `Device` for the given ordinal. Does not validate — use
    /// [`Device::all`] if you want a checked enumeration.
    #[inline]
    pub const fn from_ordinal(ordinal: u32) -> Self {
        Self(ordinal as i32)
    }

    /// All visible devices, in ordinal order.
    pub fn all() -> Result<Vec<Self>> {
        let count = Self::count()?;
        Ok((0..count).map(Self::from_ordinal).collect())
    }

    /// Set this device as current on the calling thread. Subsequent Runtime
    /// API calls (allocations, launches, ...) operate on this device.
    pub fn set_current(&self) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_set_device()?;
        check(unsafe { cu(self.0) })
    }

    /// Retrieve the device currently selected on the calling thread.
    pub fn current() -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_get_device()?;
        let mut dev: core::ffi::c_int = 0;
        check(unsafe { cu(&mut dev) })?;
        Ok(Self(dev))
    }

    /// Ordinal of this device (`0`, `1`, ...).
    #[inline]
    pub fn ordinal(&self) -> i32 {
        self.0
    }

    /// Compute capability as `(major, minor)`.
    pub fn compute_capability(&self) -> Result<(u32, u32)> {
        Ok((
            self.attribute(Attr::COMPUTE_CAPABILITY_MAJOR)? as u32,
            self.attribute(Attr::COMPUTE_CAPABILITY_MINOR)? as u32,
        ))
    }

    /// Multiprocessor count.
    pub fn multiprocessor_count(&self) -> Result<u32> {
        Ok(self.attribute(Attr::MULTIPROCESSOR_COUNT)? as u32)
    }

    /// Warp size in threads.
    pub fn warp_size(&self) -> Result<u32> {
        Ok(self.attribute(Attr::WARP_SIZE)? as u32)
    }

    /// Raw device-attribute query. See [`baracuda_cuda_sys::runtime::types::cudaDeviceAttr`].
    pub fn attribute(&self, attr: i32) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_device_get_attribute()?;
        let mut val: core::ffi::c_int = 0;
        check(unsafe { cu(&mut val, attr, self.0) })?;
        Ok(val)
    }

    /// `true` if this device can peer-access `peer`'s allocations (P2P).
    /// Call [`Device::enable_peer_access`] before actually using peer
    /// pointers in kernels.
    pub fn can_access_peer(&self, peer: &Device) -> Result<bool> {
        let r = runtime()?;
        let cu = r.cuda_device_can_access_peer()?;
        let mut v: core::ffi::c_int = 0;
        check(unsafe { cu(&mut v, self.0, peer.0) })?;
        Ok(v != 0)
    }

    /// Enable peer access from the *current* device to `peer`'s
    /// allocations. Call `Device::set_current()` on the accessing device
    /// first.
    pub fn enable_peer_access(peer: &Device) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_device_enable_peer_access()?;
        check(unsafe { cu(peer.0, 0) })
    }

    /// Disable peer access previously enabled via
    /// [`Device::enable_peer_access`].
    pub fn disable_peer_access(peer: &Device) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_device_disable_peer_access()?;
        check(unsafe { cu(peer.0) })
    }
}
