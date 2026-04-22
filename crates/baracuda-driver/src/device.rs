//! Physical-GPU query and enumeration.

use core::ffi::c_char;

use baracuda_cuda_sys::types::CUdevice_attribute as Attr;
use baracuda_cuda_sys::{driver, CUdevice};

use crate::error::{check, Result};
use crate::init::init;

/// A CUDA device (a physical GPU, or a logical slice of one under MIG).
///
/// Cheap `Copy` type — it's just an ordinal.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Device(pub(crate) CUdevice);

impl Device {
    /// Number of CUDA devices visible to the process.
    pub fn count() -> Result<u32> {
        init()?;
        let d = driver()?;
        let cu = d.cu_device_get_count()?;
        let mut n: core::ffi::c_int = 0;
        // SAFETY: `out` points to a writable i32.
        check(unsafe { cu(&mut n) })?;
        Ok(n as u32)
    }

    /// Retrieve the device with the given ordinal.
    pub fn get(ordinal: u32) -> Result<Self> {
        init()?;
        let d = driver()?;
        let cu = d.cu_device_get()?;
        let mut dev = CUdevice::default();
        // SAFETY: `dev` points to a writable CUdevice; the cast is widening on 64-bit.
        check(unsafe { cu(&mut dev, ordinal as core::ffi::c_int) })?;
        Ok(Self(dev))
    }

    /// All visible devices, in ordinal order.
    pub fn all() -> Result<Vec<Self>> {
        let count = Self::count()?;
        (0..count).map(Self::get).collect()
    }

    /// Raw ordinal (`0`, `1`, ...).
    #[inline]
    pub fn ordinal(&self) -> i32 {
        self.0 .0
    }

    /// Human-readable name, e.g. `"NVIDIA GeForce RTX 4090"`.
    pub fn name(&self) -> Result<String> {
        let d = driver()?;
        let cu = d.cu_device_get_name()?;
        let mut buf = vec![0u8; 256];
        // SAFETY: `buf` is valid for writes of `buf.len()` bytes; the
        // function is documented to null-terminate within the buffer.
        check(unsafe {
            cu(
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as core::ffi::c_int,
                self.0,
            )
        })?;
        let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..nul]).into_owned())
    }

    /// Total global memory on this device, in bytes.
    pub fn total_memory(&self) -> Result<u64> {
        let d = driver()?;
        let cu = d.cu_device_total_mem()?;
        let mut bytes: usize = 0;
        // SAFETY: writable pointer to `usize`; CUDA writes `size_t`.
        check(unsafe { cu(&mut bytes, self.0) })?;
        Ok(bytes as u64)
    }

    /// Compute capability as `(major, minor)`, e.g. `(9, 0)` for Hopper.
    pub fn compute_capability(&self) -> Result<(u32, u32)> {
        let major = self.attribute(Attr::COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self.attribute(Attr::COMPUTE_CAPABILITY_MINOR)?;
        Ok((major as u32, minor as u32))
    }

    /// Multiprocessor count (SM count).
    pub fn multiprocessor_count(&self) -> Result<u32> {
        Ok(self.attribute(Attr::MULTIPROCESSOR_COUNT)? as u32)
    }

    /// Warp size in threads (almost always 32).
    pub fn warp_size(&self) -> Result<u32> {
        Ok(self.attribute(Attr::WARP_SIZE)? as u32)
    }

    /// Query an arbitrary `CUdevice_attribute`. See
    /// [`baracuda_cuda_sys::types::CUdevice_attribute`] for the full list.
    pub fn attribute(&self, attr: i32) -> Result<i32> {
        let d = driver()?;
        let cu = d.cu_device_get_attribute()?;
        let mut val: core::ffi::c_int = 0;
        // SAFETY: writable i32; `attr` is a valid CUDA attribute selector
        // (caller supplied, but CUDA returns an error for invalid selectors
        // rather than UB).
        check(unsafe { cu(&mut val, attr, self.0) })?;
        Ok(val)
    }

    /// The raw `CUdevice` handle. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUdevice {
        self.0
    }

    /// Return the device's 16-byte UUID.
    pub fn uuid(&self) -> Result<[u8; 16]> {
        let d = driver()?;
        let cu = d.cu_device_get_uuid()?;
        let mut out = [0u8; 16];
        check(unsafe { cu(out.as_mut_ptr(), self.0) })?;
        Ok(out)
    }

    /// Return the device's Windows LUID and 32-bit device-node mask
    /// (Windows only; Linux returns zeros).
    pub fn luid(&self) -> Result<([u8; 8], u32)> {
        let d = driver()?;
        let cu = d.cu_device_get_luid()?;
        let mut luid = [0i8; 8];
        let mut mask: core::ffi::c_uint = 0;
        check(unsafe { cu(luid.as_mut_ptr(), &mut mask, self.0) })?;
        Ok((luid.map(|b| b as u8), mask))
    }

    /// Query a peer-to-peer attribute between `self` (as source) and
    /// `peer` (as destination). Pass a constant from
    /// [`baracuda_cuda_sys::types::CUdevice_P2PAttribute`].
    pub fn p2p_attribute(&self, peer: &Device, attr: i32) -> Result<i32> {
        let d = driver()?;
        let cu = d.cu_device_get_p2p_attribute()?;
        let mut v: core::ffi::c_int = 0;
        check(unsafe { cu(&mut v, attr, self.0, peer.0) })?;
        Ok(v)
    }

    /// Query whether this device supports a given exec-affinity type
    /// (e.g. SM-count partitioning at context-creation time).
    pub fn exec_affinity_support(&self, affinity_type: i32) -> Result<bool> {
        let d = driver()?;
        let cu = d.cu_device_get_exec_affinity_support()?;
        let mut v: core::ffi::c_int = 0;
        check(unsafe { cu(&mut v, affinity_type, self.0) })?;
        Ok(v != 0)
    }

    /// `true` if this device can directly access allocations on `peer`.
    /// Peer access still requires a matching `Context::enable_peer_access`
    /// on the accessing side before kernels can dereference peer pointers.
    pub fn can_access_peer(&self, peer: &Device) -> Result<bool> {
        let d = driver()?;
        let cu = d.cu_device_can_access_peer()?;
        let mut out: core::ffi::c_int = 0;
        check(unsafe { cu(&mut out, self.0, peer.0) })?;
        Ok(out != 0)
    }

    /// Query the primary-context state for this device.
    /// Returns `(flags, active)` — `flags` is the same bitmask
    /// [`crate::Context::with_flags`] takes, `active` is `true` if some
    /// caller currently holds a retained primary-context reference.
    pub fn primary_ctx_state(&self) -> Result<(u32, bool)> {
        let d = driver()?;
        let cu = d.cu_device_primary_ctx_get_state()?;
        let mut flags: core::ffi::c_uint = 0;
        let mut active: core::ffi::c_int = 0;
        check(unsafe { cu(self.0, &mut flags, &mut active) })?;
        Ok((flags, active != 0))
    }

    /// Set the flags used when the primary context is later created.
    /// Returns `CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE` if the primary context
    /// already exists; reset with `Context::reset_primary` first.
    pub fn set_primary_ctx_flags(&self, flags: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_device_primary_ctx_set_flags()?;
        check(unsafe { cu(self.0, flags) })
    }
}
