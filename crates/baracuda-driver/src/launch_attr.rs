//! Typed builders for [`CUlaunchAttribute`] entries consumed by
//! [`crate::LaunchBuilder::launch_ex`] (CUDA 12.0+).
//!
//! CUDA's launch-attribute system is a tagged 80-byte entry — 4-byte
//! `id`, 4-byte pad, 64-byte payload union, 8-byte tail pad. Each
//! attribute ID picks out a different union slot (cluster dims,
//! priority, access-policy window, etc.). These builders give you a
//! Rust-native way to construct those entries without hand-writing byte
//! layouts.
//!
//! ```no_run
//! # use baracuda_driver::launch_attr::LaunchAttr;
//! let mut attrs = [
//!     LaunchAttr::priority(-5).into_raw(),
//!     LaunchAttr::cluster_dim(2, 1, 1).into_raw(),
//!     LaunchAttr::cooperative().into_raw(),
//! ];
//! # let _ = attrs;
//! ```

use baracuda_cuda_sys::types::{
    CUaccessPolicyWindow, CUlaunchAttribute, CUlaunchAttributeID, CUlaunchAttributeValue,
};
use baracuda_cuda_sys::CUevent;

pub use baracuda_cuda_sys::types::CUaccessPolicyWindow as AccessPolicyWindow;

/// Typed constructor for a launch-attribute entry.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct LaunchAttr(CUlaunchAttribute);

impl core::fmt::Debug for LaunchAttr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LaunchAttr")
            .field("id", &self.0.id)
            .finish_non_exhaustive()
    }
}

impl LaunchAttr {
    /// Consume this wrapper and return the raw `CUlaunchAttribute`. Pass
    /// the resulting array to [`crate::LaunchBuilder::launch_ex`].
    #[inline]
    pub fn into_raw(self) -> CUlaunchAttribute {
        self.0
    }

    #[inline]
    pub fn as_raw(&self) -> &CUlaunchAttribute {
        &self.0
    }

    fn empty(id: u32) -> Self {
        Self(CUlaunchAttribute {
            id,
            pad: [0; 4],
            value: CUlaunchAttributeValue([0u8; 64]),
        })
    }

    /// Write `value` at offset 0 of the payload. `T` must be `Copy` with
    /// `size_of::<T>() <= 64`.
    #[inline]
    fn with_value<T: Copy>(mut self, value: T) -> Self {
        assert!(
            core::mem::size_of::<T>() <= 64,
            "launch attribute payload too large ({} > 64)",
            core::mem::size_of::<T>()
        );
        // SAFETY: we asserted the payload fits, and CUlaunchAttributeValue
        // is `[u8; 64]` with alignment 1 — writing via write_unaligned is
        // well-defined.
        unsafe {
            let p = self.0.value.0.as_mut_ptr() as *mut T;
            p.write_unaligned(value);
        }
        self
    }

    /// 3-D thread-block cluster dimensions (Hopper+). Grid-in-clusters.
    pub fn cluster_dim(x: u32, y: u32, z: u32) -> Self {
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct ClusterDim {
            x: u32,
            y: u32,
            z: u32,
        }
        Self::empty(CUlaunchAttributeID::CLUSTER_DIMENSION).with_value(ClusterDim { x, y, z })
    }

    /// Per-launch priority override (lower = higher priority).
    pub fn priority(priority: i32) -> Self {
        Self::empty(CUlaunchAttributeID::PRIORITY).with_value(priority)
    }

    /// Mark the kernel as cooperative (thread-block-grid-wide sync).
    pub fn cooperative() -> Self {
        Self::empty(CUlaunchAttributeID::COOPERATIVE).with_value(1i32)
    }

    /// Opt in to programmatic stream serialization for this launch.
    pub fn programmatic_stream_serialization(enabled: bool) -> Self {
        Self::empty(CUlaunchAttributeID::PROGRAMMATIC_STREAM_SERIALIZATION).with_value(if enabled {
            1i32
        } else {
            0i32
        })
    }

    /// Hopper+ cluster scheduling policy — pass a value from
    /// `CUclusterSchedulingPolicy` (DEFAULT=0, SPREAD=1, LOAD_BALANCING=2).
    pub fn cluster_scheduling_policy(policy: i32) -> Self {
        Self::empty(CUlaunchAttributeID::CLUSTER_SCHEDULING_POLICY_PREFERENCE).with_value(policy)
    }

    /// Synchronization policy — pass a value from `CUsynchronizationPolicy`
    /// (AUTO=1, SPIN=2, YIELD=3, BLOCKING_SYNC=4).
    pub fn synchronization_policy(policy: i32) -> Self {
        Self::empty(CUlaunchAttributeID::SYNCHRONIZATION_POLICY).with_value(policy)
    }

    /// Access-policy window for L2 persistence hints. Reserve a region
    /// of memory with a target hit-ratio and per-hit / per-miss caching
    /// properties.
    pub fn access_policy_window(window: CUaccessPolicyWindow) -> Self {
        Self::empty(CUlaunchAttributeID::ACCESS_POLICY_WINDOW).with_value(window)
    }

    /// Programmatic event — records `event` at the specified trigger
    /// point during launch. `flags` is typically 0.
    /// `trigger_at_block_start` tells CUDA to trigger when each block
    /// starts (vs. the whole grid finishing).
    pub fn programmatic_event(event: CUevent, flags: i32, trigger_at_block_start: bool) -> Self {
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct ProgEvent {
            event: CUevent,
            flags: i32,
            trigger_at_block_start: i32,
        }
        Self::empty(CUlaunchAttributeID::PROGRAMMATIC_EVENT).with_value(ProgEvent {
            event,
            flags,
            trigger_at_block_start: if trigger_at_block_start { 1 } else { 0 },
        })
    }

    /// Launch-completion event — records `event` when all grid work is
    /// complete (lighter than recording after a full sync).
    pub fn launch_completion_event(event: CUevent, flags: i32) -> Self {
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct LaunchEvent {
            event: CUevent,
            flags: i32,
        }
        Self::empty(CUlaunchAttributeID::LAUNCH_COMPLETION_EVENT)
            .with_value(LaunchEvent { event, flags })
    }

    /// Raw attribute constructor — use this when CUDA adds a new
    /// attribute ID that baracuda hasn't yet typed.
    ///
    /// # Safety
    ///
    /// `payload` must match the layout CUDA expects for `id`. Wrong
    /// payloads yield UB at kernel launch (corrupt state, device fault).
    pub unsafe fn raw<T: Copy>(id: u32, payload: T) -> Self {
        Self::empty(id).with_value(payload)
    }
}

/// Convenience: convert a slice of typed attrs into an owned vector of
/// raw `CUlaunchAttribute`s suitable for `launch_ex`.
pub fn into_raw_vec(attrs: &[LaunchAttr]) -> Vec<CUlaunchAttribute> {
    attrs.iter().map(|a| a.into_raw()).collect()
}
