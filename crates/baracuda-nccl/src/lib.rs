//! Safe Rust wrappers for NVIDIA NCCL (multi-GPU collective communication).
//!
//! Layered on top of [`baracuda-nccl-sys`](https://docs.rs/baracuda-nccl-sys).
//! Use this crate directly for typed, RAII-managed communicators +
//! collectives; reach for `-sys` only when adding a function the safe
//! layer doesn't expose yet.
//!
//! ## Scope
//!
//! - **Communicator lifecycle**: single-process multi-GPU via
//!   `ncclCommInitAll`, multi-process via `ncclCommInitRank` +
//!   `UniqueId` exchange, communicator destruction, error querying.
//! - **All collectives**: `all_reduce`, `all_gather`, `broadcast`,
//!   `reduce`, `reduce_scatter`, `send` / `recv` (point-to-point used
//!   by `baracuda-kernels`'s Ring Attention K/V chunk rotation).
//! - **Group operations**: `group_start` / `group_end` for batched
//!   collective launches (essential for Megatron-LM TP all-reduce
//!   patterns).
//! - **Communicator features**: stream binding, error checking,
//!   abort + finalize for graceful shutdown.
//! - **Datatype helpers**: f32/f64/f16/bf16/u8/i32/i64/u64 reduction
//!   support via `DataType` enum.
//! - **Reduction ops**: `sum`, `prod`, `min`, `max`, `avg`,
//!   `pre_mul_sum`.
//!
//! ## Platform support
//!
//! NCCL is primarily a **Linux library**. Windows has experimental
//! support in newer NCCL versions but is uncommon. The crate compiles
//! on Windows but [`Communicator::init_all`] returns
//! `LoaderError::LibraryNotFound` at runtime on hosts without NCCL —
//! single-device callers can detect this and fall back gracefully.
//!
//! ## Usage with baracuda-kernels
//!
//! `baracuda-kernels`'s Ring Attention plan (Phase 56,
//! `ring_attention` feature) and Megatron-LM TP primitives (Phase 57,
//! `megatron_tp` feature) both consume this crate. Direct callers
//! commonly use the communicator for synchronous data-parallel
//! training all-reduce.

#![warn(missing_debug_implementations)]

use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_nccl_sys::{
    nccl, ncclComm_t, ncclDataType_t, ncclRedOp_t, ncclResult_t, ncclUniqueId,
};
use baracuda_types::DeviceRepr;

/// Error type for NCCL operations.
pub type Error = baracuda_core::Error<ncclResult_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: ncclResult_t) -> Result<()> {
    Error::check(status)
}

/// Reduction operation for `all_reduce` / `reduce`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum RedOp {
    /// Element-wise sum across ranks.
    #[default]
    Sum,
    /// Element-wise product across ranks.
    Prod,
    /// Element-wise maximum across ranks.
    Max,
    /// Element-wise minimum across ranks.
    Min,
    /// Arithmetic mean. NCCL 2.10+.
    Avg,
    /// Custom op id returned by [`Communicator::create_pre_mul_sum`].
    /// NCCL 2.11+.
    Custom(i32),
}

impl RedOp {
    fn raw(self) -> ncclRedOp_t {
        match self {
            RedOp::Sum => ncclRedOp_t::Sum,
            RedOp::Prod => ncclRedOp_t::Prod,
            RedOp::Max => ncclRedOp_t::Max,
            RedOp::Min => ncclRedOp_t::Min,
            RedOp::Avg => ncclRedOp_t::Avg,
            RedOp::Custom(id) => ncclRedOp_t(id),
        }
    }
}

/// Where the scalar passed to [`Communicator::create_pre_mul_sum`] lives.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ScalarResidence {
    /// Scalar pointer is in host memory; NCCL captures the value at call time.
    Host = 0,
    /// Scalar pointer is in device memory; NCCL captures it at collective launch.
    Device = 1,
}

/// Element type for NCCL buffers. Implemented by baracuda-types primitives
/// via a sealed trait.
pub trait NcclScalar: DeviceRepr + sealed::Sealed {
    #[doc(hidden)]
    fn raw() -> ncclDataType_t;
}

macro_rules! impl_nccl_scalar {
    ($ty:ty, $variant:ident) => {
        impl NcclScalar for $ty {
            fn raw() -> ncclDataType_t {
                ncclDataType_t::$variant
            }
        }
        impl sealed::Sealed for $ty {}
    };
}

impl_nccl_scalar!(i8, Int8);
impl_nccl_scalar!(u8, Uint8);
impl_nccl_scalar!(i32, Int32);
impl_nccl_scalar!(u32, Uint32);
impl_nccl_scalar!(i64, Int64);
impl_nccl_scalar!(u64, Uint64);
impl_nccl_scalar!(f32, Float32);
impl_nccl_scalar!(f64, Float64);

// Half-precision types from the `half` crate. Gated on `half-crate`
// (which transitively pulls in `baracuda-types/half-crate` so the
// `DeviceRepr` supertrait is already satisfied).
#[cfg(feature = "half-crate")]
impl_nccl_scalar!(half::f16, Float16);
#[cfg(feature = "half-crate")]
impl_nccl_scalar!(half::bf16, BFloat16);

mod sealed {
    /// Seal so only baracuda-authorized types implement `NcclScalar`.
    /// Extra impls under feature gates are added directly on the sealed
    /// trait in the parent module via `impl_nccl_scalar!`.
    pub trait Sealed {}
}

#[cfg(all(test, feature = "half-crate"))]
mod half_scalar_tests {
    use super::*;

    #[test]
    fn half_types_are_nccl_scalars() {
        fn require_scalar<T: NcclScalar>() -> ncclDataType_t {
            T::raw()
        }
        assert_eq!(
            require_scalar::<half::f16>(),
            ncclDataType_t::Float16,
            "half::f16 must map to ncclFloat16"
        );
        assert_eq!(
            require_scalar::<half::bf16>(),
            ncclDataType_t::BFloat16,
            "half::bf16 must map to ncclBfloat16"
        );
    }
}

/// A 128-byte opaque identifier for establishing a multi-process NCCL
/// communicator. One process calls [`UniqueId::new`] and distributes the
/// bytes to all other processes via a user-provided channel (TCP, MPI, …);
/// every process then calls [`Communicator::init_rank`] with the same id.
#[derive(Copy, Clone, Debug)]
pub struct UniqueId(ncclUniqueId);

impl UniqueId {
    /// Generate a fresh unique id on this process.
    pub fn new() -> Result<Self> {
        let n = nccl()?;
        let cu = n.nccl_get_unique_id()?;
        let mut id = ncclUniqueId::default();
        check(unsafe { cu(&mut id) })?;
        Ok(Self(id))
    }

    /// Raw 128-byte representation. Transmit over the wire as-is.
    pub fn as_bytes(&self) -> [u8; 128] {
        let mut out = [0u8; 128];
        for (o, b) in out.iter_mut().zip(&self.0.internal) {
            *o = *b as u8;
        }
        out
    }

    /// Rebuild from the 128 bytes received from another process.
    pub fn from_bytes(bytes: [u8; 128]) -> Self {
        let mut id = ncclUniqueId::default();
        for (i, b) in id.internal.iter_mut().zip(&bytes) {
            *i = *b as i8;
        }
        Self(id)
    }
}

/// A NCCL communicator — one rank's view of a distributed group.
///
/// Holds the NCCL handle plus cached `rank` and `world_size` (read once
/// from NCCL at construction). The cached form keeps the common
/// `comm.rank()` / `comm.world_size()` calls infallible — both values
/// are immutable for the lifetime of a communicator.
pub struct Communicator {
    handle: ncclComm_t,
    rank: i32,
    world_size: i32,
}

unsafe impl Send for Communicator {}

impl core::fmt::Debug for Communicator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("nccl::Communicator")
            .field("handle", &self.handle)
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish()
    }
}

impl Communicator {
    /// Construct a single-process / single-GPU communicator (rank 0 of 1).
    /// Useful for exercising the NCCL API surface on hosts with only one
    /// GPU — collectives degenerate to no-ops but exercise the full
    /// loader / dispatch path.
    pub fn new_single_gpu(device: i32) -> Result<Self> {
        let mut comms = Self::init_all(&[device])?;
        // `init_all(&[device])` asks NCCL for exactly one communicator;
        // on success NCCL fills the slot — pop is infallible in practice.
        Ok(comms.pop().expect(
            "ncclCommInitAll returned Success but produced no communicators",
        ))
    }

    /// Multi-process initialization. `id` is a 128-byte unique identifier
    /// generated on rank 0 via [`UniqueId::new`] / [`NcclUniqueId::generate`]
    /// and broadcast (over MPI / TCP / a shared file / …) to every other
    /// rank before they call this constructor.
    pub fn new_with_id(id: UniqueId, world_size: i32, rank: i32) -> Result<Self> {
        Self::init_rank(world_size, id, rank)
    }

    /// Initialize `ndev` communicators (one per device) in this process.
    /// The returned vector is ordered to match `devices`.
    ///
    /// This is the single-process "data-parallel on local GPUs" path.
    pub fn init_all(devices: &[i32]) -> Result<Vec<Self>> {
        let n = nccl()?;
        let cu = n.nccl_comm_init_all()?;
        let ndev = devices.len() as core::ffi::c_int;
        let mut comms = vec![core::ptr::null_mut::<core::ffi::c_void>(); devices.len()];
        check(unsafe { cu(comms.as_mut_ptr(), ndev, devices.as_ptr()) })?;
        comms
            .into_iter()
            .map(|handle| Self::from_raw_handle(handle))
            .collect()
    }

    /// Initialize one rank of a multi-process communicator.
    pub fn init_rank(nranks: i32, id: UniqueId, rank: i32) -> Result<Self> {
        let n = nccl()?;
        let cu = n.nccl_comm_init_rank()?;
        let mut handle: ncclComm_t = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, nranks, id.0, rank) })?;
        Self::from_raw_handle(handle)
    }

    /// Like [`Self::init_rank`] but takes a pointer to a configured
    /// `ncclConfig_t`. NCCL 2.13+. Pass `core::ptr::null_mut()` for
    /// defaults — equivalent to [`Self::init_rank`]. The struct shape
    /// (blocking flag, CGA cluster size, splitShare, netName, …)
    /// changes between NCCL versions, so we don't model it as a typed
    /// Rust struct; build it through the C API or as a `[u8; N]`.
    ///
    /// # Safety
    ///
    /// `config` must be a properly-initialized `ncclConfig_t` for the
    /// installed NCCL version, or null.
    pub unsafe fn init_rank_config(
        nranks: i32,
        id: UniqueId,
        rank: i32,
        config: *mut core::ffi::c_void,
    ) -> Result<Self> { unsafe {
        let n = nccl()?;
        let cu = n.nccl_comm_init_rank_config()?;
        let mut handle: ncclComm_t = core::ptr::null_mut();
        check(cu(&mut handle, nranks, id.0, rank, config))?;
        Self::from_raw_handle(handle)
    }}

    /// Wrap a raw `ncclComm_t` produced by NCCL itself (e.g. after
    /// `ncclCommSplit`). Reads rank / world_size from the handle.
    fn from_raw_handle(handle: ncclComm_t) -> Result<Self> {
        let n = nccl()?;
        let cu_rank = n.nccl_comm_user_rank()?;
        let cu_count = n.nccl_comm_count()?;
        let mut rank: core::ffi::c_int = 0;
        let mut count: core::ffi::c_int = 0;
        check(unsafe { cu_rank(handle, &mut rank) })?;
        check(unsafe { cu_count(handle, &mut count) })?;
        Ok(Self {
            handle,
            rank,
            world_size: count,
        })
    }

    /// This rank's index within the communicator (0..world_size).
    /// Cached at construction — never re-queries NCCL.
    #[inline]
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Total number of ranks in the communicator.
    /// Cached at construction — never re-queries NCCL.
    #[inline]
    pub fn world_size(&self) -> i32 {
        self.world_size
    }

    /// Deprecated alias for [`Self::world_size`]. Returns a `Result` for
    /// source-compatibility with the pre-Phase-52 API; never errors.
    #[deprecated(note = "Use `world_size()` (infallible, cached)")]
    pub fn nranks(&self) -> Result<i32> {
        Ok(self.world_size)
    }

    /// Raw communicator handle. Use with care.
    #[inline]
    pub fn as_raw(&self) -> ncclComm_t {
        self.handle
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        if let Ok(n) = nccl() {
            if let Ok(cu) = n.nccl_comm_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// All-reduce: each rank sends `send` and receives the per-element
/// reduction (across every rank) into `recv`. In-place use (`send == recv`) is legal.
#[allow(clippy::too_many_arguments)]
pub fn all_reduce<T: NcclScalar>(
    send: &DeviceBuffer<T>,
    recv: &mut DeviceBuffer<T>,
    count: usize,
    op: RedOp,
    comm: &Communicator,
    stream: &Stream,
) -> Result<()> {
    assert!(send.len() >= count && recv.len() >= count);
    let n = nccl()?;
    let cu = n.nccl_all_reduce()?;
    check(unsafe {
        cu(
            send.as_raw().0 as *const core::ffi::c_void,
            recv.as_raw().0 as *mut core::ffi::c_void,
            count,
            T::raw(),
            op.raw(),
            comm.handle,
            stream.as_raw() as _,
        )
    })
}

/// Broadcast the data at `root`'s `send` buffer to every rank's `recv` buffer.
pub fn broadcast<T: NcclScalar>(
    send: &DeviceBuffer<T>,
    recv: &mut DeviceBuffer<T>,
    count: usize,
    root: i32,
    comm: &Communicator,
    stream: &Stream,
) -> Result<()> {
    let n = nccl()?;
    let cu = n.nccl_broadcast()?;
    check(unsafe {
        cu(
            send.as_raw().0 as *const core::ffi::c_void,
            recv.as_raw().0 as *mut core::ffi::c_void,
            count,
            T::raw(),
            root,
            comm.handle,
            stream.as_raw() as _,
        )
    })
}

/// Begin a group of collectives that must be submitted atomically (e.g.
/// in single-process multi-GPU all-reduce).
pub fn group_start() -> Result<()> {
    let n = nccl()?;
    let cu = n.nccl_group_start()?;
    check(unsafe { cu() })
}

/// End the current collective group.
pub fn group_end() -> Result<()> {
    let n = nccl()?;
    let cu = n.nccl_group_end()?;
    check(unsafe { cu() })
}

/// NCCL library version as a packed integer (e.g. `22100` for NCCL 2.21.0).
pub fn version() -> Result<i32> {
    let n = nccl()?;
    let cu = n.nccl_get_version()?;
    let mut v: core::ffi::c_int = 0;
    check(unsafe { cu(&mut v) })?;
    Ok(v)
}

/// Human-readable name for a status code.
pub fn error_string(status: ncclResult_t) -> Result<&'static str> {
    let n = nccl()?;
    let cu = n.nccl_get_error_string()?;
    let p = unsafe { cu(status) };
    if p.is_null() {
        return Ok("unknown");
    }
    Ok(unsafe { core::ffi::CStr::from_ptr(p) }
        .to_str()
        .unwrap_or("unknown"))
}

// ---- Full collective surface ----

impl Communicator {
    /// `recvbuf = reduce(sendbuf[root])` on root only; non-root `recvbuf` is unchanged.
    pub fn reduce<T: NcclScalar>(
        &self,
        sendbuf: &DeviceBuffer<T>,
        recvbuf: &mut DeviceBuffer<T>,
        count: usize,
        op: RedOp,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_reduce()?;
        check(unsafe {
            cu(
                sendbuf.as_raw().0 as *const core::ffi::c_void,
                recvbuf.as_raw().0 as *mut core::ffi::c_void,
                count,
                T::raw(),
                op.raw(),
                root,
                self.handle,
                stream.as_raw(),
            )
        })
    }

    /// `recvbuf[r * sendcount..] = sendbuf` from rank `r`.
    pub fn all_gather<T: NcclScalar>(
        &self,
        sendbuf: &DeviceBuffer<T>,
        recvbuf: &mut DeviceBuffer<T>,
        sendcount: usize,
        stream: &Stream,
    ) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_all_gather()?;
        check(unsafe {
            cu(
                sendbuf.as_raw().0 as *const core::ffi::c_void,
                recvbuf.as_raw().0 as *mut core::ffi::c_void,
                sendcount,
                T::raw(),
                self.handle,
                stream.as_raw(),
            )
        })
    }

    /// Combined reduce + scatter: `recvbuf = reduce(sendbuf[r * recvcount..])`
    /// across ranks r = 0..nranks.
    pub fn reduce_scatter<T: NcclScalar>(
        &self,
        sendbuf: &DeviceBuffer<T>,
        recvbuf: &mut DeviceBuffer<T>,
        recvcount: usize,
        op: RedOp,
        stream: &Stream,
    ) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_reduce_scatter()?;
        check(unsafe {
            cu(
                sendbuf.as_raw().0 as *const core::ffi::c_void,
                recvbuf.as_raw().0 as *mut core::ffi::c_void,
                recvcount,
                T::raw(),
                op.raw(),
                self.handle,
                stream.as_raw(),
            )
        })
    }

    /// Point-to-point send to `peer`. Pair with [`Self::recv`] inside a
    /// group-call bracket.
    pub fn send<T: NcclScalar>(
        &self,
        sendbuf: &DeviceBuffer<T>,
        count: usize,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_send()?;
        check(unsafe {
            cu(
                sendbuf.as_raw().0 as *const core::ffi::c_void,
                count,
                T::raw(),
                peer,
                self.handle,
                stream.as_raw(),
            )
        })
    }

    /// Point-to-point recv from `peer`.
    pub fn recv<T: NcclScalar>(
        &self,
        recvbuf: &mut DeviceBuffer<T>,
        count: usize,
        peer: i32,
        stream: &Stream,
    ) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_recv()?;
        check(unsafe {
            cu(
                recvbuf.as_raw().0 as *mut core::ffi::c_void,
                count,
                T::raw(),
                peer,
                self.handle,
                stream.as_raw(),
            )
        })
    }

    /// Abort all outstanding operations on this communicator. Forces
    /// pending collectives to return with an error. Drop still destroys.
    pub fn abort(&self) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_comm_abort()?;
        check(unsafe { cu(self.handle) })
    }

    /// Mark the communicator as done. After `finalize` you can still
    /// call [`Communicator::get_async_error`] but no new collectives.
    pub fn finalize(&self) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_comm_finalize()?;
        check(unsafe { cu(self.handle) })
    }

    /// Poll the communicator's async error state (non-blocking).
    /// Returns `Ok(Success)` if there's no pending error.
    pub fn get_async_error(&self) -> Result<ncclResult_t> {
        let n = nccl()?;
        let cu = n.nccl_comm_get_async_error()?;
        let mut s = ncclResult_t::Success;
        check(unsafe { cu(self.handle, &mut s) })?;
        Ok(s)
    }

    /// CUDA device ordinal this communicator is bound to.
    pub fn cuda_device(&self) -> Result<i32> {
        let n = nccl()?;
        let cu = n.nccl_comm_cu_device()?;
        let mut d: core::ffi::c_int = 0;
        check(unsafe { cu(self.handle, &mut d) })?;
        Ok(d)
    }

    /// Split a communicator — ranks with the same `color` end up in the
    /// same new communicator, ordered by `key`. Pass `color = -1` to
    /// drop a rank from the new communicator.
    pub fn split(&self, color: i32, key: i32) -> Result<Communicator> {
        let n = nccl()?;
        let cu = n.nccl_comm_split()?;
        let mut new_comm: ncclComm_t = core::ptr::null_mut();
        check(unsafe { cu(self.handle, color, key, &mut new_comm, core::ptr::null_mut()) })?;
        Communicator::from_raw_handle(new_comm)
    }

    /// Register a device buffer for zero-copy collective use. Returns an
    /// opaque handle to pass to [`Self::deregister`] later.
    ///
    /// # Safety
    ///
    /// `dev_ptr` must be a live device-memory allocation.
    pub unsafe fn register(
        &self,
        dev_ptr: *mut core::ffi::c_void,
        size: usize,
    ) -> Result<*mut core::ffi::c_void> { unsafe {
        let n = nccl()?;
        let cu = n.nccl_comm_register()?;
        let mut handle: *mut core::ffi::c_void = core::ptr::null_mut();
        check(cu(self.handle, dev_ptr, size, &mut handle))?;
        Ok(handle)
    }}

    /// Deregister a previously-registered buffer.
    ///
    /// # Safety
    ///
    /// `handle` must come from a [`Self::register`] call on this comm.
    pub unsafe fn deregister(&self, handle: *mut core::ffi::c_void) -> Result<()> { unsafe {
        let n = nccl()?;
        let cu = n.nccl_comm_deregister()?;
        check(cu(self.handle, handle))
    }}

    /// Create a custom pre-multiplied-sum reduction op:
    /// `out = sum_i (scalar * x_i)`. Use the returned [`RedOp::Custom`]
    /// in any subsequent [`all_reduce`] / [`Communicator::reduce`] /
    /// [`Communicator::reduce_scatter`] on this communicator.
    /// Destroy it with [`Self::destroy_red_op`] when you're done.
    /// NCCL 2.11+.
    ///
    /// # Safety
    ///
    /// `scalar` must point to a single value of type `T` whose
    /// residence matches `residence` (host or device memory) and stay
    /// valid until the next collective using this op completes.
    pub unsafe fn create_pre_mul_sum<T: NcclScalar>(
        &self,
        scalar: *mut core::ffi::c_void,
        residence: ScalarResidence,
    ) -> Result<RedOp> { unsafe {
        let n = nccl()?;
        let cu = n.nccl_red_op_create_pre_mul_sum()?;
        let mut op = ncclRedOp_t(0);
        check(cu(&mut op, scalar, T::raw(), residence as i32, self.handle))?;
        Ok(RedOp::Custom(op.0))
    }}

    /// Destroy a custom op previously returned by [`Self::create_pre_mul_sum`].
    /// NCCL 2.11+. Calling on a built-in op (Sum/Prod/Max/Min/Avg) is a
    /// no-op error from NCCL — guard against that yourself.
    pub fn destroy_red_op(&self, op: RedOp) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_red_op_destroy()?;
        check(unsafe { cu(op.raw(), self.handle) })
    }

    /// Most recent error string produced on this communicator.
    /// NCCL 2.13+. Returns `"unknown"` if the loader can't resolve
    /// the symbol or the C library returns null.
    pub fn last_error(&self) -> Result<&'static str> {
        let n = nccl()?;
        let cu = n.nccl_get_last_error()?;
        let p = unsafe { cu(self.handle) };
        if p.is_null() {
            return Ok("unknown");
        }
        Ok(unsafe { core::ffi::CStr::from_ptr(p) }
            .to_str()
            .unwrap_or("unknown"))
    }
}

/// NCCL-managed device allocation. Drop calls `ncclMemFree`.
#[derive(Debug)]
pub struct NcclMem {
    ptr: *mut core::ffi::c_void,
}

impl NcclMem {
    /// Allocate `size` bytes through NCCL — these are GPU-direct-
    /// friendly (pre-registered with the transport). Use with
    /// [`Communicator::register`] for zero-copy collectives.
    pub fn new(size: usize) -> Result<Self> {
        let n = nccl()?;
        let cu = n.nccl_mem_alloc()?;
        let mut p: *mut core::ffi::c_void = core::ptr::null_mut();
        check(unsafe { cu(&mut p, size) })?;
        Ok(Self { ptr: p })
    }

    /// Raw device pointer to the underlying NCCL allocation.
    #[inline]
    pub fn as_raw(&self) -> *mut core::ffi::c_void {
        self.ptr
    }
}

impl Drop for NcclMem {
    fn drop(&mut self) {
        if let Ok(n) = nccl() {
            if let Ok(cu) = n.nccl_mem_free() {
                let _ = unsafe { cu(self.ptr) };
            }
        }
    }
}

// ---- spec-named aliases (Phase 52) ---------------------------------------
//
// The free-function / shorter-name forms in this file predate the formal
// Phase 52 spec. The aliases below match the spec verbatim so downstream
// distributed-roadmap code (Ring Attention, Megatron TP, FSDP, …) can use
// the canonical names directly. They are zero-cost re-exports / wrappers.

/// Spec-name alias for [`RedOp`] — the NCCL reduction operation enum.
pub use RedOp as NcclReduceOp;

/// Spec-name alias for [`UniqueId`] — the 128-byte multi-rank handshake.
pub use UniqueId as NcclUniqueId;

/// Spec-name alias for the [`NcclScalar`] trait — sealed element-type
/// trait identifying types that map to an `ncclDataType_t`.
pub use NcclScalar as NcclDataType;

/// Re-export the raw `ncclDataType_t` enum so callers can pattern-match
/// or pass it to lower-level helpers if needed.
pub use baracuda_nccl_sys::ncclDataType_t as RawNcclDataType;

impl UniqueId {
    /// Spec-name alias for [`UniqueId::new`] — generate a fresh 128-byte
    /// unique id on rank 0. Broadcast the result (e.g. via
    /// [`UniqueId::as_bytes`]) to every other rank before they call
    /// [`Communicator::new_with_id`].
    #[inline]
    pub fn generate() -> Result<Self> {
        Self::new()
    }
}

impl Communicator {
    /// Instance-method form of the free function [`all_reduce`].
    ///
    /// `send` and `recv` may alias (in-place AllReduce is legal).
    /// `world_size = 1` (single-GPU communicator) makes this a stream-
    /// ordered device-to-device copy — useful for smoke-testing the
    /// API surface without multi-GPU hardware.
    pub fn all_reduce<T: NcclScalar>(
        &self,
        send: &DeviceBuffer<T>,
        recv: &mut DeviceBuffer<T>,
        op: RedOp,
        stream: &Stream,
    ) -> Result<()> {
        let count = core::cmp::min(send.len(), recv.len());
        all_reduce(send, recv, count, op, self, stream)
    }

    /// Instance-method form of the free function [`broadcast`].
    /// In-place broadcast is the typical caller pattern — pass `buf`
    /// as both `send` and `recv`.
    pub fn broadcast<T: NcclScalar>(
        &self,
        buf: &mut DeviceBuffer<T>,
        root: i32,
        stream: &Stream,
    ) -> Result<()> {
        // In-place broadcast: NCCL is documented to accept `send == recv`
        // on every rank. We extract the device pointer from the single
        // `&mut DeviceBuffer<T>` and pass it as both the const and the
        // mut pointer to `ncclBroadcast` directly — this avoids the
        // `&mut → &` reborrow that earlier versions of this method used
        // (which trips `#[deny(invalid_reference_casting)]` and is UB
        // under Stacked Borrows even though the FFI call respects the
        // memcpy-style contract).
        let count = buf.len();
        let n = nccl()?;
        let cu = n.nccl_broadcast()?;
        let ptr = buf.as_raw().0 as *mut core::ffi::c_void;
        check(unsafe {
            cu(
                ptr as *const core::ffi::c_void,
                ptr,
                count,
                T::raw(),
                root,
                self.handle,
                stream.as_raw() as _,
            )
        })
    }

    /// Open a group of collectives that should be submitted atomically.
    /// See [`group_start`] (free function) for details.
    #[inline]
    pub fn group_start() -> Result<()> {
        group_start()
    }

    /// Close the most recently opened group. See [`group_end`].
    #[inline]
    pub fn group_end() -> Result<()> {
        group_end()
    }
}

#[cfg(test)]
mod compile_time_alias_checks {
    //! These tests compile-only — they assert that the spec-named
    //! aliases resolve to the same types as the existing names.

    use super::*;

    #[allow(dead_code)]
    fn red_op_alias_is_same(x: NcclReduceOp) -> RedOp {
        x
    }

    #[allow(dead_code)]
    fn unique_id_alias_is_same(x: NcclUniqueId) -> UniqueId {
        x
    }

    #[allow(dead_code)]
    fn nccl_data_type_alias_is_sealed<T: NcclDataType>(_: &T) {
        // Compile-only — proves `NcclDataType` is reachable as a trait.
    }
}
