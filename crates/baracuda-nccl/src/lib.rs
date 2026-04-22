//! Safe Rust wrappers for NVIDIA NCCL.
//!
//! v0.1 covers the communicator (single-process multi-GPU via
//! `ncclCommInitAll`, multi-process via `ncclCommInitRank` + `UniqueId`) and
//! the `all_reduce` + `broadcast` collectives — enough for synchronous
//! data-parallel training.
//!
//! NCCL is a Linux library; Windows has experimental support but no
//! general distribution. On hosts without NCCL, [`Communicator::init_all`]
//! returns `LoaderError::LibraryNotFound` — callers can fall back to
//! single-device execution.

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
    #[default]
    Sum,
    Prod,
    Max,
    Min,
    /// Arithmetic mean. NCCL 2.10+.
    Avg,
}

impl RedOp {
    fn raw(self) -> ncclRedOp_t {
        match self {
            RedOp::Sum => ncclRedOp_t::Sum,
            RedOp::Prod => ncclRedOp_t::Prod,
            RedOp::Max => ncclRedOp_t::Max,
            RedOp::Min => ncclRedOp_t::Min,
            RedOp::Avg => ncclRedOp_t::Avg,
        }
    }
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

mod sealed {
    pub trait Sealed {}
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
pub struct Communicator {
    handle: ncclComm_t,
}

unsafe impl Send for Communicator {}

impl core::fmt::Debug for Communicator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("nccl::Communicator")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Communicator {
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
        Ok(comms.into_iter().map(|handle| Self { handle }).collect())
    }

    /// Initialize one rank of a multi-process communicator.
    pub fn init_rank(nranks: i32, id: UniqueId, rank: i32) -> Result<Self> {
        let n = nccl()?;
        let cu = n.nccl_comm_init_rank()?;
        let mut handle: ncclComm_t = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, nranks, id.0, rank) })?;
        Ok(Self { handle })
    }

    /// Number of ranks in the communicator.
    pub fn nranks(&self) -> Result<i32> {
        let n = nccl()?;
        let cu = n.nccl_comm_count()?;
        let mut c: core::ffi::c_int = 0;
        check(unsafe { cu(self.handle, &mut c) })?;
        Ok(c)
    }

    /// Rank of this communicator within the group.
    pub fn rank(&self) -> Result<i32> {
        let n = nccl()?;
        let cu = n.nccl_comm_user_rank()?;
        let mut r: core::ffi::c_int = 0;
        check(unsafe { cu(self.handle, &mut r) })?;
        Ok(r)
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
        Ok(Communicator { handle: new_comm })
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
    ) -> Result<*mut core::ffi::c_void> {
        let n = nccl()?;
        let cu = n.nccl_comm_register()?;
        let mut handle: *mut core::ffi::c_void = core::ptr::null_mut();
        check(cu(self.handle, dev_ptr, size, &mut handle))?;
        Ok(handle)
    }

    /// Deregister a previously-registered buffer.
    ///
    /// # Safety
    ///
    /// `handle` must come from a [`Self::register`] call on this comm.
    pub unsafe fn deregister(&self, handle: *mut core::ffi::c_void) -> Result<()> {
        let n = nccl()?;
        let cu = n.nccl_comm_deregister()?;
        check(cu(self.handle, handle))
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
