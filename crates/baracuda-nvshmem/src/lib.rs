//! Safe Rust wrappers for the **NVIDIA NVSHMEM host API**.
//!
//! NVSHMEM is the OpenSHMEM symmetric-heap model on GPUs: every PE
//! (processing element — typically one GPU) allocates from a *symmetric
//! heap* at a shared virtual address, and any PE can read/write another PE's
//! heap directly via one-sided `put` / `get`. This is the fine-grained,
//! one-sided complement to [`baracuda-nccl`]'s collectives — the two coexist
//! and a single program may use both.
//!
//! ## What this crate covers (Tier 1)
//!
//! - [`Context`] — process-wide NVSHMEM lifetime (init / finalize) plus
//!   cached `my_pe` / `n_pes`, and the barrier / quiet / fence ordering
//!   primitives.
//! - [`Team`] — a subset of PEs created via strided split.
//! - [`SymmetricBuffer`] — a typed allocation on the symmetric heap.
//! - Host-initiated RMA — blocking and stream-ordered [`Context::put`] /
//!   [`Context::get`].
//!
//! ## What it does *not* cover
//!
//! The **device-side** API — the `__device__` `nvshmem_int_p` /
//! `nvshmem_putmem_nbi` calls issued from *inside* a CUDA kernel — requires
//! linking `libnvshmem_device.a` into the consumer's kernel binary and is
//! out of scope (it cannot be a lazily-loaded host symbol). A consumer that
//! needs device-side NVSHMEM writes its own `.cu` that includes the NVSHMEM
//! headers and links the device archive.
//!
//! ## Availability
//!
//! NVSHMEM is a Linux library requiring compute capability sm_70+ (every
//! baracuda-supported GPU qualifies). On hosts without the NVSHMEM runtime
//! installed, [`Context::init`] returns `LoaderError::LibraryNotFound`, so
//! callers can fall back to single-process execution.
//!
//! [`baracuda-nccl`]: https://docs.rs/baracuda-nccl

#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};

use baracuda_driver::Stream;
use baracuda_nvshmem_sys::{nvshmem, nvshmemResult_t, nvshmem_team_t, nvshmemx_uniqueid_t};
use baracuda_types::DeviceRepr;

/// Error type for NVSHMEM operations.
pub type Error = baracuda_core::Error<nvshmemResult_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: nvshmemResult_t) -> Result<()> {
    Error::check(status)
}

#[inline]
fn stream_raw(stream: &Stream) -> baracuda_cuda_sys::runtime::cudaStream_t {
    // `CUstream` (driver) and `cudaStream_t` (runtime) are the same opaque
    // handle at the ABI level — the runtime stream wraps the driver one.
    stream.as_raw() as _
}

// ---- Context --------------------------------------------------------------

/// The process-wide NVSHMEM runtime, from this PE's point of view.
///
/// NVSHMEM is a **process singleton**: `nvshmem_init` / `nvshmem_finalize`
/// must be called exactly once each per process. Construct a single
/// `Context` near program start and drop it (or call [`Context::finalize`])
/// at shutdown. `my_pe` / `n_pes` are read once at init and cached, so the
/// hot accessors are infallible.
pub struct Context {
    my_pe: i32,
    n_pes: i32,
    finalized: bool,
}

impl core::fmt::Debug for Context {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("nvshmem::Context")
            .field("my_pe", &self.my_pe)
            .field("n_pes", &self.n_pes)
            .finish()
    }
}

impl Context {
    /// Initialize NVSHMEM using the environment-selected bootstrap
    /// (`NVSHMEM_BOOTSTRAP` — PMI / MPI / PMIx). This is the common launcher
    /// path (`nvshmrun` / `mpirun`). Call exactly once per process.
    pub fn init() -> Result<Self> {
        let n = nvshmem()?;
        let init = n.nvshmem_init()?;
        unsafe { init() };
        Self::from_initialized()
    }

    /// Initialize NVSHMEM through `nvshmemx_init_attr`, passing a caller-built
    /// attributes struct (e.g. one populated from a
    /// [`nvshmemx_uniqueid_t`](baracuda_nvshmem_sys::nvshmemx_uniqueid_t) via
    /// the raw `nvshmemx_set_attr_uniqueid_args`). Pass `flags` and a pointer
    /// to a valid `nvshmemx_init_attr_t` (or null for defaults).
    ///
    /// The attribute struct's layout is NVSHMEM-version-specific and so is
    /// **not** modeled as a typed Rust struct — build it through the raw
    /// [`baracuda-nvshmem-sys`] helpers.
    ///
    /// # Safety
    ///
    /// `attr` must be null or a properly-initialized `nvshmemx_init_attr_t`
    /// for the installed NVSHMEM version, and must outlive the call.
    ///
    /// [`baracuda-nvshmem-sys`]: baracuda_nvshmem_sys
    pub unsafe fn init_with_attr(flags: u32, attr: *mut c_void) -> Result<Self> {
        let n = nvshmem()?;
        let init = n.nvshmemx_init_attr()?;
        check(unsafe { init(flags, attr) })?;
        Self::from_initialized()
    }

    fn from_initialized() -> Result<Self> {
        let n = nvshmem()?;
        let my_pe = unsafe { (n.nvshmem_my_pe()?)() };
        let n_pes = unsafe { (n.nvshmem_n_pes()?)() };
        Ok(Self {
            my_pe,
            n_pes,
            finalized: false,
        })
    }

    /// This PE's global index (0..`n_pes`). Cached at init.
    #[inline]
    pub fn my_pe(&self) -> i32 {
        self.my_pe
    }

    /// Total number of PEs in the program. Cached at init.
    #[inline]
    pub fn n_pes(&self) -> i32 {
        self.n_pes
    }

    /// NVSHMEM version as `(major, minor)`.
    pub fn version(&self) -> Result<(i32, i32)> {
        let n = nvshmem()?;
        let cu = n.nvshmem_info_get_version()?;
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        unsafe { cu(&mut major, &mut minor) };
        Ok((major, minor))
    }

    /// Allocate `len` elements of `T` on the symmetric heap. The returned
    /// buffer occupies the **same virtual address on every PE**, so its
    /// pointer doubles as a remote address in [`Context::put`] /
    /// [`Context::get`].
    pub fn malloc<T: DeviceRepr>(&self, len: usize) -> Result<SymmetricBuffer<T>> {
        SymmetricBuffer::new(len)
    }

    /// The team of all PEs ([`Team::WORLD`]).
    #[inline]
    pub fn world(&self) -> Team {
        Team::WORLD
    }

    // -- ordering / synchronization --

    /// Global barrier: every PE arrives **and** all RMA issued before the
    /// call has completed remotely.
    pub fn barrier_all(&self) -> Result<()> {
        let n = nvshmem()?;
        unsafe { (n.nvshmem_barrier_all()?)() };
        Ok(())
    }

    /// Stream-ordered [`Self::barrier_all`].
    pub fn barrier_all_on_stream(&self, stream: &Stream) -> Result<()> {
        let n = nvshmem()?;
        let cu = n.nvshmemx_barrier_all_on_stream()?;
        unsafe { cu(stream_raw(stream)) };
        Ok(())
    }

    /// Lighter barrier — PE arrival only, without the RMA remote-completion
    /// guarantee of [`Self::barrier_all`].
    pub fn sync_all(&self) -> Result<()> {
        let n = nvshmem()?;
        unsafe { (n.nvshmem_sync_all()?)() };
        Ok(())
    }

    /// Block until all RMA issued by this PE has completed remotely.
    pub fn quiet(&self) -> Result<()> {
        let n = nvshmem()?;
        unsafe { (n.nvshmem_quiet()?)() };
        Ok(())
    }

    /// Order (but do not wait for completion of) outstanding RMA from this PE.
    pub fn fence(&self) -> Result<()> {
        let n = nvshmem()?;
        unsafe { (n.nvshmem_fence()?)() };
        Ok(())
    }

    // -- host-initiated RMA --

    /// Blocking host put: copy `count` elements from the local `src` buffer
    /// into PE `pe`'s copy of the symmetric `dest` buffer. Returns after the
    /// data has left the local PE.
    pub fn put<T: DeviceRepr>(
        &self,
        dest: &SymmetricBuffer<T>,
        src: &SymmetricBuffer<T>,
        count: usize,
        pe: i32,
    ) -> Result<()> {
        assert!(count <= dest.len() && count <= src.len(), "put out of range");
        let n = nvshmem()?;
        let cu = n.nvshmem_putmem()?;
        unsafe {
            cu(
                dest.ptr,
                src.ptr as *const c_void,
                count * core::mem::size_of::<T>(),
                pe,
            )
        };
        Ok(())
    }

    /// Blocking host get: copy `count` elements from PE `pe`'s copy of the
    /// symmetric `src` buffer into the local `dest` buffer.
    pub fn get<T: DeviceRepr>(
        &self,
        dest: &SymmetricBuffer<T>,
        src: &SymmetricBuffer<T>,
        count: usize,
        pe: i32,
    ) -> Result<()> {
        assert!(count <= dest.len() && count <= src.len(), "get out of range");
        let n = nvshmem()?;
        let cu = n.nvshmem_getmem()?;
        unsafe {
            cu(
                dest.ptr,
                src.ptr as *const c_void,
                count * core::mem::size_of::<T>(),
                pe,
            )
        };
        Ok(())
    }

    /// Stream-ordered [`Self::put`]. Completes in `stream` order; pair with
    /// [`Self::quiet`] (or a [`Self::barrier_all`]) for remote completion.
    pub fn put_on_stream<T: DeviceRepr>(
        &self,
        dest: &SymmetricBuffer<T>,
        src: &SymmetricBuffer<T>,
        count: usize,
        pe: i32,
        stream: &Stream,
    ) -> Result<()> {
        assert!(count <= dest.len() && count <= src.len(), "put out of range");
        let n = nvshmem()?;
        let cu = n.nvshmemx_putmem_on_stream()?;
        unsafe {
            cu(
                dest.ptr,
                src.ptr as *const c_void,
                count * core::mem::size_of::<T>(),
                pe,
                stream_raw(stream),
            )
        };
        Ok(())
    }

    /// Stream-ordered [`Self::get`].
    pub fn get_on_stream<T: DeviceRepr>(
        &self,
        dest: &SymmetricBuffer<T>,
        src: &SymmetricBuffer<T>,
        count: usize,
        pe: i32,
        stream: &Stream,
    ) -> Result<()> {
        assert!(count <= dest.len() && count <= src.len(), "get out of range");
        let n = nvshmem()?;
        let cu = n.nvshmemx_getmem_on_stream()?;
        unsafe {
            cu(
                dest.ptr,
                src.ptr as *const c_void,
                count * core::mem::size_of::<T>(),
                pe,
                stream_raw(stream),
            )
        };
        Ok(())
    }

    /// Explicitly finalize NVSHMEM. Idempotent — also run on [`Drop`]. After
    /// this no further NVSHMEM calls are valid.
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        let n = nvshmem()?;
        unsafe { (n.nvshmem_finalize()?)() };
        self.finalized = true;
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        if let Ok(n) = nvshmem() {
            if let Ok(cu) = n.nvshmem_finalize() {
                unsafe { cu() };
            }
        }
    }
}

// ---- Team -----------------------------------------------------------------

/// A team — a named subset of PEs. Teams created via
/// [`Team::split_strided`] must be released with [`Team::destroy`]; the
/// predefined [`Team::WORLD`] / [`Team::SHARED`] must **not** be destroyed.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Team(nvshmem_team_t);

impl Team {
    /// The team of every PE in the program.
    pub const WORLD: Self = Self(nvshmem_team_t::WORLD);
    /// The team of PEs sharing a compute node.
    pub const SHARED: Self = Self(nvshmem_team_t::SHARED);

    /// Create a sub-team of `size` PEs from this team, starting at PE `start`
    /// (in this team's index space) and taking every `stride`-th PE. Returns
    /// `None` on the PEs that are **not** members of the new team.
    ///
    /// Defaults are used for the team config (`config = null`,
    /// `config_mask = 0`).
    pub fn split_strided(
        &self,
        start: i32,
        stride: i32,
        size: i32,
    ) -> Result<Option<Team>> {
        let n = nvshmem()?;
        let cu = n.nvshmem_team_split_strided()?;
        let mut new_team = nvshmem_team_t::INVALID;
        check(unsafe {
            cu(
                self.0,
                start,
                stride,
                size,
                core::ptr::null(),
                0,
                &mut new_team,
            )
        })?;
        if new_team == nvshmem_team_t::INVALID {
            Ok(None)
        } else {
            Ok(Some(Team(new_team)))
        }
    }

    /// This PE's index *within this team* (0..`n_pes`), or `-1` if this PE is
    /// not a member.
    pub fn my_pe(&self) -> Result<i32> {
        let n = nvshmem()?;
        let cu = n.nvshmem_team_my_pe()?;
        Ok(unsafe { cu(self.0) })
    }

    /// Number of PEs in this team.
    pub fn n_pes(&self) -> Result<i32> {
        let n = nvshmem()?;
        let cu = n.nvshmem_team_n_pes()?;
        Ok(unsafe { cu(self.0) })
    }

    /// Translate `src_pe` (an index in this team) into its index in
    /// `dest_team`. Returns `-1` if `src_pe` is not in `dest_team`.
    pub fn translate_pe(&self, src_pe: i32, dest_team: Team) -> Result<i32> {
        let n = nvshmem()?;
        let cu = n.nvshmem_team_translate_pe()?;
        Ok(unsafe { cu(self.0, src_pe, dest_team.0) })
    }

    /// Destroy a team created via [`Self::split_strided`]. Destroying a
    /// predefined team ([`Self::WORLD`] / [`Self::SHARED`]) is a programmer
    /// error and is rejected here.
    pub fn destroy(self) -> Result<()> {
        if self == Team::WORLD || self == Team::SHARED {
            // Predefined teams are owned by the runtime — don't free them.
            return Ok(());
        }
        let n = nvshmem()?;
        let cu = n.nvshmem_team_destroy()?;
        unsafe { cu(self.0) };
        Ok(())
    }

    /// The raw team handle.
    #[inline]
    pub fn as_raw(&self) -> nvshmem_team_t {
        self.0
    }
}

// ---- SymmetricBuffer ------------------------------------------------------

/// A typed allocation on the NVSHMEM symmetric heap. The same virtual address
/// is valid on every PE, so the pointer can be used as a remote address in
/// [`Context::put`] / [`Context::get`]. Freed on [`Drop`] via `nvshmem_free`.
///
/// `SymmetricBuffer` is **not** `Send`/`Sync`: NVSHMEM is bound to the PE's
/// owning thread/process and the buffer must be freed on the same.
pub struct SymmetricBuffer<T: DeviceRepr> {
    ptr: *mut c_void,
    len: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: DeviceRepr> core::fmt::Debug for SymmetricBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SymmetricBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl<T: DeviceRepr> SymmetricBuffer<T> {
    /// Allocate `len` elements on the symmetric heap. This is a **collective**
    /// call — every PE must call it with the same `len` (NVSHMEM contract).
    pub fn new(len: usize) -> Result<Self> {
        let n = nvshmem()?;
        let cu = n.nvshmem_malloc()?;
        let bytes = len.checked_mul(core::mem::size_of::<T>()).expect("size overflow");
        let ptr = unsafe { cu(bytes) };
        if ptr.is_null() && bytes != 0 {
            // nvshmem_malloc aborts internally on real OOM; a null with a
            // non-zero request still warrants an error rather than a silent
            // dangling buffer.
            return Err(Error::Status {
                status: nvshmemResult_t(1),
            });
        }
        Ok(Self {
            ptr,
            len,
            _marker: core::marker::PhantomData,
        })
    }

    /// Element count.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The symmetric device pointer (same VA on every PE).
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// The symmetric device pointer, mutable.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }
}

impl<T: DeviceRepr> Drop for SymmetricBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Ok(n) = nvshmem() {
            if let Ok(cu) = n.nvshmem_free() {
                unsafe { cu(self.ptr) };
            }
        }
    }
}

// ---- UniqueId -------------------------------------------------------------

/// A 128-byte identifier for the unique-id bootstrap (the NVSHMEM analogue of
/// NCCL's `UniqueId`). One PE calls [`UniqueId::new`] and distributes the
/// bytes to every other PE; each then feeds it to the raw
/// `nvshmemx_set_attr_uniqueid_args` + [`Context::init_with_attr`] path.
///
/// Wiring the id into init requires the version-specific
/// `nvshmemx_init_attr_t` struct, which this safe layer deliberately does not
/// model — use the raw [`baracuda-nvshmem-sys`] helpers for that step.
///
/// [`baracuda-nvshmem-sys`]: baracuda_nvshmem_sys
#[derive(Copy, Clone, Debug)]
pub struct UniqueId(nvshmemx_uniqueid_t);

impl UniqueId {
    /// Generate a fresh unique id on this PE.
    pub fn new() -> Result<Self> {
        let n = nvshmem()?;
        let cu = n.nvshmemx_get_uniqueid()?;
        let mut id = nvshmemx_uniqueid_t::default();
        check(unsafe { cu(&mut id) })?;
        Ok(Self(id))
    }

    /// Raw representation. Transmit verbatim to the other PEs.
    pub fn as_raw(&self) -> nvshmemx_uniqueid_t {
        self.0
    }

    /// Rebuild from a raw id received from another PE.
    pub fn from_raw(id: nvshmemx_uniqueid_t) -> Self {
        Self(id)
    }
}

/// Convenience: NVSHMEM library version as `(major, minor)` without holding a
/// [`Context`]. Useful for capability probes. Errors if NVSHMEM is not
/// installed.
pub fn version() -> Result<(i32, i32)> {
    let n = nvshmem()?;
    let cu = n.nvshmem_info_get_version()?;
    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    unsafe { cu(&mut major, &mut minor) };
    Ok((major, minor))
}
