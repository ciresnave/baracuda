//! Raw FFI + dynamic loader for the **NVIDIA NVSHMEM host library**
//! (`libnvshmem_host.so`).
//!
//! `baracuda-nvshmem` wraps this with a safe, typed API. Use this
//! crate directly only if you need a function that the safe layer
//! hasn't wrapped yet (in which case please file a bug).
//!
//! [NVSHMEM](https://developer.nvidia.com/nvshmem) is NVIDIA's
//! implementation of the OpenSHMEM symmetric-heap programming model on
//! GPUs. Where [NCCL](../baracuda_nccl_sys/index.html) provides *collective*
//! communication, NVSHMEM adds **one-sided** RDMA: every PE (processing
//! element — typically one GPU) allocates from a *symmetric heap* at a
//! shared virtual address, and any PE can `putmem` / `getmem` directly into
//! another PE's heap without involving the remote CPU.
//!
//! ## Scope of this crate
//!
//! This crate binds the **host-side** API only — initialization, PE
//! discovery, team management, the symmetric heap allocator, host-initiated
//! RMA (`nvshmem(x)_putmem` / `getmem` and their `_on_stream` variants), and
//! the memory-ordering / synchronization primitives. The **device-side**
//! API (the `__device__` `nvshmem_int_p` / `nvshmem_putmem_nbi` etc. callable
//! from inside a CUDA kernel) lives in the static archive
//! `libnvshmem_device.a`, which must be linked into the *consumer's* kernel
//! binary; that is intentionally out of scope here (see the workspace
//! ROADMAP for the deferred device-shim crate).
//!
//! ## Loading model
//!
//! Like [`baracuda-nccl-sys`], symbols resolve lazily via
//! [`libloading`] — there is **no** link-time dependency on NVSHMEM, and the
//! crate compiles on hosts (including Windows) where NVSHMEM is not
//! installed. [`nvshmem()`] returns `LoaderError::LibraryNotFound` at runtime
//! when the host library is absent, so callers can degrade to single-process
//! execution. NVSHMEM is a Linux library in practice; NVIDIA does not ship a
//! general Windows distribution.
//!
//! ## Minimum hardware
//!
//! NVSHMEM requires compute capability **sm_70+** (Volta). baracuda targets
//! sm_80+, so every supported baracuda GPU can run NVSHMEM.
//!
//! [`baracuda-nccl-sys`]: ../baracuda_nccl_sys/index.html
//! [`libloading`]: https://docs.rs/libloading

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_int, c_long, c_uint, c_void};
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

// ---- handle / scalar types -----------------------------------------------

/// A team handle. NVSHMEM models teams as a small integer handle (mirroring
/// OpenSHMEM's `shmem_team_t`), with a set of predefined teams. Modeled as a
/// transparent newtype rather than a closed enum because user teams created
/// via [`PFN_nvshmem_team_split_strided`] return library-assigned ids.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct nvshmem_team_t(pub i32);

impl nvshmem_team_t {
    /// The team containing every PE in the NVSHMEM program. Always valid
    /// after [`PFN_nvshmem_init`].
    pub const WORLD: Self = Self(0);
    /// The team of PEs sharing a compute node (NVLink / PCIe island).
    pub const SHARED: Self = Self(1);
    /// Sentinel returned by team-creation calls that produce no team (e.g. a
    /// PE excluded from the requested stride).
    pub const INVALID: Self = Self(-1);
}

/// Opaque, version-specific initialization-attributes struct
/// (`nvshmemx_init_attr_t`). Its layout changes between NVSHMEM releases, so
/// it is **not** modeled as a typed Rust struct — callers that need the
/// attribute-driven bootstrap allocate a sufficiently large zeroed buffer and
/// populate it through [`PFN_nvshmemx_set_attr_uniqueid_args`]. Pass a null
/// `*mut c_void` to [`PFN_nvshmemx_init_attr`] for default (PMI/MPI-bootstrap)
/// behavior.
pub type nvshmemx_init_attr_t = c_void;

/// Opaque team-configuration struct (`nvshmem_team_config_t`) passed to
/// [`PFN_nvshmem_team_split_strided`]. Pass a null `*const c_void` with a
/// `config_mask` of `0` for defaults.
pub type nvshmem_team_config_t = c_void;

/// A 128-byte unique identifier for the unique-id bootstrap path
/// (`nvshmemx_uniqueid_t`), analogous to NCCL's `ncclUniqueId`. One PE calls
/// [`PFN_nvshmemx_get_uniqueid`] and distributes the bytes to every other PE
/// over a user-provided channel; each PE then feeds it to
/// [`PFN_nvshmemx_set_attr_uniqueid_args`].
///
/// The C struct is `{ int version; char internal[...]; }` padded to a fixed
/// size; we model it as a leading version field plus a byte tail so the total
/// is 128 bytes. Treat the contents as opaque — only transmit them verbatim.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct nvshmemx_uniqueid_t {
    /// Structure version tag.
    pub version: c_int,
    /// Internal field.
    pub internal: [c_char; 124],
}

impl core::fmt::Debug for nvshmemx_uniqueid_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("nvshmemx_uniqueid_t")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl Default for nvshmemx_uniqueid_t {
    fn default() -> Self {
        Self {
            version: 0,
            internal: [0; 124],
        }
    }
}

// ---- status ---------------------------------------------------------------

/// Status code returned by the few NVSHMEM host calls that report one
/// (`nvshmemx_init_attr`, `nvshmem_team_split_strided`,
/// `nvshmemx_get_uniqueid`). `0` is success; most NVSHMEM host calls return
/// `void` and abort the process internally on a fatal error rather than
/// surfacing a code.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct nvshmemResult_t(pub i32);

impl nvshmemResult_t {
    /// NVSHMEM result code `Success`.
    pub const Success: Self = Self(0);

    /// `is_success` method on `nvshmemResult_t`.
    #[inline]
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for nvshmemResult_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        if self.0 == 0 {
            "nvshmemSuccess"
        } else {
            "nvshmemError"
        }
    }
    fn description(self) -> &'static str {
        if self.0 == 0 {
            "success"
        } else {
            "NVSHMEM host call returned a non-zero status"
        }
    }
    fn is_success(self) -> bool {
        nvshmemResult_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "nvshmem"
    }
}

// ---- function-pointer types ----------------------------------------------
//
// Init / finalize / query.

/// `void nvshmem_init(void)` — bootstrap via the environment (PMI / MPI /
/// PMIx, selected by `NVSHMEM_BOOTSTRAP`).
pub type PFN_nvshmem_init = unsafe extern "C" fn();
/// `void nvshmem_finalize(void)`.
pub type PFN_nvshmem_finalize = unsafe extern "C" fn();
/// `int nvshmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t *attr)`.
pub type PFN_nvshmemx_init_attr =
    unsafe extern "C" fn(flags: c_uint, attr: *mut nvshmemx_init_attr_t) -> nvshmemResult_t;
/// `int nvshmemx_init_status(void)` — current init state (0 = none).
pub type PFN_nvshmemx_init_status = unsafe extern "C" fn() -> c_int;

/// `int nvshmem_my_pe(void)`.
pub type PFN_nvshmem_my_pe = unsafe extern "C" fn() -> c_int;
/// `int nvshmem_n_pes(void)`.
pub type PFN_nvshmem_n_pes = unsafe extern "C" fn() -> c_int;
/// `void nvshmem_info_get_version(int *major, int *minor)`.
pub type PFN_nvshmem_info_get_version =
    unsafe extern "C" fn(major: *mut c_int, minor: *mut c_int);

// Unique-id bootstrap.

/// `int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t *id)`.
pub type PFN_nvshmemx_get_uniqueid =
    unsafe extern "C" fn(id: *mut nvshmemx_uniqueid_t) -> nvshmemResult_t;
/// `void nvshmemx_set_attr_uniqueid_args(int rank, int nranks,
/// nvshmemx_uniqueid_t *id, nvshmemx_init_attr_t *attr)`.
pub type PFN_nvshmemx_set_attr_uniqueid_args = unsafe extern "C" fn(
    rank: c_int,
    nranks: c_int,
    id: *mut nvshmemx_uniqueid_t,
    attr: *mut nvshmemx_init_attr_t,
);

// Team management.

/// `int nvshmem_team_split_strided(nvshmem_team_t parent, int start,
/// int stride, int size, const nvshmem_team_config_t *config,
/// long config_mask, nvshmem_team_t *new_team)`.
pub type PFN_nvshmem_team_split_strided = unsafe extern "C" fn(
    parent_team: nvshmem_team_t,
    start: c_int,
    stride: c_int,
    size: c_int,
    config: *const nvshmem_team_config_t,
    config_mask: c_long,
    new_team: *mut nvshmem_team_t,
) -> nvshmemResult_t;
/// `void nvshmem_team_destroy(nvshmem_team_t team)`.
pub type PFN_nvshmem_team_destroy = unsafe extern "C" fn(team: nvshmem_team_t);
/// `int nvshmem_team_my_pe(nvshmem_team_t team)`.
pub type PFN_nvshmem_team_my_pe = unsafe extern "C" fn(team: nvshmem_team_t) -> c_int;
/// `int nvshmem_team_n_pes(nvshmem_team_t team)`.
pub type PFN_nvshmem_team_n_pes = unsafe extern "C" fn(team: nvshmem_team_t) -> c_int;
/// `int nvshmem_team_translate_pe(nvshmem_team_t src, int src_pe,
/// nvshmem_team_t dest)`.
pub type PFN_nvshmem_team_translate_pe = unsafe extern "C" fn(
    src_team: nvshmem_team_t,
    src_pe: c_int,
    dest_team: nvshmem_team_t,
) -> c_int;

// Symmetric heap allocator.

/// `void *nvshmem_malloc(size_t size)`.
pub type PFN_nvshmem_malloc = unsafe extern "C" fn(size: usize) -> *mut c_void;
/// `void nvshmem_free(void *ptr)`.
pub type PFN_nvshmem_free = unsafe extern "C" fn(ptr: *mut c_void);
/// `void *nvshmem_align(size_t alignment, size_t size)`.
pub type PFN_nvshmem_align =
    unsafe extern "C" fn(alignment: usize, size: usize) -> *mut c_void;
/// `void *nvshmem_calloc(size_t count, size_t size)`.
pub type PFN_nvshmem_calloc =
    unsafe extern "C" fn(count: usize, size: usize) -> *mut c_void;

// Host-initiated RMA.

/// `void nvshmem_putmem(void *dest, const void *source, size_t bytes,
/// int pe)` — blocking host put into `pe`'s symmetric heap.
pub type PFN_nvshmem_putmem =
    unsafe extern "C" fn(dest: *mut c_void, source: *const c_void, bytes: usize, pe: c_int);
/// `void nvshmem_getmem(void *dest, const void *source, size_t bytes,
/// int pe)` — blocking host get from `pe`'s symmetric heap.
pub type PFN_nvshmem_getmem =
    unsafe extern "C" fn(dest: *mut c_void, source: *const c_void, bytes: usize, pe: c_int);
/// `void nvshmemx_putmem_on_stream(void *dest, const void *source,
/// size_t bytes, int pe, cudaStream_t stream)`.
pub type PFN_nvshmemx_putmem_on_stream = unsafe extern "C" fn(
    dest: *mut c_void,
    source: *const c_void,
    bytes: usize,
    pe: c_int,
    stream: cudaStream_t,
);
/// `void nvshmemx_getmem_on_stream(void *dest, const void *source,
/// size_t bytes, int pe, cudaStream_t stream)`.
pub type PFN_nvshmemx_getmem_on_stream = unsafe extern "C" fn(
    dest: *mut c_void,
    source: *const c_void,
    bytes: usize,
    pe: c_int,
    stream: cudaStream_t,
);

// Memory ordering & synchronization.

/// `void nvshmem_barrier_all(void)` — global barrier + remote-completion.
pub type PFN_nvshmem_barrier_all = unsafe extern "C" fn();
/// `void nvshmemx_barrier_all_on_stream(cudaStream_t stream)`.
pub type PFN_nvshmemx_barrier_all_on_stream = unsafe extern "C" fn(stream: cudaStream_t);
/// `void nvshmem_sync_all(void)` — lighter barrier (no remote-completion
/// guarantee on RMA, just PE arrival).
pub type PFN_nvshmem_sync_all = unsafe extern "C" fn();
/// `void nvshmem_quiet(void)` — block until all outstanding RMA issued by
/// this PE has completed remotely.
pub type PFN_nvshmem_quiet = unsafe extern "C" fn();
/// `void nvshmem_fence(void)` — order (but do not complete) outstanding RMA.
pub type PFN_nvshmem_fence = unsafe extern "C" fn();

// ---- loader --------------------------------------------------------------

fn nvshmem_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        // Modern NVSHMEM (2.x+) splits host/device libraries; older
        // single-library builds shipped `libnvshmem.so`.
        &[
            "libnvshmem_host.so.3",
            "libnvshmem_host.so",
            "libnvshmem.so.3",
            "libnvshmem.so",
        ]
    }
    #[cfg(target_os = "windows")]
    {
        // NVIDIA does not ship a general Windows NVSHMEM distribution; these
        // are defensive names in case a vendor build provides one.
        &["nvshmem_host.dll", "nvshmem.dll"]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

macro_rules! nvshmem_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        /// NVSHMEM host-library dynamic-loader handle.
        pub struct Nvshmem {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Nvshmem {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Nvshmem").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Nvshmem {
            $(
                /// `func` (func).
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
        }
    };
}

nvshmem_fns! {
    nvshmem_init as "nvshmem_init": PFN_nvshmem_init;
    nvshmem_finalize as "nvshmem_finalize": PFN_nvshmem_finalize;
    nvshmemx_init_attr as "nvshmemx_init_attr": PFN_nvshmemx_init_attr;
    nvshmemx_init_status as "nvshmemx_init_status": PFN_nvshmemx_init_status;
    nvshmem_my_pe as "nvshmem_my_pe": PFN_nvshmem_my_pe;
    nvshmem_n_pes as "nvshmem_n_pes": PFN_nvshmem_n_pes;
    nvshmem_info_get_version as "nvshmem_info_get_version": PFN_nvshmem_info_get_version;
    nvshmemx_get_uniqueid as "nvshmemx_get_uniqueid": PFN_nvshmemx_get_uniqueid;
    nvshmemx_set_attr_uniqueid_args as "nvshmemx_set_attr_uniqueid_args": PFN_nvshmemx_set_attr_uniqueid_args;
    nvshmem_team_split_strided as "nvshmem_team_split_strided": PFN_nvshmem_team_split_strided;
    nvshmem_team_destroy as "nvshmem_team_destroy": PFN_nvshmem_team_destroy;
    nvshmem_team_my_pe as "nvshmem_team_my_pe": PFN_nvshmem_team_my_pe;
    nvshmem_team_n_pes as "nvshmem_team_n_pes": PFN_nvshmem_team_n_pes;
    nvshmem_team_translate_pe as "nvshmem_team_translate_pe": PFN_nvshmem_team_translate_pe;
    nvshmem_malloc as "nvshmem_malloc": PFN_nvshmem_malloc;
    nvshmem_free as "nvshmem_free": PFN_nvshmem_free;
    nvshmem_align as "nvshmem_align": PFN_nvshmem_align;
    nvshmem_calloc as "nvshmem_calloc": PFN_nvshmem_calloc;
    nvshmem_putmem as "nvshmem_putmem": PFN_nvshmem_putmem;
    nvshmem_getmem as "nvshmem_getmem": PFN_nvshmem_getmem;
    nvshmemx_putmem_on_stream as "nvshmemx_putmem_on_stream": PFN_nvshmemx_putmem_on_stream;
    nvshmemx_getmem_on_stream as "nvshmemx_getmem_on_stream": PFN_nvshmemx_getmem_on_stream;
    nvshmem_barrier_all as "nvshmem_barrier_all": PFN_nvshmem_barrier_all;
    nvshmemx_barrier_all_on_stream as "nvshmemx_barrier_all_on_stream": PFN_nvshmemx_barrier_all_on_stream;
    nvshmem_sync_all as "nvshmem_sync_all": PFN_nvshmem_sync_all;
    nvshmem_quiet as "nvshmem_quiet": PFN_nvshmem_quiet;
    nvshmem_fence as "nvshmem_fence": PFN_nvshmem_fence;
}

/// Resolve (once) and return the process-wide NVSHMEM host loader. Returns
/// `LoaderError::LibraryNotFound` on hosts without NVSHMEM installed.
pub fn nvshmem() -> Result<&'static Nvshmem, LoaderError> {
    static NVSHMEM: OnceLock<Nvshmem> = OnceLock::new();
    if let Some(n) = NVSHMEM.get() {
        return Ok(n);
    }
    let lib = Library::open("nvshmem", nvshmem_candidates())?;
    let n = Nvshmem::empty(lib);
    let _ = NVSHMEM.set(n);
    Ok(NVSHMEM.get().expect("OnceLock set or lost race"))
}
