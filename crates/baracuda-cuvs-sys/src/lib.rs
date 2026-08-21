//! Raw FFI + dynamic loader for NVIDIA cuVS (GPU vector search / ANN).
//!
//! cuVS is part of NVIDIA RAPIDS and ships as `libcuvs.so` (which pulls in
//! `libraft.so`). There is no native Windows distribution today — RAPIDS is
//! Linux / WSL2 only. To keep the workspace building everywhere, the entire
//! cuVS surface lives behind the off-by-default `cuvs` cargo feature and is
//! resolved lazily at runtime via `libloading` — enabling the feature never
//! adds a link-time dependency, and [`cuvs()`] returns
//! `LoaderError::LibraryNotFound` on hosts without a RAPIDS install.
//!
//! # Data interchange
//!
//! cuVS's C API takes datasets / queries / outputs as DLPack
//! [`DLManagedTensor`] pointers rather than bare device pointers. This crate
//! defines the (stable-ABI) DLPack structs so callers can hand cuVS a tensor
//! view over a baracuda `DeviceBuffer`. cuVS only *reads* the tensor metadata
//! during the call and does not take ownership, so input tensors may use a
//! `deleter` of `None`.
//!
//! # Status codes
//!
//! Note the unusual convention: [`cuvsError_t::SUCCESS`] is `1`, not `0`
//! (`CUVS_ERROR` is `0`). [`cuvsError_t::is_success`] encodes this.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

#[cfg(feature = "cuvs")]
mod ffi {
    use core::ffi::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    use baracuda_core::{Library, LoaderError};
    use baracuda_cuda_sys::runtime::cudaStream_t;
    use baracuda_types::CudaStatus;

    // ---- DLPack (stable ABI; mirrors dlpack.h) ---------------------------

    /// `DLDeviceType::kDLCPU`.
    pub const K_DL_CPU: c_int = 1;
    /// `DLDeviceType::kDLCUDA`.
    pub const K_DL_CUDA: c_int = 2;

    /// `DLDataTypeCode::kDLInt`.
    pub const K_DL_INT: u8 = 0;
    /// `DLDataTypeCode::kDLUInt`.
    pub const K_DL_UINT: u8 = 1;
    /// `DLDataTypeCode::kDLFloat`.
    pub const K_DL_FLOAT: u8 = 2;
    /// `DLDataTypeCode::kDLBfloat`.
    pub const K_DL_BFLOAT: u8 = 4;

    /// DLPack device descriptor.
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct DLDevice {
        /// `DLDeviceType` (`K_DL_CUDA` for device memory).
        pub device_type: c_int,
        /// Device ordinal (the CUDA device index for `K_DL_CUDA`).
        pub device_id: i32,
    }

    /// DLPack scalar type descriptor: `code` + `bits` (+ SIMD `lanes`).
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct DLDataType {
        /// Type-family code (`K_DL_INT` / `K_DL_UINT` / `K_DL_FLOAT` / `K_DL_BFLOAT`).
        pub code: u8,
        /// Bit width of one scalar (e.g. `32` for f32, `16` for f16/bf16).
        pub bits: u8,
        /// SIMD lane count — `1` for an ordinary scalar element.
        pub lanes: u16,
    }

    /// A non-owning tensor view (DLPack `DLTensor`).
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct DLTensor {
        /// Base pointer to the tensor's data (device or host per [`device`](Self::device)).
        pub data: *mut c_void,
        /// Physical device the [`data`](Self::data) pointer lives on.
        pub device: DLDevice,
        /// Number of dimensions (the length of [`shape`](Self::shape)).
        pub ndim: i32,
        /// Element scalar type.
        pub dtype: DLDataType,
        /// `ndim` shape entries. Must outlive any FFI call that reads it.
        pub shape: *mut i64,
        /// Strides in elements, or null for compact row-major.
        pub strides: *mut i64,
        /// Offset in **bytes** from `data` to the first element (usually `0`).
        pub byte_offset: u64,
    }

    /// An owning DLPack tensor with an optional `deleter`. For tensors handed
    /// *to* cuVS as inputs, `manager_ctx` / `deleter` may be null — cuVS does
    /// not take ownership of caller buffers.
    #[repr(C)]
    pub struct DLManagedTensor {
        /// The tensor view.
        pub dl_tensor: DLTensor,
        /// Opaque context passed to [`deleter`](Self::deleter); null when unmanaged.
        pub manager_ctx: *mut c_void,
        /// Destructor the owner calls to release the tensor, or `None` for a
        /// caller-owned input buffer cuVS only borrows.
        pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
    }

    impl core::fmt::Debug for DLManagedTensor {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("DLManagedTensor")
                .field("dl_tensor", &self.dl_tensor)
                .finish_non_exhaustive()
        }
    }

    // ---- status ----------------------------------------------------------

    /// cuVS status code. **`SUCCESS` is `1`** (`CUVS_ERROR` is `0`).
    #[repr(transparent)]
    #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
    pub struct cuvsError_t(pub c_int);

    impl cuvsError_t {
        /// `CUVS_ERROR` — the call failed; see `cuvsGetLastErrorText`.
        pub const ERROR: Self = Self(0);
        /// `CUVS_SUCCESS`.
        pub const SUCCESS: Self = Self(1);

        /// Whether this status is `SUCCESS` (`1`) — note the inverted convention
        /// (`0` is the error code, not success).
        #[inline]
        pub const fn is_success(self) -> bool {
            self.0 == 1
        }
    }

    impl CudaStatus for cuvsError_t {
        fn code(self) -> i32 {
            self.0
        }
        fn name(self) -> &'static str {
            match self.0 {
                0 => "CUVS_ERROR",
                1 => "CUVS_SUCCESS",
                _ => "cuvsUnrecognizedStatus",
            }
        }
        fn description(self) -> &'static str {
            match self.0 {
                0 => "cuVS call failed (see cuvsGetLastErrorText)",
                1 => "success",
                _ => "unrecognized cuVS status code",
            }
        }
        fn is_success(self) -> bool {
            cuvsError_t::is_success(self)
        }
        fn library(self) -> &'static str {
            "cuvs"
        }
    }

    /// Opaque cuVS resources handle (`typedef uintptr_t cuvsResources_t`).
    pub type cuvsResources_t = usize;

    // ---- distance metrics (cuvs/distance/distance.h) ---------------------

    /// `cuvsDistanceType` — the full RAFT/cuVS distance enum.
    #[repr(i32)]
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum cuvsDistanceType {
        /// Squared Euclidean (L2²) via the expanded `‖x‖² + ‖y‖² − 2·x·y` form.
        L2Expanded = 0,
        /// Euclidean (√L2) via the expanded form.
        L2SqrtExpanded = 1,
        /// Cosine distance (`1 − cosine similarity`), expanded form.
        CosineExpanded = 2,
        /// Manhattan / city-block distance (L1).
        L1 = 3,
        /// Squared Euclidean (L2²) via the direct (unexpanded) computation.
        L2Unexpanded = 4,
        /// Euclidean (√L2) via the direct (unexpanded) computation.
        L2SqrtUnexpanded = 5,
        /// Inner (dot) product — larger is nearer.
        InnerProduct = 6,
        /// Chebyshev / L-infinity (max coordinate difference).
        Linf = 7,
        /// Canberra distance.
        Canberra = 8,
        /// Minkowski Lp distance with exponent `p` (`p` given by `metric_arg`).
        LpUnexpanded = 9,
        /// Pearson-correlation distance, expanded form.
        CorrelationExpanded = 10,
        /// Jaccard distance, expanded form.
        JaccardExpanded = 11,
        /// Hellinger distance, expanded form.
        HellingerExpanded = 12,
        /// Great-circle (haversine) distance on the unit sphere.
        Haversine = 13,
        /// Bray–Curtis dissimilarity.
        BrayCurtis = 14,
        /// Jensen–Shannon divergence.
        JensenShannon = 15,
        /// Hamming distance (fraction of differing coordinates), direct form.
        HammingUnexpanded = 16,
        /// Kullback–Leibler divergence.
        KLDivergence = 17,
        /// Russell–Rao distance, expanded form.
        RusselRaoExpanded = 18,
        /// Dice (Sørensen–Dice) distance, expanded form.
        DiceExpanded = 19,
        /// Bitwise Hamming distance over packed-bit vectors.
        BitwiseHamming = 20,
        /// Precomputed distances — the caller supplies the distance matrix.
        Precomputed = 100,
    }

    // ---- prefilter (cuvs/neighbors/common.h) -----------------------------

    /// `cuvsFilterType`.
    #[repr(C)]
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum cuvsFilterType {
        /// No filter applied — pass for an unfiltered search.
        NO_FILTER = 0,
        /// A bitset over the dataset rows — one bit per vector (kept/excluded).
        BITSET = 1,
        /// A bitmap over the (query × dataset) pairs — per-query allow-lists.
        BITMAP = 2,
    }

    /// `cuvsFilter` — an optional search prefilter. Use [`cuvsFilter::none`]
    /// for an ordinary unfiltered search.
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct cuvsFilter {
        /// `uintptr_t` to a bitset/bitmap, or `0` for `NO_FILTER`.
        pub addr: usize,
        /// Which structure [`addr`](Self::addr) points at (or `NO_FILTER`).
        pub type_: cuvsFilterType,
    }

    impl cuvsFilter {
        /// The "no prefilter" sentinel.
        #[inline]
        pub const fn none() -> Self {
            Self {
                addr: 0,
                type_: cuvsFilterType::NO_FILTER,
            }
        }
    }

    // ---- IVF-Flat (cuvs/neighbors/ivf_flat.h) ----------------------------

    /// `cuvsIvfFlatIndexParams`.
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct cuvsIvfFlatIndexParams {
        /// Distance metric the index is built for.
        pub metric: cuvsDistanceType,
        /// Metric parameter (e.g. the `p` exponent for `LpUnexpanded`).
        pub metric_arg: f32,
        /// Add the dataset vectors to the lists during build (vs. an empty index
        /// populated later by `extend`).
        pub add_data_on_build: bool,
        /// Number of inverted lists (Voronoi cells / k-means clusters).
        pub n_lists: u32,
        /// k-means iterations used to train the list centroids.
        pub kmeans_n_iters: u32,
        /// Fraction of the dataset sampled to train the centroids (`0..=1`).
        pub kmeans_trainset_fraction: f64,
        /// Re-fit each list's center to its assigned vectors after build (higher
        /// recall, but the centers drift from the k-means partition).
        pub adaptive_centers: bool,
        /// Trade build/search speed for lower peak memory during construction.
        pub conservative_memory_allocation: bool,
    }
    /// Pointer to a [`cuvsIvfFlatIndexParams`] (opaque handle in the C API).
    pub type cuvsIvfFlatIndexParams_t = *mut cuvsIvfFlatIndexParams;

    /// `cuvsIvfFlatSearchParams`.
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct cuvsIvfFlatSearchParams {
        /// Number of inverted lists probed per query (recall/latency trade-off).
        pub n_probes: u32,
    }
    /// Pointer to a [`cuvsIvfFlatSearchParams`] (opaque handle in the C API).
    pub type cuvsIvfFlatSearchParams_t = *mut cuvsIvfFlatSearchParams;

    /// `cuvsIvfFlatIndex` — opaque index handle plus its trained dtype.
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct cuvsIvfFlatIndex {
        /// `uintptr_t` to the underlying C++ index object.
        pub addr: usize,
        /// Element dtype the index was trained on.
        pub dtype: DLDataType,
    }
    /// Pointer to a [`cuvsIvfFlatIndex`] (opaque handle in the C API).
    pub type cuvsIvfFlatIndex_t = *mut cuvsIvfFlatIndex;

    // ---- Brute-force (cuvs/neighbors/brute_force.h) ----------------------

    /// `cuvsBruteForceIndex` — opaque index handle plus its trained dtype.
    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct cuvsBruteForceIndex {
        /// `uintptr_t` to the underlying C++ index object.
        pub addr: usize,
        /// Element dtype the index was trained on.
        pub dtype: DLDataType,
    }
    /// Pointer to a [`cuvsBruteForceIndex`] (opaque handle in the C API).
    pub type cuvsBruteForceIndex_t = *mut cuvsBruteForceIndex;

    // ---- function-pointer types ------------------------------------------

    // Core / resources.
    /// `cuvsResourcesCreate` — allocate a resources handle (stream, memory pools).
    pub type PFN_cuvsResourcesCreate =
        unsafe extern "C" fn(res: *mut cuvsResources_t) -> cuvsError_t;
    /// `cuvsResourcesDestroy` — free a resources handle.
    pub type PFN_cuvsResourcesDestroy = unsafe extern "C" fn(res: cuvsResources_t) -> cuvsError_t;
    /// `cuvsStreamSet` — bind the CUDA stream cuVS runs its work on.
    pub type PFN_cuvsStreamSet =
        unsafe extern "C" fn(res: cuvsResources_t, stream: cudaStream_t) -> cuvsError_t;
    /// `cuvsStreamGet` — read back the resources' current CUDA stream.
    pub type PFN_cuvsStreamGet =
        unsafe extern "C" fn(res: cuvsResources_t, stream: *mut cudaStream_t) -> cuvsError_t;
    /// `cuvsStreamSync` — block until the resources' stream is idle.
    pub type PFN_cuvsStreamSync = unsafe extern "C" fn(res: cuvsResources_t) -> cuvsError_t;
    /// `cuvsGetLastErrorText` — the last error message for this thread (or null).
    pub type PFN_cuvsGetLastErrorText = unsafe extern "C" fn() -> *const c_char;

    // IVF-Flat.
    /// `cuvsIvfFlatIndexParamsCreate` — allocate default IVF-Flat build params.
    pub type PFN_cuvsIvfFlatIndexParamsCreate =
        unsafe extern "C" fn(params: *mut cuvsIvfFlatIndexParams_t) -> cuvsError_t;
    /// `cuvsIvfFlatIndexParamsDestroy` — free IVF-Flat build params.
    pub type PFN_cuvsIvfFlatIndexParamsDestroy =
        unsafe extern "C" fn(params: cuvsIvfFlatIndexParams_t) -> cuvsError_t;
    /// `cuvsIvfFlatSearchParamsCreate` — allocate default IVF-Flat search params.
    pub type PFN_cuvsIvfFlatSearchParamsCreate =
        unsafe extern "C" fn(params: *mut cuvsIvfFlatSearchParams_t) -> cuvsError_t;
    /// `cuvsIvfFlatSearchParamsDestroy` — free IVF-Flat search params.
    pub type PFN_cuvsIvfFlatSearchParamsDestroy =
        unsafe extern "C" fn(params: cuvsIvfFlatSearchParams_t) -> cuvsError_t;
    /// `cuvsIvfFlatIndexCreate` — allocate an empty IVF-Flat index handle.
    pub type PFN_cuvsIvfFlatIndexCreate =
        unsafe extern "C" fn(index: *mut cuvsIvfFlatIndex_t) -> cuvsError_t;
    /// `cuvsIvfFlatIndexDestroy` — free an IVF-Flat index.
    pub type PFN_cuvsIvfFlatIndexDestroy =
        unsafe extern "C" fn(index: cuvsIvfFlatIndex_t) -> cuvsError_t;
    /// `cuvsIvfFlatBuild` — train the lists on `dataset` and populate `index`.
    pub type PFN_cuvsIvfFlatBuild = unsafe extern "C" fn(
        res: cuvsResources_t,
        params: cuvsIvfFlatIndexParams_t,
        dataset: *mut DLManagedTensor,
        index: cuvsIvfFlatIndex_t,
    ) -> cuvsError_t;
    /// `cuvsIvfFlatSearch` — k-NN search: fill `neighbors` + `distances` for
    /// `queries`, honoring an optional prefilter.
    pub type PFN_cuvsIvfFlatSearch = unsafe extern "C" fn(
        res: cuvsResources_t,
        search_params: cuvsIvfFlatSearchParams_t,
        index: cuvsIvfFlatIndex_t,
        queries: *mut DLManagedTensor,
        neighbors: *mut DLManagedTensor,
        distances: *mut DLManagedTensor,
        filter: cuvsFilter,
    ) -> cuvsError_t;

    // Brute-force.
    /// `cuvsBruteForceIndexCreate` — allocate an empty brute-force index handle.
    pub type PFN_cuvsBruteForceIndexCreate =
        unsafe extern "C" fn(index: *mut cuvsBruteForceIndex_t) -> cuvsError_t;
    /// `cuvsBruteForceIndexDestroy` — free a brute-force index.
    pub type PFN_cuvsBruteForceIndexDestroy =
        unsafe extern "C" fn(index: cuvsBruteForceIndex_t) -> cuvsError_t;
    /// `cuvsBruteForceBuild` — wrap `dataset` as an exact-search index under `metric`.
    pub type PFN_cuvsBruteForceBuild = unsafe extern "C" fn(
        res: cuvsResources_t,
        dataset: *mut DLManagedTensor,
        metric: cuvsDistanceType,
        metric_arg: f32,
        index: cuvsBruteForceIndex_t,
    ) -> cuvsError_t;
    /// `cuvsBruteForceSearch` — exact k-NN: fill `neighbors` + `distances` for
    /// `queries`, honoring an optional prefilter.
    pub type PFN_cuvsBruteForceSearch = unsafe extern "C" fn(
        res: cuvsResources_t,
        index: cuvsBruteForceIndex_t,
        queries: *mut DLManagedTensor,
        neighbors: *mut DLManagedTensor,
        distances: *mut DLManagedTensor,
        prefilter: cuvsFilter,
    ) -> cuvsError_t;

    // ---- loader ----------------------------------------------------------

    fn cuvs_candidates() -> &'static [&'static str] {
        #[cfg(target_os = "linux")]
        {
            // RAPIDS ships an unversioned symlink plus SONAME-versioned files.
            &["libcuvs.so", "libcuvs.so.0"]
        }
        #[cfg(target_os = "windows")]
        {
            &["cuvs.dll", "libcuvs.dll"]
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            &[]
        }
    }

    macro_rules! cuvs_fns {
        ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
            /// Resolved cuVS entry points (lazy, cached per symbol).
            pub struct Cuvs {
                lib: Library,
                $($name: OnceLock<$pfn>,)*
            }
            impl core::fmt::Debug for Cuvs {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.debug_struct("Cuvs").field("lib", &self.lib).finish_non_exhaustive()
                }
            }
            impl Cuvs {
                $(
                    /// Resolve (and cache) this cuVS entry point from the loaded
                    /// library. Returns [`LoaderError`] if the symbol is absent.
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

    cuvs_fns! {
        cuvs_resources_create as "cuvsResourcesCreate": PFN_cuvsResourcesCreate;
        cuvs_resources_destroy as "cuvsResourcesDestroy": PFN_cuvsResourcesDestroy;
        cuvs_stream_set as "cuvsStreamSet": PFN_cuvsStreamSet;
        cuvs_stream_get as "cuvsStreamGet": PFN_cuvsStreamGet;
        cuvs_stream_sync as "cuvsStreamSync": PFN_cuvsStreamSync;
        cuvs_get_last_error_text as "cuvsGetLastErrorText": PFN_cuvsGetLastErrorText;
        cuvs_ivf_flat_index_params_create as "cuvsIvfFlatIndexParamsCreate": PFN_cuvsIvfFlatIndexParamsCreate;
        cuvs_ivf_flat_index_params_destroy as "cuvsIvfFlatIndexParamsDestroy": PFN_cuvsIvfFlatIndexParamsDestroy;
        cuvs_ivf_flat_search_params_create as "cuvsIvfFlatSearchParamsCreate": PFN_cuvsIvfFlatSearchParamsCreate;
        cuvs_ivf_flat_search_params_destroy as "cuvsIvfFlatSearchParamsDestroy": PFN_cuvsIvfFlatSearchParamsDestroy;
        cuvs_ivf_flat_index_create as "cuvsIvfFlatIndexCreate": PFN_cuvsIvfFlatIndexCreate;
        cuvs_ivf_flat_index_destroy as "cuvsIvfFlatIndexDestroy": PFN_cuvsIvfFlatIndexDestroy;
        cuvs_ivf_flat_build as "cuvsIvfFlatBuild": PFN_cuvsIvfFlatBuild;
        cuvs_ivf_flat_search as "cuvsIvfFlatSearch": PFN_cuvsIvfFlatSearch;
        cuvs_brute_force_index_create as "cuvsBruteForceIndexCreate": PFN_cuvsBruteForceIndexCreate;
        cuvs_brute_force_index_destroy as "cuvsBruteForceIndexDestroy": PFN_cuvsBruteForceIndexDestroy;
        cuvs_brute_force_build as "cuvsBruteForceBuild": PFN_cuvsBruteForceBuild;
        cuvs_brute_force_search as "cuvsBruteForceSearch": PFN_cuvsBruteForceSearch;
    }

    /// Resolve (and cache) the cuVS dynamic library. Returns
    /// `LoaderError::LibraryNotFound` on hosts without a RAPIDS install.
    pub fn cuvs() -> Result<&'static Cuvs, LoaderError> {
        static CUVS: OnceLock<Cuvs> = OnceLock::new();
        if let Some(c) = CUVS.get() {
            return Ok(c);
        }
        let lib = Library::open("cuvs", cuvs_candidates())?;
        let c = Cuvs::empty(lib);
        let _ = CUVS.set(c);
        Ok(CUVS.get().expect("OnceLock set or lost race"))
    }
}

#[cfg(feature = "cuvs")]
pub use ffi::*;
