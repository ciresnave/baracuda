//! Safe Rust wrappers for NVIDIA cuVS — GPU vector search / approximate
//! nearest neighbours (ANN), part of NVIDIA RAPIDS.
//!
//! Tier-1 coverage: the [`IvfFlat`] inverted-file index and exact
//! [`BruteForce`] k-NN, over `f32` (and `f16` behind the `half-crate`
//! feature) vectors, with L2 / cosine / inner-product metrics.
//!
//! The whole API is behind the off-by-default **`cuvs`** cargo feature
//! because cuVS ships only with RAPIDS (`libcuvs.so` + `libraft.so`), which
//! has no native Windows distribution (Linux / WSL2 only). Symbols are
//! resolved lazily at runtime, so on a host without cuVS the constructors
//! return [`baracuda_core::Error::Loader`] rather than failing to link.
//!
//! ```no_run
//! # #[cfg(feature = "cuvs")]
//! # fn demo() -> baracuda_cuvs::Result<()> {
//! use baracuda_cuvs::{Resources, IvfFlat, IvfFlatBuildParams, IvfFlatSearchParams, Metric};
//! use baracuda_driver::{Context, Device, DeviceBuffer};
//!
//! let ctx = Context::new(&Device::get(0)?)?;
//! let res = Resources::new()?;
//!
//! // 1000 vectors of dim 128, row-major, already on the device.
//! let dataset: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1000 * 128)?;
//! let index = IvfFlat::<f32>::build(
//!     &res, &dataset, 1000, 128,
//!     IvfFlatBuildParams { metric: Metric::L2Expanded, n_lists: 50, ..Default::default() },
//! )?;
//!
//! // 5 queries, k = 10 nearest.
//! let queries: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 5 * 128)?;
//! let (neighbors, distances) = index.search(&res, &queries, 5, 10, IvfFlatSearchParams::default())?;
//! # let _ = (neighbors, distances); Ok(())
//! # }
//! ```
//!
//! # Stream ordering
//!
//! cuVS enqueues work on the stream bound to its [`Resources`] (the default
//! stream unless [`Resources::set_stream`] was called). Both `build` and
//! `search` here call [`Resources::sync`] before returning, so the results
//! are ready to read on the host — the API is synchronous for simplicity.
//!
//! # Lifetime quirk: brute-force keeps a dataset view
//!
//! cuVS's brute-force index stores a *non-owning view* of the dataset (plus
//! precomputed norms), so the dataset buffer must outlive the index. This is
//! encoded in Rust: [`BruteForce::build`] borrows the dataset and the
//! returned [`BruteForce`] carries that borrow's lifetime. IVF-Flat (with the
//! default `add_data_on_build = true`) copies the vectors into the index, so
//! [`IvfFlat`] owns its data and has no such borrow.

#![warn(missing_debug_implementations)]

#[cfg(feature = "cuvs")]
mod imp {
    use core::ffi::{CStr, c_void};
    use core::marker::PhantomData;

    use baracuda_cuda_sys::runtime::cudaStream_t;
    use baracuda_cuvs_sys::{
        DLDataType, DLDevice, DLManagedTensor, DLTensor, K_DL_CUDA, K_DL_FLOAT, K_DL_INT, cuvs,
        cuvsBruteForceIndex_t, cuvsDistanceType, cuvsError_t, cuvsFilter, cuvsIvfFlatIndex_t,
        cuvsIvfFlatIndexParams_t, cuvsIvfFlatSearchParams_t, cuvsResources_t,
    };
    use baracuda_driver::{Context, DeviceBuffer, Stream};
    use baracuda_types::DeviceRepr;

    /// Error type for cuVS operations.
    pub type Error = baracuda_core::Error<cuvsError_t>;
    /// Result alias.
    pub type Result<T, E = Error> = core::result::Result<T, E>;

    #[inline]
    fn check(status: cuvsError_t) -> Result<()> {
        Error::check(status)
    }

    /// Collapse a baracuda-driver error (e.g. a failed output-buffer
    /// allocation) into a cuVS `Error`. cuVS's status enum has no dedicated
    /// allocation-failure code, so this maps to the generic `CUVS_ERROR`
    /// (mirrors `baracuda-cusolver`'s `alloc_fail`).
    #[inline]
    fn alloc_fail<E>(_e: E) -> Error {
        Error::Status {
            status: cuvsError_t::ERROR,
        }
    }

    /// The most recent cuVS error message (from `cuvsGetLastErrorText`), or
    /// `None` if cuVS isn't loadable or has no message. Useful to enrich the
    /// generic [`Error::Status`] after a failed call.
    pub fn last_error_text() -> Option<String> {
        let c = cuvs().ok()?;
        let f = c.cuvs_get_last_error_text().ok()?;
        let p = unsafe { f() };
        if p.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    // ---- element types ---------------------------------------------------

    /// Vector element type understood by cuVS. Sealed: implemented for `f32`
    /// (always) and `half::f16` (under the `half-crate` feature).
    pub trait CuvsScalar: DeviceRepr + sealed::Sealed {
        #[doc(hidden)]
        fn dl_dtype() -> DLDataType;
    }

    mod sealed {
        pub trait Sealed {}
    }

    impl CuvsScalar for f32 {
        fn dl_dtype() -> DLDataType {
            DLDataType {
                code: K_DL_FLOAT,
                bits: 32,
                lanes: 1,
            }
        }
    }
    impl sealed::Sealed for f32 {}

    #[cfg(feature = "half-crate")]
    impl CuvsScalar for half::f16 {
        fn dl_dtype() -> DLDataType {
            DLDataType {
                code: K_DL_FLOAT,
                bits: 16,
                lanes: 1,
            }
        }
    }
    #[cfg(feature = "half-crate")]
    impl sealed::Sealed for half::f16 {}

    /// DLPack dtype for the `int64` neighbour-index output.
    fn neighbors_dtype() -> DLDataType {
        DLDataType {
            code: K_DL_INT,
            bits: 64,
            lanes: 1,
        }
    }

    /// DLPack dtype for the `float32` distance output.
    fn distances_dtype() -> DLDataType {
        DLDataType {
            code: K_DL_FLOAT,
            bits: 32,
            lanes: 1,
        }
    }

    // ---- distance metric -------------------------------------------------

    /// Distance metric for index build / search.
    #[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
    pub enum Metric {
        /// Squared Euclidean with the expansion `||x||² + ||y||² - 2x·y`
        /// (cuVS default; ranks identically to true L2).
        #[default]
        L2Expanded,
        /// True (non-squared) Euclidean distance.
        L2SqrtExpanded,
        /// Cosine distance (`1 - cosine similarity`).
        Cosine,
        /// Negative inner product (larger dot product ⇒ smaller distance).
        InnerProduct,
    }

    impl Metric {
        fn raw(self) -> cuvsDistanceType {
            match self {
                Metric::L2Expanded => cuvsDistanceType::L2Expanded,
                Metric::L2SqrtExpanded => cuvsDistanceType::L2SqrtExpanded,
                Metric::Cosine => cuvsDistanceType::CosineExpanded,
                Metric::InnerProduct => cuvsDistanceType::InnerProduct,
            }
        }
    }

    // ---- resources -------------------------------------------------------

    /// A cuVS resources handle (CUDA stream, device, workspace allocators).
    /// One per worker thread; cheap to keep around and reuse across builds
    /// and searches.
    pub struct Resources {
        raw: cuvsResources_t,
    }

    // SAFETY: the handle is an opaque uintptr_t; cuVS serializes work on its
    // bound stream. Moving across threads is fine; concurrent use is not
    // (mirrors every other baracuda handle wrapper).
    unsafe impl Send for Resources {}

    impl core::fmt::Debug for Resources {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("cuvs::Resources")
                .field("raw", &self.raw)
                .finish()
        }
    }

    impl Resources {
        /// Create a fresh resources handle.
        pub fn new() -> Result<Self> {
            let c = cuvs()?;
            let f = c.cuvs_resources_create()?;
            let mut raw: cuvsResources_t = 0;
            check(unsafe { f(&mut raw) })?;
            Ok(Self { raw })
        }

        /// Bind a CUDA stream so cuVS enqueues its work there. Without this,
        /// cuVS uses its internal default stream.
        pub fn set_stream(&self, stream: &Stream) -> Result<()> {
            let c = cuvs()?;
            let f = c.cuvs_stream_set()?;
            check(unsafe { f(self.raw, stream.as_raw() as cudaStream_t) })
        }

        /// Block until all work on the bound stream completes.
        pub fn sync(&self) -> Result<()> {
            let c = cuvs()?;
            let f = c.cuvs_stream_sync()?;
            check(unsafe { f(self.raw) })
        }

        /// Raw `cuvsResources_t`. Use with care — `Resources` still owns it.
        #[inline]
        pub fn as_raw(&self) -> cuvsResources_t {
            self.raw
        }
    }

    impl Drop for Resources {
        fn drop(&mut self) {
            if let Ok(c) = cuvs() {
                if let Ok(f) = c.cuvs_resources_destroy() {
                    let _ = unsafe { f(self.raw) };
                }
            }
        }
    }

    // ---- DLPack helpers --------------------------------------------------

    /// Build a non-owning row-major DLPack view over a device pointer.
    /// `shape` must outlive every FFI call that reads the returned tensor.
    fn device_tensor(
        data: *mut c_void,
        shape: &mut [i64],
        dtype: DLDataType,
        device_id: i32,
    ) -> DLManagedTensor {
        DLManagedTensor {
            dl_tensor: DLTensor {
                data,
                device: DLDevice {
                    device_type: K_DL_CUDA,
                    device_id,
                },
                ndim: shape.len() as i32,
                dtype,
                shape: shape.as_mut_ptr(),
                // null ⇒ compact row-major, per the DLPack contract.
                strides: core::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: core::ptr::null_mut(),
            deleter: None,
        }
    }

    #[inline]
    fn device_id_of<T: DeviceRepr>(buf: &DeviceBuffer<T>) -> i32 {
        buf.context().device().ordinal()
    }

    /// Allocate the `(neighbors: i64, distances: f32)` output pair for a
    /// search of `n_queries` queries returning `k` results each.
    fn alloc_outputs(
        ctx: &Context,
        n_queries: usize,
        k: usize,
    ) -> Result<(DeviceBuffer<i64>, DeviceBuffer<f32>)> {
        let n = n_queries * k;
        let neighbors = DeviceBuffer::<i64>::new(ctx, n).map_err(alloc_fail)?;
        let distances = DeviceBuffer::<f32>::new(ctx, n).map_err(alloc_fail)?;
        Ok((neighbors, distances))
    }

    // ---- IVF-Flat --------------------------------------------------------

    /// Build-time parameters for an [`IvfFlat`] index. Defaults match cuVS's
    /// own (`n_lists = 1024`, `kmeans_n_iters = 20`,
    /// `kmeans_trainset_fraction = 0.5`, `add_data_on_build = true`).
    ///
    /// `n_lists` is the number of inverted-file clusters; it should be well
    /// below the dataset row count (a common rule of thumb is `~sqrt(n_rows)`).
    #[derive(Copy, Clone, Debug)]
    pub struct IvfFlatBuildParams {
        pub metric: Metric,
        pub n_lists: u32,
        pub kmeans_n_iters: u32,
        pub kmeans_trainset_fraction: f64,
        /// Add the dataset to the index during build. Keep `true` so the
        /// index owns its vectors (the Rust wrapper relies on this — see the
        /// crate-level lifetime note).
        pub add_data_on_build: bool,
        pub adaptive_centers: bool,
        pub conservative_memory_allocation: bool,
    }

    impl Default for IvfFlatBuildParams {
        fn default() -> Self {
            Self {
                metric: Metric::default(),
                n_lists: 1024,
                kmeans_n_iters: 20,
                kmeans_trainset_fraction: 0.5,
                add_data_on_build: true,
                adaptive_centers: false,
                conservative_memory_allocation: false,
            }
        }
    }

    /// Search-time parameters for an [`IvfFlat`] index.
    #[derive(Copy, Clone, Debug)]
    pub struct IvfFlatSearchParams {
        /// Number of inverted lists to probe per query. Higher ⇒ better
        /// recall, slower search. Clamped to `n_lists` by cuVS.
        pub n_probes: u32,
    }

    impl Default for IvfFlatSearchParams {
        fn default() -> Self {
            Self { n_probes: 20 }
        }
    }

    /// A trained IVF-Flat index. Owns its vectors (built with
    /// `add_data_on_build = true`).
    pub struct IvfFlat<T: CuvsScalar> {
        raw: cuvsIvfFlatIndex_t,
        dim: usize,
        _marker: PhantomData<T>,
    }

    // SAFETY: opaque handle; same contract as `Resources`.
    unsafe impl<T: CuvsScalar> Send for IvfFlat<T> {}

    impl<T: CuvsScalar> core::fmt::Debug for IvfFlat<T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("IvfFlat")
                .field("dim", &self.dim)
                .field("dtype", &core::any::type_name::<T>())
                .finish_non_exhaustive()
        }
    }

    impl<T: CuvsScalar> IvfFlat<T> {
        /// Train an IVF-Flat index over `dataset` — `n_rows` row-major vectors
        /// of dimension `dim` (so `dataset.len() == n_rows * dim`).
        pub fn build(
            res: &Resources,
            dataset: &DeviceBuffer<T>,
            n_rows: usize,
            dim: usize,
            params: IvfFlatBuildParams,
        ) -> Result<Self> {
            assert_eq!(
                dataset.len(),
                n_rows * dim,
                "dataset length {} != n_rows {n_rows} * dim {dim}",
                dataset.len(),
            );
            let c = cuvs()?;

            // Create + populate the index params (mutate named fields so we
            // stay forward-compatible if cuVS grows the struct).
            let mut p: cuvsIvfFlatIndexParams_t = core::ptr::null_mut();
            check(unsafe { c.cuvs_ivf_flat_index_params_create()?(&mut p) })?;
            // SAFETY: `p` is a live params allocation from cuVS.
            unsafe {
                (*p).metric = params.metric.raw();
                (*p).n_lists = params.n_lists;
                (*p).kmeans_n_iters = params.kmeans_n_iters;
                (*p).kmeans_trainset_fraction = params.kmeans_trainset_fraction;
                (*p).add_data_on_build = params.add_data_on_build;
                (*p).adaptive_centers = params.adaptive_centers;
                (*p).conservative_memory_allocation = params.conservative_memory_allocation;
            }

            let mut index: cuvsIvfFlatIndex_t = core::ptr::null_mut();
            let build_result = (|| -> Result<()> {
                check(unsafe { c.cuvs_ivf_flat_index_create()?(&mut index) })?;
                let mut shape = [n_rows as i64, dim as i64];
                let mut ds = device_tensor(
                    dataset.as_raw().0 as *mut c_void,
                    &mut shape,
                    T::dl_dtype(),
                    device_id_of(dataset),
                );
                check(unsafe { c.cuvs_ivf_flat_build()?(res.raw, p, &mut ds, index) })?;
                res.sync()
            })();

            // Params are consumed by build; destroy regardless of outcome.
            if let Ok(f) = c.cuvs_ivf_flat_index_params_destroy() {
                let _ = unsafe { f(p) };
            }

            match build_result {
                Ok(()) => Ok(Self {
                    raw: index,
                    dim,
                    _marker: PhantomData,
                }),
                Err(e) => {
                    if !index.is_null() {
                        if let Ok(f) = c.cuvs_ivf_flat_index_destroy() {
                            let _ = unsafe { f(index) };
                        }
                    }
                    Err(e)
                }
            }
        }

        /// Search for the `k` nearest neighbours of each of `n_queries`
        /// row-major query vectors (so `queries.len() == n_queries * dim`).
        ///
        /// Returns `(neighbors, distances)`, each a row-major
        /// `n_queries × k` device buffer — `neighbors` holds `i64` dataset
        /// row indices, `distances` holds `f32` metric values. The bound
        /// stream is synchronized before returning.
        pub fn search(
            &self,
            res: &Resources,
            queries: &DeviceBuffer<T>,
            n_queries: usize,
            k: usize,
            params: IvfFlatSearchParams,
        ) -> Result<(DeviceBuffer<i64>, DeviceBuffer<f32>)> {
            assert_eq!(
                queries.len(),
                n_queries * self.dim,
                "queries length {} != n_queries {n_queries} * dim {}",
                queries.len(),
                self.dim,
            );
            let c = cuvs()?;

            let mut sp: cuvsIvfFlatSearchParams_t = core::ptr::null_mut();
            check(unsafe { c.cuvs_ivf_flat_search_params_create()?(&mut sp) })?;
            unsafe {
                (*sp).n_probes = params.n_probes;
            }

            let (neighbors, distances) = alloc_outputs(queries.context(), n_queries, k)?;

            let result = (|| -> Result<()> {
                let dev = device_id_of(queries);
                let mut q_shape = [n_queries as i64, self.dim as i64];
                let mut n_shape = [n_queries as i64, k as i64];
                let mut d_shape = [n_queries as i64, k as i64];
                let mut q = device_tensor(
                    queries.as_raw().0 as *mut c_void,
                    &mut q_shape,
                    T::dl_dtype(),
                    dev,
                );
                let mut n = device_tensor(
                    neighbors.as_raw().0 as *mut c_void,
                    &mut n_shape,
                    neighbors_dtype(),
                    dev,
                );
                let mut d = device_tensor(
                    distances.as_raw().0 as *mut c_void,
                    &mut d_shape,
                    distances_dtype(),
                    dev,
                );
                check(unsafe {
                    c.cuvs_ivf_flat_search()?(
                        res.raw,
                        sp,
                        self.raw,
                        &mut q,
                        &mut n,
                        &mut d,
                        cuvsFilter::none(),
                    )
                })?;
                res.sync()
            })();

            if let Ok(f) = c.cuvs_ivf_flat_search_params_destroy() {
                let _ = unsafe { f(sp) };
            }

            result.map(|()| (neighbors, distances))
        }

        /// Vector dimension this index was trained on.
        #[inline]
        pub fn dim(&self) -> usize {
            self.dim
        }
    }

    impl<T: CuvsScalar> Drop for IvfFlat<T> {
        fn drop(&mut self) {
            if self.raw.is_null() {
                return;
            }
            if let Ok(c) = cuvs() {
                if let Ok(f) = c.cuvs_ivf_flat_index_destroy() {
                    let _ = unsafe { f(self.raw) };
                }
            }
        }
    }

    // ---- Brute-force -----------------------------------------------------

    /// An exact brute-force k-NN index.
    ///
    /// cuVS keeps a **non-owning view** of the dataset (plus precomputed
    /// norms), so the index borrows `dataset` for its whole lifetime — the
    /// borrow checker guarantees the buffer outlives the index.
    pub struct BruteForce<'a, T: CuvsScalar> {
        raw: cuvsBruteForceIndex_t,
        dim: usize,
        _dataset: &'a DeviceBuffer<T>,
        _marker: PhantomData<T>,
    }

    // SAFETY: opaque handle; same contract as `Resources`.
    unsafe impl<T: CuvsScalar> Send for BruteForce<'_, T> {}

    impl<T: CuvsScalar> core::fmt::Debug for BruteForce<'_, T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("BruteForce")
                .field("dim", &self.dim)
                .field("dtype", &core::any::type_name::<T>())
                .finish_non_exhaustive()
        }
    }

    impl<'a, T: CuvsScalar> BruteForce<'a, T> {
        /// Build a brute-force index over `dataset` — `n_rows` row-major
        /// vectors of dimension `dim`. The returned index borrows `dataset`,
        /// which must stay alive (and unchanged) for every later `search`.
        pub fn build(
            res: &Resources,
            dataset: &'a DeviceBuffer<T>,
            n_rows: usize,
            dim: usize,
            metric: Metric,
        ) -> Result<Self> {
            assert_eq!(
                dataset.len(),
                n_rows * dim,
                "dataset length {} != n_rows {n_rows} * dim {dim}",
                dataset.len(),
            );
            let c = cuvs()?;

            let mut index: cuvsBruteForceIndex_t = core::ptr::null_mut();
            check(unsafe { c.cuvs_brute_force_index_create()?(&mut index) })?;

            let build_result = (|| -> Result<()> {
                let mut shape = [n_rows as i64, dim as i64];
                let mut ds = device_tensor(
                    dataset.as_raw().0 as *mut c_void,
                    &mut shape,
                    T::dl_dtype(),
                    device_id_of(dataset),
                );
                check(unsafe {
                    c.cuvs_brute_force_build()?(res.raw, &mut ds, metric.raw(), 0.0, index)
                })?;
                res.sync()
            })();

            match build_result {
                Ok(()) => Ok(Self {
                    raw: index,
                    dim,
                    _dataset: dataset,
                    _marker: PhantomData,
                }),
                Err(e) => {
                    if !index.is_null() {
                        if let Ok(f) = c.cuvs_brute_force_index_destroy() {
                            let _ = unsafe { f(index) };
                        }
                    }
                    Err(e)
                }
            }
        }

        /// Exact k-NN search. Returns `(neighbors: i64, distances: f32)`, each
        /// a row-major `n_queries × k` device buffer. The bound stream is
        /// synchronized before returning.
        pub fn search(
            &self,
            res: &Resources,
            queries: &DeviceBuffer<T>,
            n_queries: usize,
            k: usize,
        ) -> Result<(DeviceBuffer<i64>, DeviceBuffer<f32>)> {
            assert_eq!(
                queries.len(),
                n_queries * self.dim,
                "queries length {} != n_queries {n_queries} * dim {}",
                queries.len(),
                self.dim,
            );
            let c = cuvs()?;
            let (neighbors, distances) = alloc_outputs(queries.context(), n_queries, k)?;

            let dev = device_id_of(queries);
            let mut q_shape = [n_queries as i64, self.dim as i64];
            let mut n_shape = [n_queries as i64, k as i64];
            let mut d_shape = [n_queries as i64, k as i64];
            let mut q = device_tensor(
                queries.as_raw().0 as *mut c_void,
                &mut q_shape,
                T::dl_dtype(),
                dev,
            );
            let mut n = device_tensor(
                neighbors.as_raw().0 as *mut c_void,
                &mut n_shape,
                neighbors_dtype(),
                dev,
            );
            let mut d = device_tensor(
                distances.as_raw().0 as *mut c_void,
                &mut d_shape,
                distances_dtype(),
                dev,
            );
            check(unsafe {
                c.cuvs_brute_force_search()?(
                    res.raw,
                    self.raw,
                    &mut q,
                    &mut n,
                    &mut d,
                    cuvsFilter::none(),
                )
            })?;
            res.sync()?;
            Ok((neighbors, distances))
        }

        /// Vector dimension this index was trained on.
        #[inline]
        pub fn dim(&self) -> usize {
            self.dim
        }
    }

    impl<T: CuvsScalar> Drop for BruteForce<'_, T> {
        fn drop(&mut self) {
            if self.raw.is_null() {
                return;
            }
            if let Ok(c) = cuvs() {
                if let Ok(f) = c.cuvs_brute_force_index_destroy() {
                    let _ = unsafe { f(self.raw) };
                }
            }
        }
    }
}

#[cfg(feature = "cuvs")]
pub use imp::*;
