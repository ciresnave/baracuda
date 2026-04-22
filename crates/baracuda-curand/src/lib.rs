//! Safe Rust wrappers for NVIDIA cuRAND.
//!
//! ```no_run
//! use baracuda_driver::{Context, Device, DeviceBuffer};
//! use baracuda_curand::{Generator, RngKind};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 1024)?;
//!
//! let mut gen = Generator::new(RngKind::Default)?;
//! gen.seed(0xDEAD_BEEF)?;
//! gen.uniform(&mut buf)?;
//! # Ok(()) }
//! ```

#![warn(missing_debug_implementations)]

use std::sync::Arc;

use baracuda_curand_sys::{
    curand, curandGenerator_t, curandOrdering_t, curandRngType_t, curandStatus_t,
};
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_types::DeviceRepr;

/// Error type for cuRAND operations.
pub type Error = baracuda_core::Error<curandStatus_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: curandStatus_t) -> Result<()> {
    Error::check(status)
}

/// Pseudo-random number generator algorithm.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum RngKind {
    /// Default (XORWOW).
    #[default]
    Default,
    XORWOW,
    MRG32K3A,
    MTGP32,
    MT19937,
    /// Philox 4×32 with 10 rounds — usually the best general-purpose choice.
    Philox4_32_10,
    /// Low-discrepancy Sobol 32-bit sequence (quasi-random).
    Sobol32,
    /// Scrambled Sobol 32-bit — shifts Sobol's output to remove structural correlations.
    ScrambledSobol32,
    /// Sobol 64-bit.
    Sobol64,
    ScrambledSobol64,
}

impl RngKind {
    #[inline]
    fn raw(self) -> curandRngType_t {
        match self {
            RngKind::Default => curandRngType_t::PSEUDO_DEFAULT,
            RngKind::XORWOW => curandRngType_t::PSEUDO_XORWOW,
            RngKind::MRG32K3A => curandRngType_t::PSEUDO_MRG32K3A,
            RngKind::MTGP32 => curandRngType_t::PSEUDO_MTGP32,
            RngKind::MT19937 => curandRngType_t::PSEUDO_MT19937,
            RngKind::Philox4_32_10 => curandRngType_t::PSEUDO_PHILOX4_32_10,
            RngKind::Sobol32 => curandRngType_t::QUASI_SOBOL32,
            RngKind::ScrambledSobol32 => curandRngType_t::QUASI_SCRAMBLED_SOBOL32,
            RngKind::Sobol64 => curandRngType_t::QUASI_SOBOL64,
            RngKind::ScrambledSobol64 => curandRngType_t::QUASI_SCRAMBLED_SOBOL64,
        }
    }
}

/// Ordering of the generated sequence.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum Ordering {
    PseudoBest,
    #[default]
    PseudoDefault,
    PseudoSeeded,
    PseudoLegacy,
    PseudoDynamic,
    QuasiDefault,
}

impl Ordering {
    #[inline]
    fn raw(self) -> curandOrdering_t {
        match self {
            Ordering::PseudoBest => curandOrdering_t::PSEUDO_BEST,
            Ordering::PseudoDefault => curandOrdering_t::PSEUDO_DEFAULT,
            Ordering::PseudoSeeded => curandOrdering_t::PSEUDO_SEEDED,
            Ordering::PseudoLegacy => curandOrdering_t::PSEUDO_LEGACY,
            Ordering::PseudoDynamic => curandOrdering_t::PSEUDO_DYNAMIC,
            Ordering::QuasiDefault => curandOrdering_t::QUASI_DEFAULT,
        }
    }
}

/// A cuRAND pseudo-random number generator.
#[derive(Clone)]
pub struct Generator {
    inner: Arc<GeneratorInner>,
}

struct GeneratorInner {
    handle: curandGenerator_t,
}

unsafe impl Send for GeneratorInner {}

impl core::fmt::Debug for GeneratorInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("curand::Generator")
            .field("handle", &self.handle)
            .finish()
    }
}

impl core::fmt::Debug for Generator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Generator {
    /// Create a new generator of the given kind.
    pub fn new(kind: RngKind) -> Result<Self> {
        let c = curand()?;
        let cu = c.curand_create_generator()?;
        let mut handle: curandGenerator_t = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, kind.raw()) })?;
        // !Sync on purpose: cuRAND generators are not thread-safe.
        #[allow(clippy::arc_with_non_send_sync)]
        let inner = Arc::new(GeneratorInner { handle });
        Ok(Self { inner })
    }

    /// Reseed the generator. Call before any `uniform`/`normal` to make
    /// subsequent output deterministic.
    pub fn seed(&self, seed: u64) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_set_pseudo_random_generator_seed()?;
        check(unsafe { cu(self.inner.handle, seed) })
    }

    /// Bind the generator to a CUDA stream. Subsequent generate calls
    /// issue on that stream asynchronously.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_set_stream()?;
        check(unsafe { cu(self.inner.handle, stream.as_raw() as _) })
    }

    /// Fill `buf` with uniform `(0, 1]` samples of type `f32`.
    pub fn uniform(&self, buf: &mut DeviceBuffer<f32>) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_uniform()?;
        check(unsafe { cu(self.inner.handle, buf.as_raw().0 as *mut f32, buf.len()) })
    }

    /// Fill `buf` with uniform `(0, 1]` samples of type `f64`.
    pub fn uniform_f64(&self, buf: &mut DeviceBuffer<f64>) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_uniform_double()?;
        check(unsafe { cu(self.inner.handle, buf.as_raw().0 as *mut f64, buf.len()) })
    }

    /// Fill `buf` with `N(mean, stddev²)` samples of type `f32`.
    ///
    /// `buf.len()` must be **even** — cuRAND generates normals in pairs
    /// (Box–Muller).
    pub fn normal(&self, buf: &mut DeviceBuffer<f32>, mean: f32, stddev: f32) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_normal()?;
        check(unsafe {
            cu(
                self.inner.handle,
                buf.as_raw().0 as *mut f32,
                buf.len(),
                mean,
                stddev,
            )
        })
    }

    /// Fill `buf` with `N(mean, stddev²)` samples of type `f64`.
    pub fn normal_f64(&self, buf: &mut DeviceBuffer<f64>, mean: f64, stddev: f64) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_normal_double()?;
        check(unsafe {
            cu(
                self.inner.handle,
                buf.as_raw().0 as *mut f64,
                buf.len(),
                mean,
                stddev,
            )
        })
    }

    /// Create a host-side generator (samples live in host memory).
    pub fn new_host(kind: RngKind) -> Result<Self> {
        let c = curand()?;
        let cu = c.curand_create_generator_host()?;
        let mut handle: curandGenerator_t = core::ptr::null_mut();
        check(unsafe { cu(&mut handle, kind.raw()) })?;
        #[allow(clippy::arc_with_non_send_sync)]
        let inner = Arc::new(GeneratorInner { handle });
        Ok(Self { inner })
    }

    /// Absolute offset into the sequence — skips samples without generating.
    pub fn set_offset(&self, offset: u64) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_set_generator_offset()?;
        check(unsafe { cu(self.inner.handle, offset) })
    }

    /// Ordering of the generated sequence across threads.
    pub fn set_ordering(&self, order: Ordering) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_set_generator_ordering()?;
        check(unsafe { cu(self.inner.handle, order.raw()) })
    }

    /// Number of dimensions for quasi-random (Sobol) generators.
    pub fn set_quasi_random_dimensions(&self, num_dimensions: u32) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_set_quasi_random_generator_dimensions()?;
        check(unsafe { cu(self.inner.handle, num_dimensions) })
    }

    /// Explicit seed-derivation step (useful before offset resets).
    pub fn generate_seeds(&self) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_seeds()?;
        check(unsafe { cu(self.inner.handle) })
    }

    /// Fill `buf` with raw u32 samples.
    pub fn uniform_u32(&self, buf: &mut DeviceBuffer<u32>) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate()?;
        check(unsafe { cu(self.inner.handle, buf.as_raw().0 as *mut u32, buf.len()) })
    }

    /// Fill `buf` with raw u64 samples.
    pub fn uniform_u64(&self, buf: &mut DeviceBuffer<u64>) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_long_long()?;
        check(unsafe { cu(self.inner.handle, buf.as_raw().0 as *mut u64, buf.len()) })
    }

    /// Log-normal samples (`exp(N(mean, stddev²))`) in `f32`.
    pub fn log_normal(
        &self,
        buf: &mut DeviceBuffer<f32>,
        mean: f32,
        stddev: f32,
    ) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_log_normal()?;
        check(unsafe {
            cu(
                self.inner.handle,
                buf.as_raw().0 as *mut f32,
                buf.len(),
                mean,
                stddev,
            )
        })
    }

    /// Log-normal samples in `f64`.
    pub fn log_normal_f64(
        &self,
        buf: &mut DeviceBuffer<f64>,
        mean: f64,
        stddev: f64,
    ) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_log_normal_double()?;
        check(unsafe {
            cu(
                self.inner.handle,
                buf.as_raw().0 as *mut f64,
                buf.len(),
                mean,
                stddev,
            )
        })
    }

    /// Poisson-distributed `u32` samples with rate `lambda`.
    pub fn poisson(&self, buf: &mut DeviceBuffer<u32>, lambda: f64) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_poisson()?;
        check(unsafe { cu(self.inner.handle, buf.as_raw().0 as *mut u32, buf.len(), lambda) })
    }

    /// Binomial-distributed `u32` samples (`trials` trials, success
    /// probability `prob`).
    pub fn binomial(
        &self,
        buf: &mut DeviceBuffer<u32>,
        trials: u32,
        prob: f64,
    ) -> Result<()> {
        let c = curand()?;
        let cu = c.curand_generate_binomial()?;
        check(unsafe {
            cu(
                self.inner.handle,
                buf.as_raw().0 as *mut u32,
                buf.len(),
                trials,
                prob,
            )
        })
    }

    /// Raw `curandGenerator_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> curandGenerator_t {
        self.inner.handle
    }
}

/// cuRAND library version (e.g. `10400` = 10.4.0).
pub fn version() -> Result<i32> {
    let c = curand()?;
    let cu = c.curand_get_version()?;
    let mut v: core::ffi::c_int = 0;
    check(unsafe { cu(&mut v) })?;
    Ok(v)
}

impl Drop for GeneratorInner {
    fn drop(&mut self) {
        if let Ok(c) = curand() {
            if let Ok(cu) = c.curand_destroy_generator() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Unused trait binding; keeps `DeviceRepr` visible at the top level so
/// safe users don't need an extra import.
#[allow(dead_code)]
fn _bind_device_repr<T: DeviceRepr>() {}
