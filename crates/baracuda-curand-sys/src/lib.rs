//! Raw FFI + dynamic loader for NVIDIA cuRAND.
//!
//! `baracuda-curand` wraps this with a safe, typed API. Use this crate
//! directly only if you need a function that the safe layer hasn't
//! wrapped yet (in which case please file a bug).

#![allow(non_camel_case_types)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

/// Opaque cuRAND generator handle.
pub type curandGenerator_t = *mut c_void;

/// Random number generator type.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum curandRngType_t {
    /// Default pseudo-random: XORWOW.
    PSEUDO_DEFAULT = 100,
    /// XORWOW pseudo-random (Marsaglia).
    PSEUDO_XORWOW = 101,
    /// MRG32k3a pseudo-random (L'Ecuyer).
    PSEUDO_MRG32K3A = 121,
    /// Mersenne-Twister GP-32 pseudo-random.
    PSEUDO_MTGP32 = 141,
    /// Mersenne-Twister 19937 pseudo-random.
    PSEUDO_MT19937 = 142,
    /// Philox 4x32-10 counter-based pseudo-random.
    PSEUDO_PHILOX4_32_10 = 161,
    // Quasi-random (low-discrepancy) generators.
    /// Default quasi-random (low-discrepancy) generator.
    QUASI_DEFAULT = 200,
    /// 32-bit Sobol quasi-random.
    QUASI_SOBOL32 = 201,
    /// 32-bit scrambled Sobol quasi-random.
    QUASI_SCRAMBLED_SOBOL32 = 202,
    /// 64-bit Sobol quasi-random.
    QUASI_SOBOL64 = 203,
    /// 64-bit scrambled Sobol quasi-random.
    QUASI_SCRAMBLED_SOBOL64 = 204,
}

/// Ordering for the generated sequence.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum curandOrdering_t {
    /// Library-chosen best ordering for pseudo-random.
    PSEUDO_BEST = 100,
    /// Default pseudo-random ordering.
    PSEUDO_DEFAULT = 101,
    /// Seeded pseudo-random ordering.
    PSEUDO_SEEDED = 102,
    /// Legacy pseudo-random ordering (pre-9.0 compatibility).
    PSEUDO_LEGACY = 103,
    /// Dynamic pseudo-random ordering (auto-tuned).
    PSEUDO_DYNAMIC = 104,
    /// Default quasi-random ordering.
    QUASI_DEFAULT = 201,
}

/// Dimension mode for quasi-random generators.
pub type curandDirectionVectorSet_t = i32;

/// Return code from a cuRAND call.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct curandStatus_t(pub i32);

impl curandStatus_t {
    /// `CURAND_STATUS_SUCCESS` — operation succeeded.
    pub const SUCCESS: Self = Self(0);
    /// `CURAND_STATUS_VERSION_MISMATCH` — header/library version mismatch.
    pub const VERSION_MISMATCH: Self = Self(100);
    /// `CURAND_STATUS_NOT_INITIALIZED` — the generator is not initialized.
    pub const NOT_INITIALIZED: Self = Self(101);
    /// `CURAND_STATUS_ALLOCATION_FAILED` — an allocation failed.
    pub const ALLOCATION_FAILED: Self = Self(102);
    /// `CURAND_STATUS_TYPE_ERROR` — an argument has an invalid type.
    pub const TYPE_ERROR: Self = Self(103);
    /// `CURAND_STATUS_OUT_OF_RANGE` — an argument is out of range.
    pub const OUT_OF_RANGE: Self = Self(104);
    /// `CURAND_STATUS_LENGTH_NOT_MULTIPLE` — length must be a multiple of the generator's period.
    pub const LENGTH_NOT_MULTIPLE: Self = Self(105);
    /// `CURAND_STATUS_DOUBLE_PRECISION_REQUIRED` — double-precision required but not supported on device.
    pub const DOUBLE_PRECISION_REQUIRED: Self = Self(106);
    /// `CURAND_STATUS_LAUNCH_FAILURE` — a GPU kernel launch failed.
    pub const LAUNCH_FAILURE: Self = Self(201);
    /// `CURAND_STATUS_PREEXISTING_FAILURE` — a prior call failed and the handle is poisoned.
    pub const PREEXISTING_FAILURE: Self = Self(202);
    /// `CURAND_STATUS_INITIALIZATION_FAILED` — initialization failed.
    pub const INITIALIZATION_FAILED: Self = Self(203);
    /// `CURAND_STATUS_ARCH_MISMATCH` — the device architecture is unsupported.
    pub const ARCH_MISMATCH: Self = Self(204);
    /// `CURAND_STATUS_INTERNAL_ERROR` — an internal cuRAND error occurred.
    pub const INTERNAL_ERROR: Self = Self(999);

    /// Return `true` if the status code denotes success.
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for curandStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CURAND_STATUS_SUCCESS",
            100 => "CURAND_STATUS_VERSION_MISMATCH",
            101 => "CURAND_STATUS_NOT_INITIALIZED",
            102 => "CURAND_STATUS_ALLOCATION_FAILED",
            103 => "CURAND_STATUS_TYPE_ERROR",
            104 => "CURAND_STATUS_OUT_OF_RANGE",
            105 => "CURAND_STATUS_LENGTH_NOT_MULTIPLE",
            106 => "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
            201 => "CURAND_STATUS_LAUNCH_FAILURE",
            202 => "CURAND_STATUS_PREEXISTING_FAILURE",
            203 => "CURAND_STATUS_INITIALIZATION_FAILED",
            204 => "CURAND_STATUS_ARCH_MISMATCH",
            999 => "CURAND_STATUS_INTERNAL_ERROR",
            _ => "CURAND_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            100 => "version mismatch",
            101 => "generator not initialized",
            102 => "allocation failed",
            103 => "type error",
            104 => "out-of-range argument",
            105 => "output length must be a multiple of the generator's period",
            106 => "double-precision not supported",
            201 => "GPU kernel launch failed",
            202 => "preexisting failure",
            203 => "initialization failed",
            204 => "architecture mismatch",
            999 => "internal cuRAND error",
            _ => "unrecognized cuRAND status code",
        }
    }
    fn is_success(self) -> bool {
        curandStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "curand"
    }
}

// ---- function-pointer types ----

/// Function-pointer type for `curandCreateGenerator` (create a device RNG). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandCreateGenerator =
    unsafe extern "C" fn(generator: *mut curandGenerator_t, ty: curandRngType_t) -> curandStatus_t;
/// Function-pointer type for `curandDestroyGenerator` (destroy an RNG). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandDestroyGenerator =
    unsafe extern "C" fn(generator: curandGenerator_t) -> curandStatus_t;
/// Function-pointer type for `curandSetStream` (bind a CUDA stream to an RNG). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandSetStream =
    unsafe extern "C" fn(generator: curandGenerator_t, stream: cudaStream_t) -> curandStatus_t;
/// Function-pointer type for `curandSetPseudoRandomGeneratorSeed` (set RNG seed). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandSetPseudoRandomGeneratorSeed =
    unsafe extern "C" fn(generator: curandGenerator_t, seed: u64) -> curandStatus_t;
/// Function-pointer type for `curandGenerateUniform` (generate uniform-distributed f32). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateUniform =
    unsafe extern "C" fn(generator: curandGenerator_t, out: *mut f32, n: usize) -> curandStatus_t;
/// Function-pointer type for `curandGenerateUniformDouble` (generate uniform-distributed f64). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateUniformDouble =
    unsafe extern "C" fn(generator: curandGenerator_t, out: *mut f64, n: usize) -> curandStatus_t;
/// Function-pointer type for `curandGenerateNormal` (generate normal-distributed f32). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateNormal = unsafe extern "C" fn(
    generator: curandGenerator_t,
    out: *mut f32,
    n: usize,
    mean: f32,
    stddev: f32,
) -> curandStatus_t;
/// Function-pointer type for `curandGenerateNormalDouble` (generate normal-distributed f64). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateNormalDouble = unsafe extern "C" fn(
    generator: curandGenerator_t,
    out: *mut f64,
    n: usize,
    mean: f64,
    stddev: f64,
) -> curandStatus_t;
/// Function-pointer type for `curandGetVersion` (query cuRAND library version). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGetVersion = unsafe extern "C" fn(version: *mut c_int) -> curandStatus_t;

// ---- Additional generator configuration ----

/// Function-pointer type for `curandCreateGeneratorHost` (create a host RNG). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandCreateGeneratorHost = unsafe extern "C" fn(
    generator: *mut curandGenerator_t,
    ty: curandRngType_t,
) -> curandStatus_t;

/// Function-pointer type for `curandSetGeneratorOffset` (set generator offset). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandSetGeneratorOffset =
    unsafe extern "C" fn(generator: curandGenerator_t, offset: u64) -> curandStatus_t;

/// Function-pointer type for `curandSetGeneratorOrdering` (set generator ordering). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandSetGeneratorOrdering = unsafe extern "C" fn(
    generator: curandGenerator_t,
    order: curandOrdering_t,
) -> curandStatus_t;

/// Function-pointer type for `curandSetQuasiRandomGeneratorDimensions` (set dimensions for quasi-random generator). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandSetQuasiRandomGeneratorDimensions = unsafe extern "C" fn(
    generator: curandGenerator_t,
    num_dimensions: u32,
) -> curandStatus_t;

/// Function-pointer type for `curandGetDirectionVectors32` (fetch 32-bit Sobol direction vectors). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGetDirectionVectors32 = unsafe extern "C" fn(
    vectors_out: *mut *mut u32,
    set: curandDirectionVectorSet_t,
) -> curandStatus_t;

/// Function-pointer type for `curandGetDirectionVectors64` (fetch 64-bit Sobol direction vectors). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGetDirectionVectors64 = unsafe extern "C" fn(
    vectors_out: *mut *mut u64,
    set: curandDirectionVectorSet_t,
) -> curandStatus_t;

/// Function-pointer type for `curandGetScrambleConstants32` (fetch 32-bit scramble constants). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGetScrambleConstants32 =
    unsafe extern "C" fn(constants_out: *mut *const u32) -> curandStatus_t;

/// Function-pointer type for `curandGetScrambleConstants64` (fetch 64-bit scramble constants). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGetScrambleConstants64 =
    unsafe extern "C" fn(constants_out: *mut *const u64) -> curandStatus_t;

/// Function-pointer type for `curandGetProperty` (query a cuRAND property value). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGetProperty =
    unsafe extern "C" fn(prop: c_int, value_out: *mut c_int) -> curandStatus_t;

// ---- Additional integer / uint / float distributions ----

/// Function-pointer type for `curandGenerate` (generate uniform-distributed u32). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerate =
    unsafe extern "C" fn(generator: curandGenerator_t, out: *mut u32, n: usize) -> curandStatus_t;

/// Function-pointer type for `curandGenerateLongLong` (generate uniform-distributed u64). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateLongLong =
    unsafe extern "C" fn(generator: curandGenerator_t, out: *mut u64, n: usize) -> curandStatus_t;

/// Function-pointer type for `curandGenerateLogNormal` (generate log-normal-distributed f32). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateLogNormal = unsafe extern "C" fn(
    generator: curandGenerator_t,
    out: *mut f32,
    n: usize,
    mean: f32,
    stddev: f32,
) -> curandStatus_t;

/// Function-pointer type for `curandGenerateLogNormalDouble` (generate log-normal-distributed f64). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateLogNormalDouble = unsafe extern "C" fn(
    generator: curandGenerator_t,
    out: *mut f64,
    n: usize,
    mean: f64,
    stddev: f64,
) -> curandStatus_t;

/// Function-pointer type for `curandGeneratePoisson` (generate Poisson-distributed u32). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGeneratePoisson = unsafe extern "C" fn(
    generator: curandGenerator_t,
    out: *mut u32,
    n: usize,
    lambda: f64,
) -> curandStatus_t;

/// Function-pointer type for `curandGenerateBinomial` (generate binomial-distributed u32). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateBinomial = unsafe extern "C" fn(
    generator: curandGenerator_t,
    out: *mut u32,
    n: usize,
    trials: u32,
    prob: f64,
) -> curandStatus_t;

// ---- Seed generation for pseudo-random generators ----

/// Function-pointer type for `curandGenerateSeeds` (force generation of seeds (pseudo-random)). See <https://docs.nvidia.com/cuda/curand/index.html>.
pub type PFN_curandGenerateSeeds = unsafe extern "C" fn(generator: curandGenerator_t) -> curandStatus_t;

// ---- loader ----

fn curand_candidates() -> Vec<String> {
    // cuRAND has historically stayed on .10 for Linux; more recent CUDA
    // toolkits also ship .12 / .13. Probe all.
    platform::versioned_library_candidates("curand", &["10", "13", "12", "11"])
}

macro_rules! curand_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        /// Lazily-resolved cuRAND function-pointer table.
        pub struct Curand {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }

        impl core::fmt::Debug for Curand {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Curand").field("lib", &self.lib).finish_non_exhaustive()
            }
        }

        impl Curand {
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
            $(
                #[doc = concat!("Resolve `", $sym, "`.")]
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
        }
    };
}

curand_fns! {
    curand_create_generator as "curandCreateGenerator": PFN_curandCreateGenerator;
    curand_create_generator_host as "curandCreateGeneratorHost": PFN_curandCreateGeneratorHost;
    curand_destroy_generator as "curandDestroyGenerator": PFN_curandDestroyGenerator;
    curand_set_stream as "curandSetStream": PFN_curandSetStream;
    curand_set_pseudo_random_generator_seed as "curandSetPseudoRandomGeneratorSeed":
        PFN_curandSetPseudoRandomGeneratorSeed;
    curand_set_generator_offset as "curandSetGeneratorOffset": PFN_curandSetGeneratorOffset;
    curand_set_generator_ordering as "curandSetGeneratorOrdering":
        PFN_curandSetGeneratorOrdering;
    curand_set_quasi_random_generator_dimensions as "curandSetQuasiRandomGeneratorDimensions":
        PFN_curandSetQuasiRandomGeneratorDimensions;
    curand_get_direction_vectors32 as "curandGetDirectionVectors32":
        PFN_curandGetDirectionVectors32;
    curand_get_direction_vectors64 as "curandGetDirectionVectors64":
        PFN_curandGetDirectionVectors64;
    curand_get_scramble_constants32 as "curandGetScrambleConstants32":
        PFN_curandGetScrambleConstants32;
    curand_get_scramble_constants64 as "curandGetScrambleConstants64":
        PFN_curandGetScrambleConstants64;
    curand_get_property as "curandGetProperty": PFN_curandGetProperty;

    // Generators
    curand_generate as "curandGenerate": PFN_curandGenerate;
    curand_generate_long_long as "curandGenerateLongLong": PFN_curandGenerateLongLong;
    curand_generate_uniform as "curandGenerateUniform": PFN_curandGenerateUniform;
    curand_generate_uniform_double as "curandGenerateUniformDouble":
        PFN_curandGenerateUniformDouble;
    curand_generate_normal as "curandGenerateNormal": PFN_curandGenerateNormal;
    curand_generate_normal_double as "curandGenerateNormalDouble":
        PFN_curandGenerateNormalDouble;
    curand_generate_log_normal as "curandGenerateLogNormal": PFN_curandGenerateLogNormal;
    curand_generate_log_normal_double as "curandGenerateLogNormalDouble":
        PFN_curandGenerateLogNormalDouble;
    curand_generate_poisson as "curandGeneratePoisson": PFN_curandGeneratePoisson;
    curand_generate_binomial as "curandGenerateBinomial": PFN_curandGenerateBinomial;
    curand_generate_seeds as "curandGenerateSeeds": PFN_curandGenerateSeeds;

    curand_get_version as "curandGetVersion": PFN_curandGetVersion;
}

/// Return the lazily-loaded cuRAND library accessor.
pub fn curand() -> Result<&'static Curand, LoaderError> {
    static CURAND: OnceLock<Curand> = OnceLock::new();
    if let Some(c) = CURAND.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = curand_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("curand", candidates_leaked)?;
    let _ = CURAND.set(Curand::empty(lib));
    Ok(CURAND.get().expect("OnceLock set or lost race"))
}
