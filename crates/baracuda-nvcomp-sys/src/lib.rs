//! Raw FFI + dynamic loader skeleton for NVIDIA nvCOMP (GPU compression).
//!
//! `baracuda-nvcomp` wraps this with a safe, typed API. Use this
//! crate directly only if you need a function that the safe layer
//! hasn't wrapped yet (in which case please file a bug).
//!
//! v0.1 ships the loader + status enum so the crate compiles everywhere;
//! the wide surface of per-codec entry points (LZ4, Snappy, GDeflate, Zstd,
//! Bitcomp, ANS, batched helpers) lands when we have a testable install.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_types::CudaStatus;

/// nvCOMP status code.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct nvcompStatus_t(pub i32);

impl nvcompStatus_t {
    /// nvCOMP status code `SUCCESS`.
    pub const SUCCESS: Self = Self(0);
    /// nvCOMP status code `ERROR_INVALID_VALUE`.
    pub const ERROR_INVALID_VALUE: Self = Self(10);
    /// nvCOMP status code `ERROR_NOT_SUPPORTED`.
    pub const ERROR_NOT_SUPPORTED: Self = Self(11);
    /// nvCOMP status code `ERROR_CANNOT_DECOMPRESS`.
    pub const ERROR_CANNOT_DECOMPRESS: Self = Self(12);
    /// nvCOMP status code `ERROR_BAD_CHECKSUM`.
    pub const ERROR_BAD_CHECKSUM: Self = Self(13);
    /// nvCOMP status code `ERROR_CUDA`.
    pub const ERROR_CUDA: Self = Self(1000);
    /// nvCOMP status code `ERROR_INTERNAL`.
    pub const ERROR_INTERNAL: Self = Self(10000);

    /// `is_success` method on `nvcompStatus_t`.
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for nvcompStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "nvcompSuccess",
            10 => "nvcompErrorInvalidValue",
            11 => "nvcompErrorNotSupported",
            12 => "nvcompErrorCannotDecompress",
            13 => "nvcompErrorBadChecksum",
            1000 => "nvcompErrorCudaError",
            10000 => "nvcompErrorInternal",
            _ => "nvcompErrorUnrecognized",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            10 => "invalid argument",
            11 => "operation not supported",
            12 => "decompression failed",
            13 => "bad checksum",
            1000 => "CUDA error",
            _ => "unrecognized nvCOMP status code",
        }
    }
    fn is_success(self) -> bool {
        nvcompStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "nvcomp"
    }
}

use core::ffi::c_void;

/// `nvcompType_t` — element type for batched APIs.
#[allow(non_snake_case)]
pub mod nvcompType {
    /// `CHAR` value of `nvcompType`.
    pub const CHAR: i32 = 0;
    /// `UCHAR` value of `nvcompType`.
    pub const UCHAR: i32 = 1;
    /// `SHORT` value of `nvcompType`.
    pub const SHORT: i32 = 2;
    /// `USHORT` value of `nvcompType`.
    pub const USHORT: i32 = 3;
    /// `INT` value of `nvcompType`.
    pub const INT: i32 = 4;
    /// `UINT` value of `nvcompType`.
    pub const UINT: i32 = 5;
    /// `LONGLONG` value of `nvcompType`.
    pub const LONGLONG: i32 = 6;
    /// `ULONGLONG` value of `nvcompType`.
    pub const ULONGLONG: i32 = 7;
    /// `BITS` value of `nvcompType`.
    pub const BITS: i32 = 0xff;
}

// --- LZ4 batched options (nvCOMP 5.x layout — 32 bytes) ---
// The exact field layout drifts across versions; we keep the known
// ones typed and pad the tail so the struct is ABI-forward-compatible
// with zero-default values.

/// `nvcompDecompressBackend_t` — decompression-engine selector (nvCOMP 5+).
#[allow(non_snake_case)]
pub mod nvcompDecompressBackend {
    /// `DEFAULT` value of `nvcompDecompressBackend`.
    pub const DEFAULT: i32 = 0;
    /// `CUDA` value of `nvcompDecompressBackend`.
    pub const CUDA: i32 = 1;
    /// `HARDWARE` value of `nvcompDecompressBackend`.
    pub const HARDWARE: i32 = 2;
}

/// LZ4 decompression options (v5+, 64 bytes: 4 backend + 4 sort_before_hw +
/// 4 data_type + 4 bitshuffle_mode + 48 reserved).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedLZ4DecompressOpts_t {
    pub backend: i32,
    pub sort_before_hw_decompress: i32,
    pub data_type: i32,
    pub bitshuffle_mode: i32,
    _reserved: [u8; 48],
}

impl Default for nvcompBatchedLZ4DecompressOpts_t {
    fn default() -> Self {
        Self {
            backend: 0,
            sort_before_hw_decompress: 0,
            data_type: 0,
            bitshuffle_mode: 0,
            _reserved: [0; 48],
        }
    }
}

/// Snappy decompression options (shape parallel to LZ4's).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedSnappyDecompressOpts_t {
    pub backend: i32,
    pub sort_before_hw_decompress: i32,
    _pad: i64,
    _reserved: [u8; 48],
}

impl Default for nvcompBatchedSnappyDecompressOpts_t {
    fn default() -> Self {
        Self {
            backend: 0,
            sort_before_hw_decompress: 0,
            _pad: 0,
            _reserved: [0; 48],
        }
    }
}

/// Zstd decompression options.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedZstdDecompressOpts_t {
    pub backend: i32,
    pub sort_before_hw_decompress: i32,
    _pad: i64,
    _reserved: [u8; 48],
}

impl Default for nvcompBatchedZstdDecompressOpts_t {
    fn default() -> Self {
        Self {
            backend: 0,
            sort_before_hw_decompress: 0,
            _pad: 0,
            _reserved: [0; 48],
        }
    }
}

/// Gdeflate decompression options.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedGdeflateDecompressOpts_t {
    pub backend: i32,
    pub sort_before_hw_decompress: i32,
    _pad: i64,
    _reserved: [u8; 48],
}

impl Default for nvcompBatchedGdeflateDecompressOpts_t {
    fn default() -> Self {
        Self {
            backend: 0,
            sort_before_hw_decompress: 0,
            _pad: 0,
            _reserved: [0; 48],
        }
    }
}

// nvCOMP 5.x CompressOpts_t structs are all 64 bytes:
//   4-byte data_type + 4-byte bitshuffle_mode + 56-byte reserved pad.

/// LZ4 batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedLZ4Opts_t {
    pub data_type: i32,
    pub bitshuffle_mode: i32,
    _reserved: [u8; 56],
}

impl Default for nvcompBatchedLZ4Opts_t {
    fn default() -> Self {
        Self {
            data_type: 0,
            bitshuffle_mode: 0,
            _reserved: [0; 56],
        }
    }
}

/// Snappy batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedSnappyOpts_t {
    pub reserved_field: i32,
    pub _pad0: i32,
    _reserved: [u8; 56],
}

impl Default for nvcompBatchedSnappyOpts_t {
    fn default() -> Self {
        Self {
            reserved_field: 0,
            _pad0: 0,
            _reserved: [0; 56],
        }
    }
}

/// Zstd batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedZstdOpts_t {
    pub reserved_field: i32,
    pub _pad0: i32,
    _reserved: [u8; 56],
}

impl Default for nvcompBatchedZstdOpts_t {
    fn default() -> Self {
        Self {
            reserved_field: 0,
            _pad0: 0,
            _reserved: [0; 56],
        }
    }
}

/// Gdeflate batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedGdeflateOpts_t {
    pub algo: i32,
    pub _pad0: i32,
    _reserved: [u8; 56],
}

impl Default for nvcompBatchedGdeflateOpts_t {
    fn default() -> Self {
        Self {
            algo: 0,
            _pad0: 0,
            _reserved: [0; 56],
        }
    }
}

// ---- PFN types ----

/// nvCOMP 5+ signature — adds `max_total_uncompressed_bytes` tail param.
pub type PFN_nvcompBatchedLZ4CompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedLZ4Opts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get max output chunk size for LZ4 batched compression.
pub type PFN_nvcompBatchedLZ4CompressGetMaxOutputChunkSize = unsafe extern "C" fn(
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedLZ4Opts_t,
    max_compressed_bytes_out: *mut usize,
)
    -> nvcompStatus_t;

/// launch LZ4 batched compression on a CUDA stream.
pub type PFN_nvcompBatchedLZ4CompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedLZ4Opts_t,
    device_statuses: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// nvCOMP 5+ signature: adds `decompress_opts` + `max_total_uncompressed_bytes`.
pub type PFN_nvcompBatchedLZ4DecompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    decompress_opts: nvcompBatchedLZ4DecompressOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// nvCOMP 5+ signature: adds `decompress_opts` (by value).
pub type PFN_nvcompBatchedLZ4DecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    decompress_opts: nvcompBatchedLZ4DecompressOpts_t,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// launch Snappy batched compression on a CUDA stream.
pub type PFN_nvcompBatchedSnappyCompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedSnappyOpts_t,
    device_statuses: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// launch Zstd batched compression on a CUDA stream.
pub type PFN_nvcompBatchedZstdCompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedZstdOpts_t,
    device_statuses: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// --- Required-alignment structs (nvCOMP 4+) ---

/// alignment hints returned by nvCOMP for buffer planning.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvcompAlignmentRequirements_t {
    pub input: usize,
    pub output: usize,
    pub temp: usize,
}

// --- Additional codec options ---

/// Bitcomp batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvcompBatchedBitcompOpts_t {
    pub algorithm_type: i32,
    pub data_type: i32,
}

/// ANS batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvcompBatchedANSOpts_t {
    pub reserved: i32,
}

/// Deflate batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvcompBatchedDeflateOpts_t {
    pub algo: i32,
}

/// Cascaded batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct nvcompBatchedCascadedOpts_t {
    pub chunk_size: usize,
    pub type_: i32,
    pub num_rles: i32,
    pub num_deltas: i32,
    pub use_bp: i32,
}

// ---- Generalized codec PFN signatures (all codecs share this shape) ----
// Each codec `X` exposes CompressGetTempSize/CompressGetMaxOutputChunkSize/
// CompressAsync/DecompressGetTempSize/DecompressAsync (+ alignment / size
// helpers in nvCOMP 4+). Below we declare each PFN individually for
// readability.

// LZ4 extras
/// get required-alignment hints for LZ4 batched compression.
pub type PFN_nvcompBatchedLZ4CompressGetRequiredAlignments = unsafe extern "C" fn(
    opts: nvcompBatchedLZ4Opts_t,
    alignments_out: *mut nvcompAlignmentRequirements_t,
)
    -> nvcompStatus_t;

/// get required-alignment hints for LZ4 batched decompression.
pub type PFN_nvcompBatchedLZ4DecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;

/// get decompressed-size estimate for LZ4 batched chunks.
pub type PFN_nvcompBatchedLZ4GetDecompressSizeAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes_out: *mut usize,
    batch_size: usize,
    stream: *mut c_void,
) -> nvcompStatus_t;

// Snappy full set
/// get scratch size for Snappy batched compression.
pub type PFN_nvcompBatchedSnappyCompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedSnappyOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get max output chunk size for Snappy batched compression.
pub type PFN_nvcompBatchedSnappyCompressGetMaxOutputChunkSize =
    unsafe extern "C" fn(
        max_uncompressed_chunk_size: usize,
        opts: nvcompBatchedSnappyOpts_t,
        max_compressed_bytes_out: *mut usize,
    ) -> nvcompStatus_t;

/// get scratch size for Snappy batched decompression.
pub type PFN_nvcompBatchedSnappyDecompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    decompress_opts: nvcompBatchedSnappyDecompressOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// launch Snappy batched decompression on a CUDA stream.
pub type PFN_nvcompBatchedSnappyDecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    decompress_opts: nvcompBatchedSnappyDecompressOpts_t,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// Zstd full set
/// get scratch size for Zstd batched compression.
pub type PFN_nvcompBatchedZstdCompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedZstdOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get max output chunk size for Zstd batched compression.
pub type PFN_nvcompBatchedZstdCompressGetMaxOutputChunkSize =
    unsafe extern "C" fn(
        max_uncompressed_chunk_size: usize,
        opts: nvcompBatchedZstdOpts_t,
        max_compressed_bytes_out: *mut usize,
    ) -> nvcompStatus_t;

/// get scratch size for Zstd batched decompression.
pub type PFN_nvcompBatchedZstdDecompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    decompress_opts: nvcompBatchedZstdDecompressOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// launch Zstd batched decompression on a CUDA stream.
pub type PFN_nvcompBatchedZstdDecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    decompress_opts: nvcompBatchedZstdDecompressOpts_t,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// GDeflate
/// get scratch size for Gdeflate batched compression.
pub type PFN_nvcompBatchedGdeflateCompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedGdeflateOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get max output chunk size for Gdeflate batched compression.
pub type PFN_nvcompBatchedGdeflateCompressGetMaxOutputChunkSize =
    unsafe extern "C" fn(
        max_uncompressed_chunk_size: usize,
        opts: nvcompBatchedGdeflateOpts_t,
        max_compressed_bytes_out: *mut usize,
    ) -> nvcompStatus_t;

/// launch Gdeflate batched compression on a CUDA stream.
pub type PFN_nvcompBatchedGdeflateCompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedGdeflateOpts_t,
    device_statuses: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// get scratch size for Gdeflate batched decompression.
pub type PFN_nvcompBatchedGdeflateDecompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    decompress_opts: nvcompBatchedGdeflateDecompressOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// launch Gdeflate batched decompression on a CUDA stream.
pub type PFN_nvcompBatchedGdeflateDecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    decompress_opts: nvcompBatchedGdeflateDecompressOpts_t,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// Deflate (zlib/raw-deflate stream path)
/// launch Deflate batched compression on a CUDA stream.
pub type PFN_nvcompBatchedDeflateCompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedDeflateOpts_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// launch Deflate batched decompression on a CUDA stream.
pub type PFN_nvcompBatchedDeflateDecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// Bitcomp
/// launch Bitcomp batched compression on a CUDA stream.
pub type PFN_nvcompBatchedBitcompCompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedBitcompOpts_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// launch Bitcomp batched decompression on a CUDA stream.
pub type PFN_nvcompBatchedBitcompDecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// ANS
/// launch ANS batched compression on a CUDA stream.
pub type PFN_nvcompBatchedANSCompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedANSOpts_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// launch ANS batched decompression on a CUDA stream.
pub type PFN_nvcompBatchedANSDecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// Cascaded
/// launch Cascaded batched compression on a CUDA stream.
pub type PFN_nvcompBatchedCascadedCompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedCascadedOpts_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// launch Cascaded batched decompression on a CUDA stream.
pub type PFN_nvcompBatchedCascadedDecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// Generic DecompressGetSizeAsync — returns original uncompressed sizes
// from compressed buffers. Same signature across codecs; we expose one
// alias per codec via loader entries.
/// generic decompressed-size estimator (codec-agnostic).
pub type PFN_nvcompGetDecompressSizeAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes_out: *mut usize,
    batch_size: usize,
    stream: *mut c_void,
) -> nvcompStatus_t;

// ---- Gzip codec (nvCOMP 5+) ----

/// Gzip batched compression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedGzipOpts_t {
    pub reserved_field: i32,
    pub _pad0: i32,
    _reserved: [u8; 56],
}

impl Default for nvcompBatchedGzipOpts_t {
    fn default() -> Self {
        Self {
            reserved_field: 0,
            _pad0: 0,
            _reserved: [0; 56],
        }
    }
}

/// Gzip batched decompression options struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvcompBatchedGzipDecompressOpts_t {
    pub backend: i32,
    pub sort_before_hw_decompress: i32,
    _pad: i64,
    _reserved: [u8; 48],
}

impl Default for nvcompBatchedGzipDecompressOpts_t {
    fn default() -> Self {
        Self {
            backend: 0,
            sort_before_hw_decompress: 0,
            _pad: 0,
            _reserved: [0; 48],
        }
    }
}

/// get scratch size for Gzip batched compression.
pub type PFN_nvcompBatchedGzipCompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedGzipOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get max output chunk size for Gzip batched compression.
pub type PFN_nvcompBatchedGzipCompressGetMaxOutputChunkSize =
    unsafe extern "C" fn(
        max_uncompressed_chunk_size: usize,
        opts: nvcompBatchedGzipOpts_t,
        max_compressed_bytes_out: *mut usize,
    ) -> nvcompStatus_t;

/// launch Gzip batched compression on a CUDA stream.
pub type PFN_nvcompBatchedGzipCompressAsync = unsafe extern "C" fn(
    device_uncompressed_ptrs: *const *const c_void,
    device_uncompressed_bytes: *const usize,
    max_uncompressed_chunk_size: usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_compressed_ptrs: *const *mut c_void,
    device_compressed_bytes: *mut usize,
    opts: nvcompBatchedGzipOpts_t,
    device_statuses: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// get scratch size for Gzip batched decompression.
pub type PFN_nvcompBatchedGzipDecompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    decompress_opts: nvcompBatchedGzipDecompressOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// launch Gzip batched decompression on a CUDA stream.
pub type PFN_nvcompBatchedGzipDecompressAsync = unsafe extern "C" fn(
    device_compressed_ptrs: *const *const c_void,
    device_compressed_bytes: *const usize,
    device_uncompressed_bytes: *const usize,
    device_actual_uncompressed_bytes: *mut usize,
    batch_size: usize,
    device_temp_ptr: *mut c_void,
    temp_bytes: usize,
    device_uncompressed_ptrs: *const *mut c_void,
    decompress_opts: nvcompBatchedGzipDecompressOpts_t,
    device_status_ptrs: *mut nvcompStatus_t,
    stream: *mut c_void,
) -> nvcompStatus_t;

// ---- CRC32 batched (nvCOMP 5+) ----

/// launch batched CRC32 on a CUDA stream.
pub type PFN_nvcompBatchedCRC32Async = unsafe extern "C" fn(
    device_data_ptrs: *const *const c_void,
    device_data_bytes: *const usize,
    batch_size: usize,
    config: *const c_void, // CRC32 config struct — opaque to wrapper
    device_crc_out: *mut u32,
    stream: *mut c_void,
) -> nvcompStatus_t;

/// get heuristic CRC32 launch config.
pub type PFN_nvcompBatchedCRC32GetHeuristicConf =
    unsafe extern "C" fn(max_chunk_bytes: usize, config_out: *mut c_void) -> nvcompStatus_t;

/// search for an optimal CRC32 launch config.
pub type PFN_nvcompBatchedCRC32SearchConf =
    unsafe extern "C" fn(max_chunk_bytes: usize, config_out: *mut c_void) -> nvcompStatus_t;

// ---- Library-level metadata ----

/// get nvCOMP library version/build properties.
pub type PFN_nvcompGetProperties = unsafe extern "C" fn(version_out: *mut c_void) -> nvcompStatus_t;

/// stringify an `nvcompStatus_t` code.
pub type PFN_nvcompGetStatusString =
    unsafe extern "C" fn(status: nvcompStatus_t) -> *const core::ffi::c_char;

// ---- Per-codec alignment / size helpers (all codecs, nvCOMP 5+) ----
//
// Every codec exposes `XCompressGetRequiredAlignments`,
// `XDecompressGetRequiredAlignments`, `XGetDecompressSizeAsync`. We
// reuse the PFN type shapes defined earlier (for LZ4).

/// get required-alignment hints for ANS batched compression.
pub type PFN_nvcompBatchedANSCompressGetRequiredAlignments = unsafe extern "C" fn(
    opts: nvcompBatchedANSOpts_t,
    alignments_out: *mut nvcompAlignmentRequirements_t,
)
    -> nvcompStatus_t;
/// get required-alignment hints for Bitcomp batched compression.
pub type PFN_nvcompBatchedBitcompCompressGetRequiredAlignments =
    unsafe extern "C" fn(
        opts: nvcompBatchedBitcompOpts_t,
        alignments_out: *mut nvcompAlignmentRequirements_t,
    ) -> nvcompStatus_t;
/// get required-alignment hints for Cascaded batched compression.
pub type PFN_nvcompBatchedCascadedCompressGetRequiredAlignments =
    unsafe extern "C" fn(
        opts: nvcompBatchedCascadedOpts_t,
        alignments_out: *mut nvcompAlignmentRequirements_t,
    ) -> nvcompStatus_t;
/// get required-alignment hints for Deflate batched compression.
pub type PFN_nvcompBatchedDeflateCompressGetRequiredAlignments =
    unsafe extern "C" fn(
        opts: nvcompBatchedDeflateOpts_t,
        alignments_out: *mut nvcompAlignmentRequirements_t,
    ) -> nvcompStatus_t;
/// get required-alignment hints for Gdeflate batched compression.
pub type PFN_nvcompBatchedGdeflateCompressGetRequiredAlignments =
    unsafe extern "C" fn(
        opts: nvcompBatchedGdeflateOpts_t,
        alignments_out: *mut nvcompAlignmentRequirements_t,
    ) -> nvcompStatus_t;
/// get required-alignment hints for Snappy batched compression.
pub type PFN_nvcompBatchedSnappyCompressGetRequiredAlignments =
    unsafe extern "C" fn(
        opts: nvcompBatchedSnappyOpts_t,
        alignments_out: *mut nvcompAlignmentRequirements_t,
    ) -> nvcompStatus_t;
/// get required-alignment hints for Zstd batched compression.
pub type PFN_nvcompBatchedZstdCompressGetRequiredAlignments =
    unsafe extern "C" fn(
        opts: nvcompBatchedZstdOpts_t,
        alignments_out: *mut nvcompAlignmentRequirements_t,
    ) -> nvcompStatus_t;

/// get required-alignment hints for ANS batched decompression.
pub type PFN_nvcompBatchedANSDecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;
/// get required-alignment hints for Bitcomp batched decompression.
pub type PFN_nvcompBatchedBitcompDecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;
/// get required-alignment hints for Cascaded batched decompression.
pub type PFN_nvcompBatchedCascadedDecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;
/// get required-alignment hints for Deflate batched decompression.
pub type PFN_nvcompBatchedDeflateDecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;
/// get required-alignment hints for Gdeflate batched decompression.
pub type PFN_nvcompBatchedGdeflateDecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;
/// get required-alignment hints for Gzip batched decompression.
pub type PFN_nvcompBatchedGzipDecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;
/// get required-alignment hints for Snappy batched decompression.
pub type PFN_nvcompBatchedSnappyDecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;
/// get required-alignment hints for Zstd batched decompression.
pub type PFN_nvcompBatchedZstdDecompressGetRequiredAlignments =
    unsafe extern "C" fn(alignments_out: *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t;

// ---- Missing max-output-chunk-size PFNs (ANS / Bitcomp / Cascaded / Deflate) ----

/// get max output chunk size for ANS batched compression.
pub type PFN_nvcompBatchedANSCompressGetMaxOutputChunkSize = unsafe extern "C" fn(
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedANSOpts_t,
    max_compressed_bytes_out: *mut usize,
)
    -> nvcompStatus_t;

/// get max output chunk size for Bitcomp batched compression.
pub type PFN_nvcompBatchedBitcompCompressGetMaxOutputChunkSize =
    unsafe extern "C" fn(
        max_uncompressed_chunk_size: usize,
        opts: nvcompBatchedBitcompOpts_t,
        max_compressed_bytes_out: *mut usize,
    ) -> nvcompStatus_t;

/// get max output chunk size for Cascaded batched compression.
pub type PFN_nvcompBatchedCascadedCompressGetMaxOutputChunkSize =
    unsafe extern "C" fn(
        max_uncompressed_chunk_size: usize,
        opts: nvcompBatchedCascadedOpts_t,
        max_compressed_bytes_out: *mut usize,
    ) -> nvcompStatus_t;

/// get max output chunk size for Deflate batched compression.
pub type PFN_nvcompBatchedDeflateCompressGetMaxOutputChunkSize =
    unsafe extern "C" fn(
        max_uncompressed_chunk_size: usize,
        opts: nvcompBatchedDeflateOpts_t,
        max_compressed_bytes_out: *mut usize,
    ) -> nvcompStatus_t;

// ---- Missing compress-temp-size PFNs (ANS / Bitcomp / Cascaded / Deflate) ----

/// get scratch size for ANS batched compression.
pub type PFN_nvcompBatchedANSCompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedANSOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get scratch size for Bitcomp batched compression.
pub type PFN_nvcompBatchedBitcompCompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedBitcompOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get scratch size for Cascaded batched compression.
pub type PFN_nvcompBatchedCascadedCompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedCascadedOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get scratch size for Deflate batched compression.
pub type PFN_nvcompBatchedDeflateCompressGetTempSize = unsafe extern "C" fn(
    batch_size: usize,
    max_uncompressed_chunk_size: usize,
    opts: nvcompBatchedDeflateOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

// ---- Missing decompress-temp-size PFNs (ANS / Bitcomp / Cascaded / Deflate) ----

// ANS + Bitcomp + Cascaded don't have a DecompressOpts_t; they share
// the LZ4-style "num_chunks + max_chunk + temp_out + max_total" signature.

/// get scratch size for Simple batched decompression.
pub type PFN_nvcompBatchedSimpleDecompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

/// get scratch size for Deflate batched decompression.
pub type PFN_nvcompBatchedDeflateDecompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_uncompressed_chunk_bytes: usize,
    decompress_opts: nvcompBatchedDeflateOpts_t,
    temp_bytes_out: *mut usize,
    max_total_uncompressed_bytes: usize,
) -> nvcompStatus_t;

// ---- Loader ----

macro_rules! nvcomp_fns {
    ($($(#[$attr:meta])* fn $name:ident as $sym:literal : $pfn:ty;)*) => {
        /// nvCOMP dynamic-loader handle.
        pub struct Nvcomp {
            pub lib: Library,
            $(
                $name: OnceLock<$pfn>,
            )*
        }

        impl core::fmt::Debug for Nvcomp {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Nvcomp").field("lib", &self.lib).finish_non_exhaustive()
            }
        }

        impl Nvcomp {
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
            $(
                $(#[$attr])*
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

nvcomp_fns! {
    // LZ4 — nvCOMP 5.x uses `*GetTempSizeSync` for the host-sizing call.
    fn lz4_compress_get_temp_size as "nvcompBatchedLZ4CompressGetTempSizeAsync":
        PFN_nvcompBatchedLZ4CompressGetTempSize;
    fn lz4_compress_get_max_output_chunk_size as "nvcompBatchedLZ4CompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedLZ4CompressGetMaxOutputChunkSize;
    fn lz4_compress_async as "nvcompBatchedLZ4CompressAsync": PFN_nvcompBatchedLZ4CompressAsync;
    fn lz4_compress_get_required_alignments as "nvcompBatchedLZ4CompressGetRequiredAlignments":
        PFN_nvcompBatchedLZ4CompressGetRequiredAlignments;
    fn lz4_decompress_get_temp_size as "nvcompBatchedLZ4DecompressGetTempSizeAsync":
        PFN_nvcompBatchedLZ4DecompressGetTempSize;
    fn lz4_decompress_async as "nvcompBatchedLZ4DecompressAsync":
        PFN_nvcompBatchedLZ4DecompressAsync;
    fn lz4_decompress_get_required_alignments as "nvcompBatchedLZ4DecompressGetRequiredAlignments":
        PFN_nvcompBatchedLZ4DecompressGetRequiredAlignments;
    fn lz4_get_decompress_size_async as "nvcompBatchedLZ4GetDecompressSizeAsync":
        PFN_nvcompBatchedLZ4GetDecompressSizeAsync;

    // Snappy
    fn snappy_compress_get_temp_size as "nvcompBatchedSnappyCompressGetTempSizeAsync":
        PFN_nvcompBatchedSnappyCompressGetTempSize;
    fn snappy_compress_get_max_output_chunk_size
        as "nvcompBatchedSnappyCompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedSnappyCompressGetMaxOutputChunkSize;
    fn snappy_compress_async as "nvcompBatchedSnappyCompressAsync":
        PFN_nvcompBatchedSnappyCompressAsync;
    fn snappy_decompress_get_temp_size as "nvcompBatchedSnappyDecompressGetTempSizeAsync":
        PFN_nvcompBatchedSnappyDecompressGetTempSize;
    fn snappy_decompress_async as "nvcompBatchedSnappyDecompressAsync":
        PFN_nvcompBatchedSnappyDecompressAsync;
    fn snappy_get_decompress_size_async as "nvcompBatchedSnappyGetDecompressSizeAsync":
        PFN_nvcompGetDecompressSizeAsync;

    // Zstd
    fn zstd_compress_get_temp_size as "nvcompBatchedZstdCompressGetTempSizeAsync":
        PFN_nvcompBatchedZstdCompressGetTempSize;
    fn zstd_compress_get_max_output_chunk_size
        as "nvcompBatchedZstdCompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedZstdCompressGetMaxOutputChunkSize;
    fn zstd_compress_async as "nvcompBatchedZstdCompressAsync":
        PFN_nvcompBatchedZstdCompressAsync;
    fn zstd_decompress_get_temp_size as "nvcompBatchedZstdDecompressGetTempSizeAsync":
        PFN_nvcompBatchedZstdDecompressGetTempSize;
    fn zstd_decompress_async as "nvcompBatchedZstdDecompressAsync":
        PFN_nvcompBatchedZstdDecompressAsync;
    fn zstd_get_decompress_size_async as "nvcompBatchedZstdGetDecompressSizeAsync":
        PFN_nvcompGetDecompressSizeAsync;

    // GDeflate
    fn gdeflate_compress_get_temp_size as "nvcompBatchedGdeflateCompressGetTempSizeAsync":
        PFN_nvcompBatchedGdeflateCompressGetTempSize;
    fn gdeflate_compress_get_max_output_chunk_size
        as "nvcompBatchedGdeflateCompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedGdeflateCompressGetMaxOutputChunkSize;
    fn gdeflate_compress_async as "nvcompBatchedGdeflateCompressAsync":
        PFN_nvcompBatchedGdeflateCompressAsync;
    fn gdeflate_decompress_get_temp_size as "nvcompBatchedGdeflateDecompressGetTempSizeAsync":
        PFN_nvcompBatchedGdeflateDecompressGetTempSize;
    fn gdeflate_decompress_async as "nvcompBatchedGdeflateDecompressAsync":
        PFN_nvcompBatchedGdeflateDecompressAsync;

    // Deflate
    fn deflate_compress_async as "nvcompBatchedDeflateCompressAsync":
        PFN_nvcompBatchedDeflateCompressAsync;
    fn deflate_decompress_async as "nvcompBatchedDeflateDecompressAsync":
        PFN_nvcompBatchedDeflateDecompressAsync;

    // Bitcomp
    fn bitcomp_compress_async as "nvcompBatchedBitcompCompressAsync":
        PFN_nvcompBatchedBitcompCompressAsync;
    fn bitcomp_decompress_async as "nvcompBatchedBitcompDecompressAsync":
        PFN_nvcompBatchedBitcompDecompressAsync;

    // ANS
    fn ans_compress_async as "nvcompBatchedANSCompressAsync":
        PFN_nvcompBatchedANSCompressAsync;
    fn ans_decompress_async as "nvcompBatchedANSDecompressAsync":
        PFN_nvcompBatchedANSDecompressAsync;

    // Cascaded
    fn cascaded_compress_async as "nvcompBatchedCascadedCompressAsync":
        PFN_nvcompBatchedCascadedCompressAsync;
    fn cascaded_decompress_async as "nvcompBatchedCascadedDecompressAsync":
        PFN_nvcompBatchedCascadedDecompressAsync;

    // ---- Gzip ----
    fn gzip_compress_get_temp_size as "nvcompBatchedGzipCompressGetTempSizeAsync":
        PFN_nvcompBatchedGzipCompressGetTempSize;
    fn gzip_compress_get_max_output_chunk_size
        as "nvcompBatchedGzipCompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedGzipCompressGetMaxOutputChunkSize;
    fn gzip_compress_async as "nvcompBatchedGzipCompressAsync":
        PFN_nvcompBatchedGzipCompressAsync;
    fn gzip_decompress_get_temp_size as "nvcompBatchedGzipDecompressGetTempSizeAsync":
        PFN_nvcompBatchedGzipDecompressGetTempSize;
    fn gzip_decompress_async as "nvcompBatchedGzipDecompressAsync":
        PFN_nvcompBatchedGzipDecompressAsync;
    fn gzip_get_decompress_size_async as "nvcompBatchedGzipGetDecompressSizeAsync":
        PFN_nvcompGetDecompressSizeAsync;

    // ---- CRC32 batched checksums ----
    fn crc32_async as "nvcompBatchedCRC32Async": PFN_nvcompBatchedCRC32Async;
    fn crc32_get_heuristic_conf as "nvcompBatchedCRC32GetHeuristicConf":
        PFN_nvcompBatchedCRC32GetHeuristicConf;
    fn crc32_search_conf as "nvcompBatchedCRC32SearchConf":
        PFN_nvcompBatchedCRC32SearchConf;

    // ---- Library metadata ----
    fn nvcomp_get_properties as "nvcompGetProperties": PFN_nvcompGetProperties;
    fn nvcomp_get_status_string as "nvcompGetStatusString": PFN_nvcompGetStatusString;

    // ---- Per-codec alignment helpers (non-LZ4) ----
    fn ans_compress_get_required_alignments as "nvcompBatchedANSCompressGetRequiredAlignments":
        PFN_nvcompBatchedANSCompressGetRequiredAlignments;
    fn ans_decompress_get_required_alignments as "nvcompBatchedANSDecompressGetRequiredAlignments":
        PFN_nvcompBatchedANSDecompressGetRequiredAlignments;
    fn bitcomp_compress_get_required_alignments
        as "nvcompBatchedBitcompCompressGetRequiredAlignments":
        PFN_nvcompBatchedBitcompCompressGetRequiredAlignments;
    fn bitcomp_decompress_get_required_alignments
        as "nvcompBatchedBitcompDecompressGetRequiredAlignments":
        PFN_nvcompBatchedBitcompDecompressGetRequiredAlignments;
    fn cascaded_compress_get_required_alignments
        as "nvcompBatchedCascadedCompressGetRequiredAlignments":
        PFN_nvcompBatchedCascadedCompressGetRequiredAlignments;
    fn cascaded_decompress_get_required_alignments
        as "nvcompBatchedCascadedDecompressGetRequiredAlignments":
        PFN_nvcompBatchedCascadedDecompressGetRequiredAlignments;
    fn deflate_compress_get_required_alignments
        as "nvcompBatchedDeflateCompressGetRequiredAlignments":
        PFN_nvcompBatchedDeflateCompressGetRequiredAlignments;
    fn deflate_decompress_get_required_alignments
        as "nvcompBatchedDeflateDecompressGetRequiredAlignments":
        PFN_nvcompBatchedDeflateDecompressGetRequiredAlignments;
    fn gdeflate_compress_get_required_alignments
        as "nvcompBatchedGdeflateCompressGetRequiredAlignments":
        PFN_nvcompBatchedGdeflateCompressGetRequiredAlignments;
    fn gdeflate_decompress_get_required_alignments
        as "nvcompBatchedGdeflateDecompressGetRequiredAlignments":
        PFN_nvcompBatchedGdeflateDecompressGetRequiredAlignments;
    fn gzip_decompress_get_required_alignments
        as "nvcompBatchedGzipDecompressGetRequiredAlignments":
        PFN_nvcompBatchedGzipDecompressGetRequiredAlignments;
    fn snappy_compress_get_required_alignments
        as "nvcompBatchedSnappyCompressGetRequiredAlignments":
        PFN_nvcompBatchedSnappyCompressGetRequiredAlignments;
    fn snappy_decompress_get_required_alignments
        as "nvcompBatchedSnappyDecompressGetRequiredAlignments":
        PFN_nvcompBatchedSnappyDecompressGetRequiredAlignments;
    fn zstd_compress_get_required_alignments as "nvcompBatchedZstdCompressGetRequiredAlignments":
        PFN_nvcompBatchedZstdCompressGetRequiredAlignments;
    fn zstd_decompress_get_required_alignments
        as "nvcompBatchedZstdDecompressGetRequiredAlignments":
        PFN_nvcompBatchedZstdDecompressGetRequiredAlignments;

    // ---- Get-decompress-size-async for non-LZ4 codecs ----
    fn ans_get_decompress_size_async as "nvcompBatchedANSGetDecompressSizeAsync":
        PFN_nvcompGetDecompressSizeAsync;
    fn bitcomp_get_decompress_size_async as "nvcompBatchedBitcompGetDecompressSizeAsync":
        PFN_nvcompGetDecompressSizeAsync;
    fn cascaded_get_decompress_size_async as "nvcompBatchedCascadedGetDecompressSizeAsync":
        PFN_nvcompGetDecompressSizeAsync;
    fn deflate_get_decompress_size_async as "nvcompBatchedDeflateGetDecompressSizeAsync":
        PFN_nvcompGetDecompressSizeAsync;
    fn gdeflate_get_decompress_size_async as "nvcompBatchedGdeflateGetDecompressSizeAsync":
        PFN_nvcompGetDecompressSizeAsync;

    // ---- Missing max-output-chunk for ANS / Bitcomp / Cascaded / Deflate ----
    fn ans_compress_get_max_output_chunk_size as "nvcompBatchedANSCompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedANSCompressGetMaxOutputChunkSize;
    fn bitcomp_compress_get_max_output_chunk_size
        as "nvcompBatchedBitcompCompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedBitcompCompressGetMaxOutputChunkSize;
    fn cascaded_compress_get_max_output_chunk_size
        as "nvcompBatchedCascadedCompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedCascadedCompressGetMaxOutputChunkSize;
    fn deflate_compress_get_max_output_chunk_size
        as "nvcompBatchedDeflateCompressGetMaxOutputChunkSize":
        PFN_nvcompBatchedDeflateCompressGetMaxOutputChunkSize;

    // ---- Missing compress temp-size for ANS / Bitcomp / Cascaded / Deflate ----
    fn ans_compress_get_temp_size as "nvcompBatchedANSCompressGetTempSizeAsync":
        PFN_nvcompBatchedANSCompressGetTempSize;
    fn bitcomp_compress_get_temp_size as "nvcompBatchedBitcompCompressGetTempSizeAsync":
        PFN_nvcompBatchedBitcompCompressGetTempSize;
    fn cascaded_compress_get_temp_size as "nvcompBatchedCascadedCompressGetTempSizeAsync":
        PFN_nvcompBatchedCascadedCompressGetTempSize;
    fn deflate_compress_get_temp_size as "nvcompBatchedDeflateCompressGetTempSizeAsync":
        PFN_nvcompBatchedDeflateCompressGetTempSize;

    // ---- Missing decompress temp-size for ANS / Bitcomp / Cascaded / Deflate ----
    fn ans_decompress_get_temp_size as "nvcompBatchedANSDecompressGetTempSizeAsync":
        PFN_nvcompBatchedSimpleDecompressGetTempSize;
    fn bitcomp_decompress_get_temp_size as "nvcompBatchedBitcompDecompressGetTempSizeAsync":
        PFN_nvcompBatchedSimpleDecompressGetTempSize;
    fn cascaded_decompress_get_temp_size as "nvcompBatchedCascadedDecompressGetTempSizeAsync":
        PFN_nvcompBatchedSimpleDecompressGetTempSize;
    fn deflate_decompress_get_temp_size as "nvcompBatchedDeflateDecompressGetTempSizeAsync":
        PFN_nvcompBatchedDeflateDecompressGetTempSize;
}

fn nvcomp_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &[
            "libnvcomp.so.5",
            "libnvcomp.so.4",
            "libnvcomp.so.3",
            "libnvcomp.so",
        ]
    }
    #[cfg(target_os = "windows")]
    {
        &[
            "nvcomp64_5.dll",
            "nvcomp64_4.dll",
            "nvcomp64.dll",
            "nvcomp.dll",
        ]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

/// Extra Windows search paths where NVIDIA's nvCOMP installer drops the
/// DLL — typically `C:\Program Files\NVIDIA nvCOMP\v<ver>\bin\<cuda>\`.
#[cfg(target_os = "windows")]
fn nvcomp_extra_dirs() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
    let mut out = Vec::new();
    let progfiles = std::env::var("ProgramFiles").unwrap_or_else(|_| "C:\\Program Files".into());
    for root_name in ["NVIDIA nvCOMP", "NVIDIA\\nvCOMP"] {
        let root = PathBuf::from(&progfiles).join(root_name);
        if let Ok(top) = std::fs::read_dir(&root) {
            for ent in top.flatten() {
                let p = ent.path();
                if p.is_dir() {
                    for sub in ["bin\\13", "bin\\12", "bin\\11", "bin", "lib\\13", "lib\\12"] {
                        out.push(p.join(sub));
                    }
                }
            }
        }
    }
    out
}

/// resolve and return the process-wide nvCOMP loader.
pub fn nvcomp() -> Result<&'static Nvcomp, LoaderError> {
    static NVCOMP: OnceLock<Nvcomp> = OnceLock::new();
    if let Some(n) = NVCOMP.get() {
        return Ok(n);
    }
    let lib = match Library::open("nvcomp", nvcomp_candidates()) {
        Ok(l) => l,
        Err(e) => {
            #[cfg(target_os = "windows")]
            {
                let mut found: Option<Library> = None;
                for dir in nvcomp_extra_dirs() {
                    for candidate in nvcomp_candidates() {
                        let full = dir.join(candidate);
                        if let Ok(l) = Library::open_at("nvcomp", &full) {
                            found = Some(l);
                            break;
                        }
                    }
                    if found.is_some() {
                        break;
                    }
                }
                match found {
                    Some(l) => l,
                    None => return Err(e),
                }
            }
            #[cfg(not(target_os = "windows"))]
            {
                return Err(e);
            }
        }
    };
    let _ = NVCOMP.set(Nvcomp::empty(lib));
    Ok(NVCOMP.get().expect("OnceLock set or lost race"))
}
