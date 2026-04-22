//! Safe Rust wrappers for NVIDIA nvCOMP (GPU compression).
//!
//! nvCOMP ships host-side batched APIs for several codecs:
//!
//! | Codec     | Compress | Decompress | Notes |
//! | --------- | :------: | :--------: | ----- |
//! | LZ4       |   ✓      |    ✓       | Lowest CPU overhead; good general purpose |
//! | Snappy    |   ✓      |    ✓       | OpenCV-compatible; ~LZ4-level ratio |
//! | Zstd      |   ✓      |    ✓       | Higher ratio, slower |
//! | GDeflate  |   ✓      |    ✓       | Optimized for GPU; random-access friendly |
//! | Deflate   |   ✓      |    ✓       | Strict zlib/raw deflate |
//! | Bitcomp   |   ✓      |    ✓       | Columnar/numerical data |
//! | ANS       |   ✓      |    ✓       | Range-coded entropy |
//! | Cascaded  |   ✓      |    ✓       | Numerical RLE + delta + bit-packing |
//!
//! Every codec shares the same flow:
//!
//! 1. Sizing: `compress_get_temp_size(batch, max_chunk, opts)`, then
//!    `compress_get_max_output_chunk_size(max_chunk, opts)` to pre-allocate.
//! 2. Compress async: `compress_async(in_ptrs, in_bytes, max_chunk, batch,
//!    temp, out_ptrs, out_bytes, opts, stream)`.
//! 3. Decompress sizing: `decompress_get_temp_size(batch, max_chunk)`.
//! 4. Optional: `get_decompress_size_async` to fill a `uncompressed_bytes`
//!    array from compressed buffers (nvCOMP 4+, per-codec).
//! 5. Decompress async: `decompress_async(...)`.
//!
//! Device pointers are passed as arrays of `batch` pointers (input /
//! output chunks) and matching arrays of sizes. See nvCOMP headers for
//! the exact device-side layout — this crate is a thin FFI surface.

#![warn(missing_debug_implementations)]

use core::ffi::c_void;

use baracuda_nvcomp_sys::{nvcomp, nvcompStatus_t};

pub use baracuda_nvcomp_sys::{
    nvcompAlignmentRequirements_t, nvcompBatchedANSOpts_t, nvcompBatchedBitcompOpts_t,
    nvcompBatchedCascadedOpts_t, nvcompBatchedDeflateOpts_t, nvcompBatchedGdeflateDecompressOpts_t,
    nvcompBatchedGdeflateOpts_t, nvcompBatchedGzipDecompressOpts_t, nvcompBatchedGzipOpts_t,
    nvcompBatchedLZ4DecompressOpts_t, nvcompBatchedLZ4Opts_t, nvcompBatchedSnappyDecompressOpts_t,
    nvcompBatchedSnappyOpts_t, nvcompBatchedZstdDecompressOpts_t, nvcompBatchedZstdOpts_t,
    nvcompDecompressBackend, nvcompType,
};

/// Human-readable name for an `nvcompStatus_t`.
pub fn status_string(status: nvcompStatus_t) -> Result<&'static str> {
    let n = nvcomp()?;
    let cu = n.nvcomp_get_status_string()?;
    let p = unsafe { cu(status) };
    if p.is_null() {
        return Ok("unknown");
    }
    // SAFETY: nvCOMP returns a static C string.
    Ok(unsafe { core::ffi::CStr::from_ptr(p) }
        .to_str()
        .unwrap_or("unknown"))
}

/// Library-level properties. The `properties` pointer is passed through
/// raw since nvCOMP's `nvcompProperties_t` layout is headers-only — use
/// `baracuda_nvcomp_sys::PFN_nvcompGetProperties` directly if you need
/// the typed fields.
///
/// # Safety
///
/// `props` must point at an `nvcompProperties_t` buffer.
pub unsafe fn get_properties(props: *mut core::ffi::c_void) -> Result<()> {
    let n = nvcomp()?;
    let cu = n.nvcomp_get_properties()?;
    check(cu(props))
}

pub mod crc32 {
    //! Batched CRC32 checksums (nvCOMP 5+).

    use super::*;

    /// Pick a reasonable config for `max_chunk_bytes`.
    ///
    /// # Safety
    ///
    /// `config_out` must point at an `nvcompCRC32Config_t` buffer.
    pub unsafe fn get_heuristic_conf(
        max_chunk_bytes: usize,
        config_out: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.crc32_get_heuristic_conf()?;
        check(cu(max_chunk_bytes, config_out))
    }

    /// Brute-force search for the best config (slower; offline use).
    ///
    /// # Safety
    ///
    /// Same as [`get_heuristic_conf`].
    pub unsafe fn search_conf(max_chunk_bytes: usize, config_out: *mut c_void) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.crc32_search_conf()?;
        check(cu(max_chunk_bytes, config_out))
    }

    /// Compute per-chunk CRC32s on `stream`.
    ///
    /// # Safety
    ///
    /// `config` must be filled by [`get_heuristic_conf`] or [`search_conf`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn compute_async(
        dev_data_ptrs: *const *const c_void,
        dev_data_bytes: *const usize,
        batch_size: usize,
        config: *const c_void,
        dev_crc_out: *mut u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.crc32_async()?;
        check(cu(
            dev_data_ptrs,
            dev_data_bytes,
            batch_size,
            config,
            dev_crc_out,
            stream,
        ))
    }
}

/// Query compress-side alignment requirements for any codec that
/// follows the standard (opts-by-value, alignments-out) PFN shape.
///
/// # Safety
///
/// `pfn` must be the PFN for the codec's `CompressGetRequiredAlignments`;
/// `opts` must match the type the PFN expects.
pub unsafe fn compress_alignment<Opts: Copy>(
    pfn: unsafe extern "C" fn(Opts, *mut nvcompAlignmentRequirements_t) -> nvcompStatus_t,
    opts: Opts,
) -> Result<nvcompAlignmentRequirements_t> {
    let mut a = nvcompAlignmentRequirements_t::default();
    check(pfn(opts, &mut a))?;
    Ok(a)
}

/// Query decompress-side alignment requirements.
///
/// # Safety
///
/// Same as [`compress_alignment`].
pub unsafe fn decompress_alignment(
    pfn: unsafe extern "C" fn(*mut nvcompAlignmentRequirements_t) -> nvcompStatus_t,
) -> Result<nvcompAlignmentRequirements_t> {
    let mut a = nvcompAlignmentRequirements_t::default();
    check(pfn(&mut a))?;
    Ok(a)
}

/// Query per-chunk uncompressed sizes from compressed buffers (v5+
/// codecs). Shape is shared across LZ4/Snappy/Zstd/GDeflate/Gzip/
/// Deflate/ANS/Bitcomp/Cascaded.
///
/// # Safety
///
/// `pfn` must be the codec's `GetDecompressSizeAsync`; pointers must be
/// live + batch-consistent.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_decompress_size_async(
    pfn: unsafe extern "C" fn(
        *const *const c_void,
        *const usize,
        *mut usize,
        usize,
        *mut c_void,
    ) -> nvcompStatus_t,
    dev_compressed_ptrs: *const *const c_void,
    dev_compressed_bytes: *const usize,
    dev_uncompressed_bytes_out: *mut usize,
    batch_size: usize,
    stream: *mut c_void,
) -> Result<()> {
    check(pfn(
        dev_compressed_ptrs,
        dev_compressed_bytes,
        dev_uncompressed_bytes_out,
        batch_size,
        stream,
    ))
}

/// Error type for nvCOMP operations.
pub type Error = baracuda_core::Error<nvcompStatus_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: nvcompStatus_t) -> Result<()> {
    Error::check(status)
}

/// Verify nvCOMP is loadable on this host.
pub fn probe() -> Result<()> {
    nvcomp()?;
    Ok(())
}

/// Raw byte-pointer arrays passed to compress/decompress. Kept in a
/// dedicated type so the call sites read nicer.
#[allow(clippy::missing_safety_doc)]
pub mod raw {
    use super::*;

    /// Result-agnostic launch of `compress_async` for any codec.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn compress_async<Opts>(
        pfn: unsafe extern "C" fn(
            *const *const c_void,
            *const usize,
            usize,
            usize,
            *mut c_void,
            usize,
            *const *mut c_void,
            *mut usize,
            Opts,
            *mut c_void,
        ) -> nvcompStatus_t,
        dev_uncompressed_ptrs: *const *const c_void,
        dev_uncompressed_bytes: *const usize,
        max_chunk_bytes: usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_compressed_ptrs: *const *mut c_void,
        dev_compressed_bytes: *mut usize,
        opts: Opts,
        stream: *mut c_void,
    ) -> Result<()> {
        super::check(pfn(
            dev_uncompressed_ptrs,
            dev_uncompressed_bytes,
            max_chunk_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_compressed_ptrs,
            dev_compressed_bytes,
            opts,
            stream,
        ))
    }

    /// Result-agnostic launch of `decompress_async` for any codec that
    /// follows the standard 10-argument signature.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decompress_async(
        pfn: unsafe extern "C" fn(
            *const *const c_void,
            *const usize,
            *const usize,
            *mut usize,
            usize,
            *mut c_void,
            usize,
            *const *mut c_void,
            *mut nvcompStatus_t,
            *mut c_void,
        ) -> nvcompStatus_t,
        dev_compressed_ptrs: *const *const c_void,
        dev_compressed_bytes: *const usize,
        dev_uncompressed_bytes: *const usize,
        dev_actual_uncompressed_bytes: *mut usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_uncompressed_ptrs: *const *mut c_void,
        dev_status_ptrs: *mut nvcompStatus_t,
        stream: *mut c_void,
    ) -> Result<()> {
        super::check(pfn(
            dev_compressed_ptrs,
            dev_compressed_bytes,
            dev_uncompressed_bytes,
            dev_actual_uncompressed_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_uncompressed_ptrs,
            dev_status_ptrs,
            stream,
        ))
    }
}

/// Macro generating a codec module with the standard set of functions.
macro_rules! codec_mod {
    (
        $(#[$m:meta])*
        $modname:ident,
        $copts:ty,
        $dopts:ty,
        $compress_get_temp:ident,
        $compress_get_max:ident,
        $compress:ident,
        $decompress_get_temp:ident,
        $decompress:ident $(,)?
    ) => {
        $(#[$m])*
        pub mod $modname {
            use super::*;

            /// Device scratch-buffer size for a compress op.
            ///
            /// `max_total_uncompressed_bytes` is a sizing hint — pass
            /// `batch_size * max_chunk_bytes` for the default.
            pub fn compress_get_temp_size(
                batch_size: usize,
                max_chunk_bytes: usize,
                opts: $copts,
                max_total_uncompressed_bytes: usize,
            ) -> Result<usize> {
                let n = nvcomp()?;
                let cu = n.$compress_get_temp()?;
                let mut out: usize = 0;
                check(unsafe {
                    cu(
                        batch_size,
                        max_chunk_bytes,
                        opts,
                        &mut out,
                        max_total_uncompressed_bytes,
                    )
                })?;
                Ok(out)
            }

            /// Worst-case compressed output size per chunk.
            pub fn compress_get_max_output_chunk_size(
                max_chunk_bytes: usize,
                opts: $copts,
            ) -> Result<usize> {
                let n = nvcomp()?;
                let cu = n.$compress_get_max()?;
                let mut out: usize = 0;
                check(unsafe { cu(max_chunk_bytes, opts, &mut out) })?;
                Ok(out)
            }

            /// Launch batched compression on `stream` (v5+ signature
            /// with per-chunk `device_statuses`).
            ///
            /// # Safety
            ///
            /// All device pointers must be live and batch-consistent.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn compress_async(
                dev_uncompressed_ptrs: *const *const c_void,
                dev_uncompressed_bytes: *const usize,
                max_chunk_bytes: usize,
                batch_size: usize,
                dev_temp_ptr: *mut c_void,
                temp_bytes: usize,
                dev_compressed_ptrs: *const *mut c_void,
                dev_compressed_bytes: *mut usize,
                opts: $copts,
                dev_statuses: *mut nvcompStatus_t,
                stream: *mut c_void,
            ) -> Result<()> {
                let n = nvcomp()?;
                let cu = n.$compress()?;
                check(cu(
                    dev_uncompressed_ptrs,
                    dev_uncompressed_bytes,
                    max_chunk_bytes,
                    batch_size,
                    dev_temp_ptr,
                    temp_bytes,
                    dev_compressed_ptrs,
                    dev_compressed_bytes,
                    opts,
                    dev_statuses,
                    stream,
                ))
            }

            /// Device scratch-buffer size for decompression.
            pub fn decompress_get_temp_size(
                num_chunks: usize,
                max_chunk_bytes: usize,
                opts: $dopts,
                max_total_uncompressed_bytes: usize,
            ) -> Result<usize> {
                let n = nvcomp()?;
                let cu = n.$decompress_get_temp()?;
                let mut out: usize = 0;
                check(unsafe {
                    cu(
                        num_chunks,
                        max_chunk_bytes,
                        opts,
                        &mut out,
                        max_total_uncompressed_bytes,
                    )
                })?;
                Ok(out)
            }

            /// Launch batched decompression on `stream` (v5+ signature
            /// with `decompress_opts` + `device_statuses`).
            ///
            /// # Safety
            ///
            /// Same as [`compress_async`].
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn decompress_async(
                dev_compressed_ptrs: *const *const c_void,
                dev_compressed_bytes: *const usize,
                dev_uncompressed_bytes: *const usize,
                dev_actual_uncompressed_bytes: *mut usize,
                batch_size: usize,
                dev_temp_ptr: *mut c_void,
                temp_bytes: usize,
                dev_uncompressed_ptrs: *const *mut c_void,
                opts: $dopts,
                dev_status_ptrs: *mut nvcompStatus_t,
                stream: *mut c_void,
            ) -> Result<()> {
                let n = nvcomp()?;
                let cu = n.$decompress()?;
                check(cu(
                    dev_compressed_ptrs,
                    dev_compressed_bytes,
                    dev_uncompressed_bytes,
                    dev_actual_uncompressed_bytes,
                    batch_size,
                    dev_temp_ptr,
                    temp_bytes,
                    dev_uncompressed_ptrs,
                    opts,
                    dev_status_ptrs,
                    stream,
                ))
            }
        }
    };
}

codec_mod! {
    /// LZ4 batched compression / decompression.
    lz4, nvcompBatchedLZ4Opts_t, nvcompBatchedLZ4DecompressOpts_t,
    lz4_compress_get_temp_size, lz4_compress_get_max_output_chunk_size,
    lz4_compress_async, lz4_decompress_get_temp_size, lz4_decompress_async,
}

codec_mod! {
    /// Snappy batched compression / decompression.
    snappy, nvcompBatchedSnappyOpts_t, nvcompBatchedSnappyDecompressOpts_t,
    snappy_compress_get_temp_size, snappy_compress_get_max_output_chunk_size,
    snappy_compress_async, snappy_decompress_get_temp_size, snappy_decompress_async,
}

codec_mod! {
    /// Zstandard batched compression / decompression.
    zstd, nvcompBatchedZstdOpts_t, nvcompBatchedZstdDecompressOpts_t,
    zstd_compress_get_temp_size, zstd_compress_get_max_output_chunk_size,
    zstd_compress_async, zstd_decompress_get_temp_size, zstd_decompress_async,
}

codec_mod! {
    /// GDeflate batched compression / decompression (GPU-optimized zlib family).
    gdeflate, nvcompBatchedGdeflateOpts_t, nvcompBatchedGdeflateDecompressOpts_t,
    gdeflate_compress_get_temp_size, gdeflate_compress_get_max_output_chunk_size,
    gdeflate_compress_async, gdeflate_decompress_get_temp_size, gdeflate_decompress_async,
}

codec_mod! {
    /// Gzip batched compression / decompression (nvCOMP 5+).
    gzip, nvcompBatchedGzipOpts_t, nvcompBatchedGzipDecompressOpts_t,
    gzip_compress_get_temp_size, gzip_compress_get_max_output_chunk_size,
    gzip_compress_async, gzip_decompress_get_temp_size, gzip_decompress_async,
}

pub mod deflate {
    //! Raw Deflate / zlib-compatible stream compression.
    use super::*;

    /// # Safety
    ///
    /// Same as [`lz4::compress_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn compress_async(
        dev_uncompressed_ptrs: *const *const c_void,
        dev_uncompressed_bytes: *const usize,
        max_chunk_bytes: usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_compressed_ptrs: *const *mut c_void,
        dev_compressed_bytes: *mut usize,
        opts: nvcompBatchedDeflateOpts_t,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.deflate_compress_async()?;
        check(cu(
            dev_uncompressed_ptrs,
            dev_uncompressed_bytes,
            max_chunk_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_compressed_ptrs,
            dev_compressed_bytes,
            opts,
            stream,
        ))
    }

    /// # Safety
    ///
    /// Same as [`lz4::decompress_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decompress_async(
        dev_compressed_ptrs: *const *const c_void,
        dev_compressed_bytes: *const usize,
        dev_uncompressed_bytes: *const usize,
        dev_actual_uncompressed_bytes: *mut usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_uncompressed_ptrs: *const *mut c_void,
        dev_status_ptrs: *mut nvcompStatus_t,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.deflate_decompress_async()?;
        check(cu(
            dev_compressed_ptrs,
            dev_compressed_bytes,
            dev_uncompressed_bytes,
            dev_actual_uncompressed_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_uncompressed_ptrs,
            dev_status_ptrs,
            stream,
        ))
    }
}

pub mod bitcomp {
    //! Bitcomp batched compression (columnar/numerical data).
    use super::*;

    /// # Safety
    ///
    /// Same as [`lz4::compress_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn compress_async(
        dev_uncompressed_ptrs: *const *const c_void,
        dev_uncompressed_bytes: *const usize,
        max_chunk_bytes: usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_compressed_ptrs: *const *mut c_void,
        dev_compressed_bytes: *mut usize,
        opts: nvcompBatchedBitcompOpts_t,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.bitcomp_compress_async()?;
        check(cu(
            dev_uncompressed_ptrs,
            dev_uncompressed_bytes,
            max_chunk_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_compressed_ptrs,
            dev_compressed_bytes,
            opts,
            stream,
        ))
    }

    /// # Safety
    ///
    /// Same as [`lz4::decompress_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decompress_async(
        dev_compressed_ptrs: *const *const c_void,
        dev_compressed_bytes: *const usize,
        dev_uncompressed_bytes: *const usize,
        dev_actual_uncompressed_bytes: *mut usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_uncompressed_ptrs: *const *mut c_void,
        dev_status_ptrs: *mut nvcompStatus_t,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.bitcomp_decompress_async()?;
        check(cu(
            dev_compressed_ptrs,
            dev_compressed_bytes,
            dev_uncompressed_bytes,
            dev_actual_uncompressed_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_uncompressed_ptrs,
            dev_status_ptrs,
            stream,
        ))
    }
}

pub mod ans {
    //! ANS (Asymmetric Numeral Systems) batched compression.
    use super::*;

    /// # Safety
    ///
    /// Same as [`lz4::compress_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn compress_async(
        dev_uncompressed_ptrs: *const *const c_void,
        dev_uncompressed_bytes: *const usize,
        max_chunk_bytes: usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_compressed_ptrs: *const *mut c_void,
        dev_compressed_bytes: *mut usize,
        opts: nvcompBatchedANSOpts_t,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.ans_compress_async()?;
        check(cu(
            dev_uncompressed_ptrs,
            dev_uncompressed_bytes,
            max_chunk_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_compressed_ptrs,
            dev_compressed_bytes,
            opts,
            stream,
        ))
    }

    /// # Safety
    ///
    /// Same as [`lz4::decompress_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decompress_async(
        dev_compressed_ptrs: *const *const c_void,
        dev_compressed_bytes: *const usize,
        dev_uncompressed_bytes: *const usize,
        dev_actual_uncompressed_bytes: *mut usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_uncompressed_ptrs: *const *mut c_void,
        dev_status_ptrs: *mut nvcompStatus_t,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.ans_decompress_async()?;
        check(cu(
            dev_compressed_ptrs,
            dev_compressed_bytes,
            dev_uncompressed_bytes,
            dev_actual_uncompressed_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_uncompressed_ptrs,
            dev_status_ptrs,
            stream,
        ))
    }
}

pub mod cascaded {
    //! Cascaded compression (RLE + delta + bit-packing for numerical data).
    use super::*;

    /// # Safety
    ///
    /// Same as [`lz4::compress_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn compress_async(
        dev_uncompressed_ptrs: *const *const c_void,
        dev_uncompressed_bytes: *const usize,
        max_chunk_bytes: usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_compressed_ptrs: *const *mut c_void,
        dev_compressed_bytes: *mut usize,
        opts: nvcompBatchedCascadedOpts_t,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.cascaded_compress_async()?;
        check(cu(
            dev_uncompressed_ptrs,
            dev_uncompressed_bytes,
            max_chunk_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_compressed_ptrs,
            dev_compressed_bytes,
            opts,
            stream,
        ))
    }

    /// # Safety
    ///
    /// Same as [`lz4::decompress_async`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decompress_async(
        dev_compressed_ptrs: *const *const c_void,
        dev_compressed_bytes: *const usize,
        dev_uncompressed_bytes: *const usize,
        dev_actual_uncompressed_bytes: *mut usize,
        batch_size: usize,
        dev_temp_ptr: *mut c_void,
        temp_bytes: usize,
        dev_uncompressed_ptrs: *const *mut c_void,
        dev_status_ptrs: *mut nvcompStatus_t,
        stream: *mut c_void,
    ) -> Result<()> {
        let n = nvcomp()?;
        let cu = n.cascaded_decompress_async()?;
        check(cu(
            dev_compressed_ptrs,
            dev_compressed_bytes,
            dev_uncompressed_bytes,
            dev_actual_uncompressed_bytes,
            batch_size,
            dev_temp_ptr,
            temp_bytes,
            dev_uncompressed_ptrs,
            dev_status_ptrs,
            stream,
        ))
    }
}

/// Alignment requirements for LZ4 input / output / temp buffers
/// (nvCOMP 4+).
pub fn lz4_compress_alignment(
    opts: nvcompBatchedLZ4Opts_t,
) -> Result<nvcompAlignmentRequirements_t> {
    let n = nvcomp()?;
    let cu = n.lz4_compress_get_required_alignments()?;
    let mut align = nvcompAlignmentRequirements_t::default();
    check(unsafe { cu(opts, &mut align) })?;
    Ok(align)
}

/// Alignment requirements for LZ4 decompression buffers.
pub fn lz4_decompress_alignment() -> Result<nvcompAlignmentRequirements_t> {
    let n = nvcomp()?;
    let cu = n.lz4_decompress_get_required_alignments()?;
    let mut align = nvcompAlignmentRequirements_t::default();
    check(unsafe { cu(&mut align) })?;
    Ok(align)
}

/// Query per-chunk uncompressed sizes from compressed LZ4 buffers.
///
/// # Safety
///
/// Pointers must be live and batch-consistent.
pub unsafe fn lz4_get_decompress_size_async(
    dev_compressed_ptrs: *const *const c_void,
    dev_compressed_bytes: *const usize,
    dev_uncompressed_bytes_out: *mut usize,
    batch_size: usize,
    stream: *mut c_void,
) -> Result<()> {
    let n = nvcomp()?;
    let cu = n.lz4_get_decompress_size_async()?;
    check(cu(
        dev_compressed_ptrs,
        dev_compressed_bytes,
        dev_uncompressed_bytes_out,
        batch_size,
        stream,
    ))
}

/// Same as [`lz4_get_decompress_size_async`] but for Snappy.
///
/// # Safety
///
/// Same as above.
pub unsafe fn snappy_get_decompress_size_async(
    dev_compressed_ptrs: *const *const c_void,
    dev_compressed_bytes: *const usize,
    dev_uncompressed_bytes_out: *mut usize,
    batch_size: usize,
    stream: *mut c_void,
) -> Result<()> {
    let n = nvcomp()?;
    let cu = n.snappy_get_decompress_size_async()?;
    check(cu(
        dev_compressed_ptrs,
        dev_compressed_bytes,
        dev_uncompressed_bytes_out,
        batch_size,
        stream,
    ))
}

/// Same as [`lz4_get_decompress_size_async`] but for Zstd.
///
/// # Safety
///
/// Same as above.
pub unsafe fn zstd_get_decompress_size_async(
    dev_compressed_ptrs: *const *const c_void,
    dev_compressed_bytes: *const usize,
    dev_uncompressed_bytes_out: *mut usize,
    batch_size: usize,
    stream: *mut c_void,
) -> Result<()> {
    let n = nvcomp()?;
    let cu = n.zstd_get_decompress_size_async()?;
    check(cu(
        dev_compressed_ptrs,
        dev_compressed_bytes,
        dev_uncompressed_bytes_out,
        batch_size,
        stream,
    ))
}
