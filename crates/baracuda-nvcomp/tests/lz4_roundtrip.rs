//! nvCOMP integration test — LZ4 batched round-trip.
//!
//! Compresses a small batch of known input buffers, decompresses, and
//! checks the result matches the original. Skips if nvCOMP isn't
//! installed.

use core::ffi::c_void;

use baracuda_nvcomp::{lz4, nvcompBatchedLZ4DecompressOpts_t, nvcompBatchedLZ4Opts_t};
use baracuda_runtime::{Device, DeviceBuffer, Stream};

#[test]
#[ignore = "requires nvCOMP installed + NVIDIA GPU"]
fn lz4_compress_decompress_roundtrip() {
    if baracuda_nvcomp::probe().is_err() {
        eprintln!("nvCOMP not installed — skipping");
        return;
    }

    Device::from_ordinal(0).set_current().unwrap();
    let stream = Stream::new().unwrap();
    let opts = nvcompBatchedLZ4Opts_t::default();

    // Build a batch of 4 chunks; use repeating data so LZ4 actually compresses.
    let chunk_size = 64 * 1024usize;
    let batch: Vec<Vec<u8>> = (0..4)
        .map(|i| {
            let mut v = vec![0u8; chunk_size];
            // Write base pattern.
            for (j, b) in v.iter_mut().enumerate() {
                *b = (((i + 1) as u64 * (j as u64 + 1)).wrapping_mul(0x9E37_79B1) & 0xFF) as u8;
            }
            // Overlay repeating segments every 128 bytes so LZ4 compresses well.
            let mut j = 128;
            while j < chunk_size {
                let end = (j + 64).min(chunk_size);
                for (offset, k) in (j..end).enumerate() {
                    v[k] = (i as u8).wrapping_add(offset as u8);
                }
                j += 128;
            }
            v
        })
        .collect();
    let batch_size = batch.len();

    // Allocate device-side input buffers + pointer/size tables.
    let d_inputs: Vec<DeviceBuffer<u8>> = batch
        .iter()
        .map(|chunk| DeviceBuffer::from_slice(chunk).unwrap())
        .collect();
    // Pointers are 8-byte values on 64-bit Windows/Linux; ship them as u64
    // buffers since DeviceBuffer requires a DeviceRepr element type.
    let input_ptrs_u64: Vec<u64> = d_inputs.iter().map(|b| b.as_raw() as u64).collect();
    let input_sizes: Vec<usize> = batch.iter().map(|b| b.len()).collect();

    let d_input_ptrs: DeviceBuffer<u64> = DeviceBuffer::from_slice(&input_ptrs_u64).unwrap();
    let d_input_sizes: DeviceBuffer<usize> = DeviceBuffer::from_slice(&input_sizes).unwrap();

    // Sizing.
    let max_total = batch_size * chunk_size;
    eprintln!("sizing: temp_bytes...");
    let temp_bytes = lz4::compress_get_temp_size(batch_size, chunk_size, opts, max_total).unwrap();
    eprintln!("  temp_bytes = {temp_bytes}");
    eprintln!("sizing: max_compressed...");
    let max_compressed = lz4::compress_get_max_output_chunk_size(chunk_size, opts).unwrap();
    eprintln!("  max_compressed = {max_compressed}");

    let d_temp: DeviceBuffer<u8> = DeviceBuffer::new(temp_bytes).unwrap();

    // Allocate compressed output buffers, one per chunk.
    let d_comp_bufs: Vec<DeviceBuffer<u8>> = (0..batch_size)
        .map(|_| DeviceBuffer::new(max_compressed).unwrap())
        .collect();
    let comp_ptrs_u64: Vec<u64> = d_comp_bufs.iter().map(|b| b.as_raw() as u64).collect();
    let d_comp_ptrs: DeviceBuffer<u64> = DeviceBuffer::from_slice(&comp_ptrs_u64).unwrap();
    let d_comp_sizes: DeviceBuffer<usize> = DeviceBuffer::new(batch_size).unwrap();

    // Compress (v5+ signature includes device_statuses array).
    let d_comp_statuses: DeviceBuffer<i32> = DeviceBuffer::new(batch_size).unwrap();
    unsafe {
        lz4::compress_async(
            d_input_ptrs.as_raw() as *const *const c_void,
            d_input_sizes.as_raw() as *const usize,
            chunk_size,
            batch_size,
            d_temp.as_raw(),
            temp_bytes,
            d_comp_ptrs.as_raw() as *const *mut c_void,
            d_comp_sizes.as_raw() as *mut usize,
            opts,
            d_comp_statuses.as_raw() as *mut baracuda_nvcomp_sys::nvcompStatus_t,
            stream.as_raw(),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    // Read back actual compressed sizes — just for sanity.
    let mut host_comp_sizes = vec![0usize; batch_size];
    d_comp_sizes.copy_to_host(&mut host_comp_sizes).unwrap();
    eprintln!("LZ4 compressed sizes: {host_comp_sizes:?} (uncompressed = {chunk_size})");

    // Decompress.
    let decomp_opts = nvcompBatchedLZ4DecompressOpts_t::default();
    let decomp_temp_bytes =
        lz4::decompress_get_temp_size(batch_size, chunk_size, decomp_opts, max_total).unwrap();
    let d_decomp_temp: DeviceBuffer<u8> = DeviceBuffer::new(decomp_temp_bytes).unwrap();

    let d_decomp_bufs: Vec<DeviceBuffer<u8>> = (0..batch_size)
        .map(|_| DeviceBuffer::new(chunk_size).unwrap())
        .collect();
    let decomp_ptrs_u64: Vec<u64> = d_decomp_bufs.iter().map(|b| b.as_raw() as u64).collect();
    let d_decomp_ptrs: DeviceBuffer<u64> = DeviceBuffer::from_slice(&decomp_ptrs_u64).unwrap();
    let d_decomp_actual_sizes: DeviceBuffer<usize> = DeviceBuffer::new(batch_size).unwrap();
    let d_decomp_statuses: DeviceBuffer<i32> = DeviceBuffer::new(batch_size).unwrap();
    let uncompressed_size_array: Vec<usize> = vec![chunk_size; batch_size];
    let d_uncompressed_sizes: DeviceBuffer<usize> =
        DeviceBuffer::from_slice(&uncompressed_size_array).unwrap();

    // Const-pointer array pointing at the same compressed buffers.
    let d_comp_const_ptrs: DeviceBuffer<u64> = DeviceBuffer::from_slice(&comp_ptrs_u64).unwrap();

    unsafe {
        lz4::decompress_async(
            d_comp_const_ptrs.as_raw() as *const *const c_void,
            d_comp_sizes.as_raw() as *const usize,
            d_uncompressed_sizes.as_raw() as *const usize,
            d_decomp_actual_sizes.as_raw() as *mut usize,
            batch_size,
            d_decomp_temp.as_raw(),
            decomp_temp_bytes,
            d_decomp_ptrs.as_raw() as *const *mut c_void,
            decomp_opts,
            d_decomp_statuses.as_raw() as *mut baracuda_nvcomp_sys::nvcompStatus_t,
            stream.as_raw(),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();

    // Compare each chunk to the original.
    for i in 0..batch_size {
        let mut host = vec![0u8; chunk_size];
        d_decomp_bufs[i].copy_to_host(&mut host).unwrap();
        assert_eq!(host, batch[i], "chunk {i} round-trip mismatch");
    }
    eprintln!("nvCOMP LZ4 round-trip OK on {batch_size} chunks of {chunk_size}B each");
}
