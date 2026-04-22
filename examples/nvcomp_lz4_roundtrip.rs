//! nvCOMP LZ4 batched round-trip demo.
//!
//! Compresses a batch of 64-KiB chunks with mildly-redundant content,
//! decompresses them back, and reports compression ratio + timing.

use core::ffi::c_void;

use baracuda_nvcomp::{lz4, nvcompBatchedLZ4DecompressOpts_t, nvcompBatchedLZ4Opts_t};
use baracuda_runtime::{Device, DeviceBuffer, Stream};

fn main() {
    if let Err(e) = baracuda_nvcomp::probe() {
        eprintln!("nvCOMP not available: {e:?}");
        std::process::exit(1);
    }

    Device::from_ordinal(0).set_current().unwrap();
    let stream = Stream::new().unwrap();

    // Build a batch of compressible 64-KiB chunks.
    let chunk = 64 * 1024;
    let batch: Vec<Vec<u8>> = (0..8)
        .map(|k| {
            let mut v = vec![0u8; chunk];
            for (i, b) in v.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(0x9E37_79B1) + k * 7) & 0xFF) as u8;
            }
            // Sprinkle in 128-byte repeats so LZ4 has something to find.
            let mut off = 256;
            while off + 64 < chunk {
                for (i, src) in (off - 128..off - 64).enumerate() {
                    v[off + i] = v[src];
                }
                off += 256;
            }
            v
        })
        .collect();
    let batch_size = batch.len();
    let total_in = batch.iter().map(|c| c.len()).sum::<usize>();
    let max_total = batch_size * chunk;

    println!(
        "nvCOMP LZ4: batch of {} × {} B = {} B",
        batch_size, chunk, total_in
    );

    // Upload chunks + build pointer/size tables.
    let d_inputs: Vec<DeviceBuffer<u8>> = batch
        .iter()
        .map(|c| DeviceBuffer::from_slice(c).unwrap())
        .collect();
    let input_ptrs_u64: Vec<u64> = d_inputs.iter().map(|b| b.as_raw() as u64).collect();
    let input_sizes: Vec<usize> = batch.iter().map(|c| c.len()).collect();
    let d_in_ptrs: DeviceBuffer<u64> = DeviceBuffer::from_slice(&input_ptrs_u64).unwrap();
    let d_in_sizes: DeviceBuffer<usize> = DeviceBuffer::from_slice(&input_sizes).unwrap();

    let copts = nvcompBatchedLZ4Opts_t::default();
    let dopts = nvcompBatchedLZ4DecompressOpts_t::default();

    let temp_bytes = lz4::compress_get_temp_size(batch_size, chunk, copts, max_total).unwrap();
    let max_out = lz4::compress_get_max_output_chunk_size(chunk, copts).unwrap();
    let d_temp: DeviceBuffer<u8> = DeviceBuffer::new(temp_bytes).unwrap();
    let d_out_bufs: Vec<DeviceBuffer<u8>> = (0..batch_size)
        .map(|_| DeviceBuffer::new(max_out).unwrap())
        .collect();
    let out_ptrs_u64: Vec<u64> = d_out_bufs.iter().map(|b| b.as_raw() as u64).collect();
    let d_out_ptrs: DeviceBuffer<u64> = DeviceBuffer::from_slice(&out_ptrs_u64).unwrap();
    let d_out_sizes: DeviceBuffer<usize> = DeviceBuffer::new(batch_size).unwrap();
    let d_c_stat: DeviceBuffer<i32> = DeviceBuffer::new(batch_size).unwrap();

    let t0 = std::time::Instant::now();
    unsafe {
        lz4::compress_async(
            d_in_ptrs.as_raw() as *const *const c_void,
            d_in_sizes.as_raw() as *const usize,
            chunk,
            batch_size,
            d_temp.as_raw(),
            temp_bytes,
            d_out_ptrs.as_raw() as *const *mut c_void,
            d_out_sizes.as_raw() as *mut usize,
            copts,
            d_c_stat.as_raw() as *mut baracuda_nvcomp_sys::nvcompStatus_t,
            stream.as_raw(),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let comp_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut comp_sizes = vec![0usize; batch_size];
    d_out_sizes.copy_to_host(&mut comp_sizes).unwrap();
    let total_out: usize = comp_sizes.iter().sum();

    // Decompress back.
    let decomp_temp = lz4::decompress_get_temp_size(batch_size, chunk, dopts, max_total).unwrap();
    let d_decomp_temp: DeviceBuffer<u8> = DeviceBuffer::new(decomp_temp).unwrap();
    let d_decomp_bufs: Vec<DeviceBuffer<u8>> = (0..batch_size)
        .map(|_| DeviceBuffer::new(chunk).unwrap())
        .collect();
    let dec_ptrs_u64: Vec<u64> = d_decomp_bufs.iter().map(|b| b.as_raw() as u64).collect();
    let d_decomp_ptrs: DeviceBuffer<u64> = DeviceBuffer::from_slice(&dec_ptrs_u64).unwrap();
    let d_actual_sizes: DeviceBuffer<usize> = DeviceBuffer::new(batch_size).unwrap();
    let d_uncompressed_sizes: DeviceBuffer<usize> =
        DeviceBuffer::from_slice(&vec![chunk; batch_size]).unwrap();
    let d_comp_const_ptrs: DeviceBuffer<u64> = DeviceBuffer::from_slice(&out_ptrs_u64).unwrap();
    let d_d_stat: DeviceBuffer<i32> = DeviceBuffer::new(batch_size).unwrap();

    let t1 = std::time::Instant::now();
    unsafe {
        lz4::decompress_async(
            d_comp_const_ptrs.as_raw() as *const *const c_void,
            d_out_sizes.as_raw() as *const usize,
            d_uncompressed_sizes.as_raw() as *const usize,
            d_actual_sizes.as_raw() as *mut usize,
            batch_size,
            d_decomp_temp.as_raw(),
            decomp_temp,
            d_decomp_ptrs.as_raw() as *const *mut c_void,
            dopts,
            d_d_stat.as_raw() as *mut baracuda_nvcomp_sys::nvcompStatus_t,
            stream.as_raw(),
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let decomp_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Verify.
    for (i, d_out) in d_decomp_bufs.iter().enumerate() {
        let mut host = vec![0u8; chunk];
        d_out.copy_to_host(&mut host).unwrap();
        assert_eq!(host, batch[i], "chunk {i} mismatch");
    }

    let ratio = total_in as f64 / total_out as f64;
    let comp_gbps = (total_in as f64 / 1e9) / (comp_ms / 1000.0);
    let decomp_gbps = (total_in as f64 / 1e9) / (decomp_ms / 1000.0);

    println!(
        "  compressed: {} → {} bytes ({ratio:.2}× ratio)",
        total_in, total_out
    );
    println!("  compress:   {:.2} ms   ({:.2} GB/s)", comp_ms, comp_gbps);
    println!(
        "  decompress: {:.2} ms   ({:.2} GB/s)",
        decomp_ms, decomp_gbps
    );
    println!("OK — round-trip bit-exact");
}
