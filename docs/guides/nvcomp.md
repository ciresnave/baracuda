# nvCOMP guide

nvCOMP is NVIDIA's GPU compression library. baracuda wraps the full
**batched** host-side API for nvCOMP 5.x: LZ4, Snappy, Zstd, GDeflate,
Deflate, Bitcomp, ANS, Cascaded.

## Installation

Windows: `C:\Program Files\NVIDIA nvCOMP\v<ver>\bin\<cuda-major>\nvcomp64_5.dll`.
Linux: `libnvcomp.so.5` in standard library paths.

## ABI note: nvCOMP 5 changed the surface

If you're upgrading from nvCOMP 3/4 samples, the v5 API is different:

1. Compress/decompress **options are split** — `LZ4CompressOpts_t` ≠
   `LZ4DecompressOpts_t`.
2. `GetTempSize` → `GetTempSizeAsync` and `GetTempSizeSync` (we wrap
   `Async` as the host-sizing call since it doesn't touch the device).
3. `GetTempSize*` functions gained a `max_total_uncompressed_bytes`
   hint parameter.
4. `CompressAsync` and `DecompressAsync` gained a `device_statuses`
   array for per-chunk error reporting.

baracuda's `codec_mod!` macro generates the uniform v5 surface
per-codec.

## Flow: LZ4 batched round-trip

```rust
use baracuda_nvcomp::{lz4, nvcompBatchedLZ4Opts_t, nvcompBatchedLZ4DecompressOpts_t};

let opts = nvcompBatchedLZ4Opts_t::default();
let batch_size = 4;
let max_chunk = 64 * 1024;
let max_total = batch_size * max_chunk;

// Scratch sizing.
let temp_bytes = lz4::compress_get_temp_size(batch_size, max_chunk, opts, max_total)?;
let max_out   = lz4::compress_get_max_output_chunk_size(max_chunk, opts)?;
let decomp_temp = lz4::decompress_get_temp_size(batch_size, max_chunk,
    nvcompBatchedLZ4DecompressOpts_t::default(), max_total)?;

// ... allocate device pointer/size arrays, temp buffer, per-chunk output buffers ...

unsafe {
    lz4::compress_async(
        dev_uncompressed_ptrs, dev_uncompressed_sizes, max_chunk, batch_size,
        dev_temp_ptr, temp_bytes,
        dev_compressed_ptrs, dev_compressed_sizes,
        opts, dev_statuses, stream,
    )?;
}
```

## Device pointer / size arrays

Every batched call wants device-resident arrays of `batch_size`
pointers and matching arrays of `batch_size` sizes. The usual pattern:

1. Allocate each chunk's input buffer on-device via
   [`baracuda_runtime::DeviceBuffer::from_slice`].
2. Collect the device pointers into a host-side `Vec<u64>` (cast to
   `u64` since `DeviceBuffer<T>` requires `T: DeviceRepr` and raw
   pointers aren't `DeviceRepr`).
3. Upload the `Vec<u64>` itself to a `DeviceBuffer<u64>`, then cast
   its raw pointer to `*const *const c_void` for the call.

See [`crates/baracuda-nvcomp/tests/lz4_roundtrip.rs`](../../crates/baracuda-nvcomp/tests/lz4_roundtrip.rs)
for the full boilerplate.

## Codec selection

| Codec     | Ratio        | Throughput (dec) | Notes                                 |
| --------- | ------------ | ---------------- | ------------------------------------- |
| LZ4       | low–moderate | very high        | Default choice; lowest CPU overhead   |
| Snappy    | low–moderate | high             | OpenCV-compatible                     |
| Zstd      | high         | moderate         | Best general-purpose ratio            |
| GDeflate  | high         | very high        | GPU-native zlib variant; random-access |
| Deflate   | high         | moderate         | Strict zlib/raw deflate compatibility |
| Bitcomp   | varies       | very high        | Columnar / sorted numerical data      |
| ANS       | high         | moderate         | Range-coded entropy                   |
| Cascaded  | very high    | moderate         | RLE + delta + bit-packing for ints    |

## Alignment hints (nvCOMP 5+)

```rust
let align = baracuda_nvcomp::lz4_compress_alignment(opts)?;
// align.input, align.output, align.temp — minimum required alignments
```

Providing over-aligned buffers (16 or 32 bytes) often boosts
throughput; under-aligned buffers cause `ERROR_INVALID_VALUE`.
