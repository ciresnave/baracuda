# baracuda-nvcomp

Safe Rust wrappers for **NVIDIA nvCOMP** — GPU-accelerated compression
and decompression.

## Coverage

All shipped codecs, with stream-async compress / decompress, temp-size
queries, and chunk-size queries:

- **LZ4** (low-latency, moderate ratio)
- **Snappy**
- **Zstd**
- **GDeflate** (NVIDIA's GPU-friendly Deflate variant)
- **Gzip** / **Deflate** (standard reference)
- **Bitcomp** (lossless, integer-friendly)
- **ANS** (asymmetric numeral systems)
- **Cascaded** (multi-pass: bit-pack → delta → run-length)
- **CRC32** (for end-to-end integrity, not compression)

```rust,no_run
use baracuda_nvcomp::lz4::{Lz4Manager, Lz4Options};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let stream = Stream::new(&ctx)?;

let manager = Lz4Manager::new(Lz4Options::default(), &stream)?;
let input: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 1 << 20)?;
let comp_size = manager.compressed_size(input.len())?;
let mut compressed: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, comp_size)?;
let actual = manager.compress(&input, &mut compressed, &stream)?;
# Ok(()) }
```

## Status / property queries

Every codec exposes `status_string`, version, and feature properties so
you can detect which codecs the installed nvCOMP build supports without
a runtime crash.

Pairs with [`baracuda-nvcomp-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvcomp-sys`]: https://docs.rs/baracuda-nvcomp-sys
