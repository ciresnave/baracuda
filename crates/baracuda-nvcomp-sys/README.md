# baracuda-nvcomp-sys

Raw FFI bindings + dynamic loader for **NVIDIA nvCOMP** — GPU-accelerated
compression / decompression (LZ4, Snappy, Zstd, GDeflate, Gzip, Deflate,
Bitcomp, ANS, Cascaded).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libnvcomp.so` / `nvcomp64_*.dll`.

**Most users want [`baracuda-nvcomp`]** — that crate exposes typed
codec selection, async compress / decompress on a stream, temp-size
and chunk-size queries, and the CRC32 helpers.

## What's exposed

- All codec namespaces: LZ4, Snappy, Zstd, GDeflate, Gzip, Deflate,
  Bitcomp, ANS, Cascaded.
- For each codec: `compress_async`, `decompress_async`, temp-size
  query, max-chunk-size query, alignment query.
- Status helpers, property queries, CRC32.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvcomp`]: https://docs.rs/baracuda-nvcomp
