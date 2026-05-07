# baracuda-cufile

Safe Rust wrappers for **NVIDIA cuFile** — the GPUDirect Storage
user-space library. Direct DMA between NVMe / network storage and GPU
memory, no host-RAM bounce buffer.

## Coverage

Comprehensive across the cuFile API:

- **Driver lifecycle**: `Driver::open`, `Driver::close`,
  `Driver::properties`, `Driver::set_poll_mode`.
- **File-handle register**: typed RAII `FileHandle` that registers with
  cuFile on construction and unregisters on drop.
- **Buffer register**: typed `BufferRegistration` for pre-registering
  device or pinned host buffers (avoids per-IO registration overhead).
- **Sync read / write**: `read`, `write`.
- **Async read / write** on a stream: `read_async`, `write_async`.
- **BatchIO**: `BatchHandle::setup` / `submit` / `poll` / `cancel` /
  `destroy`. Lets you submit dozens of IOs in one syscall, poll for
  completion, and overlap with kernel execution.
- **Limits**: configurable direct-I/O size, cache size, pinned-memory
  size — all via `Driver::set_*`.
- **Error reporting**: typed `OpStatus` with `error_string` helper.

```rust,no_run
use baracuda_cufile::{Driver, FileHandle, OpenFlags};
use baracuda_driver::{Context, Device, DeviceBuffer};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let _driver = Driver::open()?; // RAII: closes on drop
let ctx = Context::new(&Device::get(0)?)?;
let mut buf: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 4 << 20)?;

let fh = FileHandle::open("dataset.bin", OpenFlags::ReadOnly)?;
let bytes_read = fh.read(&mut buf, 0)?;
# Ok(()) }
```

## Platform support

cuFile / GPUDirect Storage is **Linux-only**. The loader fails fast
with a clear error on Windows / macOS.

Pairs with [`baracuda-cufile-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cufile-sys`]: https://docs.rs/baracuda-cufile-sys
