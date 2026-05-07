# baracuda-cufile-sys

Raw FFI bindings + dynamic loader for **NVIDIA cuFile** — the GPUDirect
Storage user-space library, enabling direct DMA between NVMe / network
storage and GPU memory without bouncing through host RAM.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcufile.so`.

**Most users want [`baracuda-cufile`]** — that crate exposes typed
driver-lifecycle / file-handle / buffer-registration / read-write /
batch-IO APIs in idiomatic Rust.

## Platform support

cuFile / GPUDirect Storage is **Linux-only**. The loader fails fast
with a clear error on Windows / macOS.

## What's exposed

- Driver lifecycle: `cuFileDriverOpen`, `cuFileDriverClose`,
  `cuFileDriverGetProperties`, `cuFileDriverSetPollMode` etc.
- File-handle register / deregister.
- Buffer register / deregister (for both pinned and managed memory).
- Sync read / write: `cuFileRead`, `cuFileWrite`.
- Async read / write on a stream.
- BatchIO: setup, submit, poll, cancel, destroy.
- Configurable direct-I/O / cache / pinned-mem limits.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cufile`]: https://docs.rs/baracuda-cufile
