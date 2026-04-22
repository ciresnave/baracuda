# cuFile guide

cuFile (GPUDirect Storage) lets CUDA read/write files directly into
device memory, bypassing the host CPU for I/O. baracuda wraps the full
host-side surface: driver lifecycle, handle registration, buffer
registration, sync + async read/write, and batched I/O.

## Platform support

**Linux-only.** cuFile's kernel module and NVMe path have no
Windows/macOS analogue. baracuda's loader returns
`LoaderError::UnsupportedPlatform { platform: "windows" }` etc. on
non-Linux; every safe API forwards that as `Error::Loader(...)`.

Requires a GDS-capable filesystem (ext4 / XFS on NVMe with the NVIDIA
GDS kernel module). `/tmp` usually isn't GDS-capable; use `/mnt/nvme`
or similar.

## Concepts

- [`Driver`](../../crates/baracuda-cufile/src/lib.rs) — RAII over
  `cuFileDriverOpen` / `cuFileDriverClose`. Open once per process.
- [`FileHandle`] — wrapped file descriptor. `FileHandle::register(fd)`
  corresponds to `cuFileHandleRegister`.
- [`BufRegistration`] — optional but strongly recommended. Registering
  a device buffer lets cuFile hit the fastest DMA path.
- [`StreamRegistration`] — for async I/O, you register each stream
  once before queueing `read_async` / `write_async` on it.
- [`BatchIO`] — submit many I/O entries in one call, poll for
  completion with [`BatchIO::poll`].

## Flow: sync write → read round-trip

```rust
use std::os::fd::AsRawFd;
use baracuda_cufile::{Driver, FileHandle, BufRegistration};
use baracuda_runtime::{Device, DeviceBuffer};

let _driver = Driver::open()?;   // keep alive until all I/O is done
Device::from_ordinal(0).set_current()?;

let host: Vec<u32> = (0..4096).collect();
let d_src: DeviceBuffer<u32> = DeviceBuffer::from_slice(&host)?;
let d_dst: DeviceBuffer<u32> = DeviceBuffer::new(4096)?;

let file = std::fs::File::options()
    .read(true).write(true).create(true).truncate(true)
    .open("/mnt/nvme/data.bin")?;
let fh = unsafe { FileHandle::register(file.as_raw_fd()) }?;

// Optional: register the device buffers for faster DMA.
let bytes = 4096 * 4;
let _buf_reg = unsafe { BufRegistration::register(d_src.as_raw(), bytes, 0) }?;

unsafe { fh.write(d_src.as_raw(), bytes, /*file_off=*/ 0, /*buf_off=*/ 0) }?;
unsafe { fh.read (d_dst.as_raw(), bytes, 0, 0) }?;

let mut back = vec![0u32; 4096];
d_dst.copy_to_host(&mut back)?;
assert_eq!(host, back);
```

## Flow: async I/O (stream-ordered)

```rust
use baracuda_cufile::{StreamRegistration, FileHandle};

let _sreg = unsafe { StreamRegistration::register(stream.as_raw(), 0) }?;

let mut size = 4096u64 * 4;
let mut file_off = 0i64;
let mut buf_off = 0i64;
let mut bytes_read: isize = 0;

unsafe {
    fh.read_async(
        d_dst.as_raw(),
        &mut size as *mut usize as *mut usize,  // sized-as-usize field
        &mut file_off,
        &mut buf_off,
        &mut bytes_read,
        stream.as_raw(),
    )?;
}
// The read completes when the stream reaches this op.
stream.synchronize()?;
```

Async I/O lets you interleave storage and compute on a stream:
`h2d → read_from_file → kernel → write_to_file → d2h` with no host
synchronization between steps.

## Flow: batched I/O

Submit many reads/writes in one call and reap them together:

```rust
use baracuda_cufile::{BatchIO, CUfileIOParams_t, CUfileIOEvents_t, CUfileOpcode};

let batch = BatchIO::new(64)?;
let mut params: Vec<CUfileIOParams_t> = (0..10)
    .map(|i| CUfileIOParams_t {
        fh: fh.as_raw(),
        opcode: CUfileOpcode::READ,
        dev_ptr_base: d_dst.as_raw(),
        file_offset: i as i64 * 4096,
        dev_ptr_offset: i as i64 * 4096,
        size: 4096,
        ..Default::default()
    })
    .collect();

unsafe { batch.submit(&mut params, 0) }?;

let mut events = vec![CUfileIOEvents_t::default(); 10];
let completed = unsafe { batch.poll(10, &mut events) }?;
assert_eq!(completed, 10);
```

## Common pitfalls

1. **Filesystem not GDS-capable.** `FileHandle::register` returns
   `CU_FILE_IO_NOT_SUPPORTED`. Move the file to an NVMe + ext4/XFS mount.
2. **Driver not opened.** All I/O returns `DRIVER_NOT_INITIALIZED` if
   you forgot `Driver::open()` first. `Driver` is a guard type — keep
   it in scope.
3. **`Driver` drop ordering.** On process teardown, drop `FileHandle`
   and `BufRegistration` *before* `Driver`. With RAII this happens
   naturally if they're local, but be careful with `lazy_static!` /
   `OnceCell` globals.
4. **Pointers must be device-memory.** cuFile does DMA; passing a
   host-only pointer fails (or writes garbage to the file).
