# Memory patterns

Which allocation strategy to pick depends on the access pattern. baracuda
wraps all the common ones; this page is a cheat-sheet.

## Picking an allocator

| You need…                                  | Use                          |
| ------------------------------------------ | ---------------------------- |
| Plain device-only buffer                   | `DeviceBuffer<T>`            |
| Short-lived scratch, many allocations      | `DeviceBuffer::from_pool(&pool, len)` |
| Zero-copy host↔device (small, hot)         | `UnifiedBuffer<T>`           |
| Fast H2D / D2H throughput                  | `PinnedVec<T>` + stream API  |
| Share memory between processes             | `IpcMemory` (driver API)     |
| Import from Vulkan / D3D12 / external      | `ExternalMemory`             |

Each of these is documented on its type; this page shows the common flows.

## Device buffer

The workhorse.

```rust
use baracuda::driver::{Context, DeviceBuffer};

let buf = DeviceBuffer::<f32>::zeros(&ctx, 1_000_000)?;           // allocated + zeroed
let buf = DeviceBuffer::from_slice(&ctx, &host_data)?;            // allocated + copied
let mut buf = DeviceBuffer::<f32>::new(&ctx, 1_000_000)?;         // allocated, contents undefined

// Synchronous transfer:
buf.copy_from_host(&host_data)?;
buf.copy_to_host(&mut host_out)?;

// Async transfer (on a stream):
buf.copy_from_host_async(&host_data, &stream)?;
buf.copy_to_host_async(&mut host_out, &stream)?;
```

## Stream-ordered memory pools

For workloads that allocate and free many buffers per frame, the pool
allocator amortizes the overhead:

```rust
use baracuda::driver::MemoryPool;

let pool = MemoryPool::for_device(&device)?;
pool.set_release_threshold(256 * 1024 * 1024)?;  // keep 256 MiB warm

let stream = Stream::new(&ctx)?;
let buf = DeviceBuffer::from_pool(&pool, 1_000_000, &stream)?;
// ... use buf ...
drop(buf);  // memory returns to the pool; reuse is instantaneous
```

Pool allocations are *stream-ordered* — the driver tracks which stream owns
the allocation and only reuses freed memory after downstream work on that
stream finishes.

## Unified memory

CUDA unified memory is simpler than pinned + device-buffer pairs at the cost
of some throughput. The driver page-migrates between host and device on
access.

```rust
use baracuda::driver::UnifiedBuffer;

let mut u: UnifiedBuffer<f32> = UnifiedBuffer::zeros(&ctx, 1_000_000)?;

// Direct host access: no explicit H2D needed.
u.as_mut_slice()[0] = 3.14;

// Pass to a kernel:
kernel.launch()...arg(&u)...launch()?;

// Kernel result visible to host:
stream.synchronize()?;
let first = u.as_slice()[0];
```

Unified works best when the access pattern is coherent (host OR device uses
each page, not both simultaneously). Use advice hooks to help the driver
prefetch: `u.advise_read_mostly()`, `u.prefetch_async(&device, &stream)`.

## Pinned host memory

For maximum H2D / D2H bandwidth, pair a `PinnedVec<T>` with an async copy:

```rust
use baracuda::driver::PinnedVec;

let mut pinned: PinnedVec<f32> = PinnedVec::new(&ctx, 1_000_000)?;
pinned.as_mut_slice().fill(1.0);

d_buf.copy_from_host_async(&pinned, &stream)?;
stream.synchronize()?;
```

Pinned pages bypass the kernel's page cache and enable true DMA transfers.
Used with pageable `Vec<T>`, the driver has to stage the copy through a
pinned bounce buffer first.

## Shared memory between contexts (IPC)

`IpcMemory` lets another process (or another CUDA context in the same
process) open a handle to memory this one allocated:

```rust
use baracuda::driver::{DeviceBuffer, IpcMemory};

let buf = DeviceBuffer::<f32>::zeros(&ctx, 1_000_000)?;
let ipc = IpcMemory::export(&buf)?;
let handle_bytes = ipc.as_bytes();  // send these to the other process

// On the other side:
let imported = IpcMemory::import(handle_bytes, &other_ctx)?;
```

Linux only; Windows uses a different mechanism.

## External memory

Interop with Vulkan, D3D12, NvSci, or OpenGL — you import an external
allocation and treat it like a CUDA buffer:

```rust
use baracuda::driver::{ExternalMemory, ExternalMemoryHandleDesc};

let handle = ExternalMemoryHandleDesc::opaque_fd(fd, size);
let ext = ExternalMemory::import(&ctx, &handle)?;
let mapped = ext.map_buffer(0, size)?;  // returns a DeviceSlice<u8>
```

Same semantics for external semaphores (`ExternalSemaphore`).

## Debugging tips

- **"free-failed" warnings at Drop** — you've freed a buffer while a kernel
  is still referencing it. Sync the stream before dropping.
- **Pinned allocations > RAM / 4 failing** — most OSes cap how much
  pinned memory a process can hold. Tune `ulimit -l` on Linux.
- **Unified memory thrashing** — the driver is migrating pages back and
  forth. Check your access pattern, and use `prefetch_async` to hint the
  right GPU ahead of the access.
