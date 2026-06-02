# baracuda-nvshmem

Safe Rust wrappers for the **NVIDIA NVSHMEM host API** — the OpenSHMEM
symmetric-heap, one-sided GPU communication model.

NVSHMEM is the fine-grained, one-sided complement to [`baracuda-nccl`]'s
collectives: every PE (typically one GPU) allocates from a *symmetric heap*
at a shared virtual address, and any PE can `put` / `get` directly into
another PE's heap. The two crates coexist — collectives via NCCL, one-sided
RDMA via NVSHMEM — and a single program may use both.

```rust,no_run
use baracuda_nvshmem::Context;

let ctx = Context::init()?;                 // environment-bootstrapped
let me = ctx.my_pe();
let world = ctx.n_pes();

let local  = ctx.malloc::<f32>(1024)?;      // collective symmetric alloc
let remote = ctx.malloc::<f32>(1024)?;

// one-sided: push our `local` into PE (me+1)'s `remote`
let peer = (me + 1) % world;
ctx.put(&remote, &local, 1024, peer)?;
ctx.barrier_all()?;                         // remote completion + sync
# Ok::<(), baracuda_nvshmem::Error>(())
```

## Surface (Tier 1)

- `Context` — init / finalize, cached `my_pe` / `n_pes`, `version`, and the
  `barrier_all` / `sync_all` / `quiet` / `fence` ordering primitives (plus
  `*_on_stream` variants).
- `Team` — `split_strided`, `my_pe`, `n_pes`, `translate_pe`, `destroy`,
  with predefined `Team::WORLD` / `Team::SHARED`.
- `SymmetricBuffer<T>` — typed symmetric-heap allocation (RAII free).
- Host-initiated RMA — `put` / `get` and their stream-ordered forms.
- `UniqueId` — the unique-id bootstrap seed (see the crate docs for the
  version-specific init wiring).

## Not covered

The **device-side** API (`nvshmem_int_p` / `nvshmem_putmem_nbi` etc. called
from inside a CUDA kernel) requires linking `libnvshmem_device.a` into the
consumer's kernel binary and cannot be a lazily-loaded host symbol. Consumers
that need it write their own `.cu` against the NVSHMEM headers.

## Availability

NVSHMEM is Linux-only in practice and requires compute capability sm_70+
(every baracuda-supported GPU qualifies). On hosts without the runtime,
`Context::init` returns `LoaderError::LibraryNotFound` so callers can fall
back to single-process execution. NVSHMEM is **not** bundled — it is loaded
at runtime from the user's installation.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nccl`]: https://docs.rs/baracuda-nccl
