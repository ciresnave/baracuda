# baracuda-nvshmem-sys

Raw FFI bindings + dynamic loader for the **NVIDIA NVSHMEM host library**
(`libnvshmem_host.so`) — the OpenSHMEM symmetric-heap, one-sided RDMA
programming model on GPUs.

Where [`baracuda-nccl-sys`] binds *collective* communication, NVSHMEM adds
**one-sided** put/get: every PE allocates from a symmetric heap at a shared
virtual address, and any PE can write into another PE's heap directly.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading); there
is **no** link-time dependency on NVSHMEM, so the crate compiles on hosts
where NVSHMEM is not installed. [`nvshmem()`](crate::nvshmem) returns
`LoaderError::LibraryNotFound` at runtime in that case.

**Most users want [`baracuda-nvshmem`]** — that crate exposes a safe
`Context` handle, `Team`, a `SymmetricBuffer` heap allocator, host-initiated
put/get, and the barrier/quiet/fence primitives.

## What's exposed (host side only)

- Init / finalize / query: `nvshmem_init`, `nvshmemx_init_attr`,
  `nvshmemx_init_status`, `nvshmem_finalize`, `nvshmem_my_pe`,
  `nvshmem_n_pes`, `nvshmem_info_get_version`.
- Unique-id bootstrap: `nvshmemx_get_uniqueid`,
  `nvshmemx_set_attr_uniqueid_args`.
- Teams: `nvshmem_team_split_strided`, `nvshmem_team_destroy`,
  `nvshmem_team_my_pe`, `nvshmem_team_n_pes`, `nvshmem_team_translate_pe`.
- Symmetric heap: `nvshmem_malloc`, `nvshmem_free`, `nvshmem_align`,
  `nvshmem_calloc`.
- Host-initiated RMA: `nvshmem_putmem`, `nvshmem_getmem`,
  `nvshmemx_putmem_on_stream`, `nvshmemx_getmem_on_stream`.
- Ordering / sync: `nvshmem_barrier_all`, `nvshmemx_barrier_all_on_stream`,
  `nvshmem_sync_all`, `nvshmem_quiet`, `nvshmem_fence`.

## What's *not* exposed

The **device-side** API (the `__device__` `nvshmem_int_p` /
`nvshmem_putmem_nbi` etc. callable from inside a CUDA kernel) lives in the
static archive `libnvshmem_device.a`, which must be linked into the
consumer's own kernel binary. That belongs in a future device-shim crate, not
in this lazily-loaded host wrapper.

## Platform support

NVSHMEM is **Linux-only** in practice and requires compute capability
**sm_70+** (every baracuda-supported GPU qualifies, since baracuda targets
sm_80+). It also requires the NVSHMEM runtime to be installed on the host for
*operation* — the build never needs it.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0. (NVSHMEM itself is **not** bundled — it is loaded at
runtime from the user's installation.)

[`baracuda-nvshmem`]: https://docs.rs/baracuda-nvshmem
[`baracuda-nccl-sys`]: https://docs.rs/baracuda-nccl-sys
