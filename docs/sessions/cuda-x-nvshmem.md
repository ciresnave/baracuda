# Session prompt — Add `baracuda-nvshmem` wrapper for NVSHMEM

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
Per the Phase 65 CUDA-X audit, NVSHMEM is a high-ROI addition that
enables multi-GPU patterns NCCL can't do (one-sided RDMA, symmetric
heap, fine-grained device-side put/get). Other library-addition
sessions may be running in parallel.

## Context

[NVSHMEM](https://developer.nvidia.com/nvshmem) is NVIDIA's
implementation of the OpenSHMEM symmetric-heap programming model on
GPUs. Unlike NCCL (collectives only), NVSHMEM provides:

- **Symmetric heap** — each PE (processing element, typically one GPU)
  has an allocation at the same virtual address that other PEs can
  read/write directly.
- **One-sided put/get** — `nvshmem_putmem(dst, src, nbytes, target_pe)`
  writes to `target_pe`'s heap at `dst` without involving target's
  CPU/GPU.
- **Device-side calls** — from within a CUDA kernel, threads can call
  `nvshmem_int_p(addr, val, target_pe)` to RDMA-write a single int to
  a remote GPU.
- **MoE expert-parallel substrate** — DeepEP, the standard MoE
  all-to-all library, is built on NVSHMEM.

baracuda has `baracuda-nccl` for collectives. NVSHMEM is the natural
sibling for fine-grained one-sided comms.

## Scope

**Crates to create:**

1. `crates/baracuda-nvshmem-sys/` — `extern "C"` FFI declarations over
   `libnvshmem_host.so`. NVSHMEM has both host-side init/finalize and
   device-side per-kernel operations; the device-side ones are
   `__device__` functions that must be called from inside CUDA
   kernels.
2. `crates/baracuda-nvshmem/` — safe wrapper. Mirror the
   `baracuda-nccl` design: opaque `NvshmemContext` handle (representing
   the SHMEM team) + typed put/get/atomic methods.

## Reference patterns

- `crates/baracuda-nccl/` — similar multi-GPU comms wrapper. Mirror
  the structure (lazy lib loading via `libloading`, no bindgen, no
  link-time NCCL dependency).
- NVSHMEM doesn't have the same "no NCCL on dev box" convenience —
  it requires NVSHMEM runtime to be installed for actual operation.
  But the build can succeed without it (only runtime initialization
  fails).

## Linking

- Host side: `libnvshmem_host.so` (Linux) / `nvshmem_host.dll`
  (Windows, if NVIDIA ships one). Use `libloading` for lazy resolution
  (matches baracuda-nccl pattern); no `find_library` in build.rs.
- Device side: NVSHMEM provides `libnvshmem_device.a` that has to be
  linked into the user kernel binary. This is awkward for baracuda's
  C-ABI surface — likely the right pattern is for `baracuda-nvshmem`
  to expose host-side API only (init, finalize, team creation,
  collective sync barriers), and let consumers who want device-side
  NVSHMEM call write their own `.cu` file that includes NVSHMEM
  headers + links `libnvshmem_device.a`.

## Tier 1 deliverables (this session)

1. Host-side wrapper covering:
   - `nvshmem_init` / `nvshmem_init_attr` / `nvshmem_finalize`
   - `nvshmem_my_pe` / `nvshmem_n_pes`
   - `nvshmem_team_create` / `nvshmem_team_destroy`
   - `nvshmem_barrier_all` / `nvshmem_quiet` / `nvshmem_fence`
   - Symmetric heap `nvshmem_malloc` / `nvshmem_free`
2. Cargo feature `nvshmem` on both `-sys` and `baracuda-kernels` (off
   by default).
3. Smoke tests requiring NVSHMEM runtime (mark `#[ignore]`).

## Tier 2 deferrable (next session)

- Device-side wrappers (one-sided put/get from inside kernels). This
  requires a separate `.cu` shim crate that links against
  `libnvshmem_device.a`.
- DeepEP integration (MoE all-to-all on top of NVSHMEM).
- Ring Attention via NVSHMEM (cleaner than the NCCL ring rotation
  Phase 56 does today; future MoE expert-parallel candidate).

## Coordination with `baracuda-nccl`

These crates SHOULD coexist. NCCL = collectives, NVSHMEM =
fine-grained. No code overlap. A future MoE Plan might use both.

## Out of scope

- Don't try to make NVSHMEM and NCCL share a single context — they
  don't and aren't supposed to.
- Don't add HOST-side bindings for the OpenSHMEM standard ops baracuda
  has no use for (atomic-fetch-and-add, lock-related ops, etc.) —
  add when needed.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase69-nvshmem`
- No version bump, no publish.
- Commit on branch + push + stop.

## Stop conditions

- If NVSHMEM's redistribution license requires bundling: stop, ask
  Eric. Most distros don't redistribute — runtime must be installed
  by user. baracuda-nccl handles this with lazy lib loading.
- If NVSHMEM's required compute capability is higher than baracuda's
  target (sm_80+): document the minimum + still ship; just feature-gate.
- If the entire `baracuda-nvshmem` crate pair already exists: stop, report.

## Memory file

After completion, write `project_phase69_complete.md`.
