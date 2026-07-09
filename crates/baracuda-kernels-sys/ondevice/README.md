# On-device validation harnesses (baracuda-kernels-sys)

Manual `nvcc` harnesses (NOT wired into `cargo test` or `build.rs`) that launch the
bespoke `.cu` kernels on the GPU and check behavior only catchable on device. Each
harness `#include`s the kernel header by relative path and instantiates the launcher
macros locally, so it links standalone (not against the crate's static archive).

**Run (Windows):** from a Visual Studio dev shell so `nvcc` finds `cl.exe`
(`Enter-VsDevShell`), or an x64 Native Tools prompt.

---

## `write_slice_doff_validate.cu` — form-B `_doff` WriteSlice (device-resident dyn start)

Acceptance for the form-B `_doff` WriteSlice variant: the range-start of ONE axis
(`dyn_axis`) is read from a **device** pointer (`dyn_start_dev[0]`, a single i64) at
kernel entry, instead of being host-baked into the by-value `range_start`. This is a
**separate** kernel/launcher family (`write_slice_byte_doff_kernel<Blob>`,
`baracuda_kernels_write_slice_b{1,2,4,8}_doff_run` / `_doff_can_implement`) added
purely additively beside the existing `_run` family — the `_run` kernel's PTX and
every pre-existing symbol are untouched.

**Why.** Fuel's KV-cache decode wants CUDA-graph replay (`cuGraphLaunch` once/token
vs ~150 launches). The by-value `range_start` is marshaled into the captured graph
node's param space, so a host-baked seq position **freezes** at the captured token.
The `_doff` variant reads the position from device memory, which the host updates per
token via a fixed-address H2D memcpy that is **capture-tolerant** — form B survives
graph replay where the baked `_run` (form A) would freeze.

**Cells:**

- **A — stream-capture + replay (the gate).** Capture `{ cudaMemcpyAsync(dyn_start_dev
  ← pinned h_pos), the _doff launch }`; `cudaGraphInstantiate`; replay over a seq sweep
  `{0,1,7,S/2,S-1}`, mutating `*h_pos` each token. Assert (i) every head's slab lands
  at the **updated** `p` and (ii) no other row moves (a host mirror carries the
  built-up KV cache, so prior writes must be preserved).
- **A′ — node-param update path.** Same, but the per-token position is pushed via
  `cudaGraphExecMemcpyNodeSetParams` on the retrieved memcpy node (the exact mechanism
  Fuel's DecodeSession will use).
- **B — additive byte-identity.** `_doff` at a fixed device start vs the baked `_run`
  at the matched host start, whole-buffer `memcmp == 0`, across b1/b2/b4/b8, dyn edges
  `{0, S/2, S-1}`, seeded with the house probe classes (NaN payloads / ±Inf / subnormal
  / ±0 / extremes — a byte-level copy round-trips every pattern). The runtime witness
  that the `_run` path is unperturbed.
- **C — sanitizer/edge.** `p = S-1` (slab touches the buffer end) under
  `compute-sanitizer memcheck` + `racecheck`, via the `san` arg (small shapes). The
  in-bounds edge witness — note the kernel deliberately does **not** clamp the device
  start (clamping would perturb the index math vs `_run`); `cached_len + seq ≤ max_seq`
  is the caller's (Fuel/DecodeSession) contract.

**Coverage:** dtype `{b1,b2,b4,b8}` (b2/f16 = the KV default) × heads `{8}` × seq-edge
`{0, mid, S-1}` × mechanism `{stream-capture-replay, node-param-set, non-graph memcmp}`
× sanitizer `{memcheck, racecheck}`. Two layouts: rank-4 `[1,H,S,D]` with `dyn_axis=2`
(the CapturedRun consumer) and rank-3 `[S,H,D]` with `dyn_axis=0`.

**Build / run (from a VS dev shell):**

```sh
nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
     crates/baracuda-kernels-sys/ondevice/write_slice_doff_validate.cu \
     -o write_slice_doff_validate
./write_slice_doff_validate            # full sweep (Cells A / A' / B)
./write_slice_doff_validate san        # small shapes for the sanitizers
compute-sanitizer --tool memcheck   ./write_slice_doff_validate san
compute-sanitizer --tool racecheck  ./write_slice_doff_validate san
```

**`_run` byte-identity — static proof (risk #2).** Compile the pre-change (git HEAD)
`write_slice.cu` and the current one to PTX + object and diff:

- **Device PTX:** the entire `_run` device code (6 `.entry` — `write_slice_nibble_kernel`
  + `write_slice_byte_kernel<Blob1/2/4/8/16>`) is **byte-for-byte identical**; the 4 new
  `write_slice_byte_doff_kernel<Blob1/2/4/8>` entries are appended.
- **Host object symbols:** the ONLY additions are the **8** new `_doff` symbols
  (`baracuda_kernels_write_slice_b{1,2,4,8}_doff_run` + `..._doff_can_implement`); every
  pre-existing `_run` / `_can_implement` / `nibble` / `b16` symbol is preserved unchanged.

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **ALL PASSED**
(2026-07-09):

- Cell A / A′: all 12 capture+replay sweeps PASS (b1/b2/b4/b8 rank-4 dyn_axis=2 ×
  {mem-mutate, node-set} = 8; b2/b4 rank-3 dyn_axis=0 × {mem-mutate, node-set} = 4) —
  every replayed write lands at the updated slab and prior KV rows are preserved.
- Cell B: all 24 `_doff == _run` whole-buffer memcmp cases bit-identical (b1/b2/b4/b8 ×
  edges × probe classes), both layouts.
- Cell C: `compute-sanitizer memcheck` = **0 errors**, `racecheck` = **0 hazards** at
  `p = S-1`.
- Static: `_run` device PTX byte-identical (0 diff over 3068 lines); object symbols show
  exactly the 8 new `_doff` symbols added, none removed/changed.
