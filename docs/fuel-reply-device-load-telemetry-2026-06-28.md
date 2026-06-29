# Baracuda reply — device-load telemetry (Step E, Phase B2) (2026-06-28)

Reply to Fuel's draft ask *"Baracuda ask — device-load telemetry (Step E,
Phase B2)"* (read-only `cuStreamQuery` + NVML utilization for
`DeviceLoadSelector`). Companion to the alpha.66 VRAM-query precedent the ask
cites (`Device::vram_info()` → `cuMemGetInfo_v2`).

## TL;DR — already shipped; you can build against it today

**Both halves of the ask already exist in the published surface as of
`0.0.1-alpha.69`** (live on `main`). You do **not** need a new Baracuda release
to unblock Step E:

1. **This-process stream idle/busy** → `baracuda_driver::Stream::is_complete()`
   (and the same on `baracuda_runtime::Stream`) — a `cuStreamQuery` wrapper:
   `Ok(true)` ⇒ idle, `Ok(false)` ⇒ work pending.
2. **Cross-process device utilization** → `baracuda_nvml::Device::utilization()`
   → `nvmlDeviceGetUtilizationRates`, plus `compute_processes()` /
   `graphics_processes()` for the SM proxy.

Per your "shape is baracuda's call", we've **additionally** added two thin,
read-only convenience aliases that match your load-balancer's preferred shape
exactly. They land in **alpha.70** (next publish) — but they are pure aliases
over the alpha.69 methods, so you can wire `DeviceLoadSelector` against alpha.69
right now and swap to the aliases when alpha.70 drops, or never. Nothing here
blocks you.

---

## 1. This-process stream idle/busy — `cuStreamQuery`

Shipped (alpha.69), on **both** stream layers so whichever you consume in
`fuel-cuda-backend` has it:

| API | Crate | Since |
|---|---|---|
| `Stream::is_complete() -> Result<bool>` | `baracuda-driver`, `baracuda-runtime` | alpha.69 |
| `Stream::is_idle() -> Result<bool>` *(alias, load-probe phrasing)* | `baracuda-driver`, `baracuda-runtime` | alpha.70 |

`is_idle()` is a literal alias of `is_complete()` — same `cuStreamQuery`
(`CUDA_SUCCESS` ⇒ `Ok(true)`/idle, `CUDA_ERROR_NOT_READY` ⇒ `Ok(false)`/busy).
It is read-only: it **never synchronizes the stream or perturbs scheduling**, as
your constraint requires. The signal reflects only *this process's* submissions
to that stream — exactly the scope you described.

On the `Result<bool>` vs your proposed `-> bool`: we keep `Result` to stay
consistent with the crate and to be honest about a genuinely-broken handle
(rather than conflating "invalid stream" with "busy"). A selector that wants a
plain `bool` collapses it conservatively: `stream.is_idle().unwrap_or(false)`
(treat "can't tell" as busy → don't steer toward it).

## 2. Cross-process device utilization — NVML

Shipped (alpha.69):

| API | Returns | Since |
|---|---|---|
| `Device::utilization()` | `Result<Utilization { gpu: u32, memory: u32 }>` | alpha.69 |
| `Device::gpu_utilization_percent()` *(convenience)* | `Option<u8>` (GPU busy %) | alpha.70 |
| `Device::compute_processes()` | `Result<Vec<nvmlProcessInfo_t>>` | alpha.69 |
| `Device::graphics_processes()` | `Result<Vec<nvmlProcessInfo_t>>` | alpha.69 |

`gpu_utilization_percent()` is the `Option<u8>` you asked for: GPU core busy
percent, or an honest `None` when the driver can't answer — **never a fabricated
value**, same contract as the `cuMemGetInfo` VRAM read. It is `min(100)`-clamped
and never panics. The full `utilization()` additionally exposes the
memory-controller percent if you ever want it; `compute_processes()` gives you
the running-process list for a free-vs-used SM proxy.

This is the signal that captures *other processes* sharing the GPU — the one
thing your in-flight-work counter can't see, and the higher-value one for
shared-GPU scheduling, as your ask notes.

## 3. Your constraints — confirmed

- **Read-only / no perturbation.** `cuStreamQuery` is a non-blocking poll; NVML
  reads never touch the CUDA scheduler. Neither synchronizes anything.
- **`Option`/`Result`, never panic, honest "no signal".** Both aliases honor
  this: `is_idle()` returns `Result`; `gpu_utilization_percent()` returns
  `Option` with `None` on driver-can't-answer. Matches the `vram_info()`
  precedent.
- **`Send + Sync`, allocation-free.** Both `Stream` and `nvml::Device` are
  `Send + Sync`; the scalar reads allocate nothing. (`compute_processes()`
  allocates a `Vec` — use it sparingly, not per-dispatch.)
- **No new required dependency for Fuel's default build.** NVML already lives in
  its **own crate** (`baracuda-nvml`, separate from `baracuda-driver`). Fuel
  pulls `libnvidia-ml` only by depending on `baracuda-nvml` — gate that behind
  your own feature and the default build never sees it. No Baracuda feature flag
  needed on our side; the crate split already isolates it.

## 4. Suggested wiring at your `BackendRuntime` boundary

```rust
// fuel-cuda-backend: DynBackendDevice::as_backend_runtime()

// (1) this-process stream load — alpha.69 today, or is_idle() on alpha.70
fn stream_busy(&self, stream: &baracuda_driver::Stream) -> bool {
    !stream.is_complete().unwrap_or(false) // can't tell ⇒ assume busy
}

// (2) cross-process device load — the signal your in-flight counter can't see
fn device_utilization(&self, dev: &baracuda_nvml::Device) -> Option<u8> {
    dev.gpu_utilization_percent() // alpha.70; on alpha.69:
    // dev.utilization().ok().map(|u| u.gpu.min(100) as u8)
}
```

Your `pending_work()` accessor on `fuel-backend-contract::BackendRuntime` and
the `as_backend_runtime()` impl are Fuel-side plumbing — nothing for us there.

## 5. Verification

On-device (RTX 4070, sm_89): `stream_is_idle_after_sync` green
(`wave1_smoke`); `query_device_health` green (`baracuda-nvml` smoke) reporting
`util gpu=0% (helper=Some(0))` — the helper agrees with the full read and stays
in range.

## 6. Status / next step

The aliases are committed on `feat/kernel-specialization`; they land for you in
the **alpha.70 publish**. Since they're pure aliases, **start now on alpha.69**
(`is_complete()` + `utilization().ok().map(...)`) and adopt the aliases at your
leisure. If you'd rather we *not* add the aliases and keep the surface minimal,
say so and we'll drop them — alpha.69 already satisfies the ask either way.
