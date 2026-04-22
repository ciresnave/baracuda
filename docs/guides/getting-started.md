# Getting started with baracuda

This guide walks you from a clean machine to running your first CUDA kernel
through baracuda, then points you at the common follow-ups.

## What you need

- **A Rust toolchain** — `rustup`, Rust 1.75+ (`rust-version` in the workspace
  manifest). Install with `curl https://sh.rustup.rs -sSf | sh` on Linux/macOS
  or from [rustup.rs](https://rustup.rs) on Windows.
- **An NVIDIA driver.** Any driver that supports CUDA 11.4 or newer is enough
  for `baracuda-driver` + `baracuda-runtime`. Some libraries (cuBLASLt,
  nvJitLink, green contexts, conditional graph nodes) require CUDA 12+.
- **Optionally: the CUDA Toolkit.** Only needed if you want to use the
  non-driver libraries (cuBLAS, cuDNN, etc.) *at runtime*. baracuda still
  *builds* without any CUDA artifacts on the host — the dynamic loader resolves
  symbols when your program actually calls them.

> **WSL2:** install an NVIDIA Windows driver with WSL support, then in your
> distro `sudo apt install libnvidia-compute-*`. baracuda probes
> `/usr/lib/wsl/lib/` for the driver stub automatically.

## Adding the dependency

```toml
[dependencies]
baracuda = { version = "0.1", features = ["driver", "runtime"] }
```

The default feature set is just `driver` + `runtime`. Turn on additional
libraries per your needs — see [feature-flags.md](feature-flags.md).

## Hello, device

```rust
use baracuda::runtime::{set_device, DeviceBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_device(0)?;

    let host = vec![1.0f32, 2.0, 3.0, 4.0];
    let device = DeviceBuffer::from_slice(&host)?;

    let mut back = vec![0.0f32; host.len()];
    device.copy_to_host(&mut back)?;

    assert_eq!(host, back);
    println!("H2D + D2H round-trip successful ✓");
    Ok(())
}
```

Run it with `cargo run`. You should see the checkmark; the buffer traveled to
the GPU and back.

## Your first kernel

baracuda can load a kernel that was compiled ahead of time to PTX (via `nvcc
--ptx`), or one you compile at runtime with [NVRTC](nvrtc.md). For the
precompiled path:

```rust
use baracuda::driver::{Context, Device, DeviceBuffer, Module, Stream};

const PTX: &str = include_str!("vector_add.ptx");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;

    let module = Module::load_ptx(&ctx, PTX)?;
    let kernel = module.get_function("vector_add")?;

    let a = DeviceBuffer::from_slice(&ctx, &[1.0f32, 2.0, 3.0, 4.0])?;
    let b = DeviceBuffer::from_slice(&ctx, &[10.0f32, 20.0, 30.0, 40.0])?;
    let mut c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4)?;
    let n = 4u32;

    let stream = Stream::new(&ctx)?;
    kernel
        .launch()
        .grid((1, 1, 1))
        .block((4, 1, 1))
        .arg(&a).arg(&b).arg(&mut c).arg(&n)
        .stream(&stream)
        .launch()?;
    stream.synchronize()?;

    let mut result = vec![0.0f32; 4];
    c.copy_to_host(&mut result)?;
    println!("{result:?}"); // [11, 22, 33, 44]
    Ok(())
}
```

If you'd rather compile the CUDA C++ at runtime, skip to
[nvrtc.md](nvrtc.md) — you won't need a precompiled `.ptx` file.

## Common pitfalls

- **"Library not found"** — means the loader couldn't find the shared
  library. Check that the relevant NVIDIA package is installed and on
  `LD_LIBRARY_PATH` (Linux) / `PATH` (Windows). The error enumerates
  everywhere baracuda tried.
- **"Symbol not found"** — means your driver / runtime is older than the
  function being called. baracuda's feature-gating routes these to
  `Error::FeatureNotSupported` at the safe-crate level wherever possible.
- **Silent "zero-time" kernel launches** — you forgot `stream.synchronize()`
  (or `cudaDeviceSynchronize`). Until a stream syncs, host-side timing
  measures only the kernel-submission cost.
- **Freeing a descriptor before the data** — safe wrappers prevent this at
  compile time via lifetime parameters, but if you reach for `as_raw()` you
  lose that protection.
- **Mixing Driver and Runtime handles** — they *are* compatible (they're the
  same underlying types), but the lifecycle rules differ. Prefer picking one
  API per program.

## Where to go next

- [Matrix multiply with cuBLAS](matmul-cublas.md) — your first real
  computation.
- [NVRTC runtime compilation](nvrtc.md) — skip the precompiled `.ptx`.
- [Streams and events](streams-and-events.md) — overlap H2D / compute / D2H.
- [Feature flags](feature-flags.md) — picking the right bundle.
