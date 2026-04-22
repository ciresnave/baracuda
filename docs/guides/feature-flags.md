# Feature flags

baracuda is one umbrella crate on top of ~40 smaller crates. Features control
which libraries get compiled and linked transitively. Every feature is
opt-in; the defaults are just `driver` + `runtime`.

## Individual features

| Feature      | Turns on                                    | Shared library at runtime       |
| ------------ | ------------------------------------------- | ------------------------------- |
| `driver`     | `baracuda-driver`                           | `libcuda.so.1` / `nvcuda.dll`   |
| `runtime`    | `baracuda-runtime`                          | `libcudart.so.12` / `cudart64_*.dll` |
| `nvrtc`      | `baracuda-nvrtc`                            | `libnvrtc.so.12`                |
| `nvjitlink`  | `baracuda-nvjitlink`                        | `libnvJitLink.so`               |
| `cublas`     | `baracuda-cublas` (incl. cuBLASLt + cuBLASXt)| `libcublas.so` + `libcublasLt.so` |
| `cufft`      | `baracuda-cufft`                            | `libcufft.so`                   |
| `cusparse`   | `baracuda-cusparse`                         | `libcusparse.so`                |
| `cusolver`   | `baracuda-cusolver` (Dn + Sp + Rf + Mg)     | `libcusolver.so` + `libcusolverMg.so` |
| `curand`     | `baracuda-curand`                           | `libcurand.so`                  |
| `cutensor`   | `baracuda-cutensor` *(separate NVIDIA download)* | `libcutensor.so`           |
| `cudnn`      | `baracuda-cudnn`                            | `libcudnn.so.9`                 |
| `nccl`       | `baracuda-nccl`                             | `libnccl.so.2`                  |
| `tensorrt`   | `baracuda-tensorrt`                         | `libnvinfer.so.10`              |
| `npp`        | `baracuda-npp`                              | `libnpp*.so`                    |
| `nvjpeg`     | `baracuda-nvjpeg`                           | `libnvjpeg.so`                  |
| `nvcomp`     | `baracuda-nvcomp`                           | `libnvcomp.so`                  |
| `cvcuda`     | `baracuda-cvcuda` *(Linux-primary)*         | `libcvcuda.so`                  |
| `nvml`       | `baracuda-nvml` *(ships with driver)*       | `libnvidia-ml.so.1`             |
| `cupti`      | `baracuda-cupti`                            | `libcupti.so.*`                 |
| `cufile`     | `baracuda-cufile` *(Linux-only)*            | `libcufile.so`                  |
| `cudf`       | `baracuda-cudf` *(RAPIDS, tracks libcudf_c)* | `libcudf.so`                   |

## Convenience bundles

| Bundle     | Contents                                                          |
| ---------- | ----------------------------------------------------------------- |
| `math`     | `cublas`, `cufft`, `cusparse`, `cusolver`, `curand`               |
| `dl`       | `cudnn`, `nccl`, `tensorrt`                                       |
| `imaging`  | `npp`, `nvjpeg`, `cvcuda`                                         |
| `ml`       | `driver` + `runtime` + `nvrtc` + `nvjitlink` + `math` + `dl`      |
| `full`     | `ml` + `imaging` + `nvcomp` + `nvml` + `cupti` + `cufile` + `cudf` + `cutensor` |

Examples:

```toml
# Just the runtime
[dependencies]
baracuda = { version = "0.1" }   # driver + runtime by default

# Typical ML training setup
[dependencies]
baracuda = { version = "0.1", features = ["ml"] }

# Profiling-focused build
[dependencies]
baracuda = { version = "0.1", features = ["driver", "runtime", "cupti", "nvml"] }

# Everything
[dependencies]
baracuda = { version = "0.1", features = ["full"] }
```

## Cross-cutting features

| Feature    | Effect                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------- |
| `async`    | `Event::wait_future()` returns a `Future`. Pulls in `futures-core` only (no runtime).          |
| `fp16`     | Enable `Half` (IEEE 754 binary16) adapters; interop with the `half` crate when combined with `half-crate`. |
| `bf16`     | Enable `BFloat16` adapters.                                                                    |
| `half-crate` | Add `From`/`Into` between baracuda's `Half`/`BFloat16` and the `half` crate's types.          |
| `num-complex-crate` | Add `From`/`Into` between baracuda's `Complex32`/`Complex64` and `num_complex::Complex<f32>`/`<f64>`. |

## Direct `-sys`-only dependency

If you want the raw FFI without pulling in the safe wrapper (for prototyping
a new API or matching an existing C call site exactly):

```toml
[dependencies]
baracuda-cublas-sys = "0.1"
baracuda-types = "0.1"
baracuda-core = "0.1"  # for the loader + Error plumbing
```

## Features we *don't* expose

- **`static-link`** — compiling the NVIDIA libraries statically. This would
  contradict the dynamic-load-everywhere design. If you specifically need
  static linking, drop to the `-sys` crate and its `build.rs`.
- **`bundle-drivers`** — shipping NVIDIA's `.so` / `.dll` with your binary.
  Not allowed by NVIDIA's EULA for most of these libraries; users must
  obtain them separately.

## Checking what's actually loaded

Every `-sys` crate has a `probe()` or equivalent function you can call to
verify the shared library loads without touching any real GPU state:

```rust
baracuda_cublas_sys::cublas()?;   // returns LoaderError if missing
baracuda_cudnn_sys::cudnn()?;
```

At a higher level, `baracuda::types::CudaVersion::detect()` returns the
driver version once the Driver API is loadable.
