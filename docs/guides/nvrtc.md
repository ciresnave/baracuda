# NVRTC — compile CUDA C++ at runtime

NVRTC lets you emit PTX from CUDA C++ source at runtime, which means:

- **No `nvcc` in your build pipeline.** The NVIDIA driver's runtime compiler
  does the work.
- **Specialized kernels per input.** Bake tile sizes, loop bounds, and types
  into the source at launch time.
- **Distributable binaries.** Your Rust crate ships a string, not a
  precompiled `.ptx` targeted at a specific compute capability.

## Minimal example

```rust
use baracuda::driver::{Context, Device, DeviceBuffer, Module, Stream};
use baracuda::nvrtc::{compile_ptx, CompileOptions};

const SRC: &str = r#"
extern "C" __global__
void saxpy(int n, float a, const float* x, float* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;

    let opts = CompileOptions::new().arch("sm_80");  // target Ampere
    let ptx = compile_ptx(SRC, "saxpy.cu", &opts)?;

    let module = Module::load_ptx(&ctx, ptx.as_str())?;
    let kernel = module.get_function("saxpy")?;

    let n = 1_000_000i32;
    let a = 2.0f32;
    let x = DeviceBuffer::from_slice(&ctx, &vec![1.0f32; n as usize])?;
    let mut y = DeviceBuffer::from_slice(&ctx, &vec![3.0f32; n as usize])?;

    let stream = Stream::new(&ctx)?;
    let block = 256;
    let grid = (n as u32 + block - 1) / block;

    kernel
        .launch()
        .grid((grid, 1, 1))
        .block((block, 1, 1))
        .arg(&n).arg(&a).arg(&x).arg(&mut y)
        .stream(&stream)
        .launch()?;
    stream.synchronize()?;

    let mut host = vec![0.0f32; n as usize];
    y.copy_to_host(&mut host)?;
    assert!(host.iter().all(|&v| (v - 5.0).abs() < 1e-5));
    Ok(())
}
```

## Compile options

```rust
use baracuda::nvrtc::CompileOptions;

let opts = CompileOptions::new()
    .arch("sm_90a")            // target Hopper architecture-specific features
    .include("/opt/cuda/include")
    .define("TILE_M", "128")   // -DTILE_M=128
    .define("TILE_N", "128")
    .rdc(true)                 // relocatable device code — needed for cuBLASDX etc.
    .line_info(true);          // preserve source location for cuda-gdb
```

Common flags:

- `--gpu-architecture=sm_XX` — pick a target SM version.
- `-I path` — include-path entries.
- `-D NAME=value` — preprocessor macros.
- `--use_fast_math` — fast approximations for transcendentals.
- `--std=c++17` — selects the C++ dialect.
- `--relocatable-device-code=true` — needed to link against other PTX or
  device libraries.

## Inspecting the compile log

When compilation fails, the PTX builder returns `Error::NvrtcCompileFailed`
with the compiler log attached. On success the log may still contain
warnings:

```rust
match compile_ptx(SRC, "kernel.cu", &opts) {
    Ok(ptx) => {
        if let Some(log) = ptx.compile_log() {
            eprintln!("NVRTC warnings:\n{log}");
        }
        // use ptx.as_str()
    }
    Err(e) => {
        eprintln!("NVRTC failed: {e}");
        return Err(e.into());
    }
}
```

## Device-side headers

NVRTC can resolve `#include <cuda_fp16.h>` and the other device-side headers
if you pass `-I path/to/cuda/include`. You don't need the rest of the
toolkit — just the headers.

For offline compilation, point `-I` at the directory that shipped with the
driver:

- Linux: `/usr/local/cuda/include`
- Windows: `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\include`

## Linking multiple PTXs with nvJitLink

Once you're compiling multiple modules at runtime (e.g. kernel + device-side
helper library), use [nvJitLink](../../crates/baracuda-nvjitlink) to stitch
them before loading into a CUDA module. That sits right above NVRTC in the
same flow.
