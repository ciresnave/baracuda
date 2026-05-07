# baracuda-nvrtc

Safe Rust wrappers for **NVIDIA NVRTC** — runtime CUDA C++ compilation.

NVRTC compiles CUDA C++ source to PTX (or CUBIN, on CUDA 11.1+) without
spawning `nvcc`, all in-process. Useful when kernel source is generated
at runtime, embedded as a string literal, or selected dynamically based
on input shapes.

```rust,no_run
use baracuda_nvrtc::Program;

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let src = r#"
extern "C" __global__ void scale(float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}
"#;

let mut prog = Program::new(src, "scale.cu")?;
prog.compile(&["--gpu-architecture=compute_80"])?;
let ptx = prog.ptx()?;
// Hand `ptx` to baracuda_driver::Module::load_ptx.
# Ok(()) }
```

## Coverage

- `Program` create / compile / destroy with options-as-`&[&str]`.
- PTX retrieval (`ptx()`, `ptx_size()`).
- CUBIN retrieval on CUDA 11.1+.
- Compile-log retrieval, surfaced inside the typed error on failure.
- Include-path registration (`add_name_expression`, `name_expression`).
- Name-mangled lookup for `__global__` C++ entry points.

## NVRTC vs `baracuda-forge`

- **`baracuda-nvrtc`**: in-process compilation at *runtime*. Use when
  source isn't known until then.
- **[`baracuda-forge`]**: out-of-process compilation at *build time*
  via `nvcc`. Better when source is fixed, you want incremental builds,
  parallel compile, multi-arch fat binaries, or CUTLASS dependencies.

Both load PTX through `baracuda-driver`'s `Module::load_ptx`.

Pairs with [`baracuda-nvrtc-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-forge`]: https://docs.rs/baracuda-forge
[`baracuda-nvrtc-sys`]: https://docs.rs/baracuda-nvrtc-sys
