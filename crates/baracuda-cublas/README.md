# baracuda-cublas

Safe Rust wrappers for **NVIDIA cuBLAS**, **cuBLASLt**, and **cuBLASXt**
— GPU-accelerated dense BLAS, with tile-tunable matmul and multi-GPU
GEMM as siblings.

Generic over scalar type (`f32` / `f64` / `Complex32` / `Complex64`)
where the math allows; concrete batched and Ex variants for everything
else.

```rust,no_run
use baracuda_cublas::{cublas, gemm, Op};
use baracuda_driver::{Context, Device, DeviceBuffer};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let handle = cublas::Handle::new(&ctx)?;

let m = 64; let n = 64; let k = 64;
let a: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, m * k)?;
let b: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, k * n)?;
let mut c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, m * n)?;

gemm(&handle, Op::N, Op::N, m as i32, n as i32, k as i32,
     1.0, &a, m as i32, &b, k as i32, 0.0, &mut c, m as i32)?;
# Ok(()) }
```

## Coverage

Comprehensive across all three sub-libraries:

- **cuBLAS core**: full L1 / L2 / L3 in `S` / `D` / `C` / `Z` plus
  real-only L2 (symv, trmv, trsv, ger, syr) and `Ex` variants
  (axpy, dot, nrm2, scal, rot).
- **Batched GEMM**: `gemm_batched`, `gemm_strided_batched`.
- **Direct batched solvers**: `getrf`, `getrs`, `getri`, `matinv` —
  batched-only, no host loops.
- **GemmEx / GemmStridedBatchedEx**: arbitrary input/output dtypes plus
  compute type, including FP16 in / FP32 accumulate.
- **cuBLASLt**: `MatmulDescriptor`, `MatrixLayout`, `MatmulPreference`,
  `MatmulHeuristics::query`, `matmul`. Fine-grained tuning when the
  default cuBLAS heuristic isn't picking the right algorithm.
- **cuBLASXt**: multi-GPU GEMM with affinity control.

## Stream binding

Set the active stream on a handle with `Handle::set_stream`. All
subsequent ops dispatch to that stream until you change it again.

Pairs with [`baracuda-cublas-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cublas-sys`]: https://docs.rs/baracuda-cublas-sys
