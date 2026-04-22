# Matrix multiply with cuBLAS

cuBLAS is the right tool when "`C = alpha * A * B + beta * C` at scale" is in
your critical path. baracuda wraps three surfaces in the same crate:

- **Classic cuBLAS** — `baracuda_cublas::gemm`, generic over
  `f32` / `f64` / `Complex32` / `Complex64`.
- **`cublasGemmEx`** — mixed-precision GEMM (fp16 / bf16 / int8 input with fp32
  accumulate).
- **cuBLASLt** — descriptor-based GEMM with tuning heuristics, epilogue fusion
  (bias + activation), and full multi-GPU via cuBLASXt.

## SGEMM in three calls

```rust
use baracuda::cublas::{gemm, Handle, Op};
use baracuda::driver::{Context, Device, DeviceBuffer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;
    let handle = Handle::new()?;

    // 2×3 × 3×2 = 2×2, column-major.
    let a_host: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];       // 2×3
    let b_host: Vec<f32> = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0];    // 3×2
    let a = DeviceBuffer::from_slice(&ctx, &a_host)?;
    let b = DeviceBuffer::from_slice(&ctx, &b_host)?;
    let mut c: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4)?;

    gemm(&handle, Op::N, Op::N, 2, 2, 3, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2)?;

    let mut host = vec![0.0f32; 4];
    c.copy_to_host(&mut host)?;
    println!("{host:?}"); // [58, 139, 64, 154] column-major
    Ok(())
}
```

**Column-major is not optional.** cuBLAS inherits the Fortran convention. If
your data is row-major, swap the operands and transpose with `Op::T` to get
the same effect.

## Complex and double-precision

`BlasScalar` is sealed over `f32` / `f64` / `Complex32` / `Complex64`. Picking
the element type is enough to route to `cublas?gemm_v2`:

```rust
use baracuda_types::Complex32;
use baracuda::cublas::{gemm, Op};

gemm::<Complex32>(
    &handle, Op::N, Op::N, m, n, k,
    Complex32::new(1.0, 0.0),
    &a, m,
    &b, k,
    Complex32::ZERO,
    &mut c, m,
)?;
```

## Batched GEMM (for attention)

Transformer attention is dominated by batched GEMMs with a constant stride
between matrices. Use `gemm_strided_batched`:

```rust
use baracuda::cublas::{gemm_strided_batched, Op};

// Each matrix is m×n = 64×64, stride = 64*64 = 4096 elements.
gemm_strided_batched::<f32>(
    &handle, Op::N, Op::T,
    64, 64, 128,            // m, n, k
    1.0,
    &q, 64, 64 * 128,       // A: 64×128 per batch
    &k_t, 128, 128 * 64,    // B: 128×64 per batch
    0.0,
    &mut scores, 64, 64 * 64,
    batch_size,
)?;
```

If your per-matrix strides aren't constant, drop to `gemm_batched` which takes
device-side arrays of pointers (one per matrix).

## Mixed-precision (`gemm_ex`)

FP16 math with FP32 accumulators — the standard pattern for training on
Hopper / Ampere tensor cores:

```rust
use baracuda::cublas::{gemm_ex, Op, cudaDataType_t, cublasComputeType_t};

unsafe {
    gemm_ex(
        &handle, Op::N, Op::N, m, n, k,
        &alpha as *const f32 as *const _,
        a_fp16_ptr as *const _, cudaDataType_t::R_16F, m,
        b_fp16_ptr as *const _, cudaDataType_t::R_16F, k,
        &beta as *const f32 as *const _,
        c_fp32_ptr as *mut _,  cudaDataType_t::R_32F, m,
        cublasComputeType_t::Compute32F,
        -1,  // default algo
    )?;
}
```

The pointers are `*const c_void` because cuBLAS is type-erased here; pair
this with `baracuda_types::Half` buffers and you'll stay ergonomic.

## cuBLASLt with heuristics

The Lt API lets you ask cuBLAS "what algorithms can run this matmul?",
optionally with an epilogue like bias + activation:

```rust
use baracuda::cublas::lt::{LtHandle, MatmulDesc, MatrixLayout, MatmulPreference, heuristics_search, matmul};
use baracuda_cublas_sys::{cublasOperation_t};
use baracuda_cublas_sys::functions::{cublasComputeType_t, cudaDataType_t};

let lt = LtHandle::new()?;
let desc = MatmulDesc::new(cublasComputeType_t::Compute32F, cudaDataType_t::R_32F)?;
desc.set_transa(cublasOperation_t::N)?;
desc.set_transb(cublasOperation_t::N)?;

let layout_a = MatrixLayout::new(cudaDataType_t::R_32F, m as u64, k as u64, m as i64)?;
let layout_b = MatrixLayout::new(cudaDataType_t::R_32F, k as u64, n as u64, k as i64)?;
let layout_c = MatrixLayout::new(cudaDataType_t::R_32F, m as u64, n as u64, m as i64)?;

let pref = MatmulPreference::new()?;
pref.set_max_workspace_bytes(32 * 1024 * 1024)?;  // 32 MiB scratch

let heurs = heuristics_search(&lt, &desc, &layout_a, &layout_b, &layout_c, &layout_c, &pref, 1)?;
let algo = heurs[0].algo;
let workspace_size = heurs[0].workspace_size;

// ... allocate workspace, then:
unsafe {
    matmul(
        &lt, &desc,
        &alpha as *const _ as _, a_ptr as _, &layout_a,
        b_ptr as _, &layout_b,
        &beta as *const _ as _, c_ptr as _, &layout_c,
        c_ptr, &layout_c,
        Some(&algo),
        workspace_ptr, workspace_size,
        Some(&stream),
    )?;
}
```

See `baracuda::cublas::lt` for the full attribute catalog.

## Rank-1 updates, triangular solves, and friends

- **BLAS-2**: `gemv`, `symv`, `trmv`, `trsv`, `ger`, `syr`.
- **BLAS-3**: `gemm`, `symm`, `hemm`, `syrk`, `herk`, `trmm`, `trsm`.
- **BLAS-1**: `axpy`, `dot`, `scal`, `nrm2`, `asum`, `iamax`, `iamin`, `copy`,
  plus the mixed-precision Ex variants (`ex::axpy`, `ex::dot`, `ex::nrm2`,
  `ex::scal`, `ex::rot`).

All four kinds are generic over `f32` / `f64` / `Complex32` / `Complex64`
where the BLAS defines the op; Hermitian ops (`hemm`, `herk`) sensibly require
complex types at the trait level.

## Multi-GPU GEMM with cuBLASXt

Host-pointer operands, automatic device tiling:

```rust
use baracuda::cublas::xt::{gemm as xt_gemm, XtHandle};
use baracuda::cublas::Op;

let handle = XtHandle::new()?;
handle.device_select(&[0, 1])?;  // stripe across GPUs 0 and 1
handle.set_block_dim(2048)?;

unsafe {
    xt_gemm::<f32>(
        &handle, Op::N, Op::N, m, n, k,
        1.0,
        host_a.as_ptr(), m,
        host_b.as_ptr(), k,
        0.0,
        host_c.as_mut_ptr(), m,
    )?;
}
```

## Validating your code

Always cross-check against a CPU reference the first time. `ndarray` or
`nalgebra` are both fine — baracuda doesn't care which linear-algebra crate
you use for that.
