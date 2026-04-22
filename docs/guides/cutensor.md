# cuTENSOR guide

cuTENSOR is NVIDIA's high-performance tensor-primitive library —
einsum-style contractions, element-wise ops, reductions, and
permutations. baracuda wraps the full v2 host-side surface.

## Installation

cuTENSOR is a separate NVIDIA download. On Linux it's usually in
`/usr/lib/x86_64-linux-gnu/libcutensor.so.2`. On Windows the installer
drops it into `C:\Program Files\NVIDIA cuTENSOR\v<ver>\bin\<cuda-major>\cutensor.dll`;
baracuda's loader searches that path automatically.

## Concepts

- [`Handle`](../../crates/baracuda-cutensor/src/lib.rs) — per-process
  library handle; owns the plan cache.
- [`TensorDescriptor`] — shape + strides + dtype + alignment for one tensor.
- [`OperationDescriptor`] — an un-compiled op. Built via
  [`Contraction::new`], [`Reduction::new`],
  [`ElementwiseBinary::new`], [`ElementwiseTrinary::new`], or
  [`Permutation::new`].
- [`PlanPreference`] — algorithm selection + JIT mode.
- [`Plan`] — compiled op, bound to a workspace size.

The execute path is per-kind on [`Plan`]: `plan.contract(...)`,
`plan.reduce(...)`, `plan.elementwise_binary(...)`, etc.

## Flow: matmul as a contraction

`D[m,n] = Σₖ A[m,k] · B[k,n]` written as `C[m,n] = Σ A[m,k] · B[k,n]`:

```rust
use baracuda_cutensor::*;

let handle = Handle::new()?;
let m = 128; let n = 128; let k = 64;

// cuTENSOR's default layout is column-major. Pass row-major strides.
let a = TensorDescriptor::new(&handle, &[m, k], Some(&[k, 1]), DataType::F32, 128)?;
let b = TensorDescriptor::new(&handle, &[k, n], Some(&[n, 1]), DataType::F32, 128)?;
let d = TensorDescriptor::new(&handle, &[m, n], Some(&[n, 1]), DataType::F32, 128)?;

// Mode labels — any distinct i32 set works. Contracted mode is the one
// that appears on both inputs but not the output.
let modes_a = [0i32, 2];  // [m, k]
let modes_b = [2i32, 1];  // [k, n]
let modes_d = [0i32, 1];  // [m, n]

let compute = handle.compute_desc_32f()?;
let op = unsafe {
    Contraction::new(&handle, &a, &modes_a, &b, &modes_b,
                     &d, &modes_d,   // C (β = 0 below makes it write-only)
                     &d, &modes_d, compute)
}?;

let pref = PlanPreference::default_for(&handle)?;
let ws = op.estimate_workspace(&pref, WorkspaceKind::Default)?;
let plan = Plan::new(&op, &pref, ws)?;

let alpha: f32 = 1.0; let beta: f32 = 0.0;
unsafe {
    plan.contract(&alpha as *const _ as *const _,
                  d_a_ptr, d_b_ptr,
                  &beta as *const _ as *const _, d_d_ptr, d_d_ptr,
                  d_workspace_ptr, ws, stream)?;
}
```

See [`crates/baracuda-cutensor/tests/contract_gemm.rs`](../../crates/baracuda-cutensor/tests/contract_gemm.rs)
for the end-to-end test with CPU verification.

## Compute descriptors

cuTENSOR v2 requires a non-null `cutensorComputeDescriptor_t`. baracuda
resolves the library's exported globals (`CUTENSOR_COMPUTE_DESC_32F`
etc.) as data symbols — use the helper accessors:

```rust
let compute = handle.compute_desc_32f()?;   // F32 compute
let compute = handle.compute_desc_16f()?;   // FP16 compute
let compute = handle.compute_desc_tf32()?;  // TF32 (Ampere+)
let compute = handle.compute_desc_3xtf32()?;// 3×TF32 mantissa-extended
```

## Unary + binary operators

Applied per-operand before the main op:

- [`UnaryOp`]: `Identity`, `Sqrt`, `Relu`, `Conj`, `Rcp`, `Sigmoid`, `Tanh`
- [`BinaryOp`]: `Add`, `Mul`, `Max`, `Min`

Used by [`ElementwiseBinary`] / [`ElementwiseTrinary`] / [`Reduction`].

## Plan cache

Plans are compiled each time from `(op_desc, pref, workspace_size)`.
The library keeps a host-side cache keyed by op shape. Persist it
across runs:

```rust
handle.resize_plan_cache(256)?;       // default is 64
handle.write_cache_to_file("cache.bin")?;
// … later:
handle.read_cache_from_file("cache.bin")?;
```

## Layout gotcha — default is column-major

cuTENSOR infers strides from the extents when you pass `None` for
strides. The inferred layout is **column-major** (Fortran order), which
surprises most Rust code. **Always pass explicit row-major strides**
(`strides[i] = product of extents after i`) unless you actually want
column-major.
