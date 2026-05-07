# baracuda-types

Shared type vocabulary for the [baracuda](https://github.com/ciresnave/baracuda)
CUDA stack.

This crate contains only pure-data types — no I/O, no `libloading`, no
runtime machinery. It's intended to be a cheap dependency for any crate
(including user-authored CUDA wrappers) that wants to speak baracuda's
type vocabulary without pulling in the loader infrastructure.

## What's here

- **`Half`, `BFloat16`, `Complex32`, `Complex64`** — first-class scalar
  types that match CUDA's `__half`, `__nv_bfloat16`, `cuFloatComplex`,
  `cuDoubleComplex` ABI-for-ABI.
- **`DeviceRepr`** — marker trait for types with a stable, ABI-compatible
  layout. Implemented for every primitive numeric type, fixed-size arrays
  of `DeviceRepr` elements, and tuples up to arity 12. Derive it on
  your own `#[repr(C)]` types via `#[derive(DeviceRepr)]` (with the
  `derive` feature).
- **`KernelArg`** — blanket-impl'd for `&T where T: DeviceRepr`, so any
  `DeviceRepr` value can be passed by reference to a CUDA kernel launch
  builder.
- **`CudaVersion`** — `(major, minor)` pair with `parse_version_string`.
- **`Feature`** — runtime feature-availability tag used by the safe
  wrapper crates to gate APIs added after CUDA 11.4.
- **`CudaStatus`** — unifying error trait that every per-library error
  type implements, so application-level error types can absorb all of
  them.

## Optional integrations (feature flags)

| Feature | Pulls in | What it adds |
| --- | --- | --- |
| `derive` | `baracuda-types-derive` | `#[derive(DeviceRepr)]` |
| `half-crate` | `half` | `DeviceRepr` for `half::f16` and `half::bf16` |
| `num-complex-crate` | `num-complex` | `DeviceRepr` for `Complex<f32>` / `Complex<f64>` |
| `bytemuck` | `bytemuck` | `Pod` / `Zeroable` for the scalar types |
| `f8-crate` | `float8` | `DeviceRepr` for `F8E4M3` / `F8E5M2` |

## Example

```rust
use baracuda_types::{DeviceRepr, Half};

#[repr(C)]
#[derive(Copy, Clone, baracuda_types::DeviceRepr)]
struct MyKernelParams {
    scale: f32,
    half_offset: Half,
    flags: u32,
}
```

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.
