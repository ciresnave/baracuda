# baracuda-cufft

Safe Rust wrappers for **NVIDIA cuFFT** — GPU Fast Fourier Transforms.

```rust,no_run
use baracuda_cufft::{Plan1d, TransformType};
use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_types::Complex32;

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let mut data: DeviceBuffer<Complex32> = DeviceBuffer::zeros(&ctx, 1024)?;

let plan = Plan1d::new(1024, TransformType::C2C, 1)?;
plan.exec_c2c(&mut data, baracuda_cufft::Direction::Forward)?;
# Ok(()) }
```

## Coverage

- **Plans**: `Plan1d`, `Plan2d`, `Plan3d`, `PlanMany` (arbitrary-rank
  batched FFTs with strides).
- **Transform types**: R2C / C2R / C2C / D2Z / Z2D / Z2Z.
- **cuFFT-XT** for multi-GPU plans (`PlanXt1d`, `PlanXtMany`).
- **Callbacks** for load and store transforms.
- Stream binding via `Plan::set_stream`; work-area override via
  `Plan::set_work_area`.

Pairs with [`baracuda-cufft-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cufft-sys`]: https://docs.rs/baracuda-cufft-sys
