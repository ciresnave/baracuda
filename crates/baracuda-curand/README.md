# baracuda-curand

Safe Rust wrappers for **NVIDIA cuRAND** — GPU pseudo-random and
quasi-random number generation.

```rust,no_run
use baracuda_curand::{Generator, RngKind};
use baracuda_driver::{Context, Device, DeviceBuffer};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 1024)?;

let mut rng = Generator::new(RngKind::Default)?;
rng.seed(0xDEAD_BEEF)?;
rng.uniform(&mut buf)?;
# Ok(()) }
```

## Coverage

- **All RNG kinds**: XORWOW, MRG32k3a, MTGP32, Philox4_32_10, MT19937,
  Sobol32 / Sobol64, Scrambled Sobol32 / 64.
- **All distributions**: uniform / uniform-double, normal / normal-double,
  log-normal / log-normal-double, Poisson, binomial. Quasi-random
  variants where applicable.
- **Stream-async generation**: `Generator::set_stream` for per-stream
  dispatch.
- **Determinism controls**: `seed`, `set_offset`, `set_ordering`.

## Quasi-random support

For Sobol generators, baracuda-curand exposes
`set_quasi_random_dimensions`, `direction_vectors_32 / _64`, and
`scramble_constants_32 / _64` so you can drive the sequence with the
same direction vectors NVIDIA ships.

Pairs with [`baracuda-curand-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-curand-sys`]: https://docs.rs/baracuda-curand-sys
