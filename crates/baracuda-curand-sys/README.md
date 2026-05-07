# baracuda-curand-sys

Raw FFI bindings + dynamic loader for **NVIDIA cuRAND** — GPU-side
pseudo-random and quasi-random number generation.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcurand.so` / `curand64_*.dll`.

**Most users want [`baracuda-curand`]** — that crate exposes a typed
`Generator` with seed control, all documented distributions (uniform,
normal, log-normal, Poisson, binomial), and stream-async generation.

## What's exposed

- **Host generators** for every `curandRngType_t` (XORWOW, MRG32k3a,
  MTGP32, Philox4_32_10, MT19937, Sobol, Scrambled Sobol).
- **Distributions**: uniform / uniform-double, normal / normal-double,
  log-normal / log-normal-double, Poisson, binomial.
- Quasi-random direction-vectors and scramble-constants.
- Set-stream, set-seed, set-offset, set-ordering.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-curand`]: https://docs.rs/baracuda-curand
