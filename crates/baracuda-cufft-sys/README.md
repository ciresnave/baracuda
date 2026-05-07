# baracuda-cufft-sys

Raw FFI bindings + dynamic loader for **NVIDIA cuFFT** — GPU-side Fast
Fourier Transforms.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcufft.so` / `cufft64_*.dll`.

**Most users want [`baracuda-cufft`]** — that crate exposes typed plan
builders for 1-D / 2-D / 3-D and many-sized batch FFTs, with all
real/complex transform types (R2C, C2R, C2C, D2Z, Z2D, Z2Z), cuFFT-XT
multi-GPU plans, and load/store callbacks.

## What's exposed

- Plan creation (`cufftPlan1d`, `cufftPlan2d`, `cufftPlan3d`,
  `cufftPlanMany`).
- Execution (`cufftExecC2C`, `cufftExecZ2Z`, `cufftExecR2C`,
  `cufftExecC2R`, `cufftExecD2Z`, `cufftExecZ2D`).
- cuFFT-XT multi-GPU API.
- Set-stream, work-area control, callback registration.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cufft`]: https://docs.rs/baracuda-cufft
