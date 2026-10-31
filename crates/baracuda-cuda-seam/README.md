# baracuda-cuda-seam

The Fuel kernel-seam `Synthesizer` implementation over the CUDA backend.

`BaracudaSynthesizer` answers Fuel's JIT-on-request envelope (`fuel-kernel-seam`) using `baracuda-cuda-emit`; without the `nvrtc` feature it falls back to `unpopped::StubCompiler`.
It depends on `fuel-kernel-seam` and `fuel-kernel-seam-types` from crates.io, which must resolve to the same `unpopped-vocab` as the rest of the graph.

Part of [baracuda](https://github.com/ciresnave/baracuda); licensed under MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`).
