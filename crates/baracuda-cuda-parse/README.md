# baracuda-cuda-parse

CUDA to IR frontend for the neutral `unpopped` kernel generator.

The tree-sitter grammar binding (`parse_cuda`), the CUDA residue list and the `CUDA` `Frontend` descriptor.
It mirrors `baracuda-cuda-emit` on the parse side. Its MAJOR.MINOR follows the `unpopped` release it builds against; the PATCH is baracuda's own counter.

Part of [baracuda](https://github.com/ciresnave/baracuda); licensed under MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`).
