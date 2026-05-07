# baracuda-cudnn-sys

Raw FFI bindings + dynamic loader for **NVIDIA cuDNN** — deep-learning
primitive ops (convolution, pooling, activation, normalization, RNN,
attention, ...).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcudnn.so` / `cudnn64_*.dll`.

cuDNN 9's main DLL on Windows is a facade that depends on companion
DLLs (`cudnn_ops64_9.dll`, `cudnn_graph64_9.dll`, ...) in the same
directory. This crate's loader detects that and adds the cuDNN bin
directory to PATH on Windows so `LoadLibraryExW` resolves the
companions.

**Most users want [`baracuda-cudnn`]** — that crate exposes typed
descriptors for tensors / filters / convolutions / RNNs, the classic op
API (forward + backward), and the modern Backend (Graph) API
scaffolding.

## What's exposed

- All `cudnnHandle_t` / `cudnnTensorDescriptor_t` /
  `cudnnFilterDescriptor_t` / `cudnnConvolutionDescriptor_t` etc. types.
- Classic op API: activation, convolution forward + backward (data /
  filter / bias), pooling, softmax, batch-norm (inference / training /
  backward), LRN, dropout, op-tensor, reduce, transform, add, scale,
  spatial transformer, CTC loss, RNN forward + backward.
- Backend (Graph) API for cuDNN v8+ — descriptors, attribute setters,
  finalize/execute.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cudnn`]: https://docs.rs/baracuda-cudnn
