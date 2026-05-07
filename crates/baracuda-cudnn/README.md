# baracuda-cudnn

Safe Rust wrappers for **NVIDIA cuDNN** — deep-learning primitive ops:
convolution, pooling, activation, normalization, RNN, attention.

## Coverage

Workhorse-level coverage. Covers the **classic op API** comprehensively
plus scaffolding for the modern **Backend (Graph) API**:

- **Activation**: forward + backward; ReLU, sigmoid, tanh, ELU, etc.
- **Convolution**: forward + backward-data + backward-filter +
  backward-bias, all algorithm variants. N-D `Filter` / `Convolution`
  descriptors.
- **Pooling**: max, avg, deterministic; forward + backward.
- **Softmax + LogSoftmax**: forward + backward.
- **Batch normalization**: inference, training, backward.
- **LRN, dropout, op-tensor, reduce-tensor, transform, add, scale.**
- **Spatial transformer.**
- **CTC loss.**
- **RNN**: forward + backward; LSTM / GRU / vanilla cell types.
- **N-D tensor / filter / pooling / convolution descriptors.**
- **Backend (Graph) API**: `BackendDescriptor` with
  `set_attribute_raw`, finalize, execute. Lower-level than the classic
  ops but supports operations the classic API doesn't.

## Half-precision interop

Enable the `half-crate` feature to use `half::f16` and `half::bf16` as
`CudnnDataType` directly, without going through baracuda's `Half` /
`BFloat16` conversion types:

```toml
baracuda-cudnn = { version = "0.0.1-alpha.7", features = ["half-crate"] }
```

## Stream binding

Set the active stream on a handle with `Handle::set_stream`. All ops
on that handle dispatch to that stream until you change it again.

Pairs with [`baracuda-cudnn-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cudnn-sys`]: https://docs.rs/baracuda-cudnn-sys
