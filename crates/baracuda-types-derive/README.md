# baracuda-types-derive

Proc-macros for [`baracuda-types`]:

- `#[derive(DeviceRepr)]` — marks a `#[repr(C)]` type as ABI-stable for
  passing through CUDA kernel arguments and across the device boundary.

`#[derive(KernelArg)]` is deliberately *not* provided: `KernelArg` is
already implemented for `&T where T: DeviceRepr` via a blanket impl in
[`baracuda-types`], so deriving `DeviceRepr` is sufficient for a type to
be usable as a kernel argument via `&my_value`.

You normally enable this by adding `baracuda-types` with the `derive`
feature, which pulls this crate in as a transitive dependency:

```toml
[dependencies]
baracuda-types = { version = "0.0.1-alpha.7", features = ["derive"] }
```

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-types`]: https://docs.rs/baracuda-types
