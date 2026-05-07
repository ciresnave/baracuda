# baracuda-nccl

Safe Rust wrappers for **NVIDIA NCCL** — multi-GPU and multi-node
collective communication for ML / HPC training.

```rust,no_run
use baracuda_nccl::{Communicator, ReduceOp};
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let ctx = Context::new(&Device::get(0)?)?;
let stream = Stream::new(&ctx)?;
let comm = Communicator::single(&ctx, 0, 1)?; // rank 0 of 1 (single-GPU example)

let mut buf: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1024)?;
comm.all_reduce(&buf, &mut buf, 1024, ReduceOp::Sum, &stream)?;
# Ok(()) }
```

## Coverage

- **Communicators**: `Communicator::single` (single-process / multi-rank
  via NCCL_COMM_ID), `Communicator::from_unique_id` (cluster join).
- **All collectives**: all-reduce, reduce, reduce-scatter, all-gather,
  broadcast.
- **Point-to-point**: send / recv.
- **Group API**: `Group::new` + RAII auto-end, for batching multiple
  collectives into one launch.
- **Communicator splits**: `Communicator::split` for sub-team
  collectives.
- **Memory registration**: `MemoryRegistration` for pre-registering
  buffers used in many collectives.
- **Communicator lifecycle**: `abort`, `finalize`.

## Half-precision interop

Enable the `half-crate` feature to use `half::f16` and `half::bf16` as
`NcclScalar` directly:

```toml
baracuda-nccl = { version = "0.0.1-alpha.7", features = ["half-crate"] }
```

## Platform support

NCCL is **Linux-only**. The loader fails fast with a clear error on
Windows / macOS.

Pairs with [`baracuda-nccl-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nccl-sys`]: https://docs.rs/baracuda-nccl-sys
