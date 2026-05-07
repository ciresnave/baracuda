# baracuda-nccl-sys

Raw FFI bindings + dynamic loader for **NVIDIA NCCL** — multi-GPU /
multi-node collective communication (all-reduce, broadcast, all-gather,
reduce-scatter, send/recv).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libnccl.so`.

**Most users want [`baracuda-nccl`]** — that crate exposes typed
`Communicator` handles, scalar-typed collectives, p2p send/recv, group
APIs, and stream-async dispatch.

## What's exposed

- `ncclComm_t` create / init / destroy + `ncclCommSplit` /
  `ncclCommAbort` / `ncclCommFinalize`.
- All collectives: `ncclAllReduce`, `ncclReduce`, `ncclReduceScatter`,
  `ncclAllGather`, `ncclBroadcast`.
- Point-to-point: `ncclSend`, `ncclRecv`.
- Group API: `ncclGroupStart`, `ncclGroupEnd`.
- Buffer registration: `ncclMemAlloc`, `ncclMemFree`,
  `ncclCommRegister`, `ncclCommDeregister`.

## Platform support

NCCL is **Linux-only**. The loader fails fast with a clear error on
Windows / macOS.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nccl`]: https://docs.rs/baracuda-nccl
