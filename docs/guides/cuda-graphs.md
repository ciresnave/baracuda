# CUDA Graphs — capture once, replay forever

CUDA Graphs let you record a sequence of operations (kernel launches,
memcpys, library calls) as a DAG, then launch that DAG as a single unit on
subsequent iterations. The launch overhead drops from one driver call per op
to one driver call per iteration. For training loops and inference kernels,
the savings are often 2–10× on small-op-count workloads.

## Two ways to build a graph

1. **Stream capture** — record everything a stream does between `begin` and
   `end` calls. Works for any existing CUDA code; you don't have to learn the
   graph-building API.
2. **Explicit construction** — build nodes one at a time with
   `Graph::add_kernel_node`, `add_memcpy_node`, etc. More boilerplate, but
   fully deterministic.

baracuda exposes both. Stream capture is the common case.

## Stream capture example

```rust
use baracuda::driver::{Context, Device, DeviceBuffer, Graph, GraphExec, Stream};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;
    let kernel = load_my_kernel(&ctx)?;

    let mut input = DeviceBuffer::<f32>::zeros(&ctx, 1_000_000)?;
    let mut temp  = DeviceBuffer::<f32>::zeros(&ctx, 1_000_000)?;
    let mut out   = DeviceBuffer::<f32>::zeros(&ctx, 1_000_000)?;

    // Capture: every op on `stream` between begin/end lands in the graph.
    stream.begin_capture()?;
    kernel.launch()/* ... */.stream(&stream).launch()?;
    kernel.launch()/* ... */.stream(&stream).launch()?;
    temp.copy_to_device_async(&out, &stream)?;
    let graph: Graph = stream.end_capture()?;

    // Instantiate once, then replay on every iteration.
    let exec: GraphExec = graph.instantiate()?;

    for _ in 0..1_000 {
        exec.launch(&stream)?;
    }
    stream.synchronize()?;
    Ok(())
}
```

**Gotcha:** captured streams can't call synchronous APIs (including
`stream.synchronize()`) during capture. If you need a dependency between
captured ops, use events.

## Updating a graph between launches

A key feature: `GraphExec::update` swaps in new parameters (grid dims, kernel
args, memcpy sources) without rebuilding the graph:

```rust
let mut updated = graph.clone();
let new_kernel_node = updated.find_kernel_node("my_kernel")?;
new_kernel_node.set_args(&[&new_input, &new_output, &new_n])?;
exec.update(&updated)?;
exec.launch(&stream)?;
```

This avoids the instantiation cost when only parameters change.

## Conditional nodes (CUDA 12.3+)

Conditional and switch nodes let a graph branch based on a device-side
flag. baracuda exposes them behind a feature gate — check
[`Feature::GraphConditionalNodes`](../design/loader.md#feature-negotiation).

```rust
#[cfg(feature = "graph-conditional")]
{
    let cond = graph.add_conditional_node(&condition_flag, ConditionalKind::If)?;
    // ... add child nodes inside cond
}
```

At runtime, `supports(CudaVersion::detect(), Feature::GraphConditionalNodes)`
tells you whether the driver actually has them.

## When to use graphs

Graphs pay off when:

- You issue a lot of small ops per iteration (< 1 ms each).
- The op sequence repeats without changing topology.
- You can tolerate a one-time instantiation cost (hundreds of μs).

They don't pay off when:

- You launch one huge kernel per iteration.
- The op sequence changes every iteration.
- Host-side decisions drive which ops to run.

## Comparing with stream pipelines

Streams + events give you overlap of independent work across stages; graphs
give you lower launch overhead for a fixed sequence. They compose: you can
submit multiple `GraphExec` instances to different streams to get both
benefits.

## Further reading

- NVIDIA: [CUDA Graphs programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs).
- The `baracuda::driver::graph` module documents each node kind.
- [`examples/graph_capture.rs`](../../examples/graph_capture.rs) for a full
  runnable example.
