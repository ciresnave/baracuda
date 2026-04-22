# Streams and events

Streams let CUDA overlap independent work on the GPU. Events let the host
or a stream wait on a specific point in another stream's history. Together,
they let you build pipelines that look like this:

```text
stream 0: [H2D chunk 0] [H2D chunk 1]               [H2D chunk 2]
stream 1:                [compute 0] [compute 1]                  [compute 2]
stream 2:                            [D2H chunk 0] [D2H chunk 1]               [D2H chunk 2]
```

— three stages, pipelined across streams, saturating the PCIe bus while
the SMs stay busy.

## Basic usage

```rust
use baracuda::driver::{Context, Device, DeviceBuffer, Stream};

let device = Device::get(0)?;
let ctx = Context::new(&device)?;

let s = Stream::new(&ctx)?;       // default: blocking, normal priority
let hi = Stream::new_with_priority(&ctx, -1)?;  // -1 = higher priority
```

Every cuBLAS / cuDNN / cuFFT handle accepts `.set_stream(&stream)`; once set,
every subsequent op issues on that stream. Reset with a different stream or
re-use across cooperating handles.

## Events: "has X happened yet?"

```rust
use baracuda::driver::Event;

let start = Event::new(&ctx)?;
let stop  = Event::new(&ctx)?;

start.record(&stream)?;
kernel.launch()...launch()?;
stop.record(&stream)?;

stop.synchronize()?;  // block host until stop has been reached
println!("kernel took {} ms", start.elapsed_ms(&stop)?);
```

Events are the standard way to time GPU work. `Event::new_disable_timing` is
cheaper but loses the `elapsed_ms` capability — use it for pure
synchronization.

## Cross-stream dependencies

```rust
let h2d = Stream::new(&ctx)?;
let compute = Stream::new(&ctx)?;

let done_copying = Event::new(&ctx)?;

// Stream 1: H2D
d_input.copy_from_host_async(&host, &h2d)?;
done_copying.record(&h2d)?;

// Stream 2: compute — waits for the copy to finish
compute.wait_event(&done_copying)?;
kernel.launch()...stream(&compute).launch()?;
```

`wait_event` is non-blocking from the host's perspective — it inserts a
dependency in the submitted command stream.

## Three-stage pipeline

The canonical pattern. Assume you're iterating over chunks of a large
dataset:

```rust
let n_chunks = 8;
let streams: Vec<_> = (0..2).map(|_| Stream::new(&ctx).unwrap()).collect();
let events:  Vec<_> = (0..2).map(|_| Event::new(&ctx).unwrap()).collect();

for chunk in 0..n_chunks {
    let slot = chunk % 2;  // double-buffer
    let s = &streams[slot];
    let e = &events[slot];

    // wait for the previous round using this slot to finish
    if chunk >= 2 { e.synchronize()?; }

    // stage 1: H2D
    d_in.copy_from_host_async(&host_chunks[chunk], s)?;

    // stage 2: compute
    kernel.launch()...stream(s).launch()?;

    // stage 3: D2H
    d_out.copy_to_host_async(&mut host_out[chunk], s)?;

    e.record(s)?;
}
for e in &events { e.synchronize()?; }
```

Double-buffering with two streams is a sweet spot: enough concurrency to
overlap transfer with compute, without the overhead of more context switches.

## Per-thread default streams

Classical CUDA has a single "default stream" that synchronizes with *all*
other streams. That's almost always not what you want. baracuda defaults to
**per-thread default streams** (the `_ptsz` variants). You can override at
process start:

```rust
use baracuda_core::stream_mode::{init, StreamMode};

init(StreamMode::Legacy);  // one global default stream
```

Call this *before* any baracuda API — it controls symbol resolution in the
Driver loader.

## `async` support

Behind the `async` feature, `Event::wait_future()` returns a `Future` that
resolves when the event fires. Implementation uses `cuLaunchHostFunc` to wake
a waker — no polling thread, no tokio dependency.

```rust
#[cfg(feature = "async")]
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let stream = Stream::new(&ctx)?;
    kernel.launch()...stream(&stream).launch()?;
    let done = Event::new(&ctx)?;
    done.record(&stream)?;
    done.wait_future().await?;
    Ok(())
}
```

## Debugging tips

- **"kernel appears to take 0 ns"** — you forgot the `stop.synchronize()`
  (or equivalent) before reading the elapsed time.
- **"`wait_event` doesn't block"** — correct, it's lazy. The wait is enforced
  when the dependent op runs on the GPU, not at host submission time.
- **`Event::elapsed_ms` returns `NOT_READY`** — you called it before the event
  was reached. Either sync first or use `Event::query`.
