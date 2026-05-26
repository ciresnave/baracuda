# baracuda-kernels

Unified Rust ML-op facade for the baracuda CUDA ecosystem.

One crate, one API style. Internally dispatches to whichever backend
(NVIDIA library wrapper or bespoke `.cu` kernel) is best for the
selected op × dtype × layout × architecture. The dispatch choice is
observable via `Plan::sku()` for telemetry but doesn't leak into the
call site.

For the high-level project pitch, the layered design, and the full
roadmap see the workspace
[`README.md`](../../README.md) and
[`ARCHITECTURE.md`](../../ARCHITECTURE.md).

## What this crate exposes

The full PyTorch (`torch.*` + `nn.functional`) and JAX (`jax.lax.*` +
`jax.numpy.*`) op union as `Plan`-based Rust types. As of alpha.25 the
following families are wired with forward + backward kernels across the
expected dtype matrices:

- **GEMM** — `GemmPlan`, `BatchedGemmPlan`, `GroupedGemmPlan`,
  `IntGemmPlan`, `Fp8GemmPlan`, `Int4GemmPlan`, `BinGemmPlan`.
- **Elementwise** — `UnaryPlan`, `BinaryPlan`, `TernaryPlan`,
  `BinaryCmpPlan`, `WherePlan`, `AffinePlan`, `CastPlan`,
  `GatedActivationPlan`, `UnaryParamPlan` / `BinaryParamPlan` (PReLU,
  Lerp, Threshold, …), all with paired `*BackwardPlan`.
- **Shape / layout** — `ConcatPlan`, `PadPlan`, `RepeatPlan`,
  `RollPlan`, `FlipPlan`, `PermutePlan`, `FillPlan`, all with BW.
- **Reductions** — `ReducePlan`, `ArgReducePlan`, `BoolReducePlan`,
  `CountReducePlan`, `TracePlan`.
- **Scans** — `ScanPlan` (cumsum, cumprod, cummax, cummin, logcumsumexp).
- **Softmax family** — `SoftmaxPlan`, `GumbelSoftmaxPlan`,
  `SparsemaxPlan`, all with BW.
- **Normalization** — `RMSNormPlan`, `LayerNormPlan`, `BatchNormPlan`,
  `GroupNormPlan`, `InstanceNormPlan`, all with BW.
- **Loss** — MSE / L1 / Huber / SmoothL1 / NLL / CrossEntropy / BCE /
  BCEWithLogits / KLDiv / GaussianNLL / PoissonNLL / Cosine /
  HingeEmbedding / MarginRanking / MultiMargin / MultilabelMargin /
  MultilabelSoftMargin / TripletMargin / CTCLoss.
- **Random** — `RandomPlan`, `DropoutPlan`.
- **Attention** — `SdpaPlan`, `FlashSdpaPlan`, `FlashSdpaSm89Plan`
  (sm_89 sibling), `RopePlan`, `AlibiPlan`, `KvCacheAppendPlan`.
- **Linalg** — `CholeskyPlan`, `LuPlan`, `QrPlan`, `BatchedQrPlan`,
  `SvdPlan`, `BatchedSvdPlan`, `BatchedSvdaPlan`, `EigPlan`, `EighPlan`,
  `InversePlan`, `SolvePlan`, `LstSqPlan`, `BatchedOrmqrPlan` /
  `BatchedOrmqrWyPlan`, `BatchedQrMaterializePlan`.
- **Convolution + Pooling** (cuDNN-backed; `cudnn` feature) —
  `Conv2dPlan`, `MaxPool2dPlan`, `AvgPool2dPlan`.
- **FFT** — `FftPlan`, `RfftPlan`, `IrfftPlan`, `FftNdPlan`,
  `RfftNdPlan`, `IrfftNdPlan`, `FftShiftPlan`, `FftShiftNdPlan`.
- **Indexing** — `GatherPlan`, `ScatterAddPlan`, `IndexSelectPlan`,
  `MaskedFillPlan`, `OneHotPlan`, `NonzeroPlan`.
- **Embedding** — `EmbeddingPlan`, `EmbeddingBagPlan`.
- **Segment ops** — sorted + unsorted variants of segment sum / mean /
  max / min / prod.
- **Quantization** — per-tensor / per-channel / per-token / per-group
  quantize + dequantize, `FakeQuantizePlan`, `DynamicRangeQuantizePlan`,
  `QuantizedLinearPlan`, plus GGUF block-format dequant + MMVQ for
  Q4_0..Q8_K + k-quants.
- **MoE** — fused per-token-dispatch + expert-matmul + accumulate
  (`MoePlan` with `MoeVariant::Wmma`, `ScalarGguf`, `WmmaGguf`).
- **Image** — `InterpolatePlan`, `GridSamplePlan`, `AffineGridPlan`,
  `PixelShufflePlan`, `RoiAlignPlan`, `RoiPoolPlan`, `NmsPlan`.
- **Sort / topk** — `SortPlan`, `ArgsortPlan`, `TopkPlan`,
  `KthvaluePlan`, `MsortPlan`, `SearchsortedPlan`, `BincountPlan`,
  `HistogramPlan`, `UniquePlan`, `UniqueConsecutivePlan`.

The shared vocabulary (`KernelDtype`, `Element`, `TensorRef`,
`KernelSku`, `PlanPreference`, `Workspace`, every op-kind enum) is
re-exported from `baracuda-kernels-types` so callers import one crate.

## `Element` vs `KernelDtype` — which to bound on

[`KernelDtype`] is the **umbrella marker** for every kernel-usable
dtype, including the sub-byte / FP8 / packed-bit newtypes (`S4`,
`U4`, `S8`, `U8`, `Fp8E4M3`, `Fp8E5M2`, `Bin`) that have their own
kernel families. The op-shaped sub-traits (`Element`, `IntElement`,
`FpElement`, `BinElement`) all extend `KernelDtype`, so a function
bounded by `<T: KernelDtype>` accepts any kernel-usable type.

In practice plans bound on `Element`, `IntElement`, `FpElement`, or
`BinElement` — whichever family the plan's kernel set fits —
because they parameterize the plan shape. Reach for the umbrella
`KernelDtype` bound only when the receiver needs to handle the
**union** of every dtype (a generic dtype-size helper, a telemetry
function, a wrapper crate downstream).

See [`baracuda-kernels-types`](../baracuda-kernels-types/README.md)
for the full trait map and the per-trait dtype list.

## `#[non_exhaustive]` and forward-compat

Phase 28 marked the op-family discriminant enums and several
auxiliary tag enums `#[non_exhaustive]` in preparation for the 1.0
freeze. Downstream `match` arms must include a `_ =>` catch-all —
adding new op variants in future phases then no longer breaks the
build. `ElementKind`, `LayoutSku`, `ArchSku`, `EpilogueKind`,
`ActivationKind`, and `Workspace<'a>` are intentionally left
exhaustive because they're hot-path-matched by the kernel
dispatchers; new variants there are a deliberate breaking-change
event. See the
[`baracuda-kernels-types` README](../baracuda-kernels-types/README.md)
for the full classification.

## Quick start

```rust,no_run
use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    EpilogueKind, IntGemmArgs, IntGemmDescriptor, IntGemmPlan,
    LayoutSku, MatrixMut, MatrixRef, PlanPreference, S8, Workspace,
};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new(&Device::get(0)?)?;
    let stream = Stream::new(&ctx)?;

    let m = 128i32; let n = 128i32; let k = 128i32;
    let dev_a: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, (m * k) as usize)?;
    let dev_b: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, (k * n) as usize)?;
    let mut dev_d: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, (m * n) as usize)?;

    // Rrr dispatches to bespoke int8 kernels in baracuda-kernels-sys.
    // Switching to Rcr would dispatch the same call through CUTLASS.
    let desc = IntGemmDescriptor {
        m, n, k,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = IntGemmPlan::<S8>::select(&stream, &desc, PlanPreference::default())?;

    let args = IntGemmArgs::<S8, f32> {
        a: MatrixRef { data: dev_a.as_slice(), rows: m, cols: k, ld: k as i64 },
        b: MatrixRef { data: dev_b.as_slice(), rows: k, cols: n, ld: n as i64 },
        c: None,
        d: MatrixMut { data: dev_d.as_slice_mut(), rows: m, cols: n, ld: n as i64 },
        bias: None,
        alpha: 0.125,
        beta: 0.0,
    };
    plan.run(&stream, Workspace::None, args)?;
    stream.synchronize()?;
    Ok(())
}
```

The same lifecycle — `Descriptor → Plan::select → query_workspace_size →
Args → Plan::run` — applies to every op family in the crate. See
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) for the design rationale
behind the triple.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `sm80` | yes | Build the Ampere-baseline kernel set. Runs forward-compatibly on Ada and Hopper. |
| `sm89` | no | Build the Ada Lovelace specializations: FP8 GEMM, `FlashSdpaSm89Plan`. |
| `sm90a` | no | Build the Hopper-specialized kernels (stubs today). |
| `cudnn` | no | Link cuDNN and enable `Conv2dPlan`, `MaxPool2dPlan`, `AvgPool2dPlan`, `CtcLossCudnnPlan`. |

`cudnn` is off by default because cuDNN is a separate NVIDIA download
not bundled with the stock CUDA toolkit installer. See the workspace
[`README.md`](../../README.md) for the auto-discovery paths the build
script probes.

## Verifying the API surface compiles

```bash
cargo check -p baracuda-kernels --features sm89,cudnn
```

The GPU integration tests are gated behind `#[ignore]`; run them with
`cargo test -p baracuda-kernels --release -- --ignored` on a host with
a working NVIDIA driver. The full regression covers ~1630 tests on an
RTX 4070.

## See also

- [`baracuda-kernels-types`](../baracuda-kernels-types) — the shared
  type vocabulary.
- [`baracuda-kernels-sys`](../baracuda-kernels-sys) — raw FFI to the
  bespoke `.cu` kernels behind this facade.
- [`baracuda-cutlass`](../baracuda-cutlass) — CUTLASS plan types
  re-exported here unchanged.
- [`baracuda-kernels-bench`](../baracuda-kernels-bench) — criterion
  bench harness for sm_89 perf sweeps.
