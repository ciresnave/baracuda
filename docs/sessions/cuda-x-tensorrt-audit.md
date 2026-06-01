# Session prompt — Audit `baracuda-tensorrt` safe wrapper completeness

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
This is an audit-and-fill-gaps task (not a from-scratch addition).
Other parallel sessions may be running.

## Context

baracuda has `baracuda-tensorrt-sys` + `baracuda-tensorrt` crates,
created at some prior phase. TensorRT is NVIDIA's inference compiler
that takes a network definition (built up via the TensorRT API or
imported from ONNX) and produces an optimized engine for a specific
GPU + dtype combination. It's central to production LLM inference.

Status of baracuda's TensorRT wrappers is unclear. They exist as
crates but the safe-wrapper completeness against the full TensorRT
inference surface needs verification. This session audits + fills the
gaps.

## Audit checklist

For each of the following TensorRT capability areas, verify
`baracuda-tensorrt` exposes a usable safe wrapper. For gaps, add the
wrapper.

**Engine build path:**
- `INetworkDefinition` construction (layer-by-layer API)
- `IBuilder` + `IBuilderConfig` (precision flags, workspace size, etc.)
- ONNX import via `IParser` (most callers will use this)
- INT8 calibration (`IInt8Calibrator2` / `IInt8EntropyCalibrator2`)
- Plugin registration (`IPluginRegistry`, `IPluginV2Layer`)

**Engine run path:**
- `IExecutionContext` (per-stream inference instance)
- Binding (input/output tensor pointer + shape setup)
- `enqueueV3` (the modern async inference call)
- Multi-stream concurrent execution

**Memory + tooling:**
- `ICudaEngine` serialization / deserialization (save engine to disk)
- Optimization profile (dynamic shapes)
- `setOutputType` / `setMemoryPoolLimit` / etc. (the dial knobs)
- Runtime version + plugin compatibility checks

## Methodology

1. Read `crates/baracuda-tensorrt/src/lib.rs` and any nested modules.
2. Read `crates/baracuda-tensorrt-sys/src/lib.rs` for the FFI surface.
3. For each capability above, determine:
   - **Exposed**: passes through to working FFI, has a smoke test if applicable.
   - **Partial**: FFI exists, no safe wrapper, or safe wrapper exists but skips some args.
   - **Missing**: no exposure at all.
4. Produce an inventory table + propose fixes for partial/missing.
5. Implement Tier-1 fixes (the inference-runtime path: engine deserialize → execute → bindings). Defer Tier-2 (calibration, plugins) if scope grows.

## Tier 1 deliverables

If the audit finds the inference runtime path is incomplete:

1. Complete `ICudaEngine::deserialize_engine` safe wrapper
2. Complete `IExecutionContext` lifecycle + `enqueueV3`
3. Complete tensor binding (set_tensor_address for each input/output)
4. Smoke test deserializing a small precompiled engine + running 1
   inference (use a stub engine for testing or a minimal hand-built
   one).

If the inference runtime path is complete:

1. Add the engine BUILD path Tier-1 (network definition + builder
   config + ONNX parser).
2. Tests for the build path.

## Reference patterns

Look at `baracuda-cudnn/` for a similar "complex C++ API wrapped in
safe Rust" pattern. cuDNN's descriptor + plan + execute lifecycle
mirrors TensorRT's network + engine + context lifecycle.

## Coordination with baracuda-kernels

TensorRT engines run on a CUDA stream provided by the caller. The
existing baracuda-driver `Stream` type should be passable. Verify the
binding wrapper accepts `&Stream` not raw `cudaStream_t`.

## Out of scope

- Don't add TensorRT plugin authoring tools (writing custom CUDA
  kernels as TensorRT plugins). That's a separate phase.
- Don't add TensorRT-LLM (the LLM-specific extensions). Separate
  crate.
- Don't add ONNX runtime integration. TensorRT-only.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase68-tensorrt-audit`
- No version bump, no publish.
- Commit on branch + push + stop.

## Stop conditions

- If the audit reveals the existing `baracuda-tensorrt` is essentially
  empty (stub crate only): report and ask Eric whether to do a full
  ground-up build in this session or defer.
- If TensorRT linking is broken on the dev machine (libnvinfer.so
  missing): document the install path requirement, mark tests
  `#[ignore]` for environments without TensorRT, proceed with what
  you can.
- If you find recent baracuda commits already addressing the audit
  (someone got here first): stop, report.

## Memory file

Write `project_phase68_complete.md` summarizing the audit + what was
filled + what was deferred.
