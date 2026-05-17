//! Real-GPU smoke test for `KvCacheAppendPlan + AttentionKind::KvCache`.
//!
//! Pure copy → bit-exact (`to_bits()` equality) across every wired dtype.
//!
//! Three scenarios per dtype:
//!   1. append at offset 0 (initial cache fill)
//!   2. append at a non-zero uniform offset (subsequent decode step)
//!   3. ragged offsets — different `cache_offsets[b]` per sample
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, KvCacheAppendArgs, KvCacheAppendDescriptor,
    KvCacheAppendPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// -------------------------------------------------------------------------
// Shared CPU reference
// -------------------------------------------------------------------------
//
// Bit-copies `T` cells from `k_new` / `v_new` (host) into the
// corresponding row of `k_cache` / `v_cache` (host) for each sample's
// offset. Mirrors the kernel exactly — out-of-range cells are skipped.

fn cpu_kv_cache_append<T: Copy>(
    batch: usize,
    heads: usize,
    new_len: usize,
    max_cache_len: usize,
    d: usize,
    new_buf: &[T],
    cache_buf: &mut [T],
    offsets: &[i64],
) {
    for b in 0..batch {
        let off = offsets[b];
        for h in 0..heads {
            for l in 0..new_len {
                let cache_pos = off + l as i64;
                if cache_pos < 0 || cache_pos >= max_cache_len as i64 {
                    continue;
                }
                let src_base = ((b * heads + h) * new_len + l) * d;
                let dst_base =
                    ((b * heads + h) * max_cache_len + cache_pos as usize) * d;
                for di in 0..d {
                    cache_buf[dst_base + di] = new_buf[src_base + di];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Dtype-erased runner — takes f32 raw values, converts to the target
// dtype, runs the plan, downloads, and compares bit-exactly to the host
// reference (also produced from the dtype-converted host buffers).
// -------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
struct Shape {
    batch: i32,
    heads: i32,
    new_len: i32,
    max_cache_len: i32,
    d_k: i32,
    d_v: i32,
}

fn run_case<T>(
    shape: Shape,
    offsets: &[i64],
    k_new_f32: &[f32],
    v_new_f32: &[f32],
    k_cache_init_f32: &[f32],
    v_cache_init_f32: &[f32],
    to_t: impl Fn(f32) -> T,
    kind: ElementKind,
) where
    T: baracuda_kernels::Element + Copy + core::fmt::Debug + PartialEq,
{
    let (ctx, stream) = setup();
    let b = shape.batch as usize;
    let h = shape.heads as usize;
    let l_new = shape.new_len as usize;
    let l_max = shape.max_cache_len as usize;
    let d_k = shape.d_k as usize;
    let d_v = shape.d_v as usize;

    // Host buffers in target dtype
    let k_new_t: Vec<T> = k_new_f32.iter().map(|&v| to_t(v)).collect();
    let v_new_t: Vec<T> = v_new_f32.iter().map(|&v| to_t(v)).collect();
    let k_cache_init_t: Vec<T> =
        k_cache_init_f32.iter().map(|&v| to_t(v)).collect();
    let v_cache_init_t: Vec<T> =
        v_cache_init_f32.iter().map(|&v| to_t(v)).collect();

    // Build expected cache via host reference
    let mut expected_k = k_cache_init_t.clone();
    let mut expected_v = v_cache_init_t.clone();
    cpu_kv_cache_append(
        b, h, l_new, l_max, d_k, &k_new_t, &mut expected_k, offsets,
    );
    cpu_kv_cache_append(
        b, h, l_new, l_max, d_v, &v_new_t, &mut expected_v, offsets,
    );

    let dev_k_new = DeviceBuffer::from_slice(&ctx, &k_new_t).expect("up k_new");
    let dev_v_new = DeviceBuffer::from_slice(&ctx, &v_new_t).expect("up v_new");
    let dev_offsets =
        DeviceBuffer::from_slice(&ctx, offsets).expect("up offsets");
    let mut dev_k_cache =
        DeviceBuffer::from_slice(&ctx, &k_cache_init_t).expect("up k_cache");
    let mut dev_v_cache =
        DeviceBuffer::from_slice(&ctx, &v_cache_init_t).expect("up v_cache");

    let desc = KvCacheAppendDescriptor {
        batch_size: shape.batch,
        num_heads: shape.heads,
        new_len: shape.new_len,
        max_cache_len: shape.max_cache_len,
        d_k: shape.d_k,
        d_v: shape.d_v,
        element: kind,
    };
    let plan =
        KvCacheAppendPlan::<T>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let shape_k_new = [shape.batch, shape.heads, shape.new_len, shape.d_k];
    let shape_v_new = [shape.batch, shape.heads, shape.new_len, shape.d_v];
    let shape_k_cache =
        [shape.batch, shape.heads, shape.max_cache_len, shape.d_k];
    let shape_v_cache =
        [shape.batch, shape.heads, shape.max_cache_len, shape.d_v];

    plan.run(
        &stream,
        Workspace::None,
        KvCacheAppendArgs {
            k_new: TensorRef {
                data: dev_k_new.as_slice(),
                shape: shape_k_new,
                stride: contiguous_stride(shape_k_new),
            },
            v_new: TensorRef {
                data: dev_v_new.as_slice(),
                shape: shape_v_new,
                stride: contiguous_stride(shape_v_new),
            },
            cache_offsets: TensorRef {
                data: dev_offsets.as_slice(),
                shape: [shape.batch],
                stride: [1],
            },
            k_cache: TensorMut {
                data: dev_k_cache.as_slice_mut(),
                shape: shape_k_cache,
                stride: contiguous_stride(shape_k_cache),
            },
            v_cache: TensorMut {
                data: dev_v_cache.as_slice_mut(),
                shape: shape_v_cache,
                stride: contiguous_stride(shape_v_cache),
            },
        },
    )
    .expect("run");
    stream.synchronize().expect("sync");

    let mut got_k = vec![to_t(0.0); b * h * l_max * d_k];
    let mut got_v = vec![to_t(0.0); b * h * l_max * d_v];
    dev_k_cache.copy_to_host(&mut got_k).expect("dl k");
    dev_v_cache.copy_to_host(&mut got_v).expect("dl v");

    // Bit-exact — pure copy.
    for (i, (g, e)) in got_k.iter().zip(expected_k.iter()).enumerate() {
        assert_eq!(g, e, "k_cache mismatch at flat index {i}");
    }
    for (i, (g, e)) in got_v.iter().zip(expected_v.iter()).enumerate() {
        assert_eq!(g, e, "v_cache mismatch at flat index {i}");
    }
}

// -------------------------------------------------------------------------
// Per-dtype dispatch and three offset patterns
// -------------------------------------------------------------------------

fn build_inputs(
    shape: Shape,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_new_k = (shape.batch * shape.heads * shape.new_len * shape.d_k) as usize;
    let n_new_v = (shape.batch * shape.heads * shape.new_len * shape.d_v) as usize;
    let n_cache_k =
        (shape.batch * shape.heads * shape.max_cache_len * shape.d_k) as usize;
    let n_cache_v =
        (shape.batch * shape.heads * shape.max_cache_len * shape.d_v) as usize;
    // Synthetic patterns with distinct ranges so a stale slot is
    // visibly wrong if the kernel writes the wrong destination.
    let k_new: Vec<f32> = (0..n_new_k).map(|i| 1.0 + (i as f32) * 0.0625).collect();
    let v_new: Vec<f32> = (0..n_new_v)
        .map(|i| -1.0 - (i as f32) * 0.0625)
        .collect();
    let k_init: Vec<f32> = (0..n_cache_k)
        .map(|i| 100.0 + (i as f32) * 0.5)
        .collect();
    let v_init: Vec<f32> = (0..n_cache_v)
        .map(|i| -100.0 - (i as f32) * 0.5)
        .collect();
    (k_new, v_new, k_init, v_init)
}

fn run_three_cases<T>(
    to_t: impl Fn(f32) -> T + Copy,
    kind: ElementKind,
) where
    T: baracuda_kernels::Element + Copy + core::fmt::Debug + PartialEq,
{
    // Common geometry. d_k != d_v on purpose.
    let shape = Shape {
        batch: 3,
        heads: 4,
        new_len: 5,
        max_cache_len: 16,
        d_k: 8,
        d_v: 6,
    };
    let (k_new, v_new, k_init, v_init) = build_inputs(shape);

    // Case 1: offset 0 (initial fill).
    let offsets_zero = vec![0i64; shape.batch as usize];
    run_case::<T>(
        shape,
        &offsets_zero,
        &k_new,
        &v_new,
        &k_init,
        &v_init,
        to_t,
        kind,
    );

    // Case 2: same non-zero offset for every sample (mid-decode).
    let offsets_mid = vec![7i64; shape.batch as usize];
    run_case::<T>(
        shape,
        &offsets_mid,
        &k_new,
        &v_new,
        &k_init,
        &v_init,
        to_t,
        kind,
    );

    // Case 3: ragged offsets — every sample lands at a different
    // position. Last sample is set such that the very last new-row
    // falls just beyond max_cache_len, exercising the boundary skip.
    let offsets_ragged = vec![1i64, 9i64, 12i64]; // 12 + 5 = 17 > 16 → tail skip
    assert_eq!(offsets_ragged.len(), shape.batch as usize);
    run_case::<T>(
        shape,
        &offsets_ragged,
        &k_new,
        &v_new,
        &k_init,
        &v_init,
        to_t,
        kind,
    );
}

#[test]
#[ignore]
fn kv_cache_append_f32() {
    run_three_cases::<f32>(|v| v, ElementKind::F32);
}

#[test]
#[ignore]
fn kv_cache_append_f16() {
    run_three_cases::<f16>(f16::from_f32, ElementKind::F16);
}

#[test]
#[ignore]
fn kv_cache_append_bf16() {
    run_three_cases::<bf16>(bf16::from_f32, ElementKind::Bf16);
}

#[test]
#[ignore]
fn kv_cache_append_f64() {
    run_three_cases::<f64>(|v| v as f64, ElementKind::F64);
}
