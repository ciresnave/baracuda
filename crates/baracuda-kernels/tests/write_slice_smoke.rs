//! Real-GPU smoke tests for the WriteSlice trailblazer
//! (`WriteSlicePlan<T, N>`).
//!
//! Coverage:
//!   * KV-cache append shape (rank-3, full-width minor axes, single-
//!     row slab) on f32, f16, bf16, f64 — exercises the
//!     `ContiguousChunk` fast path (single `cuMemcpyDtoDAsync`).
//!   * Interior 2-D slab (rank-2, both axes partial) on f32 —
//!     exercises the generic per-slab-element kernel path.
//!   * 1-D slab (rank-1) on i32 — minimal sanity.
//!   * Nibble-packed (rank-2 on S4, even alignment) — exercises the
//!     nibble kernel symbol.
//!   * Negative test: slab out of bounds → `select` rejects.
//!   * Negative test: nibble innermost odd start → `select` rejects.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm80 \
//!   --test write_slice_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, S4, TensorMut, TensorRef, Workspace,
    WriteSliceArgs, WriteSliceDescriptor, WriteSlicePlan,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host reference: write `source` into the per-axis range window of
/// `dest`. Assumes both tensors are contiguous row-major.
fn cpu_write_slice<const N: usize>(
    dest_init: &[u8],
    dest_shape: [i32; N],
    source: &[u8],
    source_shape: [i32; N],
    ranges: [(i32, i32); N],
    byte_width: usize,
) -> Vec<u8> {
    let mut out = dest_init.to_vec();
    let source_numel: usize = source_shape.iter().map(|&d| d as usize).product();
    if source_numel == 0 {
        return out;
    }
    // Compute row-major strides in element units.
    let mut dest_strides = [1i64; N];
    let mut source_strides = [1i64; N];
    if N > 0 {
        for d in (0..N - 1).rev() {
            dest_strides[d] = dest_strides[d + 1] * dest_shape[d + 1] as i64;
            source_strides[d] = source_strides[d + 1] * source_shape[d + 1] as i64;
        }
    }
    for i in 0..source_numel {
        let mut linear = i as i64;
        let mut coord = [0i64; N];
        for d in (0..N).rev() {
            let s = source_shape[d] as i64;
            if s == 0 {
                coord[d] = 0;
            } else {
                coord[d] = linear % s;
                linear /= s;
            }
        }
        let mut dest_off: i64 = 0;
        let mut source_off: i64 = 0;
        for d in 0..N {
            dest_off += (coord[d] + ranges[d].0 as i64) * dest_strides[d];
            source_off += coord[d] * source_strides[d];
        }
        // Byte-level copy.
        let dst_byte = dest_off as usize * byte_width;
        let src_byte = source_off as usize * byte_width;
        for b in 0..byte_width {
            out[dst_byte + b] = source[src_byte + b];
        }
    }
    out
}

/// KV-cache append: rank-3 dest [batch, max_seq, head_dim], slab
/// inserted at sequence position `t` covers a single row across the
/// full head_dim and full batch — but the layout we care about is
/// [max_seq, batch, head_dim] so the slab on axis 0 covers minors
/// fully (one row), giving the `ContiguousChunk` fast path. f32.
#[test]
#[ignore]
fn write_slice_kv_cache_append_f32() {
    let (ctx, stream) = setup();
    let dest_shape = [32i32, 4, 64]; // [max_seq, batch, head_dim]
    let source_shape = [1i32, 4, 64]; // append at position t=7
    let ranges = [(7, 8), (0, 4), (0, 64)];
    let byte_width = 4;

    let dest_numel = (dest_shape[0] * dest_shape[1] * dest_shape[2]) as usize;
    let source_numel = (source_shape[0] * source_shape[1] * source_shape[2]) as usize;

    let dest_init: Vec<f32> = (0..dest_numel).map(|i| (i as f32) * 0.001).collect();
    let source: Vec<f32> = (0..source_numel).map(|i| (i as f32) + 1000.0).collect();

    let dest_bytes = bytemuck_slice(&dest_init);
    let source_bytes = bytemuck_slice(&source);
    let expected_bytes = cpu_write_slice::<3>(
        &dest_bytes, dest_shape, &source_bytes, source_shape, ranges, byte_width,
    );

    let mut dev_dest = DeviceBuffer::from_slice(&ctx, &dest_init).expect("upload dest");
    let dev_source = DeviceBuffer::from_slice(&ctx, &source).expect("upload source");

    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::F32,
    };
    let plan = WriteSlicePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = WriteSliceArgs::<f32, 3> {
        dest: TensorMut {
            data: dev_dest.as_slice_mut(),
            shape: dest_shape,
            stride: contiguous_stride(dest_shape),
        },
        source: TensorRef {
            data: dev_source.as_slice(),
            shape: source_shape,
            stride: contiguous_stride(source_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dest_numel];
    dev_dest.copy_to_host(&mut got).expect("download");
    let got_bytes = bytemuck_slice(&got);
    for i in 0..expected_bytes.len() {
        assert_eq!(
            got_bytes[i], expected_bytes[i],
            "kv_cache_append_f32 mismatch @ byte {i}"
        );
    }
}

/// Same KV-cache append shape, f16. Exercises the b2 byte-width FFI
/// symbol via the fast path.
#[test]
#[ignore]
fn write_slice_kv_cache_append_f16() {
    let (ctx, stream) = setup();
    let dest_shape = [16i32, 2, 32];
    let source_shape = [1i32, 2, 32];
    let ranges = [(3, 4), (0, 2), (0, 32)];

    let dest_numel = (dest_shape[0] * dest_shape[1] * dest_shape[2]) as usize;
    let source_numel = (source_shape[0] * source_shape[1] * source_shape[2]) as usize;

    let dest_init: Vec<f16> = (0..dest_numel)
        .map(|i| f16::from_f32(i as f32 * 0.01))
        .collect();
    let source: Vec<f16> = (0..source_numel)
        .map(|i| f16::from_f32(i as f32 + 50.0))
        .collect();

    let mut dev_dest = DeviceBuffer::from_slice(&ctx, &dest_init).expect("upload dest");
    let dev_source = DeviceBuffer::from_slice(&ctx, &source).expect("upload source");

    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::F16,
    };
    let plan = WriteSlicePlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = WriteSliceArgs::<f16, 3> {
        dest: TensorMut {
            data: dev_dest.as_slice_mut(),
            shape: dest_shape,
            stride: contiguous_stride(dest_shape),
        },
        source: TensorRef {
            data: dev_source.as_slice(),
            shape: source_shape,
            stride: contiguous_stride(source_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; dest_numel];
    dev_dest.copy_to_host(&mut got).expect("download");

    // Build expected on host.
    let mut expected = dest_init.clone();
    let dest_minor_stride = (dest_shape[1] * dest_shape[2]) as usize;
    let row_elems = source_numel;
    let row_off = ranges[0].0 as usize * dest_minor_stride;
    expected[row_off..row_off + row_elems].copy_from_slice(&source);
    for i in 0..dest_numel {
        assert_eq!(
            got[i].to_bits(), expected[i].to_bits(),
            "kv_cache_append_f16 mismatch @ {i}"
        );
    }
}

/// KV-cache append on bf16.
#[test]
#[ignore]
fn write_slice_kv_cache_append_bf16() {
    let (ctx, stream) = setup();
    let dest_shape = [12i32, 3, 16];
    let source_shape = [1i32, 3, 16];
    let ranges = [(5, 6), (0, 3), (0, 16)];

    let dest_numel = (dest_shape[0] * dest_shape[1] * dest_shape[2]) as usize;
    let source_numel = (source_shape[0] * source_shape[1] * source_shape[2]) as usize;

    let dest_init: Vec<bf16> = (0..dest_numel)
        .map(|i| bf16::from_f32(i as f32 * 0.5))
        .collect();
    let source: Vec<bf16> = (0..source_numel)
        .map(|i| bf16::from_f32(i as f32 + 200.0))
        .collect();

    let mut dev_dest = DeviceBuffer::from_slice(&ctx, &dest_init).expect("upload dest");
    let dev_source = DeviceBuffer::from_slice(&ctx, &source).expect("upload source");

    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::Bf16,
    };
    let plan = WriteSlicePlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = WriteSliceArgs::<bf16, 3> {
        dest: TensorMut {
            data: dev_dest.as_slice_mut(),
            shape: dest_shape,
            stride: contiguous_stride(dest_shape),
        },
        source: TensorRef {
            data: dev_source.as_slice(),
            shape: source_shape,
            stride: contiguous_stride(source_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; dest_numel];
    dev_dest.copy_to_host(&mut got).expect("download");
    let mut expected = dest_init.clone();
    let dest_minor_stride = (dest_shape[1] * dest_shape[2]) as usize;
    let row_off = ranges[0].0 as usize * dest_minor_stride;
    expected[row_off..row_off + source_numel].copy_from_slice(&source);
    for i in 0..dest_numel {
        assert_eq!(
            got[i].to_bits(), expected[i].to_bits(),
            "kv_cache_append_bf16 mismatch @ {i}"
        );
    }
}

/// KV-cache append on f64. Exercises the b8 byte-width FFI symbol.
#[test]
#[ignore]
fn write_slice_kv_cache_append_f64() {
    let (ctx, stream) = setup();
    let dest_shape = [8i32, 2, 16];
    let source_shape = [1i32, 2, 16];
    let ranges = [(2, 3), (0, 2), (0, 16)];

    let dest_numel = (dest_shape[0] * dest_shape[1] * dest_shape[2]) as usize;
    let source_numel = (source_shape[0] * source_shape[1] * source_shape[2]) as usize;

    let dest_init: Vec<f64> = (0..dest_numel).map(|i| (i as f64) * 0.001).collect();
    let source: Vec<f64> = (0..source_numel).map(|i| (i as f64) + 1.0e6).collect();

    let mut dev_dest = DeviceBuffer::from_slice(&ctx, &dest_init).expect("upload dest");
    let dev_source = DeviceBuffer::from_slice(&ctx, &source).expect("upload source");

    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::F64,
    };
    let plan = WriteSlicePlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = WriteSliceArgs::<f64, 3> {
        dest: TensorMut {
            data: dev_dest.as_slice_mut(),
            shape: dest_shape,
            stride: contiguous_stride(dest_shape),
        },
        source: TensorRef {
            data: dev_source.as_slice(),
            shape: source_shape,
            stride: contiguous_stride(source_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; dest_numel];
    dev_dest.copy_to_host(&mut got).expect("download");
    let mut expected = dest_init.clone();
    let dest_minor_stride = (dest_shape[1] * dest_shape[2]) as usize;
    let row_off = ranges[0].0 as usize * dest_minor_stride;
    expected[row_off..row_off + source_numel].copy_from_slice(&source);
    for i in 0..dest_numel {
        assert_eq!(
            got[i].to_bits(), expected[i].to_bits(),
            "kv_cache_append_f64 mismatch @ {i}"
        );
    }
}

/// Interior 2-D slab on f32 — both axes partial. This forces the
/// generic kernel path because no axis past the first is full-width.
#[test]
#[ignore]
fn write_slice_interior_2d_f32() {
    let (ctx, stream) = setup();
    let dest_shape = [16i32, 20];
    let source_shape = [5i32, 7];
    let ranges = [(4, 9), (8, 15)];

    let dest_numel = (dest_shape[0] * dest_shape[1]) as usize;
    let source_numel = (source_shape[0] * source_shape[1]) as usize;

    let dest_init: Vec<f32> = (0..dest_numel).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let source: Vec<f32> = (0..source_numel).map(|i| (i as f32) + 1000.0).collect();

    let dest_bytes = bytemuck_slice(&dest_init);
    let source_bytes = bytemuck_slice(&source);
    let expected_bytes = cpu_write_slice::<2>(
        &dest_bytes, dest_shape, &source_bytes, source_shape, ranges, 4,
    );

    let mut dev_dest = DeviceBuffer::from_slice(&ctx, &dest_init).expect("upload dest");
    let dev_source = DeviceBuffer::from_slice(&ctx, &source).expect("upload source");

    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::F32,
    };
    let plan = WriteSlicePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = WriteSliceArgs::<f32, 2> {
        dest: TensorMut {
            data: dev_dest.as_slice_mut(),
            shape: dest_shape,
            stride: contiguous_stride(dest_shape),
        },
        source: TensorRef {
            data: dev_source.as_slice(),
            shape: source_shape,
            stride: contiguous_stride(source_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dest_numel];
    dev_dest.copy_to_host(&mut got).expect("download");
    let got_bytes = bytemuck_slice(&got);
    for i in 0..expected_bytes.len() {
        assert_eq!(
            got_bytes[i], expected_bytes[i],
            "interior_2d_f32 mismatch @ byte {i}"
        );
    }
}

/// 1-D slab on i32 — minimal sanity. Rank-1 always hits the
/// `ContiguousChunk` fast path.
#[test]
#[ignore]
fn write_slice_1d_i32() {
    let (ctx, stream) = setup();
    let dest_shape = [32i32];
    let source_shape = [8i32];
    let ranges = [(10, 18)];

    let dest_init: Vec<i32> = (0..32).map(|i| i * 10).collect();
    let source: Vec<i32> = (0..8).map(|i| 999 - i).collect();

    let mut dev_dest = DeviceBuffer::from_slice(&ctx, &dest_init).expect("upload dest");
    let dev_source = DeviceBuffer::from_slice(&ctx, &source).expect("upload source");

    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::I32,
    };
    let plan = WriteSlicePlan::<i32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = WriteSliceArgs::<i32, 1> {
        dest: TensorMut {
            data: dev_dest.as_slice_mut(),
            shape: dest_shape,
            stride: contiguous_stride(dest_shape),
        },
        source: TensorRef {
            data: dev_source.as_slice(),
            shape: source_shape,
            stride: contiguous_stride(source_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; 32];
    dev_dest.copy_to_host(&mut got).expect("download");
    let mut expected = dest_init.clone();
    expected[10..18].copy_from_slice(&source);
    assert_eq!(got, expected);
}

/// Nibble-packed write on S4. Rank-2, even alignment on innermost
/// axis (all even). Exercises the dedicated nibble kernel.
#[test]
#[ignore]
fn write_slice_nibble_s4_rank2() {
    let (ctx, stream) = setup();
    // 4 rows × 8 elements (= 4 bytes per row). Storage = 4*4 = 16 bytes.
    let dest_shape = [4i32, 8];
    // Slab is rows [1..3], inner [2..6] — 2 rows × 4 elements (= 2 bytes).
    // Source storage = 2*2 = 4 bytes.
    let source_shape = [2i32, 4];
    let ranges = [(1, 3), (2, 6)];

    let dest_storage: usize = (4 * 8) / 2;
    let source_storage: usize = (2 * 4) / 2;

    // Init dest bytes with a distinctive pattern.
    let dest_init: Vec<u8> = (0..dest_storage as u8).map(|i| i.wrapping_mul(17)).collect();
    let source_init: Vec<u8> = (0..source_storage as u8).map(|i| 0xA0 | (i & 0x0F)).collect();

    // Wrap u8 buffers as S4 via view_as.
    let mut dev_dest_u8 = DeviceBuffer::from_slice(&ctx, &dest_init).expect("upload dest");
    let dev_source_u8 = DeviceBuffer::from_slice(&ctx, &source_init).expect("upload source");

    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::S4,
    };
    let plan = WriteSlicePlan::<S4, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    {
        let dev_dest_s4 = dev_dest_u8.view_as_mut::<S4>();
        let dev_source_s4 = dev_source_u8.view_as::<S4>();
        let args = WriteSliceArgs::<S4, 2> {
            dest: TensorMut {
                data: dev_dest_s4,
                shape: dest_shape,
                stride: contiguous_stride(dest_shape),
            },
            source: TensorRef {
                data: dev_source_s4,
                shape: source_shape,
                stride: contiguous_stride(source_shape),
            },
        };
        plan.run(&stream, Workspace::None, args).expect("run");
        stream.synchronize().expect("sync");
    }

    let mut got = vec![0u8; dest_storage];
    dev_dest_u8.copy_to_host(&mut got).expect("download");

    // Expected: byte-level write of the (start_bytes, len_bytes) chunk
    // per row. Row stride = 4 bytes; inner byte range = [1, 3) per row
    // (4 elements = 2 bytes; start_elem=2 → start_byte=1).
    let mut expected = dest_init.clone();
    let row_byte_stride = 4usize;
    let inner_start_byte = 2usize / 2; // 1
    let inner_len_bytes = (6 - 2) / 2;  // 2
    for r in 0..2 {
        let dest_row = (ranges[0].0 as usize + r) * row_byte_stride;
        let src_row = r * inner_len_bytes;
        for b in 0..inner_len_bytes {
            expected[dest_row + inner_start_byte + b] = source_init[src_row + b];
        }
    }
    assert_eq!(got, expected, "nibble_s4_rank2 mismatch");
}

/// Negative: range out of bounds → `select` rejects.
#[test]
#[ignore]
fn write_slice_out_of_bounds_rejected() {
    let (_ctx, stream) = setup();
    let dest_shape = [16i32, 8];
    let source_shape = [4i32, 4];
    let ranges = [(14, 18), (0, 4)]; // 18 > 16 on axis 0
    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::F32,
    };
    let res = WriteSlicePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default());
    assert!(res.is_err(), "out-of-bounds range must be rejected at select");
}

/// Negative: nibble innermost odd start → `select` rejects with
/// `Unsupported`.
#[test]
#[ignore]
fn write_slice_nibble_odd_start_rejected() {
    let (_ctx, stream) = setup();
    let dest_shape = [4i32, 8];
    let source_shape = [2i32, 4];
    let ranges = [(1, 3), (1, 5)]; // 1 on innermost is odd
    let desc = WriteSliceDescriptor {
        dest_shape,
        source_shape,
        ranges,
        element: ElementKind::S4,
    };
    let res = WriteSlicePlan::<S4, 2>::select(&stream, &desc, PlanPreference::default());
    assert!(
        res.is_err(),
        "nibble write with odd-aligned innermost start must be rejected"
    );
}

/// Reinterpret an `&[T]` as `&[u8]` for byte-level reference computations.
fn bytemuck_slice<T: Copy>(s: &[T]) -> Vec<u8> {
    let bytes_per = core::mem::size_of::<T>();
    let mut out = vec![0u8; s.len() * bytes_per];
    unsafe {
        core::ptr::copy_nonoverlapping(
            s.as_ptr() as *const u8,
            out.as_mut_ptr(),
            s.len() * bytes_per,
        );
    }
    out
}
