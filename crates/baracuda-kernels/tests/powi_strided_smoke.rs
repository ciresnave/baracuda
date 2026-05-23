//! Real-GPU smoke test for `UnaryParamPlan + UnaryKind::PowI` strided
//! sibling (Phase 14.2).
//!
//! Mirrors `unary_powi_smoke.rs` but exercises the strided FFI path:
//! transposed input, non-contig output, and the contig fast-path
//! through the same plan (which should pick `_run` rather than
//! `_strided_run`).
//!
//! Coverage:
//!   * Contig FW (`n = 2`) — confirms the dispatcher still picks the
//!     fast path when both operands are canonical.
//!   * Transposed-input FW (`n = 3`) — strided correctness against a
//!     host-side multi-coord reference.
//!   * Strided-output FW (`n = 2`) — `y` view writes through a stride
//!     pattern that's not canonical (e.g. row-stride doubled).
//!   * Contig BW (`n = 3`) — fast-path BW sanity.
//!   * Strided BW (`n = 3`) — transposed `dy`/`dx` views against the
//!     analytical `dx = n · x^(n-1) · dy`.
//!   * Per-dtype FW spot-check via the transposed path at `n = 2`:
//!     f32 already covered above; f16 / bf16 / f64 verify all dtypes
//!     wired through the strided launcher.
//!
//! `#[ignore]` by default; run with `--ignored` on a CUDA host.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryKind,
    UnaryParamArgs, UnaryParamBackwardArgs, UnaryParamBackwardDescriptor, UnaryParamBackwardPlan,
    UnaryParamDescriptor, UnaryParamPlan, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Same shape every call so we can pre-bake the [M, N] sizes.
const M: i32 = 8;
const N_DIM: i32 = 6;

fn make_x_f32(numel: usize) -> Vec<f32> {
    (0..numel)
        .map(|i| {
            let raw = (i as f32) * 0.005 - 2.55;
            if raw == 0.0 { 0.001 } else { raw }
        })
        .collect()
}

// ============================================================================
// Forward
// ============================================================================

#[test]
#[ignore]
fn powi_strided_fw_f32_contig_fastpath() {
    // Plan should choose `_run` (contig) rather than `_strided_run`,
    // but either way the numerics must match the host reference. This
    // doubles as a sanity check that adding the strided sibling did
    // not regress the contig path.
    let (ctx, stream) = setup();
    let shape = [M, N_DIM];
    let numel: usize = (M * N_DIM) as usize;
    let host_x = make_x_f32(numel);
    let host_expected: Vec<f32> = host_x.iter().map(|&v| v.powi(2)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);

    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape,
        element: ElementKind::F32,
        params: [2.0, 0.0],
    };
    let plan = UnaryParamPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f32, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "contig n=2 @ {i}: got {g}, exp {e}");
    }
}

#[test]
#[ignore]
fn powi_strided_fw_f32_transposed_input() {
    // Build x in [M, N] contig layout, then describe it to the plan as
    // [N, M] with strides [1, N] — a logical transpose. y is contig
    // [N, M]. Reference computes pow against the transposed view.
    let (ctx, stream) = setup();
    let m = M as usize;
    let n = N_DIM as usize;
    let host_x_phys = make_x_f32(m * n); // physical row-major [M, N]
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_phys).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, m * n).expect("alloc y");

    // Logical shape after transpose: [N, M]; transposed-view strides.
    let logical_shape = [N_DIM, M];
    let stride_x_t = [1i64, N_DIM as i64]; // [row_stride, col_stride] for transposed view
    let stride_y_contig = contiguous_stride(logical_shape);

    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape: logical_shape,
        element: ElementKind::F32,
        params: [3.0, 0.0],
    };
    let plan = UnaryParamPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f32, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape: logical_shape, stride: stride_x_t },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: stride_y_contig,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; m * n];
    dev_y.copy_to_host(&mut got).expect("download");

    // Reference: for each logical (i, j) in [N, M], the source element
    // is x_phys[j * N + i] (the transposed read), and the output index
    // in contig [N, M] is i * M + j.
    for i in 0..n {
        for j in 0..m {
            let src = host_x_phys[j * n + i];
            let exp = src.powi(3);
            let g = got[i * m + j];
            let denom = exp.abs().max(1e-6);
            let rel = (g - exp).abs() / denom;
            assert!(
                rel < 1e-5,
                "transposed-x n=3 @ ({i},{j}): src={src}, got {g}, exp {exp}, rel {rel}"
            );
        }
    }
}

#[test]
#[ignore]
fn powi_strided_fw_f32_strided_output() {
    // x contig [M, N]; y is a sliced view onto a larger buffer
    // [M, 2*N] picking every other column. The kernel writes to the
    // strided positions only; the in-between positions should remain
    // at their initialised value (zero).
    let (ctx, stream) = setup();
    let m = M as usize;
    let n = N_DIM as usize;
    let shape = [M, N_DIM];
    let host_x = make_x_f32(m * n);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    // Larger output buffer [M, 2*N] zero-init.
    let mut dev_y_big: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, m * 2 * n).expect("alloc y");
    // Strided view: row_stride = 2*N, col_stride = 2. Writes columns
    // 0, 2, 4, …, 2*N-2 of the underlying buffer.
    let stride_y_strided = [(2 * n) as i64, 2i64];
    let stride_x_contig = contiguous_stride(shape);

    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape,
        element: ElementKind::F32,
        params: [2.0, 0.0],
    };
    let plan = UnaryParamPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f32, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: stride_x_contig },
        y: TensorMut {
            data: dev_y_big.as_slice_mut(),
            shape,
            stride: stride_y_strided,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; m * 2 * n];
    dev_y_big.copy_to_host(&mut got).expect("download");
    for i in 0..m {
        for j in 0..n {
            let exp = host_x[i * n + j].powi(2);
            let g_writ = got[i * 2 * n + 2 * j];
            assert_eq!(
                g_writ.to_bits(),
                exp.to_bits(),
                "strided-y n=2 @ ({i},{j}): got {g_writ}, exp {exp}"
            );
            // Unwritten slot — should still be zero.
            let g_gap = got[i * 2 * n + 2 * j + 1];
            assert_eq!(
                g_gap, 0.0,
                "strided-y n=2 @ ({i},{j}+gap): scribble! got {g_gap}"
            );
        }
    }
}

#[test]
#[ignore]
fn powi_strided_fw_f16_transposed() {
    // Same transposed-input pattern as the f32 case; n=2 collapses to
    // a single multiply so half-precision rounding is bounded.
    let (ctx, stream) = setup();
    let m = M as usize;
    let n = N_DIM as usize;
    let host_x_phys: Vec<f16> = make_x_f32(m * n)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_phys).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, m * n).expect("alloc y");
    let logical_shape = [N_DIM, M];
    let stride_x_t = [1i64, N_DIM as i64];
    let stride_y_contig = contiguous_stride(logical_shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape: logical_shape,
        element: ElementKind::F16,
        params: [2.0, 0.0],
    };
    let plan = UnaryParamPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f16, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape: logical_shape, stride: stride_x_t },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: stride_y_contig,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); m * n];
    dev_y.copy_to_host(&mut got).expect("download");
    for i in 0..n {
        for j in 0..m {
            let src = host_x_phys[j * n + i].to_f32();
            let exp = f16::from_f32(src * src);
            let g = got[i * m + j];
            let diff = (g.to_bits() as i32 - exp.to_bits() as i32).abs();
            assert!(
                diff <= 1,
                "transposed-x f16 n=2 @ ({i},{j}): got {} (bits {:#x}), exp {} (bits {:#x})",
                g.to_f32(),
                g.to_bits(),
                exp.to_f32(),
                exp.to_bits()
            );
        }
    }
}

#[test]
#[ignore]
fn powi_strided_fw_bf16_transposed() {
    let (ctx, stream) = setup();
    let m = M as usize;
    let n = N_DIM as usize;
    let host_x_phys: Vec<bf16> = make_x_f32(m * n)
        .iter()
        .map(|&v| bf16::from_f32(v))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_phys).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, m * n).expect("alloc y");
    let logical_shape = [N_DIM, M];
    let stride_x_t = [1i64, N_DIM as i64];
    let stride_y_contig = contiguous_stride(logical_shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape: logical_shape,
        element: ElementKind::Bf16,
        params: [2.0, 0.0],
    };
    let plan = UnaryParamPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<bf16, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape: logical_shape, stride: stride_x_t },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: stride_y_contig,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); m * n];
    dev_y.copy_to_host(&mut got).expect("download");
    for i in 0..n {
        for j in 0..m {
            let src = host_x_phys[j * n + i].to_f32();
            let exp = bf16::from_f32(src * src);
            let g = got[i * m + j];
            let diff = (g.to_bits() as i32 - exp.to_bits() as i32).abs();
            assert!(
                diff <= 1,
                "transposed-x bf16 n=2 @ ({i},{j}): got {} (bits {:#x}), exp {} (bits {:#x})",
                g.to_f32(),
                g.to_bits(),
                exp.to_f32(),
                exp.to_bits()
            );
        }
    }
}

#[test]
#[ignore]
fn powi_strided_fw_f64_transposed() {
    let (ctx, stream) = setup();
    let m = M as usize;
    let n = N_DIM as usize;
    let host_x_phys: Vec<f64> = (0..(m * n))
        .map(|i| {
            let raw = (i as f64) * 0.005 - 2.55;
            if raw == 0.0 { 0.001 } else { raw }
        })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_phys).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, m * n).expect("alloc y");
    let logical_shape = [N_DIM, M];
    let stride_x_t = [1i64, N_DIM as i64];
    let stride_y_contig = contiguous_stride(logical_shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::PowI,
        shape: logical_shape,
        element: ElementKind::F64,
        params: [2.0, 0.0],
    };
    let plan = UnaryParamPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f64, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape: logical_shape, stride: stride_x_t },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: logical_shape,
            stride: stride_y_contig,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; m * n];
    dev_y.copy_to_host(&mut got).expect("download");
    for i in 0..n {
        for j in 0..m {
            let src = host_x_phys[j * n + i];
            // n=2 reduces to a single multiply; bit-exact against `x*x`.
            let exp = src * src;
            let g = got[i * m + j];
            assert_eq!(
                g.to_bits(),
                exp.to_bits(),
                "transposed-x f64 n=2 @ ({i},{j}): got {g}, exp {exp}"
            );
        }
    }
}

// ============================================================================
// Backward
// ============================================================================

#[test]
#[ignore]
fn powi_strided_bw_f32_contig_fastpath() {
    // Same as the FW contig-fastpath — confirms BW dispatcher routes
    // canonical args through `_run` and matches the host reference.
    let (ctx, stream) = setup();
    let shape = [M, N_DIM];
    let numel: usize = (M * N_DIM) as usize;
    let host_x = make_x_f32(numel);
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 1.5).collect();
    let n_exp = 3;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let stride = contiguous_stride(shape);

    let desc = UnaryParamBackwardDescriptor {
        kind: UnaryKind::PowI,
        shape,
        element: ElementKind::F32,
        params: [n_exp as f32, 0.0],
    };
    let plan = UnaryParamBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamBackwardArgs::<f32, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..numel {
        let exp = (n_exp as f32) * host_x[i].powi(n_exp - 1) * host_dy[i];
        let denom = exp.abs().max(1e-6);
        let rel = (got[i] - exp).abs() / denom;
        assert!(
            rel < 1e-5,
            "contig BW n={n_exp} @ {i}: x={}, dy={}, got {}, exp {}, rel {rel}",
            host_x[i],
            host_dy[i],
            got[i],
            exp
        );
    }
}

#[test]
#[ignore]
fn powi_strided_bw_f32_transposed() {
    // dy and x are described as transposed views of physical [M, N]
    // buffers. dx is contig [N, M]. The kernel must read from the
    // transposed positions and write contiguously.
    let (ctx, stream) = setup();
    let m = M as usize;
    let n = N_DIM as usize;
    let host_x_phys = make_x_f32(m * n);
    let host_dy_phys: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.05 - 1.5).collect();
    let n_exp = 3;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x_phys).expect("upload x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy_phys).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, m * n).expect("alloc dx");

    let logical_shape = [N_DIM, M];
    let stride_t = [1i64, N_DIM as i64]; // both x and dy are transposed
    let stride_dx_contig = contiguous_stride(logical_shape);

    let desc = UnaryParamBackwardDescriptor {
        kind: UnaryKind::PowI,
        shape: logical_shape,
        element: ElementKind::F32,
        params: [n_exp as f32, 0.0],
    };
    let plan = UnaryParamBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamBackwardArgs::<f32, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: logical_shape, stride: stride_t },
        x: TensorRef { data: dev_x.as_slice(), shape: logical_shape, stride: stride_t },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: logical_shape,
            stride: stride_dx_contig,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; m * n];
    dev_dx.copy_to_host(&mut got).expect("download");
    for i in 0..n {
        for j in 0..m {
            let src_x = host_x_phys[j * n + i];
            let src_dy = host_dy_phys[j * n + i];
            let exp = (n_exp as f32) * src_x.powi(n_exp - 1) * src_dy;
            let g = got[i * m + j];
            let denom = exp.abs().max(1e-6);
            let rel = (g - exp).abs() / denom;
            assert!(
                rel < 1e-5,
                "transposed BW n={n_exp} @ ({i},{j}): x={src_x}, dy={src_dy}, got {g}, \
                 exp {exp}, rel {rel}"
            );
        }
    }
}
