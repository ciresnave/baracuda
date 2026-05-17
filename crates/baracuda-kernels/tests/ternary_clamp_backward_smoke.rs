//! Real-GPU smoke test for `TernaryBackwardPlan<T, N> + TernaryKind::Clamp`
//! across f32 / f16 / bf16 / f64.
//!
//! Forward: `y = min(max(a, b), c)` (b = lo, c = hi). Backward:
//!   da = dy if b <= a <= c else 0
//!   db = dy if a <  b      else 0
//!   dc = dy if a >  c      else 0
//!
//! Bit-exact at every dtype — the kernel only does compare + select,
//! no math. Test inputs are constructed so that each cell lands in a
//! different branch (some below lo, some in-range, some above hi).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test ternary_clamp_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TernaryBackwardArgs,
    TernaryBackwardDescriptor, TernaryBackwardPlan, TernaryKind, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Returns (a, b, c) values for cell `i` covering all three branches:
//   i % 3 == 0 → a < b   (below lo, db active)
//   i % 3 == 1 → b ≤ a ≤ c (in range, da active)
//   i % 3 == 2 → a > c   (above hi, dc active)
// `b = -1.0`, `c = +1.0` fixed; `a` varies in {-3, 0, +3} (+ small jitter).
fn gen_abc_f32(i: usize) -> (f32, f32, f32) {
    let branch = i % 3;
    let jitter = ((i / 3) % 7) as f32 * 0.0625; // small additive noise
    let a = match branch {
        0 => -3.0 - jitter,                 // < b
        1 => -0.5 + 0.125 * (branch as f32) + jitter * 0.25, // in [b, c]
        _ => 3.0 + jitter,                  // > c
    };
    (a, -1.0, 1.0)
}

#[test]
#[ignore]
fn clamp_backward_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32, 96];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 23.0).collect();
    let (host_a, host_b, host_c): (Vec<f32>, Vec<f32>, Vec<f32>) = {
        let mut a = Vec::with_capacity(numel);
        let mut b = Vec::with_capacity(numel);
        let mut c = Vec::with_capacity(numel);
        for i in 0..numel {
            let (ai, bi, ci) = gen_abc_f32(i);
            a.push(ai); b.push(bi); c.push(ci);
        }
        (a, b, c)
    };
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).unwrap();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).unwrap();
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dc: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let stride = contiguous_stride(shape);
    let desc = TernaryBackwardDescriptor {
        kind: TernaryKind::Clamp,
        shape,
        element: ElementKind::F32,
        scale: 1.0,
    };
    let plan = TernaryBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
        dc: TensorMut { data: dev_dc.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f32; numel];
    let mut got_db = vec![0f32; numel];
    let mut got_dc = vec![0f32; numel];
    dev_da.copy_to_host(&mut got_da).unwrap();
    dev_db.copy_to_host(&mut got_db).unwrap();
    dev_dc.copy_to_host(&mut got_dc).unwrap();
    // Track branch coverage so we know all three masks are exercised.
    let (mut n_below, mut n_in, mut n_above) = (0usize, 0usize, 0usize);
    for i in 0..numel {
        let a = host_a[i]; let b = host_b[i]; let c = host_c[i]; let dy = host_dy[i];
        let ed_a = if b <= a && a <= c { dy } else { 0.0 };
        let ed_b = if a <  b           { dy } else { 0.0 };
        let ed_c = if a >  c           { dy } else { 0.0 };
        if a < b { n_below += 1; } else if a > c { n_above += 1; } else { n_in += 1; }
        assert_eq!(got_da[i].to_bits(), ed_a.to_bits(), "clamp BW f32 da @ {i}");
        assert_eq!(got_db[i].to_bits(), ed_b.to_bits(), "clamp BW f32 db @ {i}");
        assert_eq!(got_dc[i].to_bits(), ed_c.to_bits(), "clamp BW f32 dc @ {i}");
    }
    assert!(n_below > 0 && n_in > 0 && n_above > 0,
        "clamp BW f32 test must cover all three branches; got {n_below}/{n_in}/{n_above}");
}

#[test]
#[ignore]
fn clamp_backward_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32, 96];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..numel).map(|i| f16::from_f32(((i % 47) as f32) * 0.25 - 5.0)).collect();
    let mut host_a: Vec<f16> = Vec::with_capacity(numel);
    let mut host_b: Vec<f16> = Vec::with_capacity(numel);
    let mut host_c: Vec<f16> = Vec::with_capacity(numel);
    for i in 0..numel {
        let (a, b, c) = gen_abc_f32(i);
        host_a.push(f16::from_f32(a));
        host_b.push(f16::from_f32(b));
        host_c.push(f16::from_f32(c));
    }
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).unwrap();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).unwrap();
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dc: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let stride = contiguous_stride(shape);
    let desc = TernaryBackwardDescriptor {
        kind: TernaryKind::Clamp,
        shape,
        element: ElementKind::F16,
        scale: 1.0,
    };
    let plan = TernaryBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
        dc: TensorMut { data: dev_dc.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![f16::from_f32(0.0); numel];
    let mut got_db = vec![f16::from_f32(0.0); numel];
    let mut got_dc = vec![f16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).unwrap();
    dev_db.copy_to_host(&mut got_db).unwrap();
    dev_dc.copy_to_host(&mut got_dc).unwrap();
    let zero = f16::from_f32(0.0);
    for i in 0..numel {
        let af = host_a[i].to_f32();
        let bf = host_b[i].to_f32();
        let cf = host_c[i].to_f32();
        let ed_a = if bf <= af && af <= cf { host_dy[i] } else { zero };
        let ed_b = if af <  bf              { host_dy[i] } else { zero };
        let ed_c = if af >  cf              { host_dy[i] } else { zero };
        assert_eq!(got_da[i].to_bits(), ed_a.to_bits(), "clamp BW f16 da @ {i}");
        assert_eq!(got_db[i].to_bits(), ed_b.to_bits(), "clamp BW f16 db @ {i}");
        assert_eq!(got_dc[i].to_bits(), ed_c.to_bits(), "clamp BW f16 dc @ {i}");
    }
}

#[test]
#[ignore]
fn clamp_backward_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32, 96];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..numel).map(|i| bf16::from_f32(((i % 47) as f32) * 0.25 - 5.0)).collect();
    let mut host_a: Vec<bf16> = Vec::with_capacity(numel);
    let mut host_b: Vec<bf16> = Vec::with_capacity(numel);
    let mut host_c: Vec<bf16> = Vec::with_capacity(numel);
    for i in 0..numel {
        let (a, b, c) = gen_abc_f32(i);
        host_a.push(bf16::from_f32(a));
        host_b.push(bf16::from_f32(b));
        host_c.push(bf16::from_f32(c));
    }
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).unwrap();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).unwrap();
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dc: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let stride = contiguous_stride(shape);
    let desc = TernaryBackwardDescriptor {
        kind: TernaryKind::Clamp,
        shape,
        element: ElementKind::Bf16,
        scale: 1.0,
    };
    let plan = TernaryBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
        dc: TensorMut { data: dev_dc.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![bf16::from_f32(0.0); numel];
    let mut got_db = vec![bf16::from_f32(0.0); numel];
    let mut got_dc = vec![bf16::from_f32(0.0); numel];
    dev_da.copy_to_host(&mut got_da).unwrap();
    dev_db.copy_to_host(&mut got_db).unwrap();
    dev_dc.copy_to_host(&mut got_dc).unwrap();
    let zero = bf16::from_f32(0.0);
    for i in 0..numel {
        let af = host_a[i].to_f32();
        let bf_ = host_b[i].to_f32();
        let cf = host_c[i].to_f32();
        let ed_a = if bf_ <= af && af <= cf { host_dy[i] } else { zero };
        let ed_b = if af <  bf_              { host_dy[i] } else { zero };
        let ed_c = if af >  cf               { host_dy[i] } else { zero };
        assert_eq!(got_da[i].to_bits(), ed_a.to_bits(), "clamp BW bf16 da @ {i}");
        assert_eq!(got_db[i].to_bits(), ed_b.to_bits(), "clamp BW bf16 db @ {i}");
        assert_eq!(got_dc[i].to_bits(), ed_c.to_bits(), "clamp BW bf16 dc @ {i}");
    }
}

#[test]
#[ignore]
fn clamp_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 32, 96];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 23.0).collect();
    let mut host_a: Vec<f64> = Vec::with_capacity(numel);
    let mut host_b: Vec<f64> = Vec::with_capacity(numel);
    let mut host_c: Vec<f64> = Vec::with_capacity(numel);
    for i in 0..numel {
        let (a, b, c) = gen_abc_f32(i);
        host_a.push(a as f64);
        host_b.push(b as f64);
        host_c.push(c as f64);
    }
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).unwrap();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).unwrap();
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).unwrap();
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).unwrap();
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dc: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let stride = contiguous_stride(shape);
    let desc = TernaryBackwardDescriptor {
        kind: TernaryKind::Clamp,
        shape,
        element: ElementKind::F64,
        scale: 1.0,
    };
    let plan = TernaryBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        da: TensorMut { data: dev_da.as_slice_mut(), shape, stride },
        db: TensorMut { data: dev_db.as_slice_mut(), shape, stride },
        dc: TensorMut { data: dev_dc.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got_da = vec![0f64; numel];
    let mut got_db = vec![0f64; numel];
    let mut got_dc = vec![0f64; numel];
    dev_da.copy_to_host(&mut got_da).unwrap();
    dev_db.copy_to_host(&mut got_db).unwrap();
    dev_dc.copy_to_host(&mut got_dc).unwrap();
    for i in 0..numel {
        let a = host_a[i]; let b = host_b[i]; let c = host_c[i]; let dy = host_dy[i];
        let ed_a = if b <= a && a <= c { dy } else { 0.0 };
        let ed_b = if a <  b           { dy } else { 0.0 };
        let ed_c = if a >  c           { dy } else { 0.0 };
        assert_eq!(got_da[i].to_bits(), ed_a.to_bits(), "clamp BW f64 da @ {i}");
        assert_eq!(got_db[i].to_bits(), ed_b.to_bits(), "clamp BW f64 db @ {i}");
        assert_eq!(got_dc[i].to_bits(), ed_c.to_bits(), "clamp BW f64 dc @ {i}");
    }
}
