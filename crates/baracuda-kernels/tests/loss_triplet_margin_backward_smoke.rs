//! Real-GPU smoke test for `TripletMarginLossBackwardPlan`. BW × 4 dtypes.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    ElementKind, LossReduction, PlanPreference, TensorMut, TensorRef,
    TripletMarginLossBackwardArgs, TripletMarginLossBackwardDescriptor,
    TripletMarginLossBackwardPlan, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn host_triplet_bw_f64(
    a: &[f64], p: &[f64], n: &[f64], rows: usize, d: usize,
    margin: f64, pn: f64, dy: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let scale = dy / (rows as f64);
    let numel = rows * d;
    let mut da = vec![0.0; numel];
    let mut dp = vec![0.0; numel];
    let mut dn = vec![0.0; numel];
    for r in 0..rows {
        let mut sp = 0.0;
        let mut sn = 0.0;
        for j in 0..d {
            sp += (a[r * d + j] - p[r * d + j]).abs().powf(pn);
            sn += (a[r * d + j] - n[r * d + j]).abs().powf(pn);
        }
        let pd = sp.powf(1.0 / pn);
        let nd = sn.powf(1.0 / pn);
        let loss = pd - nd + margin;
        if loss <= 0.0 {
            continue;
        }
        let pd_pm1 = pd.powf(pn - 1.0).max(1e-30);
        let nd_pm1 = nd.powf(pn - 1.0).max(1e-30);
        for j in 0..d {
            let d1 = a[r * d + j] - p[r * d + j];
            let d2 = a[r * d + j] - n[r * d + j];
            let sgn1 = if d1 > 0.0 { 1.0 } else if d1 < 0.0 { -1.0 } else { 0.0 };
            let sgn2 = if d2 > 0.0 { 1.0 } else if d2 < 0.0 { -1.0 } else { 0.0 };
            let pa_pj = d1.abs().powf(pn - 1.0) * sgn1 / pd_pm1;
            let pa_nj = d2.abs().powf(pn - 1.0) * sgn2 / nd_pm1;
            da[r * d + j] = (pa_pj - pa_nj) * scale;
            dp[r * d + j] = (-pa_pj) * scale;
            dn[r * d + j] = (pa_nj) * scale;
        }
    }
    (da, dp, dn)
}

#[test]
#[ignore]
fn loss_triplet_margin_backward_f32_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 1.0f32;
    let p_norm = 2.0f32;
    let numel = n * d;
    let h_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let h_p: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.02).collect();
    let h_n: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let dy_host = [1.0f32];
    let (ea, ep, en) = host_triplet_bw_f64(
        &h_a.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &h_p.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &h_n.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        n, d, margin as f64, p_norm as f64, 1.0,
    );
    let dev_a = DeviceBuffer::from_slice(&ctx, &h_a).unwrap();
    let dev_p = DeviceBuffer::from_slice(&ctx, &h_p).unwrap();
    let dev_n = DeviceBuffer::from_slice(&ctx, &h_n).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_da: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dp: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dn: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = TripletMarginLossBackwardDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F32,
    };
    let plan = TripletMarginLossBackwardPlan::<f32>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        TripletMarginLossBackwardArgs {
            anchor: TensorRef { data: dev_a.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            positive: TensorRef { data: dev_p.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            negative: TensorRef { data: dev_n.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            d_anchor: TensorMut { data: dev_da.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            d_positive: TensorMut { data: dev_dp.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            d_negative: TensorMut { data: dev_dn.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_a = vec![0f32; numel];
    let mut got_p = vec![0f32; numel];
    let mut got_n = vec![0f32; numel];
    dev_da.copy_to_host(&mut got_a).unwrap();
    dev_dp.copy_to_host(&mut got_p).unwrap();
    dev_dn.copy_to_host(&mut got_n).unwrap();
    for i in 0..numel {
        let wa = ea[i] as f32;
        let wp = ep[i] as f32;
        let wn = en[i] as f32;
        let tol = (wa.abs().max(wp.abs()).max(wn.abs())).max(1.0) * 32.0 * f32::EPSILON + 1e-5;
        assert!((got_a[i] - wa).abs() <= tol, "f32 Triplet BW da @{i}: got={} want={}", got_a[i], wa);
        assert!((got_p[i] - wp).abs() <= tol);
        assert!((got_n[i] - wn).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_triplet_margin_backward_f64_mean() {
    let (ctx, stream) = setup();
    let n = 2usize;
    let d = 4usize;
    let margin = 1.0f32;
    let p_norm = 2.0f32;
    let numel = n * d;
    let h_a: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05).collect();
    let h_p: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05 + 0.02).collect();
    let h_n: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.05 - 0.3).collect();
    let dy_host = [2.0f64];
    let (ea, ep, en) = host_triplet_bw_f64(&h_a, &h_p, &h_n, n, d, margin as f64, p_norm as f64, 2.0);
    let dev_a = DeviceBuffer::from_slice(&ctx, &h_a).unwrap();
    let dev_p = DeviceBuffer::from_slice(&ctx, &h_p).unwrap();
    let dev_n = DeviceBuffer::from_slice(&ctx, &h_n).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_da: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dp: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dn: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = TripletMarginLossBackwardDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F64,
    };
    let plan = TripletMarginLossBackwardPlan::<f64>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        TripletMarginLossBackwardArgs {
            anchor: TensorRef { data: dev_a.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            positive: TensorRef { data: dev_p.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            negative: TensorRef { data: dev_n.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            d_anchor: TensorMut { data: dev_da.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            d_positive: TensorMut { data: dev_dp.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            d_negative: TensorMut { data: dev_dn.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_a = vec![0f64; numel];
    let mut got_p = vec![0f64; numel];
    let mut got_n = vec![0f64; numel];
    dev_da.copy_to_host(&mut got_a).unwrap();
    dev_dp.copy_to_host(&mut got_p).unwrap();
    dev_dn.copy_to_host(&mut got_n).unwrap();
    for i in 0..numel {
        let tol = (ea[i].abs().max(ep[i].abs()).max(en[i].abs())).max(1.0) * 32.0 * f64::EPSILON + 1e-11;
        assert!((got_a[i] - ea[i]).abs() <= tol);
        assert!((got_p[i] - ep[i]).abs() <= tol);
        assert!((got_n[i] - en[i]).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_triplet_margin_backward_f16_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 1.0f32;
    let p_norm = 2.0f32;
    let numel = n * d;
    let h_a_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let h_p_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.02).collect();
    let h_n_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let h_a: Vec<f16> = h_a_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_p: Vec<f16> = h_p_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let h_n: Vec<f16> = h_n_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dy_host = [f16::from_f32(1.0)];
    let a64: Vec<f64> = h_a.iter().map(|&v| v.to_f32() as f64).collect();
    let p64: Vec<f64> = h_p.iter().map(|&v| v.to_f32() as f64).collect();
    let n64: Vec<f64> = h_n.iter().map(|&v| v.to_f32() as f64).collect();
    let (ea, ep, en) = host_triplet_bw_f64(&a64, &p64, &n64, n, d, margin as f64, p_norm as f64, 1.0);
    let dev_a = DeviceBuffer::from_slice(&ctx, &h_a).unwrap();
    let dev_p = DeviceBuffer::from_slice(&ctx, &h_p).unwrap();
    let dev_n = DeviceBuffer::from_slice(&ctx, &h_n).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_da: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dp: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dn: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = TripletMarginLossBackwardDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::F16,
    };
    let plan = TripletMarginLossBackwardPlan::<f16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        TripletMarginLossBackwardArgs {
            anchor: TensorRef { data: dev_a.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            positive: TensorRef { data: dev_p.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            negative: TensorRef { data: dev_n.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            d_anchor: TensorMut { data: dev_da.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            d_positive: TensorMut { data: dev_dp.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            d_negative: TensorMut { data: dev_dn.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_a = vec![f16::ZERO; numel];
    let mut got_p = vec![f16::ZERO; numel];
    let mut got_n = vec![f16::ZERO; numel];
    dev_da.copy_to_host(&mut got_a).unwrap();
    dev_dp.copy_to_host(&mut got_p).unwrap();
    dev_dn.copy_to_host(&mut got_n).unwrap();
    for i in 0..numel {
        let wa = ea[i] as f32;
        let wp = ep[i] as f32;
        let wn = en[i] as f32;
        let ga = got_a[i].to_f32();
        let gp = got_p[i].to_f32();
        let gn = got_n[i].to_f32();
        let tol = (wa.abs().max(wp.abs()).max(wn.abs())).max(1.0) * 32.0 * 9.77e-4_f32 + 1e-2;
        assert!((ga - wa).abs() <= tol, "f16 Triplet BW da @{i}: got={} want={}", ga, wa);
        assert!((gp - wp).abs() <= tol);
        assert!((gn - wn).abs() <= tol);
    }
}

#[test]
#[ignore]
fn loss_triplet_margin_backward_bf16_mean() {
    let (ctx, stream) = setup();
    let n = 3usize;
    let d = 4usize;
    let margin = 1.0f32;
    let p_norm = 2.0f32;
    let numel = n * d;
    let h_a_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05).collect();
    let h_p_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 + 0.02).collect();
    let h_n_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let h_a: Vec<bf16> = h_a_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_p: Vec<bf16> = h_p_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let h_n: Vec<bf16> = h_n_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_host = [bf16::from_f32(1.0)];
    let a64: Vec<f64> = h_a.iter().map(|&v| v.to_f32() as f64).collect();
    let p64: Vec<f64> = h_p.iter().map(|&v| v.to_f32() as f64).collect();
    let n64: Vec<f64> = h_n.iter().map(|&v| v.to_f32() as f64).collect();
    let (ea, ep, en) = host_triplet_bw_f64(&a64, &p64, &n64, n, d, margin as f64, p_norm as f64, 1.0);
    let dev_a = DeviceBuffer::from_slice(&ctx, &h_a).unwrap();
    let dev_p = DeviceBuffer::from_slice(&ctx, &h_p).unwrap();
    let dev_n = DeviceBuffer::from_slice(&ctx, &h_n).unwrap();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dev_da: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dp: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let mut dev_dn: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = TripletMarginLossBackwardDescriptor {
        n_rows: n as i32,
        d_extent: d as i32,
        reduction: LossReduction::Mean,
        margin,
        p_norm,
        element: ElementKind::Bf16,
    };
    let plan = TripletMarginLossBackwardPlan::<bf16>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .unwrap();
    plan.run(
        &stream,
        Workspace::None,
        TripletMarginLossBackwardArgs {
            anchor: TensorRef { data: dev_a.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            positive: TensorRef { data: dev_p.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            negative: TensorRef { data: dev_n.as_slice(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            dy: TensorRef { data: dev_dy.as_slice(), shape: [1, 1], stride: [1, 1] },
            d_anchor: TensorMut { data: dev_da.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            d_positive: TensorMut { data: dev_dp.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
            d_negative: TensorMut { data: dev_dn.as_slice_mut(), shape: [n as i32, d as i32], stride: [d as i64, 1] },
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let mut got_a = vec![bf16::ZERO; numel];
    let mut got_p = vec![bf16::ZERO; numel];
    let mut got_n = vec![bf16::ZERO; numel];
    dev_da.copy_to_host(&mut got_a).unwrap();
    dev_dp.copy_to_host(&mut got_p).unwrap();
    dev_dn.copy_to_host(&mut got_n).unwrap();
    for i in 0..numel {
        let wa = ea[i] as f32;
        let wp = ep[i] as f32;
        let wn = en[i] as f32;
        let ga = got_a[i].to_f32();
        let gp = got_p[i].to_f32();
        let gn = got_n[i].to_f32();
        let tol = (wa.abs().max(wp.abs()).max(wn.abs())).max(1.0) * 32.0 * 7.81e-3_f32 + 5e-2;
        assert!((ga - wa).abs() <= tol, "bf16 Triplet BW da @{i}: got={} want={}", ga, wa);
        assert!((gp - wp).abs() <= tol);
        assert!((gn - wn).abs() <= tol);
    }
}
