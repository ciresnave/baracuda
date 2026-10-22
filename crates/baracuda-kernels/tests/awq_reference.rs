//! AWQ W4A16 GEMM vs a dequantise-then-GEMM reference at real shapes, with
//! non-trivial (random) zero-points.
//!
//! The reference is built from the raw 4-bit values, zero-points and f16
//! scales the test generated, never from the packed buffers the kernel reads.
//! Each shape also runs a known-bad control: one scale is corrupted on the
//! device only, and the SAME comparison must then fail in exactly that column.
//!
//! Buffer layout is the one the vendored kernel indexes
//! (`gemm_cuda_gen.cu`: `B_ptr`, `zeros_ptr`, `scaling_factors_ptr`), all
//! OC-major: weights `[OC, IC/8]` (nibble `ic % 8` of word `ic / 8`), zeros
//! `[OC, zeros_w]` (nibble `g % 8` of word `g / 8`), scales `[OC, 8*zeros_w]`,
//! where `g` is the group index and, with `groups = IC / group_size` and
//! `mult = 128 / group_size`, `zeros_w = ceil(ceil(groups / 8) / mult) * mult`
//! (the kernel pads each row; padding is never read as data). The shapes
//! include `groups % 8 != 0` (padded rows), group size 64, and `M > 128`.
//!
//! Device test (feature `device-tests`, not `#[ignore]`): run via
//! `cargo gpu-test -p baracuda-kernels --features sm89,awq,device-tests --test awq_reference`.

#![cfg(all(feature = "awq", feature = "device-tests"))]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init, require};
use baracuda_kernels::{
    Int4AwqGemmArgs, Int4AwqGemmDescriptor, Int4AwqGemmPlan, PlanPreference, TensorMut, TensorRef,
    Workspace, contiguous_stride,
};
use half::f16;

/// Deterministic xorshift so a failure reproduces from the shape alone.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Case {
    a: Vec<f16>,      // [M, IC]
    q: Vec<u8>,       // [IC, OC] 0..=15
    z: Vec<u8>,       // [groups, OC] 0..=15
    scales: Vec<f16>, // [groups, OC]
}

fn make_case(m: usize, ic: usize, oc: usize, group: usize) -> Case {
    let mut rng = Rng(0xD1B54A32D192ED03 ^ ((m * 31 + ic) * 31 + oc) as u64);
    let groups = ic / group;
    Case {
        a: (0..m * ic)
            .map(|_| f16::from_f32(rng.unit() * 2.0 - 1.0))
            .collect(),
        q: (0..ic * oc).map(|_| (rng.next() & 0xF) as u8).collect(),
        z: (0..groups * oc).map(|_| (rng.next() & 0xF) as u8).collect(),
        scales: (0..groups * oc)
            .map(|_| f16::from_f32(0.01 + rng.unit() * 0.04))
            .collect(),
    }
}

/// Kernel-native packed buffers `(weights, zeros, scales)`, with the kernel's
/// row padding (see the module docs).
fn pack(case: &Case, ic: usize, oc: usize, group: usize) -> (Vec<i32>, Vec<i32>, Vec<f16>) {
    let groups = ic / group;
    let mult = 128 / group;
    let zeros_w = groups.div_ceil(8).div_ceil(mult) * mult;
    let sf_w = zeros_w * 8;
    let mut w = vec![0u32; oc * ic / 8];
    let mut z = vec![0u32; oc * zeros_w];
    let mut s = vec![f16::ZERO; oc * sf_w];
    for o in 0..oc {
        for i in 0..ic {
            w[o * (ic / 8) + i / 8] |= (case.q[i * oc + o] as u32) << (4 * (i % 8));
        }
        for g in 0..groups {
            z[o * zeros_w + g / 8] |= (case.z[g * oc + o] as u32) << (4 * (g % 8));
            s[o * sf_w + g] = case.scales[g * oc + o];
        }
    }
    (
        w.into_iter().map(|x| x as i32).collect(),
        z.into_iter().map(|x| x as i32).collect(),
        s,
    )
}

/// `(reference, abs_sum)` per output cell; the weight is rounded to f16 as the
/// kernel does (`__float2half(scale * (q - zero))`), accumulated in f64.
fn reference(case: &Case, m: usize, ic: usize, oc: usize, group: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0f64; m * oc];
    let mut abs = vec![0f64; m * oc];
    for row in 0..m {
        for o in 0..oc {
            let (mut acc, mut asum) = (0f64, 0f64);
            for i in 0..ic {
                let g = i / group;
                let w = f16::from_f32(
                    case.scales[g * oc + o].to_f32()
                        * (case.q[i * oc + o] as f32 - case.z[g * oc + o] as f32),
                )
                .to_f64();
                let term = case.a[row * ic + i].to_f64() * w;
                acc += term;
                asum += term.abs();
            }
            out[row * oc + o] = acc;
            abs[row * oc + o] = asum;
        }
    }
    (out, abs)
}

/// Problem shape: `[M, IC] x [IC, OC]`, quantised in groups of `group` along IC.
#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    ic: usize,
    oc: usize,
    group: usize,
}

/// An initialised device, context and stream.
struct Gpu {
    ctx: Context,
    stream: Stream,
}

impl Gpu {
    fn new(dev: &Device) -> Self {
        let ctx = Context::new(dev).expect("context");
        let stream = Stream::new(&ctx).expect("stream");
        Self { ctx, stream }
    }
}

/// Kernel-native packed buffers: weights, zeros, scales.
#[derive(Clone)]
struct Packed {
    w: Vec<i32>,
    z: Vec<i32>,
    s: Vec<f16>,
}

fn tref(d: &DeviceBuffer<i32>, shape: [i32; 2]) -> TensorRef<'_, i32, 2> {
    TensorRef::new(d.as_slice(), shape, contiguous_stride(shape)).unwrap()
}

fn run_kernel(gpu: &Gpu, case: &Case, packed: &Packed, shape: Shape) -> Vec<f16> {
    let (ctx, stream) = (&gpu.ctx, &gpu.stream);
    let Shape { m, ic, oc, group } = shape;
    let split_k_iters = 8i32;
    let (mi, ici, oci, g) = (m as i32, ic as i32, oc as i32, group as i32);
    let dev_a = DeviceBuffer::from_slice(ctx, &case.a).expect("upload A");
    let dev_w = DeviceBuffer::from_slice(ctx, &packed.w).expect("upload W");
    let dev_z = DeviceBuffer::from_slice(ctx, &packed.z).expect("upload Z");
    let dev_s = DeviceBuffer::from_slice(ctx, &packed.s).expect("upload S");
    let mut dev_c: DeviceBuffer<f16> = DeviceBuffer::zeros(ctx, m * oc).expect("alloc C");
    let desc = Int4AwqGemmDescriptor::new(mi, ici, oci)
        .with_group_size(g)
        .with_split_k_iters(split_k_iters);
    let plan = Int4AwqGemmPlan::<f16>::select(stream, &desc, PlanPreference::default())
        .expect("plan select");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, plan.workspace_size()).expect("alloc workspace");
    // The shape labels below are the plan's documented ones; the element
    // counts equal the OC-major buffers' because `IC / group` is a multiple of 8.
    let args = Int4AwqGemmArgs::<f16> {
        activation: TensorRef::new(dev_a.as_slice(), [mi, ici], contiguous_stride([mi, ici]))
            .unwrap(),
        weight_packed: tref(&dev_w, [oci, ici / 8]),
        scales: TensorRef::new(
            dev_s.as_slice(),
            [ici / g, oci],
            contiguous_stride([ici / g, oci]),
        )
        .unwrap(),
        zeros: tref(&dev_z, [ici / g, oci / 8]),
        output: TensorMut::new(
            dev_c.as_slice_mut(),
            [mi, oci],
            contiguous_stride([mi, oci]),
        )
        .unwrap(),
    };
    plan.run(stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("awq run");
    stream.synchronize().expect("sync");
    let mut host = vec![f16::ZERO; m * oc];
    dev_c.copy_to_host(&mut host).expect("download C");
    host
}

/// Cells whose |got - ref| exceeds 1% of the cell's own sum of |terms| (+ a
/// floor), as `(row, col)`.
fn mismatches(got: &[f16], want: &[f64], abs: &[f64], oc: usize) -> Vec<(usize, usize)> {
    (0..got.len())
        .filter(|&i| {
            let g = got[i].to_f64();
            !g.is_finite() || (g - want[i]).abs() > 0.01 * abs[i] + 1e-3
        })
        .map(|i| (i / oc, i % oc))
        .collect()
}

/// Everything a comparison needs: inputs, packed device buffers and the
/// reference output.
struct Fixture {
    case: Case,
    packed: Packed,
    want: Vec<f64>,
    abs: Vec<f64>,
    shape: Shape,
}

impl Fixture {
    fn run(&self, gpu: &Gpu, packed: &Packed) -> Vec<f16> {
        run_kernel(gpu, &self.case, packed, self.shape)
    }

    fn mismatches(&self, got: &[f16]) -> Vec<(usize, usize)> {
        mismatches(got, &self.want, &self.abs, self.shape.oc)
    }
}

/// Build the case with one group-0 scale tagged (`probe_col`, `probe_val`, a
/// value outside the generated range) so its packed slot can be found by value.
fn build_fixture(shape: Shape, probe_col: usize, probe_val: f16) -> Fixture {
    let Shape { m, ic, oc, group } = shape;
    let mut case = make_case(m, ic, oc, group);
    case.scales[probe_col] = probe_val;
    let (w, z, s) = pack(&case, ic, oc, group);
    let (want, abs) = reference(&case, m, ic, oc, group);
    Fixture {
        case,
        packed: Packed { w, z, s },
        want,
        abs,
        shape,
    }
}

/// Known-bad control: scale the tagged scale by 16 (a x2 error in one of 32
/// groups can hide inside the 1% bound through sign cancellation). Only the
/// tagged column, group 0, is affected.
fn check_corruption_is_detected(gpu: &Gpu, fx: &Fixture, probe_col: usize, probe_val: f16) {
    let Shape { m, ic, oc, group } = fx.shape;
    assert_eq!(fx.packed.s.iter().filter(|&&x| x == probe_val).count(), 1);
    let slot = fx.packed.s.iter().position(|&x| x == probe_val).unwrap();
    let mut corrupted = fx.packed.clone();
    corrupted.s[slot] = f16::from_f32(corrupted.s[slot].to_f32() * 16.0);
    let bad_cells = fx.mismatches(&fx.run(gpu, &corrupted));
    assert!(
        !bad_cells.is_empty(),
        "M={m} IC={ic} OC={oc} G={group}: corrupting a scale did NOT change the result beyond tolerance;          the comparison cannot detect a wrong answer"
    );
    assert!(
        bad_cells.iter().all(|&(_, c)| c == probe_col),
        "M={m} IC={ic} OC={oc} G={group}: corruption leaked outside column {probe_col}: {:?}",
        &bad_cells[..bad_cells.len().min(8)]
    );
}

fn check_shape(m: usize, ic: usize, oc: usize, group: usize) {
    init().expect("driver init");
    let dev = require!(Device::get(0), "a CUDA device (AWQ reference test)");
    let gpu = Gpu::new(&dev);
    let shape = Shape { m, ic, oc, group };
    let (probe_col, probe_val) = (oc / 2 + 3, f16::from_f32(0.0777));
    let fx = build_fixture(shape, probe_col, probe_val);

    let bad = fx.mismatches(&fx.run(&gpu, &fx.packed));
    assert!(
        bad.is_empty(),
        "M={m} IC={ic} OC={oc} G={group}: {} of {} cells mismatch the dequant+GEMM reference; first {:?}",
        bad.len(),
        m * oc,
        &bad[..bad.len().min(8)]
    );
    check_corruption_is_detected(&gpu, &fx, probe_col, probe_val);
}

#[test]
fn awq_matches_dequant_gemm_m1() {
    check_shape(1, 4096, 4096, 128);
}

#[test]
fn awq_matches_dequant_gemm_m16() {
    check_shape(16, 4096, 4096, 128);
}

#[test]
fn awq_matches_dequant_gemm_small() {
    check_shape(16, 1024, 1024, 128);
}

#[test]
fn awq_matches_dequant_gemm_group64() {
    check_shape(16, 4096, 4096, 64);
}

/// `IC / group_size = 10`, not a multiple of 8: the kernel's padded rows.
#[test]
fn awq_matches_dequant_gemm_padded_groups_g128() {
    check_shape(16, 1280, 1024, 128);
}

/// `IC / group_size = 20` at group 64: padded rows with the doubled multiplier.
#[test]
fn awq_matches_dequant_gemm_padded_groups_g64() {
    check_shape(16, 1280, 1024, 64);
}

/// `M > 128`: more than one kernel row block (and the padded split-k staging).
#[test]
fn awq_matches_dequant_gemm_m_over_128() {
    check_shape(130, 1024, 1024, 128);
}
