//! `write_slice` plan — Phase 13.1 trailblazer.
//!
//! `write_slice(dest, source, ranges) -> dest`:
//!
//!   `dest[start_0..end_0, ..., start_{N-1}..end_{N-1}] = source`
//!
//! Assign semantics (not accumulate — that distinguishes
//! [`WriteSlicePlan`] from `ScatterAddPlan`). Drives Fuel team's
//! persistent KV-cache append during autoregressive decoding —
//! step 9c E.3.3 of their Phase 7.6 integration.
//!
//! Dtype coverage spans the entire baracuda element bank via
//! byte-width dispatch (`sizeof(T) ∈ {1, 2, 4, 8, 16}`), with a
//! separate nibble-packed kernel for [`S4`] / [`U4`]. Bound is
//! `T: DeviceRepr + Copy + 'static` (same as [`TensorRef`]) so the
//! same plan covers `Element`-family, `IntElement`-family, and
//! `FpElement`-family dtypes uniformly.
//!
//! No backward — `write_slice` is non-differentiable in Fuel's
//! autograd model.
//!
//! ## Fast paths
//!
//! 1. **Full-width minor axes** — when `ranges[i] == (0, dest_shape[i])`
//!    for all `i > 0`, the source maps to one contiguous chunk of
//!    `dest` starting at offset `start_0 * stride[0] * sizeof(T)`. A
//!    single `cuMemcpyDtoDAsync` does the copy. This is the KV-cache
//!    append shape and the most performance-critical case.
//! 2. **Whole dest covered** — when source-shape == dest-shape and
//!    ranges fully cover dest, a single `cuMemcpyDtoDAsync` of the
//!    whole buffer (degenerate of case 1).
//! 3. **Otherwise** — generic per-slab-element kernel. One thread per
//!    source element computes the dest linear offset from the slab
//!    coord shifted by `range_start`.
//!
//! ## S4 / U4 constraint
//!
//! Nibble-packed dtypes pack two elements per `u8`. To avoid
//! read-modify-write across the byte boundary, the trailblazer
//! requires that `start_{N-1}` and `end_{N-1}` on the innermost axis
//! be **even**. A non-even innermost range returns
//! [`Error::Unsupported`] at `select` time.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, KernelSku, MathPrecision, OpCategory, PlanPreference,
    PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};
use baracuda_types::DeviceRepr;

/// Descriptor for a `write_slice` op.
///
/// `dest_shape[d]` is the per-axis extent of the destination tensor.
/// `source_shape[d]` must equal `ranges[d].1 - ranges[d].0` for every
/// axis (the slab extent). `ranges[d] = (start, end)` selects the
/// inclusive-start / exclusive-end window on axis `d`.
/// `element` is the logical element kind of both tensors (they share
/// dtype). Used to drive byte-width / nibble dispatch.
#[derive(Copy, Clone, Debug)]
pub struct WriteSliceDescriptor<const N: usize> {
    /// Shape of the destination tensor.
    pub dest_shape: [i32; N],
    /// Shape of the source tensor (== `ranges[i].1 - ranges[i].0`
    /// per axis).
    pub source_shape: [i32; N],
    /// Per-axis `(start, end)` window. `0 ≤ start ≤ end ≤ dest_shape[d]`.
    pub ranges: [(i32, i32); N],
    /// Element kind of both tensors. Used to compute the byte width
    /// (and to detect S4 / U4 for the nibble path).
    pub element: ElementKind,
}

/// Args bundle for a `write_slice` launch.
///
/// `dest` is mutated in place. `source` is read once. Both must be
/// contiguous row-major with zero offset relative to their backing
/// device buffer (Fuel's plan layer materializes strided / offset
/// inputs upstream via `Contiguize`).
pub struct WriteSliceArgs<'a, T: DeviceRepr + Copy + 'static, const N: usize> {
    /// Destination tensor — written in the per-axis range window.
    /// Bytes outside the window are untouched.
    pub dest: TensorMut<'a, T, N>,
    /// Source tensor — same dtype as `dest`, shape == slab extent.
    pub source: TensorRef<'a, T, N>,
}

/// `write_slice` plan.
///
/// `dest[start_0..end_0, ..., start_{N-1}..end_{N-1}] = source` —
/// assign (not accumulate). Drives Fuel team's persistent KV-cache
/// append.
///
/// **When to use**: in-place per-axis range write. Distinct from
/// [`ScatterAddPlan`](crate::ScatterAddPlan) (which accumulates
/// per-index) and from [`PadPlan`](crate::PadPlan) (which produces a
/// larger output tensor). No backward — non-differentiable.
///
/// **Dtypes**: every byte-aligned element kind in baracuda's element
/// bank — `f16, bf16, f32, F32Strict, f64, i32, i64, Bool, S8, U8,
/// Fp8E4M3, Fp8E5M2, Complex32, Complex64`. Plus nibble-packed
/// `S4 / U4` with the even-alignment constraint on the innermost axis.
/// `Bin` (1-bit packed) is out of scope.
///
/// **Shape limits**: rank in `[1, 8]`; per-axis
/// `0 ≤ start ≤ end ≤ dest_shape[d]`; `source_shape[d] = end - start`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact (no
/// arithmetic — pure memcpy / index + copy).
pub struct WriteSlicePlan<T: DeviceRepr + Copy + 'static, const N: usize> {
    desc: WriteSliceDescriptor<N>,
    sku: KernelSku,
    byte_width: i32,
    is_nibble: bool,
    /// Fast-path discriminant computed once at `select` time.
    fast_path: FastPath,
    _marker: PhantomData<T>,
}

#[derive(Copy, Clone, Debug)]
enum FastPath {
    /// Source covers exactly the dest (whole-buffer copy).
    WholeDest,
    /// `ranges[i] == (0, dest_shape[i])` for all `i > 0` — the slab is
    /// one contiguous chunk in dest. Offset (in elements) of the
    /// chunk's start is stored.
    ContiguousChunk { dest_offset_elems: i64, source_numel: i64 },
    /// Neither fast path applies — fall through to the generic kernel.
    Generic,
}

impl<T: DeviceRepr + Copy + 'static, const N: usize> WriteSlicePlan<T, N> {
    /// Pick a kernel for `desc`. Validates rank, range bounds, source
    /// shape consistency, dtype coverage, and the nibble-axis-alignment
    /// constraint for S4 / U4. Detects the available fast path.
    pub fn select(
        _stream: &Stream,
        desc: &WriteSliceDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::WriteSlicePlan: rank-0 tensors not supported",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::WriteSlicePlan: tensor rank > 8 not supported",
            ));
        }
        // Validate ranges + source shape.
        for d in 0..N {
            let (s, e) = desc.ranges[d];
            if s < 0 || e < s || e > desc.dest_shape[d] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::WriteSlicePlan: ranges[d] must satisfy \
                     0 <= start <= end <= dest_shape[d]",
                ));
            }
            if desc.source_shape[d] != e - s {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::WriteSlicePlan: source_shape[d] must equal \
                     ranges[d].1 - ranges[d].0",
                ));
            }
            if desc.dest_shape[d] < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::WriteSlicePlan: dest_shape dims must be non-negative",
                ));
            }
        }

        let (byte_width, is_nibble) = match dispatch_kind(desc.element) {
            Some(b) => b,
            None => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::WriteSlicePlan: dtype out of scope. Supported set: \
                     {f16, bf16, f32, F32Strict, f64, i32, i64, Bool, S8, U8, S4, U4, \
                      Fp8E4M3, Fp8E5M2, Complex32, Complex64}",
                ));
            }
        };

        // Nibble-axis-alignment guard. Both start and end on the
        // innermost axis must be even so no byte straddles two halves
        // of the kernel write set.
        if is_nibble {
            let (s, e) = desc.ranges[N - 1];
            if (s & 1) != 0 || (e & 1) != 0 {
                return Err(Error::Unsupported(
                    "baracuda-kernels::WriteSlicePlan: WriteSlice on S4 / U4 requires \
                     even start/end on innermost axis (no read-modify-write at byte \
                     boundary in the trailblazer kernel)",
                ));
            }
            // Also require the innermost dest extent to be even — the
            // nibble byte-shape on the innermost axis is dest_shape/2.
            if (desc.dest_shape[N - 1] & 1) != 0 {
                return Err(Error::Unsupported(
                    "baracuda-kernels::WriteSlicePlan: WriteSlice on S4 / U4 requires \
                     even dest_shape on innermost axis",
                ));
            }
        }

        let fast_path = detect_fast_path::<N>(desc);

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // No arithmetic — pure memcpy + linear write.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::WriteSlice as u16,
            element: desc.element,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            byte_width,
            is_nibble,
            fast_path,
            _marker: PhantomData,
        })
    }

    /// Validate `args` against the descriptor: shapes match, device
    /// buffers are large enough.
    pub fn can_implement(&self, args: &WriteSliceArgs<'_, T, N>) -> Result<()> {
        if args.dest.shape != self.desc.dest_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::WriteSlicePlan: dest shape mismatch with descriptor",
            ));
        }
        if args.source.shape != self.desc.source_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::WriteSlicePlan: source shape mismatch with descriptor",
            ));
        }
        // The kernel assumes both tensors are contiguous row-major.
        if !args.dest.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::WriteSlicePlan: dest must be contiguous row-major",
            ));
        }
        if !args.source.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::WriteSlicePlan: source must be contiguous row-major",
            ));
        }
        // Buffer-size checks. Nibble case: storage element count is
        // numel/2 (rounded up — innermost extent is even by select-time
        // guard, so numel is even too on the nibble path).
        let dest_numel = product_i64(self.desc.dest_shape);
        let source_numel = product_i64(self.desc.source_shape);
        let dest_storage = if self.is_nibble { (dest_numel + 1) / 2 } else { dest_numel };
        let source_storage = if self.is_nibble { (source_numel + 1) / 2 } else { source_numel };
        if (args.dest.data.len() as i64) < dest_storage {
            return Err(Error::BufferTooSmall {
                needed: dest_storage as usize,
                got: args.dest.data.len(),
            });
        }
        if (args.source.data.len() as i64) < source_storage {
            return Err(Error::BufferTooSmall {
                needed: source_storage as usize,
                got: args.source.data.len(),
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch on `stream`. `workspace` is ignored (always zero).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: WriteSliceArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let source_numel = product_i64(self.desc.source_shape);
        if source_numel == 0 {
            return Ok(());
        }
        let dest_ptr_u64 = args.dest.data.as_raw().0;
        let source_ptr_u64 = args.source.data.as_raw().0;
        let stream_ptr = stream.as_raw() as *mut c_void;

        // -------------------- Fast paths --------------------
        match self.fast_path {
            FastPath::WholeDest | FastPath::ContiguousChunk { .. } => {
                // Bytes to copy and per-side offsets:
                //   - source: always starts at offset 0 with source_numel elems
                //   - dest: starts at `dest_offset_elems` (0 for WholeDest)
                let (dest_off_elems, copy_elems) = match self.fast_path {
                    FastPath::WholeDest => (0i64, source_numel),
                    FastPath::ContiguousChunk { dest_offset_elems, source_numel: n } => {
                        (dest_offset_elems, n)
                    }
                    FastPath::Generic => unreachable!(),
                };
                // Byte counts. Nibble: 2 elements per byte (innermost
                // axis alignment is guaranteed even by select-time
                // guard, so both offset and count are integer bytes).
                let (dest_off_bytes, copy_bytes) = if self.is_nibble {
                    (dest_off_elems / 2, copy_elems / 2)
                } else {
                    let bw = self.byte_width as i64;
                    (dest_off_elems * bw, copy_elems * bw)
                };
                return copy_d2d_async(
                    dest_ptr_u64.wrapping_add(dest_off_bytes as u64),
                    source_ptr_u64,
                    copy_bytes as usize,
                    stream_ptr,
                );
            }
            FastPath::Generic => {}
        }

        // -------------------- Generic kernel path --------------------
        let rank = N as i32;
        let dest_shape = self.desc.dest_shape;
        let source_shape = self.desc.source_shape;
        let mut range_start = [0i32; N];
        for d in 0..N {
            range_start[d] = self.desc.ranges[d].0;
        }

        let status = if self.is_nibble {
            // Nibble kernel: shape arrays on the innermost axis are
            // byte-counted (= elements / 2). select() guarantees both
            // innermost dest extent and innermost start are even, so
            // the divisions are exact.
            let mut dest_byte_shape = dest_shape;
            let mut source_byte_shape = source_shape;
            let mut range_start_bytes = range_start;
            dest_byte_shape[N - 1] /= 2;
            source_byte_shape[N - 1] /= 2;
            range_start_bytes[N - 1] /= 2;
            let source_byte_numel = source_numel / 2;
            unsafe {
                baracuda_kernels_sys::baracuda_kernels_write_slice_nibble_run(
                    dest_ptr_u64 as *mut c_void,
                    source_ptr_u64 as *const c_void,
                    source_byte_numel,
                    rank,
                    dest_byte_shape.as_ptr(),
                    source_byte_shape.as_ptr(),
                    range_start_bytes.as_ptr(),
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            }
        } else {
            // Byte-aligned: dispatch on byte width.
            unsafe {
                let dest = dest_ptr_u64 as *mut c_void;
                let source = source_ptr_u64 as *const c_void;
                let ds = dest_shape.as_ptr();
                let ss = source_shape.as_ptr();
                let rs = range_start.as_ptr();
                match self.byte_width {
                    1 => baracuda_kernels_sys::baracuda_kernels_write_slice_b1_run(
                        dest, source, source_numel, rank, ds, ss, rs,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    2 => baracuda_kernels_sys::baracuda_kernels_write_slice_b2_run(
                        dest, source, source_numel, rank, ds, ss, rs,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    4 => baracuda_kernels_sys::baracuda_kernels_write_slice_b4_run(
                        dest, source, source_numel, rank, ds, ss, rs,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    8 => baracuda_kernels_sys::baracuda_kernels_write_slice_b8_run(
                        dest, source, source_numel, rank, ds, ss, rs,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    16 => baracuda_kernels_sys::baracuda_kernels_write_slice_b16_run(
                        dest, source, source_numel, rank, ds, ss, rs,
                        core::ptr::null_mut(), 0, stream_ptr,
                    ),
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::WriteSlicePlan::run: unsupported byte width \
                         (select() should have caught this)",
                    )),
                }
            }
        };
        map_status(status)
    }
}

/// Per-`ElementKind` byte width + nibble-flag mapping. Returns `None`
/// for unsupported kinds (today: `Bin`).
fn dispatch_kind(k: ElementKind) -> Option<(i32, bool)> {
    Some(match k {
        ElementKind::Bool => (1, false),
        ElementKind::S8 => (1, false),
        ElementKind::U8 => (1, false),
        ElementKind::Fp8E4M3 => (1, false),
        ElementKind::Fp8E5M2 => (1, false),
        ElementKind::F16 => (2, false),
        ElementKind::Bf16 => (2, false),
        ElementKind::F32 => (4, false),
        ElementKind::F32Strict => (4, false),
        ElementKind::I32 => (4, false),
        ElementKind::F64 => (8, false),
        ElementKind::I64 => (8, false),
        ElementKind::Complex32 => (8, false),
        ElementKind::Complex64 => (16, false),
        ElementKind::S4 => (1, true),
        ElementKind::U4 => (1, true),
        // Bin (1-bit packed) is out of scope — distinct packing model.
        ElementKind::Bin => return None,
    })
}

fn detect_fast_path<const N: usize>(desc: &WriteSliceDescriptor<N>) -> FastPath {
    // WholeDest: ranges cover every axis fully and source_shape == dest_shape.
    let mut whole = true;
    for d in 0..N {
        let (s, e) = desc.ranges[d];
        if s != 0 || e != desc.dest_shape[d] {
            whole = false;
            break;
        }
    }
    if whole {
        return FastPath::WholeDest;
    }

    // ContiguousChunk: ranges[i] == (0, dest_shape[i]) for all i > 0.
    // The slab is one contiguous block in dest's row-major layout
    // starting at `start_0 * (product of dest_shape[1..])` elements.
    if N == 1 {
        // Rank-1 partial — contiguous chunk by definition (just one axis).
        let (s, _) = desc.ranges[0];
        let source_numel = product_i64(desc.source_shape);
        return FastPath::ContiguousChunk {
            dest_offset_elems: s as i64,
            source_numel,
        };
    }
    let mut minors_full = true;
    for d in 1..N {
        let (s, e) = desc.ranges[d];
        if s != 0 || e != desc.dest_shape[d] {
            minors_full = false;
            break;
        }
    }
    if minors_full {
        let mut minor_prod: i64 = 1;
        for d in 1..N {
            minor_prod = minor_prod.saturating_mul(desc.dest_shape[d] as i64);
        }
        let start_0 = desc.ranges[0].0 as i64;
        let source_numel = product_i64(desc.source_shape);
        return FastPath::ContiguousChunk {
            dest_offset_elems: start_0 * minor_prod,
            source_numel,
        };
    }
    FastPath::Generic
}

#[inline]
fn product_i64<const N: usize>(shape: [i32; N]) -> i64 {
    let mut p: i64 = 1;
    for d in 0..N {
        p = p.saturating_mul(shape[d] as i64);
    }
    p
}

/// Device-to-device async copy on `stream`. Thin wrapper around
/// `cuMemcpyDtoDAsync_v2` — matches the same pattern used by the
/// `kthvalue` plan's H2D / D2H helpers.
fn copy_d2d_async(
    dst_dev: u64,
    src_dev: u64,
    bytes: usize,
    stream: *mut c_void,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[allow(non_camel_case_types)]
    type CUresult = i32;
    unsafe extern "system" {
        fn cuMemcpyDtoDAsync_v2(
            dst_device: u64,
            src_device: u64,
            byte_count: usize,
            h_stream: *mut c_void,
        ) -> CUresult;
    }
    let status = unsafe { cuMemcpyDtoDAsync_v2(dst_dev, src_dev, bytes, stream) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}

fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
