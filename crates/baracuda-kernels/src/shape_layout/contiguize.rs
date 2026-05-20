//! `contiguize` plan — strided→contiguous materialization
//! (`torch.Tensor.contiguous()`). Phase 13.2.
//!
//! Closes the D2H→CPU contiguize→H2D fallback cliff in Fuel's CUDA
//! backend for non-contiguous CUDA inputs. The kernel is byte-level
//! dtype-agnostic: a single sizeof(T)-templated CUDA body covers every
//! byte-aligned dtype (f16, bf16, f32, f64, F32Strict, i32, i64, Bool,
//! S8, U8, Fp8E4M3, Fp8E5M2, Complex32, Complex64). A separate nibble
//! kernel handles S4 / U4 with a documented innermost-stride
//! constraint.
//!
//! Three host-side fast paths land in this trailblazer:
//!   1. Source already contiguous + zero offset → single
//!      `cudaMemcpyAsync(DeviceToDevice)`.
//!   2. Innermost stride == 1 → per-outer-coord contiguous-run kernel.
//!   3. Generic per-element kernel (one thread per output element).
//!
//! No backward — strides are metadata, autograd doesn't observe
//! contiguize.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, KernelSku, MathPrecision, OpCategory, PlanPreference,
    PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};
use baracuda_types::DeviceRepr;

/// Descriptor for a Contiguize op.
///
/// `shape` is the logical output shape (equal to the source's logical
/// shape — Contiguize is a layout op, not a shape transform).
/// `source_strides` are the SIGNED element strides of the source view
/// (signed because Flip ops produce negatives and BroadcastTo produces
/// zeros). `source_offset` is the source-side base offset **in
/// elements**, not bytes.
#[derive(Copy, Clone, Debug)]
pub struct ContiguizeDescriptor<const N: usize> {
    /// Logical tensor shape (source and dest agree).
    pub shape: [i32; N],
    /// Source-view element strides. Signed so Flip-produced negative
    /// strides and BroadcastTo-produced zero strides are both legal.
    pub source_strides: [i64; N],
    /// Source-view base offset, **in elements** (not bytes). The
    /// kernel scales by `sizeof(T)` internally.
    pub source_offset: i64,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Contiguize launch.
///
/// **Stride-field convention**: the kernel does NOT use `source.stride`
/// — it uses `ContiguizeDescriptor::source_strides`. The `TensorRef`'s
/// `stride` field is a load-bearing descriptor for OTHER ops; here it
/// MAY be the canonical contiguous stride of the underlying buffer or
/// the descriptor's strides. The kernel treats `source.data` as a flat
/// element buffer and indexes it via descriptor strides + element-
/// scaled offset.
///
/// `dest` MUST be a contiguous, zero-offset allocation. This is enforced
/// at `can_implement` time (`dest.is_contiguous()` + no offset is
/// implicit because dest is a raw [`TensorMut`] over a fresh buffer).
pub struct ContiguizeArgs<'a, T: DeviceRepr + Copy + 'static, const N: usize> {
    /// Source: raw byte buffer. The kernel reads
    /// `source.data[source_offset + Σ i_k * source_strides[k]]` for
    /// each output coord `(i_0, …, i_{N-1})`.
    pub source: TensorRef<'a, T, N>,
    /// Destination: contiguous, zero-offset, caller-allocated.
    pub dest: TensorMut<'a, T, N>,
}

/// `contiguize` plan.
///
/// `dest = source.contiguous()` — materializes a row-major contiguous
/// tensor from an arbitrary strided source view. Single fast-path
/// dispatch (already-contig / inner-stride-1 / generic) baked into the
/// CUDA launcher.
///
/// **When to use**: a downstream kernel requires contiguous input but
/// the upstream tensor has a non-canonical layout (transpose, slice,
/// flip, broadcast). Today Fuel's CUDA backend D2H→CPU-contiguize→H2D
/// in this case; this plan keeps the work device-resident.
///
/// **Dtypes**: every byte-aligned dtype baracuda's element bank
/// exposes — `{f16, bf16, f32, f64, F32Strict, i32, i64, Bool, S8, U8,
/// Fp8E4M3, Fp8E5M2, Complex32, Complex64}` — plus nibble-packed
/// `{S4, U4}` with an innermost-stride constraint. `Bin` (1-bit) is
/// **out of scope** for Phase 13.2.
///
/// **Nibble constraint (S4 / U4)**: the source's innermost stride MUST
/// be one of `{1, -1, 2}` — anything else breaks nibble alignment and
/// returns [`Error::Unsupported`] at run time. The dest is always
/// nibble-aligned by construction (zero offset + canonical strides).
///
/// **Shape limits**: rank in `[1, 8]`; `shape[d]` non-negative; strides
/// may be any signed i64 (including 0 and negative).
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact —
/// pure copy, no arithmetic.
pub struct ContiguizePlan<T: DeviceRepr + Copy + 'static, const N: usize> {
    desc: ContiguizeDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: DeviceRepr + Copy + 'static, const N: usize> ContiguizePlan<T, N> {
    /// Pick a kernel for `desc`.
    ///
    /// Unlike most `*Plan` types in this crate, the type parameter `T`
    /// is bounded by [`baracuda_types::DeviceRepr`] only (no `Element` /
    /// `IntElement` family) so the byte-level-agnostic kernel can serve
    /// `T = S4 / U4 / S8 / U8 / Fp8E4M3 / Fp8E5M2` alongside the
    /// classic `Element`s (`f16` / `bf16` / `f32` / `f64` / `i32` /
    /// `i64` / `Bool` / `Complex32` / `Complex64` / `F32Strict`).
    /// `desc.element` is the source of truth for dispatch.
    pub fn select(
        _stream: &Stream,
        desc: &ContiguizeDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        // Defensive: caller must size-match T to desc.element. We can't
        // statically verify this without a `KIND` const on every
        // `DeviceRepr` type, so we check that `sizeof::<T>` matches the
        // expected byte width of `desc.element`.
        if !type_size_matches_kind::<T>(desc.element) {
            return Err(Error::Unsupported(
                "baracuda-kernels::ContiguizePlan: sizeof::<T> doesn't match \
                 descriptor element kind (T must be the Rust type that backs \
                 desc.element — see ContiguizeDescriptor docs)",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ContiguizePlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ContiguizePlan: tensor rank > 8 not supported",
            ));
        }
        // Coverage matrix — every byte-aligned baracuda dtype plus
        // nibble-packed S4 / U4. Reject Bin (1-bit) — out of scope.
        let supported = matches!(
            desc.element,
            ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
                | ElementKind::I32
                | ElementKind::I64
                | ElementKind::Bool
                | ElementKind::S8
                | ElementKind::U8
                | ElementKind::Fp8E4M3
                | ElementKind::Fp8E5M2
                | ElementKind::Complex32
                | ElementKind::Complex64
                | ElementKind::S4
                | ElementKind::U4
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ContiguizePlan: dtype not in coverage \
                 (Bin is out of scope; everything else byte-aligned or nibble-packed)",
            ));
        }
        // S4 / U4 innermost-stride alignment check — duplicated on the
        // device side defensively, but we surface it earlier here so
        // mis-configured plans fail at select() rather than run().
        if matches!(desc.element, ElementKind::S4 | ElementKind::U4) && N >= 1 {
            let inner = desc.source_strides[N - 1];
            if !(inner == 1 || inner == -1 || inner == 2) {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ContiguizePlan: S4 / U4 source's innermost \
                     stride must be one of {1, -1, 2} for nibble alignment",
                ));
            }
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Pure copy — no arithmetic, hence trivially bit-stable.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Contiguize as u16,
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
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &ContiguizeArgs<'_, T, N>) -> Result<()> {
        if args.source.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ContiguizePlan: source shape mismatch with descriptor",
            ));
        }
        if args.dest.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ContiguizePlan: dest shape mismatch with descriptor",
            ));
        }
        if !args.dest.is_contiguous() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ContiguizePlan: dest must be canonical contiguous \
                 (the whole point of this op is to MATERIALIZE a contiguous view)",
            ));
        }
        let numel = args.dest.numel();
        let dest_len = args.dest.data.len() as i64;
        // Nibble-packed dtypes (S4/U4) store two logical elements per
        // byte/storage-slot, so the storage requirement is
        // ceil(numel/2), not numel.
        let needed_storage = match self.desc.element {
            ElementKind::S4 | ElementKind::U4 => (numel + 1) / 2,
            _ => numel,
        };
        if dest_len < needed_storage {
            return Err(Error::BufferTooSmall {
                needed: needed_storage as usize,
                got: dest_len as usize,
            });
        }
        // Source buffer length sanity: numel must fit, but we don't
        // tightly bound because the source view may legitimately
        // index well past `source.data.len()` once `source_offset`
        // and signed strides are applied (the buffer behind a strided
        // view is opaque to baracuda-kernels). The caller owns this
        // contract.
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
    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: ContiguizeArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dest.numel();
        if numel == 0 {
            return Ok(());
        }
        let source_ptr = args.source.data.as_raw().0 as *const c_void;
        let dest_ptr = args.dest.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let shape = self.desc.shape;
        let source_strides = self.desc.source_strides;
        let source_offset = self.desc.source_offset;
        let rank = N as i32;

        // Byte-width dispatch. Each ElementKind picks a single FFI
        // symbol — the kernel body is dtype-agnostic at that point
        // (raw byte memcpy of `ElemBytes`).
        let status = match self.desc.element {
            // 1-byte payloads.
            ElementKind::Bool
            | ElementKind::S8
            | ElementKind::U8
            | ElementKind::Fp8E4M3
            | ElementKind::Fp8E5M2 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_contiguize_b1_run(
                    dest_ptr,
                    source_ptr,
                    shape.as_ptr(),
                    source_strides.as_ptr(),
                    source_offset,
                    rank,
                    stream_ptr,
                )
            },
            // 2-byte payloads.
            ElementKind::F16 | ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_contiguize_b2_run(
                    dest_ptr,
                    source_ptr,
                    shape.as_ptr(),
                    source_strides.as_ptr(),
                    source_offset,
                    rank,
                    stream_ptr,
                )
            },
            // 4-byte payloads.
            ElementKind::F32 | ElementKind::F32Strict | ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_contiguize_b4_run(
                    dest_ptr,
                    source_ptr,
                    shape.as_ptr(),
                    source_strides.as_ptr(),
                    source_offset,
                    rank,
                    stream_ptr,
                )
            },
            // 8-byte payloads — f64 (8 B), i64 (8 B), Complex32 (4+4 B).
            ElementKind::F64 | ElementKind::I64 | ElementKind::Complex32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_contiguize_b8_run(
                    dest_ptr,
                    source_ptr,
                    shape.as_ptr(),
                    source_strides.as_ptr(),
                    source_offset,
                    rank,
                    stream_ptr,
                )
            },
            // 16-byte payloads — Complex64 (8+8 B).
            ElementKind::Complex64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_contiguize_b16_run(
                    dest_ptr,
                    source_ptr,
                    shape.as_ptr(),
                    source_strides.as_ptr(),
                    source_offset,
                    rank,
                    stream_ptr,
                )
            },
            // Nibble-packed (S4 / U4) — single symbol with documented
            // innermost-stride constraint enforced device-side.
            ElementKind::S4 | ElementKind::U4 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_contiguize_nibble_run(
                    dest_ptr,
                    source_ptr,
                    shape.as_ptr(),
                    source_strides.as_ptr(),
                    source_offset,
                    rank,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ContiguizePlan::run: dtype not in coverage \
                     (Bin / unknown — should have been rejected at select())",
                ));
            }
        };
        map_status(status)
    }
}

/// Defensive size-vs-kind check used by `select()`. Returns `true` iff
/// `core::mem::size_of::<T>()` matches the on-device byte width
/// expected for `kind`. The plan's CUDA dispatch picks the byte-width
/// FFI symbol from `desc.element`, so a mismatch between `T` and
/// `desc.element` would silently scribble memory; we reject at select
/// time instead.
fn type_size_matches_kind<T>(kind: ElementKind) -> bool {
    let want = match kind {
        ElementKind::Bool
        | ElementKind::S8
        | ElementKind::U8
        | ElementKind::Fp8E4M3
        | ElementKind::Fp8E5M2
        | ElementKind::S4
        | ElementKind::U4 => 1,
        ElementKind::F16 | ElementKind::Bf16 => 2,
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::I32 => 4,
        ElementKind::F64 | ElementKind::I64 | ElementKind::Complex32 => 8,
        ElementKind::Complex64 => 16,
        ElementKind::Bin => return false, // out of scope
    };
    core::mem::size_of::<T>() == want
}

fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration \
             (likely S4 / U4 source innermost stride not in {1, -1, 2})",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
