//! Multi-M GGUF MMVQ — Phase 33 prefill-MMVQ port.
//!
//! Computes `M` matrix-vector multiplies against a single shared weight
//! matrix in one kernel launch, reusing each weight gmem read across
//! all `M` activation vectors. The kernel design is ported from
//! llama.cpp's `mul_mat_vec_q<ncols_y, ...>` template via Fuel; see
//! `crates/baracuda-kernels-sys/kernels/include/baracuda_mmvq_multim.cuh`
//! for the kernel-level lineage.
//!
//! Op shape:
//!
//! ```text
//! out[m, r] = Σ_c W_q[r, c] · y[m, c]    for m ∈ [0, M), r ∈ [0, nrows)
//! ```
//!
//! Where `W_q` is the GGUF-packed weight matrix (one matrix shared
//! across all M activations) and `y` is the fp activation matrix
//! (`M × ncols`). The plan internally stages `y` into the Q8_1 block
//! format before dispatching the multi-M dot kernel. Output `out`
//! is fp32, shape `[M, nrows]`.
//!
//! ## Scope (Phase 33)
//!
//! - **Block format**: `Q8_0` only. Remaining 9 GGUF formats follow
//!   in a future phase — each needs a per-format `vec_dot_q*_q8_1`
//!   helper plus the per-format constants `(qk, qi, vdr)`.
//! - **Activation dtypes**: f32 / f16 / bf16 (cast to f32 during the
//!   Q8_1 staging step).
//! - **Output dtype**: f32 only (the multi-M kernel writes f32 directly;
//!   downstream cast left to the caller).
//! - **M values**: 1, 2, 4, 8. M > 8 falls through to a per-token loop
//!   over the M=1 kernel (with the same staged activations).
//!
//! ## When to use
//!
//! - **Prefill** (M ∈ [2, 8]): up to 8× gmem bandwidth save vs the
//!   per-token M=1 dispatch in [`crate::GgufMmvqPlan`]. Target speedup
//!   3-7× on a 7B-class model.
//! - **Decode** (M == 1): comparable to the M=1 path (no weight reuse
//!   to exploit), but the staging step + DP4A dot can still be faster
//!   than the fp-dequant baseline when memory bandwidth is plentiful
//!   relative to SM throughput.
//!
//! ## Numerical equivalence
//!
//! The staging path quantizes activations into 8-bit before the dot;
//! relative error is bounded by ~1e-3 per dot (depends on activation
//! distribution). Bit-exact equivalence with the M=1 FP path is NOT
//! provided — see kernel header for details.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, GgufBlockFormat, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace, U8,
};

use crate::quantize::gguf::mmvq::GgufMmvqActivation;
use crate::quantize::map_status;

/// Descriptor for a multi-M GGUF MMVQ op.
#[derive(Copy, Clone, Debug)]
pub struct GgufMmvqMultiMDescriptor {
    /// Output rows = packed weight matrix rows.
    pub nrows: i32,
    /// Unpacked weight columns = activation K. Must be a multiple of 32
    /// (the Q8_1 block size and Q8_0 weight block size).
    pub ncols: i32,
    /// Number of activation vectors. Each ≤ 8 vector launches one
    /// compile-time-specialized kernel; M > 8 fans out to multiple
    /// chained launches (currently: a sequence of M=8 kernels and a
    /// trailing M ∈ {1, 2, 4} cleanup).
    pub m: i32,
    /// GGUF block format. Phase 33 supports only `Q8_0`; other formats
    /// return `Error::Unsupported` from `select`.
    pub block_format: GgufBlockFormat,
    /// Byte offset into the `weight` allocation at which this matrix
    /// starts. Mirrors `GgufMmvqDescriptor::w_start_byte_offset`.
    pub w_start_byte_offset: i64,
}

impl Default for GgufMmvqMultiMDescriptor {
    fn default() -> Self {
        Self {
            nrows: 0,
            ncols: 0,
            m: 1,
            block_format: GgufBlockFormat::Q8_0,
            w_start_byte_offset: 0,
        }
    }
}

/// Args bundle for a multi-M GGUF MMVQ launch.
pub struct GgufMmvqMultiMArgs<'a, T: GgufMmvqActivation = f32> {
    /// Packed GGUF weight bytes (single matrix).
    pub weight: TensorRef<'a, U8, 1>,
    /// Activations, shape `[M, ncols]`, contiguous row-major.
    pub activations: TensorRef<'a, T, 2>,
    /// Output, shape `[M, nrows]` f32.
    pub output: TensorMut<'a, f32, 2>,
}

/// Multi-M MMVQ plan.
///
/// See the module docs for the full op shape, scope, and numerical
/// caveats. Workspace bytes are the staged Q8_1 activation buffer
/// (`M × ceil(ncols / 32) × 36`).
pub struct GgufMmvqMultiMPlan<T: GgufMmvqActivation = f32> {
    desc: GgufMmvqMultiMDescriptor,
    sku: KernelSku,
    workspace_bytes: usize,
    _phantom: PhantomData<T>,
}

impl<T: GgufMmvqActivation> GgufMmvqMultiMPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &GgufMmvqMultiMDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.nrows < 0 || desc.ncols < 0 || desc.m < 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqMultiMPlan: nrows / ncols / m must be non-negative",
            ));
        }
        if desc.block_format != GgufBlockFormat::Q8_0 {
            return Err(Error::Unsupported(
                "GgufMmvqMultiMPlan: only Q8_0 is supported in Phase 33 \
                 (remaining 9 block formats land in a future phase)",
            ));
        }
        // Q8_0 block size = 32; multi-M kernel requires ncols % 32 == 0.
        if desc.ncols % 32 != 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqMultiMPlan: ncols must be a multiple of 32",
            ));
        }
        if desc.w_start_byte_offset < 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqMultiMPlan: w_start_byte_offset must be non-negative",
            ));
        }

        // Workspace = staged Q8_1 activations for M rows of ncols cols.
        // = M × ceil(ncols / 32) × 36 bytes per block.
        let blocks_per_row = ((desc.ncols + 31) / 32) as usize;
        let workspace_bytes = (desc.m as usize) * blocks_per_row * 36;

        Ok(Self {
            desc: *desc,
            sku: build_sku(T::KIND),
            workspace_bytes,
            _phantom: PhantomData,
        })
    }

    /// Validate args against the plan.
    pub fn can_implement(&self, args: &GgufMmvqMultiMArgs<'_, T>) -> Result<()> {
        if args.activations.shape != [self.desc.m, self.desc.ncols] {
            return Err(Error::InvalidProblem(
                "GgufMmvqMultiMPlan: activations shape != [M, ncols]",
            ));
        }
        if args.output.shape != [self.desc.m, self.desc.nrows] {
            return Err(Error::InvalidProblem(
                "GgufMmvqMultiMPlan: output shape != [M, nrows]",
            ));
        }
        // Stride checks: kernel assumes activations + output are contig
        // along the inner dim (= K and = nrows respectively).
        if args.activations.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "GgufMmvqMultiMPlan: activations must be contig along K",
            ));
        }
        if args.output.stride[1] != 1 {
            return Err(Error::InvalidProblem(
                "GgufMmvqMultiMPlan: output must be contig along nrows",
            ));
        }
        let bs = self.desc.block_format.block_size() as i32;
        let blocks_per_row = self.desc.ncols / bs;
        let expected_bytes =
            (self.desc.nrows as i64) * (blocks_per_row as i64) * (self.desc.block_format.type_size() as i64);
        let weight_len_bytes = args.weight.shape[0] as i64;
        let need_bytes = self.desc.w_start_byte_offset + expected_bytes;
        if weight_len_bytes < need_bytes {
            return Err(Error::InvalidProblem(
                "GgufMmvqMultiMPlan: weight byte length < offset + nrows * blocks_per_row * type_size",
            ));
        }
        Ok(())
    }

    /// Workspace bytes (staged Q8_1 activation buffer).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes
    }

    /// Identity of the selected kernel.
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
        workspace: Workspace<'_>,
        args: GgufMmvqMultiMArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.nrows == 0 || self.desc.ncols == 0 || self.desc.m == 0 {
            return Ok(());
        }

        // Workspace: staged Q8_1 activation buffer.
        let need = self.workspace_bytes;
        let ws_ptr = match workspace {
            Workspace::None => {
                if need > 0 {
                    return Err(Error::WorkspaceTooSmall { needed: need, got: 0 });
                }
                core::ptr::null_mut()
            }
            Workspace::Borrowed(slice) => {
                let got = slice.len();
                if got < need {
                    return Err(Error::WorkspaceTooSmall { needed: need, got });
                }
                slice.as_raw().0 as *mut c_void
            }
        };

        let w_ptr = args.weight.data.as_raw().0 as *const c_void;
        let y_ptr = args.activations.data.as_raw().0 as *const c_void;
        let dst_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let ncols = self.desc.ncols;
        let nrows = self.desc.nrows;
        let w_off = self.desc.w_start_byte_offset;

        // Step 1: stage activations into Q8_1.
        let stage_status = unsafe {
            stage_q8_1::<T>(
                ncols as i64,
                self.desc.m as i64,
                y_ptr,
                ws_ptr,
                stream_ptr,
            )
        };
        map_status(stage_status)?;

        // Step 2: dispatch the multi-M MMVQ kernel(s). For M in {1, 2, 4, 8}
        // we use the compile-time-specialized launcher directly. For larger
        // M we tile into 8-wide blocks plus a trailing power-of-two cleanup.
        let mut m_done = 0i32;
        while m_done < self.desc.m {
            let m_remaining = self.desc.m - m_done;
            let m_chunk = pick_chunk_size(m_remaining);

            // Pointers offset by m_done * (per-row size).
            let blocks_per_row = ((ncols + 31) / 32) as i64;
            let ws_row_bytes = blocks_per_row * 36;
            let chunk_ws_ptr = unsafe {
                (ws_ptr as *mut u8).offset((m_done as i64 * ws_row_bytes) as isize)
            } as *const c_void;
            let chunk_dst_ptr = unsafe {
                (dst_ptr as *mut f32).offset((m_done as isize) * (nrows as isize))
            } as *mut c_void;

            let status = unsafe {
                dispatch_q8_0_multim(
                    m_chunk, ncols, nrows, w_ptr, w_off, chunk_ws_ptr, chunk_dst_ptr, stream_ptr,
                )
            };
            map_status(status)?;
            m_done += m_chunk;
        }

        Ok(())
    }
}

/// Decide which compile-time M to dispatch for a given remaining count.
/// Greedy power-of-two: use 8 → 4 → 2 → 1.
fn pick_chunk_size(remaining: i32) -> i32 {
    if remaining >= 8 {
        8
    } else if remaining >= 4 {
        4
    } else if remaining >= 2 {
        2
    } else {
        1
    }
}

/// Dispatch the Q8_1 staging kernel for the given activation dtype.
unsafe fn stage_q8_1<T: GgufMmvqActivation>(
    kx: i64,
    ny: i64,
    src: *const c_void,
    dst: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    match T::KIND {
        ElementKind::F32 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_quantize_q8_1_f32_run(
                kx, ny, src, dst, stream,
            )
        },
        ElementKind::F16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_quantize_q8_1_f16_run(
                kx, ny, src, dst, stream,
            )
        },
        ElementKind::Bf16 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_quantize_q8_1_bf16_run(
                kx, ny, src, dst, stream,
            )
        },
        _ => 3, // GgufMmvqActivation is sealed to f32 / f16 / bf16.
    }
}

/// Dispatch to the right compile-time-specialized M kernel.
#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_q8_0_multim(
    m: i32,
    ncols: i32,
    nrows: i32,
    w_ptr: *const c_void,
    w_off: i64,
    activations_q8_1: *const c_void,
    dst: *mut c_void,
    stream: *mut c_void,
) -> i32 {
    let ws = core::ptr::null_mut();
    match m {
        1 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_mmvq_multim_q8_0_m1_run(
                ncols, nrows, w_ptr, w_off, activations_q8_1, dst, ws, 0, stream,
            )
        },
        2 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_mmvq_multim_q8_0_m2_run(
                ncols, nrows, w_ptr, w_off, activations_q8_1, dst, ws, 0, stream,
            )
        },
        4 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_mmvq_multim_q8_0_m4_run(
                ncols, nrows, w_ptr, w_off, activations_q8_1, dst, ws, 0, stream,
            )
        },
        8 => unsafe {
            baracuda_kernels_sys::baracuda_kernels_mmvq_multim_q8_0_m8_run(
                ncols, nrows, w_ptr, w_off, activations_q8_1, dst, ws, 0, stream,
            )
        },
        _ => 2, // pick_chunk_size only yields 1/2/4/8.
    }
}

fn build_sku(act_kind: ElementKind) -> KernelSku {
    KernelSku {
        category: OpCategory::Quantization,
        op: QuantizeKind::GgufMmvq as u16,
        element: act_kind,
        aux_element: Some(ElementKind::U8),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee: PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Within a fixed M (single compile-time kernel) the result is
            // bit-stable on identical hardware (single-warp / nwarps reduction,
            // no atomics). The Q8_1 staging step is itself deterministic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        },
    }
}

