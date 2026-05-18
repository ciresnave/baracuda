//! Sparsemax forward plan — projection onto the probability simplex.
//!
//! `y = ProjSimplex(x)` via the standard sort-then-threshold closed
//! form: sort logits descending, find the largest `k` such that
//! `1 + (k+1) · x_sorted[k] > Σ_{i ≤ k} x_sorted[i]`, set
//! `τ = (Σ_{i ≤ k_max} x_sorted[i] - 1) / (k_max + 1)`, and emit
//! `y[i] = max(0, x[i] - τ)`.
//!
//! Wired today: `T ∈ {f32, f16, bf16, f64}`. Row extent (softmax axis
//! size) limited to 1024 (Phase 11.6) — the kernel uses a
//! `cub::BlockRadixSort` + `cub::BlockScan` block-cooperative
//! algorithm: one CUDA thread block per row, 256 threads per block,
//! and two compiled tile-size specializations
//! (`ITEMS_PER_THREAD = 1` for extents ≤ 256, `ITEMS_PER_THREAD = 4`
//! for extents 257..=1024). Larger extents return `Unsupported` at
//! plan-select time.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SoftmaxKind, TensorMut, TensorRef, Workspace,
};

/// Maximum supported extent along the sparsemax axis. Phase 11.6 lifts
/// this from the trailblazer 64 → 1024 by switching the forward kernel
/// from a per-thread serial sort to a block-cooperative
/// `cub::BlockRadixSort` + `cub::BlockScan`. Mirrors the C++ macro
/// `BARACUDA_SPARSEMAX_MAX_EXTENT`.
pub const SPARSEMAX_MAX_EXTENT: i32 = 1024;

/// Descriptor for a Sparsemax forward op.
#[derive(Copy, Clone, Debug)]
pub struct SparsemaxDescriptor<const N: usize> {
    /// Tensor shape (input and output share it).
    pub input_shape: [i32; N],
    /// Axis along which to apply sparsemax. Must be in `[0, N)`.
    /// Extent along this axis must be `<= SPARSEMAX_MAX_EXTENT`.
    pub softmax_axis: u8,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Sparsemax forward launch.
pub struct SparsemaxArgs<'a, T: Element, const N: usize> {
    /// Input logits.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — same shape as input.
    pub y: TensorMut<'a, T, N>,
}

/// Sparsemax forward plan.
pub struct SparsemaxPlan<T: Element, const N: usize> {
    desc: SparsemaxDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> SparsemaxPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SparsemaxDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SparsemaxPlan: descriptor element != T",
            ));
        }
        if (desc.softmax_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SparsemaxPlan: softmax_axis out of range",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SparsemaxPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SparsemaxPlan: tensor rank > 8 not supported",
            ));
        }
        let extent = desc.input_shape[desc.softmax_axis as usize];
        if extent > SPARSEMAX_MAX_EXTENT {
            return Err(Error::Unsupported(
                "baracuda-kernels::SparsemaxPlan: extent along softmax_axis > 1024 \
                 not supported (block-cooperative BlockRadixSort tile capped at 1024)",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::SparsemaxPlan: wired today: {f32, f16, bf16, f64}",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: match T::KIND {
                ElementKind::F64 => ElementKind::F64,
                _ => ElementKind::F32,
            },
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Softmax,
            op: SoftmaxKind::Sparsemax as u16,
            element: T::KIND,
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
    pub fn can_implement(&self, args: &SparsemaxArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SparsemaxPlan: x shape mismatch",
            ));
        }
        if args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SparsemaxPlan: y shape mismatch",
            ));
        }
        let numel = args.x.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if x_len < numel || y_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: x_len.min(y_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Kernel SKU identity.
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
        args: SparsemaxArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.x.numel();
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let axis = self.desc.softmax_axis as usize;
        let shape = self.desc.input_shape;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;
        let extent = shape[axis];
        let stride_x_axis = stride_x[axis];
        let stride_y_axis = stride_y[axis];

        macro_rules! dispatch {
            ($sym:ident) => {
                unsafe {
                    baracuda_kernels_sys::$sym(
                        numel,
                        rank,
                        shape.as_ptr(),
                        stride_x.as_ptr(),
                        stride_y.as_ptr(),
                        axis as i32,
                        extent,
                        stride_x_axis,
                        stride_y_axis,
                        x_ptr,
                        y_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            };
        }
        let status = match T::KIND {
            ElementKind::F32 => dispatch!(baracuda_kernels_sparsemax_f32_run),
            ElementKind::F16 => dispatch!(baracuda_kernels_sparsemax_f16_run),
            ElementKind::Bf16 => dispatch!(baracuda_kernels_sparsemax_bf16_run),
            ElementKind::F64 => dispatch!(baracuda_kernels_sparsemax_f64_run),
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SparsemaxPlan::run unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
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
