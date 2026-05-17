//! Single-axis `count_nonzero` reduction. Heterogeneous output dtype:
//! input is `T: Element`, output is always `i64` (PyTorch convention —
//! `torch.count_nonzero` returns int64).
//!
//! Sibling of [`crate::ReducePlan`] (same-dtype-in-out),
//! [`crate::ArgReducePlan`] (i64 index out), and
//! [`crate::BoolReducePlan`] (Bool out for Any / All). Lives in its own
//! plan shape because of the heterogeneous output. PyTorch parity:
//! `torch.count_nonzero(x, dim=k)`.
//!
//! Wired matrix: `CountNonzero × {f32, f16, bf16, f64, i32, i64, Bool}`
//! — 7 SKUs. NaN is counted as non-zero (`NaN != 0` is true — matches
//! PyTorch / IEEE 754). Non-differentiable; no backward plan ships.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ReduceKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a count-nonzero axis reduction.
#[derive(Copy, Clone, Debug)]
pub struct CountReduceDescriptor<const N: usize> {
    /// Must be [`ReduceKind::CountNonzero`].
    pub kind: ReduceKind,
    /// Input tensor shape.
    pub input_shape: [i32; N],
    /// Axis to reduce along.
    pub reduce_axis: u8,
    /// Input element type.
    pub element: ElementKind,
}

impl<const N: usize> CountReduceDescriptor<N> {
    /// Output shape: input shape with reduce axis = 1 (keepdim).
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = self.input_shape;
        out[self.reduce_axis as usize] = 1;
        out
    }
}

/// Args bundle. Output is always `i64` regardless of input dtype.
pub struct CountReduceArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — always i64.
    pub y: TensorMut<'a, i64, N>,
}

/// Plan for a count-nonzero axis reduction.
pub struct CountReducePlan<T: Element, const N: usize> {
    desc: CountReduceDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> CountReducePlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &CountReduceDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CountReducePlan: descriptor element != type parameter T",
            ));
        }
        if (desc.reduce_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CountReducePlan: reduce_axis must be < rank",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::CountReducePlan: input_shape dims must be non-negative",
                ));
            }
        }
        if !matches!(desc.kind, ReduceKind::CountNonzero) {
            return Err(Error::Unsupported(
                "baracuda-kernels::CountReducePlan: kind must be CountNonzero",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32
                | ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::F64
                | ElementKind::I32
                | ElementKind::I64
                | ElementKind::Bool
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::CountReducePlan: supported input dtypes are \
                 {f32, f16, bf16, f64, i32, i64, Bool}",
            ));
        }
        // Pure integer accumulation in i64 — bit-stable on the same
        // hardware (no FP math).
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::I64,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Reduction,
            op: desc.kind as u16,
            element: T::KIND,
            // Output dtype is i64; ElementKind has an I64 variant.
            aux_element: Some(ElementKind::I64),
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
    pub fn can_implement(&self, args: &CountReduceArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CountReducePlan: X shape mismatch with descriptor",
            ));
        }
        let expected_out = self.desc.output_shape();
        if args.y.shape != expected_out {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CountReducePlan: Y shape mismatch with derived output \
                 shape (input shape with reduce_axis collapsed to 1)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::CountReducePlan: tensor rank > 8 not supported",
            ));
        }
        let y_numel = args.y.numel();
        let x_numel = args.x.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
            });
        }
        if x_len < x_numel {
            return Err(Error::BufferTooSmall {
                needed: x_numel as usize,
                got: x_len as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always 0 for the naive trailblazer.
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
        args: CountReduceArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let output_numel = args.y.numel();
        if output_numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let output_shape = self.desc.output_shape();
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;
        let reduce_axis = self.desc.reduce_axis as i32;
        let reduce_extent = self.desc.input_shape[self.desc.reduce_axis as usize];
        let reduce_stride_x = args.x.stride[self.desc.reduce_axis as usize];

        macro_rules! dispatch {
            ($sym:ident) => {{
                unsafe {
                    baracuda_kernels_sys::$sym(
                        output_numel,
                        rank,
                        output_shape.as_ptr(),
                        stride_x.as_ptr(),
                        stride_y.as_ptr(),
                        reduce_axis,
                        reduce_extent,
                        reduce_stride_x,
                        x_ptr,
                        y_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            }};
        }

        let status = match (self.desc.kind, T::KIND) {
            (ReduceKind::CountNonzero, ElementKind::F32) => {
                dispatch!(baracuda_kernels_reduce_count_nonzero_f32_run)
            }
            (ReduceKind::CountNonzero, ElementKind::F16) => {
                dispatch!(baracuda_kernels_reduce_count_nonzero_f16_run)
            }
            (ReduceKind::CountNonzero, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_count_nonzero_bf16_run)
            }
            (ReduceKind::CountNonzero, ElementKind::F64) => {
                dispatch!(baracuda_kernels_reduce_count_nonzero_f64_run)
            }
            (ReduceKind::CountNonzero, ElementKind::I32) => {
                dispatch!(baracuda_kernels_reduce_count_nonzero_i32_run)
            }
            (ReduceKind::CountNonzero, ElementKind::I64) => {
                dispatch!(baracuda_kernels_reduce_count_nonzero_i64_run)
            }
            (ReduceKind::CountNonzero, ElementKind::Bool) => {
                dispatch!(baracuda_kernels_reduce_count_nonzero_bool_run)
            }
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::CountReducePlan::run: only `CountNonzero × \
                     {f32, f16, bf16, f64, i32, i64, Bool}` wired",
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
