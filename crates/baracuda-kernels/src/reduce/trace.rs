//! Trace plan — `y = sum(diag(M))` for a 2-D square matrix.
//!
//! Doesn't fit `ReducePlan<T, N>` cleanly because trace reduces *both*
//! axes via the `i == i` constraint rather than a single reduce_axis,
//! and the output is rank-0 (scalar) rather than keepdim. Trace gets
//! its own plan shape — `TracePlan<T>` — but reuses the
//! [`OpCategory::Reduction`] category and the
//! [`ReduceKind::Trace`] discriminant for telemetry / SKU-tagging
//! consistency with the rest of the reduction family.
//!
//! Wired for `{f32, f16, bf16, f64}` — 4 dtype cells. f16 / bf16
//! accumulate in f32 internally (f32-detour).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ReduceKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a matrix-trace reduction.
///
/// `n` is the matrix dimension (rows == cols). The kernel walks the
/// diagonal of an `n × n` matrix, accumulating in f32 for half-precision
/// inputs (native dtype for f32 / f64).
#[derive(Copy, Clone, Debug)]
pub struct TraceDescriptor {
    /// Matrix dimension — rows == cols == `n`. Must be non-negative.
    pub n: i32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a trace launch.
///
/// `x` is the 2-D input matrix of shape `[n, n]`. `y` is a rank-0
/// scalar output buffer (empty `shape == []`).
pub struct TraceArgs<'a, T: Element> {
    /// Input matrix, shape `[n, n]`.
    pub x: TensorRef<'a, T, 2>,
    /// Scalar output — rank-0 tensor (empty shape).
    pub y: TensorMut<'a, T, 0>,
}

/// Matrix-trace plan — `y = sum(diag(M))`.
///
/// `T: Element` is the element type. Supported: `f32`, `f16`, `bf16`,
/// `f64`.
pub struct TracePlan<T: Element> {
    desc: TraceDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> TracePlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &TraceDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TracePlan: descriptor element != type parameter T",
            ));
        }
        if desc.n < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TracePlan: n must be non-negative",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::TracePlan: supported dtypes are \
                 {f32, f16, bf16, f64}; other dtypes land in later fanout",
            ));
        }
        // Naive trailblazer: single thread walks the diagonal. f32
        // accumulator for f16/bf16; native for f32/f64. Deterministic
        // and bit-stable.
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Reduction,
            op: ReduceKind::Trace as u16,
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
    pub fn can_implement(&self, args: &TraceArgs<'_, T>) -> Result<()> {
        if args.x.shape != [self.desc.n, self.desc.n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TracePlan: X shape must be [n, n] (square)",
            ));
        }
        // y is rank-0 — `shape == []`. The const-generic forces it at
        // compile time, but assert the runtime shape is the empty array
        // anyway (mirrors the other plans' explicit shape checks).
        let y_shape: [i32; 0] = args.y.shape;
        let _expected: [i32; 0] = [];
        if y_shape != _expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TracePlan: Y must be a rank-0 scalar (empty shape)",
            ));
        }
        let n = self.desc.n as i64;
        let x_needed = n.saturating_mul(n);
        let x_len = args.x.data.len() as i64;
        if x_len < x_needed {
            return Err(Error::BufferTooSmall {
                needed: x_needed as usize,
                got: x_len as usize,
            });
        }
        if (args.y.data.len() as i64) < 1 {
            return Err(Error::BufferTooSmall {
                needed: 1,
                got: args.y.data.len(),
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0` for the naive trailblazer.
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

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: TraceArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.n == 0 {
            // Trace of an empty (0×0) matrix is 0. The kernel still
            // launches and stores 0 into y[0] via the empty-loop path.
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let n = self.desc.n;
        let stride_row = args.x.stride[0];
        let stride_col = args.x.stride[1];

        macro_rules! dispatch {
            ($sym:ident) => {{
                unsafe {
                    baracuda_kernels_sys::$sym(
                        n,
                        stride_row,
                        stride_col,
                        x_ptr,
                        y_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            }};
        }

        let status = match T::KIND {
            ElementKind::F32 => dispatch!(baracuda_kernels_trace_f32_run),
            ElementKind::F16 => dispatch!(baracuda_kernels_trace_f16_run),
            ElementKind::Bf16 => dispatch!(baracuda_kernels_trace_bf16_run),
            ElementKind::F64 => dispatch!(baracuda_kernels_trace_f64_run),
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TracePlan::run: dtype not wired",
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
