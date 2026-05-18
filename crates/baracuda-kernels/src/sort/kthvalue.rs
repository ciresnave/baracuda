//! `kthvalue` plan — returns the k-th smallest value + its index along
//! the last dimension.
//!
//! PyTorch `torch.kthvalue(x, k, dim)` returns `(value, index)` for
//! the **k-th smallest** value (1-indexed in PyTorch). Our descriptor
//! uses 0-indexed `k` (the user passes `k = 0` for the smallest).
//!
//! Composition: invokes [`crate::sort::TopkPlan`] with `largest=false`
//! and `k = desc.k + 1`, then reads cell `(k)` of the bottom-(k+1)
//! result. The composition lives at the Rust plan layer — no separate
//! kthvalue kernel is shipped.
//!
//! Trailblazer dtype coverage: `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_kernels_types::{
    contiguous_stride, Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee,
    SortKind, TensorMut, TensorRef, Workspace,
};

use super::sort::build_sku;
use super::topk::{TopkArgs, TopkDescriptor, TopkPlan};
use super::TOPK_MAX_K;

/// Descriptor for a `kthvalue` op.
///
/// `k` is 0-indexed (k = 0 → smallest value, k = row_len - 1 →
/// largest).
#[derive(Copy, Clone, Debug)]
pub struct KthvalueDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each row.
    pub row_len: i32,
    /// Which order statistic to return (0-indexed). Trailblazer cap:
    /// `k < 64` (composes a bottom-(k+1) topk).
    pub k: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `kthvalue` launch.
pub struct KthvalueArgs<'a, T: Element> {
    /// Input `[batch, row_len]`.
    pub input: TensorRef<'a, T, 2>,
    /// Output values `[batch]` (one cell per row).
    pub values: TensorMut<'a, T, 1>,
    /// Output indices `[batch]` (one i32 per row).
    pub indices: TensorMut<'a, i32, 1>,
}

/// `kthvalue` plan.
///
/// Returns the k-th smallest value and its index along the last axis
/// (PyTorch `torch.kthvalue`; 0-indexed `k` here, vs PyTorch's
/// 1-indexed). Composed at the plan layer as a bottom-(k+1)
/// [`TopkPlan`](crate::TopkPlan), reading cell `(k)` of the result.
///
/// **When to use**: order-statistic queries (median, quantile pickup
/// in fixed K range). Pair with
/// [`KthvalueBackwardPlan`](crate::KthvalueBackwardPlan).
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: input `[batch, row_len]`; outputs `[batch]`;
/// `row_len ≤ 1024`; `k < 64` (composes a bottom-(k+1) topk).
///
/// **Workspace**: zero in [`Workspace`]; plan internally allocates a
/// scratch `[batch, k+1]` topk-result buffer per launch.
///
/// **Precision guarantee**: deterministic, bit-stable (inherits topk's
/// fixed-network guarantee).
pub struct KthvaluePlan<T: Element> {
    desc: KthvalueDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> KthvaluePlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &KthvalueDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::KthvaluePlan: descriptor element != type parameter T",
            ));
        }
        if desc.batch < 0 || desc.row_len < 0 || desc.k < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvaluePlan: batch / row_len / k must be non-negative",
            ));
        }
        if desc.k >= desc.row_len {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvaluePlan: k must be < row_len (0-indexed)",
            ));
        }
        if desc.k + 1 > TOPK_MAX_K {
            return Err(Error::Unsupported(
                "baracuda-kernels::KthvaluePlan: k+1 > 64 not supported (composes topk)",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::KthvaluePlan: today only f32 / f64 wired (TopkPlan limit)",
            ));
        }
        let sku = build_sku::<T>(SortKind::Kthvalue);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &KthvalueArgs<'_, T>) -> Result<()> {
        if args.input.shape != [self.desc.batch, self.desc.row_len] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvaluePlan: input shape != [batch, row_len]",
            ));
        }
        if args.values.shape != [self.desc.batch] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvaluePlan: values shape != [batch]",
            ));
        }
        if args.indices.shape != [self.desc.batch] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KthvaluePlan: indices shape != [batch]",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes. Internal device buffers are allocated
    /// fresh at run() time.
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

    /// Launch. Composes a bottom-(k+1) topk; reads the last cell as the
    /// k-th smallest. Allocates two intermediate device buffers and
    /// round-trips the bottom-(k+1) cells through host memory to
    /// extract the (k)-th slot per row (the data is small — batch *
    /// (k+1) cells with `k+1 ≤ 64`).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: KthvalueArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 {
            return Ok(());
        }

        let kp1 = self.desc.k + 1;
        let topk_desc = TopkDescriptor {
            batch: self.desc.batch,
            row_len: self.desc.row_len,
            k: kp1,
            largest: false,
            element: T::KIND,
        };
        let topk_plan = TopkPlan::<T>::select(stream, &topk_desc, PlanPreference::default())?;

        let ctx = stream.context();
        let total = (self.desc.batch as usize) * (kp1 as usize);
        let mut topk_vals: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, total).map_err(|_| {
            Error::InvalidProblem(
                "baracuda-kernels::KthvaluePlan: failed to allocate topk values buffer",
            )
        })?;
        let mut topk_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, total).map_err(|_| {
            Error::InvalidProblem(
                "baracuda-kernels::KthvaluePlan: failed to allocate topk indices buffer",
            )
        })?;

        let topk_args = TopkArgs::<T> {
            input: args.input,
            values: TensorMut {
                data: topk_vals.as_slice_mut(),
                shape: [self.desc.batch, kp1],
                stride: contiguous_stride([self.desc.batch, kp1]),
            },
            indices: TensorMut {
                data: topk_idx.as_slice_mut(),
                shape: [self.desc.batch, kp1],
                stride: contiguous_stride([self.desc.batch, kp1]),
            },
        };
        topk_plan.run(stream, Workspace::None, topk_args)?;
        stream
            .synchronize()
            .map_err(|_| Error::CutlassInternal(-1))?;

        // Bring the bottom-(k+1) tiles host-side as raw bytes, pick
        // the k-th cell per row, and ship the two compacted [batch]
        // vectors back to the device. We use byte buffers to avoid a
        // `T: Default` bound on `Element` (which doesn't exist).
        let val_bytes = total * core::mem::size_of::<T>();
        let idx_bytes_total = total * core::mem::size_of::<i32>();
        let mut host_vals: Vec<u8> = vec![0u8; val_bytes];
        let mut host_idx_bytes: Vec<u8> = vec![0u8; idx_bytes_total];
        unsafe {
            copy_d2h_async(
                host_vals.as_mut_ptr() as *mut c_void,
                topk_vals.as_raw().0,
                val_bytes,
                stream,
            )?;
            copy_d2h_async(
                host_idx_bytes.as_mut_ptr() as *mut c_void,
                topk_idx.as_raw().0,
                idx_bytes_total,
                stream,
            )?;
        }
        stream
            .synchronize()
            .map_err(|_| Error::CutlassInternal(-1))?;

        let out_val_bytes = (self.desc.batch as usize) * core::mem::size_of::<T>();
        let out_idx_bytes = (self.desc.batch as usize) * core::mem::size_of::<i32>();
        let mut out_vals: Vec<u8> = vec![0u8; out_val_bytes];
        let mut out_idx: Vec<u8> = vec![0u8; out_idx_bytes];
        let stride_v = core::mem::size_of::<T>();
        let stride_i = core::mem::size_of::<i32>();
        for row in 0..self.desc.batch as usize {
            let src_v_off = (row * (kp1 as usize) + self.desc.k as usize) * stride_v;
            let src_i_off = (row * (kp1 as usize) + self.desc.k as usize) * stride_i;
            let dst_v_off = row * stride_v;
            let dst_i_off = row * stride_i;
            out_vals[dst_v_off..dst_v_off + stride_v]
                .copy_from_slice(&host_vals[src_v_off..src_v_off + stride_v]);
            out_idx[dst_i_off..dst_i_off + stride_i]
                .copy_from_slice(&host_idx_bytes[src_i_off..src_i_off + stride_i]);
        }

        unsafe {
            copy_h2d_async(
                args.values.data.as_raw().0 as *mut c_void,
                out_vals.as_ptr() as *const c_void,
                out_val_bytes,
                stream,
            )?;
            copy_h2d_async(
                args.indices.data.as_raw().0 as *mut c_void,
                out_idx.as_ptr() as *const c_void,
                out_idx_bytes,
                stream,
            )?;
        }
        // Keep the host buffers alive until the H2D completes.
        stream
            .synchronize()
            .map_err(|_| Error::CutlassInternal(-1))?;
        drop(out_vals);
        drop(out_idx);
        drop(host_vals);
        drop(host_idx_bytes);
        Ok(())
    }
}

/// H2D copy helper — same pattern as `linalg/qr.rs` (no `cu` dep in
/// this crate; we declare the symbol locally).
unsafe fn copy_h2d_async(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[allow(non_camel_case_types)]
    type CUresult = i32;
    unsafe extern "system" {
        fn cuMemcpyHtoDAsync_v2(
            dst_device: u64,
            src_host: *const c_void,
            byte_count: usize,
            h_stream: *mut c_void,
        ) -> CUresult;
    }
    let status =
        unsafe { cuMemcpyHtoDAsync_v2(dst as u64, src, bytes, stream.as_raw() as *mut c_void) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}

/// D2H copy helper.
unsafe fn copy_d2h_async(
    dst: *mut c_void,
    src: u64,
    bytes: usize,
    stream: &Stream,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[allow(non_camel_case_types)]
    type CUresult = i32;
    unsafe extern "system" {
        fn cuMemcpyDtoHAsync_v2(
            dst_host: *mut c_void,
            src_device: u64,
            byte_count: usize,
            h_stream: *mut c_void,
        ) -> CUresult;
    }
    let status =
        unsafe { cuMemcpyDtoHAsync_v2(dst, src, bytes, stream.as_raw() as *mut c_void) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}
