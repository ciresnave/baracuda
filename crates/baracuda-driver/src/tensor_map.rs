//! Hopper Tensor Memory Accelerator (TMA) descriptors.
//!
//! CUDA 12.0+ introduced `cuTensorMapEncodeTiled` / `cuTensorMapEncodeIm2col`
//! to produce `CUtensorMap` descriptors that TMA instructions in kernels
//! consume to asynchronously move multi-dimensional tiles between global
//! and shared memory. This is a Hopper-only hardware feature (SM 9.0+),
//! but the descriptor *encoding* itself is pure host code and works on
//! any device.
//!
//! See the [`TensorMap`] builder for a typed wrapper around
//! `cuTensorMapEncodeTiled`.

use baracuda_cuda_sys::types::CUtensorMap;
use baracuda_cuda_sys::{driver, CUdeviceptr};

use crate::error::{check, Result};

pub use baracuda_cuda_sys::types::{
    CUtensorMapDataType as DataType, CUtensorMapFloatOOBfill as OOBFill,
    CUtensorMapInterleave as Interleave, CUtensorMapL2promotion as L2Promotion,
    CUtensorMapSwizzle as Swizzle,
};

/// A 128-byte Hopper TMA descriptor. Pass to a kernel as a `__grid_constant__`
/// parameter of type `CUtensorMap` for use with TMA instructions.
pub struct TensorMap {
    inner: CUtensorMap,
}

impl core::fmt::Debug for TensorMap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorMap")
            .field(
                "non_zero_words",
                &self.inner.opaque.iter().filter(|w| **w != 0).count(),
            )
            .finish_non_exhaustive()
    }
}

impl TensorMap {
    /// Build a tiled TMA descriptor.
    ///
    /// - `data_type`: element type (one of the `DataType::*` constants).
    /// - `global_base`: pointer to the first element of the tensor.
    /// - `global_dim`: per-axis size of the global tensor (innermost-to-outermost).
    /// - `global_strides`: per-axis byte strides between successive elements.
    /// - `box_dim`: per-axis shape of the tile copied at a time.
    /// - `element_strides`: per-axis element-strides (typically all 1).
    ///
    /// All arrays must have length `rank = global_dim.len()`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_tiled(
        data_type: i32,
        global_base: CUdeviceptr,
        global_dim: &[u64],
        global_strides: &[u64],
        box_dim: &[u32],
        element_strides: &[u32],
        interleave: i32,
        swizzle: i32,
        l2_promotion: i32,
        oob_fill: i32,
    ) -> Result<Self> {
        let rank = global_dim.len();
        assert_eq!(global_strides.len(), rank);
        assert_eq!(box_dim.len(), rank);
        assert_eq!(element_strides.len(), rank);
        let d = driver()?;
        let cu = d.cu_tensor_map_encode_tiled()?;
        let mut map = CUtensorMap::default();
        check(unsafe {
            cu(
                &mut map,
                data_type,
                rank as core::ffi::c_uint,
                global_base.0 as *mut core::ffi::c_void,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )
        })?;
        Ok(Self { inner: map })
    }

    /// Swap the global base address of an existing descriptor in place.
    /// Lets you reuse one `TensorMap` across multiple buffers of the same
    /// shape/stride.
    pub fn replace_address(&mut self, new_base: CUdeviceptr) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_tensor_map_replace_address()?;
        check(unsafe { cu(&mut self.inner, new_base.0 as *mut core::ffi::c_void) })
    }

    /// Raw pointer to the 128-byte descriptor — pass this to kernels that
    /// take a `CUtensorMap` parameter.
    #[inline]
    pub fn as_raw(&self) -> &CUtensorMap {
        &self.inner
    }

    /// Mutable raw access (for FFI calls that want `*mut CUtensorMap`).
    #[inline]
    pub fn as_raw_mut(&mut self) -> &mut CUtensorMap {
        &mut self.inner
    }
}
