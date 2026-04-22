//! Extended launch attributes + `cudaLaunchKernelEx` (cluster launches,
//! programmatic stream serialization, preferred shmem carveout).
//!
//! Modern launches go through [`LaunchExBuilder`]:
//!
//! ```no_run
//! # use baracuda_runtime::launch_attr::LaunchExBuilder;
//! # use baracuda_runtime::{Stream, Library};
//! # let stream: Stream = todo!();
//! # let kernel = todo!();
//! # let mut args: [*mut core::ffi::c_void; 0] = [];
//! unsafe {
//!     LaunchExBuilder::new(&stream, (32, 1, 1), (256, 1, 1))
//!         .cluster_dim((2, 1, 1))
//!         .cooperative(true)
//!         .launch(kernel, &mut args)
//!         .unwrap();
//! }
//! ```

use core::ffi::c_void;

use baracuda_cuda_sys::runtime::runtime;
use baracuda_cuda_sys::runtime::types::{
    cudaLaunchAttribute, cudaLaunchAttributeID, cudaLaunchAttributeValue, cudaLaunchConfig_t, dim3,
};

use crate::error::{check, Result};
use crate::launch::Dim3;
use crate::module::Kernel;
use crate::stream::Stream;

/// Builder for `cudaLaunchKernelEx` — accepts up to ~14 attribute kinds.
#[derive(Debug)]
pub struct LaunchExBuilder<'s> {
    config: cudaLaunchConfig_t,
    attrs: Vec<cudaLaunchAttribute>,
    _stream: &'s Stream,
}

impl<'s> LaunchExBuilder<'s> {
    pub fn new(stream: &'s Stream, grid: impl Into<Dim3>, block: impl Into<Dim3>) -> Self {
        let g: Dim3 = grid.into();
        let b: Dim3 = block.into();
        Self {
            config: cudaLaunchConfig_t {
                grid_dim: dim3::new(g.x, g.y, g.z),
                block_dim: dim3::new(b.x, b.y, b.z),
                dynamic_smem_bytes: 0,
                stream: stream.as_raw(),
                attrs: core::ptr::null_mut(),
                num_attrs: 0,
            },
            attrs: Vec::new(),
            _stream: stream,
        }
    }

    pub fn dynamic_shared_memory(mut self, bytes: usize) -> Self {
        self.config.dynamic_smem_bytes = bytes;
        self
    }

    fn push(mut self, id: i32, val: cudaLaunchAttributeValue) -> Self {
        self.attrs.push(cudaLaunchAttribute { id, _pad: 0, val });
        self
    }

    /// Hopper cluster dimension (x, y, z) in blocks.
    pub fn cluster_dim(self, dims: impl Into<Dim3>) -> Self {
        let d: Dim3 = dims.into();
        self.push(
            cudaLaunchAttributeID::CLUSTER_DIMENSION,
            cudaLaunchAttributeValue::cluster_dimension(d.x, d.y, d.z),
        )
    }

    /// Enable a cooperative launch.
    pub fn cooperative(self, enable: bool) -> Self {
        self.push(
            cudaLaunchAttributeID::COOPERATIVE,
            cudaLaunchAttributeValue::cooperative(enable),
        )
    }

    /// Assign a priority to this launch (overrides stream priority for
    /// this kernel).
    pub fn priority(self, prio: i32) -> Self {
        self.push(
            cudaLaunchAttributeID::PRIORITY,
            cudaLaunchAttributeValue::priority(prio),
        )
    }

    /// Push a raw attribute slot — escape hatch for IDs this builder
    /// doesn't expose typed.
    pub fn raw_attr(self, id: i32, val: cudaLaunchAttributeValue) -> Self {
        self.push(id, val)
    }

    /// Execute the launch.
    ///
    /// # Safety
    ///
    /// `args` must match `kernel`'s C signature in count / order / types
    /// exactly (the marshaling is bytewise).
    pub unsafe fn launch(mut self, kernel: &Kernel, args: &mut [*mut c_void]) -> Result<()> {
        if !self.attrs.is_empty() {
            self.config.attrs = self.attrs.as_mut_ptr();
            self.config.num_attrs = self.attrs.len() as core::ffi::c_uint;
        }
        let r = runtime()?;
        let cu = r.cuda_launch_kernel_ex()?;
        check(cu(
            &self.config,
            kernel.as_launch_ptr(),
            if args.is_empty() {
                core::ptr::null_mut()
            } else {
                args.as_mut_ptr()
            },
        ))
    }
}
