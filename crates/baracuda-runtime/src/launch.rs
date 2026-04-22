//! Kernel launch builder for the Runtime API.

use core::ffi::c_void;

use baracuda_cuda_sys::runtime::{cudaStream_t, runtime, types::dim3};
use baracuda_types::KernelArg;

use crate::error::{check, Result};
use crate::module::Kernel;
use crate::stream::Stream;

/// Grid / block size triple, matching [`baracuda_driver::Dim3`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    #[inline]
    fn to_sys(self) -> dim3 {
        dim3::new(self.x, self.y, self.z)
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Self { x, y, z: 1 }
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self { x, y, z }
    }
}

impl Kernel {
    /// Start a kernel-launch builder for this kernel.
    #[inline]
    pub fn launch(&self) -> LaunchBuilder<'_> {
        LaunchBuilder {
            kernel: self,
            grid: Dim3 { x: 1, y: 1, z: 1 },
            block: Dim3 { x: 1, y: 1, z: 1 },
            shared_mem_bytes: 0,
            stream: None,
            args: Vec::new(),
        }
    }
}

/// Builder produced by [`Kernel::launch`].
#[must_use = "the launch builder does nothing until `.launch()` is called"]
pub struct LaunchBuilder<'k> {
    kernel: &'k Kernel,
    grid: Dim3,
    block: Dim3,
    shared_mem_bytes: usize,
    stream: Option<&'k Stream>,
    args: Vec<*mut c_void>,
}

impl core::fmt::Debug for LaunchBuilder<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LaunchBuilder")
            .field("grid", &self.grid)
            .field("block", &self.block)
            .field("shared_mem_bytes", &self.shared_mem_bytes)
            .field("arg_count", &self.args.len())
            .finish_non_exhaustive()
    }
}

impl<'k> LaunchBuilder<'k> {
    #[inline]
    pub fn grid(mut self, grid: impl Into<Dim3>) -> Self {
        self.grid = grid.into();
        self
    }

    #[inline]
    pub fn block(mut self, block: impl Into<Dim3>) -> Self {
        self.block = block.into();
        self
    }

    #[inline]
    pub fn shared_mem_bytes(mut self, bytes: usize) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    #[inline]
    pub fn stream(mut self, stream: &'k Stream) -> Self {
        self.stream = Some(stream);
        self
    }

    #[inline]
    pub fn arg<K: KernelArg>(mut self, arg: K) -> Self {
        self.args.push(arg.as_kernel_arg_ptr());
        self
    }

    /// Enqueue the kernel.
    ///
    /// # Safety
    ///
    /// Same rules as [`baracuda_driver::LaunchBuilder::launch`]: argument
    /// types and order must match the kernel's C signature, referenced
    /// device memory must stay valid for the duration of device execution,
    /// and grid/block dims must be within device limits.
    pub unsafe fn launch(mut self) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_launch_kernel()?;
        let stream_handle: cudaStream_t = self.stream.map_or(core::ptr::null_mut(), |s| s.as_raw());
        let args_ptr = if self.args.is_empty() {
            core::ptr::null_mut()
        } else {
            self.args.as_mut_ptr()
        };
        check(cu(
            self.kernel.as_launch_ptr(),
            self.grid.to_sys(),
            self.block.to_sys(),
            args_ptr,
            self.shared_mem_bytes,
            stream_handle,
        ))
    }

    /// Launch as a cooperative kernel — grid-wide sync via
    /// `cooperative_groups::this_grid()`. All blocks must fit resident
    /// on the device simultaneously; use
    /// [`crate::Kernel::max_active_blocks_per_multiprocessor`] to size
    /// the grid.
    ///
    /// # Safety
    ///
    /// Same as [`launch`](Self::launch) plus the kernel must be
    /// compiled with cooperative-groups support.
    pub unsafe fn launch_cooperative(mut self) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_launch_cooperative_kernel()?;
        let stream_handle: cudaStream_t = self.stream.map_or(core::ptr::null_mut(), |s| s.as_raw());
        let args_ptr = if self.args.is_empty() {
            core::ptr::null_mut()
        } else {
            self.args.as_mut_ptr()
        };
        check(cu(
            self.kernel.as_launch_ptr(),
            self.grid.to_sys(),
            self.block.to_sys(),
            args_ptr,
            self.shared_mem_bytes,
            stream_handle,
        ))
    }
}
