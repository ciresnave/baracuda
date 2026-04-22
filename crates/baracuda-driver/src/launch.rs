//! Kernel launch builder — the Rust equivalent of CUDA C's triple-chevron syntax.
//!
//! ```no_run
//! # use baracuda_driver::{Context, Device, Stream, Module, DeviceBuffer};
//! # fn example() -> baracuda_driver::Result<()> {
//! # let device = Device::get(0)?;
//! # let ctx = Context::new(&device)?;
//! # let stream = Stream::new(&ctx)?;
//! # let module = Module::load_ptx(&ctx, "")?;
//! # let kernel = module.get_function("vector_add")?;
//! # let mut d_c: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 1024)?;
//! # let d_a: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 1024)?;
//! # let d_b: DeviceBuffer<f32> = DeviceBuffer::new(&ctx, 1024)?;
//! # let n = 1024u32;
//! unsafe {
//!     kernel.launch()
//!         .grid((n.div_ceil(256), 1, 1))
//!         .block((256, 1, 1))
//!         .stream(&stream)
//!         .arg(&d_a.as_raw())
//!         .arg(&d_b.as_raw())
//!         .arg(&d_c.as_raw())
//!         .arg(&n)
//!         .launch()?;
//! }
//! # Ok(())
//! # }
//! ```

use core::ffi::c_void;

use baracuda_cuda_sys::{driver, CUstream};
use baracuda_types::KernelArg;

use crate::error::{check, Result};
use crate::module::Function;
use crate::stream::Stream;

/// Three-dimensional grid/block size.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
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

impl Function {
    /// Start a kernel-launch builder for this function.
    #[inline]
    pub fn launch(&self) -> LaunchBuilder<'_> {
        LaunchBuilder {
            function: self,
            grid: Dim3 { x: 1, y: 1, z: 1 },
            block: Dim3 { x: 1, y: 1, z: 1 },
            shared_mem_bytes: 0,
            stream: None,
            args: Vec::new(),
        }
    }
}

/// Builder produced by [`Function::launch`]. Call [`LaunchBuilder::launch`]
/// to actually enqueue the kernel.
#[must_use = "the launch builder does nothing until `launch()` is called"]
pub struct LaunchBuilder<'f> {
    function: &'f Function,
    grid: Dim3,
    block: Dim3,
    shared_mem_bytes: u32,
    stream: Option<&'f Stream>,
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

impl<'f> LaunchBuilder<'f> {
    /// Set the grid (number of blocks per axis).
    #[inline]
    pub fn grid(mut self, grid: impl Into<Dim3>) -> Self {
        self.grid = grid.into();
        self
    }

    /// Set the block (number of threads per block per axis).
    #[inline]
    pub fn block(mut self, block: impl Into<Dim3>) -> Self {
        self.block = block.into();
        self
    }

    /// Set the amount of dynamic shared memory (bytes). Defaults to 0.
    #[inline]
    pub fn shared_mem_bytes(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Launch on the specified [`Stream`]. Defaults to the legacy null stream.
    #[inline]
    pub fn stream(mut self, stream: &'f Stream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Append one kernel argument. Pass `&value` for each kernel parameter:
    /// the referent must remain alive until [`launch`](Self::launch) is
    /// called. CUDA copies argument bytes at submission time, so the
    /// values don't need to outlive the *device* execution — only the
    /// submission.
    #[inline]
    pub fn arg<K: KernelArg>(mut self, arg: K) -> Self {
        self.args.push(arg.as_kernel_arg_ptr());
        self
    }

    /// Actually enqueue the kernel.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// 1. The number, order, and types of arguments match what the kernel
    ///    expects. Baracuda cannot see the kernel's signature, so a
    ///    mismatch here causes undefined behavior (typically corrupted
    ///    output, a device fault, or silent memory corruption).
    /// 2. Any pointer-typed argument (e.g. `DeviceBuffer::as_raw()`) is
    ///    live for the duration of kernel execution — use streams + events
    ///    to manage this.
    /// 3. Grid and block dimensions are within the device's supported
    ///    limits (see [`crate::Device::attribute`]).
    pub unsafe fn launch(mut self) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_launch_kernel()?;
        let stream_handle: CUstream = self.stream.map_or(core::ptr::null_mut(), |s| s.as_raw());
        let args_ptr = if self.args.is_empty() {
            core::ptr::null_mut()
        } else {
            self.args.as_mut_ptr()
        };
        check(cu(
            self.function.as_raw(),
            self.grid.x,
            self.grid.y,
            self.grid.z,
            self.block.x,
            self.block.y,
            self.block.z,
            self.shared_mem_bytes,
            stream_handle,
            args_ptr,
            core::ptr::null_mut(), // extras — unused; we always pass args via the kernel_params slot
        ))
    }

    /// Enqueue the kernel via `cuLaunchKernelEx` (CUDA 12.0+), letting the
    /// caller attach launch attributes (cluster dims, programmatic stream
    /// serialization, priority, …). Pass an empty slice when you just want
    /// the modern launch entry point with no attributes.
    ///
    /// # Safety
    ///
    /// Same responsibilities as [`launch`](Self::launch). Attribute payloads
    /// in `attributes` must be populated correctly per
    /// [`baracuda_cuda_sys::types::CUlaunchAttributeID`] — invalid
    /// attribute payloads cause undefined behavior on the device.
    pub unsafe fn launch_ex(
        mut self,
        attributes: &mut [baracuda_cuda_sys::types::CUlaunchAttribute],
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_launch_kernel_ex()?;
        let stream_handle: CUstream = self.stream.map_or(core::ptr::null_mut(), |s| s.as_raw());
        let args_ptr = if self.args.is_empty() {
            core::ptr::null_mut()
        } else {
            self.args.as_mut_ptr()
        };
        let config = baracuda_cuda_sys::types::CUlaunchConfig {
            grid_dim_x: self.grid.x,
            grid_dim_y: self.grid.y,
            grid_dim_z: self.grid.z,
            block_dim_x: self.block.x,
            block_dim_y: self.block.y,
            block_dim_z: self.block.z,
            shared_mem_bytes: self.shared_mem_bytes,
            stream: stream_handle,
            attrs: if attributes.is_empty() {
                core::ptr::null_mut()
            } else {
                attributes.as_mut_ptr()
            },
            num_attrs: attributes.len() as core::ffi::c_uint,
        };
        check(cu(
            &config,
            self.function.as_raw(),
            args_ptr,
            core::ptr::null_mut(),
        ))
    }
}
