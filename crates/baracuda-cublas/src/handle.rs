//! `cublas::Handle` — the owning wrapper around a `cublasHandle_t`.

use std::sync::Arc;

use baracuda_cublas_sys::{cublas, cublasHandle_t, cublasMath_t, cublasPointerMode_t};
use baracuda_driver::Stream;

use crate::error::{check, Result};

/// Owned cuBLAS context.
///
/// Clones share ownership via `Arc`. Handles are `Send` but **not** `Sync`:
/// NVIDIA documents that a cuBLAS handle should only be used from one host
/// thread at a time. Wrap in `Arc<Mutex<Handle>>` if you need to share
/// across threads.
#[derive(Clone)]
pub struct Handle {
    inner: Arc<HandleInner>,
}

struct HandleInner {
    handle: cublasHandle_t,
}

// SAFETY: NVIDIA explicitly permits cuBLAS handles to be moved between
// threads; concurrent use from multiple threads is UB.
unsafe impl Send for HandleInner {}

impl core::fmt::Debug for HandleInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("cublas::Handle")
            .field("handle", &self.handle)
            .finish()
    }
}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Handle {
    /// Create a new cuBLAS handle on the current device.
    pub fn new() -> Result<Self> {
        let c = cublas()?;
        let cu = c.cublas_create()?;
        let mut h: cublasHandle_t = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        // `HandleInner` is `!Sync` on purpose — cuBLAS handles are not
        // safe for concurrent use from multiple threads. Arc is still
        // correct: we want cheap clones for sharing within a thread.
        #[allow(clippy::arc_with_non_send_sync)]
        let inner = Arc::new(HandleInner { handle: h });
        Ok(Self { inner })
    }

    /// Bind this handle to `stream`. All subsequent cuBLAS operations will
    /// issue on `stream`. Returns the previous stream binding is discarded;
    /// keep a reference to `stream` alive until the operations complete.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = cublas()?;
        let cu = c.cublas_set_stream()?;
        // SAFETY: both handles are alive for the call.
        check(unsafe { cu(self.inner.handle, stream.as_raw() as _) })
    }

    /// Alpha/beta scalars can live in host or device memory. Default is host.
    pub fn set_pointer_mode(&self, device_pointers: bool) -> Result<()> {
        let c = cublas()?;
        let cu = c.cublas_set_pointer_mode()?;
        let mode = if device_pointers {
            cublasPointerMode_t::Device
        } else {
            cublasPointerMode_t::Host
        };
        check(unsafe { cu(self.inner.handle, mode) })
    }

    /// Configure the math mode (tensor-core preferences, TF32, etc.).
    pub fn set_math_mode(&self, mode: cublasMath_t) -> Result<()> {
        let c = cublas()?;
        let cu = c.cublas_set_math_mode()?;
        check(unsafe { cu(self.inner.handle, mode) })
    }

    /// cuBLAS library version, e.g. `120604` for cuBLAS 12.6.4.
    pub fn version(&self) -> Result<i32> {
        let c = cublas()?;
        let cu = c.cublas_get_version()?;
        let mut v: core::ffi::c_int = 0;
        check(unsafe { cu(self.inner.handle, &mut v) })?;
        Ok(v)
    }

    /// Raw `cublasHandle_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cublasHandle_t {
        self.inner.handle
    }
}

impl Drop for HandleInner {
    fn drop(&mut self) {
        if let Ok(c) = cublas() {
            if let Ok(cu) = c.cublas_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
