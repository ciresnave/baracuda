//! Runtime-API graph user objects (CUDA 12.0+).
//!
//! Refcounted RAII slot you can attach to a graph via
//! [`Graph::retain_user_object`]; when the graph releases the last
//! reference, the destructor runs. Mirrors the Driver-side wrapper.

use core::ffi::c_void;

use baracuda_cuda_sys::runtime::{cudaUserObject_t, runtime};

use crate::error::{check, Result};

/// A refcounted user object.
pub struct UserObject {
    handle: cudaUserObject_t,
}

unsafe impl Send for UserObject {}
unsafe impl Sync for UserObject {}

impl core::fmt::Debug for UserObject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UserObject")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

unsafe extern "C" fn destroy_trampoline(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let f: Box<Box<dyn FnOnce() + Send>> =
        unsafe { Box::from_raw(user_data as *mut Box<dyn FnOnce() + Send>) };
    (*f)();
}

impl UserObject {
    /// Create a user object whose destructor is `destroy`.
    /// `initial_refcount` must be >= 1.
    pub fn new<F>(destroy: F, initial_refcount: u32) -> Result<Self>
    where
        F: FnOnce() + Send + 'static,
    {
        let boxed: Box<Box<dyn FnOnce() + Send>> = Box::new(Box::new(destroy));
        let raw = Box::into_raw(boxed) as *mut c_void;
        let r = runtime()?;
        let cu = r.cuda_user_object_create()?;
        let mut object: cudaUserObject_t = core::ptr::null_mut();
        // CUDA requires flags == cudaGraphUserObjectMove (1) currently.
        const CUDA_USER_OBJECT_NO_DESTRUCTOR_SYNC: core::ffi::c_uint = 1;
        let rc = unsafe {
            cu(
                &mut object,
                raw,
                Some(destroy_trampoline),
                initial_refcount,
                CUDA_USER_OBJECT_NO_DESTRUCTOR_SYNC,
            )
        };
        if rc != baracuda_cuda_sys::runtime::cudaError_t::Success {
            drop(unsafe { Box::from_raw(raw as *mut Box<dyn FnOnce() + Send>) });
            return Err(crate::error::Error::Status { status: rc });
        }
        Ok(Self { handle: object })
    }

    pub fn retain(&self, count: u32) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_user_object_retain()?;
        check(unsafe { cu(self.handle, count) })
    }

    pub fn release(&self, count: u32) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_user_object_release()?;
        check(unsafe { cu(self.handle, count) })
    }

    #[inline]
    pub fn as_raw(&self) -> cudaUserObject_t {
        self.handle
    }
}

impl Drop for UserObject {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_user_object_release() {
                let _ = unsafe { cu(self.handle, 1) };
            }
        }
    }
}

impl crate::Graph {
    /// Have this graph retain `count` references to `object`.
    pub fn retain_user_object(&self, object: &UserObject, count: u32, flags: u32) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_graph_retain_user_object()?;
        check(unsafe { cu(self.as_raw(), object.as_raw(), count, flags) })
    }

    pub fn release_user_object(&self, object: &UserObject, count: u32) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_graph_release_user_object()?;
        check(unsafe { cu(self.as_raw(), object.as_raw(), count) })
    }
}
