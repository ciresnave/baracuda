//! Safe Rust API for the NVIDIA TensorRT **builder**: define a network layer by
//! layer, configure the build, and get a serialized engine that
//! [`baracuda_tensorrt::Runtime::deserialize_engine`] loads.
//!
//! Like the runtime crate, this reaches TensorRT through the C++ shim in
//! `baracuda-tensorrt-sys`, so the whole API needs the `shim` feature (and the
//! TensorRT SDK headers at build time; see `crates/baracuda-tensorrt/NOTICE.md`).
//! Without it the crate only offers [`shim_built`].
//!
//! ```text
//! let logger = Arc::new(Logger::stderr(Severity::Warning)?);
//! let builder = Builder::new(logger)?;
//! let network = builder.create_network(NetworkFlags::STRONGLY_TYPED)?;
//! let a = network.add_input("a", DataType::Float, &Dims::new(&[4]))?;
//! let b = network.add_input("b", DataType::Float, &Dims::new(&[4]))?;
//! let sum = network.add_element_wise(&a, &b, ElementWiseOp::Sum)?.output(0)?;
//! sum.set_name("sum")?;
//! network.mark_output(&sum)?;
//! let blob = builder.build_serialized(&network, &builder.create_config()?)?;
//! ```

#![warn(missing_debug_implementations, rust_2018_idioms)]

/// Whether the C++ shim was compiled in. When `false` this crate has no API.
pub fn shim_built() -> bool {
    baracuda_tensorrt_sys::SHIM_BUILT
}

#[cfg(feature = "shim")]
mod api;
#[cfg(feature = "shim")]
pub use api::*;
