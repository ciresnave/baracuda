//! The ONNX parser: fill a [`Network`] from a serialized ONNX model.

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::sync::Arc;

use baracuda_tensorrt::{Error, Logger, Result};
use baracuda_tensorrt_sys::{self as sys, builder as ffi};

use crate::Network;

/// One problem the ONNX parser reported (`nvonnxparser::IParserError`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParserError {
    /// `nvonnxparser::ErrorCode` (0 is success; 3 is a model that failed to
    /// deserialize, 8 an unsupported node, ...).
    pub code: i32,
    /// Human-readable description.
    pub description: String,
    /// Index of the offending node in the graph, or `-1` when there is none.
    pub node: i32,
    /// Name of the offending node (may be empty).
    pub node_name: String,
    /// Operator of the offending node, such as `"Relu"` (may be empty).
    pub node_operator: String,
}

impl std::fmt::Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ONNX parser error {}: {}", self.code, self.description)?;
        if self.node >= 0 {
            write!(
                f,
                " (node {} `{}`, op `{}`)",
                self.node, self.node_name, self.node_operator
            )?;
        }
        Ok(())
    }
}

/// Why [`OnnxParser::parse`] failed: the parser's own error list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OnnxParseError(pub Vec<ParserError>);

impl std::fmt::Display for OnnxParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.first() {
            Some(first) => write!(f, "{first}")?,
            None => write!(f, "ONNX parse failed with no error recorded")?,
        }
        if self.0.len() > 1 {
            write!(f, " (+{} more)", self.0.len() - 1)?;
        }
        Ok(())
    }
}

impl std::error::Error for OnnxParseError {}

fn text(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: TensorRT returns a NUL-terminated string that stays valid while the tensor lives (a null pointer is handled above).
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// An `nvonnxparser::IParser` bound to a [`Network`].
///
/// Needs the ONNX parser library (`nvonnxparser_10.dll` / `libnvonnxparser.so.10`)
/// next to TensorRT; [`OnnxParser::new`] reports a loader error when it is absent.
#[derive(Debug)]
pub struct OnnxParser<'n> {
    raw: sys::trtIParser_t,
    // TensorRT keeps the logger pointer; the process-wide `ILogger` outlives
    // everything, but the handle keeps messages flowing to the closure.
    _logger: Arc<Logger>,
    _network: PhantomData<&'n ()>,
}

impl<'n> OnnxParser<'n> {
    /// Create a parser that fills `network`, reporting through `logger`.
    pub fn new(network: &'n Network<'_>, logger: Arc<Logger>) -> Result<Self> {
        let create = sys::onnx_parser()?.create_parser();
        // SAFETY: `create` was resolved from the ONNX parser library by the loader; `network` and the
        // process-wide `ILogger*` are live for the call, and the returned parser borrows the network ('n).
        let raw = unsafe {
            create(
                network.as_raw(),
                logger.as_raw(),
                sys::NV_ONNX_PARSER_VERSION,
            )
        };
        if raw.is_null() {
            return Err(Error::NullHandle {
                op: "createNvOnnxParser",
            });
        }
        Ok(Self {
            raw,
            _logger: logger,
            _network: PhantomData,
        })
    }

    /// Parse a serialized ONNX model (the bytes of a `.onnx` file) into the
    /// network. `path` is the model's file path, which the parser needs to find
    /// external weight files; pass `None` for a self-contained model.
    ///
    /// On failure the parser's error list is returned and then cleared.
    pub fn parse(
        &self,
        model: &[u8],
        path: Option<&str>,
    ) -> std::result::Result<(), OnnxParseError> {
        let path = match path.map(CString::new).transpose() {
            Ok(path) => path,
            Err(_) => {
                return Err(OnnxParseError(vec![ParserError {
                    code: 4,
                    description: "model path contains an interior NUL byte".into(),
                    node: -1,
                    node_name: String::new(),
                    node_operator: String::new(),
                }]));
            }
        };
        // SAFETY: `self.raw` is a live parser; `model` is a slice valid for the call and `path`, if any, is a NUL-terminated `CString` that outlives it.
        let ok = unsafe {
            ffi::trtOnnxParserParse(
                self.raw,
                model.as_ptr().cast(),
                model.len(),
                path.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()),
            )
        };
        if ok {
            return Ok(());
        }
        let errors = self.errors();
        // SAFETY: `self.raw` is a live parser owned by this value; any string argument is a NUL-terminated `CString` that outlives the call, and strings TensorRT returns are copied before the next parse.
        unsafe { ffi::trtOnnxParserClearErrors(self.raw) };
        Err(OnnxParseError(errors))
    }

    /// Whether the parser can import the ONNX operator `op` (such as `"Relu"`).
    pub fn supports_operator(&self, op: &str) -> Result<bool> {
        let op = CString::new(op)?;
        // SAFETY: `self.raw` is a live parser owned by this value; any string argument is a NUL-terminated `CString` that outlives the call, and strings TensorRT returns are copied before the next parse.
        Ok(unsafe { ffi::trtOnnxParserSupportsOperator(self.raw, op.as_ptr()) })
    }

    /// The errors recorded by the last parse.
    pub fn errors(&self) -> Vec<ParserError> {
        // SAFETY: `self.raw` is a live parser owned by this value; any string argument is a NUL-terminated `CString` that outlives the call, and strings TensorRT returns are copied before the next parse.
        let count = unsafe { ffi::trtOnnxParserGetNbErrors(self.raw) };
        (0..count)
            .map(|i| unsafe {
                ParserError {
                    code: ffi::trtOnnxParserErrorCode(self.raw, i),
                    description: text(ffi::trtOnnxParserErrorDesc(self.raw, i)),
                    node: ffi::trtOnnxParserErrorNode(self.raw, i),
                    node_name: text(ffi::trtOnnxParserErrorNodeName(self.raw, i)),
                    node_operator: text(ffi::trtOnnxParserErrorNodeOperator(self.raw, i)),
                }
            })
            .collect()
    }

    /// Raw `nvonnxparser::IParser*`.
    pub fn as_raw(&self) -> sys::trtIParser_t {
        self.raw
    }
}

impl Drop for OnnxParser<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from the matching create call, is destroyed exactly once here, and nothing can still borrow it (children borrow `&self`).
        unsafe { ffi::trtOnnxParserDestroy(self.raw) };
    }
}
