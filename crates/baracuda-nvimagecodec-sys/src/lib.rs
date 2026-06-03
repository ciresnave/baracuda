//! Raw FFI + dynamic loader for NVIDIA **nvImageCodec** — the unified GPU
//! image codec library (JPEG / JPEG2000 / TIFF / PNG / BMP / WebP / ...).
//!
//! `baracuda-nvimagecodec` wraps this with a safe, typed API. Use this
//! crate directly only if you need a function that the safe layer
//! hasn't wrapped yet (in which case please file a bug).
//!
//! nvImageCodec supersedes the standalone nvJPEG (which baracuda also wraps
//! via [`baracuda-nvjpeg-sys`](https://docs.rs/baracuda-nvjpeg-sys)) and
//! exposes a single batched high-level decode/encode pipeline:
//!
//! ```text
//! Instance  ──►  CodeStream (input bitstream)  ──►  GetImageInfo
//!                Image      (output buffer)
//!                Decoder    ──►  Decode(streams[], images[], batch) ──► Future
//! ```
//!
//! Symbols resolve lazily via `libloading`; there is no link-time dependency
//! on `libnvimgcodec.so` / `nvimgcodec_0.dll`. The struct layouts and enum
//! discriminants below target the **nvImageCodec 0.x** C ABI
//! (`nvimgcodec.h`); the versioned `struct_type` / `struct_size` preamble on
//! every public struct is the library's forward-compat mechanism and is
//! populated by the `new()` constructors here.
//!
//! Most users want the safe [`baracuda-nvimagecodec`] wrapper.
//!
//! [`baracuda-nvimagecodec`]: https://docs.rs/baracuda-nvimagecodec

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_int, c_uchar, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

// ---- opaque handles -------------------------------------------------------

pub type nvimgcodecInstance_t = *mut c_void;
pub type nvimgcodecCodeStream_t = *mut c_void;
pub type nvimgcodecImage_t = *mut c_void;
pub type nvimgcodecDecoder_t = *mut c_void;
pub type nvimgcodecEncoder_t = *mut c_void;
pub type nvimgcodecFuture_t = *mut c_void;

// ---- constants ------------------------------------------------------------

/// Fixed length of the `codec_name` field in [`nvimgcodecImageInfo_t`].
pub const NVIMGCODEC_MAX_CODEC_NAME_SIZE: usize = 256;
/// Maximum number of image planes carried by [`nvimgcodecImageInfo_t`].
pub const NVIMGCODEC_MAX_NUM_PLANES: usize = 32;
/// Maximum region dimensionality.
pub const NVIMGCODEC_MAX_NUM_DIM: usize = 5;
/// Maximum out-of-bounds fill channels.
pub const NVIMGCODEC_MAX_ROI_FILL_CHANNELS: usize = 5;

/// `device_id` sentinel: use the current CUDA device.
pub const NVIMGCODEC_DEVICE_CURRENT: c_int = -1;
/// `device_id` sentinel: CPU-only decode.
pub const NVIMGCODEC_DEVICE_CPU_ONLY: c_int = -99999;

// ---- structure type tag ---------------------------------------------------

/// Versioned struct discriminator. Every public struct begins with one of
/// these; the value must match the struct it heads. Ordering follows
/// `nvimgcodec.h` (sequential, no explicit hex values).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvimgcodecStructureType_t {
    Unknown = 0,
    Properties = 1,
    InstanceCreateInfo = 2,
    DeviceAllocator = 3,
    PinnedAllocator = 4,
    DecodeParams = 5,
    EncodeParams = 6,
    Orientation = 7,
    Region = 8,
    CodeStreamView = 9,
    CodeStreamInfo = 10,
    ImageInfo = 11,
    ImagePlaneInfo = 12,
    JpegImageInfo = 13,
    JpegEncodeParams = 14,
    TileGeometryInfo = 15,
    Jpeg2kEncodeParams = 16,
    Backend = 17,
    IoStreamDesc = 18,
    FrameworkDesc = 19,
    DecoderDesc = 20,
    EncoderDesc = 21,
    ParserDesc = 22,
    ImageDesc = 23,
    CodeStreamDesc = 24,
    DebugMessengerDesc = 25,
    DebugMessageData = 26,
    ExtensionDesc = 27,
    ExecutorDesc = 28,
    BackendParams = 29,
    ExecutionParams = 30,
}

// ---- value enums ----------------------------------------------------------

/// Output sample layout. `P_*` are planar, `I_*` interleaved.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvimgcodecSampleFormat_t {
    Unknown = 0,
    /// Planar, channels unchanged from source.
    PUnchanged = 1,
    /// Interleaved, channels unchanged from source.
    IUnchanged = 2,
    PY = 3,
    IY = 4,
    PYA = 5,
    IYA = 6,
    PRGB = 7,
    IRGB = 8,
    PBGR = 9,
    IBGR = 10,
    PYUV = 11,
    IYUV = 12,
    PRGBA = 13,
    IRGBA = 14,
    PYCCK = 15,
    IYCCK = 16,
    PCMYK = 17,
    ICMYK = 18,
}

/// Per-sample numeric type. The encoded value packs the byte-width in the
/// upper nibble (e.g. `UINT8 = 0x0802`, `FLOAT32 = 0x200B`).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvimgcodecSampleDataType_t {
    Unknown = 0,
    Int8 = 0x0801,
    Uint8 = 0x0802,
    Int16 = 0x1003,
    Uint16 = 0x1004,
    Int32 = 0x2005,
    Uint32 = 0x2006,
    Int64 = 0x4007,
    Uint64 = 0x4008,
    Float16 = 0x1009,
    Float32 = 0x200B,
    Float64 = 0x400D,
}

/// Chroma subsampling of the encoded image.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvimgcodecChromaSubsampling_t {
    /// 4:4:4 (no subsampling).
    None = 0,
    Css422 = 2,
    Css420 = 3,
    Css440 = 4,
    Css411 = 5,
    Css410 = 6,
    Gray = 7,
    Css410V = 8,
}

/// Color interpretation of the samples.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvimgcodecColorSpec_t {
    Unknown = 0,
    Srgb = 1,
    Gray = 2,
    Sycc = 3,
    Cmyk = 4,
    Ycck = 5,
    Palette = 6,
    IccProfile = 7,
}

/// Where the image `buffer` pointer lives.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvimgcodecImageBufferKind_t {
    Unknown = 0,
    /// Strided device (GPU) memory.
    StridedDevice = 1,
    /// Strided pinned/pageable host memory.
    StridedHost = 2,
}

// ---- processing status (bitfield) -----------------------------------------

/// Per-image processing status returned by the future. `uint32_t` bitfield;
/// `SUCCESS` (`0x1`) means the image decoded cleanly.
pub type nvimgcodecProcessingStatus_t = u32;

pub const NVIMGCODEC_PROCESSING_STATUS_UNKNOWN: nvimgcodecProcessingStatus_t = 0x0;
pub const NVIMGCODEC_PROCESSING_STATUS_SUCCESS: nvimgcodecProcessingStatus_t = 0x1;
pub const NVIMGCODEC_PROCESSING_STATUS_SATURATED: nvimgcodecProcessingStatus_t = 0x2;
pub const NVIMGCODEC_PROCESSING_STATUS_FAIL: nvimgcodecProcessingStatus_t = 0x3;
pub const NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED: nvimgcodecProcessingStatus_t = 0x7;
pub const NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED: nvimgcodecProcessingStatus_t = 0xb;
pub const NVIMGCODEC_PROCESSING_STATUS_BACKEND_UNSUPPORTED: nvimgcodecProcessingStatus_t = 0x13;
pub const NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED: nvimgcodecProcessingStatus_t = 0x83;

// ---- status ---------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct nvimgcodecStatus_t(pub i32);

impl nvimgcodecStatus_t {
    pub const SUCCESS: Self = Self(0);
    pub const NOT_INITIALIZED: Self = Self(1);
    pub const INVALID_PARAMETER: Self = Self(2);
    pub const BAD_CODESTREAM: Self = Self(3);
    pub const CODESTREAM_UNSUPPORTED: Self = Self(4);
    pub const ALLOCATOR_FAILURE: Self = Self(5);
    pub const EXECUTION_FAILED: Self = Self(6);
    pub const ARCH_MISMATCH: Self = Self(7);
    pub const INTERNAL_ERROR: Self = Self(8);
    pub const IMPLEMENTATION_UNSUPPORTED: Self = Self(9);
    pub const MISSED_DEPENDENCIES: Self = Self(10);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for nvimgcodecStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "NVIMGCODEC_STATUS_SUCCESS",
            1 => "NVIMGCODEC_STATUS_NOT_INITIALIZED",
            2 => "NVIMGCODEC_STATUS_INVALID_PARAMETER",
            3 => "NVIMGCODEC_STATUS_BAD_CODESTREAM",
            4 => "NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED",
            5 => "NVIMGCODEC_STATUS_ALLOCATOR_FAILURE",
            6 => "NVIMGCODEC_STATUS_EXECUTION_FAILED",
            7 => "NVIMGCODEC_STATUS_ARCH_MISMATCH",
            8 => "NVIMGCODEC_STATUS_INTERNAL_ERROR",
            9 => "NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED",
            10 => "NVIMGCODEC_STATUS_MISSED_DEPENDENCIES",
            _ => "NVIMGCODEC_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            3 => "codestream (bitstream) is malformed",
            4 => "codestream format is not supported by any backend",
            9 => "no installed backend can satisfy the request",
            10 => "a runtime dependency (codec extension) is missing",
            _ => "unrecognized nvImageCodec status code",
        }
    }
    fn is_success(self) -> bool {
        nvimgcodecStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "nvimgcodec"
    }
}

// ---- structs --------------------------------------------------------------

/// Library properties (version numbers). Returned by `nvimgcodecGetProperties`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvimgcodecProperties_t {
    pub struct_type: nvimgcodecStructureType_t,
    pub struct_size: usize,
    pub struct_next: *mut c_void,
    /// Encoded library version.
    pub version: u32,
    /// CUDA runtime version the library was built against.
    pub cudart_version: u32,
}

impl nvimgcodecProperties_t {
    pub fn new() -> Self {
        Self {
            struct_type: nvimgcodecStructureType_t::Properties,
            struct_size: core::mem::size_of::<Self>(),
            struct_next: core::ptr::null_mut(),
            version: 0,
            cudart_version: 0,
        }
    }
}

impl Default for nvimgcodecProperties_t {
    fn default() -> Self {
        Self::new()
    }
}

/// EXIF-style orientation descriptor embedded in [`nvimgcodecImageInfo_t`].
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvimgcodecOrientation_t {
    pub struct_type: nvimgcodecStructureType_t,
    pub struct_size: usize,
    pub struct_next: *mut c_void,
    /// Rotation in degrees (0 / 90 / 180 / 270).
    pub rotated: c_int,
    pub flip_x: c_int,
    pub flip_y: c_int,
}

impl nvimgcodecOrientation_t {
    pub fn new() -> Self {
        Self {
            struct_type: nvimgcodecStructureType_t::Orientation,
            struct_size: core::mem::size_of::<Self>(),
            struct_next: core::ptr::null_mut(),
            rotated: 0,
            flip_x: 0,
            flip_y: 0,
        }
    }
}

impl Default for nvimgcodecOrientation_t {
    fn default() -> Self {
        Self::new()
    }
}

/// One image plane's geometry + sample type.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvimgcodecImagePlaneInfo_t {
    pub struct_type: nvimgcodecStructureType_t,
    pub struct_size: usize,
    pub struct_next: *mut c_void,
    pub width: u32,
    pub height: u32,
    /// Bytes between successive rows.
    pub row_stride: usize,
    pub num_channels: u32,
    pub sample_type: nvimgcodecSampleDataType_t,
    /// Bits of precision actually used (0 = full width of `sample_type`).
    pub precision: u8,
}

impl nvimgcodecImagePlaneInfo_t {
    pub fn new() -> Self {
        Self {
            struct_type: nvimgcodecStructureType_t::ImagePlaneInfo,
            struct_size: core::mem::size_of::<Self>(),
            struct_next: core::ptr::null_mut(),
            width: 0,
            height: 0,
            row_stride: 0,
            num_channels: 0,
            sample_type: nvimgcodecSampleDataType_t::Unknown,
            precision: 0,
        }
    }
}

impl Default for nvimgcodecImagePlaneInfo_t {
    fn default() -> Self {
        Self::new()
    }
}

/// Full image descriptor — both the queried metadata (from a code stream)
/// and the output spec (for an output image). The single most important
/// struct in the API.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct nvimgcodecImageInfo_t {
    pub struct_type: nvimgcodecStructureType_t,
    pub struct_size: usize,
    pub struct_next: *mut c_void,
    /// NUL-terminated codec name (e.g. `b"jpeg"`, `b"png"`, `b"tiff"`).
    pub codec_name: [c_char; NVIMGCODEC_MAX_CODEC_NAME_SIZE],
    pub color_spec: nvimgcodecColorSpec_t,
    pub chroma_subsampling: nvimgcodecChromaSubsampling_t,
    pub sample_format: nvimgcodecSampleFormat_t,
    pub orientation: nvimgcodecOrientation_t,
    pub num_planes: u32,
    pub plane_info: [nvimgcodecImagePlaneInfo_t; NVIMGCODEC_MAX_NUM_PLANES],
    /// Output buffer pointer (device or host per `buffer_kind`). The library
    /// derives the required size from `plane_info` (`row_stride * height` per
    /// plane); there is no separate `buffer_size` field in `nvimgcodec.h`.
    pub buffer: *mut c_void,
    pub buffer_kind: nvimgcodecImageBufferKind_t,
    pub cuda_stream: cudaStream_t,
}

impl nvimgcodecImageInfo_t {
    pub fn new() -> Self {
        Self {
            struct_type: nvimgcodecStructureType_t::ImageInfo,
            struct_size: core::mem::size_of::<Self>(),
            struct_next: core::ptr::null_mut(),
            codec_name: [0; NVIMGCODEC_MAX_CODEC_NAME_SIZE],
            color_spec: nvimgcodecColorSpec_t::Unknown,
            chroma_subsampling: nvimgcodecChromaSubsampling_t::None,
            sample_format: nvimgcodecSampleFormat_t::Unknown,
            orientation: nvimgcodecOrientation_t::new(),
            num_planes: 0,
            plane_info: [nvimgcodecImagePlaneInfo_t::new(); NVIMGCODEC_MAX_NUM_PLANES],
            buffer: core::ptr::null_mut(),
            buffer_kind: nvimgcodecImageBufferKind_t::Unknown,
            cuda_stream: core::ptr::null_mut(),
        }
    }
}

impl Default for nvimgcodecImageInfo_t {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for nvimgcodecImageInfo_t {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("nvimgcodecImageInfo_t")
            .field("sample_format", &self.sample_format)
            .field("color_spec", &self.color_spec)
            .field("num_planes", &self.num_planes)
            .field("buffer", &self.buffer)
            .field("buffer_kind", &self.buffer_kind)
            .finish_non_exhaustive()
    }
}

/// Library-instance creation parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvimgcodecInstanceCreateInfo_t {
    pub struct_type: nvimgcodecStructureType_t,
    pub struct_size: usize,
    pub struct_next: *mut c_void,
    /// Load the built-in codecs (jpeg, png, ...). Non-zero = yes.
    pub load_builtin_modules: c_int,
    /// Discover and load external codec extension modules. Non-zero = yes.
    pub load_extension_modules: c_int,
    /// Optional override for the extension search path (NUL-terminated).
    pub extension_modules_path: *const c_char,
    pub create_debug_messenger: c_int,
    pub debug_messenger_desc: *const c_void,
    pub message_severity: u32,
    pub message_category: u32,
}

impl nvimgcodecInstanceCreateInfo_t {
    /// Default instance: built-in + extension modules on, no debug messenger.
    pub fn new() -> Self {
        Self {
            struct_type: nvimgcodecStructureType_t::InstanceCreateInfo,
            struct_size: core::mem::size_of::<Self>(),
            struct_next: core::ptr::null_mut(),
            load_builtin_modules: 1,
            load_extension_modules: 1,
            extension_modules_path: core::ptr::null(),
            create_debug_messenger: 0,
            debug_messenger_desc: core::ptr::null(),
            message_severity: 0,
            message_category: 0,
        }
    }
}

impl Default for nvimgcodecInstanceCreateInfo_t {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode-time parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvimgcodecDecodeParams_t {
    pub struct_type: nvimgcodecStructureType_t,
    pub struct_size: usize,
    pub struct_next: *mut c_void,
    /// Apply embedded EXIF orientation during decode. Non-zero = yes.
    pub apply_exif_orientation: c_int,
}

impl nvimgcodecDecodeParams_t {
    pub fn new() -> Self {
        Self {
            struct_type: nvimgcodecStructureType_t::DecodeParams,
            struct_size: core::mem::size_of::<Self>(),
            struct_next: core::ptr::null_mut(),
            apply_exif_orientation: 1,
        }
    }
}

impl Default for nvimgcodecDecodeParams_t {
    fn default() -> Self {
        Self::new()
    }
}

/// Decoder/encoder execution parameters (device, threads, backends).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvimgcodecExecutionParams_t {
    pub struct_type: nvimgcodecStructureType_t,
    pub struct_size: usize,
    pub struct_next: *mut c_void,
    pub device_allocator: *mut c_void,
    pub pinned_allocator: *mut c_void,
    pub max_num_cpu_threads: c_int,
    pub executor: *mut c_void,
    pub device_id: c_int,
    pub pre_init: c_int,
    pub skip_pre_sync: c_int,
    pub num_backends: c_int,
    pub backends: *const c_void,
}

impl nvimgcodecExecutionParams_t {
    /// Default: current device, library-chosen thread count, all backends.
    pub fn new() -> Self {
        Self {
            struct_type: nvimgcodecStructureType_t::ExecutionParams,
            struct_size: core::mem::size_of::<Self>(),
            struct_next: core::ptr::null_mut(),
            device_allocator: core::ptr::null_mut(),
            pinned_allocator: core::ptr::null_mut(),
            max_num_cpu_threads: 0,
            executor: core::ptr::null_mut(),
            device_id: NVIMGCODEC_DEVICE_CURRENT,
            pre_init: 0,
            skip_pre_sync: 0,
            num_backends: 0,
            backends: core::ptr::null(),
        }
    }
}

impl Default for nvimgcodecExecutionParams_t {
    fn default() -> Self {
        Self::new()
    }
}

// ---- function-pointer types ----------------------------------------------

pub type PFN_nvimgcodecGetProperties =
    unsafe extern "C" fn(properties: *mut nvimgcodecProperties_t) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecInstanceCreate = unsafe extern "C" fn(
    instance: *mut nvimgcodecInstance_t,
    create_info: *const nvimgcodecInstanceCreateInfo_t,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecInstanceDestroy =
    unsafe extern "C" fn(instance: nvimgcodecInstance_t) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecCodeStreamCreateFromHostMem = unsafe extern "C" fn(
    instance: nvimgcodecInstance_t,
    code_stream: *mut nvimgcodecCodeStream_t,
    data: *const c_uchar,
    length: usize,
    // Optional `const nvimgcodecCodeStreamView_t*` (parse offset / image
    // limit). Added on nvImageCodec `main`; older 0.x releases omit it. We
    // always pass NULL — correct on new libraries, and on the C ABIs
    // baracuda targets (x86-64 SysV / Win64) an extra trailing pointer the
    // callee ignores is harmless on older ones.
    code_stream_view: *const c_void,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecCodeStreamCreateFromFile = unsafe extern "C" fn(
    instance: nvimgcodecInstance_t,
    code_stream: *mut nvimgcodecCodeStream_t,
    file_name: *const c_char,
    // Optional `const nvimgcodecCodeStreamView_t*`; see the HostMem variant.
    code_stream_view: *const c_void,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecCodeStreamGetImageInfo = unsafe extern "C" fn(
    code_stream: nvimgcodecCodeStream_t,
    image_info: *mut nvimgcodecImageInfo_t,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecCodeStreamDestroy =
    unsafe extern "C" fn(code_stream: nvimgcodecCodeStream_t) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecImageCreate = unsafe extern "C" fn(
    instance: nvimgcodecInstance_t,
    image: *mut nvimgcodecImage_t,
    image_info: *const nvimgcodecImageInfo_t,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecImageGetImageInfo = unsafe extern "C" fn(
    image: nvimgcodecImage_t,
    image_info: *mut nvimgcodecImageInfo_t,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecImageDestroy =
    unsafe extern "C" fn(image: nvimgcodecImage_t) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecDecoderCreate = unsafe extern "C" fn(
    instance: nvimgcodecInstance_t,
    decoder: *mut nvimgcodecDecoder_t,
    exec_params: *const nvimgcodecExecutionParams_t,
    options: *const c_char,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecDecoderDestroy =
    unsafe extern "C" fn(decoder: nvimgcodecDecoder_t) -> nvimgcodecStatus_t;

/// Batched decode: `streams[0..batch_size]` ➜ `images[0..batch_size]`.
/// Asynchronous — completion is signalled through `*future`.
pub type PFN_nvimgcodecDecoderDecode = unsafe extern "C" fn(
    decoder: nvimgcodecDecoder_t,
    streams: *const nvimgcodecCodeStream_t,
    images: *const nvimgcodecImage_t,
    batch_size: c_int,
    params: *const nvimgcodecDecodeParams_t,
    future: *mut nvimgcodecFuture_t,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecFutureWaitForAll =
    unsafe extern "C" fn(future: nvimgcodecFuture_t) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecFutureGetProcessingStatus = unsafe extern "C" fn(
    future: nvimgcodecFuture_t,
    processing_status: *mut nvimgcodecProcessingStatus_t,
    size: *mut usize,
) -> nvimgcodecStatus_t;

pub type PFN_nvimgcodecFutureDestroy =
    unsafe extern "C" fn(future: nvimgcodecFuture_t) -> nvimgcodecStatus_t;

// ---- loader ---------------------------------------------------------------

fn nvimgcodec_candidates() -> Vec<String> {
    // nvImageCodec does not follow the `<name>64_<major>.dll` CUDA DLL
    // convention; the Windows DLL is `nvimgcodec_0.dll` and the Linux
    // soname is `libnvimgcodec.so.0`. Provide an explicit candidate list.
    match platform::os_family() {
        platform::OsFamily::Linux => vec![
            "libnvimgcodec.so.0".to_string(),
            "libnvimgcodec.so".to_string(),
        ],
        platform::OsFamily::Windows => {
            vec!["nvimgcodec_0.dll".to_string(), "nvimgcodec.dll".to_string()]
        }
        _ => Vec::new(),
    }
}

macro_rules! nvimgcodec_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        pub struct Nvimgcodec {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Nvimgcodec {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Nvimgcodec").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Nvimgcodec {
            $(
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
        }
    };
}

nvimgcodec_fns! {
    // Library properties
    nvimgcodec_get_properties as "nvimgcodecGetProperties": PFN_nvimgcodecGetProperties;

    // Instance lifecycle
    nvimgcodec_instance_create as "nvimgcodecInstanceCreate": PFN_nvimgcodecInstanceCreate;
    nvimgcodec_instance_destroy as "nvimgcodecInstanceDestroy": PFN_nvimgcodecInstanceDestroy;

    // Code stream (input bitstream)
    nvimgcodec_code_stream_create_from_host_mem as "nvimgcodecCodeStreamCreateFromHostMem":
        PFN_nvimgcodecCodeStreamCreateFromHostMem;
    nvimgcodec_code_stream_create_from_file as "nvimgcodecCodeStreamCreateFromFile":
        PFN_nvimgcodecCodeStreamCreateFromFile;
    nvimgcodec_code_stream_get_image_info as "nvimgcodecCodeStreamGetImageInfo":
        PFN_nvimgcodecCodeStreamGetImageInfo;
    nvimgcodec_code_stream_destroy as "nvimgcodecCodeStreamDestroy":
        PFN_nvimgcodecCodeStreamDestroy;

    // Image (output buffer descriptor)
    nvimgcodec_image_create as "nvimgcodecImageCreate": PFN_nvimgcodecImageCreate;
    nvimgcodec_image_get_image_info as "nvimgcodecImageGetImageInfo":
        PFN_nvimgcodecImageGetImageInfo;
    nvimgcodec_image_destroy as "nvimgcodecImageDestroy": PFN_nvimgcodecImageDestroy;

    // Decoder
    nvimgcodec_decoder_create as "nvimgcodecDecoderCreate": PFN_nvimgcodecDecoderCreate;
    nvimgcodec_decoder_destroy as "nvimgcodecDecoderDestroy": PFN_nvimgcodecDecoderDestroy;
    nvimgcodec_decoder_decode as "nvimgcodecDecoderDecode": PFN_nvimgcodecDecoderDecode;

    // Future (async completion)
    nvimgcodec_future_wait_for_all as "nvimgcodecFutureWaitForAll":
        PFN_nvimgcodecFutureWaitForAll;
    nvimgcodec_future_get_processing_status as "nvimgcodecFutureGetProcessingStatus":
        PFN_nvimgcodecFutureGetProcessingStatus;
    nvimgcodec_future_destroy as "nvimgcodecFutureDestroy": PFN_nvimgcodecFutureDestroy;
}

/// Resolve the process-wide nvImageCodec loader, opening `libnvimgcodec`
/// lazily on first call.
pub fn nvimgcodec() -> Result<&'static Nvimgcodec, LoaderError> {
    static NVIMGCODEC: OnceLock<Nvimgcodec> = OnceLock::new();
    if let Some(n) = NVIMGCODEC.get() {
        return Ok(n);
    }
    let candidates: Vec<&'static str> = nvimgcodec_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("nvimgcodec", candidates_leaked)?;
    let n = Nvimgcodec::empty(lib);
    let _ = NVIMGCODEC.set(n);
    Ok(NVIMGCODEC.get().expect("OnceLock set or lost race"))
}
