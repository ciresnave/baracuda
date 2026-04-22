//! Raw FFI + dynamic loader for NVIDIA nvJPEG (GPU JPEG decode/encode).

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_uchar, c_void};
use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

pub type nvjpegHandle_t = *mut c_void;
pub type nvjpegJpegState_t = *mut c_void;
pub type nvjpegEncoderState_t = *mut c_void;
pub type nvjpegEncoderParams_t = *mut c_void;
pub type nvjpegJpegStream_t = *mut c_void;
pub type nvjpegJpegDecoder_t = *mut c_void;
pub type nvjpegBufferPinned_t = *mut c_void;
pub type nvjpegBufferDevice_t = *mut c_void;
pub type nvjpegDecodeParams_t = *mut c_void;

/// nvJPEG backend selector.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvjpegBackend_t {
    Default = 0,
    Hybrid = 1,
    GpuHybrid = 2,
    Hardware = 3,
    GpuHybridDevice = 4,
    HardwareWithBackup = 5,
    Lossless = 6,
}

/// Chroma subsampling.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvjpegChromaSubsampling_t {
    Css444 = 0,
    Css422 = 1,
    Css420 = 2,
    Css440 = 3,
    Css411 = 4,
    Css410 = 5,
    CssGray = 6,
    Css410V = 7,
}

/// An output image's per-channel pointer + pitch table.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct nvjpegImage_t {
    pub channel: [*mut c_uchar; 4],
    pub pitch: [usize; 4],
}

impl Default for nvjpegImage_t {
    fn default() -> Self {
        Self {
            channel: [core::ptr::null_mut(); 4],
            pitch: [0; 4],
        }
    }
}

/// nvJPEG output format. Subset covering common decode modes.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum nvjpegOutputFormat_t {
    Unchanged = 0,
    Yuv = 1,
    Y = 2,
    Rgb = 3,
    Bgr = 4,
    /// RGB interleaved (single output pointer, 3 bytes per pixel).
    Rgbi = 5,
    /// BGR interleaved.
    Bgri = 6,
}

// ---- status ---------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct nvjpegStatus_t(pub i32);

impl nvjpegStatus_t {
    pub const SUCCESS: Self = Self(0);
    pub const NOT_INITIALIZED: Self = Self(1);
    pub const INVALID_PARAMETER: Self = Self(2);
    pub const BAD_JPEG: Self = Self(3);
    pub const JPEG_NOT_SUPPORTED: Self = Self(4);
    pub const ALLOCATOR_FAILURE: Self = Self(5);
    pub const EXECUTION_FAILED: Self = Self(6);
    pub const ARCH_MISMATCH: Self = Self(7);
    pub const INTERNAL_ERROR: Self = Self(8);
    pub const IMPLEMENTATION_NOT_SUPPORTED: Self = Self(9);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for nvjpegStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "NVJPEG_STATUS_SUCCESS",
            1 => "NVJPEG_STATUS_NOT_INITIALIZED",
            2 => "NVJPEG_STATUS_INVALID_PARAMETER",
            3 => "NVJPEG_STATUS_BAD_JPEG",
            4 => "NVJPEG_STATUS_JPEG_NOT_SUPPORTED",
            5 => "NVJPEG_STATUS_ALLOCATOR_FAILURE",
            6 => "NVJPEG_STATUS_EXECUTION_FAILED",
            _ => "NVJPEG_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            3 => "JPEG bitstream is malformed",
            4 => "JPEG feature not supported",
            _ => "unrecognized nvJPEG status code",
        }
    }
    fn is_success(self) -> bool {
        nvjpegStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "nvjpeg"
    }
}

// ---- function-pointer types ----------------------------------------------

pub type PFN_nvjpegCreateSimple =
    unsafe extern "C" fn(handle: *mut nvjpegHandle_t) -> nvjpegStatus_t;
pub type PFN_nvjpegDestroy = unsafe extern "C" fn(handle: nvjpegHandle_t) -> nvjpegStatus_t;
pub type PFN_nvjpegJpegStateCreate =
    unsafe extern "C" fn(handle: nvjpegHandle_t, state: *mut nvjpegJpegState_t) -> nvjpegStatus_t;
pub type PFN_nvjpegJpegStateDestroy =
    unsafe extern "C" fn(state: nvjpegJpegState_t) -> nvjpegStatus_t;
pub type PFN_nvjpegGetImageInfo = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    data: *const c_uchar,
    length: usize,
    n_components: *mut c_int,
    subsampling: *mut c_int,
    widths: *mut c_int,
    heights: *mut c_int,
) -> nvjpegStatus_t;
pub type PFN_nvjpegDecode = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
    data: *const c_uchar,
    length: usize,
    output_format: nvjpegOutputFormat_t,
    destination: *mut nvjpegImage_t,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

// ---- Full handle creation (backend-aware) ----

pub type PFN_nvjpegCreate = unsafe extern "C" fn(
    backend: nvjpegBackend_t,
    allocator: *mut c_void,
    pinned_allocator: *mut c_void,
    handle: *mut nvjpegHandle_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegCreateEx = unsafe extern "C" fn(
    backend: nvjpegBackend_t,
    allocator: *mut c_void,
    pinned_allocator: *mut c_void,
    flags: u32,
    handle: *mut nvjpegHandle_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegGetProperty =
    unsafe extern "C" fn(prop: c_int, value_out: *mut c_int) -> nvjpegStatus_t;

// ---- Batched decode ----

pub type PFN_nvjpegDecodeBatchedInitialize = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
    batch_size: c_int,
    max_cpu_threads: c_int,
    output_format: nvjpegOutputFormat_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeBatched = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
    data: *mut *const c_uchar,
    lengths: *const usize,
    destinations: *mut nvjpegImage_t,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeBatchedPreAllocate = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
    batch_size: c_int,
    width: c_int,
    height: c_int,
    subsampling: nvjpegChromaSubsampling_t,
    output_format: nvjpegOutputFormat_t,
) -> nvjpegStatus_t;

// ---- Hybrid / pipelined decoder (separate phase controls) ----

pub type PFN_nvjpegDecoderCreate = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    backend: nvjpegBackend_t,
    decoder_out: *mut nvjpegJpegDecoder_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegDecoderDestroy =
    unsafe extern "C" fn(decoder: nvjpegJpegDecoder_t) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeJpegHost = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    decoder: nvjpegJpegDecoder_t,
    state: nvjpegJpegState_t,
    decode_params: nvjpegDecodeParams_t,
    stream_in: nvjpegJpegStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeJpegTransferToDevice = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    decoder: nvjpegJpegDecoder_t,
    state: nvjpegJpegState_t,
    stream_in: nvjpegJpegStream_t,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeJpegDevice = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    decoder: nvjpegJpegDecoder_t,
    state: nvjpegJpegState_t,
    dest: *mut nvjpegImage_t,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

// ---- JPEG-stream parsing ----

pub type PFN_nvjpegJpegStreamCreate =
    unsafe extern "C" fn(handle: nvjpegHandle_t, stream: *mut nvjpegJpegStream_t) -> nvjpegStatus_t;

pub type PFN_nvjpegJpegStreamDestroy =
    unsafe extern "C" fn(stream: nvjpegJpegStream_t) -> nvjpegStatus_t;

pub type PFN_nvjpegJpegStreamParse = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    data: *const c_uchar,
    length: usize,
    save_metadata: c_int,
    save_stream: c_int,
    stream: nvjpegJpegStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegJpegStreamGetFrameDimensions = unsafe extern "C" fn(
    stream: nvjpegJpegStream_t,
    width: *mut u32,
    height: *mut u32,
) -> nvjpegStatus_t;

pub type PFN_nvjpegJpegStreamGetComponentsNum =
    unsafe extern "C" fn(stream: nvjpegJpegStream_t, components_num: *mut u32) -> nvjpegStatus_t;

pub type PFN_nvjpegJpegStreamGetChromaSubsampling = unsafe extern "C" fn(
    stream: nvjpegJpegStream_t,
    subsampling: *mut nvjpegChromaSubsampling_t,
) -> nvjpegStatus_t;

// ---- Buffer pools (pinned + device) ----

pub type PFN_nvjpegBufferPinnedCreate = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    allocator: *mut c_void,
    buffer_out: *mut nvjpegBufferPinned_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegBufferPinnedDestroy =
    unsafe extern "C" fn(buf: nvjpegBufferPinned_t) -> nvjpegStatus_t;

pub type PFN_nvjpegBufferDeviceCreate = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    allocator: *mut c_void,
    buffer_out: *mut nvjpegBufferDevice_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegBufferDeviceDestroy =
    unsafe extern "C" fn(buf: nvjpegBufferDevice_t) -> nvjpegStatus_t;

pub type PFN_nvjpegStateAttachPinnedBuffer = unsafe extern "C" fn(
    state: nvjpegJpegState_t,
    buf: nvjpegBufferPinned_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegStateAttachDeviceBuffer = unsafe extern "C" fn(
    state: nvjpegJpegState_t,
    buf: nvjpegBufferDevice_t,
) -> nvjpegStatus_t;

// ---- Decode params ----

pub type PFN_nvjpegDecodeParamsCreate = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    params_out: *mut nvjpegDecodeParams_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeParamsDestroy =
    unsafe extern "C" fn(params: nvjpegDecodeParams_t) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeParamsSetOutputFormat = unsafe extern "C" fn(
    params: nvjpegDecodeParams_t,
    output_format: nvjpegOutputFormat_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeParamsSetROI = unsafe extern "C" fn(
    params: nvjpegDecodeParams_t,
    offset_x: c_int,
    offset_y: c_int,
    roi_width: c_int,
    roi_height: c_int,
) -> nvjpegStatus_t;

pub type PFN_nvjpegDecodeParamsSetAllowCMYK =
    unsafe extern "C" fn(params: nvjpegDecodeParams_t, allow: c_int) -> nvjpegStatus_t;

// ---- Encoder ----

pub type PFN_nvjpegEncoderStateCreate = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    state_out: *mut nvjpegEncoderState_t,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegEncoderStateDestroy =
    unsafe extern "C" fn(state: nvjpegEncoderState_t) -> nvjpegStatus_t;

pub type PFN_nvjpegEncoderParamsCreate = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    params_out: *mut nvjpegEncoderParams_t,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegEncoderParamsDestroy =
    unsafe extern "C" fn(params: nvjpegEncoderParams_t) -> nvjpegStatus_t;

pub type PFN_nvjpegEncoderParamsSetQuality = unsafe extern "C" fn(
    params: nvjpegEncoderParams_t,
    quality: c_int,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegEncoderParamsSetSamplingFactors = unsafe extern "C" fn(
    params: nvjpegEncoderParams_t,
    subsampling: nvjpegChromaSubsampling_t,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegEncoderParamsSetOptimizedHuffman = unsafe extern "C" fn(
    params: nvjpegEncoderParams_t,
    optimized: c_int,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegEncodeImage = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    state: nvjpegEncoderState_t,
    params: nvjpegEncoderParams_t,
    source: *const nvjpegImage_t,
    input_format: c_int,
    image_width: c_int,
    image_height: c_int,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegEncodeYUV = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    state: nvjpegEncoderState_t,
    params: nvjpegEncoderParams_t,
    source: *const nvjpegImage_t,
    subsampling: nvjpegChromaSubsampling_t,
    image_width: c_int,
    image_height: c_int,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

pub type PFN_nvjpegEncodeRetrieveBitstream = unsafe extern "C" fn(
    handle: nvjpegHandle_t,
    state: nvjpegEncoderState_t,
    data: *mut c_uchar,
    length_inout: *mut usize,
    stream: cudaStream_t,
) -> nvjpegStatus_t;

// ---- loader --------------------------------------------------------------

fn nvjpeg_candidates() -> Vec<String> {
    platform::versioned_library_candidates("nvjpeg", &["13", "12", "11"])
}

macro_rules! nvjpeg_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        pub struct Nvjpeg {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Nvjpeg {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Nvjpeg").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Nvjpeg {
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

nvjpeg_fns! {
    // Handle lifecycle
    nvjpeg_create_simple as "nvjpegCreateSimple": PFN_nvjpegCreateSimple;
    nvjpeg_create as "nvjpegCreate": PFN_nvjpegCreate;
    nvjpeg_create_ex as "nvjpegCreateEx": PFN_nvjpegCreateEx;
    nvjpeg_destroy as "nvjpegDestroy": PFN_nvjpegDestroy;
    nvjpeg_get_property as "nvjpegGetProperty": PFN_nvjpegGetProperty;

    // State lifecycle
    nvjpeg_jpeg_state_create as "nvjpegJpegStateCreate": PFN_nvjpegJpegStateCreate;
    nvjpeg_jpeg_state_destroy as "nvjpegJpegStateDestroy": PFN_nvjpegJpegStateDestroy;

    // Single decode
    nvjpeg_get_image_info as "nvjpegGetImageInfo": PFN_nvjpegGetImageInfo;
    nvjpeg_decode as "nvjpegDecode": PFN_nvjpegDecode;

    // Batched decode
    nvjpeg_decode_batched_initialize as "nvjpegDecodeBatchedInitialize":
        PFN_nvjpegDecodeBatchedInitialize;
    nvjpeg_decode_batched as "nvjpegDecodeBatched": PFN_nvjpegDecodeBatched;
    nvjpeg_decode_batched_pre_allocate as "nvjpegDecodeBatchedPreAllocate":
        PFN_nvjpegDecodeBatchedPreAllocate;

    // Pipelined / hybrid decoder
    nvjpeg_decoder_create as "nvjpegDecoderCreate": PFN_nvjpegDecoderCreate;
    nvjpeg_decoder_destroy as "nvjpegDecoderDestroy": PFN_nvjpegDecoderDestroy;
    nvjpeg_decode_jpeg_host as "nvjpegDecodeJpegHost": PFN_nvjpegDecodeJpegHost;
    nvjpeg_decode_jpeg_transfer_to_device as "nvjpegDecodeJpegTransferToDevice":
        PFN_nvjpegDecodeJpegTransferToDevice;
    nvjpeg_decode_jpeg_device as "nvjpegDecodeJpegDevice": PFN_nvjpegDecodeJpegDevice;

    // JPEG-stream parsing
    nvjpeg_jpeg_stream_create as "nvjpegJpegStreamCreate": PFN_nvjpegJpegStreamCreate;
    nvjpeg_jpeg_stream_destroy as "nvjpegJpegStreamDestroy": PFN_nvjpegJpegStreamDestroy;
    nvjpeg_jpeg_stream_parse as "nvjpegJpegStreamParse": PFN_nvjpegJpegStreamParse;
    nvjpeg_jpeg_stream_get_frame_dimensions as "nvjpegJpegStreamGetFrameDimensions":
        PFN_nvjpegJpegStreamGetFrameDimensions;
    nvjpeg_jpeg_stream_get_components_num as "nvjpegJpegStreamGetComponentsNum":
        PFN_nvjpegJpegStreamGetComponentsNum;
    nvjpeg_jpeg_stream_get_chroma_subsampling as "nvjpegJpegStreamGetChromaSubsampling":
        PFN_nvjpegJpegStreamGetChromaSubsampling;

    // Buffer pools
    nvjpeg_buffer_pinned_create as "nvjpegBufferPinnedCreate": PFN_nvjpegBufferPinnedCreate;
    nvjpeg_buffer_pinned_destroy as "nvjpegBufferPinnedDestroy": PFN_nvjpegBufferPinnedDestroy;
    nvjpeg_buffer_device_create as "nvjpegBufferDeviceCreate": PFN_nvjpegBufferDeviceCreate;
    nvjpeg_buffer_device_destroy as "nvjpegBufferDeviceDestroy": PFN_nvjpegBufferDeviceDestroy;
    nvjpeg_state_attach_pinned_buffer as "nvjpegStateAttachPinnedBuffer":
        PFN_nvjpegStateAttachPinnedBuffer;
    nvjpeg_state_attach_device_buffer as "nvjpegStateAttachDeviceBuffer":
        PFN_nvjpegStateAttachDeviceBuffer;

    // Decode params
    nvjpeg_decode_params_create as "nvjpegDecodeParamsCreate": PFN_nvjpegDecodeParamsCreate;
    nvjpeg_decode_params_destroy as "nvjpegDecodeParamsDestroy": PFN_nvjpegDecodeParamsDestroy;
    nvjpeg_decode_params_set_output_format as "nvjpegDecodeParamsSetOutputFormat":
        PFN_nvjpegDecodeParamsSetOutputFormat;
    nvjpeg_decode_params_set_roi as "nvjpegDecodeParamsSetROI": PFN_nvjpegDecodeParamsSetROI;
    nvjpeg_decode_params_set_allow_cmyk as "nvjpegDecodeParamsSetAllowCMYK":
        PFN_nvjpegDecodeParamsSetAllowCMYK;

    // Encoder
    nvjpeg_encoder_state_create as "nvjpegEncoderStateCreate": PFN_nvjpegEncoderStateCreate;
    nvjpeg_encoder_state_destroy as "nvjpegEncoderStateDestroy": PFN_nvjpegEncoderStateDestroy;
    nvjpeg_encoder_params_create as "nvjpegEncoderParamsCreate": PFN_nvjpegEncoderParamsCreate;
    nvjpeg_encoder_params_destroy as "nvjpegEncoderParamsDestroy":
        PFN_nvjpegEncoderParamsDestroy;
    nvjpeg_encoder_params_set_quality as "nvjpegEncoderParamsSetQuality":
        PFN_nvjpegEncoderParamsSetQuality;
    nvjpeg_encoder_params_set_sampling_factors as "nvjpegEncoderParamsSetSamplingFactors":
        PFN_nvjpegEncoderParamsSetSamplingFactors;
    nvjpeg_encoder_params_set_optimized_huffman as "nvjpegEncoderParamsSetOptimizedHuffman":
        PFN_nvjpegEncoderParamsSetOptimizedHuffman;
    nvjpeg_encode_image as "nvjpegEncodeImage": PFN_nvjpegEncodeImage;
    nvjpeg_encode_yuv as "nvjpegEncodeYUV": PFN_nvjpegEncodeYUV;
    nvjpeg_encode_retrieve_bitstream as "nvjpegEncodeRetrieveBitstream":
        PFN_nvjpegEncodeRetrieveBitstream;
}

pub fn nvjpeg() -> Result<&'static Nvjpeg, LoaderError> {
    static NVJPEG: OnceLock<Nvjpeg> = OnceLock::new();
    if let Some(n) = NVJPEG.get() {
        return Ok(n);
    }
    let candidates: Vec<&'static str> = nvjpeg_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("nvjpeg", candidates_leaked)?;
    let n = Nvjpeg::empty(lib);
    let _ = NVJPEG.set(n);
    Ok(NVJPEG.get().expect("OnceLock set or lost race"))
}
