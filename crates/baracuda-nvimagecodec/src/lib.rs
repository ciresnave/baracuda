//! Safe Rust wrappers for **NVIDIA nvImageCodec** — the unified GPU image
//! codec library (JPEG / JPEG2000 / TIFF / PNG / BMP / WebP / ...).
//!
//! nvImageCodec supersedes the standalone nvJPEG (also wrapped by baracuda,
//! see [`baracuda-nvjpeg`](https://docs.rs/baracuda-nvjpeg)) and unifies
//! decode of many container formats behind one batched pipeline. This v0.1
//! covers **single-image decode** to a caller-provided device buffer:
//!
//! ```text
//! Instance::new()
//!   └─ CodeStream::from_host_mem(&instance, &jpeg_bytes)   // input bitstream
//!        └─ .image_info()                                   // probe dims / codec
//!   └─ Decoder::new(&instance)
//!   └─ Image::new_interleaved_rgb8(&instance, &mut buf, w, h, stream)
//!        └─ decoder.decode(&stream, &image, &params)        // -> Future
//!             └─ future.wait()                              // block for completion
//! ```
//!
//! # Lifetimes
//!
//! `nvimgcodecCodeStreamCreateFromHostMem` **borrows** the input byte slice
//! — the bytes must stay valid until the [`CodeStream`] is dropped. This is
//! enforced at compile time: [`CodeStream`] carries a `PhantomData<&'data
//! [u8]>` and cannot outlive the slice it was built from.
//!
//! Likewise an [`Image`] borrows its output [`DeviceBuffer`] for `'buf`; the
//! buffer cannot be moved or freed while the image references it.
//!
//! Because decode is asynchronous on the supplied stream, the output buffer
//! must not be read until [`Future::wait`] returns (or the stream is
//! synchronized).
//!
//! Encoding, batch decode, and the non-JPEG/PNG/TIFF formats are Tier-2 and
//! land in follow-ups.

#![warn(missing_debug_implementations)]

use core::ffi::{c_char, c_int, c_uchar};
use core::marker::PhantomData;
use std::ffi::CString;
use std::path::Path;

use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_nvimagecodec_sys::{
    nvimgcodec, nvimgcodecCodeStream_t, nvimgcodecDecodeParams_t, nvimgcodecDecoder_t,
    nvimgcodecFuture_t, nvimgcodecImageBufferKind_t, nvimgcodecImageInfo_t, nvimgcodecImage_t,
    nvimgcodecInstanceCreateInfo_t, nvimgcodecInstance_t, nvimgcodecProcessingStatus_t,
    nvimgcodecSampleDataType_t, nvimgcodecStatus_t, NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
};

/// Re-exported sample-layout enum (planar `P_*` / interleaved `I_*`).
pub use baracuda_nvimagecodec_sys::nvimgcodecSampleFormat_t as SampleFormat;
/// Re-exported chroma-subsampling enum.
pub use baracuda_nvimagecodec_sys::nvimgcodecChromaSubsampling_t as ChromaSubsampling;
/// Re-exported color-spec enum.
pub use baracuda_nvimagecodec_sys::nvimgcodecColorSpec_t as ColorSpec;
/// Re-exported raw image-info struct, for advanced callers.
pub use baracuda_nvimagecodec_sys::nvimgcodecImageInfo_t as RawImageInfo;

/// Error type for nvImageCodec operations.
pub type Error = baracuda_core::Error<nvimgcodecStatus_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: nvimgcodecStatus_t) -> Result<()> {
    Error::check(status)
}

#[inline]
fn stream_raw(stream: Option<&Stream>) -> cudaStream_t {
    stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as cudaStream_t)
}

// ---- instance -------------------------------------------------------------

/// nvImageCodec library instance. Owns the loaded codec registry; shared
/// across decode sessions. Create one per process (or per device).
pub struct Instance {
    raw: nvimgcodecInstance_t,
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl core::fmt::Debug for Instance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Instance").field("raw", &self.raw).finish()
    }
}

impl Instance {
    /// Create an instance loading the built-in codecs plus any discoverable
    /// extension modules.
    pub fn new() -> Result<Self> {
        let n = nvimgcodec()?;
        let create = n.nvimgcodec_instance_create()?;
        let info = nvimgcodecInstanceCreateInfo_t::new();
        let mut raw: nvimgcodecInstance_t = core::ptr::null_mut();
        check(unsafe { create(&mut raw, &info) })?;
        Ok(Self { raw })
    }

    /// Library + CUDA-runtime version the loaded `libnvimgcodec` reports.
    pub fn properties(&self) -> Result<Properties> {
        let n = nvimgcodec()?;
        let getp = n.nvimgcodec_get_properties()?;
        let mut props = baracuda_nvimagecodec_sys::nvimgcodecProperties_t::new();
        check(unsafe { getp(&mut props) })?;
        Ok(Properties {
            version: props.version,
            cudart_version: props.cudart_version,
        })
    }

    /// Raw `nvimgcodecInstance_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> nvimgcodecInstance_t {
        self.raw
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if let Ok(n) = nvimgcodec() {
            if let Ok(destroy) = n.nvimgcodec_instance_destroy() {
                let _ = unsafe { destroy(self.raw) };
            }
        }
    }
}

/// Library version numbers reported by [`Instance::properties`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Properties {
    /// Encoded nvImageCodec version.
    pub version: u32,
    /// CUDA runtime version the library was built against.
    pub cudart_version: u32,
}

// ---- code stream (input bitstream) ---------------------------------------

/// A parsed input bitstream. Borrows the source bytes for `'data` when
/// created from host memory (see the module-level lifetime notes).
pub struct CodeStream<'data> {
    raw: nvimgcodecCodeStream_t,
    _data: PhantomData<&'data [u8]>,
}

unsafe impl Send for CodeStream<'_> {}

impl core::fmt::Debug for CodeStream<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CodeStream")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl<'data> CodeStream<'data> {
    /// Build a code stream over an in-memory bitstream. `data` must outlive
    /// the returned stream — the library may parse it lazily.
    pub fn from_host_mem(instance: &Instance, data: &'data [u8]) -> Result<Self> {
        let n = nvimgcodec()?;
        let create = n.nvimgcodec_code_stream_create_from_host_mem()?;
        let mut raw: nvimgcodecCodeStream_t = core::ptr::null_mut();
        check(unsafe {
            create(
                instance.raw,
                &mut raw,
                data.as_ptr() as *const c_uchar,
                data.len(),
                // code_stream_view: NULL = parse the whole bitstream.
                core::ptr::null(),
            )
        })?;
        Ok(Self {
            raw,
            _data: PhantomData,
        })
    }

    /// Probe the stream's image metadata (dimensions, codec, sample format)
    /// without decoding.
    pub fn image_info(&self) -> Result<ImageInfo> {
        let n = nvimgcodec()?;
        let getinfo = n.nvimgcodec_code_stream_get_image_info()?;
        let mut info = nvimgcodecImageInfo_t::new();
        check(unsafe { getinfo(self.raw, &mut info) })?;
        Ok(ImageInfo::from_raw(info))
    }

    /// Raw `nvimgcodecCodeStream_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> nvimgcodecCodeStream_t {
        self.raw
    }
}

impl CodeStream<'static> {
    /// Build a code stream that reads from a file path. The library owns the
    /// file handle, so the resulting stream has no borrow (`'static`).
    pub fn from_file(instance: &Instance, path: &Path) -> Result<Self> {
        let n = nvimgcodec()?;
        let create = n.nvimgcodec_code_stream_create_from_file()?;
        let c_path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| Error::Status {
            status: nvimgcodecStatus_t::INVALID_PARAMETER,
        })?;
        let mut raw: nvimgcodecCodeStream_t = core::ptr::null_mut();
        check(unsafe {
            create(
                instance.raw,
                &mut raw,
                c_path.as_ptr() as *const c_char,
                // code_stream_view: NULL = default parsing.
                core::ptr::null(),
            )
        })?;
        Ok(Self {
            raw,
            _data: PhantomData,
        })
    }
}

impl Drop for CodeStream<'_> {
    fn drop(&mut self) {
        if let Ok(n) = nvimgcodec() {
            if let Ok(destroy) = n.nvimgcodec_code_stream_destroy() {
                let _ = unsafe { destroy(self.raw) };
            }
        }
    }
}

/// Decoded view of an [`nvimgcodecImageInfo_t`] — the fields most callers
/// need, plus the raw struct for advanced use.
#[derive(Clone, Debug)]
pub struct ImageInfo {
    /// Image width in pixels (component 0).
    pub width: u32,
    /// Image height in pixels (component 0).
    pub height: u32,
    /// Number of planes in the source layout.
    pub num_planes: u32,
    /// Codec name as reported by the library (e.g. `"jpeg"`, `"png"`).
    pub codec_name: String,
    /// Source sample layout.
    pub sample_format: SampleFormat,
    /// Source color spec.
    pub color_spec: ColorSpec,
    raw: RawImageInfo,
}

impl ImageInfo {
    fn from_raw(raw: RawImageInfo) -> Self {
        // codec_name is a fixed NUL-terminated char buffer.
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(raw.codec_name.as_ptr() as *const u8, raw.codec_name.len())
        };
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        let codec_name = String::from_utf8_lossy(&bytes[..end]).into_owned();
        // Per-plane dimensions live in plane_info[0] for the queried stream.
        let p0 = raw.plane_info[0];
        Self {
            width: p0.width,
            height: p0.height,
            num_planes: raw.num_planes,
            codec_name,
            sample_format: raw.sample_format,
            color_spec: raw.color_spec,
            raw,
        }
    }

    /// The full underlying struct, for callers that need fields this view
    /// doesn't surface.
    #[inline]
    pub fn raw(&self) -> &RawImageInfo {
        &self.raw
    }
}

// ---- image (output buffer descriptor) ------------------------------------

/// An output image descriptor wrapping a caller-provided device buffer.
/// Borrows the buffer for `'buf`.
pub struct Image<'buf> {
    raw: nvimgcodecImage_t,
    _buf: PhantomData<&'buf mut DeviceBuffer<u8>>,
}

unsafe impl Send for Image<'_> {}

impl core::fmt::Debug for Image<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Image")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl<'buf> Image<'buf> {
    /// Build an output image for **interleaved 8-bit RGB** decode into
    /// `buffer`, which must hold at least `width * height * 3` bytes of
    /// device memory. Decode runs on `stream` (or the default stream if
    /// `None`).
    ///
    /// The buffer is borrowed mutably for the lifetime of the image; the
    /// decoded pixels are written there once the decode [`Future`] completes.
    pub fn new_interleaved_rgb8(
        instance: &Instance,
        buffer: &'buf mut DeviceBuffer<u8>,
        width: u32,
        height: u32,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        let needed = (width as usize) * (height as usize) * 3;
        assert!(
            buffer.len() >= needed,
            "output buffer too small: have {}, need {needed} for {width}x{height} RGB8",
            buffer.len(),
        );

        let mut info = nvimgcodecImageInfo_t::new();
        info.sample_format = SampleFormat::IRGB;
        info.color_spec = ColorSpec::Srgb;
        info.num_planes = 1;
        info.plane_info[0].width = width;
        info.plane_info[0].height = height;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].row_stride = (width as usize) * 3;
        info.plane_info[0].sample_type = nvimgcodecSampleDataType_t::Uint8;
        info.plane_info[0].precision = 8;
        info.buffer = buffer.as_raw().0 as *mut core::ffi::c_void;
        info.buffer_kind = nvimgcodecImageBufferKind_t::StridedDevice;
        info.cuda_stream = stream_raw(stream);

        let n = nvimgcodec()?;
        let create = n.nvimgcodec_image_create()?;
        let mut raw: nvimgcodecImage_t = core::ptr::null_mut();
        check(unsafe { create(instance.raw, &mut raw, &info) })?;
        Ok(Self {
            raw,
            _buf: PhantomData,
        })
    }

    /// Raw `nvimgcodecImage_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> nvimgcodecImage_t {
        self.raw
    }
}

impl Drop for Image<'_> {
    fn drop(&mut self) {
        if let Ok(n) = nvimgcodec() {
            if let Ok(destroy) = n.nvimgcodec_image_destroy() {
                let _ = unsafe { destroy(self.raw) };
            }
        }
    }
}

// ---- decode params --------------------------------------------------------

/// Decode-time options.
#[derive(Copy, Clone, Debug)]
pub struct DecodeParams {
    /// Apply embedded EXIF orientation while decoding.
    pub apply_exif_orientation: bool,
}

impl Default for DecodeParams {
    fn default() -> Self {
        Self {
            apply_exif_orientation: true,
        }
    }
}

impl DecodeParams {
    fn to_raw(self) -> nvimgcodecDecodeParams_t {
        let mut p = nvimgcodecDecodeParams_t::new();
        p.apply_exif_orientation = self.apply_exif_orientation as c_int;
        p
    }
}

// ---- decoder --------------------------------------------------------------

/// A decoder bound to an [`Instance`]. Drives the batched decode entry point
/// (this v0.1 exposes the single-image case via [`Decoder::decode`]).
pub struct Decoder {
    raw: nvimgcodecDecoder_t,
}

unsafe impl Send for Decoder {}

impl core::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Decoder")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl Decoder {
    /// Create a decoder on the current CUDA device with library-default
    /// execution parameters (all available backends, auto thread count).
    pub fn new(instance: &Instance) -> Result<Self> {
        let n = nvimgcodec()?;
        let create = n.nvimgcodec_decoder_create()?;
        let exec = baracuda_nvimagecodec_sys::nvimgcodecExecutionParams_t::new();
        let mut raw: nvimgcodecDecoder_t = core::ptr::null_mut();
        // `options` is an optional NUL-terminated tuning string; pass null.
        check(unsafe { create(instance.raw, &mut raw, &exec, core::ptr::null()) })?;
        Ok(Self { raw })
    }

    /// Decode a single `code_stream` into `image`. Returns immediately with
    /// a [`Future`]; the decode runs asynchronously on the image's stream.
    /// Call [`Future::wait`] (or synchronize the stream) before reading the
    /// output buffer.
    pub fn decode(
        &self,
        code_stream: &CodeStream<'_>,
        image: &Image<'_>,
        params: &DecodeParams,
    ) -> Result<Future> {
        let n = nvimgcodec()?;
        let decode = n.nvimgcodec_decoder_decode()?;
        let raw_params = params.to_raw();
        let streams = [code_stream.raw];
        let images = [image.raw];
        let mut future: nvimgcodecFuture_t = core::ptr::null_mut();
        check(unsafe {
            decode(
                self.raw,
                streams.as_ptr(),
                images.as_ptr(),
                1,
                &raw_params,
                &mut future,
            )
        })?;
        Ok(Future { raw: future })
    }

    /// Raw `nvimgcodecDecoder_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> nvimgcodecDecoder_t {
        self.raw
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        if let Ok(n) = nvimgcodec() {
            if let Ok(destroy) = n.nvimgcodec_decoder_destroy() {
                let _ = unsafe { destroy(self.raw) };
            }
        }
    }
}

// ---- future (async completion) -------------------------------------------

/// Handle to an in-flight (or completed) decode batch.
pub struct Future {
    raw: nvimgcodecFuture_t,
}

unsafe impl Send for Future {}

impl core::fmt::Debug for Future {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Future")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl Future {
    /// Block the host until every image in the batch has finished processing.
    pub fn wait(&self) -> Result<()> {
        let n = nvimgcodec()?;
        let wait = n.nvimgcodec_future_wait_for_all()?;
        check(unsafe { wait(self.raw) })
    }

    /// Per-image processing-status bitfields (one entry per batch image).
    /// `0x1` (`NVIMGCODEC_PROCESSING_STATUS_SUCCESS`) means success.
    pub fn processing_statuses(&self) -> Result<Vec<nvimgcodecProcessingStatus_t>> {
        let n = nvimgcodec()?;
        let getstatus = n.nvimgcodec_future_get_processing_status()?;
        // First query the count (size), then fetch into a sized buffer.
        let mut size: usize = 0;
        check(unsafe { getstatus(self.raw, core::ptr::null_mut(), &mut size) })?;
        let mut statuses = vec![0u32; size];
        if size > 0 {
            check(unsafe { getstatus(self.raw, statuses.as_mut_ptr(), &mut size) })?;
        }
        Ok(statuses)
    }

    /// Convenience: `true` iff every image decoded with `SUCCESS`.
    pub fn all_succeeded(&self) -> Result<bool> {
        Ok(self
            .processing_statuses()?
            .iter()
            .all(|&s| s == NVIMGCODEC_PROCESSING_STATUS_SUCCESS))
    }

    /// Raw `nvimgcodecFuture_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> nvimgcodecFuture_t {
        self.raw
    }
}

impl Drop for Future {
    fn drop(&mut self) {
        if let Ok(n) = nvimgcodec() {
            if let Ok(destroy) = n.nvimgcodec_future_destroy() {
                let _ = unsafe { destroy(self.raw) };
            }
        }
    }
}
