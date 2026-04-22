//! Safe Rust wrappers for NVIDIA nvJPEG.
//!
//! v0.1 covers single-image JPEG decode with the default CPU+GPU hybrid
//! backend. Encoding and the batched/multi-phase decoders land in
//! follow-ups.

#![warn(missing_debug_implementations)]

use core::ffi::{c_int, c_uchar};

use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_nvjpeg_sys::{
    nvjpeg, nvjpegBufferDevice_t, nvjpegBufferPinned_t, nvjpegDecodeParams_t,
    nvjpegEncoderParams_t, nvjpegEncoderState_t, nvjpegHandle_t, nvjpegImage_t, nvjpegJpegDecoder_t,
    nvjpegJpegState_t, nvjpegJpegStream_t, nvjpegOutputFormat_t, nvjpegStatus_t,
};

/// Error type for nvJPEG operations.
pub type Error = baracuda_core::Error<nvjpegStatus_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: nvjpegStatus_t) -> Result<()> {
    Error::check(status)
}

/// Output pixel format.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    /// Interleaved 24-bit RGB. Output is a single buffer with 3 bytes per pixel.
    Rgbi,
    /// Interleaved 24-bit BGR.
    Bgri,
    /// Planar RGB (3 separate channel buffers).
    Rgb,
    /// Single-channel luminance.
    Y,
}

impl PixelFormat {
    fn raw(self) -> nvjpegOutputFormat_t {
        match self {
            PixelFormat::Rgbi => nvjpegOutputFormat_t::Rgbi,
            PixelFormat::Bgri => nvjpegOutputFormat_t::Bgri,
            PixelFormat::Rgb => nvjpegOutputFormat_t::Rgb,
            PixelFormat::Y => nvjpegOutputFormat_t::Y,
        }
    }

    /// Bytes per pixel in the output buffer(s).
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::Rgbi | PixelFormat::Bgri => 3,
            PixelFormat::Rgb => 1, // per-plane
            PixelFormat::Y => 1,
        }
    }

    fn is_interleaved(self) -> bool {
        matches!(self, PixelFormat::Rgbi | PixelFormat::Bgri)
    }
}

/// nvJPEG library handle. Shared across decode sessions.
pub struct Handle {
    handle: nvjpegHandle_t,
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("nvjpeg::Handle")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Handle {
    /// Create a new nvJPEG handle with default (hybrid) backend.
    pub fn new() -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_create_simple()?;
        let mut h: nvjpegHandle_t = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self { handle: h })
    }

    /// Basic JPEG metadata: per-component dimensions (4 is max channels
    /// nvJPEG reports), subsampling code, and component count.
    pub fn image_info(&self, jpeg_bytes: &[u8]) -> Result<ImageInfo> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_get_image_info()?;
        let mut components: c_int = 0;
        let mut subsampling: c_int = 0;
        let mut widths = [0i32; 4];
        let mut heights = [0i32; 4];
        check(unsafe {
            cu(
                self.handle,
                jpeg_bytes.as_ptr() as *const c_uchar,
                jpeg_bytes.len(),
                &mut components,
                &mut subsampling,
                widths.as_mut_ptr(),
                heights.as_mut_ptr(),
            )
        })?;
        Ok(ImageInfo {
            components: components as u32,
            subsampling: subsampling as u32,
            widths,
            heights,
        })
    }

    /// Raw `nvjpegHandle_t`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> nvjpegHandle_t {
        self.handle
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Parsed JPEG metadata.
#[derive(Copy, Clone, Debug)]
pub struct ImageInfo {
    /// Number of image components (1 for Y, 3 for YCbCr/RGB).
    pub components: u32,
    /// nvJPEG chroma-subsampling enum value (e.g. `444`, `422`, `420`).
    pub subsampling: u32,
    /// Per-component widths, 4 entries (only the first `components` are valid).
    pub widths: [i32; 4],
    /// Per-component heights.
    pub heights: [i32; 4],
}

impl ImageInfo {
    /// Output width (in pixels) of component 0.
    pub fn width(&self) -> i32 {
        self.widths[0]
    }
    /// Output height.
    pub fn height(&self) -> i32 {
        self.heights[0]
    }
}

/// A decode session (transient per-image scratch state).
pub struct State {
    state: nvjpegJpegState_t,
}

unsafe impl Send for State {}

impl core::fmt::Debug for State {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("nvjpeg::State")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl State {
    /// Create a per-handle decode state.
    pub fn new(handle: &Handle) -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_jpeg_state_create()?;
        let mut st: nvjpegJpegState_t = core::ptr::null_mut();
        check(unsafe { cu(handle.handle, &mut st) })?;
        Ok(Self { state: st })
    }
}

impl Drop for State {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_jpeg_state_destroy() {
                let _ = unsafe { cu(self.state) };
            }
        }
    }
}

/// Decode a JPEG byte slice into `output`. `output` must be an interleaved
/// RGB or BGR device buffer sized at least `width * height * 3` bytes.
pub fn decode_interleaved(
    handle: &Handle,
    state: &State,
    jpeg_bytes: &[u8],
    format: PixelFormat,
    output: &mut DeviceBuffer<u8>,
    width: i32,
    stream: Option<&Stream>,
) -> Result<()> {
    assert!(
        format.is_interleaved(),
        "decode_interleaved requires an interleaved PixelFormat (Rgbi / Bgri)"
    );
    let n = nvjpeg()?;
    let cu = n.nvjpeg_decode()?;
    let mut image = nvjpegImage_t::default();
    image.channel[0] = output.as_raw().0 as *mut c_uchar;
    image.pitch[0] = (width as usize) * 3;
    let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
    check(unsafe {
        cu(
            handle.handle,
            state.state,
            jpeg_bytes.as_ptr() as *const c_uchar,
            jpeg_bytes.len(),
            format.raw(),
            &mut image,
            stream_handle,
        )
    })
}

/// Decode a JPEG into up to four planar buffers (one per component). Pass
/// the full `ImageInfo` so the function can compute pitches per plane.
pub fn decode_planar(
    handle: &Handle,
    state: &State,
    jpeg_bytes: &[u8],
    format: PixelFormat,
    channels: &mut [&mut DeviceBuffer<u8>],
    widths: [i32; 4],
    stream: Option<&Stream>,
) -> Result<()> {
    let n = nvjpeg()?;
    let cu = n.nvjpeg_decode()?;
    let mut image = nvjpegImage_t::default();
    for (i, ch) in channels.iter_mut().enumerate().take(4) {
        image.channel[i] = ch.as_raw().0 as *mut c_uchar;
        image.pitch[i] = widths[i] as usize;
    }
    let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
    check(unsafe {
        cu(
            handle.handle,
            state.state,
            jpeg_bytes.as_ptr() as *const c_uchar,
            jpeg_bytes.len(),
            format.raw(),
            &mut image,
            stream_handle,
        )
    })
}

// ---- batched decode ------------------------------------------------------

/// Initialize the batched decoder for a batch of `batch_size` images with
/// `max_cpu_threads` parallel host-side prep threads. Follow with calls to
/// [`decode_batched`].
pub fn decode_batched_initialize(
    handle: &Handle,
    state: &State,
    batch_size: i32,
    max_cpu_threads: i32,
    output_format: PixelFormat,
) -> Result<()> {
    let n = nvjpeg()?;
    let cu = n.nvjpeg_decode_batched_initialize()?;
    check(unsafe {
        cu(
            handle.handle,
            state.state,
            batch_size,
            max_cpu_threads,
            output_format.raw(),
        )
    })
}

/// Decode a batch of JPEGs that has already been initialized via
/// [`decode_batched_initialize`]. `images` must be a fully prepared array of
/// one `nvjpegImage_t` per input, with `channel[0..]` pointing to destination
/// device buffers.
///
/// # Safety
/// `images` must have at least as many entries as the initialized batch
/// size, and each `channel` pointer must be a valid device address.
pub unsafe fn decode_batched(
    handle: &Handle,
    state: &State,
    data: &mut [*const c_uchar],
    lengths: &mut [usize],
    images: *mut nvjpegImage_t,
    stream: Option<&Stream>,
) -> Result<()> {
    assert_eq!(data.len(), lengths.len(), "data/lengths must match in size");
    let n = nvjpeg()?;
    let cu = n.nvjpeg_decode_batched()?;
    let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
    check(cu(
        handle.handle,
        state.state,
        data.as_mut_ptr(),
        lengths.as_mut_ptr(),
        images,
        stream_handle,
    ))
}

// ---- encoder --------------------------------------------------------------

pub use baracuda_nvjpeg_sys::nvjpegChromaSubsampling_t as ChromaSubsampling;

/// Owned encoder state — scratch buffers for a streaming encoder.
pub struct EncoderState {
    raw: nvjpegEncoderState_t,
}

unsafe impl Send for EncoderState {}

impl core::fmt::Debug for EncoderState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncoderState")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl EncoderState {
    pub fn new(handle: &Handle, stream: Option<&Stream>) -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_encoder_state_create()?;
        let mut raw: nvjpegEncoderState_t = core::ptr::null_mut();
        let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
        check(unsafe { cu(handle.handle, &mut raw, stream_handle) })?;
        Ok(Self { raw })
    }

    #[inline]
    pub fn as_raw(&self) -> nvjpegEncoderState_t {
        self.raw
    }
}

impl Drop for EncoderState {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_encoder_state_destroy() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

/// Encoder parameters: quality (1–100), chroma-subsampling, Huffman mode.
pub struct EncoderParams {
    raw: nvjpegEncoderParams_t,
}

unsafe impl Send for EncoderParams {}

impl core::fmt::Debug for EncoderParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncoderParams")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl EncoderParams {
    pub fn new(handle: &Handle, stream: Option<&Stream>) -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_encoder_params_create()?;
        let mut raw: nvjpegEncoderParams_t = core::ptr::null_mut();
        let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
        check(unsafe { cu(handle.handle, &mut raw, stream_handle) })?;
        Ok(Self { raw })
    }

    pub fn set_quality(&self, quality: i32, stream: Option<&Stream>) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_encoder_params_set_quality()?;
        let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
        check(unsafe { cu(self.raw, quality, stream_handle) })
    }

    pub fn set_sampling(
        &self,
        subsampling: ChromaSubsampling,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_encoder_params_set_sampling_factors()?;
        let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
        check(unsafe { cu(self.raw, subsampling, stream_handle) })
    }

    pub fn set_optimized_huffman(&self, optimize: bool, stream: Option<&Stream>) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_encoder_params_set_optimized_huffman()?;
        let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
        check(unsafe { cu(self.raw, optimize as i32, stream_handle) })
    }

    #[inline]
    pub fn as_raw(&self) -> nvjpegEncoderParams_t {
        self.raw
    }
}

impl Drop for EncoderParams {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_encoder_params_destroy() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

/// Encode an RGB/BGR/YUV image. After this returns, call
/// [`retrieve_bitstream`] to pull the JPEG bytes off the device.
///
/// # Safety
/// `channels` must be device pointers of the correct size; `input_format`
/// must match their memory layout.
#[allow(clippy::too_many_arguments)]
pub unsafe fn encode_image(
    handle: &Handle,
    state: &EncoderState,
    params: &EncoderParams,
    channels: [&DeviceBuffer<u8>; 4],
    pitches: [usize; 4],
    input_format: PixelFormat,
    width: i32,
    height: i32,
    stream: Option<&Stream>,
) -> Result<()> {
    let n = nvjpeg()?;
    let cu = n.nvjpeg_encode_image()?;
    let mut image = nvjpegImage_t::default();
    for i in 0..4 {
        image.channel[i] = channels[i].as_raw().0 as *mut c_uchar;
        image.pitch[i] = pitches[i];
    }
    let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
    check(cu(
        handle.handle,
        state.raw,
        params.raw,
        &image,
        input_format.raw() as c_int,
        width,
        height,
        stream_handle,
    ))
}

/// Pull the encoded JPEG byte stream out of the encoder state. First call
/// returns the required length; then re-call with a properly-sized buffer.
pub fn retrieve_bitstream(
    handle: &Handle,
    state: &EncoderState,
    out: Option<&mut [u8]>,
    stream: Option<&Stream>,
) -> Result<usize> {
    let n = nvjpeg()?;
    let cu = n.nvjpeg_encode_retrieve_bitstream()?;
    let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
    let mut length: usize = out.as_ref().map(|b| b.len()).unwrap_or(0);
    let ptr = out
        .map(|b| b.as_mut_ptr())
        .unwrap_or(core::ptr::null_mut());
    check(unsafe { cu(handle.handle, state.raw, ptr, &mut length, stream_handle) })?;
    Ok(length)
}

// ---- hybrid decoder (Host → Transfer → Device phases) -------------------

pub use baracuda_nvjpeg_sys::nvjpegBackend_t as Backend;

/// A JPEG decoder configured for a specific backend. Pairs with a
/// [`State`] + [`JpegStream`] + [`DecodeParams`] to drive the three-phase
/// hybrid pipeline.
pub struct Decoder {
    raw: nvjpegJpegDecoder_t,
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
    pub fn new(handle: &Handle, backend: Backend) -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_decoder_create()?;
        let mut raw: nvjpegJpegDecoder_t = core::ptr::null_mut();
        check(unsafe { cu(handle.handle, backend, &mut raw) })?;
        Ok(Self { raw })
    }

    pub fn decode_host(
        &self,
        handle: &Handle,
        state: &State,
        params: &DecodeParams,
        jpeg_stream: &JpegStream,
    ) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_decode_jpeg_host()?;
        check(unsafe { cu(handle.handle, self.raw, state.state, params.raw, jpeg_stream.raw) })
    }

    pub fn decode_transfer_to_device(
        &self,
        handle: &Handle,
        state: &State,
        jpeg_stream: &JpegStream,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_decode_jpeg_transfer_to_device()?;
        let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
        check(unsafe {
            cu(handle.handle, self.raw, state.state, jpeg_stream.raw, stream_handle)
        })
    }

    /// Final phase: emit the decoded pixels into `dest`.
    ///
    /// # Safety
    /// `dest.channel[..]` must point to device memory of the right size.
    pub unsafe fn decode_device(
        &self,
        handle: &Handle,
        state: &State,
        dest: &mut nvjpegImage_t,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_decode_jpeg_device()?;
        let stream_handle = stream.map_or(core::ptr::null_mut(), |s| s.as_raw() as _);
        check(cu(handle.handle, self.raw, state.state, dest, stream_handle))
    }

    #[inline]
    pub fn as_raw(&self) -> nvjpegJpegDecoder_t {
        self.raw
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_decoder_destroy() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

// ---- JPEG stream (parsed stream metadata) -------------------------------

pub struct JpegStream {
    raw: nvjpegJpegStream_t,
}

unsafe impl Send for JpegStream {}

impl core::fmt::Debug for JpegStream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("JpegStream")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl JpegStream {
    pub fn new(handle: &Handle) -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_jpeg_stream_create()?;
        let mut raw: nvjpegJpegStream_t = core::ptr::null_mut();
        check(unsafe { cu(handle.handle, &mut raw) })?;
        Ok(Self { raw })
    }

    pub fn parse(
        &self,
        handle: &Handle,
        jpeg_bytes: &[u8],
        save_metadata: bool,
        save_stream: bool,
    ) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_jpeg_stream_parse()?;
        check(unsafe {
            cu(
                handle.handle,
                jpeg_bytes.as_ptr() as *const c_uchar,
                jpeg_bytes.len(),
                save_metadata as c_int,
                save_stream as c_int,
                self.raw,
            )
        })
    }

    pub fn frame_dimensions(&self) -> Result<(u32, u32)> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_jpeg_stream_get_frame_dimensions()?;
        let (mut w, mut h) = (0u32, 0u32);
        check(unsafe { cu(self.raw, &mut w, &mut h) })?;
        Ok((w, h))
    }

    pub fn num_components(&self) -> Result<u32> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_jpeg_stream_get_components_num()?;
        let mut n_comp = 0u32;
        check(unsafe { cu(self.raw, &mut n_comp) })?;
        Ok(n_comp)
    }

    pub fn chroma_subsampling(&self) -> Result<ChromaSubsampling> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_jpeg_stream_get_chroma_subsampling()?;
        let mut s = ChromaSubsampling::Css444;
        check(unsafe { cu(self.raw, &mut s) })?;
        Ok(s)
    }

    #[inline]
    pub fn as_raw(&self) -> nvjpegJpegStream_t {
        self.raw
    }
}

impl Drop for JpegStream {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_jpeg_stream_destroy() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

// ---- Pinned / device buffer pools (attachable to a State) --------------

pub struct BufferPinned {
    raw: nvjpegBufferPinned_t,
}

unsafe impl Send for BufferPinned {}

impl core::fmt::Debug for BufferPinned {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BufferPinned")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl BufferPinned {
    /// Create a pinned-memory buffer that the decoder can reuse across
    /// frames. The default allocator is used (pass `None`).
    pub fn new(handle: &Handle) -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_buffer_pinned_create()?;
        let mut raw: nvjpegBufferPinned_t = core::ptr::null_mut();
        check(unsafe { cu(handle.handle, core::ptr::null_mut(), &mut raw) })?;
        Ok(Self { raw })
    }

    pub fn attach_to(&self, state: &State) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_state_attach_pinned_buffer()?;
        check(unsafe { cu(state.state, self.raw) })
    }

    #[inline]
    pub fn as_raw(&self) -> nvjpegBufferPinned_t {
        self.raw
    }
}

impl Drop for BufferPinned {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_buffer_pinned_destroy() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

pub struct BufferDevice {
    raw: nvjpegBufferDevice_t,
}

unsafe impl Send for BufferDevice {}

impl core::fmt::Debug for BufferDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BufferDevice")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl BufferDevice {
    pub fn new(handle: &Handle) -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_buffer_device_create()?;
        let mut raw: nvjpegBufferDevice_t = core::ptr::null_mut();
        check(unsafe { cu(handle.handle, core::ptr::null_mut(), &mut raw) })?;
        Ok(Self { raw })
    }

    pub fn attach_to(&self, state: &State) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_state_attach_device_buffer()?;
        check(unsafe { cu(state.state, self.raw) })
    }

    #[inline]
    pub fn as_raw(&self) -> nvjpegBufferDevice_t {
        self.raw
    }
}

impl Drop for BufferDevice {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_buffer_device_destroy() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}

// ---- Decode params (output format / ROI / CMYK) -------------------------

pub struct DecodeParams {
    raw: nvjpegDecodeParams_t,
}

unsafe impl Send for DecodeParams {}

impl core::fmt::Debug for DecodeParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodeParams")
            .field("raw", &self.raw)
            .finish_non_exhaustive()
    }
}

impl DecodeParams {
    pub fn new(handle: &Handle) -> Result<Self> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_decode_params_create()?;
        let mut raw: nvjpegDecodeParams_t = core::ptr::null_mut();
        check(unsafe { cu(handle.handle, &mut raw) })?;
        Ok(Self { raw })
    }

    pub fn set_output_format(&self, fmt: PixelFormat) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_decode_params_set_output_format()?;
        check(unsafe { cu(self.raw, fmt.raw()) })
    }

    pub fn set_roi(&self, offset_x: i32, offset_y: i32, width: i32, height: i32) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_decode_params_set_roi()?;
        check(unsafe { cu(self.raw, offset_x, offset_y, width, height) })
    }

    pub fn set_allow_cmyk(&self, allow: bool) -> Result<()> {
        let n = nvjpeg()?;
        let cu = n.nvjpeg_decode_params_set_allow_cmyk()?;
        check(unsafe { cu(self.raw, allow as c_int) })
    }

    #[inline]
    pub fn as_raw(&self) -> nvjpegDecodeParams_t {
        self.raw
    }
}

impl Drop for DecodeParams {
    fn drop(&mut self) {
        if let Ok(n) = nvjpeg() {
            if let Ok(cu) = n.nvjpeg_decode_params_destroy() {
                let _ = unsafe { cu(self.raw) };
            }
        }
    }
}
