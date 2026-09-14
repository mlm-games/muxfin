//! Stable C ABI for the `muxfin` MP4 muxer (§13).
//!
//! The core crate stays dependency-minimal and `#![forbid(unsafe_code)]`;
//! all `unsafe` required by a C boundary lives here, behind opaque handles
//! and integer status codes. Every function documents its safety contract.
//!
//! # Ownership
//!
//! - `muxfin_builder_new` / `muxfin_builder_free`: builder handle.
//! - `muxfin_build_file` / `muxfin_build_memory`: consume the builder,
//!   produce a muxer handle.
//! - `muxfin_free`: destroys a muxer handle (finish first if you want output).
//! - `muxfin_take_output` / `muxfin_free_bytes`: memory-build output.
//! - `muxfin_last_error`: thread-local detail for the last failing call.
//!
//! All pointer arguments must be non-null and point to at least `len`
//! readable bytes; all handles must come from this library.

use std::cell::RefCell;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::os::raw::{c_char, c_double, c_uchar};
use std::rc::Rc;

use muxfin::api::{AacProfile, AudioCodec, Muxer, MuxerBuilder, SubtitleCodec, VideoCodec};

/// Status code returned by every fallible FFI entry point.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuxfinStatus {
    Ok = 0,
    NullArgument = 1,
    InvalidArgument = 2,
    BuildFailed = 3,
    WriteFailed = 4,
    FinishFailed = 5,
}

std::thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn stash_error(message: String) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(message).ok();
    });
}

/// Human-readable detail for the calling thread's last failing call.
///
/// # Safety
///
/// Always safe to call; returns a null-terminated string that stays valid
/// until the next failing call on this thread (do not free it).
#[unsafe(no_mangle)]
pub extern "C" fn muxfin_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(s) => s.as_ptr(),
        None => c"".as_ptr(),
    })
}

/// Library version string (`CARGO_PKG_VERSION`), null-terminated.
///
/// # Safety
///
/// Always safe to call; the pointer is `'static` (do not free it).
#[unsafe(no_mangle)]
pub extern "C" fn muxfin_version() -> *const c_char {
    c"0.2.2".as_ptr()
}

/// Opaque builder handle (see `muxfin_builder_new`).
pub struct MuxfinBuilderHandle {
    video: Option<(VideoCodec, u32, u32, f64)>,
    audio: Option<(AudioCodec, u32, u16)>,
    subtitle: Option<(SubtitleCodec, Option<String>)>,
    flac_streaminfo: Option<Vec<u8>>,
    fast_start: bool,
}

/// In-memory sink shared with the finished muxer output.
#[derive(Clone, Default)]
pub struct SharedMem(Rc<RefCell<Vec<u8>>>);

impl Write for SharedMem {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Opaque muxer handle (see `muxfin_build_file` / `muxfin_build_memory`).
pub enum MuxfinHandle {
    File(Muxer<File>),
    Memory(Muxer<SharedMem>, SharedMem),
}

/// Create a new builder handle.
///
/// # Safety
///
/// Always safe to call. The caller owns the returned pointer and must pass
/// it to `muxfin_builder_free` (or hand it to a `muxfin_build_*` call,
/// which consumes it) exactly once. Returns null on allocation failure.
#[unsafe(no_mangle)]
pub extern "C" fn muxfin_builder_new() -> *mut MuxfinBuilderHandle {
    let handle = Box::new(MuxfinBuilderHandle {
        video: None,
        audio: None,
        subtitle: None,
        flac_streaminfo: None,
        fast_start: true,
    });
    Box::into_raw(handle)
}

/// Destroy a builder handle.
///
/// # Safety
///
/// `builder` must be a live pointer from `muxfin_builder_new` that has not
/// been consumed by `muxfin_build_*` or freed before. Null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_builder_free(builder: *mut MuxfinBuilderHandle) {
    if !builder.is_null() {
        unsafe {
            drop(Box::from_raw(builder));
        }
    }
}

fn builder_ref(
    builder: *mut MuxfinBuilderHandle,
) -> Result<&'static mut MuxfinBuilderHandle, MuxfinStatus> {
    if builder.is_null() {
        stash_error("null builder handle".to_string());
        return Err(MuxfinStatus::NullArgument);
    }
    Ok(unsafe { &mut *builder })
}

const VIDEO_H264: u32 = 0;
const VIDEO_H265: u32 = 1;
const VIDEO_AV1: u32 = 2;
const VIDEO_VP9: u32 = 3;

fn decode_video(codec: u32) -> Result<VideoCodec, MuxfinStatus> {
    match codec {
        VIDEO_H264 => Ok(VideoCodec::H264),
        VIDEO_H265 => Ok(VideoCodec::H265),
        VIDEO_AV1 => Ok(VideoCodec::Av1),
        VIDEO_VP9 => Ok(VideoCodec::Vp9),
        _ => {
            stash_error(format!("unknown video codec code {codec}"));
            Err(MuxfinStatus::InvalidArgument)
        }
    }
}

/// Configure the video track.
///
/// `codec`: 0 = H.264, 1 = H.265, 2 = AV1, 3 = VP9.
///
/// # Safety
///
/// `builder` must be a live builder pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_builder_video(
    builder: *mut MuxfinBuilderHandle,
    codec: u32,
    width: u32,
    height: u32,
    framerate: c_double,
) -> MuxfinStatus {
    let b = match builder_ref(builder) {
        Ok(b) => b,
        Err(s) => return s,
    };
    let codec = match decode_video(codec) {
        Ok(c) => c,
        Err(s) => return s,
    };
    if width == 0 || height == 0 || !framerate.is_finite() || framerate <= 0.0 {
        stash_error("video dimensions must be non-zero and framerate finite positive".to_string());
        return MuxfinStatus::InvalidArgument;
    }
    b.video = Some((codec, width, height, framerate));
    MuxfinStatus::Ok
}

fn decode_audio(codec: u32) -> Result<AudioCodec, MuxfinStatus> {
    match codec {
        0 => Ok(AudioCodec::Aac(AacProfile::Lc)),
        1 => Ok(AudioCodec::Aac(AacProfile::Main)),
        2 => Ok(AudioCodec::Aac(AacProfile::Ssr)),
        3 => Ok(AudioCodec::Aac(AacProfile::Ltp)),
        4 => Ok(AudioCodec::Aac(AacProfile::He)),
        5 => Ok(AudioCodec::Aac(AacProfile::Hev2)),
        10 => Ok(AudioCodec::Opus),
        11 => Ok(AudioCodec::Flac),
        _ => {
            stash_error(format!("unknown audio codec code {codec}"));
            Err(MuxfinStatus::InvalidArgument)
        }
    }
}

/// Configure the audio track.
///
/// `codec`: 0-5 = AAC (LC/Main/SSR/LTP/HE/HEv2), 10 = Opus, 11 = FLAC
/// (FLAC additionally requires `muxfin_builder_flac_streaminfo`).
///
/// # Safety
///
/// `builder` must be a live builder pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_builder_audio(
    builder: *mut MuxfinBuilderHandle,
    codec: u32,
    sample_rate: u32,
    channels: u16,
) -> MuxfinStatus {
    let b = match builder_ref(builder) {
        Ok(b) => b,
        Err(s) => return s,
    };
    let codec = match decode_audio(codec) {
        Ok(c) => c,
        Err(s) => return s,
    };
    if sample_rate == 0 || channels == 0 {
        stash_error("audio sample rate and channels must be non-zero".to_string());
        return MuxfinStatus::InvalidArgument;
    }
    b.audio = Some((codec, sample_rate, channels));
    MuxfinStatus::Ok
}

/// Supply the 34-byte FLAC STREAMINFO block (required for FLAC audio).
///
/// # Safety
///
/// `builder` must be live; `data` must point to at least `len` bytes.
/// `len` must be exactly 34.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_builder_flac_streaminfo(
    builder: *mut MuxfinBuilderHandle,
    data: *const u8,
    len: usize,
) -> MuxfinStatus {
    let b = match builder_ref(builder) {
        Ok(b) => b,
        Err(s) => return s,
    };
    if data.is_null() || len != 34 {
        stash_error("FLAC STREAMINFO must be exactly 34 bytes".to_string());
        return MuxfinStatus::InvalidArgument;
    }
    b.flac_streaminfo = Some(unsafe { std::slice::from_raw_parts(data, len) }.to_vec());
    MuxfinStatus::Ok
}

/// Configure the subtitle track.
///
/// `codec`: 0 = mov_text (`tx3g`), 1 = WebVTT (`wvtt`), 2 = SSA
/// (`S_TEXT/SSA`, Matroska-only), 3 = ASS (`S_TEXT/ASS`, Matroska-only). `lang` is an
/// optional ISO-639-2/T code (`lang_len` bytes, not null-terminated; may be
/// null with length 0 for none).
///
/// # Safety
///
/// `builder` must be live; `lang` must point to at least `lang_len` bytes
/// unless null with length 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_builder_subtitle(
    builder: *mut MuxfinBuilderHandle,
    codec: u32,
    lang: *const c_char,
    lang_len: usize,
) -> MuxfinStatus {
    let b = match builder_ref(builder) {
        Ok(b) => b,
        Err(s) => return s,
    };
    let codec = match codec {
        0 => SubtitleCodec::MovText,
        1 => SubtitleCodec::WebVtt,
        2 => SubtitleCodec::Ssa,
        3 => SubtitleCodec::Ass,
        _ => {
            stash_error(format!("unknown subtitle codec code {codec}"));
            return MuxfinStatus::InvalidArgument;
        }
    };
    let language = if lang.is_null() || lang_len == 0 {
        None
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(lang as *const u8, lang_len) };
        match std::str::from_utf8(bytes) {
            Ok(s) => Some(s.to_string()),
            Err(_) => {
                stash_error("subtitle language must be UTF-8".to_string());
                return MuxfinStatus::InvalidArgument;
            }
        }
    };
    b.subtitle = Some((codec, language));
    MuxfinStatus::Ok
}

/// Toggle fast-start (`moov` before `mdat`; default on).
///
/// # Safety
///
/// `builder` must be a live builder pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_builder_fast_start(
    builder: *mut MuxfinBuilderHandle,
    enabled: bool,
) -> MuxfinStatus {
    let b = match builder_ref(builder) {
        Ok(b) => b,
        Err(s) => return s,
    };
    b.fast_start = enabled;
    MuxfinStatus::Ok
}

fn apply_builder<W>(builder: MuxerBuilder<W>, handle: &MuxfinBuilderHandle) -> MuxerBuilder<W> {
    let mut b = builder;
    if let Some((codec, width, height, framerate)) = handle.video {
        b = b.video(codec, width, height, framerate);
    }
    if let Some((codec, sample_rate, channels)) = handle.audio {
        b = b.audio(codec, sample_rate, channels);
    }
    if let Some((codec, ref language)) = handle.subtitle {
        b = b.subtitle(codec, language.clone());
    }
    if let Some(ref info) = handle.flac_streaminfo {
        b = b.with_flac_streaminfo(info.clone());
    }
    b.with_fast_start(handle.fast_start)
}

fn path_str(path: *const c_char, path_len: usize) -> Result<String, MuxfinStatus> {
    if path.is_null() {
        stash_error("null output path".to_string());
        return Err(MuxfinStatus::NullArgument);
    }
    let bytes = unsafe { std::slice::from_raw_parts(path as *const u8, path_len) };
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => {
            stash_error("output path must be UTF-8".to_string());
            Err(MuxfinStatus::InvalidArgument)
        }
    }
}

/// Build a muxer writing to the file at `path` (`path_len` bytes, UTF-8,
/// not null-terminated). Consumes the builder; on success the caller owns
/// the returned muxer pointer (see `muxfin_free`).
///
/// # Safety
///
/// `builder` must be a live, unconsumed builder pointer; `path` must point
/// to at least `path_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_build_file(
    builder: *mut MuxfinBuilderHandle,
    path: *const c_char,
    path_len: usize,
) -> *mut MuxfinHandle {
    if builder.is_null() {
        stash_error("null builder handle".to_string());
        return std::ptr::null_mut();
    }
    let path = match path_str(path, path_len) {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };
    let handle = unsafe { *Box::from_raw(builder) };
    let file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            stash_error(format!("cannot create output file: {e}"));
            return std::ptr::null_mut();
        }
    };
    let b = apply_builder(MuxerBuilder::new(file), &handle);
    match b.build() {
        Ok(muxer) => Box::into_raw(Box::new(MuxfinHandle::File(muxer))),
        Err(e) => {
            stash_error(format!("build failed: {e:?}"));
            std::ptr::null_mut()
        }
    }
}

/// Build a muxer writing to memory. Consumes the builder; retrieve the
/// bytes after `muxfin_finish` with `muxfin_take_output`.
///
/// # Safety
///
/// `builder` must be a live, unconsumed builder pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_build_memory(
    builder: *mut MuxfinBuilderHandle,
) -> *mut MuxfinHandle {
    if builder.is_null() {
        stash_error("null builder handle".to_string());
        return std::ptr::null_mut();
    }
    let handle = unsafe { *Box::from_raw(builder) };
    let sink = SharedMem::default();
    let b = apply_builder(MuxerBuilder::new(sink.clone()), &handle);
    match b.build() {
        Ok(muxer) => Box::into_raw(Box::new(MuxfinHandle::Memory(muxer, sink))),
        Err(e) => {
            stash_error(format!("build failed: {e:?}"));
            std::ptr::null_mut()
        }
    }
}

/// Destroy a muxer handle.
///
/// # Safety
///
/// `muxer` must be a live pointer from `muxfin_build_*` that has not been
/// freed before. Null is a no-op. Destroying without `muxfin_finish`
/// discards the output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_free(muxer: *mut MuxfinHandle) {
    if !muxer.is_null() {
        unsafe {
            drop(Box::from_raw(muxer));
        }
    }
}

fn handle_ref(muxer: *mut MuxfinHandle) -> Result<&'static mut MuxfinHandle, MuxfinStatus> {
    if muxer.is_null() {
        stash_error("null muxer handle".to_string());
        return Err(MuxfinStatus::NullArgument);
    }
    Ok(unsafe { &mut *muxer })
}

fn input_bytes<'a>(data: *const u8, len: usize) -> Result<&'a [u8], MuxfinStatus> {
    if data.is_null() {
        stash_error("null sample data".to_string());
        return Err(MuxfinStatus::NullArgument);
    }
    if len == 0 {
        stash_error("empty sample".to_string());
        return Err(MuxfinStatus::InvalidArgument);
    }
    Ok(unsafe { std::slice::from_raw_parts(data, len) })
}

/// Write one video frame.
///
/// Annex B for H.264/H.265, OBU stream for AV1, compressed frames for VP9.
/// `dts == pts` unless B-frames are used (then pass differing values;
/// frames must arrive in decode order with strictly increasing `dts`).
///
/// # Safety
///
/// `muxer` must be live; `data` must point to at least `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_write_video(
    muxer: *mut MuxfinHandle,
    pts: c_double,
    dts: c_double,
    data: *const u8,
    len: usize,
    is_keyframe: bool,
) -> MuxfinStatus {
    let h = match handle_ref(muxer) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let data = match input_bytes(data, len) {
        Ok(d) => d,
        Err(s) => return s,
    };
    let result = match h {
        MuxfinHandle::File(m) => m.write_video_with_dts(pts, dts, data, is_keyframe),
        MuxfinHandle::Memory(m, _) => m.write_video_with_dts(pts, dts, data, is_keyframe),
    };
    match result {
        Ok(()) => MuxfinStatus::Ok,
        Err(e) => {
            stash_error(format!("write video failed: {e:?}"));
            MuxfinStatus::WriteFailed
        }
    }
}

/// Write one audio frame (AAC ADTS frame, raw Opus packet, or FLAC frame).
///
/// # Safety
///
/// `muxer` must be live; `data` must point to at least `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_write_audio(
    muxer: *mut MuxfinHandle,
    pts: c_double,
    data: *const u8,
    len: usize,
) -> MuxfinStatus {
    let h = match handle_ref(muxer) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let data = match input_bytes(data, len) {
        Ok(d) => d,
        Err(s) => return s,
    };
    let result = match h {
        MuxfinHandle::File(m) => m.write_audio(pts, data),
        MuxfinHandle::Memory(m, _) => m.write_audio(pts, data),
    };
    match result {
        Ok(()) => MuxfinStatus::Ok,
        Err(e) => {
            stash_error(format!("write audio failed: {e:?}"));
            MuxfinStatus::WriteFailed
        }
    }
}

/// Write one subtitle cue.
///
/// # Safety
///
/// `muxer` must be live; `text` must point to at least `len` UTF-8 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_write_subtitle(
    muxer: *mut MuxfinHandle,
    pts: c_double,
    duration: c_double,
    text: *const c_char,
    len: usize,
) -> MuxfinStatus {
    let h = match handle_ref(muxer) {
        Ok(h) => h,
        Err(s) => return s,
    };
    if text.is_null() || len == 0 {
        stash_error("null or empty subtitle text".to_string());
        return MuxfinStatus::InvalidArgument;
    }
    let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, len) };
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            stash_error("subtitle text must be UTF-8".to_string());
            return MuxfinStatus::InvalidArgument;
        }
    };
    let result = match h {
        MuxfinHandle::File(m) => m.write_subtitle(pts, duration, text),
        MuxfinHandle::Memory(m, _) => m.write_subtitle(pts, duration, text),
    };
    match result {
        Ok(()) => MuxfinStatus::Ok,
        Err(e) => {
            stash_error(format!("write subtitle failed: {e:?}"));
            MuxfinStatus::WriteFailed
        }
    }
}

/// Finalize the container (writes headers/trailers).
///
/// # Safety
///
/// `muxer` must be a live, unfinished muxer pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_finish(muxer: *mut MuxfinHandle) -> MuxfinStatus {
    let h = match handle_ref(muxer) {
        Ok(h) => h,
        Err(s) => return s,
    };
    let result = match h {
        MuxfinHandle::File(m) => m.finish_in_place().map(|_| ()),
        MuxfinHandle::Memory(m, _) => m.finish_in_place().map(|_| ()),
    };
    match result {
        Ok(()) => MuxfinStatus::Ok,
        Err(e) => {
            stash_error(format!("finish failed: {e:?}"));
            MuxfinStatus::FinishFailed
        }
    }
}

/// Take the output bytes of a finished memory build.
///
/// On success sets `*out_len` and returns a buffer the caller owns (free
/// with `muxfin_free_bytes`). Only valid for `muxfin_build_memory` handles
/// after `muxfin_finish`.
///
/// # Safety
///
/// `muxer` must be live; `out_len` must be a writable non-null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_take_output(
    muxer: *mut MuxfinHandle,
    out_len: *mut usize,
) -> *mut c_uchar {
    if out_len.is_null() {
        stash_error("null out_len pointer".to_string());
        return std::ptr::null_mut();
    }
    let h = match handle_ref(muxer) {
        Ok(h) => h,
        Err(_) => return std::ptr::null_mut(),
    };
    let bytes = match h {
        MuxfinHandle::Memory(_, sink) => sink.0.borrow().clone(),
        MuxfinHandle::File(_) => {
            stash_error("take_output requires a memory build".to_string());
            return std::ptr::null_mut();
        }
    };
    let mut boxed = bytes.into_boxed_slice();
    unsafe {
        *out_len = boxed.len();
    }
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

/// Free a buffer returned by `muxfin_take_output`.
///
/// # Safety
///
/// `ptr` must come from `muxfin_take_output` with the same `len`, freed
/// exactly once. Null with length 0 is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_free_bytes(ptr: *mut c_uchar, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
    }
}

/// C-visible alias of [`MuxfinStatus`]; guarantees the discriminant width.
pub type MuxfinStatusCode = i32;

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    const SPS: &[u8] = &[
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11,
    ];
    const PPS: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80];
    const IDR: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, 0x11];

    fn keyframe() -> Vec<u8> {
        [SPS, PPS, IDR].concat()
    }

    #[test]
    fn memory_roundtrip() {
        unsafe {
            let b = muxfin_builder_new();
            assert!(!b.is_null());
            assert_eq!(muxfin_builder_video(b, 0, 640, 480, 30.0), MuxfinStatus::Ok);
            assert_eq!(muxfin_builder_audio(b, 0, 48_000, 2), MuxfinStatus::Ok);
            let m = muxfin_build_memory(b);
            assert!(!m.is_null());

            let kf = keyframe();
            assert_eq!(
                muxfin_write_video(m, 0.0, 0.0, kf.as_ptr(), kf.len(), true),
                MuxfinStatus::Ok
            );
            let adts = [0xffu8, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc, 0xaa, 0xbb];
            assert_eq!(
                muxfin_write_audio(m, 0.0, adts.as_ptr(), adts.len()),
                MuxfinStatus::Ok
            );
            assert_eq!(muxfin_finish(m), MuxfinStatus::Ok);

            let mut len = 0usize;
            let ptr = muxfin_take_output(m, &mut len);
            assert!(!ptr.is_null());
            assert!(len > 100);
            let bytes = std::slice::from_raw_parts(ptr, len);
            assert_eq!(&bytes[4..8], b"ftyp");
            muxfin_free_bytes(ptr, len);
            muxfin_free(m);
        }
    }

    #[test]
    fn null_handles_rejected() {
        unsafe {
            assert_eq!(
                muxfin_write_video(std::ptr::null_mut(), 0.0, 0.0, [1u8].as_ptr(), 1, true),
                MuxfinStatus::NullArgument
            );
            assert_eq!(
                muxfin_finish(std::ptr::null_mut()),
                MuxfinStatus::NullArgument
            );
            assert!(!muxfin_last_error().is_null());
            let msg = CStr::from_ptr(muxfin_last_error());
            assert!(!msg.to_bytes().is_empty());
        }
    }

    #[test]
    fn invalid_codes_rejected() {
        unsafe {
            let b = muxfin_builder_new();
            assert_eq!(
                muxfin_builder_video(b, 99, 640, 480, 30.0),
                MuxfinStatus::InvalidArgument
            );
            assert_eq!(
                muxfin_builder_audio(b, 99, 48_000, 2),
                MuxfinStatus::InvalidArgument
            );
            muxfin_builder_free(b);
            muxfin_builder_free(std::ptr::null_mut());
        }
    }
}
