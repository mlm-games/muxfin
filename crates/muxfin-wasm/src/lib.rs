//! WebAssembly bindings for the `muxfin` MP4 muxer (§13).
//!
//! This crate re-exports the stable [`muxfin_ffi`] C ABI — the same opaque
//! handles and integer status codes — so `wasm32-unknown-unknown` builds can
//! call the muxer from JavaScript with plain `extern "C"` imports (no
//! `wasm-bindgen` dependency, keeping the workspace minimal-dependency).
//! It additionally provides explicit linear-memory helpers (`alloc`/`free`)
//! because wasm guests cannot rely on libc `malloc`.
//!
//! # JavaScript sketch
//!
//! ```js
//! // Memory helpers manage guest buffers; sample bytes are copied in.
//! const ptr = wasm.muxfin_wasm_alloc(len);
//! new Uint8Array(wasm.memory.buffer).set(sampleBytes, ptr);
//! wasm.muxfin_write_video(handle, pts, dts, ptr, len, isKeyframe);
//! wasm.muxfin_wasm_free(ptr, len);
//!
//! // After muxfin_finish on a memory build:
//! const outPtr = wasm.muxfin_take_output(handle, outLenPtr);
//! const mp4 = new Uint8Array(wasm.memory.buffer).slice(
//!   outPtr, outPtr + new Uint32Array(wasm.memory.buffer)[outLenPtr / 4]);
//! wasm.muxfin_free_bytes(outPtr, mp4.length);
//! ```
//!
//! All pointer contracts are identical to [`muxfin_ffi`]; see that crate's
//! safety documentation.

pub use muxfin_ffi::{
    MuxfinBuilderHandle, MuxfinHandle, MuxfinStatus, MuxfinStatusCode, muxfin_build_file,
    muxfin_build_memory, muxfin_builder_audio, muxfin_builder_fast_start,
    muxfin_builder_flac_streaminfo, muxfin_builder_free, muxfin_builder_new,
    muxfin_builder_subtitle, muxfin_builder_video, muxfin_finish, muxfin_free, muxfin_free_bytes,
    muxfin_last_error, muxfin_take_output, muxfin_version, muxfin_write_audio,
    muxfin_write_subtitle, muxfin_write_video,
};

/// Allocate `len` uninitialized bytes in guest linear memory.
///
/// Returns null for `len == 0` or on allocation failure; free with
/// [`muxfin_wasm_free`] using the same `len`.
///
/// # Safety
///
/// Always safe to call from the host.
#[unsafe(no_mangle)]
pub extern "C" fn muxfin_wasm_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    // Zeroed (not uninitialized): the host overwrites the buffer with
    // sample bytes before any read.
    let mut vec = vec![0u8; len];
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

/// Free a buffer from [`muxfin_wasm_alloc`].
///
/// # Safety
///
/// `ptr` must come from [`muxfin_wasm_alloc`] with the same `len`, freed
/// exactly once. Null with length 0 is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_wasm_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// One-call helper: mux a single keyframe + optional ADTS audio frame into
/// an MP4 held in guest memory.
///
/// This exists for smoke-testing the wasm build from hosts that cannot
/// easily drive the multi-call handle API. Production use should drive the
/// re-exported handle API directly. Returns null on failure (see
/// [`muxfin_last_error`]); otherwise a `muxfin_take_output`-style buffer
/// the caller frees with [`muxfin_free_bytes`](muxfin_ffi::muxfin_free_bytes)
/// after reading `*out_len`.
///
/// # Safety
///
/// `keyframe`/`audio` must point to at least `keyframe_len`/`audio_len`
/// readable bytes (either may be null with length 0 to omit); `out_len`
/// must be writable and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn muxfin_wasm_mux_single_frame(
    width: u32,
    height: u32,
    keyframe: *const u8,
    keyframe_len: usize,
    audio: *const u8,
    audio_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let builder = muxfin_ffi::muxfin_builder_new();
        if builder.is_null() {
            return std::ptr::null_mut();
        }
        if muxfin_builder_video(builder, 0, width, height, 30.0) != MuxfinStatus::Ok {
            muxfin_builder_free(builder);
            return std::ptr::null_mut();
        }
        let with_audio = !audio.is_null() && audio_len > 0;
        if with_audio && muxfin_builder_audio(builder, 0, 48_000, 2) != MuxfinStatus::Ok {
            muxfin_builder_free(builder);
            return std::ptr::null_mut();
        }
        let muxer = muxfin_build_memory(builder);
        if muxer.is_null() {
            return std::ptr::null_mut();
        }
        if !keyframe.is_null() && keyframe_len > 0 {
            let status = muxfin_write_video(muxer, 0.0, 0.0, keyframe, keyframe_len, true);
            if status != MuxfinStatus::Ok {
                muxfin_free(muxer);
                return std::ptr::null_mut();
            }
        }
        if with_audio {
            let status = muxfin_write_audio(muxer, 0.0, audio, audio_len);
            if status != MuxfinStatus::Ok {
                muxfin_free(muxer);
                return std::ptr::null_mut();
            }
        }
        if muxfin_finish(muxer) != MuxfinStatus::Ok {
            muxfin_free(muxer);
            return std::ptr::null_mut();
        }
        let out = muxfin_take_output(muxer, out_len);
        muxfin_free(muxer);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEYFRAME: &[u8] = &[
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11, 0x00,
        0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00,
        0x11,
    ];
    const ADTS: &[u8] = &[0xff, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc, 0xaa, 0xbb];

    #[test]
    fn alloc_free_roundtrip() {
        unsafe {
            let ptr = muxfin_wasm_alloc(16);
            assert!(!ptr.is_null());
            std::slice::from_raw_parts_mut(ptr, 16).fill(0xAB);
            muxfin_wasm_free(ptr, 16);
            assert!(muxfin_wasm_alloc(0).is_null());
            muxfin_wasm_free(std::ptr::null_mut(), 0);
        }
    }

    #[test]
    fn single_frame_mux() {
        unsafe {
            let mut len = 0usize;
            let ptr = muxfin_wasm_mux_single_frame(
                640,
                480,
                KEYFRAME.as_ptr(),
                KEYFRAME.len(),
                ADTS.as_ptr(),
                ADTS.len(),
                &mut len,
            );
            assert!(!ptr.is_null());
            assert!(len > 100);
            let bytes = std::slice::from_raw_parts(ptr, len);
            assert_eq!(&bytes[4..8], b"ftyp");
            muxfin_free_bytes(ptr, len);
        }
    }
}
