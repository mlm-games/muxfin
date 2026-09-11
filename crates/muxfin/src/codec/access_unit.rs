//! Raw Annex B access-unit splitting.
//!
//! Encoders and `.h264`/`.h265` files emit a continuous Annex B byte stream
//! containing many frames, but [`Muxer`](crate::api::Muxer) takes exactly one
//! access unit (frame) per `write_video` call. This module splits a raw
//! stream into its constituent access units so each slice can be passed
//! through unchanged:
//!
//! ```no_run
//! use muxfin::api::{MuxerBuilder, VideoCodec};
//! use muxfin::codec::access_unit::split_h264_access_units;
//! use std::fs::File;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let raw = std::fs::read("capture.h264")?;
//! let file = File::create("out.mp4")?;
//! let mut muxer = MuxerBuilder::new(file)
//!     .video(VideoCodec::H264, 1920, 1080, 30.0)
//!     .build()?;
//! let mut pts = 0.0;
//! for au in split_h264_access_units(&raw) {
//!     muxer.write_video(pts, au.data, au.is_keyframe)?;
//!     pts += 1.0 / 30.0;
//! }
//! muxer.finish()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Boundary rules
//!
//! Boundaries follow the access-unit structure of ITU-T H.264/H.265: a new
//! unit starts at an AUD NAL, or — when the stream has no AUDs — at
//! parameter sets that follow coded slices, or at a first slice that
//! follows the previous picture's last slice. SEI NALs never start a unit
//! on their own (a suffix SEI belongs to the picture it follows; the next
//! picture's slice then opens the new unit).
//!
//! Data partitioning (H.264 NAL types 2-4) and H.265 dependent slice
//! segments are kept with their picture: a new unit only opens at a
//! complete slice (H.264 types 1/5, H.265 types 0-9/16-21) that directly
//! follows another complete slice or a suffix SEI.

use super::common::find_start_code;
use crate::api::VideoCodec;

/// One coded picture: a byte range of the original Annex B stream.
///
/// `data` keeps its start codes, so it can be passed directly to
/// `write_video` / `write_video_with_dts` / `write_video_sample`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessUnit<'a> {
    /// Raw Annex B bytes for this picture, start codes included.
    pub data: &'a [u8],
    /// Whether this picture is a random-access (keyframe) picture:
    /// contains an IDR slice (H.264) or an IRAP picture (H.265).
    pub is_keyframe: bool,
}

/// NAL header with its byte offset, for boundary scanning.
struct NalPos {
    /// Offset of the start code.
    start: usize,
    /// Full NAL type number (H.264: `byte & 0x1f`, H.265: `(byte >> 1) & 0x3f`).
    nal_type: u8,
}

/// Collect `(start, header, nal_type)` for every NAL in the stream.
fn scan_nals(data: &[u8], h265: bool) -> Vec<NalPos> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some((pos, len)) = find_start_code(data, cursor) {
        let header = pos + len;
        if header < data.len() {
            let byte = data[header];
            let nal_type = if h265 {
                (byte >> 1) & 0x3f
            } else {
                byte & 0x1f
            };
            out.push(NalPos {
                start: pos,
                nal_type,
            });
        }
        cursor = header.max(cursor + 1);
    }
    out
}

fn is_h264_slice(t: u8) -> bool {
    t == 1 || t == 5
}

fn is_hevc_slice(t: u8) -> bool {
    t <= 9 || (16..=21).contains(&t)
}

/// Split an H.264 Annex B stream into access units.
///
/// Returns an empty vector when no start code is present (nothing to split;
/// callers should reject such input rather than guess).
pub fn split_h264_access_units(data: &[u8]) -> Vec<AccessUnit<'_>> {
    let nals = scan_nals(data, false);
    if nals.is_empty() {
        return Vec::new();
    }
    // NAL indices that open a new access unit.
    let mut boundaries = vec![0usize];
    let mut seen_slice = false;
    let mut prev_type: Option<u8> = None;
    for (i, nal) in nals.iter().enumerate() {
        let t = nal.nal_type;
        if i > 0 {
            let opens = t == 9  // AUD always opens
                || ((t == 7 || t == 8) && seen_slice) // PS after coded data
                || (is_h264_slice(t)
                    && seen_slice
                    && matches!(prev_type, Some(1) | Some(5) | Some(6)));
            if opens {
                boundaries.push(i);
                seen_slice = false;
            }
        }
        if is_h264_slice(t) {
            seen_slice = true;
        }
        prev_type = Some(t);
    }
    build_units(data, &nals, &boundaries, |types| {
        types.iter().any(|&t| t == 5)
    })
}

/// Split an H.265/HEVC Annex B stream into access units.
///
/// Same contract as [`split_h264_access_units`]; keyframes are pictures
/// containing an IRAP NAL (types 16-21: BLA/WLP/CRA/IDR).
pub fn split_hevc_access_units(data: &[u8]) -> Vec<AccessUnit<'_>> {
    let nals = scan_nals(data, true);
    if nals.is_empty() {
        return Vec::new();
    }
    let mut boundaries = vec![0usize];
    let mut seen_slice = false;
    let mut prev_type: Option<u8> = None;
    for (i, nal) in nals.iter().enumerate() {
        let t = nal.nal_type;
        if i > 0 {
            let opens = t == 35 // AUD always opens
                || ((t == 32 || t == 33 || t == 34) && seen_slice) // VPS/SPS/PPS after coded data
                || (is_hevc_slice(t)
                    && seen_slice
                    && prev_type.is_some_and(|p| is_hevc_slice(p) || p == 40));
            if opens {
                boundaries.push(i);
                seen_slice = false;
            }
        }
        if is_hevc_slice(t) {
            seen_slice = true;
        }
        prev_type = Some(t);
    }
    build_units(data, &nals, &boundaries, |types| {
        types.iter().any(|&t| (16..=21).contains(&t))
    })
}

/// Split a raw Annex B stream for the given codec.
///
/// AV1 (OBU stream) and VP9 (compressed frames) have no Annex B framing;
/// pass those through unchanged as a single unit when the input is
/// non-empty.
pub fn split_access_units(codec: VideoCodec, data: &[u8]) -> Vec<AccessUnit<'_>> {
    match codec {
        VideoCodec::H264 => split_h264_access_units(data),
        VideoCodec::H265 => split_hevc_access_units(data),
        VideoCodec::Av1 | VideoCodec::Vp9 => {
            if data.is_empty() {
                Vec::new()
            } else {
                vec![AccessUnit {
                    data,
                    is_keyframe: false,
                }]
            }
        }
    }
}

/// Slice byte ranges out of the original stream at NAL boundaries.
fn build_units<'a>(
    data: &'a [u8],
    nals: &[NalPos],
    boundaries: &[usize],
    is_keyframe: impl Fn(&[u8]) -> bool,
) -> Vec<AccessUnit<'a>> {
    let mut units = Vec::with_capacity(boundaries.len());
    for (k, &start_idx) in boundaries.iter().enumerate() {
        let start = nals[start_idx].start;
        let end = boundaries
            .get(k + 1)
            .map(|&next| nals[next].start)
            .unwrap_or(data.len());
        // Skip zero-length units from back-to-back start codes.
        if start >= end {
            continue;
        }
        let mut types = [0u8; 64];
        let mut count = 0;
        for nal in &nals[start_idx
            ..nals
                .len()
                .min(boundaries.get(k + 1).copied().unwrap_or(nals.len()))]
        {
            if count < types.len() {
                types[count] = nal.nal_type;
                count += 1;
            }
        }
        units.push(AccessUnit {
            data: &data[start..end],
            is_keyframe: is_keyframe(&types[..count]),
        });
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nal(prefix4: bool, header: u8, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        if prefix4 {
            v.extend_from_slice(&[0, 0, 0, 1]);
        } else {
            v.extend_from_slice(&[0, 0, 1]);
        }
        v.push(header);
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn h264_aud_delimited() {
        let mut stream = Vec::new();
        // AU 1: AUD SPS PPS IDR
        stream.extend(nal(true, 0x09, &[0x10]));
        stream.extend(nal(true, 0x67, &[0x42]));
        stream.extend(nal(true, 0x68, &[0xce]));
        stream.extend(nal(true, 0x65, &[0x88]));
        // AU 2: AUD P-slice
        stream.extend(nal(true, 0x09, &[0x10]));
        stream.extend(nal(true, 0x41, &[0x9a]));
        let units = split_h264_access_units(&stream);
        assert_eq!(units.len(), 2);
        assert!(units[0].is_keyframe);
        assert!(!units[1].is_keyframe);
        assert_eq!(units.concat_len(), stream.len());
    }

    #[test]
    fn h264_no_aud_splits_at_repeated_ps() {
        let mut stream = Vec::new();
        stream.extend(nal(true, 0x67, &[0x42]));
        stream.extend(nal(true, 0x68, &[0xce]));
        stream.extend(nal(true, 0x65, &[0x88]));
        stream.extend(nal(true, 0x67, &[0x42]));
        stream.extend(nal(true, 0x68, &[0xce]));
        stream.extend(nal(true, 0x41, &[0x9a]));
        let units = split_h264_access_units(&stream);
        assert_eq!(units.len(), 2);
        assert!(units[0].is_keyframe);
        assert!(!units[1].is_keyframe);
    }

    #[test]
    fn h264_back_to_back_slices_split() {
        let mut stream = Vec::new();
        stream.extend(nal(true, 0x65, &[0x88]));
        stream.extend(nal(true, 0x41, &[0x9a]));
        let units = split_h264_access_units(&stream);
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn h264_suffix_sei_stays_with_picture() {
        let mut stream = Vec::new();
        stream.extend(nal(true, 0x65, &[0x88]));
        stream.extend(nal(true, 0x06, &[0x01])); // suffix SEI
        stream.extend(nal(true, 0x41, &[0x9a]));
        let units = split_h264_access_units(&stream);
        assert_eq!(units.len(), 2);
        // Suffix SEI belongs to the first picture.
        assert!(units[0].data.len() > units[1].data.len());
    }

    #[test]
    fn h264_no_start_codes_yields_nothing() {
        assert!(split_h264_access_units(&[0x65, 0x88, 0x84]).is_empty());
        assert!(split_h264_access_units(&[]).is_empty());
    }

    #[test]
    fn hevc_aud_delimited() {
        let mut stream = Vec::new();
        // VPS=32 -> header (32<<1)=0x40, SPS=33 -> 0x42, PPS=34 -> 0x44,
        // IDR_W_RADL=19 -> 0x26, TRAIL_R=1 -> 0x02, AUD=35 -> 0x46.
        stream.extend(nal(true, 0x46, &[0x01]));
        stream.extend(nal(true, 0x40, &[0x01]));
        stream.extend(nal(true, 0x42, &[0x01]));
        stream.extend(nal(true, 0x44, &[0x01]));
        stream.extend(nal(true, 0x26, &[0x01]));
        stream.extend(nal(true, 0x46, &[0x01]));
        stream.extend(nal(true, 0x02, &[0x01]));
        let units = split_hevc_access_units(&stream);
        assert_eq!(units.len(), 2);
        assert!(units[0].is_keyframe);
        assert!(!units[1].is_keyframe);
    }

    #[test]
    fn dispatcher_covers_all_codecs() {
        let mut stream = Vec::new();
        stream.extend(nal(true, 0x65, &[0x88]));
        stream.extend(nal(true, 0x41, &[0x9a]));
        assert_eq!(split_access_units(VideoCodec::H264, &stream).len(), 2);
        let obu = [0x0a, 0x01, 0x02];
        assert_eq!(split_access_units(VideoCodec::Av1, &obu).len(), 1);
        assert!(split_access_units(VideoCodec::Vp9, &[]).is_empty());
    }

    trait ConcatLen {
        fn concat_len(&self) -> usize;
    }

    impl ConcatLen for Vec<AccessUnit<'_>> {
        fn concat_len(&self) -> usize {
            self.iter().map(|u| u.data.len()).sum()
        }
    }
}
