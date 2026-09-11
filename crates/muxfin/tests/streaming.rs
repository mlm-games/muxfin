//! Tests for the seekable-streaming writer (§5).
//!
//! The streaming layout (`ftyp`, `mdat`, `moov`) differs from the buffered
//! fast-start layout, so these tests compare decoded semantics — sample
//! counts, total durations, composition offsets, and chunk-offset sanity —
//! plus error parity with [`muxfin::api::Muxer`].

use std::io::Cursor;

use muxfin::api::{AacProfile, AudioCodec, MuxerBuilder, VideoCodec};
use muxfin::time::{EncodedSample, SampleTime};

const SPS: &[u8] = &[
    0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11,
];
const PPS: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80];
const IDR: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, 0x11];
const P_SLICE: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x41, 0x9a, 0x11, 0x22];
const ADTS: &[u8] = &[0xff, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc, 0xaa, 0xbb];

fn keyframe() -> Vec<u8> {
    [SPS, PPS, IDR].concat()
}

/// Minimal top-level box scan: (type, payload offset, payload len).
fn top_boxes(data: &[u8]) -> Vec<([u8; 4], usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let mut size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as u64;
        let typ: [u8; 4] = data[pos + 4..pos + 8].try_into().unwrap();
        let mut header = 8usize;
        if size == 1 {
            if pos + 16 > data.len() {
                break;
            }
            size = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap());
            header = 16;
        } else if size == 0 {
            size = (data.len() - pos) as u64;
        }
        if size < 8 || pos + size as usize > data.len() {
            break;
        }
        out.push((typ, pos + header, size as usize - header));
        pos += size as usize;
    }
    out
}

/// Find all boxes of `typ` anywhere (descending into container boxes).
/// Returns full boxes (size + type + payload).
fn find_boxes(data: &[u8], typ: &[u8; 4]) -> Vec<Vec<u8>> {
    let mut found = Vec::new();
    let mut stack: Vec<&[u8]> = vec![data];
    while let Some(buf) = stack.pop() {
        let mut pos = 0usize;
        while pos + 8 <= buf.len() {
            let mut size = u32::from_be_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            let t: [u8; 4] = buf[pos + 4..pos + 8].try_into().unwrap();
            let mut header = 8usize;
            if size == 1 {
                if pos + 16 > buf.len() {
                    break;
                }
                size = u64::from_be_bytes(buf[pos + 8..pos + 16].try_into().unwrap()) as usize;
                header = 16;
            } else if size == 0 {
                size = buf.len() - pos;
            }
            if size < 8 || pos + size > buf.len() {
                break;
            }
            let payload = &buf[pos + header..pos + size];
            if &t == typ {
                found.push(buf[pos..pos + size].to_vec());
            }
            if matches!(
                &t,
                b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts"
            ) {
                stack.push(payload);
            }
            pos += size;
        }
    }
    found
}

fn stsz_count(full: &[u8]) -> u32 {
    assert_eq!(&full[4..8], b"stsz", "expected stsz fullbox");
    u32::from_be_bytes(full[16..20].try_into().unwrap())
}

fn stts_total(full: &[u8]) -> u64 {
    assert_eq!(&full[4..8], b"stts", "expected stts fullbox");
    let entries = u32::from_be_bytes(full[12..16].try_into().unwrap()) as usize;
    let mut total = 0u64;
    for i in 0..entries {
        let base = 16 + i * 8;
        let count = u32::from_be_bytes(full[base..base + 4].try_into().unwrap()) as u64;
        let delta = u32::from_be_bytes(full[base + 4..base + 8].try_into().unwrap()) as u64;
        total += count * delta;
    }
    total
}

fn write_streamed_to_vec() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream.mp4");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .unwrap();
    let mut muxer = MuxerBuilder::new(file)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .build_streaming_seekable()
        .unwrap();
    muxer
        .write_audio_sample(EncodedSample {
            data: ADTS,
            timing: SampleTime::new(0, 0, 1024).unwrap(),
            is_sync: true,
        })
        .unwrap();
    muxer
        .write_video_sample(EncodedSample {
            data: &keyframe(),
            timing: SampleTime::new(90_000, 90_000, 3000).unwrap(),
            is_sync: true,
        })
        .unwrap();
    muxer
        .write_video_sample(EncodedSample {
            data: P_SLICE,
            timing: SampleTime::new(96_000, 93_000, 3000).unwrap(),
            is_sync: false,
        })
        .unwrap();
    muxer
        .write_audio_sample(EncodedSample {
            data: ADTS,
            timing: SampleTime::new(1024, 1024, 1024).unwrap(),
            is_sync: true,
        })
        .unwrap();
    muxer.finish().unwrap();
    std::fs::read(&path).unwrap()
}

#[test]
fn streaming_layout_is_ftyp_mdat_moov() {
    // File-backed streaming needs a read+write handle; use a temp file.
    let bytes = write_streamed_to_vec();
    let tops = top_boxes(&bytes);
    let types: Vec<[u8; 4]> = tops.iter().map(|(t, _, _)| *t).collect();
    assert_eq!(types, vec![*b"ftyp", *b"mdat", *b"moov"]);
    // mdat uses the fixed 16-byte largesize header (size field == 1).
    let mdat_size = u32::from_be_bytes(bytes[tops[1].1 - 16..tops[1].1 - 12].try_into().unwrap());
    assert_eq!(mdat_size, 1, "streaming mdat must use largesize form");
}

#[test]
fn streaming_tables_match_buffered_semantics() {
    let bytes = write_streamed_to_vec();
    // Two video samples, two audio samples.
    let stsz = find_boxes(&bytes, b"stsz");
    assert_eq!(stsz.len(), 2, "one stsz per track");
    let mut counts: Vec<u32> = stsz.iter().map(|p| stsz_count(p)).collect();
    counts.sort_unstable();
    assert_eq!(counts, vec![2, 2]);
    // B-frame composition offsets survive streaming (ctts present).
    let ctts = find_boxes(&bytes, b"ctts");
    assert_eq!(ctts.len(), 1, "video track with pts != dts needs ctts");
    // Edit lists preserve the audio-before-video offset.
    let elst = find_boxes(&bytes, b"elst");
    assert!(!elst.is_empty(), "late video needs edts/elst");
    // Chunk offsets land inside mdat.
    let tops = top_boxes(&bytes);
    let (mdat_off, mdat_len) = tops
        .iter()
        .find_map(|(t, off, len)| (*t == *b"mdat").then_some((*off, *len)))
        .unwrap();
    let mdat_range = mdat_off..mdat_off + mdat_len;
    for payload in find_boxes(&bytes, b"stco") {
        let entries = u32::from_be_bytes(payload[12..16].try_into().unwrap()) as usize;
        for i in 0..entries {
            let off = u32::from_be_bytes(payload[16 + i * 4..20 + i * 4].try_into().unwrap());
            // Offsets are absolute file positions; the first chunks must be in mdat.
            assert!(
                mdat_range.contains(&(off as usize)),
                "chunk offset {off} outside mdat"
            );
        }
    }
    // Total durations are sane: video 6000 ticks @90kHz; audio
    // 2x1024 @48kHz rescaled to 2x1920 = 3840 @90kHz.
    let stts = find_boxes(&bytes, b"stts");
    let mut totals: Vec<u64> = stts.iter().map(|p| stts_total(p)).collect();
    totals.sort_unstable();
    assert_eq!(totals, vec![3840, 6000]);
}

#[test]
fn streaming_error_parity_with_buffered() {
    // Non-increasing DTS rejected.
    let mut muxer = MuxerBuilder::new(Cursor::new(Vec::<u8>::new()))
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build_streaming_seekable()
        .unwrap();
    muxer
        .write_video_sample(EncodedSample {
            data: &keyframe(),
            timing: SampleTime::new(0, 0, 3000).unwrap(),
            is_sync: true,
        })
        .unwrap();
    let err = muxer
        .write_video_sample(EncodedSample {
            data: P_SLICE,
            timing: SampleTime::new(3000, 0, 3000).unwrap(),
            is_sync: false,
        })
        .unwrap_err();
    assert!(
        matches!(
            err,
            muxfin::api::MuxerError::NonIncreasingIntDts { .. }
                | muxfin::api::MuxerError::NonIncreasingVideoPts { .. }
                | muxfin::api::MuxerError::NonIncreasingDts { .. }
        ),
        "unexpected error: {err:?}"
    );

    // First frame must be a keyframe.
    let mut muxer = MuxerBuilder::new(Cursor::new(Vec::<u8>::new()))
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build_streaming_seekable()
        .unwrap();
    let err = muxer
        .write_video_sample(EncodedSample {
            data: P_SLICE,
            timing: SampleTime::new(0, 0, 3000).unwrap(),
            is_sync: false,
        })
        .unwrap_err();
    assert!(
        matches!(err, muxfin::api::MuxerError::FirstVideoFrameMustBeKeyframe),
        "unexpected error: {err:?}"
    );

    // Zero duration rejected without writing.
    assert!(SampleTime::new(0, 0, 0).is_err());
}

#[test]
fn streaming_enforces_sample_count_limit() {
    use muxfin::time::Limits;
    let limits = Limits {
        max_samples_per_track: 1,
        ..Limits::default()
    };
    let mut muxer = MuxerBuilder::new(Cursor::new(Vec::<u8>::new()))
        .video(VideoCodec::H264, 640, 480, 30.0)
        .with_limits(limits)
        .build_streaming_seekable()
        .unwrap();
    muxer
        .write_video_sample(EncodedSample {
            data: &keyframe(),
            timing: SampleTime::new(0, 0, 3000).unwrap(),
            is_sync: true,
        })
        .unwrap();
    let err = muxer
        .write_video_sample(EncodedSample {
            data: P_SLICE,
            timing: SampleTime::new(3000, 3000, 3000).unwrap(),
            is_sync: false,
        })
        .unwrap_err();
    assert!(
        matches!(err, muxfin::api::MuxerError::ResourceLimitExceeded { .. }),
        "unexpected error: {err:?}"
    );
}

#[test]
fn streaming_f64_shims_work() {
    let mut muxer = MuxerBuilder::new(Cursor::new(Vec::<u8>::new()))
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build_streaming_seekable()
        .unwrap();
    muxer.write_video(0.0, &keyframe(), true).unwrap();
    muxer
        .write_video_with_dts(0.1, 0.033, P_SLICE, false)
        .unwrap();
    let stats = muxer.finish_with_stats().unwrap();
    assert_eq!(stats.video_frames, 2);
}
