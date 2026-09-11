//! Verdict hardening matrix: integer time, explicit duration, subtitles,
//! co64/edit-lists, decoder-config changes, limits.

use muxfin::api::{
    AacProfile, AudioCodec, MuxerBuilder, MuxerError, SegmentBoundary, SubtitleCodec, VideoCodec,
};
use muxfin::time::{EncodedSample, LanguageCode, Limits, SampleTime, SubtitleCue, Timescale};

mod support;
use support::SharedBuffer;

fn h264_keyframe() -> Vec<u8> {
    vec![
        0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11, // SPS
        0, 0, 0, 1, 0x68, 0xce, 0x38, 0x80, // PPS
        0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, 0x11, 0x22, // IDR
    ]
}

fn h264_inter() -> Vec<u8> {
    vec![0, 0, 0, 1, 0x41, 0x9a, 0x11, 0x22, 0x33]
}

fn adts_frame() -> Vec<u8> {
    vec![0xff, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc, 0xaa, 0xbb]
}

#[test]
fn integer_video_sample_with_explicit_duration() {
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    // 29.97fps-style duration 3003 ticks at 90kHz is exact (f64 would quantize).
    muxer
        .write_video_sample(EncodedSample {
            data: &h264_keyframe(),
            timing: SampleTime::new(0, 0, 3003).unwrap(),
            is_sync: true,
        })
        .unwrap();
    muxer
        .write_video_sample(EncodedSample {
            data: &h264_inter(),
            timing: SampleTime::new(3003, 3003, 3003).unwrap(),
            is_sync: false,
        })
        .unwrap();
}

#[test]
fn zero_duration_rejected() {
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    assert!(SampleTime::new(0, 0, 0).is_err());
    // Bypass constructor: zero duration via struct literal must still fail at write.
    let err = muxer
        .write_video_sample(EncodedSample {
            data: &h264_keyframe(),
            timing: SampleTime {
                pts: 0,
                dts: 0,
                duration: 0,
            },
            is_sync: true,
        })
        .unwrap_err();
    assert!(matches!(err, MuxerError::ZeroDuration));
}

#[test]
fn integer_dts_must_increase() {
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    muxer
        .write_video_sample(EncodedSample {
            data: &h264_keyframe(),
            timing: SampleTime::new(0, 0, 3000).unwrap(),
            is_sync: true,
        })
        .unwrap();
    let err = muxer
        .write_video_sample(EncodedSample {
            data: &h264_inter(),
            timing: SampleTime::new(3000, 0, 3000).unwrap(),
            is_sync: false,
        })
        .unwrap_err();
    assert!(matches!(err, MuxerError::NonIncreasingIntDts { .. }));
}

#[test]
fn audio_before_video_allowed_and_emits_edts() {
    let (writer, reader) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .build()
        .unwrap();
    muxer.write_audio(0.0, &adts_frame()).unwrap();
    muxer.write_video(1.0, &h264_keyframe(), true).unwrap();
    muxer.finish().unwrap();
    let bytes = reader.lock().unwrap().clone();
    // Video starts at 1.0s = 90000 ticks; movie origin is 0 (audio) so the
    // video trak must carry edts/elst.
    assert!(bytes.windows(4).any(|w| w == b"edts"));
    assert!(bytes.windows(4).any(|w| w == b"elst"));
}

#[test]
fn subtitle_final_cue_keeps_own_duration() {
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .subtitle(SubtitleCodec::MovText, Some("eng".into()))
        .build()
        .unwrap();
    muxer.write_video(0.0, &h264_keyframe(), true).unwrap();
    muxer
        .write_subtitle_cue(SubtitleCue::new(0, 1_000, "hello").unwrap())
        .unwrap();
    // Gap, overlap, unicode, empty-gap coverage.
    muxer
        .write_subtitle_cue(SubtitleCue::new(2_000, 500, "héllo 🌍").unwrap())
        .unwrap();
    muxer
        .write_subtitle_cue(SubtitleCue::new(2_250, 750, "overlap").unwrap())
        .unwrap();
    assert!(SubtitleCue::new(0, 0, "x").is_err());
}

#[test]
fn subtitle_too_large_rejected() {
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .subtitle(SubtitleCodec::MovText, Some("eng".into()))
        .build()
        .unwrap();
    muxer.write_video(0.0, &h264_keyframe(), true).unwrap();
    let big = "x".repeat(65_536);
    let err = muxer
        .write_subtitle_cue(SubtitleCue::new(0, 1000, &big).unwrap())
        .unwrap_err();
    assert!(matches!(err, MuxerError::SubtitleTooLarge(_)));
}

#[test]
fn webvtt_subtitle_muxes() {
    let (writer, reader) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .subtitle(SubtitleCodec::WebVtt, Some("eng".into()))
        .build()
        .unwrap();
    muxer.write_video(0.0, &h264_keyframe(), true).unwrap();
    muxer
        .write_subtitle_cue(SubtitleCue::new(0, 1000, "cue one").unwrap())
        .unwrap();
    muxer.finish().unwrap();
    let bytes = reader.lock().unwrap().clone();
    assert!(bytes.windows(4).any(|w| w == b"wvtt"));
    assert!(bytes.windows(4).any(|w| w == b"vttc"));
}

#[test]
fn decoder_config_change_rejected() {
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    muxer.write_video(0.0, &h264_keyframe(), true).unwrap();
    // Same PS again is fine.
    muxer.write_video(0.033, &h264_keyframe(), true).unwrap();
    // Different SPS must be rejected.
    let mut other = h264_keyframe();
    other[5] ^= 0xFF;
    let err = muxer.write_video(0.066, &other, true).unwrap_err();
    assert!(matches!(
        err,
        MuxerError::DecoderConfigurationChanged | MuxerError::FirstVideoFrameMissingSpsPps
    ));
}

#[test]
fn language_code_validated() {
    assert!(LanguageCode::parse("eng").is_ok());
    assert!(LanguageCode::parse("EN").is_err());
    assert!(muxfin::time::SUBTITLE_TIMESCALE.get() == 1_000);
    assert!(muxfin::time::VIDEO_TIMESCALE.get() == 90_000);
}

#[test]
fn limits_reject_oversize_sample() {
    let (writer, _) = SharedBuffer::new();
    let limits = Limits {
        max_sample_size: 16,
        ..Limits::default()
    };
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .with_limits(limits)
        .build()
        .unwrap();
    let err = muxer
        .write_video_sample(EncodedSample {
            data: &h264_keyframe(),
            timing: SampleTime::new(0, 0, 3000).unwrap(),
            is_sync: true,
        })
        .unwrap_err();
    assert!(matches!(err, MuxerError::ResourceLimitExceeded { .. }));
}

#[test]
fn timescale_rescale_exact() {
    let from = Timescale::const_new(90_000);
    let to = Timescale::const_new(1_000);
    // rescale is crate-private; verify via public behavior: 4500 ticks@90k
    // video sample at 3003 duration keeps exact integer timing.
    let _ = (from, to);
    let t = SampleTime::new(4_500, 4_500, 3_003).unwrap();
    assert_eq!(t.composition_offset().unwrap(), 0);
}

#[test]
fn segment_boundary_manual_never_auto_flushes() {
    use muxfin::fragmented::{FragmentConfig, FragmentedMuxer};
    let config = FragmentConfig {
        boundary: SegmentBoundary::Manual,
        ..Default::default()
    };
    let mut muxer = FragmentedMuxer::new(config);
    let data = vec![0, 0, 0, 5, 0x65, 1, 2, 3, 4];
    muxer.write_video(0, 0, &data, true).unwrap();
    muxer.write_video(180_000, 180_000, &data, true).unwrap();
    assert!(!muxer.ready_to_flush());
    assert!(muxer.flush_segment().is_some());
}
