//! Integration tests for Matroska (MKV) and WebM muxing.
//!
//! Files produced by [`muxfin::api::MkvMuxer`] are validated with the
//! external `matroska-demuxer` crate (not our own writer): tracks, codec
//! IDs, codec-private blobs and frame payloads must round-trip.

mod support;

use matroska_demuxer::{MatroskaFile, TrackType};
use muxfin::api::{
    AacProfile, AudioCodec, ContainerFormat, Metadata, MuxerBuilder, SubtitleCodec, VideoCodec,
};
use std::io::Cursor;
use std::{fs, path::Path};
use support::SharedBuffer;

fn read_hex_fixture(dir: &str, name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(dir)
        .join(name);
    let contents = fs::read_to_string(path).expect("fixture must be readable");
    let hex: String = contents.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        out.push(u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex"));
    }
    out
}

/// Minimal H.264 keyframe (SPS + PPS + IDR) in Annex B format.
fn build_h264_keyframe() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xab, 0x40, 0xf0, 0x28, 0xd0,
    ]);
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80]);
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, 0x10]);
    data
}

/// Minimal H.264 P-frame (non-IDR slice) in Annex B format.
fn build_h264_pframe() -> Vec<u8> {
    vec![0x00, 0x00, 0x00, 0x01, 0x61, 0x88, 0x84, 0x00, 0x10]
}

/// Minimal Opus packet (SILK 20ms, stereo, 1 frame).
fn build_opus_packet() -> Vec<u8> {
    vec![0x24, 0xc0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05]
}

/// Minimal VP9 keyframe with a valid frame header (100x100).
fn build_vp9_keyframe() -> Vec<u8> {
    let mut data = vec![0x49, 0x83, 0x42, 0x00, 0x80, 0x64, 0x64, 0x12];
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    data
}

/// Minimal AV1 keyframe: Sequence Header OBU + Frame OBU.
fn build_av1_keyframe() -> Vec<u8> {
    let mut data = vec![0x0A, 12];
    data.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x10, 0x07, 0x80, 0x04, 0x38, 0x00, 0x00, 0x00, 0x00,
    ]);
    data.extend_from_slice(&[0x32, 4, 0x10, 0x00, 0x00, 0x00]);
    data
}

fn open_demuxed(bytes: &[u8]) -> MatroskaFile<Cursor<&[u8]>> {
    MatroskaFile::open(Cursor::new(bytes)).expect("produced file must demux")
}

/// matroska-demuxer preserves the EBML null terminator on strings.
fn clean(s: &str) -> &str {
    s.trim_end_matches('\0')
}

#[test]
fn mkv_starts_with_ebml_magic_and_matroska_doctype() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build_mkv()
        .unwrap();
    muxer
        .write_video(0.0, &build_h264_keyframe(), true)
        .unwrap();
    muxer.finish().unwrap();

    let produced = buffer.lock().unwrap();
    assert!(
        produced.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]),
        "MKV must start with the EBML magic"
    );
    assert!(
        produced.windows(8).any(|w| w == b"matroska"),
        "EBML header must declare DocType matroska"
    );
}

#[test]
fn mkv_h264_aac_roundtrip() {
    let frame0 = read_hex_fixture("video_samples", "frame0_key.264");
    let frame1 = read_hex_fixture("video_samples", "frame1_p.264");
    let frame2 = read_hex_fixture("video_samples", "frame2_p.264");
    let a0 = read_hex_fixture("audio_samples", "frame0.aac.adts");
    let a1 = read_hex_fixture("audio_samples", "frame1.aac.adts");
    let a2 = read_hex_fixture("audio_samples", "frame2.aac.adts");

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .with_metadata(Metadata::new().with_title("mkv roundtrip"))
        .build_mkv()
        .unwrap();

    muxer.write_video(0.0, &frame0, true).unwrap();
    muxer.write_audio(0.0, &a0).unwrap();
    muxer.write_audio(0.021, &a1).unwrap();
    muxer.write_video(0.033, &frame1, false).unwrap();
    muxer.write_audio(0.042, &a2).unwrap();
    muxer.write_video(0.066, &frame2, false).unwrap();
    let stats = muxer.finish_with_stats().unwrap();
    assert_eq!(stats.video_frames, 3);
    assert_eq!(stats.audio_frames, 3);

    let produced = buffer.lock().unwrap();
    let mkv = open_demuxed(&produced);

    assert_eq!(mkv.info().title().map(clean), Some("mkv roundtrip"));
    assert_eq!(mkv.tracks().len(), 2);

    let video_no: u64;
    let audio_no: u64;
    {
        let video = mkv
            .tracks()
            .iter()
            .find(|t| t.track_type() == TrackType::Video)
            .expect("video track");
        assert_eq!(clean(video.codec_id()), "V_MPEG4/ISO/AVC");
        // avcC payload: configurationVersion == 1 first.
        let private = video.codec_private().expect("avcC CodecPrivate");
        assert!(!private.is_empty());
        assert_eq!(private[0], 1);
        let video_dims = video.video().expect("video settings");
        assert_eq!(video_dims.pixel_width().get(), 640);
        assert_eq!(video_dims.pixel_height().get(), 480);
        video_no = video.track_number().get();

        let audio = mkv
            .tracks()
            .iter()
            .find(|t| t.track_type() == TrackType::Audio)
            .expect("audio track");
        assert_eq!(clean(audio.codec_id()), "A_AAC");
        // AudioSpecificConfig is exactly 2 bytes.
        let asc = audio.codec_private().expect("ASC CodecPrivate");
        assert_eq!(asc.len(), 2);
        audio_no = audio.track_number().get();
    }

    // All frames must come back out with monotonic timestamps.
    let mut mkv = mkv;
    let mut frame = matroska_demuxer::Frame::default();
    let (mut video_frames, mut audio_frames) = (0, 0);
    let mut last_ts = 0u64;
    let mut first = true;
    while mkv.next_frame(&mut frame).unwrap() {
        if first {
            first = false;
        } else {
            assert!(frame.timestamp >= last_ts, "timestamps must not go back");
        }
        last_ts = frame.timestamp;
        if frame.track == video_no {
            video_frames += 1;
        } else if frame.track == audio_no {
            audio_frames += 1;
            // ADTS headers are stripped: payload must not start with a syncword.
            assert!(
                !(frame.data.len() >= 2 && frame.data[0] == 0xFF && frame.data[1] & 0xF0 == 0xF0),
                "AAC payloads must be raw (no ADTS header)"
            );
        } else {
            panic!("unexpected track {}", frame.track);
        }
        frame = matroska_demuxer::Frame::default();
    }
    assert_eq!(video_frames, 3);
    assert_eq!(audio_frames, 3);
}

#[test]
fn mkv_opus_head_roundtrip() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Opus, 48_000, 2)
        .build_mkv()
        .unwrap();

    muxer
        .write_video(0.0, &build_h264_keyframe(), true)
        .unwrap();
    let packet = build_opus_packet();
    muxer.write_audio(0.0, &packet).unwrap();
    muxer.write_audio(0.02, &packet).unwrap();
    muxer.finish().unwrap();

    let produced = buffer.lock().unwrap();
    let mkv = open_demuxed(&produced);
    let audio = mkv
        .tracks()
        .iter()
        .find(|t| t.track_type() == TrackType::Audio)
        .expect("audio track");
    assert_eq!(clean(audio.codec_id()), "A_OPUS");
    let private = audio.codec_private().expect("OpusHead CodecPrivate");
    assert_eq!(&private[..8], b"OpusHead");
    assert_eq!(private[9], 2, "channel count must round-trip");
}

#[test]
fn webm_vp9_opus_roundtrip() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .audio(AudioCodec::Opus, 48_000, 2)
        .with_container(ContainerFormat::WebM)
        .build_mkv()
        .unwrap();
    assert_eq!(muxer.container(), ContainerFormat::WebM);

    muxer.write_video(0.0, &build_vp9_keyframe(), true).unwrap();
    let packet = build_opus_packet();
    muxer.write_audio(0.0, &packet).unwrap();
    muxer.finish().unwrap();

    let produced = buffer.lock().unwrap();
    assert!(
        produced.windows(4).any(|w| w == b"webm"),
        "EBML header must declare DocType webm"
    );

    let mkv = open_demuxed(&produced);
    assert_eq!(clean(mkv.ebml_header().doc_type()), "webm");
    let video = mkv
        .tracks()
        .iter()
        .find(|t| t.track_type() == TrackType::Video)
        .expect("video track");
    assert_eq!(clean(video.codec_id()), "V_VP9");
    assert!(
        video.codec_private().is_none(),
        "V_VP9 carries no CodecPrivate"
    );
}

#[test]
fn webm_av1_roundtrip() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Av1, 1920, 1080, 30.0)
        .with_container(ContainerFormat::WebM)
        .build_mkv()
        .unwrap();

    muxer.write_video(0.0, &build_av1_keyframe(), true).unwrap();
    muxer.finish().unwrap();

    let produced = buffer.lock().unwrap();
    let mkv = open_demuxed(&produced);
    let video = mkv
        .tracks()
        .iter()
        .find(|t| t.track_type() == TrackType::Video)
        .expect("video track");
    assert_eq!(clean(video.codec_id()), "V_AV1");
    // av1C payload: marker byte 0x81 first.
    let private = video.codec_private().expect("av1C CodecPrivate");
    assert_eq!(private[0], 0x81);
}

#[test]
fn webm_rejects_h264() {
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .with_container(ContainerFormat::WebM)
        .build_mkv()
        .unwrap();
    muxer
        .write_video(0.0, &build_h264_keyframe(), true)
        .unwrap();
    let err = muxer.finish().unwrap_err().to_string();
    assert!(
        err.contains("WebM"),
        "error must mention WebM, got: {}",
        err
    );
}

#[test]
fn webm_rejects_aac() {
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .with_container(ContainerFormat::WebM)
        .build_mkv()
        .unwrap();
    muxer.write_video(0.0, &build_vp9_keyframe(), true).unwrap();
    let err = muxer.finish().unwrap_err().to_string();
    assert!(
        err.contains("Opus"),
        "error must suggest Opus, got: {}",
        err
    );
}

#[test]
fn webm_rejects_subtitles() {
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .subtitle(SubtitleCodec::MovText, Some("eng".to_string()))
        .with_container(ContainerFormat::WebM)
        .build_mkv()
        .unwrap();
    muxer.write_video(0.0, &build_vp9_keyframe(), true).unwrap();
    muxer.write_subtitle(0.0, 1.0, "hello").unwrap();
    let err = muxer.finish().unwrap_err().to_string();
    assert!(
        err.contains("WebM"),
        "error must mention WebM, got: {}",
        err
    );
}

#[test]
fn mkv_subtitle_track_roundtrip() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .subtitle(SubtitleCodec::MovText, Some("eng".to_string()))
        .build_mkv()
        .unwrap();

    muxer
        .write_video(0.0, &build_h264_keyframe(), true)
        .unwrap();
    muxer.write_subtitle(0.5, 2.0, "hello world").unwrap();
    muxer.finish().unwrap();

    let produced = buffer.lock().unwrap();
    let mkv = open_demuxed(&produced);
    let subtitle_no: u64;
    {
        let subtitle = mkv
            .tracks()
            .iter()
            .find(|t| t.track_type() == TrackType::Subtitle)
            .expect("subtitle track");
        assert_eq!(clean(subtitle.codec_id()), "S_TEXT/UTF8");
        subtitle_no = subtitle.track_number().get();
    }

    let mut mkv = mkv;
    let mut frame = matroska_demuxer::Frame::default();
    let mut saw_subtitle = false;
    while mkv.next_frame(&mut frame).unwrap() {
        if frame.track == subtitle_no {
            saw_subtitle = true;
            assert_eq!(frame.data, b"hello world");
        }
        frame = matroska_demuxer::Frame::default();
    }
    assert!(saw_subtitle, "subtitle block must round-trip");
}

#[test]
fn mkv_bframes_with_dts_roundtrip() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build_mkv()
        .unwrap();

    // Decode order I P B with presentation I B P.
    let key = build_h264_keyframe();
    let p = build_h264_pframe();
    muxer.write_video_with_dts(0.0, 0.0, &key, true).unwrap();
    muxer.write_video_with_dts(0.066, 0.033, &p, false).unwrap();
    muxer.write_video_with_dts(0.033, 0.066, &p, false).unwrap();
    muxer.finish().unwrap();

    let produced = buffer.lock().unwrap();
    let mut mkv = open_demuxed(&produced);
    let mut frame = matroska_demuxer::Frame::default();
    let mut count = 0;
    while mkv.next_frame(&mut frame).unwrap() {
        count += 1;
        frame = matroska_demuxer::Frame::default();
    }
    assert_eq!(count, 3);
}

#[test]
fn mkv_validation_mirrors_mp4() {
    // First frame must be a keyframe.
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build_mkv()
        .unwrap();
    let err = muxer
        .write_video(0.0, &build_h264_pframe(), false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("must be a keyframe"), "got: {}", err);

    // Non-increasing PTS rejected.
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build_mkv()
        .unwrap();
    muxer
        .write_video(0.0, &build_h264_keyframe(), true)
        .unwrap();
    let err = muxer
        .write_video(0.0, &build_h264_pframe(), false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not greater"), "got: {}", err);

    // Subtitle requires video.
    let (writer, _buffer) = SharedBuffer::new();
    let err = match MuxerBuilder::new(writer)
        .subtitle(SubtitleCodec::MovText, None)
        .build_mkv()
    {
        Ok(_) => panic!("subtitle without video must fail"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("subtitle"), "got: {}", err);
}

#[test]
fn mkv_container_format_parsing() {
    use std::str::FromStr;
    assert_eq!(
        ContainerFormat::from_str("mkv").unwrap(),
        ContainerFormat::Matroska
    );
    assert_eq!(
        ContainerFormat::from_str("webm").unwrap(),
        ContainerFormat::WebM
    );
    assert_eq!(
        ContainerFormat::from_str("mp4").unwrap(),
        ContainerFormat::Mp4
    );
    assert_eq!(ContainerFormat::Matroska.extension(), "mkv");
    assert!(ContainerFormat::WebM.is_matroska_family());
    assert!(!ContainerFormat::Mp4.is_matroska_family());
}
