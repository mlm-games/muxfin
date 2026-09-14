//! Integration tests for Ogg Opus muxing.
//!
//! Files produced by [`muxfin::api::OggMuxer`] are validated with the
//! crate's own [`muxfin::demux::demux_ogg_opus`] (independent code path:
//! the writer derives granules from TOC durations, the demuxer verifies
//! them by differences): channels, pre-skip, PTS and packet payloads must
//! round-trip exactly.

use muxfin::api::{AudioCodec, ContainerFormat, MuxerBuilder};
use muxfin::codec::opus::OpusConfig;
use muxfin::demux::demux_ogg_opus;
use muxfin::time::{EncodedSample, SampleTime, Timescale};

/// Minimal Opus packet: TOC 0xF8 = config 31 (CELT-only FB 20 ms =
/// 960 samples @48k), mono, single frame.
fn opus_20ms() -> Vec<u8> {
    vec![0xF8, 0x00, 0x01, 0x02]
}

/// Minimal 60 ms Opus packet: TOC 0xFF = config 31, code 3 (arbitrary
/// frames), 3 frames × 20 ms = 2880 samples.
fn opus_60ms() -> Vec<u8> {
    vec![0xFF, 0x03, 0x00, 0x01, 0x02]
}

#[test]
fn ogg_roundtrips_packets_pts_and_preskip() {
    let packets = [opus_20ms(), opus_60ms(), opus_20ms()];
    let gaps = [960.0 / 48_000.0, 2880.0 / 48_000.0, 960.0 / 48_000.0];
    let mut out = Vec::<u8>::new();
    let stats = {
        let mut muxer = MuxerBuilder::new(&mut out)
            .audio(AudioCodec::Opus, 48_000, 2)
            .build_ogg()
            .expect("build should succeed");
        let mut pts = 0.0;
        for (packet, gap) in packets.iter().zip(gaps.iter()) {
            muxer
                .write_audio(pts, packet)
                .expect("write should succeed");
            pts += gap;
        }
        muxer.finish_with_stats().expect("finish should succeed")
    };
    assert_eq!(stats.audio_frames, 3);
    assert_eq!(stats.video_frames, 0);
    assert!((stats.duration_secs - 0.1).abs() < 1e-9);
    assert!(stats.bytes_written > 0);

    let track = demux_ogg_opus(&out).expect("own demuxer must verify granules");
    assert_eq!(track.channels, 2);
    assert_eq!(track.pre_skip, 312);
    assert_eq!(track.packets.len(), 3);
    for (demuxed, original) in track.packets.iter().zip(packets.iter()) {
        assert_eq!(&demuxed.data, original);
    }
    assert!((track.packets[0].pts - 0.0).abs() < 1e-9);
    assert!((track.packets[1].pts - 0.02).abs() < 1e-9);
    assert!((track.packets[2].pts - 0.08).abs() < 1e-9);
    // End-trim bound: 312 + 960 + 2880 + 960 = 5112 samples.
    assert!((track.duration - 4800.0 / 48_000.0).abs() < 1e-6);
}

#[test]
fn ogg_rejects_video_subtitle_and_non_opus() {
    use muxfin::api::VideoCodec;

    let result = MuxerBuilder::new(Vec::<u8>::new())
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .audio(AudioCodec::Opus, 48_000, 2)
        .build_ogg();
    assert!(result.is_err(), "video must be rejected for Ogg");

    let result = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Aac(muxfin::api::AacProfile::Lc), 44_100, 2)
        .build_ogg();
    assert!(result.is_err(), "AAC must be rejected for Ogg");

    let result = MuxerBuilder::new(Vec::<u8>::new()).build_ogg();
    assert!(result.is_err(), "missing audio must be rejected");
}

#[test]
fn ogg_rejects_bad_packets_and_backwards_pts() {
    let mut muxer = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Opus, 48_000, 2)
        .build_ogg()
        .expect("build should succeed");

    assert!(
        muxer.write_audio(0.0, &[]).is_err(),
        "empty packet rejected"
    );
    assert!(
        // TOC code 3 (arbitrary frame count) with no count byte follows:
        // duration is undecodable, so the packet is rejected.
        muxer.write_audio(0.0, &[0xFF]).is_err(),
        "structurally invalid packet rejected"
    );

    muxer
        .write_audio(0.0, &opus_20ms())
        .expect("first write should succeed");
    muxer
        .write_audio(0.0, &opus_20ms())
        .expect("equal PTS is non-decreasing, allowed");
    muxer
        .write_audio(0.02, &opus_20ms())
        .expect("forward PTS allowed");
    assert!(
        muxer.write_audio(0.01, &opus_20ms()).is_err(),
        "backwards PTS rejected"
    );
}

#[test]
fn ogg_integer_sample_api_and_opus_config_override() {
    let mut muxer = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Opus, 48_000, 1)
        .with_opus_config(OpusConfig::mono().with_pre_skip(120))
        .build_ogg()
        .expect("build should succeed");

    let ts48k = Timescale::new(core::num::NonZeroU32::new(48_000).unwrap());
    let packet = opus_20ms();
    muxer
        .write_audio_sample(EncodedSample {
            data: &packet,
            timing: SampleTime {
                pts: 0,
                dts: 0,
                duration: 960,
            },
            is_sync: true,
        })
        .expect("integer write should succeed");
    let _ = ts48k;
    muxer.finish().expect("finish should succeed");
}

#[test]
fn ogg_bad_channel_count_rejected_at_finalize() {
    // 0 channels passes the `u16` plumbing but cannot form an OpusHead:
    // the writer must fail at finalize, not emit a corrupt file.
    let muxer = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Opus, 48_000, 0)
        .build_ogg()
        .expect("build stores the track; validation happens at finalize");
    assert!(muxer.finish().is_err());
}

#[test]
fn ogg_container_format_strings() {
    use std::str::FromStr;
    assert_eq!(
        ContainerFormat::from_str("ogg").unwrap(),
        ContainerFormat::Ogg
    );
    assert_eq!(
        ContainerFormat::from_str("opus").unwrap(),
        ContainerFormat::Ogg
    );
    assert_eq!(ContainerFormat::Ogg.extension(), "ogg");
    assert_eq!(ContainerFormat::Ogg.to_string(), "Ogg");
    assert!(!ContainerFormat::Ogg.is_matroska_family());
}
