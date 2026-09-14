//! Integration tests for native FLAC muxing.
//!
//! Files produced by [`muxfin::api::FlacMuxer`] are validated with the
//! crate's own [`muxfin::demux::demux_flac`] (independent code path:
//! the writer derives totals from queued frames, the demuxer re-derives
//! them from the emitted bytes): STREAMINFO, frame payloads, PTS and
//! totals must round-trip exactly.

use muxfin::api::{AudioCodec, ContainerFormat, MuxerBuilder};
use muxfin::codec::flac::{build_streaminfo, crc16, parse_streaminfo, synthetic_frame_header};
use muxfin::demux::demux_flac;
use muxfin::time::{EncodedSample, SampleTime};

fn streaminfo_44100() -> Vec<u8> {
    build_streaminfo(4096, 4096, 44_100, 2, 16, 0).to_vec()
}

/// One synthetic frame: valid header + filler + footer CRC-16.
/// Fixed blocking, so frame `n` starts at sample `n * 4096`.
fn frame(frame_number: u64, payload_len: usize) -> Vec<u8> {
    let mut data = synthetic_frame_header(frame_number, 4095);
    data.extend(std::iter::repeat_n(0xABu8, payload_len));
    let crc = crc16(&data);
    data.extend_from_slice(&crc.to_be_bytes());
    data
}

#[test]
fn flac_roundtrips_frames_pts_and_totals() {
    let frames = [frame(0, 64), frame(1, 128), frame(2, 32)];
    let mut out = Vec::<u8>::new();
    let stats = {
        let mut muxer = MuxerBuilder::new(&mut out)
            .audio(AudioCodec::Flac, 44_100, 2)
            .with_flac_streaminfo(streaminfo_44100())
            .build_flac()
            .expect("build should succeed");
        assert_eq!(muxer.container(), ContainerFormat::Flac);
        for (i, frame) in frames.iter().enumerate() {
            // PTS in seconds; coded numbers are authoritative.
            muxer
                .write_audio(i as f64 * 4096.0 / 44_100.0, frame)
                .expect("write should succeed");
        }
        muxer.finish_with_stats().expect("finish should succeed")
    };
    assert_eq!(stats.audio_frames, 3);
    assert_eq!(stats.video_frames, 0);
    assert!((stats.duration_secs - 3.0 * 4096.0 / 44_100.0).abs() < 1e-9);
    assert!(stats.bytes_written > 0);

    let track = demux_flac(&out).expect("own demuxer must accept the file");
    assert_eq!(track.streaminfo.sample_rate, 44_100);
    assert_eq!(track.streaminfo.channels, 2);
    assert_eq!(track.streaminfo.bits_per_sample, 16);
    assert_eq!(track.streaminfo.total_samples, 3 * 4096);
    assert_eq!(track.frames.len(), 3);
    for (demuxed, original) in track.frames.iter().zip(frames.iter()) {
        assert_eq!(&demuxed.data, original);
    }
    assert!((track.frames[0].pts - 0.0).abs() < 1e-9);
    assert!((track.frames[1].pts - 4096.0 / 44_100.0).abs() < 1e-9);
    assert!((track.frames[2].pts - 8192.0 / 44_100.0).abs() < 1e-9);
    // Recomputed min/max frame size bound the queued frames.
    let lens: Vec<usize> = frames.iter().map(|f| f.len()).collect();
    let min = *lens.iter().min().unwrap() as u32;
    let max = *lens.iter().max().unwrap() as u32;
    assert_eq!(track.streaminfo.min_frame_size, min);
    assert_eq!(track.streaminfo.max_frame_size, max);
}

#[test]
fn flac_rejects_video_subtitle_non_flac_and_missing_streaminfo() {
    use muxfin::api::VideoCodec;

    let result = MuxerBuilder::new(Vec::<u8>::new())
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .audio(AudioCodec::Flac, 44_100, 2)
        .with_flac_streaminfo(streaminfo_44100())
        .build_flac();
    assert!(result.is_err(), "video must be rejected for FLAC");

    let result = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Opus, 48_000, 2)
        .build_flac();
    assert!(result.is_err(), "Opus must be rejected for FLAC");

    let result = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Flac, 44_100, 2)
        .build_flac();
    assert!(result.is_err(), "missing STREAMINFO must be rejected");

    // Track parameters must agree with STREAMINFO.
    let result = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Flac, 48_000, 2)
        .with_flac_streaminfo(streaminfo_44100())
        .build_flac();
    assert!(result.is_err(), "rate mismatch must be rejected");
}

#[test]
fn flac_rejects_bad_frames_and_backwards_pts() {
    let mut muxer = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Flac, 44_100, 2)
        .with_flac_streaminfo(streaminfo_44100())
        .build_flac()
        .expect("build should succeed");

    assert!(muxer.write_audio(0.0, &[]).is_err(), "empty frame rejected");
    assert!(
        muxer.write_audio(0.0, &[0xFF, 0xF8, 0x00]).is_err(),
        "truncated frame rejected"
    );

    let mut bad = frame(0, 16);
    let last = bad.len() - 1;
    bad[last] ^= 0xFF;
    assert!(
        muxer.write_audio(0.0, &bad).is_err(),
        "footer CRC mismatch rejected"
    );

    muxer
        .write_audio(0.0, &frame(0, 16))
        .expect("first write should succeed");
    assert!(
        muxer.write_audio(0.0, &frame(0, 16)).is_err(),
        "repeated coded number is non-monotonic, rejected"
    );
    muxer
        .write_audio(4096.0 / 44_100.0, &frame(1, 16))
        .expect("forward frame allowed");
    // Backwards PTS rejected even though the coded number repeats.
    let mut muxer2 = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Flac, 44_100, 2)
        .with_flac_streaminfo(streaminfo_44100())
        .build_flac()
        .expect("build should succeed");
    muxer2
        .write_audio(0.02, &frame(0, 16))
        .expect("write should succeed");
    assert!(
        muxer2.write_audio(0.01, &frame(1, 16)).is_err(),
        "backwards PTS rejected"
    );
}

#[test]
fn flac_integer_sample_api() {
    let mut muxer = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Flac, 44_100, 2)
        .with_flac_streaminfo(streaminfo_44100())
        .build_flac()
        .expect("build should succeed");

    let packet = frame(0, 16);
    muxer
        .write_audio_sample(EncodedSample {
            data: &packet,
            timing: SampleTime {
                pts: 0,
                dts: 0,
                duration: 4096,
            },
            is_sync: true,
        })
        .expect("integer write should succeed");
    muxer.finish().expect("finish should succeed");
}

#[test]
fn flac_streaminfo_body_parses_after_finalize() {
    // The emitted STREAMINFO must be well-formed on its own.
    let mut out = Vec::<u8>::new();
    {
        let mut muxer = MuxerBuilder::new(&mut out)
            .audio(AudioCodec::Flac, 44_100, 2)
            .with_flac_streaminfo(streaminfo_44100())
            .build_flac()
            .expect("build should succeed");
        muxer
            .write_audio(0.0, &frame(0, 16))
            .expect("write should succeed");
        muxer.finish().expect("finish should succeed");
    }
    assert_eq!(&out[..4], b"fLaC");
    // First metadata block: not-last STREAMINFO, 34 bytes.
    assert_eq!(out[4], 0x00);
    assert_eq!(&out[5..8], &[0, 0, 34]);
    let info = parse_streaminfo(&out[8..42]).expect("emitted STREAMINFO parses");
    assert_eq!(info.total_samples, 4096);
    assert_eq!(info.min_block_size, 4096);
}
