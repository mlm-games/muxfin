//! Cross-validation with the sibling `dissonia` crate (same author):
//! files muxed by dissonia must demux exactly in muxfin, and remux
//! into MP4. This is the `via dissonia` integration for Ogg Opus and
//! native FLAC.

use std::io::Cursor;

use dissonia::core::audio::{AudioSpec, ChannelLayout, SampleFormat};
use dissonia::core::codecs::{
    CodecId, CodecParameters, CodecSpecific, FlacStreamInfo, OpusStreamMapping,
};
use dissonia::core::formats::{Muxer, TrackSpec};
use dissonia::core::packet::EncodedPacket;
use dissonia::core::units::TimeBase;
use dissonia::{FlacMuxer, OggOpusMuxer};
use muxfin::api::{AudioCodec, MuxerBuilder};
use muxfin::codec::flac::{crc8, crc16};
use muxfin::demux::{demux_flac, demux_ogg_opus};

/// TOC 0xF8: config 31 (CELT-only FB 20 ms = 960 samples), mono.
fn opus_packet() -> Vec<u8> {
    vec![0xF8, 0x11, 0x22, 0x33]
}

#[test]
fn dissonia_ogg_opus_roundtrips_through_muxfin() {
    // ---- Mux with dissonia ------------------------------------------------
    let spec = AudioSpec::new(48_000, ChannelLayout::STEREO, SampleFormat::I16);
    let mut params = CodecParameters::new(CodecId::Opus, spec);
    params.encoder_delay = 312;
    params.codec_specific = Some(CodecSpecific::Opus(OpusStreamMapping::new(
        0,
        1,
        1,
        Box::<[u8]>::default(),
    )));

    let mut muxer = OggOpusMuxer::builder(Cursor::new(Vec::<u8>::new()))
        .serial_number(0x0BAD_F00D)
        .pre_skip(312)
        .build();
    let track = muxer
        .add_track(TrackSpec::new(params, TimeBase::audio_sample_rate(48_000)))
        .expect("add track");
    for _ in 0..3 {
        let mut packet = EncodedPacket::new(opus_packet());
        packet.duration = Some(960);
        muxer.write_packet(track, packet).expect("write packet");
    }
    muxer.finalize().expect("finalize");
    let ogg = muxer.into_inner().into_inner();

    // ---- Demux with muxfin --------------------------------------------------
    let demuxed = demux_ogg_opus(&ogg).expect("muxfin demuxes dissonia ogg");
    assert_eq!(demuxed.channels, 2);
    assert_eq!(demuxed.pre_skip, 312);
    assert_eq!(demuxed.packets.len(), 3);
    assert!((demuxed.packets[0].pts - 0.0).abs() < 1e-9);
    assert!((demuxed.packets[1].pts - 0.02).abs() < 1e-9);
    assert!((demuxed.packets[2].pts - 0.04).abs() < 1e-9);
    assert_eq!(demuxed.packets[0].data, opus_packet());

    // ---- Remux into MP4 ------------------------------------------------------
    let mut mp4 = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Opus, 48_000, 2)
        .with_opus_preskip(demuxed.pre_skip)
        .build()
        .expect("build mp4");
    for packet in &demuxed.packets {
        mp4.write_audio(packet.pts, &packet.data).expect("write");
    }
    mp4.finish().expect("finish");
}

/// Synthetic FLAC frame: fixed blocking, STREAMINFO-described format.
fn flac_frame(frame_no: u64, block_size: u16) -> Vec<u8> {
    let mut hdr = vec![0xFF, 0xF8, 0x70, 0x10, frame_no as u8];
    hdr.extend_from_slice(&(block_size - 1).to_be_bytes());
    let completion = (0..=255u8)
        .find(|&c| {
            hdr.push(c);
            let ok = crc8(&hdr) == 0;
            hdr.pop();
            ok
        })
        .unwrap();
    hdr.push(completion);
    let mut frame = hdr;
    frame.extend_from_slice(&[0x5A; 16]);
    let crc = crc16(&frame);
    frame.extend_from_slice(&crc.to_be_bytes());
    frame
}

#[test]
fn dissonia_flac_roundtrips_through_muxfin() {
    // ---- Mux with dissonia --------------------------------------------------
    let spec = AudioSpec::new(44_100, ChannelLayout::STEREO, SampleFormat::I16);
    let mut params = CodecParameters::new(CodecId::Flac, spec);
    params.codec_specific = Some(CodecSpecific::Flac(FlacStreamInfo {
        min_block_size: 4096,
        max_block_size: 4096,
        min_frame_size: 0,
        max_frame_size: 0,
        bits_per_sample: 16,
        total_samples: 0,
        md5: [0; 16],
    }));

    let mut muxer = FlacMuxer::builder(Cursor::new(Vec::<u8>::new())).build();
    let track = muxer
        .add_track(TrackSpec::new(params, TimeBase::audio_sample_rate(44_100)))
        .expect("add track");
    for frame_no in 0..3 {
        let mut packet = EncodedPacket::new(flac_frame(frame_no, 4096));
        packet.duration = Some(4096);
        muxer.write_packet(track, packet).expect("write frame");
    }
    muxer.finalize().expect("finalize");
    let flac = muxer.into_inner().into_inner();

    // ---- Demux with muxfin ----------------------------------------------------
    let stream = demux_flac(&flac).expect("muxfin demuxes dissonia flac");
    assert_eq!(stream.streaminfo.sample_rate, 44_100);
    assert_eq!(stream.streaminfo.channels, 2);
    assert_eq!(stream.streaminfo.bits_per_sample, 16);
    assert_eq!(stream.frames.len(), 3);
    assert_eq!(stream.frames[0].sample_number, 0);
    assert_eq!(stream.frames[2].sample_number, 8192);

    // ---- Remux into MP4 --------------------------------------------------------
    let mut mp4 = MuxerBuilder::new(Vec::<u8>::new())
        .audio(AudioCodec::Flac, 44_100, 2)
        .with_flac_streaminfo(stream.streaminfo_raw.to_vec())
        .build()
        .expect("build mp4");
    for frame in &stream.frames {
        mp4.write_audio(frame.pts, &frame.data).expect("write");
    }
    mp4.finish().expect("finish");
}
