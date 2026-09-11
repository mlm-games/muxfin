//! Integration tests for audio demuxing (Ogg Opus, native FLAC) and
//! FLAC output support in MP4/MKV.
//!
//! All vectors are built in-test (no external files): the Ogg page
//! builder below is an independent implementation, so these tests also
//! cross-check the library's CRC and framing against a second source.

mod support;

use muxfin::api::{AudioCodec, ContainerFormat, MuxerBuilder};
use muxfin::codec::flac::{STREAMINFO_LEN, build_streaminfo, crc8, crc16, parse_streaminfo};
use muxfin::demux::{demux_flac, demux_ogg_opus};
use support::SharedBuffer;

// ---------------------------------------------------------------------------
// Minimal Ogg builder (independent of the library implementation).
// ---------------------------------------------------------------------------

fn ogg_crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut r = (i as u32) << 24;
        for _ in 0..8 {
            r = if r & 0x8000_0000 != 0 {
                (r << 1) ^ 0x04C1_1DB7
            } else {
                r << 1
            };
        }
        *entry = r;
    }
    let mut crc = 0u32;
    for &byte in data {
        crc = (crc << 8) ^ table[(((crc >> 24) ^ u32::from(byte)) & 0xFF) as usize];
    }
    crc
}

fn ogg_page(header_type: u8, granule: u64, serial: u32, seq: u32, packets: &[&[u8]]) -> Vec<u8> {
    let mut segments = Vec::new();
    let mut payload = Vec::new();
    for packet in packets {
        let mut rest = *packet;
        loop {
            let take = rest.len().min(255);
            segments.push(take as u8);
            payload.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if take < 255 {
                break;
            }
        }
    }
    let mut page = Vec::new();
    page.extend_from_slice(b"OggS");
    page.push(0);
    page.push(header_type);
    page.extend_from_slice(&granule.to_le_bytes());
    page.extend_from_slice(&serial.to_le_bytes());
    page.extend_from_slice(&seq.to_le_bytes());
    page.extend_from_slice(&[0, 0, 0, 0]);
    page.push(segments.len() as u8);
    page.extend_from_slice(&segments);
    page.extend_from_slice(&payload);
    let crc = ogg_crc32(&page);
    page[22..26].copy_from_slice(&crc.to_le_bytes());
    page
}

fn opus_head(channels: u8, pre_skip: u16) -> Vec<u8> {
    let mut p = Vec::from(b"OpusHead".as_slice());
    p.push(1);
    p.push(channels);
    p.extend_from_slice(&pre_skip.to_le_bytes());
    p.extend_from_slice(&48_000u32.to_le_bytes());
    p.extend_from_slice(&0i16.to_le_bytes());
    p.push(0);
    p
}

/// TOC 0xF8: config 31 (CELT-only FB 20 ms = 960 samples), mono.
fn opus_packet() -> Vec<u8> {
    vec![0xF8, 0x11, 0x22, 0x33]
}

fn test_ogg_opus() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(ogg_page(0x02, 0, 0x42, 0, &[&opus_head(2, 312)]));
    data.extend(ogg_page(0x00, 0, 0x42, 1, &[b"OpusTags...."]));
    data.extend(ogg_page(
        0x00,
        312 + 2 * 960,
        0x42,
        2,
        &[&opus_packet(), &opus_packet()],
    ));
    data.extend(ogg_page(0x04, 312 + 3 * 960, 0x42, 3, &[&opus_packet()]));
    data
}

// ---------------------------------------------------------------------------
// Minimal FLAC builder.
// ---------------------------------------------------------------------------

fn flac_frame(frame_no: u64, block_size: u16) -> Vec<u8> {
    // Fixed blocking, rate/channels/bps from STREAMINFO, u16 block size.
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

fn test_flac() -> Vec<u8> {
    let mut data = Vec::from(b"fLaC".as_slice());
    // STREAMINFO: last=1, type=0, len=34; 4096-block, 44100 Hz stereo 16-bit.
    data.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]);
    data.extend_from_slice(&build_streaminfo(4096, 4096, 44_100, 2, 16, 0));
    for frame_no in 0..4 {
        data.extend(flac_frame(frame_no, 4096));
    }
    data
}

fn find_box(data: &[u8], fourcc: &[u8; 4]) -> Option<(u32, usize)> {
    data.windows(4).position(|w| w == fourcc).and_then(|pos| {
        if pos < 4 {
            return None;
        }
        let size = u32::from_be_bytes(data[pos - 4..pos].try_into().ok()?);
        Some((size, pos + 4))
    })
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn ogg_opus_demux_gives_exact_pts() {
    let track = demux_ogg_opus(&test_ogg_opus()).expect("demux");
    assert_eq!(track.channels, 2);
    assert_eq!(track.pre_skip, 312);
    assert_eq!(track.packets.len(), 3);
    assert!((track.packets[0].pts - 0.0).abs() < 1e-9);
    assert!((track.packets[1].pts - 0.02).abs() < 1e-9);
    assert!((track.packets[2].pts - 0.04).abs() < 1e-9);
}

#[test]
fn ogg_vorbis_is_rejected_with_actionable_error() {
    let mut data = Vec::new();
    data.extend(ogg_page(
        0x02,
        0,
        0x99,
        0,
        &[b"\x01vorbis.................."],
    ));
    let err = demux_ogg_opus(&data).unwrap_err();
    assert!(err.to_string().contains("vorbis"), "got: {}", err);
    assert!(err.to_string().contains("MP4"), "got: {}", err);
}

#[test]
fn flac_demux_gives_sample_accurate_pts() {
    let stream = demux_flac(&test_flac()).expect("demux");
    assert_eq!(stream.streaminfo.sample_rate, 44_100);
    assert_eq!(stream.streaminfo.channels, 2);
    assert_eq!(stream.frames.len(), 4);
    assert_eq!(stream.frames[0].sample_number, 0);
    assert_eq!(stream.frames[3].sample_number, 3 * 4096);
    assert!((stream.frames[2].pts - 8192.0 / 44_100.0).abs() < 1e-9);
}

#[test]
fn ogg_opus_remuxes_to_mp4() {
    let track = demux_ogg_opus(&test_ogg_opus()).expect("demux");
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .audio(AudioCodec::Opus, 48_000, u16::from(track.channels))
        .build()
        .unwrap();
    for packet in &track.packets {
        muxer.write_audio(packet.pts, &packet.data).unwrap();
    }
    muxer.finish().unwrap();
    let output = buffer.lock().unwrap();
    assert!(output.windows(4).any(|w| w == b"Opus"));
    assert!(output.windows(4).any(|w| w == b"dOps"));
}

#[test]
fn flac_remuxes_to_mp4_with_spec_layout() {
    let stream = demux_flac(&test_flac()).expect("demux");
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .audio(AudioCodec::Flac, 44_100, 2)
        .with_flac_streaminfo(stream.streaminfo_raw.to_vec())
        .build()
        .unwrap();
    for frame in &stream.frames {
        muxer.write_audio(frame.pts, &frame.data).unwrap();
    }
    muxer.finish().unwrap();

    let output = buffer.lock().unwrap();
    assert!(output.windows(4).any(|w| w == b"fLaC"));
    let (size, payload_off) = find_box(&output, b"dfLa").expect("dfLa present");
    // FullBox header (4) + one metadata block header (4) + STREAMINFO (34).
    assert_eq!(size, 8 + 4 + 4 + STREAMINFO_LEN as u32);
    let payload = &output[payload_off..payload_off + 8 + STREAMINFO_LEN];
    assert_eq!(&payload[0..4], &[0, 0, 0, 0]); // version 0, flags 0
    assert_eq!(payload[4], 0x80); // last=1
    assert_eq!(payload[5], 0); // type STREAMINFO
    assert_eq!(&payload[6..8], &[0, 34]);
    // STREAMINFO echo must parse and match the source.
    let echoed = parse_streaminfo(&payload[8..8 + STREAMINFO_LEN]).expect("echo parses");
    assert_eq!(echoed.sample_rate, 44_100);
    assert_eq!(echoed.channels, 2);
    assert_eq!(echoed.bits_per_sample, 16);
}

#[test]
fn flac_remuxes_to_mkv() {
    let stream = demux_flac(&test_flac()).expect("demux");
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .audio(AudioCodec::Flac, 44_100, 2)
        .with_flac_streaminfo(stream.streaminfo_raw.to_vec())
        .build_mkv()
        .unwrap();
    for frame in &stream.frames {
        muxer.write_audio(frame.pts, &frame.data).unwrap();
    }
    muxer.finish().unwrap();
    let output = buffer.lock().unwrap();
    // EBML magic + CodecPrivate echo of the 34-byte STREAMINFO.
    assert_eq!(&output[0..4], &[0x1A, 0x45, 0xDF, 0xA3]);
    let raw = stream.streaminfo_raw;
    assert!(
        output.windows(34).any(|w| w == raw),
        "STREAMINFO must be embedded as A_FLAC CodecPrivate"
    );
}

#[test]
fn flac_requires_streaminfo() {
    let (writer, _) = SharedBuffer::new();
    let result = MuxerBuilder::new(writer)
        .audio(AudioCodec::Flac, 44_100, 2)
        .build();
    let err = match result {
        Ok(_) => panic!("expected missing-STREAMINFO error"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("with_flac_streaminfo"), "got: {err}");
}

#[test]
fn flac_rejects_mismatched_params() {
    let stream = demux_flac(&test_flac()).expect("demux");
    let (writer, _) = SharedBuffer::new();
    // STREAMINFO says stereo; claim mono.
    let result = MuxerBuilder::new(writer)
        .audio(AudioCodec::Flac, 44_100, 1)
        .with_flac_streaminfo(stream.streaminfo_raw.to_vec())
        .build();
    let err = match result {
        Ok(_) => panic!("expected parameter-mismatch error"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("disagree"), "got: {err}");
}

#[test]
fn flac_rejected_in_webm() {
    let stream = demux_flac(&test_flac()).expect("demux");
    let (writer, _) = SharedBuffer::new();
    let result = MuxerBuilder::new(writer)
        .audio(AudioCodec::Flac, 44_100, 2)
        .with_flac_streaminfo(stream.streaminfo_raw.to_vec())
        .with_container(ContainerFormat::WebM)
        .build_mkv();
    let err = match result {
        Ok(_) => panic!("expected WebM rejection"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("not supported in"), "got: {err}");
}

#[test]
fn flac_codec_name_parses() {
    assert_eq!("flac".parse::<AudioCodec>().unwrap(), AudioCodec::Flac);
    assert_eq!(AudioCodec::Flac.to_string(), "FLAC");
}
