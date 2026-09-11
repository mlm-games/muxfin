use muxfin::api::{AacProfile, AudioCodec, MuxerBuilder, MuxerError};

mod support;
use support::SharedBuffer;

fn valid_adts_frame() -> Vec<u8> {
    vec![
        0xFF, 0xF1, 0x4C, 0x80, 0x01, 0x3F, 0xFC, 0x21, 0x00, 0x49, 0x90, 0x02, 0x19, 0x00, 0x23,
        0x80,
    ]
}

#[test]
fn audio_only_muxer_works() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    muxer.write_audio(0.0, &valid_adts_frame()).unwrap();
    muxer.write_audio(0.023, &valid_adts_frame()).unwrap();
    muxer.finish().unwrap();

    let output = buffer.lock().unwrap().clone();
    assert!(!output.is_empty());
    assert_eq!(&output[4..8], b"ftyp");
    assert!(
        output.windows(4).any(|w| w == b"moov"),
        "moov box not found"
    );
    assert!(
        output.windows(4).any(|w| w == b"trak"),
        "trak box not found"
    );
}

#[test]
fn audio_only_muxer_opus_works() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .audio(AudioCodec::Opus, 48000, 2)
        .build()
        .unwrap();

    // Minimal valid Opus packet: TOC byte with config 0, s=0, code=0
    let opus_packet = vec![0xF8];
    muxer.write_audio(0.0, &opus_packet).unwrap();
    muxer.finish().unwrap();

    let output = buffer.lock().unwrap().clone();
    assert!(!output.is_empty());
    assert_eq!(&output[4..8], b"ftyp");
}

#[test]
fn audio_only_muxer_rejects_video_write() {
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    // Writing video without video track should fail
    let frame = vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e];
    let err = muxer.write_video(0.0, &frame, true).unwrap_err();
    assert!(matches!(err, MuxerError::Io(_)));
}

#[test]
fn audio_only_muxer_rejects_no_config() {
    let (writer, _) = SharedBuffer::new();
    let result = MuxerBuilder::new(writer).build();
    assert!(result.is_err());
    if let Err(err) = result {
        assert!(matches!(err, MuxerError::MissingConfig));
        assert!(err.to_string().contains("configuration"));
    }
}

#[test]
fn audio_only_with_metadata() {
    use muxfin::api::Metadata;

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .with_metadata(Metadata::new().with_title("Audio Only Test"))
        .build()
        .unwrap();

    muxer.write_audio(0.0, &valid_adts_frame()).unwrap();
    muxer.finish().unwrap();

    let output = buffer.lock().unwrap().clone();
    assert!(!output.is_empty());
}

#[test]
fn audio_only_encode_audio_simple() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    muxer.encode_audio(&valid_adts_frame(), 1024).unwrap();
    muxer.encode_audio(&valid_adts_frame(), 1024).unwrap();
    muxer.finish().unwrap();

    let output = buffer.lock().unwrap().clone();
    assert!(!output.is_empty());
}
