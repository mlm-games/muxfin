mod support;

use muxfin::api::{MuxerBuilder, VideoCodec};
use support::SharedBuffer;

/// Build a minimal VP9 keyframe with a spec-compliant uncompressed header.
///
/// Bit layout (MSB-first) per the VP9 bitstream specification:
/// frame_marker=0b10, profile=0, show_existing_frame=0, frame_type=0 (KEY),
/// show_frame=1, error_resilient=0, sync_code=0x498342, color_space=1 (BT.601),
/// studio range, 4:2:0, width-1/height-1 as u16, render size = frame size.
fn build_vp9_keyframe() -> Vec<u8> {
    // 100x100, profile 0. Bytes derived from the bit layout above
    // (verified against the moq-mux 320x240 golden vector construction).
    let mut data = vec![0x82, 0x49, 0x83, 0x42, 0x20, 0x06, 0x30, 0x06, 0x30];
    // Minimal compressed payload (opaque to the muxer).
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    data
}

/// Build a minimal VP9 inter frame.
///
/// `0x84` = marker `0b10`, profile 0, show_existing_frame=0,
/// frame_type=1 (INTER), show_frame=0, error_resilient=0.
fn build_vp9_pframe() -> Vec<u8> {
    vec![0x84, 0x00, 0x00, 0x00]
}

/// Recursively search for a 4CC in an MP4 container by pattern matching
fn contains_box(data: &[u8], fourcc: &[u8; 4]) -> bool {
    data.windows(4).any(|window| window == fourcc)
}

/// Locate a box payload: returns (size, payload_offset) of first occurrence.
fn find_box(data: &[u8], fourcc: &[u8; 4]) -> Option<(u32, usize)> {
    data.windows(4).position(|w| w == fourcc).and_then(|pos| {
        let size = u32::from_be_bytes(data[pos - 4..pos].try_into().ok()?);
        Some((size, pos + 4))
    })
}

#[test]
fn vp9_first_frame_must_be_keyframe() {
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .build()
        .unwrap();

    // Try to write a P-frame first - should fail
    let pframe = build_vp9_pframe();
    let result = muxer.write_video(0.0, &pframe, false);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("first video frame must be a keyframe")
    );
}

#[test]
fn vp9_keyframe_must_have_valid_config() {
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .build()
        .unwrap();

    // Try to write an invalid VP9 frame (bad frame marker: 00, not 0b10)
    let invalid_frame = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let result = muxer.write_video(0.0, &invalid_frame, true);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("first VP9 frame must contain sequence parameters")
    );
}

#[test]
fn vp9_muxer_produces_vp09_sample_entry() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .build()
        .unwrap();

    // Write keyframe
    let keyframe = build_vp9_keyframe();
    muxer.write_video(0.0, &keyframe, true).unwrap();

    // Write P-frame
    let pframe = build_vp9_pframe();
    muxer.write_video(1.0 / 30.0, &pframe, false).unwrap();

    // Finalize
    muxer.finish().unwrap();

    // Check output contains vp09 sample entry
    let output = buffer.lock().unwrap();
    assert!(contains_box(&output, b"vp09"));
}

#[test]
fn vp9_muxer_produces_spec_compliant_vpcc() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .build()
        .unwrap();

    // Write keyframe
    let keyframe = build_vp9_keyframe();
    muxer.write_video(0.0, &keyframe, true).unwrap();

    // Finalize
    muxer.finish().unwrap();

    // Check output contains vpcC configuration box with the binding layout:
    // FullBox(version=1, flags=0) + profile + level +
    // packed(bitDepth/chroma/fullRange) + primaries + transfer + matrix +
    // codecInitDataSize(u16)=0  => 12-byte payload.
    let output = buffer.lock().unwrap();
    let (size, payload_off) = find_box(&output, b"vpcC").expect("vpcC present");
    assert_eq!(size, 8 + 12, "vpcC box must be 20 bytes total");
    let payload = &output[payload_off..payload_off + 12];
    // FullBox header
    assert_eq!(&payload[0..4], &[1, 0, 0, 0]);
    // profile 0
    assert_eq!(payload[4], 0);
    // level: 100x100 -> picture-size lower bound L1.0 = 10
    assert_eq!(payload[5], 10);
    // bitDepth=8, chroma=1 (4:2:0), fullRange=0 -> 0x82
    assert_eq!(payload[6], 0x82);
    // BT.601 mapping: primaries=5, transfer=6, matrix=5
    assert_eq!(&payload[7..10], &[5, 6, 5]);
    // codecInitializationDataSize = 0
    assert_eq!(&payload[10..12], &[0, 0]);
}

#[test]
fn vp9_config_extraction_works() {
    use muxfin::codec::vp9::extract_vp9_config;

    let keyframe = build_vp9_keyframe();
    let config = extract_vp9_config(&keyframe).unwrap();

    assert_eq!(config.width, 100);
    assert_eq!(config.height, 100);
    assert_eq!(config.profile, 0);
    assert_eq!(config.bit_depth, 8);
    assert_eq!(config.chroma_subsampling, 1); // 4:2:0
    assert_eq!(config.video_full_range_flag, 0);
    assert_eq!(config.colour_primaries, 5); // BT.601
    assert_eq!(config.transfer_characteristics, 6);
    assert_eq!(config.matrix_coefficients, 5);
}

#[test]
fn vp9_keyframe_detection_works() {
    use muxfin::codec::vp9::is_vp9_keyframe;

    let keyframe = build_vp9_keyframe();
    assert!(is_vp9_keyframe(&keyframe).unwrap());

    let pframe = build_vp9_pframe();
    assert!(!is_vp9_keyframe(&pframe).unwrap());
}

#[test]
fn vp9_rejects_legacy_fake_marker_prefix() {
    // Regression test for the outdated assumption that frames start with
    // the sync-code bytes 0x49 0x83 0x42: those bits decode as frame
    // marker 01, which is invalid. Real keyframes start with 0x82.
    use muxfin::codec::vp9::{is_valid_vp9_frame, is_vp9_keyframe};

    let fake = [0x49, 0x83, 0x42, 0x00, 0x00, 0x00];
    assert!(is_vp9_keyframe(&fake).is_err());
    assert!(!is_valid_vp9_frame(&fake));
}
