//! VP9 video codec support for MP4 muxing.
//!
//! Parsing follows the VP9 bitstream specification ("VP9 Bitstream
//! Superframe and Uncompressed Header", webmproject.org) and the
//! container binding follows "VP Codec ISO Media File Format Binding"
//! (`https://www.webmproject.org/vp9/mp4/`).
//!
//! # Bitstream notes
//!
//! A VP9 frame begins with an **uncompressed header** that is parsed
//! MSB-first at the *bit* level (not byte-aligned):
//!
//! ```text
//! frame_marker         f(2)  == 0b10
//! profile_low          u(1)
//! profile_high         u(1)  -> profile = (high << 1) | low
//! [reserved_zero       f(1)  == 0, only if profile == 3]
//! show_existing_frame  u(1)  -> if 1: frame_to_show u(3), end of header
//! frame_type           u(1)  -> 0 = KEY_FRAME, 1 = INTER_FRAME
//! show_frame           u(1)
//! error_resilient_mode u(1)
//! if KEY_FRAME:
//!   sync_code          u(24) == 0x498342
//!   color_config       (profile-dependent, see below)
//!   frame_width_minus_1  u(16)  -> width = value + 1
//!   frame_height_minus_1 u(16)  -> height = value + 1
//!   render_and_frame_size_different u(1)
//!   [render_width_minus_1 u(16), render_height_minus_1 u(16)]
//! ```
//!
//! The 3-byte sequence `0x49 0x83 0x42` is the **sync code**, which only
//! appears *inside* a keyframe (after the first header bits), never at
//! byte 0. Any parser that expects a frame to *start* with those bytes
//! rejects every real-world VP9 frame (a typical keyframe starts with
//! `0x82`, i.e. marker `10`, profile 0, `show_existing_frame = 0`,
//! `frame_type = 0`).
//!
//! # Container notes
//!
//! Samples are stored as whole VP9 frames (superframes pass through
//! opaquely; only the first frame header is inspected). The sample entry
//! is `vp09` containing a FullBox `vpcC` (version 1):
//!
//! ```text
//! version(8)=1, flags(24)=0
//! profile(8), level(8)
//! bitDepth(4) | chromaSubsampling(3) | videoFullRangeFlag(1)
//! colourPrimaries(8), transferCharacteristics(8), matrixCoefficients(8)
//! codecInitializationDataSize(16) = 0
//! ```

use crate::assert_invariant;

/// VP9 key-frame sync code (`frame_sync_code`, u(24)).
const SYNC_CODE: u32 = 0x49_83_42;

/// Codec configuration carried in the MP4 `vpcC` box, plus the decoded
/// picture size. Field names match the `VPCodecConfigurationRecord`
/// semantics from the VP Codec ISO Media File Format Binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vp9Config {
    /// Decoded picture width in pixels (render size when present).
    pub width: u32,
    /// Decoded picture height in pixels (render size when present).
    pub height: u32,
    /// VP9 profile (0-3).
    pub profile: u8,
    /// VP9 level (e.g. 10 = 1.0, 41 = 4.1). Picture-size lower bound
    /// per Annex A when the true encode level is unknown.
    pub level: u8,
    /// Luma/chroma bit depth (8, 10 or 12).
    pub bit_depth: u8,
    /// Chroma subsampling: 0 = 4:4:0, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4.
    pub chroma_subsampling: u8,
    /// 0 = legal (studio) range, 1 = full range.
    pub video_full_range_flag: u8,
    /// CICP colour primaries.
    pub colour_primaries: u8,
    /// CICP transfer characteristics.
    pub transfer_characteristics: u8,
    /// CICP matrix coefficients (0 = RGB).
    pub matrix_coefficients: u8,
}

/// Errors that can occur during VP9 parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Vp9Error {
    /// Frame data is too short to contain a valid VP9 frame header.
    FrameTooShort,
    /// First two bits are not the `0b10` frame marker.
    InvalidFrameMarker,
    /// Profile-3 reserved bit was set.
    InvalidReservedBit,
    /// Keyframe sync code is not `0x498342`.
    InvalidSyncCode,
    /// Generic parse failure with details.
    ParseError(String),
}

impl std::fmt::Display for Vp9Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Vp9Error::FrameTooShort => write!(f, "VP9 frame too short for header"),
            Vp9Error::InvalidFrameMarker => {
                write!(f, "invalid VP9 frame marker (expected 0b10)")
            }
            Vp9Error::InvalidReservedBit => {
                write!(f, "invalid VP9 reserved bit (profile 3 requires 0)")
            }
            Vp9Error::InvalidSyncCode => write!(f, "invalid VP9 sync code (expected 0x498342)"),
            Vp9Error::ParseError(msg) => write!(f, "VP9 parse error: {}", msg),
        }
    }
}

impl std::error::Error for Vp9Error {}

/// MSB-first bit reader over a byte slice.
struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn remaining_bits(&self) -> usize {
        self.data.len() * 8 - self.bit_pos
    }

    fn read(&mut self, n: u32) -> Result<u32, Vp9Error> {
        if n > 32 {
            return Err(Vp9Error::ParseError("read width > 32".into()));
        }
        if self.remaining_bits() < n as usize {
            return Err(Vp9Error::FrameTooShort);
        }
        let mut value = 0u32;
        for _ in 0..n {
            let byte = self.data[self.bit_pos / 8];
            // MSB-first: bit 0 of a byte is its high bit.
            let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
            value = (value << 1) | u32::from(bit);
            self.bit_pos += 1;
        }
        Ok(value)
    }

    fn skip(&mut self, n: u32) -> Result<(), Vp9Error> {
        self.read(n).map(|_| ())
    }
}

/// What the uncompressed header told us (stops after the fields we need).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeaderKind {
    ShowExisting { index: u8 },
    Key,
    Inter,
}

/// Parse frame marker / profile / show_existing_frame / frame_type.
///
/// Consumes only those bits; the caller continues for keyframes.
fn parse_frame_prefix(r: &mut BitReader<'_>) -> Result<(u8, HeaderKind), Vp9Error> {
    if r.read(2)? != 0b10 {
        return Err(Vp9Error::InvalidFrameMarker);
    }
    let profile_low = r.read(1)?;
    let profile_high = r.read(1)?;
    let profile = ((profile_high << 1) | profile_low) as u8;

    // INV-405: VP9 profile must be valid (0-3)
    assert_invariant!(
        profile <= 3,
        "VP9 profile must be valid (0-3)",
        "codec::vp9::parse_frame_prefix"
    );

    if profile == 3 && r.read(1)? != 0 {
        return Err(Vp9Error::InvalidReservedBit);
    }
    if r.read(1)? == 1 {
        let index = r.read(3)? as u8;
        return Ok((profile, HeaderKind::ShowExisting { index }));
    }
    let frame_type = r.read(1)?;
    r.skip(2)?; // show_frame, error_resilient_mode
    if frame_type == 0 {
        Ok((profile, HeaderKind::Key))
    } else {
        Ok((profile, HeaderKind::Inter))
    }
}

/// Check if a VP9 frame is a keyframe.
///
/// Parses the bit-level uncompressed header: `frame_marker == 0b10`,
/// `show_existing_frame == 0` and `frame_type == 0` (KEY_FRAME).
/// `show_existing_frame == 1` frames only reference a previously decoded
/// frame and are never keyframes.
pub fn is_vp9_keyframe(frame: &[u8]) -> Result<bool, Vp9Error> {
    if frame.is_empty() {
        return Err(Vp9Error::FrameTooShort);
    }
    let mut r = BitReader::new(frame);
    let (_profile, kind) = parse_frame_prefix(&mut r)?;
    match kind {
        HeaderKind::Key => {
            // A keyframe must still carry a valid sync code; reject
            // truncated/corrupt headers instead of misclassifying them.
            if r.remaining_bits() < 24 {
                return Err(Vp9Error::FrameTooShort);
            }
            if r.read(24)? != SYNC_CODE {
                return Err(Vp9Error::InvalidSyncCode);
            }
            Ok(true)
        }
        HeaderKind::ShowExisting { .. } | HeaderKind::Inter => Ok(false),
    }
}

/// VP9 `color_space` enum values (spec section on color config).
const CS_BT_601: u8 = 1;
const CS_BT_709: u8 = 2;
const CS_SMPTE_170: u8 = 3;
const CS_SMPTE_240: u8 = 4;
const CS_BT_2020: u8 = 5;
const CS_RGB: u8 = 7;

/// Color config decoded from a keyframe header.
struct ColorConfig {
    bit_depth: u8,
    subsampling_x: bool,
    subsampling_y: bool,
    full_range: bool,
    colour_primaries: u8,
    transfer_characteristics: u8,
    matrix_coefficients: u8,
}

/// Parse `color_config` for a keyframe of the given profile.
fn parse_color_config(r: &mut BitReader<'_>, profile: u8) -> Result<ColorConfig, Vp9Error> {
    // Bit depth: profiles 0/1 are always 8-bit; profiles 2/3 carry a
    // ten_or_twelve_bit flag (0 = 10-bit, 1 = 12-bit).
    let bit_depth = if profile >= 2 {
        if r.read(1)? == 1 { 12 } else { 10 }
    } else {
        8
    };

    let color_space = r.read(3)? as u8;
    let (subsampling_x, subsampling_y, full_range) = if color_space == CS_RGB {
        // sRGB: full range, 4:4:4. Profiles 1/3 still carry a reserved
        // zero bit here (libvpx `read_bitdepth_colorspace_sampling`).
        if (profile == 1 || profile == 3) && r.read(1)? != 0 {
            return Err(Vp9Error::ParseError("VP9 reserved bit set".into()));
        }
        (false, false, true)
    } else {
        let full_range = r.read(1)? == 1;
        if profile == 1 || profile == 3 {
            let sx = r.read(1)? == 1;
            let sy = r.read(1)? == 1;
            r.skip(1)?; // reserved_zero
            (sx, sy, full_range)
        } else {
            // Profiles 0/2 are always 4:2:0.
            (true, true, full_range)
        }
    };

    let (colour_primaries, transfer_characteristics, matrix_coefficients) =
        cicp_for_color_space(color_space);

    Ok(ColorConfig {
        bit_depth,
        subsampling_x,
        subsampling_y,
        full_range,
        colour_primaries,
        transfer_characteristics,
        matrix_coefficients,
    })
}

/// Lossy but deterministic mapping from the VP9 `color_space` enum to
/// CICP (primaries/transfer/matrix). The VP9 header carries only this
/// enum plus the range flag, so exact CICP values are unrecoverable;
/// callers that know better should override the [`Vp9Config`] fields.
fn cicp_for_color_space(color_space: u8) -> (u8, u8, u8) {
    match color_space {
        CS_BT_601 => (5, 6, 5),    // BT.470BG / BT.601-625
        CS_BT_709 => (1, 1, 1),    // BT.709
        CS_SMPTE_170 => (6, 6, 6), // SMPTE-170M
        CS_SMPTE_240 => (7, 7, 7), // SMPTE-240M
        CS_BT_2020 => (9, 14, 9),  // BT.2020, SDR transfer
        CS_RGB => (2, 2, 0),       // RGB: matrix 0
        _ => (2, 2, 2),            // UNKNOWN / RESERVED: unspecified
    }
}

/// `vpcC` chroma-subsampling code from subsampling flags.
fn chroma_subsampling(sx: bool, sy: bool) -> u8 {
    match (sx, sy) {
        (true, true) => 1,   // 4:2:0 colocated
        (true, false) => 2,  // 4:2:2
        (false, false) => 3, // 4:4:4
        (false, true) => 0,  // 4:4:0 (rare)
    }
}

/// Lowest VP9 level (Annex A) whose `MaxLumaPictureSize` fits
/// `width * height`. Framerate/bitrate are unknown from the header, so
/// this is a picture-size lower bound rather than the true encode level.
fn level_for_resolution(width: u32, height: u32) -> u8 {
    const LEVELS: [(u64, u8); 12] = [
        (36_864, 10),
        (73_728, 11),
        (122_880, 20),
        (245_760, 21),
        (552_960, 30),
        (983_040, 31),
        (2_228_224, 40),
        (3_342_336, 41),
        (8_912_896, 50),
        (8_912_896, 51),
        (35_651_584, 60),
        (35_651_584, 61),
    ];
    let pixels = u64::from(width) * u64::from(height);
    for (max_pixels, level) in LEVELS {
        if pixels <= max_pixels {
            return level;
        }
    }
    61
}

/// Extract VP9 configuration from a keyframe.
///
/// Parses the uncompressed header up to (and including) the render size.
/// Returns `None` for non-keyframes (`show_existing_frame`, inter
/// frames) or malformed input.
pub fn extract_vp9_config(keyframe: &[u8]) -> Option<Vp9Config> {
    if keyframe.is_empty() {
        return None;
    }
    let mut r = BitReader::new(keyframe);
    let (profile, kind) = parse_frame_prefix(&mut r).ok()?;
    if kind != HeaderKind::Key {
        return None;
    }

    // INV-401: the frame marker occupies the first two bits (MSB-first),
    // so it is the top two bits of byte 0. The prefix parse above only
    // succeeds when they equal 0b10; assert on the raw bits directly.
    let marker = (keyframe[0] >> 6) & 0x03;
    assert_invariant!(
        marker == 0b10,
        "INV-401: VP9 frame marker must be 0b10",
        "codec::vp9::extract_vp9_config"
    );
    // INV-402: profile is decoded from two bits, so it is always 0-3;
    // assert on the real decoded value.
    assert_invariant!(
        profile <= 3,
        "INV-402: VP9 profile must be valid (0-3)",
        "codec::vp9::extract_vp9_config"
    );

    if r.read(24).ok()? != SYNC_CODE {
        return None;
    }

    let color = parse_color_config(&mut r, profile).ok()?;

    if r.remaining_bits() < 33 {
        return None;
    }
    let width = r.read(16).ok()? + 1;
    let height = r.read(16).ok()? + 1;
    if width == 0 || height == 0 || width > 65535 || height > 65535 {
        return None;
    }

    let (render_width, render_height) = if r.read(1).ok()? == 1 {
        if r.remaining_bits() < 32 {
            return None;
        }
        (r.read(16).ok()? + 1, r.read(16).ok()? + 1)
    } else {
        (width, height)
    };

    Some(Vp9Config {
        width: render_width,
        height: render_height,
        profile,
        level: level_for_resolution(render_width, render_height),
        bit_depth: color.bit_depth,
        chroma_subsampling: chroma_subsampling(color.subsampling_x, color.subsampling_y),
        video_full_range_flag: u8::from(color.full_range),
        colour_primaries: color.colour_primaries,
        transfer_characteristics: color.transfer_characteristics,
        matrix_coefficients: color.matrix_coefficients,
    })
}

/// Build the 12-byte `vpcC` payload: FullBox header (version 1, flags 0)
/// followed by the 8-byte `VPCodecConfigurationRecord` with
/// `codecInitializationDataSize = 0` (mandatory zero for VP9).
///
/// Shared by the MP4 and fragmented-MP4 writers so the box layout cannot
/// drift between them.
pub fn vpcc_payload(config: &Vp9Config) -> Vec<u8> {
    let mut payload = Vec::with_capacity(12);
    // FullBox: version = 1, flags = 0.
    payload.extend_from_slice(&[1, 0, 0, 0]);
    payload.push(config.profile);
    payload.push(config.level);
    payload.push(
        ((config.bit_depth & 0x0F) << 4)
            | ((config.chroma_subsampling & 0x07) << 1)
            | (config.video_full_range_flag & 0x01),
    );
    payload.push(config.colour_primaries);
    payload.push(config.transfer_characteristics);
    payload.push(config.matrix_coefficients);
    // codecInitializationDataSize (u16) = 0 for VP9; no init data follows.
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload
}

/// Validate that a buffer contains a plausible VP9 frame.
///
/// Accepts keyframes, inter frames and `show_existing_frame` headers;
/// rejects bad markers and truncated headers.
pub fn is_valid_vp9_frame(frame: &[u8]) -> bool {
    if frame.is_empty() {
        return false;
    }
    let mut r = BitReader::new(frame);
    let Ok((_profile, kind)) = parse_frame_prefix(&mut r) else {
        return false;
    };
    match kind {
        HeaderKind::ShowExisting { .. } | HeaderKind::Inter => true,
        HeaderKind::Key => {
            if r.remaining_bits() < 24 {
                return false;
            }
            r.read(24).is_ok_and(|code| code == SYNC_CODE)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-shape VP9 keyframe: profile 0, show_frame, 8-bit, BT.601,
    /// studio range, 4:2:0, 320x240. Bytes after the render-size flag are
    /// irrelevant to the parser.
    const KEYFRAME_320X240: &[u8] = &[0x82, 0x49, 0x83, 0x42, 0x20, 0x13, 0xF0, 0x0E, 0xF0, 0x00];

    #[test]
    fn test_rejects_empty() {
        assert!(matches!(is_vp9_keyframe(&[]), Err(Vp9Error::FrameTooShort)));
        assert!(!is_valid_vp9_frame(&[]));
        assert!(extract_vp9_config(&[]).is_none());
    }

    #[test]
    fn test_rejects_bad_marker() {
        // First two bits 00, not 0b10.
        let bad = [0x00, 0x00, 0x00, 0x00];
        assert!(matches!(
            is_vp9_keyframe(&bad),
            Err(Vp9Error::InvalidFrameMarker)
        ));
        assert!(!is_valid_vp9_frame(&bad));
        assert!(extract_vp9_config(&bad).is_none());
    }

    #[test]
    fn test_rejects_old_style_fake_marker_prefix() {
        // The outdated assumption was that frames *start* with the sync
        // code bytes. Those bytes decode as marker 01 -> invalid.
        let fake = [0x49, 0x83, 0x42, 0x00, 0x00, 0x00];
        assert!(matches!(
            is_vp9_keyframe(&fake),
            Err(Vp9Error::InvalidFrameMarker)
        ));
        assert!(!is_valid_vp9_frame(&fake));
    }

    #[test]
    fn test_parses_real_keyframe() {
        assert_eq!(is_vp9_keyframe(KEYFRAME_320X240), Ok(true));
        assert!(is_valid_vp9_frame(KEYFRAME_320X240));
        let config = extract_vp9_config(KEYFRAME_320X240).expect("config");
        assert_eq!(config.width, 320);
        assert_eq!(config.height, 240);
        assert_eq!(config.profile, 0);
        assert_eq!(config.bit_depth, 8);
        assert_eq!(config.chroma_subsampling, 1);
        assert_eq!(config.video_full_range_flag, 0);
        assert_eq!(config.colour_primaries, 5);
        assert_eq!(config.transfer_characteristics, 6);
        assert_eq!(config.matrix_coefficients, 5);
        assert_eq!(config.level, 20);
    }

    #[test]
    fn test_interframe_is_not_keyframe() {
        // 0b10 marker, profile 0, show_existing 0, frame_type 1 (inter).
        let inter = [0x84, 0x00, 0x00, 0x00];
        assert_eq!(is_vp9_keyframe(&inter), Ok(false));
        assert!(is_valid_vp9_frame(&inter));
        assert!(extract_vp9_config(&inter).is_none());
    }

    #[test]
    fn test_show_existing_is_not_keyframe() {
        // 0b10 marker, profile 0, show_existing 1, index 0.
        let show_existing = [0x88, 0x00, 0x00, 0x00];
        assert_eq!(is_vp9_keyframe(&show_existing), Ok(false));
        assert!(is_valid_vp9_frame(&show_existing));
        assert!(extract_vp9_config(&show_existing).is_none());
    }

    /// Real libvpx keyframe header (ffmpeg `testsrc` 320x240, profile 1,
    /// sRGB): exercises the profile-1 reserved bit after `color_space = 7`.
    /// Trailing bytes are the start of the compressed payload.
    const LIBVPX_KEYFRAME_PREFIX: &[u8] = &[
        0xa2, 0x49, 0x83, 0x42, 0xe0, 0x13, 0xf0, 0x0e, 0xf6, 0x0a, 0x38, 0x24, 0x1c, 0x18, 0x4a,
        0x00,
    ];

    #[test]
    fn test_parses_libvpx_profile1_srgb_keyframe() {
        assert_eq!(is_vp9_keyframe(LIBVPX_KEYFRAME_PREFIX), Ok(true));
        let config = extract_vp9_config(LIBVPX_KEYFRAME_PREFIX).expect("config");
        assert_eq!((config.width, config.height), (320, 240));
        assert_eq!(config.profile, 1);
        assert_eq!(config.bit_depth, 8);
        assert_eq!(config.chroma_subsampling, 3); // sRGB implies 4:4:4
        assert_eq!(config.video_full_range_flag, 1);
        assert_eq!(config.matrix_coefficients, 0); // RGB
    }

    #[test]
    fn test_rejects_bad_sync_code() {
        let mut bad_sync = KEYFRAME_320X240.to_vec();
        bad_sync[1] = 0x00;
        assert!(matches!(
            is_vp9_keyframe(&bad_sync),
            Err(Vp9Error::InvalidSyncCode)
        ));
        assert!(!is_valid_vp9_frame(&bad_sync));
        assert!(extract_vp9_config(&bad_sync).is_none());
    }

    #[test]
    fn test_vpcc_payload_layout() {
        let config = Vp9Config {
            width: 320,
            height: 240,
            profile: 0,
            level: 20,
            bit_depth: 8,
            chroma_subsampling: 1,
            video_full_range_flag: 0,
            colour_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
        };
        let payload = vpcc_payload(&config);
        assert_eq!(payload, vec![1, 0, 0, 0, 0, 20, 0x82, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn test_level_table() {
        assert_eq!(level_for_resolution(100, 100), 10);
        assert_eq!(level_for_resolution(320, 240), 20);
        assert_eq!(level_for_resolution(1920, 1080), 40);
        assert_eq!(level_for_resolution(3840, 2160), 50);
    }

    #[test]
    fn test_chroma_subsampling_codes() {
        assert_eq!(chroma_subsampling(true, true), 1);
        assert_eq!(chroma_subsampling(true, false), 2);
        assert_eq!(chroma_subsampling(false, false), 3);
        assert_eq!(chroma_subsampling(false, true), 0);
    }
}
