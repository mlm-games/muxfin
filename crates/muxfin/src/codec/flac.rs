//! FLAC codec support for muxing.
//!
//! This module provides the minimal FLAC bitstream parsing needed to
//! remux native FLAC into MP4/MKV: STREAMINFO metadata parsing,
//! frame-header parsing (for sample-accurate timestamps), and the
//! CRC helpers used to validate frame boundaries.
//!
//! References:
//! - FLAC format (IETF draft-ietf-cellar-flac, xiph.org/flac/format.html)
//! - "Encapsulation of FLAC in ISO Base Media File Format"
//!   (xiph/flac `doc/isoflac.txt`): one MP4 sample is exactly one
//!   FLAC frame; the `dfLa` box carries native metadata blocks.
//!
//! This module performs no decoding.

use crate::assert_invariant;

/// FLAC stream marker at the start of a native FLAC stream.
pub const FLAC_MARKER: &[u8; 4] = b"fLaC";

/// STREAMINFO metadata block type.
pub const BLOCK_TYPE_STREAMINFO: u8 = 0;
/// STREAMINFO block data length in bytes.
pub const STREAMINFO_LEN: usize = 34;
/// Native metadata block header length in bytes.
pub const METADATA_BLOCK_HEADER_LEN: usize = 4;

/// Maximum FLAC sample rate in Hz (20-bit field).
pub const MAX_SAMPLE_RATE: u32 = 655_350;

/// Decoded FLAC STREAMINFO block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlacStreaminfo {
    /// Minimum block size in samples (16-bit).
    pub min_block_size: u16,
    /// Maximum block size in samples (16-bit).
    pub max_block_size: u16,
    /// Minimum frame size in bytes (24-bit, 0 = unknown).
    pub min_frame_size: u32,
    /// Maximum frame size in bytes (24-bit, 0 = unknown).
    pub max_frame_size: u32,
    /// Sample rate in Hz (20-bit).
    pub sample_rate: u32,
    /// Number of channels (stored as channels - 1 in 3 bits).
    pub channels: u8,
    /// Bits per sample (stored as bps - 1 in 5 bits).
    pub bits_per_sample: u8,
    /// Total samples in the stream (36-bit, 0 = unknown).
    pub total_samples: u64,
    /// MD5 of the unencoded audio data (16 bytes).
    pub md5: [u8; 16],
}

/// A native FLAC metadata block (header + raw block data).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlacMetadataBlock {
    /// Last-metadata-block flag.
    pub last: bool,
    /// Block type (7 bits); 0 = STREAMINFO.
    pub block_type: u8,
    /// Raw block data.
    pub data: Vec<u8>,
}

/// Blocking strategy signalled by a frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockingStrategy {
    /// Fixed block size; coded number is a frame number.
    Fixed,
    /// Variable block size; coded number is a sample number.
    Variable,
}

/// Parsed FLAC frame header (without the payload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlacFrameHeader {
    /// Fixed vs variable blocking strategy.
    pub strategy: BlockingStrategy,
    /// Frame number (fixed) or sample number (variable).
    pub coded_number: u64,
    /// Block size in samples (inter-channel).
    pub block_size: u32,
    /// Header length in bytes (including the CRC8 byte).
    pub header_len: usize,
}

/// Parse a 34-byte STREAMINFO block body. Returns `None` on invalid data.
pub fn parse_streaminfo(data: &[u8]) -> Option<FlacStreaminfo> {
    if data.len() < STREAMINFO_LEN {
        return None;
    }
    let min_block_size = u16::from_be_bytes([data[0], data[1]]);
    let max_block_size = u16::from_be_bytes([data[2], data[3]]);
    let min_frame_size =
        (u32::from(data[4]) << 16) | (u32::from(data[5]) << 8) | u32::from(data[6]);
    let max_frame_size =
        (u32::from(data[7]) << 16) | (u32::from(data[8]) << 8) | u32::from(data[9]);

    // INV-601: STREAMINFO minimum block size must be valid (16..=65535).
    // A value of 0 is only legal when max is also 0 (unknown); the muxer
    // requires a usable block size, so reject fully-unknown streams here.
    assert_invariant!(
        min_block_size >= 16 || max_block_size == 0,
        "INV-601: FLAC STREAMINFO block size must be valid",
        "codec::flac::parse_streaminfo"
    );
    if min_block_size < 16 && max_block_size != 0 {
        return None;
    }

    let sample_rate =
        (u32::from(data[10]) << 12) | (u32::from(data[11]) << 4) | (u32::from(data[12]) >> 4);
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return None;
    }
    let channels = ((data[12] >> 1) & 0x07) + 1;
    let bits_per_sample = ((data[12] & 0x01) << 4 | (data[13] >> 4)) + 1;
    if !(4..=32).contains(&bits_per_sample) {
        return None;
    }
    let total_samples = (u64::from(data[13] & 0x0F) << 32)
        | (u64::from(data[14]) << 24)
        | (u64::from(data[15]) << 16)
        | (u64::from(data[16]) << 8)
        | u64::from(data[17]);
    let mut md5 = [0u8; 16];
    md5.copy_from_slice(&data[18..34]);

    Some(FlacStreaminfo {
        min_block_size,
        max_block_size,
        min_frame_size,
        max_frame_size,
        sample_rate,
        channels,
        bits_per_sample,
        total_samples,
        md5,
    })
}

/// Build a 34-byte STREAMINFO block body (used by builders and tests).
#[allow(clippy::too_many_arguments)]
pub fn build_streaminfo(
    min_block_size: u16,
    max_block_size: u16,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    total_samples: u64,
) -> [u8; STREAMINFO_LEN] {
    let mut out = [0u8; STREAMINFO_LEN];
    out[0..2].copy_from_slice(&min_block_size.to_be_bytes());
    out[2..4].copy_from_slice(&max_block_size.to_be_bytes());
    // min/max frame size unknown (0).
    out[10] = ((sample_rate >> 12) & 0xFF) as u8;
    out[11] = ((sample_rate >> 4) & 0xFF) as u8;
    out[12] = (((sample_rate & 0x0F) << 4)
        | (u32::from(channels - 1) << 1)
        | (u32::from(bits_per_sample - 1) >> 4)) as u8;
    out[13] = (((u32::from(bits_per_sample - 1) & 0x0F) << 4)
        | ((total_samples >> 32) as u32 & 0x0F)) as u8;
    out[14..18].copy_from_slice(&(total_samples as u32).to_be_bytes());
    out
}

/// Split native metadata blocks. Input starts at the first block header
/// (after the `fLaC` marker). Returns the blocks in order.
pub fn split_metadata_blocks(data: &[u8]) -> Option<Vec<FlacMetadataBlock>> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    loop {
        if offset + METADATA_BLOCK_HEADER_LEN > data.len() {
            return None;
        }
        let last = data[offset] & 0x80 != 0;
        let block_type = data[offset] & 0x7F;
        let len = (usize::from(data[offset + 1]) << 16)
            | (usize::from(data[offset + 2]) << 8)
            | usize::from(data[offset + 3]);
        offset += METADATA_BLOCK_HEADER_LEN;
        if offset + len > data.len() {
            return None;
        }
        blocks.push(FlacMetadataBlock {
            last,
            block_type,
            data: data[offset..offset + len].to_vec(),
        });
        offset += len;
        if last {
            break;
        }
        // Sanity cap: metadata is tiny; bail on corrupt chains.
        if blocks.len() > 64 {
            return None;
        }
    }
    Some(blocks)
}

/// CRC-8 with polynomial x^8 + x^2 + x + 1 (FLAC frame-header check).
pub fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// CRC-16 with polynomial x^16 + x^15 + x^2 + 1 (FLAC frame footer).
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Decode a UTF-8-style coded number. Returns (value, bytes consumed).
fn decode_utf8_coded_number(data: &[u8]) -> Option<(u64, usize)> {
    let first = *data.first()?;
    let (value, extra) = if first & 0x80 == 0 {
        (u64::from(first & 0x7F), 0)
    } else if first & 0xE0 == 0xC0 {
        (u64::from(first & 0x1F), 1)
    } else if first & 0xF0 == 0xE0 {
        (u64::from(first & 0x0F), 2)
    } else if first & 0xF8 == 0xF0 {
        (u64::from(first & 0x07), 3)
    } else if first & 0xFC == 0xF8 {
        (u64::from(first & 0x03), 4)
    } else if first & 0xFE == 0xFC {
        (u64::from(first & 0x01), 5)
    } else if first == 0xFE {
        (0, 6)
    } else {
        return None;
    };
    if data.len() < 1 + extra {
        return None;
    }
    let mut value = value;
    for &byte in &data[1..1 + extra] {
        if byte & 0xC0 != 0x80 {
            return None;
        }
        value = (value << 6) | u64::from(byte & 0x3F);
    }
    Some((value, 1 + extra))
}

/// Block size in samples from the 4-bit header field.
fn block_size_from_code(code: u8, extra: &[u8]) -> Option<(u32, usize)> {
    match code {
        0 => None, // reserved
        1 => Some((192, 0)),
        2..=5 => Some((576 * (1 << (u32::from(code) - 2)), 0)),
        6 => {
            let b = usize::from(*extra.first()?);
            Some((b as u32 + 1, 1))
        }
        7 => {
            if extra.len() < 2 {
                return None;
            }
            let v = (u32::from(extra[0]) << 8) | u32::from(extra[1]);
            Some((v + 1, 2))
        }
        8..=15 => Some((256 * (1 << (u32::from(code) - 8)), 0)),
        _ => None,
    }
}

/// Parse a FLAC frame header at the start of `data`.
///
/// Returns the header on success. The header CRC8 is verified, which
/// keeps false sync positives out of the demuxer.
pub fn parse_frame_header(data: &[u8]) -> Option<FlacFrameHeader> {
    if data.len() < 4 {
        return None;
    }
    if data[0] != 0xFF || (data[1] != 0xF8 && data[1] != 0xF9) {
        return None;
    }
    let strategy = if data[1] == 0xF8 {
        BlockingStrategy::Fixed
    } else {
        BlockingStrategy::Variable
    };

    let block_code = (data[2] >> 4) & 0x0F;
    let rate_code = data[2] & 0x0F;
    let channel_code = (data[3] >> 4) & 0x0F;
    let bps_code = (data[3] >> 1) & 0x07;
    // Reserved bit must be zero.
    if data[3] & 0x01 != 0 {
        return None;
    }
    // Reserved channel assignments and bit depths are invalid.
    if channel_code >= 0x0B && channel_code != 0x08 {
        // 0x08-0x0A are side/mid-side codings (still valid frames);
        // 0x0B-0x0F are reserved.
        if channel_code > 0x0A {
            return None;
        }
    }
    if bps_code == 0x03 || bps_code == 0x07 {
        return None;
    }
    if rate_code == 0x0F {
        return None;
    }

    let (coded_number, number_len) = decode_utf8_coded_number(&data[4..])?;
    let mut offset = 4 + number_len;

    // Block size (may consume extra bytes).
    let (block_size, block_extra) = block_size_from_code(block_code, &data[offset..])?;
    offset += block_extra;

    // Sample rate extra bytes (values validated by the demuxer against
    // STREAMINFO only when the code carries an explicit rate).
    offset += match rate_code {
        0x0C => 1,
        0x0D | 0x0E => 2,
        _ => 0,
    };
    if offset + 1 > data.len() {
        return None;
    }

    // Header CRC8 covers everything up to and including itself.
    let header_bytes = &data[..offset + 1];
    if crc8(header_bytes) != 0 {
        return None;
    }

    Some(FlacFrameHeader {
        strategy,
        coded_number,
        block_size,
        header_len: offset + 1,
    })
}

/// Check whether `data` starts with a plausible FLAC frame.
pub fn is_valid_flac_frame(data: &[u8]) -> bool {
    parse_frame_header(data).is_some()
}

/// Build a minimal synthetic FLAC frame header for tests:
/// fixed blocking, given frame number, block size via code 0x7 + u16
/// extra, rate from STREAMINFO, stereo independent, 16-bit from STREAMINFO.
#[cfg(test)]
pub(crate) fn synthetic_frame_header(coded_number: u64, block_extra: u16) -> Vec<u8> {
    let mut hdr = vec![0xFF, 0xF8, 0x70, 0x10];
    // UTF-8 coded number (small values fit in 1-2 bytes).
    if coded_number < 0x80 {
        hdr.push(coded_number as u8);
    } else {
        hdr.push(0xC0 | ((coded_number >> 6) & 0x1F) as u8);
        hdr.push(0x80 | (coded_number & 0x3F) as u8);
    }
    hdr.extend_from_slice(&block_extra.to_be_bytes());
    // The stored CRC8 byte completes the header so the running CRC
    // over all header bytes (including itself) is zero.
    let completion = (0..=255u8).find(|&candidate| {
        hdr.push(candidate);
        let ok = crc8(&hdr) == 0;
        hdr.pop();
        ok
    });
    hdr.push(completion.expect("a CRC completion byte always exists"));
    hdr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streaminfo_44100_stereo() -> [u8; STREAMINFO_LEN] {
        build_streaminfo(4096, 4096, 44_100, 2, 16, 0)
    }

    #[test]
    fn test_streaminfo_roundtrip() {
        let raw = streaminfo_44100_stereo();
        let info = parse_streaminfo(&raw).expect("parse");
        assert_eq!(info.min_block_size, 4096);
        assert_eq!(info.max_block_size, 4096);
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
    }

    #[test]
    fn test_streaminfo_rejects_bad_rate() {
        let mut raw = streaminfo_44100_stereo();
        raw[10] = 0xFF;
        raw[11] = 0xFF;
        raw[12] = 0xF0; // rate with top bits set -> > max
        assert!(parse_streaminfo(&raw).is_none());
    }

    #[test]
    fn test_metadata_block_split() {
        // STREAMINFO (last=0) + VORBIS_COMMENT (last=1).
        let mut data = vec![0x00, 0x00, 0x00, 0x22];
        data.extend_from_slice(&[0u8; 34]);
        data.extend_from_slice(&[0x84, 0x00, 0x00, 0x04, 0x01, 0x02, 0x03, 0x04]);
        let blocks = split_metadata_blocks(&data).expect("split");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_type, 0);
        assert!(!blocks[0].last);
        assert_eq!(blocks[1].block_type, 4);
        assert!(blocks[1].last);
    }

    #[test]
    fn test_crc_vectors() {
        // Check values for ASCII "123456789": CRC-8/SMBUS (poly 0x07,
        // init 0, no reflection) and the non-reflected CRC-16 with
        // poly 0x8005 used by FLAC frame footers.
        assert_eq!(crc8(b"123456789"), 0xF4);
        assert_eq!(crc16(b"123456789"), 0xFEE8);
    }

    #[test]
    fn test_frame_header_parse() {
        let hdr = synthetic_frame_header(0, 4095);
        let parsed = parse_frame_header(&hdr).expect("parse");
        assert_eq!(parsed.strategy, BlockingStrategy::Fixed);
        assert_eq!(parsed.coded_number, 0);
        assert_eq!(parsed.block_size, 4096);
        assert!(is_valid_flac_frame(&hdr));
        assert!(!is_valid_flac_frame(&[0xFF, 0xF8, 0x00]));
    }
}
