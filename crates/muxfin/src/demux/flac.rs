//! Native FLAC demuxer (`.flac`).
//!
//! Parses the `fLaC` marker, all metadata blocks (STREAMINFO plus
//! anything else, preserved verbatim for the MP4 `dfLa` box), then
//! scans audio frames. Frame boundaries are validated with the frame
//! footer CRC-16, so false sync positives cannot corrupt timestamps.
//!
//! Timestamps are sample-accurate: variable-blocksize streams carry
//! the sample number directly; fixed-blocksize streams carry the frame
//! number, converted with the block size from the frame header.
//!
//! Reference: FLAC format (IETF draft-ietf-cellar-flac).

use crate::assert_invariant;
use crate::codec::flac::{
    BlockingStrategy, FLAC_MARKER, FlacMetadataBlock, FlacStreaminfo, STREAMINFO_LEN, crc16,
    parse_frame_header, parse_streaminfo, split_metadata_blocks,
};

/// A demuxed FLAC frame with its presentation timestamp.
#[derive(Clone, Debug, PartialEq)]
pub struct FlacFrame {
    /// Presentation timestamp in seconds.
    pub pts: f64,
    /// First sample number of this frame.
    pub sample_number: u64,
    /// Block size in samples.
    pub block_size: u32,
    /// Complete native FLAC frame bytes (ready as one MP4 sample).
    pub data: Vec<u8>,
}

/// A demuxed native FLAC stream.
#[derive(Clone, Debug, PartialEq)]
pub struct FlacStream {
    /// Decoded STREAMINFO.
    pub streaminfo: FlacStreaminfo,
    /// Raw 34-byte STREAMINFO body (for the MP4 `dfLa` / MKV private).
    pub streaminfo_raw: [u8; STREAMINFO_LEN],
    /// All native metadata blocks in order (preserved for `dfLa`).
    pub metadata_blocks: Vec<FlacMetadataBlock>,
    /// Audio frames in decode order with exact PTS.
    pub frames: Vec<FlacFrame>,
}

/// Errors from FLAC demuxing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlacError {
    /// Input is empty.
    Empty,
    /// Missing `fLaC` stream marker.
    BadMarker,
    /// Truncated metadata or frame data.
    Truncated { offset: usize },
    /// No metadata blocks, or first block is not STREAMINFO.
    MissingStreaminfo,
    /// Invalid STREAMINFO contents.
    InvalidStreaminfo,
    /// No audio frames found.
    NoAudioFrames,
    /// Frame header disagrees with STREAMINFO (channels/rate/bits).
    InconsistentFrame { frame: usize },
    /// Frame footer CRC-16 mismatch.
    BadFrameChecksum { frame: usize },
    /// Coded numbers went backwards (corrupt stream).
    NonMonotonic { frame: usize },
}

impl std::fmt::Display for FlacError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlacError::Empty => write!(f, "FLAC input is empty"),
            FlacError::BadMarker => write!(f, "missing fLaC stream marker"),
            FlacError::Truncated { offset } => {
                write!(f, "truncated FLAC data at offset {}", offset)
            }
            FlacError::MissingStreaminfo => {
                write!(f, "FLAC stream has no STREAMINFO metadata block")
            }
            FlacError::InvalidStreaminfo => write!(f, "invalid FLAC STREAMINFO block"),
            FlacError::NoAudioFrames => write!(f, "FLAC stream contains no audio frames"),
            FlacError::InconsistentFrame { frame } => write!(
                f,
                "FLAC frame {} disagrees with STREAMINFO (channels/rate/bits)",
                frame
            ),
            FlacError::BadFrameChecksum { frame } => {
                write!(f, "FLAC frame {} CRC-16 mismatch", frame)
            }
            FlacError::NonMonotonic { frame } => {
                write!(f, "FLAC frame {} coded number went backwards", frame)
            }
        }
    }
}

impl std::error::Error for FlacError {}

/// Scan for the next plausible frame sync at or after `from`.
fn find_sync(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 1 < data.len() {
        if data[i] == 0xFF && (data[i + 1] == 0xF8 || data[i + 1] == 0xF9) {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Demux a native FLAC stream into timestamped frames.
pub fn demux_flac(data: &[u8]) -> Result<FlacStream, FlacError> {
    if data.is_empty() {
        return Err(FlacError::Empty);
    }
    if data.len() < 4 || &data[0..4] != FLAC_MARKER {
        return Err(FlacError::BadMarker);
    }

    // ---- Metadata ---------------------------------------------------------
    let blocks = split_metadata_blocks(&data[4..]).ok_or(FlacError::Truncated { offset: 4 })?;
    let first = blocks.first().ok_or(FlacError::MissingStreaminfo)?;
    if first.block_type != 0 || first.data.len() != STREAMINFO_LEN {
        return Err(FlacError::MissingStreaminfo);
    }
    let mut streaminfo_raw = [0u8; STREAMINFO_LEN];
    streaminfo_raw.copy_from_slice(&first.data);
    let streaminfo = parse_streaminfo(&first.data).ok_or(FlacError::InvalidStreaminfo)?;

    let mut offset = 4;
    for block in &blocks {
        offset += 4 + block.data.len();
    }
    if offset >= data.len() {
        return Err(FlacError::NoAudioFrames);
    }
    let audio = &data[offset..];

    // ---- Frame scan ---------------------------------------------------------
    // Candidate sync offsets with valid headers; each frame runs to the
    // next candidate and is verified by its footer CRC-16.
    let mut syncs = Vec::new();
    let mut pos = 0;
    while let Some(sync) = find_sync(audio, pos) {
        if parse_frame_header(&audio[sync..]).is_some() {
            syncs.push(sync);
            pos = sync + 2;
        } else {
            pos = sync + 1;
        }
    }
    if syncs.is_empty() {
        return Err(FlacError::NoAudioFrames);
    }

    let rate = f64::from(streaminfo.sample_rate);
    let mut frames = Vec::with_capacity(syncs.len());
    let mut prev_end: Option<u64> = None;
    for (frame_idx, window) in syncs.windows(2).enumerate() {
        let (start, end) = (window[0], window[1]);
        frames.push(finish_frame(
            &audio[start..end],
            frame_idx,
            &streaminfo,
            rate,
            &mut prev_end,
        )?);
    }
    // Last frame runs to end of stream.
    let frame_idx = frames.len();
    frames.push(finish_frame(
        &audio[syncs[syncs.len() - 1]..],
        frame_idx,
        &streaminfo,
        rate,
        &mut prev_end,
    )?);

    // INV-703: demuxed FLAC PTS must be non-decreasing.
    let mut prev = -1.0f64;
    for frame in &frames {
        assert_invariant!(
            frame.pts >= prev,
            "INV-703: FLAC frame PTS must be non-decreasing",
            "demux::flac::demux_flac"
        );
        prev = frame.pts;
    }

    Ok(FlacStream {
        streaminfo,
        streaminfo_raw,
        metadata_blocks: blocks,
        frames,
    })
}

/// Validate one frame slice (footer CRC-16 + STREAMINFO consistency) and
/// convert it to a timestamped [`FlacFrame`].
fn finish_frame(
    slice: &[u8],
    frame_idx: usize,
    streaminfo: &FlacStreaminfo,
    rate: f64,
    prev_end: &mut Option<u64>,
) -> Result<FlacFrame, FlacError> {
    if slice.len() < 6 {
        return Err(FlacError::Truncated { offset: 0 });
    }
    let header = parse_frame_header(slice).ok_or(FlacError::Truncated { offset: 0 })?;

    // Fixed-blocksize frames (except possibly the stream tail, whose
    // shortness the CRC already vouches for) must fit STREAMINFO bounds.
    if header.strategy == BlockingStrategy::Fixed
        && streaminfo.max_block_size != 0
        && header.block_size > u32::from(streaminfo.max_block_size)
    {
        return Err(FlacError::InconsistentFrame { frame: frame_idx });
    }

    // Footer CRC-16 covers the whole frame except its own 2 bytes.
    let stored = u16::from_be_bytes([slice[slice.len() - 2], slice[slice.len() - 1]]);
    if crc16(&slice[..slice.len() - 2]) != stored {
        return Err(FlacError::BadFrameChecksum { frame: frame_idx });
    }

    let sample_number = match header.strategy {
        // Variable blocking carries the sample number directly.
        BlockingStrategy::Variable => header.coded_number,
        // Fixed blocking numbers nominal blocks: the stream tail may be
        // shorter than nominal, so scale by STREAMINFO's maximum block
        // size rather than the frame's own (possibly short) size.
        BlockingStrategy::Fixed => {
            let nominal = if streaminfo.max_block_size != 0 {
                u64::from(streaminfo.max_block_size)
            } else {
                u64::from(header.block_size)
            };
            header
                .coded_number
                .checked_mul(nominal)
                .ok_or(FlacError::Truncated { offset: 0 })?
        }
    };
    if let Some(prev) = *prev_end
        && sample_number < prev
    {
        return Err(FlacError::NonMonotonic { frame: frame_idx });
    }
    *prev_end = Some(sample_number + u64::from(header.block_size));

    Ok(FlacFrame {
        pts: sample_number as f64 / rate,
        sample_number,
        block_size: header.block_size,
        data: slice.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::flac::build_streaminfo;
    use crate::codec::flac::synthetic_frame_header;

    /// Build a minimal native FLAC stream: marker + STREAMINFO + frames
    /// with zero payloads and valid CRC-16 footers.
    pub(crate) fn synthetic_flac_stream(
        sample_rate: u32,
        channels: u8,
        block_size: u16,
        frame_count: u64,
    ) -> Vec<u8> {
        let mut data = Vec::from(b"fLaC".as_slice());
        let info = build_streaminfo(block_size, block_size, sample_rate, channels, 16, 0);
        data.extend_from_slice(&[0x80, 0x00, 0x00, 0x22]); // last, type 0, len 34
        data.extend_from_slice(&info);
        for frame_no in 0..frame_count {
            let hdr = synthetic_frame_header(frame_no, block_size - 1);
            // Frame = header + 8 payload bytes + CRC-16.
            let mut frame = hdr;
            frame.extend_from_slice(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
            let crc = crc16(&frame);
            frame.extend_from_slice(&crc.to_be_bytes());
            data.extend_from_slice(&frame);
        }
        data
    }

    #[test]
    fn test_demux_synthetic_stream() {
        // Note: synthetic headers use block code 0x7 (u16 extra), so any
        // u16 block size works; use 4096 @ 44100 Hz stereo.
        let data = synthetic_flac_stream(44_100, 2, 4096, 3);
        let stream = demux_flac(&data).expect("demux");
        assert_eq!(stream.streaminfo.sample_rate, 44_100);
        assert_eq!(stream.streaminfo.channels, 2);
        assert_eq!(stream.frames.len(), 3);
        assert_eq!(stream.frames[0].sample_number, 0);
        assert_eq!(stream.frames[1].sample_number, 4096);
        assert_eq!(stream.frames[2].sample_number, 8192);
        assert!((stream.frames[1].pts - 4096.0 / 44_100.0).abs() < 1e-9);
    }

    #[test]
    fn test_demux_rejects_bad_marker() {
        assert!(matches!(demux_flac(&[]), Err(FlacError::Empty)));
        assert!(matches!(demux_flac(b"OggS...."), Err(FlacError::BadMarker)));
    }

    #[test]
    fn test_demux_rejects_corrupt_frame() {
        let mut data = synthetic_flac_stream(48_000, 2, 4096, 2);
        // Corrupt a payload byte in the first frame (breaks CRC-16).
        let flip_at = data.len() - 20;
        data[flip_at] ^= 0xFF;
        assert!(matches!(
            demux_flac(&data),
            Err(FlacError::BadFrameChecksum { .. }) | Err(FlacError::Truncated { .. })
        ));
    }

    #[test]
    fn test_demux_ignores_false_sync_in_payload() {
        let mut data = synthetic_flac_stream(48_000, 2, 4096, 2);
        // Synthetic frames here are 18 bytes (8 header + 8 payload + 2 CRC).
        // First frame starts after marker(4) + block header(4) + STREAMINFO(34).
        // Overwrite its first two payload bytes with a fake sync (0xFF 0xF9):
        // header parse there fails (bad CRC8), and the frame-slice CRC-16
        // would reject it even if it parsed.
        let frame = &mut data[42..60];
        frame[8] = 0xFF;
        frame[9] = 0xF9;
        let crc = crc16(&frame[..16]);
        frame[16..].copy_from_slice(&crc.to_be_bytes());
        let stream = demux_flac(&data).expect("demux");
        assert_eq!(stream.frames.len(), 2);
    }
}
