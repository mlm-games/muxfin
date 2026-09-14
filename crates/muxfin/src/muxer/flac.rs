//! Native FLAC muxing (`.flac`, IETF draft-ietf-cellar-flac).
//!
//! [`FlacWriter`] queues native FLAC frames and emits a complete file on
//! [`finalize`](FlacWriter::finalize): the `fLaC` marker, a STREAMINFO
//! block with exact totals, a `VORBIS_COMMENT` block, then the frames
//! verbatim. The public frontend is [`crate::api::FlacMuxer`], built with
//! [`crate::api::MuxerBuilder::build_flac`].
//!
//! Unlike a seek-based writer, everything is computed from the queued
//! frames at finalize time, so the writer needs only [`std::io::Write`]:
//!
//! - total samples: summed from the frames' coded numbers (same fixed /
//!   variable blocking arithmetic as the demuxer),
//! - min/max frame size: measured from the queued frame bytes,
//! - MD5: zeros ("no checksum" per spec — muxing never decodes audio).
//!
//! Block sizes, sample rate, channels and bits-per-sample come from the
//! caller-supplied STREAMINFO (see
//! [`crate::api::MuxerBuilder::with_flac_streaminfo`]); the queued frames
//! must agree with it.

use std::io::{self, Write};

use crate::api::Metadata;
use crate::codec::flac::{
    BLOCK_TYPE_STREAMINFO, BlockingStrategy, FLAC_MARKER, build_streaminfo, crc16,
    parse_frame_header, parse_streaminfo,
};
use crate::muxer::mp4::Mp4AudioTrack;

/// Vendor string in the `VORBIS_COMMENT` block.
const VENDOR: &str = "muxfin";
/// `VORBIS_COMMENT` metadata block type.
const BLOCK_TYPE_VORBIS_COMMENT: u8 = 4;

/// Errors produced while queuing samples or finalising the file.
#[derive(Debug)]
pub enum FlacWriterError {
    /// Audio sample is not a structurally valid FLAC frame (header or
    /// footer CRC mismatch).
    InvalidFlacFrame,
    /// Queued frame disagrees with STREAMINFO (channels/rate/bits or
    /// fixed-blocksize bounds).
    InconsistentFrame,
    /// Frame coded numbers went backwards (corrupt stream).
    NonMonotonicFrame,
    /// Valid STREAMINFO was not supplied via
    /// [`crate::api::MuxerBuilder::with_flac_streaminfo`].
    MissingStreaminfo,
    /// Track parameters disagree with the supplied STREAMINFO.
    StreaminfoMismatch { what: String },
    /// Audio track is not enabled on this writer.
    AudioNotEnabled,
    /// A non-FLAC codec cannot be carried in a native FLAC file.
    UnsupportedCodec { codec: String, reason: String },
    /// The writer has already been finalised.
    AlreadyFinalized,
    /// Low-level IO error.
    Io(std::io::Error),
}

impl std::fmt::Display for FlacWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlacWriterError::InvalidFlacFrame => write!(f, "invalid FLAC frame"),
            FlacWriterError::InconsistentFrame => {
                write!(f, "FLAC frame disagrees with STREAMINFO")
            }
            FlacWriterError::NonMonotonicFrame => write!(f, "FLAC coded numbers went backwards"),
            FlacWriterError::MissingStreaminfo => {
                write!(
                    f,
                    "valid FLAC STREAMINFO (34 bytes) must be provided using with_flac_streaminfo()"
                )
            }
            FlacWriterError::StreaminfoMismatch { what } => {
                write!(f, "FLAC track parameters disagree with STREAMINFO ({what})")
            }
            FlacWriterError::AudioNotEnabled => write!(f, "audio track not enabled"),
            FlacWriterError::UnsupportedCodec { codec, reason } => {
                write!(f, "codec {codec} cannot be carried in FLAC: {reason}")
            }
            FlacWriterError::AlreadyFinalized => write!(f, "writer already finalised"),
            FlacWriterError::Io(err) => write!(f, "IO error: {err}"),
        }
    }
}

impl std::error::Error for FlacWriterError {}

/// Encode one metadata block header + payload.
fn metadata_block(last: bool, block_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + payload.len());
    out.push(if last { 0x80 } else { 0x00 } | (block_type & 0x7F));
    let len = payload.len() as u32;
    out.push((len >> 16) as u8);
    out.push((len >> 8) as u8);
    out.push(len as u8);
    out.extend_from_slice(payload);
    out
}

/// Encode a `VORBIS_COMMENT` payload: vendor + `TITLE` comment when set.
fn vorbis_comment_payload(title: Option<&str>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(VENDOR.len() as u32).to_le_bytes());
    out.extend_from_slice(VENDOR.as_bytes());
    let mut comments: Vec<Vec<u8>> = Vec::new();
    if let Some(title) = title.filter(|t| !t.is_empty()) {
        comments.push(format!("TITLE={title}").into_bytes());
    }
    out.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for comment in &comments {
        out.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        out.extend_from_slice(comment);
    }
    out
}

/// Buffered native-FLAC writer. See the [module](self) documentation.
pub struct FlacWriter<Writer> {
    writer: Writer,
    audio_track: Option<Mp4AudioTrack>,
    frames: Vec<Vec<u8>>,
    prev_end: Option<u64>,
    finalized: bool,
    bytes_written: u64,
}

impl<Writer: Write> FlacWriter<Writer> {
    /// Wraps the provided writer for native FLAC output.
    pub fn new(writer: Writer) -> Self {
        Self {
            writer,
            audio_track: None,
            frames: Vec::new(),
            prev_end: None,
            finalized: false,
            bytes_written: 0,
        }
    }

    pub(crate) fn audio_sample_count(&self) -> u64 {
        self.frames.len() as u64
    }

    pub(crate) fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Total decoded audio in samples (excludes nothing: FLAC has no
    /// pre-skip; the last queued frame's end is the stream end).
    pub(crate) fn total_samples(&self) -> u64 {
        self.prev_end.unwrap_or(0)
    }

    fn write_counted(&mut self, buf: &[u8]) -> io::Result<()> {
        self.bytes_written = self.bytes_written.saturating_add(buf.len() as u64);
        self.writer.write_all(buf)
    }

    pub fn enable_audio(&mut self, track: Mp4AudioTrack) {
        self.audio_track = Some(track);
    }

    /// Queues one native FLAC frame (exactly one MP4-style sample). The
    /// frame is validated eagerly: header CRC8, footer CRC-16, STREAMINFO
    /// consistency and monotonic coded numbers (fail fast, no guessing).
    pub fn write_audio_sample(&mut self, pts: u64, data: &[u8]) -> Result<(), FlacWriterError> {
        if self.finalized {
            return Err(FlacWriterError::AlreadyFinalized);
        }
        let track = self
            .audio_track
            .as_ref()
            .ok_or(FlacWriterError::AudioNotEnabled)?;
        if track.codec != crate::api::AudioCodec::Flac {
            return Err(FlacWriterError::UnsupportedCodec {
                codec: track.codec.to_string(),
                reason: "native FLAC carries FLAC frames only".to_string(),
            });
        }
        // `pts` is informational here (coded numbers are authoritative);
        // the frontend enforces ordering.
        let _ = pts;

        let streaminfo = track
            .flac_streaminfo
            .as_deref()
            .and_then(parse_streaminfo)
            .ok_or(FlacWriterError::MissingStreaminfo)?;

        let header = parse_frame_header(data).ok_or(FlacWriterError::InvalidFlacFrame)?;
        if data.len() < 6 {
            return Err(FlacWriterError::InvalidFlacFrame);
        }
        let stored = u16::from_be_bytes([data[data.len() - 2], data[data.len() - 1]]);
        if crc16(&data[..data.len() - 2]) != stored {
            return Err(FlacWriterError::InvalidFlacFrame);
        }

        // Fixed-blocksize frames (except possibly the stream tail, whose
        // shortness the CRC already vouches for) must fit STREAMINFO.
        if header.strategy == BlockingStrategy::Fixed
            && streaminfo.max_block_size != 0
            && u64::from(header.block_size) > u64::from(streaminfo.max_block_size)
        {
            return Err(FlacWriterError::InconsistentFrame);
        }

        let sample_number = match header.strategy {
            BlockingStrategy::Variable => header.coded_number,
            BlockingStrategy::Fixed => {
                let nominal = if streaminfo.max_block_size != 0 {
                    u64::from(streaminfo.max_block_size)
                } else {
                    u64::from(header.block_size)
                };
                header
                    .coded_number
                    .checked_mul(nominal)
                    .ok_or(FlacWriterError::InvalidFlacFrame)?
            }
        };
        if let Some(prev) = self.prev_end
            && sample_number < prev
        {
            return Err(FlacWriterError::NonMonotonicFrame);
        }
        self.prev_end = Some(sample_number + u64::from(header.block_size));

        self.frames.push(data.to_vec());
        Ok(())
    }

    /// Finalises the file: `fLaC` marker, STREAMINFO with exact totals,
    /// `VORBIS_COMMENT`, then the queued frames verbatim.
    pub fn finalize(&mut self, metadata: Option<&Metadata>) -> io::Result<()> {
        if self.finalized {
            return Err(io::Error::other("flac writer already finalised"));
        }
        self.finalized = true;

        self.finalize_inner(metadata).map_err(|e| match e {
            FlacWriterError::Io(io_err) => io_err,
            other => io::Error::other(other.to_string()),
        })
    }

    fn finalize_inner(&mut self, metadata: Option<&Metadata>) -> Result<(), FlacWriterError> {
        let track = self
            .audio_track
            .as_ref()
            .ok_or(FlacWriterError::AudioNotEnabled)?;
        if track.codec != crate::api::AudioCodec::Flac {
            return Err(FlacWriterError::UnsupportedCodec {
                codec: track.codec.to_string(),
                reason: "native FLAC carries FLAC frames only".to_string(),
            });
        }
        let supplied = track
            .flac_streaminfo
            .as_deref()
            .and_then(parse_streaminfo)
            .ok_or(FlacWriterError::MissingStreaminfo)?;
        if track.sample_rate != supplied.sample_rate
            || track.channels != u16::from(supplied.channels)
        {
            return Err(FlacWriterError::StreaminfoMismatch {
                what: format!(
                    "track ({} Hz, {} ch) vs STREAMINFO ({} Hz, {} ch)",
                    track.sample_rate, track.channels, supplied.sample_rate, supplied.channels
                ),
            });
        }
        if self.frames.is_empty() {
            return Err(FlacWriterError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "no FLAC frames queued: write at least one audio sample",
            )));
        }

        // Recompute what the queued frames determine: totals, frame-size
        // bounds, observed block-size bounds. Rate/channels/bps stay
        // verbatim from the supplied STREAMINFO; MD5 is zeros (no decode).
        let total_samples = self.prev_end.unwrap_or(0);
        let mut min_frame = u32::MAX;
        let mut max_frame = 0u32;
        for frame in &self.frames {
            let len = u32::try_from(frame.len()).unwrap_or(u32::MAX);
            min_frame = min_frame.min(len);
            max_frame = max_frame.max(len);
        }
        let streaminfo_body = build_streaminfo(
            supplied.min_block_size,
            supplied.max_block_size,
            supplied.sample_rate,
            supplied.channels,
            supplied.bits_per_sample,
            total_samples,
        );
        // Patch the recomputed min/max frame sizes into the body
        // (`build_streaminfo` leaves them unknown/zero).
        let mut streaminfo_body = streaminfo_body;
        for (range, value) in [(4..7, min_frame), (7..10, max_frame)] {
            let clamped = value.min(0xFF_FFFF);
            streaminfo_body[range].copy_from_slice(&[
                (clamped >> 16) as u8,
                (clamped >> 8) as u8,
                clamped as u8,
            ]);
        }
        debug_assert!(parse_streaminfo(&streaminfo_body).is_some());

        let mut out = Vec::new();
        out.extend_from_slice(FLAC_MARKER);
        out.extend_from_slice(&metadata_block(
            false,
            BLOCK_TYPE_STREAMINFO,
            &streaminfo_body,
        ));
        let comments = vorbis_comment_payload(metadata.and_then(|m| m.title.as_deref()));
        out.extend_from_slice(&metadata_block(true, BLOCK_TYPE_VORBIS_COMMENT, &comments));
        for frame in &self.frames {
            out.extend_from_slice(frame);
        }

        self.write_counted(&out).map_err(FlacWriterError::Io)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::flac::{STREAMINFO_LEN, synthetic_frame_header};

    fn test_streaminfo() -> [u8; STREAMINFO_LEN] {
        // 4096-block stereo 44.1 kHz 16-bit (totals recomputed at finalize).
        build_streaminfo(4096, 4096, 44_100, 2, 16, 0)
    }

    fn enable(writer: &mut FlacWriter<Vec<u8>>, streaminfo: &[u8]) {
        writer.enable_audio(Mp4AudioTrack {
            sample_rate: 44_100,
            channels: 2,
            codec: crate::api::AudioCodec::Flac,
            flac_streaminfo: Some(streaminfo.to_vec()),
            opus_preskip: None,
            aac_asc_override: None,
            opus_config_override: None,
            language: None,
        });
    }

    /// One synthetic frame: valid header + filler payload + footer CRC-16.
    fn frame(frame_number: u64, payload_len: usize) -> Vec<u8> {
        let mut data = synthetic_frame_header(frame_number, 4095);
        data.extend(std::iter::repeat_n(0xABu8, payload_len));
        let crc = crc16(&data);
        data.extend_from_slice(&crc.to_be_bytes());
        data
    }

    #[test]
    fn rejects_missing_streaminfo() {
        let mut writer = FlacWriter::new(Vec::<u8>::new());
        writer.enable_audio(Mp4AudioTrack {
            sample_rate: 44_100,
            channels: 2,
            codec: crate::api::AudioCodec::Flac,
            flac_streaminfo: None,
            opus_preskip: None,
            aac_asc_override: None,
            opus_config_override: None,
            language: None,
        });
        assert!(matches!(
            writer.write_audio_sample(0, &frame(0, 8)),
            Err(FlacWriterError::MissingStreaminfo)
        ));
    }

    #[test]
    fn rejects_bad_footer_crc() {
        let mut writer = FlacWriter::new(Vec::<u8>::new());
        enable(&mut writer, &test_streaminfo());
        let mut bad = frame(0, 8);
        let last = bad.len() - 1;
        bad[last] ^= 0xFF;
        assert!(matches!(
            writer.write_audio_sample(0, &bad),
            Err(FlacWriterError::InvalidFlacFrame)
        ));
    }

    #[test]
    fn vorbis_comment_layout() {
        let payload = vorbis_comment_payload(Some("take 5"));
        // vendor "muxfin" + 1 comment "TITLE=take 5".
        assert!(payload.windows(6).any(|w| w == b"muxfin"));
        assert!(payload.windows(12).any(|w| w == b"TITLE=take 5"));
        let empty = vorbis_comment_payload(None);
        // 4 (vendor len) + 6 ("muxfin") + 4 (zero comments).
        assert_eq!(empty.len(), 14);
        assert_eq!(u32::from_le_bytes(empty[10..14].try_into().unwrap()), 0);
    }
}
