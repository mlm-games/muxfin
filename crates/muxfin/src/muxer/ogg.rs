//! Ogg Opus muxing (RFC 7845).
//!
//! [`OggWriter`] queues raw Opus packets and emits a single-logical-stream
//! Ogg Opus file on [`finalize`](OggWriter::finalize): an `OpusHead`
//! page, an `OpusTags` page, then audio pages whose granule positions
//! derive from exact packet TOC durations (never assumed frame sizes).
//! The public frontend is [`crate::api::OggMuxer`], built with
//! [`crate::api::MuxerBuilder::build_ogg`].

use std::io::{self, Write};

use crate::api::Metadata;
use crate::codec::opus::{OpusConfig, OpusConfigError, is_valid_opus_packet, opus_packet_samples};
use crate::muxer::mp4::Mp4AudioTrack;

/// Opus clock: granule positions count 48 kHz samples (RFC 7845 §3).
pub(crate) const OPUS_CLOCK_HZ: u64 = 48_000;
/// Default encoder delay signalled as pre-skip (matches MP4 `dOps`/MKV).
const DEFAULT_PRE_SKIP: u16 = 312;
/// Vendor string in the `OpusTags` header packet.
const VENDOR: &[u8] = b"muxfin";
/// Fixed stream serial: a file holds one logical stream, and a constant
/// keeps output byte-identical across runs (golden-test friendly).
const STREAM_SERIAL: u32 = 0x4D55_5846; // "MUXF"
/// Maximum audio packets per Ogg page (bounds page size; the 255-segment
/// lacing cap is enforced independently when batching).
const MAX_PACKETS_PER_PAGE: usize = 50;

/// Errors produced while queuing samples or finalising the file.
#[derive(Debug)]
pub enum OggWriterError {
    /// Audio sample is not a valid Opus packet.
    InvalidOpusPacket,
    /// Caller-supplied Opus head config failed structural validation.
    InvalidOpusConfig { reason: String },
    /// Audio timestamps must not go backwards.
    NonIncreasingTimestamp,
    /// Audio track is not enabled on this writer.
    AudioNotEnabled,
    /// A non-Opus codec cannot be carried in Ogg Opus.
    UnsupportedCodec { codec: String, reason: String },
    /// The writer has already been finalised.
    AlreadyFinalized,
    /// Low-level IO error.
    Io(std::io::Error),
}

impl std::fmt::Display for OggWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OggWriterError::InvalidOpusPacket => write!(f, "invalid Opus packet"),
            OggWriterError::InvalidOpusConfig { reason } => {
                write!(f, "invalid Opus head config: {reason}")
            }
            OggWriterError::NonIncreasingTimestamp => write!(f, "timestamps must grow"),
            OggWriterError::AudioNotEnabled => write!(f, "audio track not enabled"),
            OggWriterError::UnsupportedCodec { codec, reason } => {
                write!(f, "codec {codec} cannot be carried in Ogg: {reason}")
            }
            OggWriterError::AlreadyFinalized => write!(f, "writer already finalised"),
            OggWriterError::Io(err) => write!(f, "IO error: {err}"),
        }
    }
}

impl std::error::Error for OggWriterError {}

/// Resolve the effective Opus head config: a full caller override wins,
/// otherwise channels + pre-skip (same precedence as the MP4 `dOps`
/// builder). The result is structurally validated, so [`opus_head_packet`]
/// can serialise it without further checks.
pub(crate) fn resolve_opus_config(track: &Mp4AudioTrack) -> Result<OpusConfig, OpusConfigError> {
    let config = match &track.opus_config_override {
        Some(c) => c.clone(),
        None => OpusConfig::default()
            .with_channels(track.channels.min(u8::MAX as u16) as u8)
            .with_pre_skip(track.opus_preskip.unwrap_or(DEFAULT_PRE_SKIP)),
    };
    config.validate()?;
    Ok(config)
}

/// Serialise an `OpusHead` packet (RFC 7845 §5.1, little-endian fields).
fn opus_head_packet(config: &OpusConfig) -> Vec<u8> {
    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead");
    // Note: this is the OpusHead version (always 1), not the `dOps`
    // version (always 0) from `OpusConfig::version`.
    head.push(1);
    head.push(config.output_channel_count);
    head.extend_from_slice(&config.pre_skip.to_le_bytes());
    head.extend_from_slice(&config.input_sample_rate.to_le_bytes());
    head.extend_from_slice(&config.output_gain.to_le_bytes());
    head.push(config.channel_mapping_family);
    if config.channel_mapping_family != 0 {
        // Validated: family ≥ 1 always carries counts and a mapping.
        head.push(config.stream_count.unwrap_or(1));
        head.push(config.coupled_count.unwrap_or(0));
        if let Some(mapping) = &config.channel_mapping {
            head.extend_from_slice(mapping);
        } else {
            for i in 0..config.output_channel_count {
                head.push(i);
            }
        }
    }
    head
}

/// Serialise an `OpusTags` packet (RFC 7845 §5.2) with an optional
/// `TITLE` user comment taken from container metadata.
fn opus_tags_packet(title: Option<&str>) -> Vec<u8> {
    let mut tags = Vec::new();
    tags.extend_from_slice(b"OpusTags");
    tags.extend_from_slice(&(VENDOR.len() as u32).to_le_bytes());
    tags.extend_from_slice(VENDOR);
    let mut comments: Vec<Vec<u8>> = Vec::new();
    if let Some(title) = title.filter(|t| !t.is_empty()) {
        comments.push(format!("TITLE={title}").into_bytes());
    }
    tags.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for comment in &comments {
        tags.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        tags.extend_from_slice(comment);
    }
    tags
}

/// Ogg CRC-32 (poly 0x04C11DB7, init 0, no reflection, xorout 0).
fn ogg_crc(data: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &byte in data {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Write one Ogg page carrying `packets` (lacing derived from lengths;
/// a packet whose length is an exact multiple of 255 still emits the
/// terminating zero lacing value).
fn write_page<W: Write>(
    w: &mut W,
    header_type: u8,
    granule: u64,
    seq: u32,
    packets: &[&[u8]],
) -> io::Result<()> {
    let mut segments = Vec::new();
    for packet in packets {
        let mut rest = *packet;
        loop {
            let take = rest.len().min(255);
            segments.push(take as u8);
            rest = &rest[take..];
            if take < 255 {
                break;
            }
        }
    }
    debug_assert!(
        segments.len() <= 255,
        "page batching must cap lacing segments at 255"
    );

    let mut page =
        Vec::with_capacity(27 + segments.len() + packets.iter().map(|p| p.len()).sum::<usize>());
    page.extend_from_slice(b"OggS");
    page.push(0); // stream version
    page.push(header_type);
    page.extend_from_slice(&granule.to_le_bytes());
    page.extend_from_slice(&STREAM_SERIAL.to_le_bytes());
    page.extend_from_slice(&seq.to_le_bytes());
    page.extend_from_slice(&[0u8; 4]); // checksum placeholder
    page.push(segments.len() as u8);
    page.extend_from_slice(&segments);
    for packet in packets {
        page.extend_from_slice(packet);
    }
    let crc = ogg_crc(&page);
    page[22..26].copy_from_slice(&crc.to_le_bytes());
    w.write_all(&page)
}

/// One queued audio packet with its exact duration (from the TOC, so
/// granule positions never assume a fixed frame size).
struct QueuedPacket {
    /// Packet duration in 48 kHz samples.
    samples: u32,
    data: Vec<u8>,
}

/// Buffered Ogg Opus writer. See the [module](self) documentation.
pub struct OggWriter<Writer> {
    writer: Writer,
    audio_track: Option<Mp4AudioTrack>,
    samples: Vec<QueuedPacket>,
    prev_pts: Option<u64>,
    finalized: bool,
    bytes_written: u64,
}

impl<Writer: Write> OggWriter<Writer> {
    /// Wraps the provided writer for Ogg Opus output.
    pub fn new(writer: Writer) -> Self {
        Self {
            writer,
            audio_track: None,
            samples: Vec::new(),
            prev_pts: None,
            finalized: false,
            bytes_written: 0,
        }
    }

    pub(crate) fn audio_sample_count(&self) -> u64 {
        self.samples.len() as u64
    }

    pub(crate) fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Total decoded audio in 48 kHz samples (excludes pre-skip).
    pub(crate) fn total_samples_48k(&self) -> u64 {
        self.samples.iter().map(|s| u64::from(s.samples)).sum()
    }

    fn write_counted(&mut self, buf: &[u8]) -> io::Result<()> {
        self.bytes_written = self.bytes_written.saturating_add(buf.len() as u64);
        self.writer.write_all(buf)
    }

    pub fn enable_audio(&mut self, track: Mp4AudioTrack) {
        self.audio_track = Some(track);
    }

    /// Queues one Opus packet. `pts` is in 48 kHz ticks and must not go
    /// backwards; the packet is validated eagerly (fail fast, no guessing).
    pub fn write_audio_sample(&mut self, pts: u64, data: &[u8]) -> Result<(), OggWriterError> {
        if self.finalized {
            return Err(OggWriterError::AlreadyFinalized);
        }
        let track = self
            .audio_track
            .as_ref()
            .ok_or(OggWriterError::AudioNotEnabled)?;
        if track.codec != crate::api::AudioCodec::Opus {
            return Err(OggWriterError::UnsupportedCodec {
                codec: track.codec.to_string(),
                reason: "Ogg carries Opus audio only".to_string(),
            });
        }
        if let Some(prev) = self.prev_pts
            && pts < prev
        {
            return Err(OggWriterError::NonIncreasingTimestamp);
        }
        let samples = opus_packet_samples(data)
            .filter(|_| is_valid_opus_packet(data))
            .ok_or(OggWriterError::InvalidOpusPacket)?;
        self.samples.push(QueuedPacket {
            samples,
            data: data.to_vec(),
        });
        self.prev_pts = Some(pts);
        Ok(())
    }

    /// Finalises the file: `OpusHead` + `OpusTags` header pages, then
    /// audio pages with exact granule positions (`pre_skip + decoded`
    /// per RFC 7845, so our own demuxer verifies them by differences).
    pub fn finalize(&mut self, metadata: Option<&Metadata>) -> io::Result<()> {
        if self.finalized {
            return Err(io::Error::other("ogg writer already finalised"));
        }
        self.finalized = true;

        self.finalize_inner(metadata).map_err(|e| match e {
            OggWriterError::Io(io_err) => io_err,
            other => io::Error::other(other.to_string()),
        })
    }

    fn finalize_inner(&mut self, metadata: Option<&Metadata>) -> Result<(), OggWriterError> {
        let track = self
            .audio_track
            .as_ref()
            .ok_or(OggWriterError::AudioNotEnabled)?;
        if track.codec != crate::api::AudioCodec::Opus {
            return Err(OggWriterError::UnsupportedCodec {
                codec: track.codec.to_string(),
                reason: "Ogg carries Opus audio only".to_string(),
            });
        }
        let config = resolve_opus_config(track).map_err(|e| OggWriterError::InvalidOpusConfig {
            reason: e.to_string(),
        })?;

        let mut out = Vec::new();
        let mut seq: u32 = 0;
        let head = opus_head_packet(&config);
        write_page(&mut out, 0x02, 0, seq, &[&head]).map_err(OggWriterError::Io)?;
        seq += 1;

        // A stream with no audio still gets EOS on the tags page so the
        // file is a terminated (if empty) logical stream.
        let no_audio = self.samples.is_empty();
        let tags = opus_tags_packet(metadata.and_then(|m| m.title.as_deref()));
        write_page(
            &mut out,
            if no_audio { 0x04 } else { 0x00 },
            0,
            seq,
            &[&tags],
        )
        .map_err(OggWriterError::Io)?;
        seq += 1;

        // Batch audio packets: at most 50 packets and 255 lacing segments
        // per page (a packet costs `len / 255 + 1` segments).
        let mut idx = 0;
        let mut samples_done: u64 = 0;
        while idx < self.samples.len() {
            let mut batch: Vec<&[u8]> = Vec::new();
            let mut seg_count = 0usize;
            while idx < self.samples.len() && batch.len() < MAX_PACKETS_PER_PAGE {
                let queued = &self.samples[idx];
                let segs = queued.data.len() / 255 + 1;
                if seg_count + segs > 255 {
                    break;
                }
                seg_count += segs;
                batch.push(queued.data.as_slice());
                samples_done += u64::from(queued.samples);
                idx += 1;
            }
            let is_last = idx >= self.samples.len();
            let granule = samples_done.saturating_add(u64::from(config.pre_skip));
            write_page(
                &mut out,
                if is_last { 0x04 } else { 0x00 },
                granule,
                seq,
                &batch,
            )
            .map_err(OggWriterError::Io)?;
            seq += 1;
        }

        self.write_counted(&out).map_err(OggWriterError::Io)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_check_value() {
        // Same Ogg CRC-32 as the demuxer: poly 0x04C11DB7, init 0.
        assert_eq!(ogg_crc(b"123456789"), 0x89A1_897F);
    }

    #[test]
    fn head_layout_family0() {
        let config = OpusConfig::default();
        let head = opus_head_packet(&config);
        assert_eq!(head.len(), 19);
        assert_eq!(&head[..8], b"OpusHead");
        assert_eq!(head[8], 1); // OpusHead version
        assert_eq!(head[9], 2);
        assert_eq!(u16::from_le_bytes([head[10], head[11]]), 312);
        assert_eq!(
            u32::from_le_bytes([head[12], head[13], head[14], head[15]]),
            48_000
        );
        assert_eq!(head[18], 0); // mapping family
    }

    #[test]
    fn head_layout_family1() {
        let config = OpusConfig::surround_51().with_pre_skip(120);
        let head = opus_head_packet(&config);
        // 19 + stream/coupled counts + 6 mapping entries.
        assert_eq!(head.len(), 27);
        assert_eq!(head[18], 1);
        assert_eq!(head[19], 4); // streams
        assert_eq!(head[20], 2); // coupled
        assert_eq!(&head[21..], &[0, 4, 1, 2, 3, 5]);
    }

    #[test]
    fn resolve_rejects_bad_channels() {
        fn track_with(channels: u16) -> Mp4AudioTrack {
            Mp4AudioTrack {
                sample_rate: 48_000,
                channels,
                codec: crate::api::AudioCodec::Opus,
                flac_streaminfo: None,
                opus_preskip: None,
                aac_asc_override: None,
                opus_config_override: None,
                language: None,
            }
        }
        assert!(resolve_opus_config(&track_with(0)).is_err());
        assert!(resolve_opus_config(&track_with(7)).is_err());
        // Sanity: 1-2 channels resolve.
        assert!(resolve_opus_config(&track_with(1)).is_ok());
        assert!(resolve_opus_config(&track_with(2)).is_ok());
    }

    #[test]
    fn page_lacing_exact_multiple_of_255() {
        // A 510-byte packet needs lacing [255, 255, 0]: the terminating
        // zero segment keeps the packet end unambiguous.
        let mut out = Vec::new();
        let packet = vec![0u8; 510];
        write_page(&mut out, 0x00, 0, 0, &[&packet]).expect("page");
        let n_segments = out[26] as usize;
        assert_eq!(n_segments, 3);
        assert_eq!(&out[27..30], &[255, 255, 0]);
        // CRC still verifies (zeroed field + recompute).
        let stored = u32::from_le_bytes(out[22..26].try_into().unwrap());
        let mut check = out.clone();
        check[22..26].fill(0);
        assert_eq!(ogg_crc(&check), stored);
    }

    #[test]
    fn finalize_empty_stream_terminates_on_tags() {
        let mut writer = OggWriter::new(Vec::<u8>::new());
        writer.enable_audio(Mp4AudioTrack {
            sample_rate: 48_000,
            channels: 2,
            codec: crate::api::AudioCodec::Opus,
            flac_streaminfo: None,
            opus_preskip: None,
            aac_asc_override: None,
            opus_config_override: None,
            language: None,
        });
        writer.finalize(None).expect("finalize");
        assert_eq!(writer.audio_sample_count(), 0);
        assert!(writer.bytes_written() > 0);
    }
}
