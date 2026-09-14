use std::fmt;
use std::io::{self, Write};

use crate::api::{AacProfile, AudioCodec, Metadata, SubtitleCodec, VideoCodec};
use crate::assert_invariant;
use crate::codec::av1::{Av1Config, extract_av1_config};
use crate::codec::flac::{STREAMINFO_LEN, is_valid_flac_frame, parse_streaminfo};
use crate::codec::h264::{AvcConfig, annexb_to_avcc, default_avc_config, extract_avc_config};
use crate::codec::h265::{HevcConfig, extract_hevc_config, hevc_annexb_to_hvcc};
use crate::codec::opus::{OPUS_SAMPLE_RATE, OpusConfig, is_valid_opus_packet};
use crate::codec::vp9::{Vp9Config, extract_vp9_config, vpcc_payload};

const MOVIE_TIMESCALE: u32 = 1000;
/// Track/media timebase used for converting `pts` seconds into MP4 sample deltas.
///
/// v0.1.0 uses a 90 kHz media timescale (common for MP4/H.264 workflows).
pub const MEDIA_TIMESCALE: u32 = 90_000;

/// Video codec configuration extracted from the first keyframe.
#[derive(Clone, Debug)]
pub enum VideoConfig {
    /// H.264/AVC configuration (SPS + PPS)
    Avc(AvcConfig),
    /// H.265/HEVC configuration (VPS + SPS + PPS)
    Hevc(HevcConfig),
    /// AV1 configuration (Sequence Header OBU)
    Av1(Av1Config),
    /// VP9 configuration (frame header parameters)
    Vp9(Vp9Config),
}

/// Minimal MP4 writer used by the early slices.
pub struct Mp4Writer<Writer> {
    writer: Writer,
    video_codec: Option<VideoCodec>,
    video_samples: Vec<SampleInfo>,
    video_prev_pts: Option<u64>,
    video_last_delta: Option<u32>,
    video_config: Option<VideoConfig>,
    caller_video_config: Option<crate::api::VideoDecoderConfig>,
    audio_track: Option<Mp4AudioTrack>,
    audio_samples: Vec<SampleInfo>,
    audio_prev_pts: Option<u64>,
    audio_last_delta: Option<u32>,
    subtitle_track: Option<Mp4SubtitleTrack>,
    subtitle_samples: Vec<SampleInfo>,
    subtitle_prev_pts: Option<u64>,
    subtitle_last_delta: Option<u32>,
    finalized: bool,
    bytes_written: u64,
}

/// Simplified video track information used when writing the header.
pub struct Mp4VideoTrack {
    pub width: u32,
    pub height: u32,
    /// Per-track ISO-639-2/T language (verdict §10); `None` falls back to metadata.
    pub language: Option<String>,
}

pub struct Mp4AudioTrack {
    pub sample_rate: u32,
    pub channels: u16,
    pub codec: AudioCodec,
    /// 34-byte STREAMINFO block for FLAC tracks; `None` otherwise.
    pub flac_streaminfo: Option<Vec<u8>>,
    /// Opus pre-skip in 48 kHz samples; `None` selects the 312 default.
    pub opus_preskip: Option<u16>,
    /// Caller-supplied AAC ASC override (HE/HEv2, verdict §8).
    pub aac_asc_override: Option<Vec<u8>>,
    /// Full Opus dOps override (verdict §8).
    pub opus_config_override: Option<crate::codec::opus::OpusConfig>,
    /// Per-track ISO-639-2/T language (verdict §10); `None` falls back to metadata.
    pub language: Option<String>,
}

pub struct Mp4SubtitleTrack {
    pub codec: SubtitleCodec,
    pub language: Option<String>,
    /// ASS/SSA CodecPrivate for Matroska `S_TEXT/ASS` / `S_TEXT/SSA`.
    /// Ignored by the MP4 writer (which rejects those codecs at build).
    pub ass_codec_private: Option<Vec<u8>>,
}

pub(crate) struct SampleInfo {
    pts: u64,
    dts: u64, // Decode time (for B-frames: dts != pts)
    data: Vec<u8>,
    is_keyframe: bool,
    duration: Option<u32>,
}

/// Metadata-only view of a sample for table building.
///
/// The seekable-streaming writer (§5) writes sample bytes straight to the
/// output and retains only these descriptors, so RAM stays O(metadata)
/// instead of O(media). `SampleTables::from_samples` consumes this type so
/// buffered and streaming paths build identical tables.
#[derive(Debug, Clone)]
pub(crate) struct SampleMeta {
    pub(crate) pts: u64,
    pub(crate) dts: u64,
    pub(crate) size: u32,
    pub(crate) is_keyframe: bool,
    pub(crate) duration: Option<u32>,
}

impl SampleInfo {
    fn meta(&self) -> SampleMeta {
        SampleMeta {
            pts: self.pts,
            dts: self.dts,
            size: u32::try_from(self.data.len()).unwrap_or(u32::MAX),
            is_keyframe: self.is_keyframe,
            duration: self.duration,
        }
    }
}

/// Metadata-only projection of buffered samples for table building.
pub(crate) fn sample_metas(samples: &[SampleInfo]) -> Vec<SampleMeta> {
    samples.iter().map(SampleInfo::meta).collect()
}

pub(crate) struct SampleTables {
    durations: Vec<u32>,
    sizes: Vec<u32>,
    keyframes: Vec<u32>,
    chunk_offsets: Vec<u64>,
    samples_per_chunk: u32,
    cts_offsets: Vec<i32>, // Composition time offsets (pts - dts) for ctts box
    has_bframes: bool,     // True if any sample has pts != dts
    first_pts: Option<u64>,
}

impl SampleTables {
    pub(crate) fn from_samples(
        samples: &[SampleMeta],
        chunk_offsets: Vec<u64>,
        samples_per_chunk: u32,
        fallback_duration: Option<u32>,
    ) -> Self {
        let sample_count = samples.len() as u32;
        let mut durations = Vec::with_capacity(sample_count as usize);
        for (idx, sample) in samples.iter().enumerate() {
            let duration = sample.duration.unwrap_or_else(|| {
                if idx == samples.len() - 1 {
                    fallback_duration.unwrap_or(1)
                } else {
                    1
                }
            });
            durations.push(duration);
        }
        let sizes: Vec<u32> = samples.iter().map(|sample| sample.size).collect();
        let keyframes = samples
            .iter()
            .enumerate()
            .filter_map(|(idx, sample)| {
                if sample.is_keyframe {
                    Some(u32::try_from(idx + 1).expect("too many samples"))
                } else {
                    None
                }
            })
            .collect();

        // Compute composition time offsets (cts = pts - dts), checked (verdict §7).
        // New integer API rejects out-of-range offsets at write time; legacy
        // f64 path saturates here to avoid silent truncation.
        let mut has_bframes = false;
        let cts_offsets: Vec<i32> = samples
            .iter()
            .map(|sample| {
                let diff = sample.pts as i64 - sample.dts as i64;
                let offset =
                    i32::try_from(diff).unwrap_or(if diff > 0 { i32::MAX } else { i32::MIN });
                if offset != 0 {
                    has_bframes = true;
                }
                offset
            })
            .collect();

        let _ = sample_count;
        let first_pts = samples.first().map(|s| s.pts);
        Self {
            durations,
            sizes,
            keyframes,
            chunk_offsets,
            samples_per_chunk,
            cts_offsets,
            has_bframes,
            first_pts,
        }
    }

    /// Calculate total duration in media timescale units
    fn total_duration(&self) -> u64 {
        self.durations.iter().map(|&d| d as u64).sum()
    }

    /// Movie-origin-relative start offset in media ticks.
    fn start_offset(&self, origin: u64) -> u64 {
        self.first_pts
            .map(|f| f.saturating_sub(origin))
            .unwrap_or(0)
    }
}

/// Build an `elst` inside `edts` for a track starting after movie origin.
/// `first_pts` is the track's first sample PTS in media ticks,
/// `movie_duration` the full movie duration in movie ticks.
///
/// Emits two entries: an empty edit covering the leading gap, then media
/// from `first_pts`. A single `{full_duration, media=first_pts}` entry
/// would instead present media `first_pts` at movie 0, shifting the whole
/// track early by the gap.
fn build_edts_box(first_pts: u64, movie_duration: u64) -> Vec<u8> {
    // Leading gap in movie ticks, rounded up so media never starts early.
    let gap_movie = (first_pts as u128 * u128::from(MOVIE_TIMESCALE))
        .div_ceil(u128::from(MEDIA_TIMESCALE)) as u64;
    let gap_movie = gap_movie.min(movie_duration);
    let rest_movie = movie_duration.saturating_sub(gap_movie);
    let use_v1 = first_pts > u64::from(u32::MAX)
        || movie_duration > u64::from(u32::MAX)
        || gap_movie > u64::from(u32::MAX);
    let mut elst_payload = Vec::new();
    if use_v1 {
        elst_payload.extend_from_slice(&0x0100_0000u32.to_be_bytes());
        elst_payload.extend_from_slice(&2u32.to_be_bytes());
        elst_payload.extend_from_slice(&gap_movie.to_be_bytes());
        elst_payload.extend_from_slice(&(-1i64).to_be_bytes());
        elst_payload.extend_from_slice(&1i16.to_be_bytes());
        elst_payload.extend_from_slice(&0i16.to_be_bytes());
        elst_payload.extend_from_slice(&rest_movie.to_be_bytes());
        elst_payload.extend_from_slice(&(first_pts as i64).to_be_bytes());
        elst_payload.extend_from_slice(&1i16.to_be_bytes());
        elst_payload.extend_from_slice(&0i16.to_be_bytes());
    } else {
        elst_payload.extend_from_slice(&0u32.to_be_bytes());
        elst_payload.extend_from_slice(&2u32.to_be_bytes());
        elst_payload.extend_from_slice(&(gap_movie as u32).to_be_bytes());
        elst_payload.extend_from_slice(&(-1i32).to_be_bytes());
        elst_payload.extend_from_slice(&1i16.to_be_bytes());
        elst_payload.extend_from_slice(&0i16.to_be_bytes());
        elst_payload.extend_from_slice(&(rest_movie as u32).to_be_bytes());
        elst_payload.extend_from_slice(&(first_pts as u32 as i32).to_be_bytes());
        elst_payload.extend_from_slice(&1i16.to_be_bytes());
        elst_payload.extend_from_slice(&0i16.to_be_bytes());
    }
    let elst = build_box(b"elst", &elst_payload);
    build_box(b"edts", &elst)
}

/// Severity level for validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ErrorSeverity {
    /// Critical error that prevents muxing (e.g., invalid syncword, corrupted data)
    Error,
    /// Warning for potential issues that might still work but are non-standard
    Warning,
}

/// Detailed ADTS validation error with comprehensive diagnostic information.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AdtsValidationError {
    /// The specific validation error that occurred.
    pub kind: AdtsErrorKind,
    /// Severity level of this error.
    pub severity: ErrorSeverity,
    /// Byte offset where the error was detected (0-based).
    pub byte_offset: usize,
    /// Expected value at the error location (if applicable).
    pub expected: Option<String>,
    /// Actual value found at the error location (if applicable).
    pub found: Option<String>,
    /// Enhanced hex dump with ASCII representation (up to 16 bytes).
    pub hex_dump: Option<String>,
    /// Recovery suggestion for fixing this error.
    pub suggestion: Option<String>,
    /// Code example showing how to fix this error.
    pub code_example: Option<String>,
    /// Technical details for developers (shown in verbose mode).
    pub technical_details: Option<String>,
    /// Related errors that occurred in the same frame.
    pub related_errors: Vec<AdtsValidationError>,
}

/// Specific types of ADTS validation errors with detailed context.
#[derive(Debug, Clone, serde::Serialize)]
pub enum AdtsErrorKind {
    /// Frame is too short to contain a valid ADTS header.
    FrameTooShort,
    /// Missing ADTS syncword (0xFFF) in first 12 bits.
    MissingSyncword,
    /// Frame length field indicates invalid size.
    InvalidFrameLength,
    /// Header length calculation doesn't match frame size.
    InvalidHeaderLength,
    /// MPEG version field contains invalid value.
    InvalidMpegVersion,
    /// Layer field is not set to 0 (reserved for AAC).
    InvalidLayer,
    /// Sample rate index is out of valid range.
    InvalidSampleRateIndex,
    /// Channel configuration is invalid.
    InvalidChannelConfig,
    /// CRC mismatch (if protection is present).
    CrcMismatch,
}

impl fmt::Display for AdtsValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Severity indicator
        let severity_icon = match self.severity {
            ErrorSeverity::Error => "🚨",
            ErrorSeverity::Warning => "⚠️",
        };
        write!(f, "{} ", severity_icon)?;

        // Main error message
        match &self.kind {
            AdtsErrorKind::FrameTooShort => {
                write!(
                    f,
                    "ADTS frame too short: need at least 7 bytes for header, got {}",
                    self.byte_offset
                )?;
            }
            AdtsErrorKind::MissingSyncword => {
                write!(
                    f,
                    "ADTS syncword missing at byte {}: expected 0xFFF in first 12 bits",
                    self.byte_offset
                )?;
                if let (Some(_expected), Some(found)) = (&self.expected, &self.found) {
                    write!(f, " (expected {}, found {})", _expected, found)?;
                }
            }
            AdtsErrorKind::InvalidFrameLength => {
                write!(
                    f,
                    "ADTS frame length invalid at byte {}: ",
                    self.byte_offset
                )?;
                if let (Some(_expected), Some(found)) = (&self.expected, &self.found) {
                    write!(f, "expected {}, found {}", _expected, found)?;
                }
                write!(
                    f,
                    " (frame length must be >= header length and <= total frame size)"
                )?;
            }
            AdtsErrorKind::InvalidHeaderLength => {
                write!(
                    f,
                    "ADTS header length mismatch at byte {}: ",
                    self.byte_offset
                )?;
                if let (Some(expected), Some(found)) = (&self.expected, &self.found) {
                    write!(f, "expected header length {}, found {}", expected, found)?;
                }
                write!(f, " (check protection_absent flag)")?;
            }
            AdtsErrorKind::InvalidMpegVersion => {
                write!(
                    f,
                    "ADTS MPEG version invalid at byte {}: ",
                    self.byte_offset
                )?;
                if let (Some(_expected), Some(found)) = (&self.expected, &self.found) {
                    write!(f, "expected {}, found {}", _expected, found)?;
                }
                write!(f, " (only MPEG-4 AAC is supported)")?;
            }
            AdtsErrorKind::InvalidLayer => {
                write!(f, "ADTS layer field invalid at byte {}: ", self.byte_offset)?;
                if let (Some(_expected), Some(found)) = (&self.expected, &self.found) {
                    write!(f, "expected {}, found {}", _expected, found)?;
                }
                write!(f, " (must be 0 for AAC)")?;
            }
            AdtsErrorKind::InvalidSampleRateIndex => {
                write!(
                    f,
                    "ADTS sample rate index invalid at byte {}: ",
                    self.byte_offset
                )?;
                if let (Some(_expected), Some(found)) = (&self.expected, &self.found) {
                    write!(f, "expected 0-12, found {}", found)?;
                }
                write!(f, " (valid range is 0-12 corresponding to 96000-7350 Hz)")?;
            }
            AdtsErrorKind::InvalidChannelConfig => {
                write!(
                    f,
                    "ADTS channel configuration invalid at byte {}: ",
                    self.byte_offset
                )?;
                if let (Some(_expected), Some(found)) = (&self.expected, &self.found) {
                    write!(f, "expected 1-7, found {}", found)?;
                }
                write!(f, " (valid range is 1-7 for mono/stereo configurations)")?;
            }
            AdtsErrorKind::CrcMismatch => {
                write!(f, "ADTS CRC mismatch at byte {}: ", self.byte_offset)?;
                write!(f, "frame data doesn't match CRC checksum")?;
            }
        }

        // Add hex dump if available
        if let Some(hex) = &self.hex_dump {
            write!(f, "\n  Hex dump: {}", hex)?;
        }

        // Add suggestion if available
        if let Some(suggestion) = &self.suggestion {
            write!(f, "\n  Suggestion: {}", suggestion)?;
        }

        // Add code example if available
        if let Some(code) = &self.code_example {
            write!(f, "\n  Code example: {}", code)?;
        }

        // Add technical details in verbose mode (if requested)
        if f.alternate()
            && let Some(tech) = &self.technical_details
        {
            write!(f, "\n🔍 Technical details: {}", tech)?;
        }

        // Show related errors
        if !self.related_errors.is_empty() {
            write!(f, "\n\n📋 Related errors in this frame:")?;
            for (i, related) in self.related_errors.iter().enumerate() {
                write!(f, "\n  {}. {}", i + 1, related)?;
            }
        }

        Ok(())
    }
}

impl AdtsValidationError {
    /// Get a JSON representation of this error for programmatic handling.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Get a compact JSON representation.
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Check if this error is critical (prevents muxing).
    pub fn is_critical(&self) -> bool {
        matches!(self.severity, ErrorSeverity::Error)
    }

    /// Get all errors in this chain (including related errors).
    pub fn all_errors(&self) -> Vec<&AdtsValidationError> {
        let mut result = vec![self];
        for related in &self.related_errors {
            result.extend(related.all_errors());
        }
        result
    }
}

/// Errors produced while queuing video samples.
#[derive(Debug)]
pub enum Mp4WriterError {
    /// Video frames must have strictly increasing timestamps.
    NonIncreasingTimestamp,
    /// The first frame must be a keyframe containing SPS/PPS data.
    FirstFrameMustBeKeyframe,
    /// The first keyframe must include SPS and PPS NAL units.
    FirstFrameMissingSpsPps,
    /// The first AV1 keyframe must include a Sequence Header OBU.
    FirstFrameMissingSequenceHeader,
    /// The first VP9 keyframe must include valid frame header parameters.
    FirstFrameMissingVp9Config,
    /// Audio sample is not a valid ADTS frame.
    #[allow(dead_code)]
    InvalidAdts,
    /// Audio sample has detailed ADTS validation errors.
    InvalidAdtsDetailed(Box<AdtsValidationError>),
    /// Audio sample is not a valid Opus packet.
    InvalidOpusPacket,
    /// Audio sample is not a valid FLAC frame.
    InvalidFlacFrame,
    /// FLAC track is missing its STREAMINFO configuration.
    MissingFlacStreaminfo,
    /// Audio track is not enabled on this writer.
    AudioNotEnabled,
    /// Subtitle track is not enabled on this writer.
    SubtitleNotEnabled,
    /// Computed sample duration overflowed a `u32`.
    DurationOverflow,
    /// Decoder configuration changed after the track started (verdict §8).
    DecoderConfigurationChanged,
    /// The writer has already been finalised.
    AlreadyFinalized,
    /// Low-level IO error.
    Io(std::io::Error),
}

impl fmt::Display for Mp4WriterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mp4WriterError::NonIncreasingTimestamp => write!(f, "timestamps must grow"),
            Mp4WriterError::FirstFrameMustBeKeyframe => {
                write!(f, "first frame must be a keyframe")
            }
            Mp4WriterError::FirstFrameMissingSpsPps => {
                write!(f, "first frame must contain SPS/PPS")
            }
            Mp4WriterError::FirstFrameMissingSequenceHeader => {
                write!(f, "first AV1 frame must contain Sequence Header OBU")
            }
            Mp4WriterError::FirstFrameMissingVp9Config => {
                write!(f, "first VP9 frame must contain valid frame header")
            }
            Mp4WriterError::InvalidAdts => write!(f, "invalid ADTS frame"),
            Mp4WriterError::InvalidAdtsDetailed(err) => write!(f, "{}", err),
            Mp4WriterError::InvalidOpusPacket => write!(f, "invalid Opus packet"),
            Mp4WriterError::InvalidFlacFrame => write!(f, "invalid FLAC frame"),
            Mp4WriterError::MissingFlacStreaminfo => {
                write!(f, "FLAC track is missing its STREAMINFO configuration")
            }
            Mp4WriterError::AudioNotEnabled => write!(f, "audio track not enabled"),
            Mp4WriterError::SubtitleNotEnabled => write!(f, "subtitle track not enabled"),
            Mp4WriterError::DurationOverflow => write!(f, "sample duration overflow"),
            Mp4WriterError::DecoderConfigurationChanged => {
                write!(f, "decoder configuration changed after the track started")
            }
            Mp4WriterError::AlreadyFinalized => write!(f, "writer already finalised"),
            Mp4WriterError::Io(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl std::error::Error for Mp4WriterError {}

impl<Writer: Write> Mp4Writer<Writer> {
    /// Wraps the provided writer for MP4 container output.
    pub fn new(writer: Writer) -> Self {
        Self {
            writer,
            video_codec: None,
            video_samples: Vec::new(),
            video_prev_pts: None,
            video_last_delta: None,
            video_config: None,
            caller_video_config: None,
            audio_track: None,
            audio_samples: Vec::new(),
            audio_prev_pts: None,
            audio_last_delta: None,
            subtitle_track: None,
            subtitle_samples: Vec::new(),
            subtitle_prev_pts: None,
            subtitle_last_delta: None,
            finalized: false,
            bytes_written: 0,
        }
    }

    pub(crate) fn video_sample_count(&self) -> u64 {
        self.video_samples.len() as u64
    }
    pub(crate) fn audio_sample_count(&self) -> u64 {
        self.audio_samples.len() as u64
    }

    pub(crate) fn subtitle_sample_count(&self) -> u64 {
        self.subtitle_samples.len() as u64
    }

    pub(crate) fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// State accessors for the seekable-streaming writer (§5), which owns an
    /// `Mp4Writer<io::Sink>` for validation/conversion and streams bytes
    /// itself. Buffered behavior is unchanged.
    pub(crate) fn video_config(&self) -> Option<&VideoConfig> {
        self.video_config.as_ref()
    }

    pub(crate) fn audio_track_ref(&self) -> Option<&Mp4AudioTrack> {
        self.audio_track.as_ref()
    }

    pub(crate) fn subtitle_track_ref(&self) -> Option<&Mp4SubtitleTrack> {
        self.subtitle_track.as_ref()
    }

    pub(crate) fn video_last_delta(&self) -> Option<u32> {
        self.video_last_delta
    }

    pub(crate) fn audio_last_delta(&self) -> Option<u32> {
        self.audio_last_delta
    }

    pub(crate) fn subtitle_last_delta(&self) -> Option<u32> {
        self.subtitle_last_delta
    }

    pub(crate) fn max_end_pts(&self) -> Option<u64> {
        fn track_end(samples: &[SampleInfo], last_delta: Option<u32>) -> Option<u64> {
            let last = samples.last()?;
            Some(last.pts + u64::from(last_delta.unwrap_or(0)))
        }

        let video_end = track_end(&self.video_samples, self.video_last_delta);
        let audio_end = track_end(&self.audio_samples, self.audio_last_delta);
        let subtitle_end = track_end(&self.subtitle_samples, self.subtitle_last_delta);

        [video_end, audio_end, subtitle_end]
            .into_iter()
            .flatten()
            .max()
    }

    fn write_counted(writer: &mut Writer, bytes_written: &mut u64, buf: &[u8]) -> io::Result<()> {
        *bytes_written = bytes_written.saturating_add(buf.len() as u64);
        writer.write_all(buf)
    }

    pub fn enable_video(&mut self, codec: VideoCodec) {
        self.video_codec = Some(codec);
    }

    /// Enable video with a caller-supplied decoder config (verdict §8).
    pub fn enable_video_with_config(
        &mut self,
        codec: VideoCodec,
        config: crate::api::VideoDecoderConfig,
    ) {
        self.video_codec = Some(codec);
        self.caller_video_config = Some(config);
    }

    pub fn enable_audio(&mut self, track: Mp4AudioTrack) {
        self.audio_track = Some(track);
    }

    pub fn enable_subtitle(&mut self, track: Mp4SubtitleTrack) {
        self.subtitle_track = Some(track);
    }

    /// Queues a video sample for later `mdat` emission.
    /// For backward compatibility, dts is assumed equal to pts.
    pub fn write_video_sample(
        &mut self,
        pts: u64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), Mp4WriterError> {
        self.write_video_sample_with_dts(pts, pts, data, is_keyframe)
    }

    /// Queues a video sample with explicit DTS for B-frame support.
    /// - `pts`: Presentation timestamp (display order)
    /// - `dts`: Decode timestamp (decode order) - must be monotonically increasing
    pub fn write_video_sample_with_dts(
        &mut self,
        pts: u64,
        dts: u64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), Mp4WriterError> {
        let converted = self.prepare_video_sample(pts, dts, data, is_keyframe)?;
        // Back-patch the previous sample's duration from the observed delta,
        // mirroring the pre-split behavior (prepare only records the delta).
        if let Some(last) = self.video_samples.last_mut() {
            last.duration = self.video_last_delta;
        }
        self.video_samples.push(SampleInfo {
            pts,
            dts,
            data: converted,
            is_keyframe,
            duration: None,
        });
        Ok(())
    }

    /// Validate, convert (Annex B -> length-prefixed), and extract/check
    /// decoder configuration for one video sample without buffering it.
    ///
    /// Shared by the buffered writer and the seekable-streaming writer (§5)
    /// so both paths enforce identical semantics; only the sink differs.
    pub(crate) fn prepare_video_sample(
        &mut self,
        _pts: u64,
        dts: u64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<Vec<u8>, Mp4WriterError> {
        if self.finalized {
            return Err(Mp4WriterError::AlreadyFinalized);
        }
        let video_codec = self.video_codec.ok_or_else(|| {
            Mp4WriterError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "video track not enabled: enable_video() must not have been called before write_video_sample()",
            ))
        })?;
        // DTS must be monotonically increasing (decode order)
        if let Some(prev) = self.video_prev_pts {
            if dts <= prev {
                return Err(Mp4WriterError::NonIncreasingTimestamp);
            }
            let delta = dts - prev;
            if delta > u64::from(u32::MAX) {
                return Err(Mp4WriterError::DurationOverflow);
            }
            self.video_last_delta = Some(delta as u32);
        } else {
            if !is_keyframe {
                return Err(Mp4WriterError::FirstFrameMustBeKeyframe);
            }
            // Extract codec configuration based on video codec type.
            // When a caller config was supplied, still extract to validate
            // the stream carries parameter sets (raw passthrough is 0.2 work).
            let config = match video_codec {
                VideoCodec::H264 => extract_avc_config(data).map(VideoConfig::Avc),
                VideoCodec::H265 => extract_hevc_config(data).map(VideoConfig::Hevc),
                VideoCodec::Av1 => extract_av1_config(data).map(VideoConfig::Av1),
                VideoCodec::Vp9 => extract_vp9_config(data).map(VideoConfig::Vp9),
            };
            if config.is_none() {
                return Err(match video_codec {
                    VideoCodec::Av1 => Mp4WriterError::FirstFrameMissingSequenceHeader,
                    VideoCodec::Vp9 => Mp4WriterError::FirstFrameMissingVp9Config,
                    _ => Mp4WriterError::FirstFrameMissingSpsPps,
                });
            }
            self.video_config = config;
        }
        // Verdict §8: detect parameter-set changes after initialization.
        // Only keyframes can carry new PS; compare against stored config.
        // Gated on `video_prev_pts` (not buffer length) so the
        // seekable-streaming writer (§5) enforces this identically.
        if is_keyframe && self.video_prev_pts.is_some() {
            let changed = match (video_codec, &self.video_config) {
                (VideoCodec::H264, Some(VideoConfig::Avc(old))) => extract_avc_config(data)
                    .is_some_and(|new| {
                        new.sps != old.sps
                            || new.pps != old.pps
                            || new.sequence_parameter_sets != old.sequence_parameter_sets
                            || new.picture_parameter_sets != old.picture_parameter_sets
                    }),
                (VideoCodec::H265, Some(VideoConfig::Hevc(old))) => extract_hevc_config(data)
                    .is_some_and(|new| {
                        new.vps != old.vps || new.sps != old.sps || new.pps != old.pps
                    }),
                (VideoCodec::Av1, Some(VideoConfig::Av1(old))) => {
                    extract_av1_config(data).is_some_and(|new| new != *old)
                }
                (VideoCodec::Vp9, Some(VideoConfig::Vp9(old))) => {
                    extract_vp9_config(data).is_some_and(|new| new != *old)
                }
                _ => false,
            };
            if changed {
                return Err(Mp4WriterError::DecoderConfigurationChanged);
            }
        }

        // Convert Annex B to length-prefixed format based on codec
        // AV1 uses OBU format which doesn't need conversion
        let converted = match video_codec {
            VideoCodec::H264 => annexb_to_avcc(data),
            VideoCodec::H265 => hevc_annexb_to_hvcc(data),
            VideoCodec::Av1 => data.to_vec(), // AV1 OBUs passed as-is
            VideoCodec::Vp9 => data.to_vec(), // VP9 compressed frames passed as-is
        };
        if converted.len() > u32::MAX as usize {
            return Err(Mp4WriterError::DurationOverflow);
        }

        self.video_prev_pts = Some(dts); // Track DTS for monotonic check
        Ok(converted)
    }

    pub fn write_audio_sample(&mut self, pts: u64, data: &[u8]) -> Result<(), Mp4WriterError> {
        let sample_data = self.prepare_audio_sample(pts, data)?;
        // Back-patch the previous sample's duration from the observed delta.
        if let Some(last) = self.audio_samples.last_mut() {
            last.duration = self.audio_last_delta;
        }
        self.audio_samples.push(SampleInfo {
            pts,
            dts: pts, // Audio: dts == pts (no B-frames)
            data: sample_data,
            is_keyframe: false,
            duration: None,
        });
        Ok(())
    }

    /// Validate and convert one audio sample without buffering it.
    ///
    /// Shared by the buffered writer and the seekable-streaming writer (§5).
    /// Updates `audio_prev_pts`/`audio_last_delta` but leaves duration
    /// back-patching to the caller (buffered: previous `SampleInfo`;
    /// streaming: previous `SampleMeta`).
    pub(crate) fn prepare_audio_sample(
        &mut self,
        pts: u64,
        data: &[u8],
    ) -> Result<Vec<u8>, Mp4WriterError> {
        if self.finalized {
            return Err(Mp4WriterError::AlreadyFinalized);
        }
        let audio_track = self
            .audio_track
            .as_ref()
            .ok_or(Mp4WriterError::AudioNotEnabled)?;

        if let Some(prev) = self.audio_prev_pts {
            if pts < prev {
                return Err(Mp4WriterError::NonIncreasingTimestamp);
            }
            let delta = pts - prev;
            if delta > u64::from(u32::MAX) {
                return Err(Mp4WriterError::DurationOverflow);
            }
            self.audio_last_delta = Some(delta as u32);
        }

        // Process audio data based on codec
        let sample_data = match audio_track.codec {
            AudioCodec::Aac(profile) => {
                // INV-020: AAC profile must be supported
                assert_invariant!(
                    matches!(
                        profile,
                        AacProfile::Lc
                            | AacProfile::Main
                            | AacProfile::Ssr
                            | AacProfile::Ltp
                            | AacProfile::He
                            | AacProfile::Hev2
                    ),
                    "INV-020: AAC profile must be one of the supported variants",
                    "aac audio processing"
                );

                let raw = adts_to_raw(data)
                    .map_err(|e| Mp4WriterError::InvalidAdtsDetailed(Box::new(e)))?;
                raw.to_vec()
            }
            AudioCodec::Opus => {
                // Validate Opus packet structure
                if !is_valid_opus_packet(data) {
                    return Err(Mp4WriterError::InvalidOpusPacket);
                }
                // Opus packets are passed through as-is (no container framing)
                data.to_vec()
            }
            AudioCodec::Flac => {
                // STREAMINFO must be configured up front (the `dfLa`
                // box is built from it at finalize time).
                if parse_streaminfo(audio_track.flac_streaminfo.as_deref().unwrap_or(&[])).is_none()
                {
                    return Err(Mp4WriterError::MissingFlacStreaminfo);
                }
                // FLAC frames are passed through as-is (one frame per sample);
                // only the header sync is validated here.
                if !is_valid_flac_frame(data) {
                    return Err(Mp4WriterError::InvalidFlacFrame);
                }
                data.to_vec()
            }
            AudioCodec::None => {
                return Err(Mp4WriterError::AudioNotEnabled);
            }
        };

        if sample_data.len() > u32::MAX as usize {
            return Err(Mp4WriterError::DurationOverflow);
        }

        self.audio_prev_pts = Some(pts);
        Ok(sample_data)
    }

    pub fn write_subtitle_sample(
        &mut self,
        pts: u64,
        duration: u32,
        data: &[u8],
    ) -> Result<(), Mp4WriterError> {
        let (bytes, duration) = self.prepare_subtitle_sample(pts, duration, data)?;
        self.subtitle_samples.push(SampleInfo {
            pts,
            dts: pts,
            data: bytes,
            is_keyframe: false,
            duration: Some(duration),
        });
        Ok(())
    }

    /// Validate one subtitle sample without buffering it (§5).
    ///
    /// Returns the owned payload plus the effective duration (`max(1)`),
    /// mirroring the buffered path exactly.
    pub(crate) fn prepare_subtitle_sample(
        &mut self,
        pts: u64,
        duration: u32,
        data: &[u8],
    ) -> Result<(Vec<u8>, u32), Mp4WriterError> {
        if self.finalized {
            return Err(Mp4WriterError::AlreadyFinalized);
        }
        if self.subtitle_track.is_none() {
            return Err(Mp4WriterError::SubtitleNotEnabled);
        }
        if let Some(prev) = self.subtitle_prev_pts
            && pts < prev
        {
            return Err(Mp4WriterError::NonIncreasingTimestamp);
        }
        if data.is_empty() {
            return Err(Mp4WriterError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "subtitle sample is empty",
            )));
        }
        if data.len() > u32::MAX as usize {
            return Err(Mp4WriterError::DurationOverflow);
        }

        let duration = duration.max(1);
        self.subtitle_prev_pts = Some(pts);
        self.subtitle_last_delta = Some(duration);
        Ok((data.to_vec(), duration))
    }

    /// Override the duration of the most recently queued video sample.
    /// Used by the integer-time API to enforce explicit durations (verdict §4).
    pub(crate) fn set_last_video_duration(&mut self, duration: u32) {
        if let Some(last) = self.video_samples.last_mut() {
            last.duration = Some(duration);
        }
        self.video_last_delta = Some(duration);
    }

    /// Override the duration of the most recently queued audio sample.
    pub(crate) fn set_last_audio_duration(&mut self, duration: u32) {
        if let Some(last) = self.audio_samples.last_mut() {
            last.duration = Some(duration);
        }
        self.audio_last_delta = Some(duration);
    }

    /// Encode a `tx3g` sample: u16 length + UTF-8 bytes, checked (verdict §9).
    pub fn encode_tx3g_sample(text: &str) -> Result<Vec<u8>, Mp4WriterError> {
        let bytes = text.as_bytes();
        let length = u16::try_from(bytes.len()).map_err(|_| Mp4WriterError::DurationOverflow)?;
        let mut output = Vec::with_capacity(2 + bytes.len());
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(bytes);
        Ok(output)
    }

    /// Finalises the MP4 file by writing the header boxes and sample data.
    pub fn finalize(
        &mut self,
        video: Option<&Mp4VideoTrack>,
        metadata: Option<&Metadata>,
        fast_start: bool,
    ) -> io::Result<()> {
        if self.finalized {
            return Err(io::Error::other("mp4 writer already finalised"));
        }
        self.finalized = true;

        let video_config = self.video_config.clone().or_else(|| {
            if self.video_samples.is_empty() {
                match self.video_codec {
                    Some(VideoCodec::H264) => Some(VideoConfig::Avc(default_avc_config())),
                    _ => None,
                }
            } else {
                None
            }
        });

        if let Some(video_config) = video_config {
            if let Some(video) = video {
                if fast_start {
                    self.finalize_fast_start(video, metadata, &video_config)
                } else {
                    self.finalize_standard(video, metadata, &video_config)
                }
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "video config extracted but no video track provided",
                ))
            }
        } else {
            // Audio-only mode
            self.finalize_audio_only(metadata)
        }
    }

    fn finalize_standard(
        &mut self,
        video: &Mp4VideoTrack,
        metadata: Option<&Metadata>,
        video_config: &VideoConfig,
    ) -> io::Result<()> {
        let ftyp_box = build_ftyp_box();
        let ftyp_len = ftyp_box.len() as u64;
        Self::write_counted(&mut self.writer, &mut self.bytes_written, &ftyp_box)?;

        let audio_present = self.audio_track.is_some();
        let subtitle_present = self.subtitle_track.is_some();

        if !audio_present && !subtitle_present {
            let chunk_offset = if !self.video_samples.is_empty() {
                let mut payload_size: u64 = 0;
                for sample in &self.video_samples {
                    payload_size = payload_size
                        .checked_add(sample.data.len() as u64)
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "MP4 payload size overflow")
                        })?;
                }

                let header = mdat_header_bytes(payload_size)?;
                let header_len = header.len() as u64;
                Self::write_counted(&mut self.writer, &mut self.bytes_written, &header)?;
                for sample in &self.video_samples {
                    Self::write_counted(&mut self.writer, &mut self.bytes_written, &sample.data)?;
                }
                Some(ftyp_len + header_len)
            } else {
                None
            };

            let (chunk_offsets, samples_per_chunk) = match chunk_offset {
                Some(offset) => (
                    vec![offset],
                    u32::try_from(self.video_samples.len()).unwrap_or(u32::MAX),
                ),
                None => (Vec::new(), 0),
            };

            let tables = SampleTables::from_samples(
                &sample_metas(&self.video_samples),
                chunk_offsets,
                samples_per_chunk,
                self.video_last_delta,
            );
            let moov_box = build_moov_box(video, &tables, None, None, video_config, metadata);
            return Self::write_counted(&mut self.writer, &mut self.bytes_written, &moov_box);
        }

        // Multi-track present - write interleaved mdat then moov
        let mut total_payload_size: u64 = 0;
        for sample in &self.video_samples {
            total_payload_size = total_payload_size
                .checked_add(sample.data.len() as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "MP4 payload size overflow")
                })?;
        }
        for sample in &self.audio_samples {
            total_payload_size = total_payload_size
                .checked_add(sample.data.len() as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "MP4 payload size overflow")
                })?;
        }
        for sample in &self.subtitle_samples {
            total_payload_size = total_payload_size
                .checked_add(sample.data.len() as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "MP4 payload size overflow")
                })?;
        }

        let mdat_header = mdat_header_bytes(total_payload_size)?;
        let mdat_hlen = mdat_header.len() as u64;
        Self::write_counted(&mut self.writer, &mut self.bytes_written, &mdat_header)?;

        // Write interleaved samples and collect chunk offsets
        let schedule = self.compute_interleave_schedule();
        let mut video_chunk_offsets = Vec::with_capacity(self.video_samples.len());
        let mut audio_chunk_offsets = Vec::with_capacity(self.audio_samples.len());
        let mut subtitle_chunk_offsets = Vec::with_capacity(self.subtitle_samples.len());
        let mut cursor = ftyp_len + mdat_hlen; // After ftyp + mdat header

        for (_, kind, idx) in schedule {
            match kind {
                TrackKind::Video => {
                    video_chunk_offsets.push(cursor);
                    let sample = &self.video_samples[idx];
                    let sample_len = sample.data.len() as u64;
                    Self::write_counted(&mut self.writer, &mut self.bytes_written, &sample.data)?;
                    cursor = cursor.saturating_add(sample_len);
                }
                TrackKind::Audio => {
                    audio_chunk_offsets.push(cursor);
                    let sample = &self.audio_samples[idx];
                    let sample_len = sample.data.len() as u64;
                    Self::write_counted(&mut self.writer, &mut self.bytes_written, &sample.data)?;
                    cursor = cursor.saturating_add(sample_len);
                }
                TrackKind::Subtitle => {
                    subtitle_chunk_offsets.push(cursor);
                    let sample = &self.subtitle_samples[idx];
                    let sample_len = sample.data.len() as u64;
                    Self::write_counted(&mut self.writer, &mut self.bytes_written, &sample.data)?;
                    cursor = cursor.saturating_add(sample_len);
                }
            }
        }

        let video_tables = SampleTables::from_samples(
            &sample_metas(&self.video_samples),
            video_chunk_offsets,
            1,
            self.video_last_delta,
        );
        let audio_tables = SampleTables::from_samples(
            &sample_metas(&self.audio_samples),
            audio_chunk_offsets,
            1,
            self.audio_last_delta,
        );
        let subtitle_tables = SampleTables::from_samples(
            &sample_metas(&self.subtitle_samples),
            subtitle_chunk_offsets,
            1,
            self.subtitle_last_delta,
        );

        let audio = if audio_present {
            Some((
                self.audio_track
                    .as_ref()
                    .expect("audio_present implies track"),
                &audio_tables,
            ))
        } else {
            None
        };
        let subtitle = self
            .subtitle_track
            .as_ref()
            .map(|track| (track, &subtitle_tables));
        let moov_box = build_moov_box(
            video,
            &video_tables,
            audio,
            subtitle,
            video_config,
            metadata,
        );
        Self::write_counted(&mut self.writer, &mut self.bytes_written, &moov_box)
    }

    fn finalize_audio_only(&mut self, metadata: Option<&Metadata>) -> io::Result<()> {
        let audio_track = self.audio_track.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "no audio track configured")
        })?;

        let ftyp_box = build_ftyp_box();
        let ftyp_len = ftyp_box.len() as u64;

        // Calculate mdat size
        let mut mdat_payload_size: u64 = 0;
        for sample in &self.audio_samples {
            mdat_payload_size = mdat_payload_size
                .checked_add(sample.data.len() as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "MP4 payload size overflow")
                })?;
        }
        let mdat_header_size = mdat_header_len(mdat_payload_size);

        // Build audio sample tables
        let chunk_offsets = if self.audio_samples.is_empty() {
            Vec::new()
        } else {
            vec![0u64]
        };
        let samples_per_chunk = if self.audio_samples.is_empty() {
            0
        } else {
            u32::try_from(self.audio_samples.len()).unwrap_or(u32::MAX)
        };
        let audio_tables = SampleTables::from_samples(
            &sample_metas(&self.audio_samples),
            chunk_offsets,
            samples_per_chunk,
            self.audio_last_delta,
        );

        // Build moov with placeholder to measure size
        let placeholder_moov = build_audio_only_moov_box(audio_track, &audio_tables, metadata);
        let moov_len = placeholder_moov.len() as u64;

        // mdat starts at ftyp_len + moov_len
        let mdat_data_start = ftyp_len + moov_len + mdat_header_size;

        // Rebuild moov with correct offsets (u64; co64 selected automatically)
        let final_chunk_offsets = if self.audio_samples.is_empty() {
            Vec::new()
        } else {
            vec![mdat_data_start]
        };
        let final_audio_tables = SampleTables::from_samples(
            &sample_metas(&self.audio_samples),
            final_chunk_offsets,
            samples_per_chunk,
            self.audio_last_delta,
        );
        let mut final_moov = build_audio_only_moov_box(audio_track, &final_audio_tables, metadata);
        // If co64 switch changed moov size, recompute once.
        if final_moov.len() as u64 != moov_len {
            let start2 = ftyp_len + final_moov.len() as u64 + mdat_header_size;
            let t2 = SampleTables::from_samples(
                &sample_metas(&self.audio_samples),
                if self.audio_samples.is_empty() {
                    Vec::new()
                } else {
                    vec![start2]
                },
                samples_per_chunk,
                self.audio_last_delta,
            );
            final_moov = build_audio_only_moov_box(audio_track, &t2, metadata);
        }

        // Write: ftyp -> moov -> mdat header -> samples
        Self::write_counted(&mut self.writer, &mut self.bytes_written, &ftyp_box)?;
        Self::write_counted(&mut self.writer, &mut self.bytes_written, &final_moov)?;
        Self::write_counted(
            &mut self.writer,
            &mut self.bytes_written,
            &mdat_header_bytes(mdat_payload_size)?,
        )?;

        for sample in &self.audio_samples {
            Self::write_counted(&mut self.writer, &mut self.bytes_written, &sample.data)?;
        }

        Ok(())
    }

    fn finalize_fast_start(
        &mut self,
        video: &Mp4VideoTrack,
        metadata: Option<&Metadata>,
        video_config: &VideoConfig,
    ) -> io::Result<()> {
        let ftyp_box = build_ftyp_box();
        let ftyp_len = ftyp_box.len() as u64;

        // Calculate total mdat payload size
        let mut mdat_payload_size: u64 = 0;
        for sample in &self.video_samples {
            mdat_payload_size = mdat_payload_size
                .checked_add(sample.data.len() as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "MP4 payload size overflow")
                })?;
        }
        for sample in &self.audio_samples {
            mdat_payload_size = mdat_payload_size
                .checked_add(sample.data.len() as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "MP4 payload size overflow")
                })?;
        }
        for sample in &self.subtitle_samples {
            mdat_payload_size = mdat_payload_size
                .checked_add(sample.data.len() as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "MP4 payload size overflow")
                })?;
        }
        let mdat_header_size = mdat_header_len(mdat_payload_size);

        let audio_present = self.audio_track.is_some();
        let subtitle_present = self.subtitle_track.is_some();

        // Build moov with placeholder offsets to measure its size
        let (placeholder_video_tables, placeholder_audio_tables, placeholder_subtitle_tables) =
            if audio_present || subtitle_present {
                let schedule = self.compute_interleave_schedule();

                let mut video_offsets: Vec<u64> = Vec::with_capacity(self.video_samples.len());
                let mut audio_offsets: Vec<u64> = Vec::with_capacity(self.audio_samples.len());
                let mut subtitle_offsets: Vec<u64> =
                    Vec::with_capacity(self.subtitle_samples.len());
                let mut cursor = 0u64;
                for (_, kind, _) in &schedule {
                    match kind {
                        TrackKind::Video => {
                            video_offsets.push(cursor);
                            cursor += 1;
                        }
                        TrackKind::Audio => {
                            audio_offsets.push(cursor);
                            cursor += 1;
                        }
                        TrackKind::Subtitle => {
                            subtitle_offsets.push(cursor);
                            cursor += 1;
                        }
                    }
                }

                let video_tables = SampleTables::from_samples(
                    &sample_metas(&self.video_samples),
                    video_offsets,
                    1,
                    self.video_last_delta,
                );
                let audio_tables = SampleTables::from_samples(
                    &sample_metas(&self.audio_samples),
                    audio_offsets,
                    1,
                    self.audio_last_delta,
                );
                let subtitle_tables = SampleTables::from_samples(
                    &sample_metas(&self.subtitle_samples),
                    subtitle_offsets,
                    1,
                    self.subtitle_last_delta,
                );
                (
                    video_tables,
                    if audio_present {
                        Some(audio_tables)
                    } else {
                        None
                    },
                    if subtitle_present {
                        Some(subtitle_tables)
                    } else {
                        None
                    },
                )
            } else {
                // Video-only: all samples in one chunk
                let chunk_offsets = if self.video_samples.is_empty() {
                    Vec::new()
                } else {
                    vec![0u64] // Single placeholder chunk offset (will be replaced with real value)
                };
                let samples_per_chunk = if self.video_samples.is_empty() {
                    0
                } else {
                    u32::try_from(self.video_samples.len()).unwrap_or(u32::MAX)
                };
                let video_tables = SampleTables::from_samples(
                    &sample_metas(&self.video_samples),
                    chunk_offsets,
                    samples_per_chunk,
                    self.video_last_delta,
                );
                (video_tables, None, None)
            };

        let placeholder_audio = placeholder_audio_tables
            .as_ref()
            .map(|tables| (self.audio_track.as_ref().unwrap(), tables));
        let placeholder_subtitle = placeholder_subtitle_tables
            .as_ref()
            .map(|tables| (self.subtitle_track.as_ref().unwrap(), tables));
        let placeholder_moov = build_moov_box(
            video,
            &placeholder_video_tables,
            placeholder_audio,
            placeholder_subtitle,
            video_config,
            metadata,
        );
        let moov_len = placeholder_moov.len() as u64;

        // Now we know: mdat starts at ftyp_len + moov_len
        let mdat_data_start = ftyp_len + moov_len + mdat_header_size;

        // Rebuild moov with correct offsets (u64; co64 auto-selected)
        let (final_video_tables, final_audio_tables, final_subtitle_tables) = if audio_present
            || subtitle_present
        {
            let schedule = self.compute_interleave_schedule();

            let mut video_offsets: Vec<u64> = Vec::with_capacity(self.video_samples.len());
            let mut audio_offsets: Vec<u64> = Vec::with_capacity(self.audio_samples.len());
            let mut subtitle_offsets: Vec<u64> = Vec::with_capacity(self.subtitle_samples.len());
            let mut cursor = mdat_data_start;

            for (_, kind, idx) in &schedule {
                match kind {
                    TrackKind::Video => {
                        video_offsets.push(cursor);
                        cursor = cursor.saturating_add(self.video_samples[*idx].data.len() as u64);
                    }
                    TrackKind::Audio => {
                        audio_offsets.push(cursor);
                        cursor = cursor.saturating_add(self.audio_samples[*idx].data.len() as u64);
                    }
                    TrackKind::Subtitle => {
                        subtitle_offsets.push(cursor);
                        cursor =
                            cursor.saturating_add(self.subtitle_samples[*idx].data.len() as u64);
                    }
                }
            }

            let video_tables = SampleTables::from_samples(
                &sample_metas(&self.video_samples),
                video_offsets,
                1,
                self.video_last_delta,
            );
            let audio_tables = SampleTables::from_samples(
                &sample_metas(&self.audio_samples),
                audio_offsets,
                1,
                self.audio_last_delta,
            );
            let subtitle_tables = SampleTables::from_samples(
                &sample_metas(&self.subtitle_samples),
                subtitle_offsets,
                1,
                self.subtitle_last_delta,
            );
            (
                video_tables,
                if audio_present {
                    Some(audio_tables)
                } else {
                    None
                },
                if subtitle_present {
                    Some(subtitle_tables)
                } else {
                    None
                },
            )
        } else {
            // Video only - all samples in one chunk
            let chunk_offsets = if self.video_samples.is_empty() {
                Vec::new()
            } else {
                vec![mdat_data_start]
            };
            let samples_per_chunk = if self.video_samples.is_empty() {
                0
            } else {
                u32::try_from(self.video_samples.len()).unwrap_or(u32::MAX)
            };
            let video_tables = SampleTables::from_samples(
                &sample_metas(&self.video_samples),
                chunk_offsets,
                samples_per_chunk,
                self.video_last_delta,
            );
            (video_tables, None, None)
        };

        let final_audio = final_audio_tables
            .as_ref()
            .map(|tables| (self.audio_track.as_ref().unwrap(), tables));
        let final_subtitle = final_subtitle_tables
            .as_ref()
            .map(|tables| (self.subtitle_track.as_ref().unwrap(), tables));
        let final_moov = build_moov_box(
            video,
            &final_video_tables,
            final_audio,
            final_subtitle,
            video_config,
            metadata,
        );

        // Write: ftyp → moov → mdat header → samples
        Self::write_counted(&mut self.writer, &mut self.bytes_written, &ftyp_box)?;
        Self::write_counted(&mut self.writer, &mut self.bytes_written, &final_moov)?;
        Self::write_counted(
            &mut self.writer,
            &mut self.bytes_written,
            &mdat_header_bytes(mdat_payload_size)?,
        )?;

        // Write samples in interleaved order
        if audio_present || subtitle_present {
            let schedule = self.compute_interleave_schedule();
            for (_, kind, idx) in schedule {
                match kind {
                    TrackKind::Video => {
                        Self::write_counted(
                            &mut self.writer,
                            &mut self.bytes_written,
                            &self.video_samples[idx].data,
                        )?;
                    }
                    TrackKind::Audio => {
                        Self::write_counted(
                            &mut self.writer,
                            &mut self.bytes_written,
                            &self.audio_samples[idx].data,
                        )?;
                    }
                    TrackKind::Subtitle => {
                        Self::write_counted(
                            &mut self.writer,
                            &mut self.bytes_written,
                            &self.subtitle_samples[idx].data,
                        )?;
                    }
                }
            }
        } else {
            for sample in &self.video_samples {
                Self::write_counted(&mut self.writer, &mut self.bytes_written, &sample.data)?;
            }
        }

        Ok(())
    }

    fn compute_interleave_schedule(&self) -> Vec<(u64, TrackKind, usize)> {
        let mut schedule: Vec<(u64, TrackKind, usize)> = Vec::new();
        for (idx, sample) in self.video_samples.iter().enumerate() {
            schedule.push((sample.pts, TrackKind::Video, idx));
        }
        for (idx, sample) in self.audio_samples.iter().enumerate() {
            schedule.push((sample.pts, TrackKind::Audio, idx));
        }
        for (idx, sample) in self.subtitle_samples.iter().enumerate() {
            schedule.push((sample.pts, TrackKind::Subtitle, idx));
        }
        schedule.sort_by_key(|(pts, kind, idx)| {
            let kind_order = match kind {
                TrackKind::Video => 0u8,
                TrackKind::Audio => 1u8,
                TrackKind::Subtitle => 2u8,
            };
            (*pts, kind_order, *idx)
        });
        schedule
    }
}

#[derive(Clone, Copy)]
enum TrackKind {
    Video,
    Audio,
    Subtitle,
}

#[allow(clippy::result_large_err)]
pub(crate) fn adts_to_raw(frame: &[u8]) -> Result<&[u8], AdtsValidationError> {
    // Enhanced hex dump with ASCII and color highlighting
    let create_hex_dump = |offset: usize, len: usize| -> String {
        let start = offset.saturating_sub(8).min(frame.len());
        let end = (offset + len + 8).min(frame.len());
        let slice = &frame[start..end];

        let mut hex = String::new();
        let mut ascii = String::new();

        for (i, &byte) in slice.iter().enumerate() {
            let global_offset = start + i;

            // Highlight error byte with red and asterisk
            if global_offset == offset {
                hex.push_str(&format!("\x1b[91m{:02x}*\x1b[0m ", byte));
            } else if global_offset >= offset && global_offset < offset + len {
                hex.push_str(&format!("\x1b[93m{:02x}\x1b[0m ", byte)); // Yellow for context
            } else {
                hex.push_str(&format!("{:02x} ", byte));
            }

            // ASCII representation
            let ascii_char = if byte.is_ascii_graphic() {
                byte as char
            } else {
                '.'
            };
            if global_offset == offset {
                ascii.push_str(&format!("\x1b[91m{}\x1b[0m", ascii_char));
            } else if global_offset >= offset && global_offset < offset + len {
                ascii.push_str(&format!("\x1b[93m{}\x1b[0m", ascii_char));
            } else {
                ascii.push(ascii_char);
            }

            // Line breaks every 16 bytes
            if (i + 1) % 16 == 0 {
                hex.push_str(&format!(" |{}|\n", ascii));
                ascii.clear();
            }
        }

        if !ascii.is_empty() {
            // Pad hex to align with ASCII
            while hex.chars().filter(|&c| c != '\x1b').count() % (16 * 3) != 0 {
                hex.push(' ');
            }
            hex.push_str(&format!(" |{}|", ascii));
        }

        format!("Hex dump around byte {}:\n{}", offset, hex)
    };

    if frame.len() < 7 {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::FrameTooShort,
            severity: ErrorSeverity::Error,
            byte_offset: frame.len(),
            expected: Some("≥7 bytes for ADTS header".to_string()),
            found: Some(format!("{} bytes", frame.len())),
            hex_dump: Some(create_hex_dump(0, frame.len())),
            suggestion: Some("Ensure you're passing complete ADTS frames. Check if the audio data is truncated or corrupted during transmission.".to_string()),
            code_example: Some("Ensure your audio frame buffer contains the complete ADTS frame before calling write_audio().".to_string()),
            technical_details: Some("ADTS header requires minimum 7 bytes: syncword (2 bytes), MPEG info (1 byte), frame length (3 bytes partial), buffer fullness (2 bytes partial).".to_string()),
            related_errors: Vec::new(),
        });
    }

    // Syncword validation: 0xFFF (12 bits) - first 12 bits should be 0xFFF
    let syncword = ((frame[0] as u16) << 4) | ((frame[1] as u16) >> 4);
    if syncword != 0xFFF {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::MissingSyncword,
            severity: ErrorSeverity::Error,
            byte_offset: 0,
            expected: Some("0xFFF (12-bit syncword)".to_string()),
            found: Some(format!("0x{:03X}", syncword)),
            hex_dump: Some(create_hex_dump(0, 2)),
            suggestion: Some("This doesn't appear to be an ADTS frame. Check if you're passing raw AAC data instead of ADTS-wrapped frames, or if the data is corrupted.".to_string()),
            code_example: Some("Check frame starts with ADTS syncword: if (frame[0] & 0xFF) == 0xFF && (frame[1] & 0xF0) == 0xF0 { /* valid ADTS */ }".to_string()),
            technical_details: Some("ADTS syncword is 0xFFF (all 1s in first 12 bits). If this is raw AAC, use AudioCodec::Aac without ADTS framing.".to_string()),
            related_errors: Vec::new(),
        });
    }

    // MPEG version check (bit 12 from syncword) - only MPEG-4 supported
    let mpeg_version = (frame[1] >> 3) & 0x01;
    if mpeg_version != 0 {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::InvalidMpegVersion,
            severity: ErrorSeverity::Error,
            byte_offset: 1,
            expected: Some("0 (MPEG-4)".to_string()),
            found: Some(format!("{} (MPEG-2)", mpeg_version)),
            hex_dump: Some(create_hex_dump(1, 1)),
            suggestion: Some(
                "Muxfin only supports MPEG-4 AAC. Convert your audio to MPEG-4 AAC format."
                    .to_string(),
            ),
            code_example: Some(
                "Use ffmpeg: ffmpeg -i input.mp3 -c:a aac -profile:a aac_low output.m4a"
                    .to_string(),
            ),
            technical_details: Some(
                "MPEG version bit: 0=MPEG-4, 1=MPEG-2. Muxfin requires MPEG-4 AAC.".to_string(),
            ),
            related_errors: Vec::new(),
        });
    }

    // Layer check (bits 13-14 from syncword) - must be 00 for AAC
    let layer = (frame[1] >> 1) & 0x03;
    if layer != 0 {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::InvalidLayer,
            severity: ErrorSeverity::Error,
            byte_offset: 1,
            expected: Some("0 (AAC)".to_string()),
            found: Some(format!("{} (Layer {})", layer, layer)),
            hex_dump: Some(create_hex_dump(1, 1)),
            suggestion: Some(
                "This appears to be MP3 or other MPEG audio format. Convert to AAC format."
                    .to_string(),
            ),
            code_example: Some(
                "Convert MP3 to AAC: ffmpeg -i input.mp3 -c:a aac -b:a 128k output.m4a".to_string(),
            ),
            technical_details: Some(
                "Layer field: 00=AAC, 01=Layer3, 10=Layer2, 11=Layer1. AAC requires 00."
                    .to_string(),
            ),
            related_errors: Vec::new(),
        });
    }

    let protection_absent = (frame[1] & 0x01) != 0;
    let header_len = if protection_absent { 7 } else { 9 };

    if frame.len() < header_len {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::InvalidHeaderLength,
            severity: ErrorSeverity::Error,
            byte_offset: 1,
            expected: Some(format!(
                "≥{} bytes (protection_absent={})",
                header_len, protection_absent
            )),
            found: Some(format!("{} bytes", frame.len())),
            hex_dump: Some(create_hex_dump(0, frame.len())),
            suggestion: Some(format!(
                "Frame is too short for {} header. Check if CRC protection is present and adjust header length calculation.",
                if protection_absent {
                    "unprotected"
                } else {
                    "protected"
                }
            )),
            code_example: None,
            technical_details: Some(format!(
                "Header length: 7 bytes (no CRC) or 9 bytes (with CRC). protection_absent bit: {}",
                protection_absent
            )),
            related_errors: Vec::new(),
        });
    }

    // Profile/Object type (bits 16-17)
    let profile = (frame[2] >> 6) & 0x03;
    let _profile_name = match profile {
        0 => "Main",
        1 => "LC (Low Complexity)",
        2 => "SSR (Scalable Sample Rate)",
        3 => "LTP (Long Term Prediction)",
        _ => "Unknown",
    };

    // Sample rate index (bits 18-21)
    let sample_rate_idx = (frame[2] >> 2) & 0x0F;
    if sample_rate_idx > 12 {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::InvalidSampleRateIndex,
            severity: ErrorSeverity::Error,
            byte_offset: 2,
            expected: Some("0-12 (96000-7350 Hz)".to_string()),
            found: Some(format!("{} (invalid)", sample_rate_idx)),
            hex_dump: Some(create_hex_dump(2, 1)),
            suggestion: Some("Invalid sample rate index. Valid values: 0=96000, 1=88200, 2=64000, 3=48000, 4=44100, 5=32000, 6=24000, 7=22050, 8=16000, 9=12000, 10=11025, 11=8000, 12=7350 Hz.".to_string()),
            code_example: Some("Common AAC sample rates: 44100 Hz (index 4), 48000 Hz (index 3), 22050 Hz (index 7)".to_string()),
            technical_details: Some("Sample rate index is 4 bits (0-12). Values 13-15 are reserved.".to_string()),
            related_errors: Vec::new(),
        });
    }

    // Channel configuration (bits 23-25)
    let channel_config = ((frame[2] & 0x01) << 2) | ((frame[3] >> 6) & 0x03);
    if channel_config == 0 || channel_config > 7 {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::InvalidChannelConfig,
            severity: ErrorSeverity::Error,
            byte_offset: 2,
            expected: Some("1-7 (mono to 7.1 surround)".to_string()),
            found: Some(format!("{} (invalid)", channel_config)),
            hex_dump: Some(create_hex_dump(2, 2)),
            suggestion: Some("Invalid channel configuration. For stereo use 2, for mono use 1. Values 0 and 8+ are reserved.".to_string()),
            code_example: Some("AAC channel configs: 1=mono, 2=stereo. Use AudioCodec::Aac(AacProfile::Lc) for 2-channel stereo.".to_string()),
            technical_details: Some("Channel config: 1=mono, 2=stereo, 3=3.0, 4=4.0, 5=5.0, 6=5.1, 7=7.1. 0=implicit, 8+=reserved.".to_string()),
            related_errors: Vec::new(),
        });
    }

    // Frame length validation (13 bits across bytes 3-5)
    let aac_frame_length: usize = (((frame[3] & 0x03) as usize) << 11)
        | ((frame[4] as usize) << 3)
        | (((frame[5] & 0xE0) as usize) >> 5);

    if aac_frame_length < header_len {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::InvalidFrameLength,
            severity: ErrorSeverity::Error,
            byte_offset: 3,
            expected: Some(format!("≥{} (header length)", header_len)),
            found: Some(format!("{} (too small)", aac_frame_length)),
            hex_dump: Some(create_hex_dump(3, 3)),
            suggestion: Some("Frame length is smaller than header. This indicates corrupted frame length field. Check bytes 3-5.".to_string()),
            code_example: None,
            technical_details: Some(format!("Frame length (13 bits): includes header + payload. Must be ≥{} for {} header.", header_len, if protection_absent { "unprotected" } else { "protected" })),
            related_errors: Vec::new(),
        });
    }

    if aac_frame_length > frame.len() {
        return Err(AdtsValidationError {
            kind: AdtsErrorKind::InvalidFrameLength,
            severity: ErrorSeverity::Error,
            byte_offset: 3,
            expected: Some(format!("≤{} (available data)", frame.len())),
            found: Some(format!("{} (too large)", aac_frame_length)),
            hex_dump: Some(create_hex_dump(3, 3)),
            suggestion: Some("Frame length exceeds available data. Frame may be truncated or frame length field corrupted.".to_string()),
            code_example: None,
            technical_details: Some(format!("Frame length {} > buffer size {}. Check if frame is complete.", aac_frame_length, frame.len())),
            related_errors: Vec::new(),
        });
    }

    // CRC validation if present
    if !protection_absent && frame.len() >= header_len + 2 {
        // Note: Full CRC validation would require implementing CRC calculation
        // For now, we just check that CRC bytes exist
        let crc_start = header_len - 2;
        if frame.len() < crc_start + 2 {
            return Err(AdtsValidationError {
                kind: AdtsErrorKind::CrcMismatch,
                severity: ErrorSeverity::Error,
                byte_offset: crc_start,
                expected: Some("2 CRC bytes".to_string()),
                found: Some(format!(
                    "{} bytes available",
                    frame.len().saturating_sub(crc_start)
                )),
                hex_dump: Some(create_hex_dump(
                    crc_start,
                    frame.len().saturating_sub(crc_start),
                )),
                suggestion: Some(
                    "CRC protection is enabled but CRC bytes are missing or truncated.".to_string(),
                ),
                code_example: None,
                technical_details: Some(
                    "CRC is 16 bits stored after header when protection_absent=0.".to_string(),
                ),
                related_errors: Vec::new(),
            });
        }
    }

    Ok(&frame[header_len..aac_frame_length])
}

pub(crate) fn build_moov_box(
    video: &Mp4VideoTrack,
    video_tables: &SampleTables,
    audio: Option<(&Mp4AudioTrack, &SampleTables)>,
    subtitle: Option<(&Mp4SubtitleTrack, &SampleTables)>,
    video_config: &VideoConfig,
    metadata: Option<&Metadata>,
) -> Vec<u8> {
    // Calculate duration in media timescale, then convert to movie timescale (ms)
    let video_duration_media = video_tables.total_duration();
    let audio_duration_media = audio.map(|(_, t)| t.total_duration()).unwrap_or(0);
    let subtitle_duration_media = subtitle.map(|(_, t)| t.total_duration()).unwrap_or(0);
    let overall_duration_media = video_duration_media
        .max(audio_duration_media)
        .max(subtitle_duration_media);
    let overall_duration_ms =
        overall_duration_media * u64::from(MOVIE_TIMESCALE) / u64::from(MEDIA_TIMESCALE);

    let next_track_id = if subtitle.is_some() {
        4
    } else if audio.is_some() {
        3
    } else {
        2
    };
    let mvhd_payload = build_mvhd_payload(overall_duration_ms, next_track_id);
    let mvhd_box = build_box(b"mvhd", &mvhd_payload);
    // Movie origin = earliest first PTS across tracks; do NOT shift samples.
    // Tracks starting later get edts/elst (verdict §7); audio may precede video.
    let mut origins = Vec::new();
    if let Some(f) = video_tables.first_pts {
        origins.push(f);
    }
    if let Some((_, t)) = audio
        && let Some(f) = t.first_pts
    {
        origins.push(f);
    }
    if let Some((_, t)) = subtitle
        && let Some(f) = t.first_pts
    {
        origins.push(f);
    }
    let origin = origins.into_iter().min().unwrap_or(0);
    let trak_box = build_trak_box_with_edts(
        video,
        video_tables,
        video_config,
        metadata,
        origin,
        overall_duration_ms,
    );

    let mut payload = Vec::new();
    payload.extend_from_slice(&mvhd_box);
    payload.extend_from_slice(&trak_box);
    if let Some((audio_track, audio_tables)) = audio {
        let audio_trak = build_audio_trak_box_with_edts(
            audio_track,
            audio_tables,
            metadata,
            origin,
            overall_duration_ms,
        );
        payload.extend_from_slice(&audio_trak);
    }
    if let Some((subtitle_track, subtitle_tables)) = subtitle {
        let subtitle_trak = build_subtitle_trak_box_with_edts(
            subtitle_track,
            subtitle_tables,
            metadata,
            origin,
            overall_duration_ms,
        );
        payload.extend_from_slice(&subtitle_trak);
    }

    // Add metadata (udta box) if present
    if let Some(meta) = metadata {
        let udta_box = build_udta_box(meta);
        if !udta_box.is_empty() {
            payload.extend_from_slice(&udta_box);
        }
    }

    build_box(b"moov", &payload)
}

pub(crate) fn build_audio_only_moov_box(
    audio: &Mp4AudioTrack,
    audio_tables: &SampleTables,
    metadata: Option<&Metadata>,
) -> Vec<u8> {
    let audio_duration_media = audio_tables.total_duration();
    let audio_duration_ms =
        audio_duration_media * u64::from(MOVIE_TIMESCALE) / u64::from(MEDIA_TIMESCALE);

    let mvhd_payload = build_mvhd_payload(audio_duration_ms, 3);
    let mvhd_box = build_box(b"mvhd", &mvhd_payload);
    let audio_trak = build_audio_trak_box(audio, audio_tables, metadata);

    let mut payload = Vec::new();
    payload.extend_from_slice(&mvhd_box);
    payload.extend_from_slice(&audio_trak);

    if let Some(meta) = metadata {
        let udta_box = build_udta_box(meta);
        if !udta_box.is_empty() {
            payload.extend_from_slice(&udta_box);
        }
    }

    build_box(b"moov", &payload)
}

fn build_audio_trak_box(
    audio: &Mp4AudioTrack,
    tables: &SampleTables,
    metadata: Option<&Metadata>,
) -> Vec<u8> {
    build_audio_trak_box_with_edts(audio, tables, metadata, 0, 0)
}

fn build_audio_trak_box_with_edts(
    audio: &Mp4AudioTrack,
    tables: &SampleTables,
    metadata: Option<&Metadata>,
    origin: u64,
    movie_duration: u64,
) -> Vec<u8> {
    let tkhd_box = build_audio_tkhd_box();
    let mdia_box = build_audio_mdia_box(audio, tables, metadata);

    let mut payload = Vec::new();
    payload.extend_from_slice(&tkhd_box);
    let offset = tables.start_offset(origin);
    if offset > 0 {
        payload.extend_from_slice(&build_edts_box(offset, movie_duration));
    }
    payload.extend_from_slice(&mdia_box);
    build_box(b"trak", &payload)
}

#[allow(dead_code)]
fn build_subtitle_trak_box(
    subtitle: &Mp4SubtitleTrack,
    tables: &SampleTables,
    metadata: Option<&Metadata>,
) -> Vec<u8> {
    build_subtitle_trak_box_with_edts(subtitle, tables, metadata, 0, 0)
}

fn build_subtitle_trak_box_with_edts(
    subtitle: &Mp4SubtitleTrack,
    tables: &SampleTables,
    metadata: Option<&Metadata>,
    origin: u64,
    movie_duration: u64,
) -> Vec<u8> {
    let tkhd_box = build_subtitle_tkhd_box();
    let mdia_box = build_subtitle_mdia_box(subtitle, tables, metadata);

    let mut payload = Vec::new();
    payload.extend_from_slice(&tkhd_box);
    let offset = tables.start_offset(origin);
    if offset > 0 {
        payload.extend_from_slice(&build_edts_box(offset, movie_duration));
    }
    payload.extend_from_slice(&mdia_box);
    build_box(b"trak", &payload)
}

fn build_audio_tkhd_box() -> Vec<u8> {
    build_tkhd_box_with_id(2, 0x0100, 0, 0)
}

fn build_subtitle_tkhd_box() -> Vec<u8> {
    build_tkhd_box_with_id(3, 0, 0, 0)
}

fn build_audio_mdia_box(
    audio: &Mp4AudioTrack,
    tables: &SampleTables,
    metadata: Option<&Metadata>,
) -> Vec<u8> {
    let duration = tables.total_duration();
    // Verdict §10: per-track language wins; metadata is fallback.
    let language = audio
        .language
        .as_deref()
        .or_else(|| metadata.and_then(|m| m.language.as_deref()));
    let mdhd_box = build_mdhd_box_with_timescale_and_duration(MEDIA_TIMESCALE, duration, language);
    let hdlr_box = build_sound_hdlr_box();
    let minf_box = build_audio_minf_box(audio, tables);

    let mut payload = Vec::new();
    payload.extend_from_slice(&mdhd_box);
    payload.extend_from_slice(&hdlr_box);
    payload.extend_from_slice(&minf_box);
    build_box(b"mdia", &payload)
}

fn build_subtitle_mdia_box(
    subtitle: &Mp4SubtitleTrack,
    tables: &SampleTables,
    metadata: Option<&Metadata>,
) -> Vec<u8> {
    let duration = tables.total_duration();
    let language = subtitle
        .language
        .as_deref()
        .or_else(|| metadata.and_then(|m| m.language.as_deref()));
    let mdhd_box = build_mdhd_box_with_timescale_and_duration(MEDIA_TIMESCALE, duration, language);
    let hdlr_box = build_subtitle_hdlr_box();
    let minf_box = build_subtitle_minf_box(subtitle, tables);

    let mut payload = Vec::new();
    payload.extend_from_slice(&mdhd_box);
    payload.extend_from_slice(&hdlr_box);
    payload.extend_from_slice(&minf_box);
    build_box(b"mdia", &payload)
}

fn build_audio_minf_box(audio: &Mp4AudioTrack, tables: &SampleTables) -> Vec<u8> {
    let smhd_box = build_smhd_box();
    let dinf_box = build_dinf_box();
    let stbl_box = build_audio_stbl_box(audio, tables);

    let mut payload = Vec::new();
    payload.extend_from_slice(&smhd_box);
    payload.extend_from_slice(&dinf_box);
    payload.extend_from_slice(&stbl_box);
    build_box(b"minf", &payload)
}

fn build_subtitle_minf_box(subtitle: &Mp4SubtitleTrack, tables: &SampleTables) -> Vec<u8> {
    let nmhd_box = build_nmhd_box();
    let dinf_box = build_dinf_box();
    let stbl_box = build_subtitle_stbl_box(subtitle, tables);

    let mut payload = Vec::new();
    payload.extend_from_slice(&nmhd_box);
    payload.extend_from_slice(&dinf_box);
    payload.extend_from_slice(&stbl_box);
    build_box(b"minf", &payload)
}

fn build_audio_stbl_box(audio: &Mp4AudioTrack, tables: &SampleTables) -> Vec<u8> {
    let stsd_box = build_audio_stsd_box(audio);
    let stts_box = build_stts_box(&tables.durations);
    let stsc_box = build_stsc_box(tables.samples_per_chunk, tables.chunk_offsets.len() as u32);
    let stsz_box = build_stsz_box(&tables.sizes);
    let stco_box = build_chunk_offset_box(&tables.chunk_offsets);

    let mut payload = Vec::new();
    payload.extend_from_slice(&stsd_box);
    payload.extend_from_slice(&stts_box);
    payload.extend_from_slice(&stsc_box);
    payload.extend_from_slice(&stsz_box);
    payload.extend_from_slice(&stco_box);
    build_box(b"stbl", &payload)
}

fn build_subtitle_stbl_box(subtitle: &Mp4SubtitleTrack, tables: &SampleTables) -> Vec<u8> {
    let stsd_box = build_subtitle_stsd_box(subtitle);
    let stts_box = build_stts_box(&tables.durations);
    let stsc_box = build_stsc_box(tables.samples_per_chunk, tables.chunk_offsets.len() as u32);
    let stsz_box = build_stsz_box(&tables.sizes);
    let stco_box = build_chunk_offset_box(&tables.chunk_offsets);

    let mut payload = Vec::new();
    payload.extend_from_slice(&stsd_box);
    payload.extend_from_slice(&stts_box);
    payload.extend_from_slice(&stsc_box);
    payload.extend_from_slice(&stsz_box);
    payload.extend_from_slice(&stco_box);
    build_box(b"stbl", &payload)
}

fn build_audio_stsd_box(audio: &Mp4AudioTrack) -> Vec<u8> {
    let sample_entry_box = match audio.codec {
        AudioCodec::Aac(_) => build_mp4a_box(audio),
        AudioCodec::Opus => build_opus_box(audio),
        AudioCodec::Flac => build_flac_box(audio),
        AudioCodec::None => build_mp4a_box(audio), // Fallback, shouldn't happen
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&sample_entry_box);
    build_box(b"stsd", &payload)
}

fn build_subtitle_stsd_box(subtitle: &Mp4SubtitleTrack) -> Vec<u8> {
    let sample_entry_box = match subtitle.codec {
        SubtitleCodec::MovText => build_tx3g_box(),
        SubtitleCodec::WebVtt => build_wvtt_box(),
        SubtitleCodec::Ssa | SubtitleCodec::Ass => {
            unreachable!("SSA/ASS subtitles are Matroska-only (rejected in MuxerBuilder::build)")
        }
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&sample_entry_box);
    build_box(b"stsd", &payload)
}

fn build_tx3g_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0u8; 6]);
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.push(0);
    payload.push(0);
    payload.extend_from_slice(&[0u8; 4]);
    payload.extend_from_slice(&0i16.to_be_bytes());
    payload.extend_from_slice(&0i16.to_be_bytes());
    payload.extend_from_slice(&0i16.to_be_bytes());
    payload.extend_from_slice(&0i16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.push(0);
    payload.push(18);
    payload.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    payload.extend_from_slice(&build_tx3g_ftab_box());
    build_box(b"tx3g", &payload)
}

fn build_tx3g_ftab_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.push(5);
    payload.extend_from_slice(b"Serif");
    build_box(b"ftab", &payload)
}

fn build_wvtt_box() -> Vec<u8> {
    // `wvtt` sample entry with empty `vttC` config (no global headers).
    let vttc = build_box(b"vttC", &[]);
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0u8; 6]);
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.extend_from_slice(&vttc);
    build_box(b"wvtt", &payload)
}

fn build_mp4a_box(audio: &Mp4AudioTrack) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&[0u8; 6]);
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&audio.channels.to_be_bytes());
    payload.extend_from_slice(&16u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    let rate_fixed = audio.sample_rate << 16;
    payload.extend_from_slice(&rate_fixed.to_be_bytes());
    let esds = build_esds_box(audio);
    payload.extend_from_slice(&esds);
    build_box(b"mp4a", &payload)
}

fn build_esds_box(audio: &Mp4AudioTrack) -> Vec<u8> {
    // Verdict §8: caller ASC wins (required for HE/HEv2 extensions).
    let asc: Vec<u8> = match &audio.aac_asc_override {
        Some(raw) => raw.clone(),
        None => build_audio_specific_config(audio.sample_rate, audio.channels).to_vec(),
    };

    let mut dec_specific = Vec::new();
    dec_specific.push(0x05);
    dec_specific.push(asc.len() as u8);
    dec_specific.extend_from_slice(&asc);

    let mut dec_config_payload = Vec::new();
    dec_config_payload.push(0x40);
    dec_config_payload.push(0x15);
    dec_config_payload.extend_from_slice(&[0x00, 0x00, 0x00]);
    dec_config_payload.extend_from_slice(&0u32.to_be_bytes());
    dec_config_payload.extend_from_slice(&0u32.to_be_bytes());
    dec_config_payload.extend_from_slice(&dec_specific);

    let mut dec_config = Vec::new();
    dec_config.push(0x04);
    dec_config.push(dec_config_payload.len() as u8);
    dec_config.extend_from_slice(&dec_config_payload);

    let sl_config = [0x06u8, 0x01u8, 0x02u8];

    let mut es_payload = Vec::new();
    es_payload.extend_from_slice(&1u16.to_be_bytes());
    es_payload.push(0);
    es_payload.extend_from_slice(&dec_config);
    es_payload.extend_from_slice(&sl_config);

    let mut es_desc = Vec::new();
    es_desc.push(0x03);
    es_desc.push(es_payload.len() as u8);
    es_desc.extend_from_slice(&es_payload);

    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&es_desc);
    build_box(b"esds", &payload)
}

fn build_audio_specific_config(sample_rate: u32, channels: u16) -> [u8; 2] {
    let sfi = match sample_rate {
        96000 => 0,
        88200 => 1,
        64000 => 2,
        48000 => 3,
        44100 => 4,
        32000 => 5,
        24000 => 6,
        22050 => 7,
        16000 => 8,
        12000 => 9,
        11025 => 10,
        8000 => 11,
        7350 => 12,
        _ => 4,
    };
    let aot = 2u8;
    let chan = (channels.min(15) as u8) & 0x0f;
    let byte0 = (aot << 3) | (sfi >> 1);
    let byte1 = ((sfi & 1) << 7) | (chan << 3);
    [byte0, byte1]
}

/// Build an Opus sample entry box.
fn build_opus_box(audio: &Mp4AudioTrack) -> Vec<u8> {
    let mut payload = Vec::new();
    // Reserved (6 bytes)
    payload.extend_from_slice(&[0u8; 6]);
    // Data reference index
    payload.extend_from_slice(&1u16.to_be_bytes());
    // Reserved (2 x u32)
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Channel count
    payload.extend_from_slice(&audio.channels.to_be_bytes());
    // Sample size (16 bits)
    payload.extend_from_slice(&16u16.to_be_bytes());
    // Pre-defined
    payload.extend_from_slice(&0u16.to_be_bytes());
    // Reserved
    payload.extend_from_slice(&0u16.to_be_bytes());
    // Sample rate (fixed point 16.16, always 48000 for Opus)
    let rate_fixed = OPUS_SAMPLE_RATE << 16;
    payload.extend_from_slice(&rate_fixed.to_be_bytes());

    // dOps box (Opus decoder configuration)
    let dops = build_dops_box(audio);
    payload.extend_from_slice(&dops);

    build_box(b"Opus", &payload)
}

/// Build the dOps (Opus Decoder Configuration) box.
///
/// Structure per ISO/IEC 14496-3 Amendment 4:
/// - Version (1 byte) = 0
/// - OutputChannelCount (1 byte)
/// - PreSkip (2 bytes, big-endian)
/// - InputSampleRate (4 bytes, big-endian)
/// - OutputGain (2 bytes, signed, big-endian)
/// - ChannelMappingFamily (1 byte)
/// - If ChannelMappingFamily != 0:
///   - StreamCount (1 byte)
///   - CoupledCount (1 byte)
///   - ChannelMapping (OutputChannelCount bytes)
fn build_dops_box(audio: &Mp4AudioTrack) -> Vec<u8> {
    // Verdict §8: full caller config wins; otherwise channel/preskip overrides.
    let config = match &audio.opus_config_override {
        Some(c) => c.clone(),
        None => OpusConfig::default()
            .with_channels(audio.channels as u8)
            .with_pre_skip(audio.opus_preskip.unwrap_or(312)),
    };

    let mut payload = Vec::new();
    // Version = 0
    payload.push(config.version);
    // OutputChannelCount
    payload.push(config.output_channel_count);
    // PreSkip (big-endian)
    payload.extend_from_slice(&config.pre_skip.to_be_bytes());
    // InputSampleRate (big-endian)
    payload.extend_from_slice(&config.input_sample_rate.to_be_bytes());
    // OutputGain (signed, big-endian)
    payload.extend_from_slice(&config.output_gain.to_be_bytes());
    // ChannelMappingFamily
    payload.push(config.channel_mapping_family);

    // Extended channel mapping for family != 0
    if config.channel_mapping_family != 0 {
        payload.push(config.stream_count.unwrap_or(1));
        payload.push(config.coupled_count.unwrap_or(0));
        if let Some(mapping) = &config.channel_mapping {
            payload.extend_from_slice(mapping);
        } else {
            // Default mapping for stereo
            for i in 0..config.output_channel_count {
                payload.push(i);
            }
        }
    }

    build_box(b"dOps", &payload)
}

/// Build a FLAC sample entry box (`fLaC`) with its `dfLa` config.
///
/// Follows xiph "Encapsulation of FLAC in ISO Base Media File Format":
/// one MP4 sample is exactly one FLAC frame, and `dfLa` (FullBox
/// version 0) carries the native STREAMINFO metadata block. Sample
/// rates above 16.16-fixed range (> 65535 Hz) are written as 0 in the
/// entry; players must read the true rate from `dfLa` STREAMINFO.
fn build_flac_box(audio: &Mp4AudioTrack) -> Vec<u8> {
    // INV-604: FLAC sample entries require a valid 34-byte STREAMINFO.
    // `MuxerBuilder` validates this up front; direct `Mp4Writer` users
    // get a loud panic here rather than a silently corrupt file.
    let streaminfo = audio.flac_streaminfo.as_deref().unwrap_or(&[]);
    assert_invariant!(
        parse_streaminfo(streaminfo).is_some(),
        "INV-604: FLAC sample entry requires valid STREAMINFO",
        "build_flac_box"
    );
    let info = parse_streaminfo(streaminfo).expect("builder-validated STREAMINFO");

    let mut payload = Vec::new();
    // Reserved (6 bytes)
    payload.extend_from_slice(&[0u8; 6]);
    // Data reference index
    payload.extend_from_slice(&1u16.to_be_bytes());
    // Reserved (u32 + u32)
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Channel count (from STREAMINFO, authoritative)
    payload.extend_from_slice(&u16::from(info.channels).to_be_bytes());
    // Sample size in bits (from STREAMINFO)
    payload.extend_from_slice(&u16::from(info.bits_per_sample).to_be_bytes());
    // Pre-defined
    payload.extend_from_slice(&0u16.to_be_bytes());
    // Reserved
    payload.extend_from_slice(&0u16.to_be_bytes());
    // Sample rate (16.16 fixed point; 0 when > 65535 Hz, see dfLa)
    let rate_fixed = if info.sample_rate > u32::from(u16::MAX) {
        0u32
    } else {
        info.sample_rate << 16
    };
    payload.extend_from_slice(&rate_fixed.to_be_bytes());

    // dfLa box with the native STREAMINFO block.
    payload.extend_from_slice(&build_dfla_box(&info));

    build_box(b"fLaC", &payload)
}

/// Build the `dfLa` (FLACSpecificBox) FullBox v0 carrying STREAMINFO.
fn build_dfla_box(info: &crate::codec::flac::FlacStreaminfo) -> Vec<u8> {
    let mut payload = Vec::new();
    // FullBox: version = 0, flags = 0.
    payload.extend_from_slice(&[0, 0, 0, 0]);
    // One FLACMetadataBlock: last=1, type=0 (STREAMINFO), length=34.
    payload.push(0x80);
    payload.extend_from_slice(&(STREAMINFO_LEN as u32).to_be_bytes()[1..4]);
    let mut raw = [0u8; STREAMINFO_LEN];
    raw[0..2].copy_from_slice(&info.min_block_size.to_be_bytes());
    raw[2..4].copy_from_slice(&info.max_block_size.to_be_bytes());
    raw[4..7].copy_from_slice(&[
        (info.min_frame_size >> 16) as u8,
        (info.min_frame_size >> 8) as u8,
        info.min_frame_size as u8,
    ]);
    raw[7..10].copy_from_slice(&[
        (info.max_frame_size >> 16) as u8,
        (info.max_frame_size >> 8) as u8,
        info.max_frame_size as u8,
    ]);
    raw[10] = ((info.sample_rate >> 12) & 0xFF) as u8;
    raw[11] = ((info.sample_rate >> 4) & 0xFF) as u8;
    raw[12] = ((((info.sample_rate & 0x0F) << 4)
        | (u32::from(info.channels - 1) << 1)
        | (u32::from(info.bits_per_sample - 1) >> 4))
        & 0xFF) as u8;
    raw[13] = ((((u32::from(info.bits_per_sample - 1) & 0x0F) << 4)
        | ((info.total_samples >> 32) as u32 & 0x0F))
        & 0xFF) as u8;
    raw[14..18].copy_from_slice(&(info.total_samples as u32).to_be_bytes());
    raw[18..34].copy_from_slice(&info.md5);
    payload.extend_from_slice(&raw);
    build_box(b"dfLa", &payload)
}

#[allow(dead_code)]
fn build_trak_box(
    video: &Mp4VideoTrack,
    tables: &SampleTables,
    video_config: &VideoConfig,
    metadata: Option<&Metadata>,
) -> Vec<u8> {
    build_trak_box_with_edts(video, tables, video_config, metadata, 0, 0)
}

fn build_trak_box_with_edts(
    video: &Mp4VideoTrack,
    tables: &SampleTables,
    video_config: &VideoConfig,
    metadata: Option<&Metadata>,
    origin: u64,
    movie_duration: u64,
) -> Vec<u8> {
    let tkhd_box = build_tkhd_box(video);
    let mdia_box = build_mdia_box(video, tables, video_config, metadata);

    let mut payload = Vec::new();
    payload.extend_from_slice(&tkhd_box);
    let offset = tables.start_offset(origin);
    if offset > 0 {
        payload.extend_from_slice(&build_edts_box(offset, movie_duration));
    }
    payload.extend_from_slice(&mdia_box);
    build_box(b"trak", &payload)
}

fn build_mdia_box(
    video: &Mp4VideoTrack,
    tables: &SampleTables,
    video_config: &VideoConfig,
    metadata: Option<&Metadata>,
) -> Vec<u8> {
    let duration = tables.total_duration();
    // Verdict §10: per-track language wins; metadata is fallback.
    let language = video
        .language
        .as_deref()
        .or_else(|| metadata.and_then(|m| m.language.as_deref()));
    let mdhd_box = build_mdhd_box_with_timescale_and_duration(MEDIA_TIMESCALE, duration, language);
    let hdlr_box = build_hdlr_box();
    let minf_box = build_minf_box(video, tables, video_config);

    let mut payload = Vec::new();
    payload.extend_from_slice(&mdhd_box);
    payload.extend_from_slice(&hdlr_box);
    payload.extend_from_slice(&minf_box);
    build_box(b"mdia", &payload)
}

fn build_minf_box(
    video: &Mp4VideoTrack,
    tables: &SampleTables,
    video_config: &VideoConfig,
) -> Vec<u8> {
    let vmhd_box = build_vmhd_box();
    let dinf_box = build_dinf_box();
    let stbl_box = build_stbl_box(video, tables, video_config);

    let mut payload = Vec::new();
    payload.extend_from_slice(&vmhd_box);
    payload.extend_from_slice(&dinf_box);
    payload.extend_from_slice(&stbl_box);
    build_box(b"minf", &payload)
}

fn build_stbl_box(
    video: &Mp4VideoTrack,
    tables: &SampleTables,
    video_config: &VideoConfig,
) -> Vec<u8> {
    let stsd_box = build_stsd_box(video, video_config);
    let stts_box = build_stts_box(&tables.durations);
    let stsc_box = build_stsc_box(tables.samples_per_chunk, tables.chunk_offsets.len() as u32);
    let stsz_box = build_stsz_box(&tables.sizes);
    let stco_box = build_chunk_offset_box(&tables.chunk_offsets);

    let mut payload = Vec::new();
    payload.extend_from_slice(&stsd_box);
    payload.extend_from_slice(&stts_box);
    // Add ctts box if B-frames are present (pts != dts for any sample)
    if tables.has_bframes {
        let ctts_box = build_ctts_box(&tables.cts_offsets);
        payload.extend_from_slice(&ctts_box);
    }
    payload.extend_from_slice(&stsc_box);
    payload.extend_from_slice(&stsz_box);
    payload.extend_from_slice(&stco_box);
    if !tables.keyframes.is_empty() {
        let stss_box = build_stss_box(&tables.keyframes);
        payload.extend_from_slice(&stss_box);
    }
    build_box(b"stbl", &payload)
}

fn build_stsd_box(video: &Mp4VideoTrack, video_config: &VideoConfig) -> Vec<u8> {
    let sample_entry = match video_config {
        VideoConfig::Avc(avc_config) => build_avc1_box(video, avc_config),
        VideoConfig::Hevc(hevc_config) => build_hvc1_box(video, hevc_config),
        VideoConfig::Av1(av1_config) => build_av01_box(video, av1_config),
        VideoConfig::Vp9(vp9_config) => build_vp09_box(video, vp9_config),
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&sample_entry);
    build_box(b"stsd", &payload)
}

fn build_stts_box(durations: &[u32]) -> Vec<u8> {
    let mut entries: Vec<(u32, u32)> = Vec::new();
    for &duration in durations {
        if let Some(last) = entries.last_mut()
            && last.1 == duration
        {
            last.0 += 1;
            continue;
        }
        entries.push((1u32, duration));
    }

    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (count, delta) in entries {
        payload.extend_from_slice(&count.to_be_bytes());
        payload.extend_from_slice(&delta.to_be_bytes());
    }
    build_box(b"stts", &payload)
}

fn build_stsc_box(samples_per_chunk: u32, chunk_count: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());

    if chunk_count == 0 || samples_per_chunk == 0 {
        payload.extend_from_slice(&0u32.to_be_bytes());
        return build_box(b"stsc", &payload);
    }

    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&samples_per_chunk.to_be_bytes());
    payload.extend_from_slice(&1u32.to_be_bytes());
    build_box(b"stsc", &payload)
}

fn build_stsz_box(sizes: &[u32]) -> Vec<u8> {
    // INV-004: No empty samples (zero-size) in stsz
    for (i, &size) in sizes.iter().enumerate() {
        assert_invariant!(
            size > 0,
            "No empty samples in stsz",
            &format!("build_stsz_box[{}]", i)
        );
    }

    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
    for size in sizes {
        payload.extend_from_slice(&size.to_be_bytes());
    }
    build_box(b"stsz", &payload)
}

fn build_stco_box(chunk_offsets: &[u32]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());

    payload.extend_from_slice(&(chunk_offsets.len() as u32).to_be_bytes());
    for offset in chunk_offsets {
        payload.extend_from_slice(&offset.to_be_bytes());
    }
    build_box(b"stco", &payload)
}

fn build_co64_box(chunk_offsets: &[u64]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    let count = u32::try_from(chunk_offsets.len()).expect("too many chunks for co64");
    payload.extend_from_slice(&count.to_be_bytes());
    for offset in chunk_offsets {
        payload.extend_from_slice(&offset.to_be_bytes());
    }
    build_box(b"co64", &payload)
}

/// Automatically choose `stco` or `co64` (verdict §6).
fn build_chunk_offset_box(offsets: &[u64]) -> Vec<u8> {
    if offsets.iter().all(|&o| u32::try_from(o).is_ok()) {
        let narrow: Vec<u32> = offsets.iter().map(|&o| o as u32).collect();
        build_stco_box(&narrow)
    } else {
        build_co64_box(offsets)
    }
}

fn build_stss_box(keyframes: &[u32]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&(keyframes.len() as u32).to_be_bytes());
    for index in keyframes {
        payload.extend_from_slice(&index.to_be_bytes());
    }
    build_box(b"stss", &payload)
}

/// Build ctts (Composition Time to Sample) box for B-frame support.
/// Uses version 1 which supports signed offsets (required for some B-frame patterns).
fn build_ctts_box(cts_offsets: &[i32]) -> Vec<u8> {
    // Run-length encode the offsets
    let mut entries: Vec<(u32, i32)> = Vec::new();
    for &offset in cts_offsets {
        if let Some(last) = entries.last_mut()
            && last.1 == offset
        {
            last.0 += 1;
            continue;
        }
        entries.push((1, offset));
    }

    let mut payload = Vec::new();
    // Version 1 (supports signed offsets), flags 0
    payload.extend_from_slice(&0x0100_0000_u32.to_be_bytes());
    payload.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (count, offset) in entries {
        payload.extend_from_slice(&count.to_be_bytes());
        payload.extend_from_slice(&offset.to_be_bytes());
    }
    build_box(b"ctts", &payload)
}

fn build_avc1_box(video: &Mp4VideoTrack, avc_config: &AvcConfig) -> Vec<u8> {
    // INV-002: Width/height must fit in 16-bit for visual sample entry
    assert_invariant!(
        video.width <= u16::MAX as u32,
        "Width must fit in 16-bit",
        "build_avc1_box"
    );
    assert_invariant!(
        video.height <= u16::MAX as u32,
        "Height must fit in 16-bit",
        "build_avc1_box"
    );

    let mut payload = Vec::new();
    payload.extend_from_slice(&[0u8; 6]);
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Width and height are 16-bit values in the visual sample entry
    payload.extend_from_slice(&(video.width as u16).to_be_bytes());
    payload.extend_from_slice(&(video.height as u16).to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&1u16.to_be_bytes());
    payload.extend_from_slice(&[0u8; 32]);
    payload.extend_from_slice(&0x0018u16.to_be_bytes());
    payload.extend_from_slice(&0xffffu16.to_be_bytes());
    let avc_c_box = build_avcc_box(avc_config);
    payload.extend_from_slice(&avc_c_box);
    build_box(b"avc1", &payload)
}

fn build_avcc_box(avc_config: &AvcConfig) -> Vec<u8> {
    build_box(b"avcC", &avcc_payload(avc_config))
}

/// Raw avcC payload (without the MP4 box header).
///
/// Shared with the Matroska muxer: `V_MPEG4/ISO/AVC` stores the same
/// avcC bytes as `CodecPrivate`.
pub(crate) fn avcc_payload(avc_config: &AvcConfig) -> Vec<u8> {
    let mut payload = Vec::new();

    let (profile_indication, profile_compat, level_indication) = if avc_config.sps.len() >= 4 {
        (avc_config.sps[1], avc_config.sps[2], avc_config.sps[3])
    } else {
        (0x42, 0x00, 0x1e)
    };

    payload.push(1);
    payload.push(profile_indication);
    payload.push(profile_compat);
    payload.push(level_indication);
    // lengthSizeMinusOne = 3 (4-byte NAL length), stored as 0xFF.
    payload.push(0xff);
    // numOfSequenceParameterSets (verdict §8: full arrays).
    let num_sps =
        u8::try_from(avc_config.sequence_parameter_sets.len().min(31)).unwrap_or(31) | 0xe0;
    payload.push(num_sps);
    for sps in &avc_config.sequence_parameter_sets {
        payload.extend_from_slice(&(u16::try_from(sps.len()).unwrap_or(u16::MAX)).to_be_bytes());
        payload.extend_from_slice(sps);
    }
    let num_pps = u8::try_from(avc_config.picture_parameter_sets.len().min(255)).unwrap_or(255);
    payload.push(num_pps);
    for pps in &avc_config.picture_parameter_sets {
        payload.extend_from_slice(&(u16::try_from(pps.len()).unwrap_or(u16::MAX)).to_be_bytes());
        payload.extend_from_slice(pps);
    }
    payload
}

/// Build an hvc1 sample entry box for HEVC video.
fn build_hvc1_box(video: &Mp4VideoTrack, hevc_config: &HevcConfig) -> Vec<u8> {
    // INV-002: Width/height must fit in 16-bit for visual sample entry
    assert_invariant!(
        video.width <= u16::MAX as u32,
        "Width must fit in 16-bit",
        "build_hvc1_box"
    );
    assert_invariant!(
        video.height <= u16::MAX as u32,
        "Height must fit in 16-bit",
        "build_hvc1_box"
    );

    let mut payload = Vec::new();
    // Reserved (6 bytes)
    payload.extend_from_slice(&[0u8; 6]);
    // Data reference index
    payload.extend_from_slice(&1u16.to_be_bytes());
    // Pre-defined + reserved
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Width and height are 16-bit values in the visual sample entry
    payload.extend_from_slice(&(video.width as u16).to_be_bytes());
    payload.extend_from_slice(&(video.height as u16).to_be_bytes());
    // Horizontal/vertical resolution (72 dpi fixed point)
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    // Reserved
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Frame count
    payload.extend_from_slice(&1u16.to_be_bytes());
    // Compressor name (32 bytes, empty)
    payload.extend_from_slice(&[0u8; 32]);
    // Depth
    payload.extend_from_slice(&0x0018u16.to_be_bytes());
    // Pre-defined
    payload.extend_from_slice(&0xffffu16.to_be_bytes());
    // hvcC box
    let hvcc_box = build_hvcc_box(hevc_config);
    payload.extend_from_slice(&hvcc_box);
    build_box(b"hvc1", &payload)
}

/// Build an hvcC configuration box for HEVC.
fn build_hvcc_box(hevc_config: &HevcConfig) -> Vec<u8> {
    build_box(b"hvcC", &hvcc_payload(hevc_config))
}

/// Raw hvcC payload (without the MP4 box header).
///
/// Shared with the Matroska muxer: `V_MPEGH/ISO/HEVC` stores the same
/// hvcC bytes as `CodecPrivate`.
pub(crate) fn hvcc_payload(hevc_config: &HevcConfig) -> Vec<u8> {
    let mut payload = Vec::new();

    // Extract profile/tier/level from SPS
    let general_profile_space = hevc_config.general_profile_space();
    let general_tier_flag = hevc_config.general_tier_flag();
    let general_profile_idc = hevc_config.general_profile_idc();
    let general_level_idc = hevc_config.general_level_idc();

    // configurationVersion = 1
    payload.push(1);

    // general_profile_space (2) + general_tier_flag (1) + general_profile_idc (5)
    let byte1 = (general_profile_space << 6)
        | (if general_tier_flag { 0x20 } else { 0 })
        | (general_profile_idc & 0x1f);
    payload.push(byte1);

    // general_profile_compatibility_flags (4 bytes)
    // For simplicity, set Main profile compatibility (bit 1)
    payload.extend_from_slice(&[0x60, 0x00, 0x00, 0x00]);

    // general_constraint_indicator_flags (6 bytes)
    payload.extend_from_slice(&[0x90, 0x00, 0x00, 0x00, 0x00, 0x00]);

    // general_level_idc
    payload.push(general_level_idc);

    // min_spatial_segmentation_idc (12 bits) with reserved (4 bits)
    payload.extend_from_slice(&[0xf0, 0x00]);

    // parallelismType (2 bits) with reserved (6 bits)
    payload.push(0xfc);

    // chromaFormat (2 bits) with reserved (6 bits) - assume 4:2:0
    payload.push(0xfd);

    // bitDepthLumaMinus8 (3 bits) with reserved (5 bits) - assume 8-bit
    payload.push(0xf8);

    // bitDepthChromaMinus8 (3 bits) with reserved (5 bits) - assume 8-bit
    payload.push(0xf8);

    // avgFrameRate (16 bits) - 0 = unspecified
    payload.extend_from_slice(&0u16.to_be_bytes());

    // constantFrameRate (2) + numTemporalLayers (3) + temporalIdNested (1) + lengthSizeMinusOne (2)
    // lengthSizeMinusOne = 3 (4-byte NAL length)
    payload.push(0x03);

    // numOfArrays = 3 (VPS, SPS, PPS)
    payload.push(3);

    // VPS array
    // array_completeness is the MSB (bit 7). nal_unit_type occupies bits 0..=5.
    payload.push(0x80 | 32); // array_completeness=1 + nal_unit_type=32 (VPS)
    // VPS array (verdict §8: full arrays)
    // array_completeness is the MSB (bit 7). nal_unit_type occupies bits 0..=5.
    payload.push(0x80 | 32); // array_completeness=1 + nal_unit_type=32 (VPS)
    payload.extend_from_slice(
        &(u16::try_from(hevc_config.video_parameter_sets.len()).unwrap_or(u16::MAX)).to_be_bytes(),
    );
    for vps in &hevc_config.video_parameter_sets {
        payload.extend_from_slice(&(u16::try_from(vps.len()).unwrap_or(u16::MAX)).to_be_bytes());
        payload.extend_from_slice(vps);
    }

    // SPS array
    payload.push(0x80 | 33); // array_completeness=1 + nal_unit_type=33 (SPS)
    payload.extend_from_slice(
        &(u16::try_from(hevc_config.sequence_parameter_sets.len()).unwrap_or(u16::MAX))
            .to_be_bytes(),
    );
    for sps in &hevc_config.sequence_parameter_sets {
        payload.extend_from_slice(&(u16::try_from(sps.len()).unwrap_or(u16::MAX)).to_be_bytes());
        payload.extend_from_slice(sps);
    }

    // PPS array
    payload.push(0x80 | 34); // array_completeness=1 + nal_unit_type=34 (PPS)
    payload.extend_from_slice(
        &(u16::try_from(hevc_config.picture_parameter_sets.len()).unwrap_or(u16::MAX))
            .to_be_bytes(),
    );
    for pps in &hevc_config.picture_parameter_sets {
        payload.extend_from_slice(&(u16::try_from(pps.len()).unwrap_or(u16::MAX)).to_be_bytes());
        payload.extend_from_slice(pps);
    }

    payload
}

/// Build an av01 sample entry box for AV1 video.
fn build_av01_box(video: &Mp4VideoTrack, av1_config: &Av1Config) -> Vec<u8> {
    // INV-002: Width/height must fit in 16-bit for visual sample entry
    assert_invariant!(
        video.width <= u16::MAX as u32,
        "Width must fit in 16-bit",
        "build_av01_box"
    );
    assert_invariant!(
        video.height <= u16::MAX as u32,
        "Height must fit in 16-bit",
        "build_av01_box"
    );

    let mut payload = Vec::new();
    // Reserved (6 bytes)
    payload.extend_from_slice(&[0u8; 6]);
    // Data reference index
    payload.extend_from_slice(&1u16.to_be_bytes());
    // Pre-defined + reserved
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Width and height are 16-bit values in the visual sample entry
    payload.extend_from_slice(&(video.width as u16).to_be_bytes());
    payload.extend_from_slice(&(video.height as u16).to_be_bytes());
    // Horizontal/vertical resolution (72 dpi fixed point)
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    // Reserved
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Frame count
    payload.extend_from_slice(&1u16.to_be_bytes());
    // Compressor name (32 bytes, empty)
    payload.extend_from_slice(&[0u8; 32]);
    // Depth (24-bit)
    payload.extend_from_slice(&0x0018u16.to_be_bytes());
    // Pre-defined (-1)
    payload.extend_from_slice(&0xffffu16.to_be_bytes());
    // av1C box
    let av1c_box = build_av1c_box(av1_config);
    payload.extend_from_slice(&av1c_box);
    build_box(b"av01", &payload)
}

/// Build an av1C configuration box for AV1.
///
/// ISO/IEC 14496-12:2022 and AV1 Codec ISO Media File Format Binding spec.
fn build_av1c_box(av1_config: &Av1Config) -> Vec<u8> {
    build_box(b"av1C", &av1c_payload(av1_config))
}

/// Raw av1C payload (without the MP4 box header).
///
/// Shared with the Matroska muxer: `V_AV1` stores the same
/// av1C bytes as `CodecPrivate`.
pub(crate) fn av1c_payload(av1_config: &Av1Config) -> Vec<u8> {
    let mut payload = Vec::new();

    // Byte 0: marker (1) + version (7) = 0x81
    payload.push(0x81);

    // Byte 1: seq_profile (3) + seq_level_idx_0 (5)
    let byte1 = ((av1_config.seq_profile & 0x07) << 5) | (av1_config.seq_level_idx & 0x1f);
    payload.push(byte1);

    // Byte 2: seq_tier_0 (1) + high_bitdepth (1) + twelve_bit (1) + monochrome (1)
    //       + chroma_subsampling_x (1) + chroma_subsampling_y (1) + chroma_sample_position (2)
    let byte2 = ((av1_config.seq_tier & 0x01) << 7)
        | (if av1_config.high_bitdepth { 0x40 } else { 0 })
        | (if av1_config.twelve_bit { 0x20 } else { 0 })
        | (if av1_config.monochrome { 0x10 } else { 0 })
        | (if av1_config.chroma_subsampling_x {
            0x08
        } else {
            0
        })
        | (if av1_config.chroma_subsampling_y {
            0x04
        } else {
            0
        })
        | (av1_config.chroma_sample_position & 0x03);
    payload.push(byte2);

    // Byte 3: reserved (1) + initial_presentation_delay_present (1) + reserved (4) OR initial_presentation_delay_minus_one (4)
    // Set to 0 (no initial presentation delay)
    payload.push(0x00);

    // configOBUs: Append the Sequence Header OBU
    payload.extend_from_slice(&av1_config.sequence_header);

    payload
}

fn build_vp09_box(video: &Mp4VideoTrack, vp9_config: &Vp9Config) -> Vec<u8> {
    // INV-002: Width/height must fit in 16-bit for visual sample entry
    assert_invariant!(
        video.width <= u16::MAX as u32,
        "Width must fit in 16-bit",
        "build_vp09_box"
    );
    assert_invariant!(
        video.height <= u16::MAX as u32,
        "Height must fit in 16-bit",
        "build_vp09_box"
    );

    let mut payload = Vec::new();
    // Reserved (6 bytes)
    payload.extend_from_slice(&[0u8; 6]);
    // Data reference index
    payload.extend_from_slice(&1u16.to_be_bytes());
    // Pre-defined + reserved
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Width and height are 16-bit values in the visual sample entry
    payload.extend_from_slice(&(video.width as u16).to_be_bytes());
    payload.extend_from_slice(&(video.height as u16).to_be_bytes());
    // Horizontal/vertical resolution (72 dpi fixed point)
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    payload.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    // Reserved
    payload.extend_from_slice(&0u32.to_be_bytes());
    // Frame count
    payload.extend_from_slice(&1u16.to_be_bytes());
    // Compressor name (32 bytes, empty)
    payload.extend_from_slice(&[0u8; 32]);
    // Depth (24-bit)
    payload.extend_from_slice(&0x0018u16.to_be_bytes());
    // Pre-defined (-1)
    payload.extend_from_slice(&0xffffu16.to_be_bytes());
    // vpcC box
    let vpcc_box = build_vpcc_box(vp9_config);
    payload.extend_from_slice(&vpcc_box);
    build_box(b"vp09", &payload)
}

/// Build a vpcC configuration box for VP9.
///
/// Layout follows "VP Codec ISO Media File Format Binding": a FullBox
/// (version 1, flags 0) whose record packs bit depth, chroma subsampling
/// and range into a single byte and ends with
/// `codecInitializationDataSize = 0` (mandatory zero for VP9).
fn build_vpcc_box(vp9_config: &Vp9Config) -> Vec<u8> {
    build_box(b"vpcC", &vpcc_payload(vp9_config))
}

fn build_vmhd_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    build_box(b"vmhd", &payload)
}

fn build_dinf_box() -> Vec<u8> {
    let dref_box = build_dref_box();
    build_box(b"dinf", &dref_box)
}

fn build_dref_box() -> Vec<u8> {
    let url_box = build_url_box();
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&url_box);
    build_box(b"dref", &payload)
}

fn build_url_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&1u32.to_be_bytes());
    build_box(b"url ", &payload)
}

fn encode_language_code(language: &str) -> [u8; 2] {
    // ISO 639-2/T language codes are packed into 16 bits as (c1<<10) | (c2<<5) | c3
    // where each character is offset by 0x60
    let chars: Vec<char> = language.chars().take(3).collect();
    let c1 = chars.first().copied().unwrap_or('u') as u16;
    let c2 = chars.get(1).copied().unwrap_or('n') as u16;
    let c3 = chars.get(2).copied().unwrap_or('d') as u16;

    let packed = ((c1.saturating_sub(0x60) & 0x1F) << 10)
        | ((c2.saturating_sub(0x60) & 0x1F) << 5)
        | (c3.saturating_sub(0x60) & 0x1F);

    packed.to_be_bytes()
}

fn build_mdhd_box_with_timescale_and_duration(
    timescale: u32,
    duration: u64,
    language: Option<&str>,
) -> Vec<u8> {
    let mut payload = Vec::new();
    if duration <= u64::from(u32::MAX) {
        payload.extend_from_slice(&0u32.to_be_bytes()); // version 0 + flags
        payload.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        payload.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        payload.extend_from_slice(&timescale.to_be_bytes());
        payload.extend_from_slice(&(duration as u32).to_be_bytes());
    } else {
        payload.extend_from_slice(&0x0100_0000u32.to_be_bytes()); // version 1
        payload.extend_from_slice(&0u64.to_be_bytes()); // creation_time
        payload.extend_from_slice(&0u64.to_be_bytes()); // modification_time
        payload.extend_from_slice(&timescale.to_be_bytes());
        payload.extend_from_slice(&duration.to_be_bytes());
    }
    payload.extend_from_slice(&encode_language_code(language.unwrap_or("und"))); // language
    payload.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    build_box(b"mdhd", &payload)
}

fn build_hdlr_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(b"vide");
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(b"VideoHandler");
    payload.push(0);
    build_box(b"hdlr", &payload)
}

fn build_sound_hdlr_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(b"soun");
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(b"SoundHandler");
    payload.push(0);
    build_box(b"hdlr", &payload)
}

fn build_subtitle_hdlr_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(b"sbtl");
    payload.extend_from_slice(&[0u8; 12]);
    payload.extend_from_slice(b"SubtitleHandler");
    payload.push(0);
    build_box(b"hdlr", &payload)
}

fn build_smhd_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    build_box(b"smhd", &payload)
}

fn build_nmhd_box() -> Vec<u8> {
    build_box(b"nmhd", &0u32.to_be_bytes())
}

fn build_tkhd_box(video: &Mp4VideoTrack) -> Vec<u8> {
    build_tkhd_box_with_id(1, 0, video.width, video.height)
}

fn build_tkhd_box_with_id(track_id: u32, volume: u16, width: u32, height: u32) -> Vec<u8> {
    build_tkhd_box_with_duration(track_id, volume, width, height, 0)
}

fn build_tkhd_box_with_duration(
    track_id: u32,
    volume: u16,
    width: u32,
    height: u32,
    duration: u64,
) -> Vec<u8> {
    let mut payload = Vec::new();
    if duration <= u64::from(u32::MAX) {
        payload.extend_from_slice(&0u32.to_be_bytes()); // v0
        payload.extend_from_slice(&0u32.to_be_bytes()); // creation
        payload.extend_from_slice(&0u32.to_be_bytes()); // modification
        payload.extend_from_slice(&track_id.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes()); // reserved
        payload.extend_from_slice(&(duration as u32).to_be_bytes());
    } else {
        payload.extend_from_slice(&0x0100_0000u32.to_be_bytes()); // v1
        payload.extend_from_slice(&0u64.to_be_bytes()); // creation
        payload.extend_from_slice(&0u64.to_be_bytes()); // modification
        payload.extend_from_slice(&track_id.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes()); // reserved
        payload.extend_from_slice(&duration.to_be_bytes());
    }
    payload.extend_from_slice(&0u32.to_be_bytes()); // reserved
    payload.extend_from_slice(&0u32.to_be_bytes()); // reserved
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(&volume.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    let matrix = [
        0x0001_0000_u32,
        0,
        0,
        0,
        0x0001_0000_u32,
        0,
        0,
        0,
        0x4000_0000_u32,
    ];
    for value in matrix {
        payload.extend_from_slice(&value.to_be_bytes());
    }
    let width_fixed = width << 16;
    let height_fixed = height << 16;
    payload.extend_from_slice(&width_fixed.to_be_bytes());
    payload.extend_from_slice(&height_fixed.to_be_bytes());
    build_box(b"tkhd", &payload)
}

pub(crate) fn build_ftyp_box() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"isom");
    payload.extend_from_slice(&0x200_u32.to_be_bytes());
    payload.extend_from_slice(b"isommp41");
    build_box(b"ftyp", &payload)
}

fn build_mvhd_payload(duration_ms: u64, next_track_id: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    if duration_ms <= u64::from(u32::MAX) {
        payload.extend_from_slice(&0u32.to_be_bytes()); // v0
        payload.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        payload.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        payload.extend_from_slice(&MOVIE_TIMESCALE.to_be_bytes());
        payload.extend_from_slice(&(duration_ms as u32).to_be_bytes());
    } else {
        payload.extend_from_slice(&0x0100_0000u32.to_be_bytes()); // v1
        payload.extend_from_slice(&0u64.to_be_bytes()); // creation_time
        payload.extend_from_slice(&0u64.to_be_bytes()); // modification_time
        payload.extend_from_slice(&MOVIE_TIMESCALE.to_be_bytes());
        payload.extend_from_slice(&duration_ms.to_be_bytes());
    }
    payload.extend_from_slice(&0x0001_0000_u32.to_be_bytes()); // rate (1.0)
    payload.extend_from_slice(&0x0100u16.to_be_bytes()); // volume (1.0)
    payload.extend_from_slice(&0u16.to_be_bytes()); // reserved
    payload.extend_from_slice(&0u64.to_be_bytes()); // reserved
    let matrix = [
        0x0001_0000_u32,
        0,
        0,
        0,
        0x0001_0000_u32,
        0,
        0,
        0,
        0x4000_0000_u32,
    ];
    for value in matrix {
        payload.extend_from_slice(&value.to_be_bytes());
    }
    for _ in 0..6 {
        payload.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
    }
    payload.extend_from_slice(&next_track_id.to_be_bytes()); // next_track_ID
    payload
}

fn build_box(typ: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let total = 8u64
        .checked_add(payload.len() as u64)
        .expect("box payload size overflow");
    let mut buffer = Vec::with_capacity((total.min(16 * 1024 * 1024)) as usize + 8);
    if total <= u64::from(u32::MAX) {
        buffer.extend_from_slice(&(total as u32).to_be_bytes());
        buffer.extend_from_slice(typ);
    } else {
        // Large-size: 32-bit size = 1 followed by 64-bit largesize.
        buffer.extend_from_slice(&1u32.to_be_bytes());
        buffer.extend_from_slice(typ);
        buffer.extend_from_slice(&total.to_be_bytes());
    }
    buffer.extend_from_slice(payload);

    // INV-001: Box size must equal header + payload length
    assert_invariant!(
        buffer.len() as u64 == total + if total > u64::from(u32::MAX) { 8 } else { 0 },
        "Box size must equal header + payload",
        "build_box"
    );

    buffer
}

/// Write a box header with 64-bit largesize when needed; returns header bytes.
fn box_header(typ: &[u8; 4], total_size: u64) -> Vec<u8> {
    if total_size <= u64::from(u32::MAX) {
        let mut h = Vec::with_capacity(8);
        h.extend_from_slice(&(total_size as u32).to_be_bytes());
        h.extend_from_slice(typ);
        h
    } else {
        let mut h = Vec::with_capacity(16);
        h.extend_from_slice(&1u32.to_be_bytes());
        h.extend_from_slice(typ);
        h.extend_from_slice(&total_size.to_be_bytes());
        h
    }
}

/// mdat header bytes (8 or 16) for a given payload size (verdict §6).
fn mdat_header_bytes(payload_size: u64) -> io::Result<Vec<u8>> {
    let total = 8u64
        .checked_add(payload_size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "mdat size overflow"))?;
    // Large mdat needs 16-byte header (size=1 + type + largesize).
    let total_with_large = if total > u64::from(u32::MAX) {
        16u64
            .checked_add(payload_size)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "mdat size overflow"))?
    } else {
        total
    };
    Ok(box_header(b"mdat", total_with_large))
}

/// mdat header length (8 or 16) for offset math.
fn mdat_header_len(payload_size: u64) -> u64 {
    let total = 8u64.saturating_add(payload_size);
    if total > u64::from(u32::MAX) { 16 } else { 8 }
}

// ============================================================================
// Metadata (udta/meta/ilst) box building
// ============================================================================

fn build_udta_box(metadata: &Metadata) -> Vec<u8> {
    let mut ilst_payload = Vec::new();

    if let Some(title) = &metadata.title {
        ilst_payload.extend_from_slice(&build_ilst_string_item(b"\xa9nam", title));
    }

    if let Some(creation_time) = metadata.creation_time {
        // Format as ISO 8601: "YYYY-MM-DDTHH:MM:SSZ"
        let date_str = format_unix_timestamp(creation_time);
        ilst_payload.extend_from_slice(&build_ilst_string_item(b"\xa9day", &date_str));
    }

    if ilst_payload.is_empty() {
        return Vec::new(); // No metadata, skip udta entirely
    }

    let ilst_box = build_box(b"ilst", &ilst_payload);

    // meta box requires hdlr
    let hdlr_box = build_meta_hdlr_box();

    // meta is a full box (version + flags)
    let mut meta_payload = vec![0u8; 4]; // version=0, flags=0
    meta_payload.extend_from_slice(&hdlr_box);
    meta_payload.extend_from_slice(&ilst_box);
    let meta_box = build_box(b"meta", &meta_payload);

    build_box(b"udta", &meta_box)
}

fn build_ilst_string_item(atom_type: &[u8; 4], value: &str) -> Vec<u8> {
    // data box: type indicator (1 = UTF-8) + locale (0) + string
    let mut data_payload = Vec::new();
    data_payload.extend_from_slice(&[0, 0, 0, 1]); // type = UTF-8
    data_payload.extend_from_slice(&[0, 0, 0, 0]); // locale = 0
    data_payload.extend_from_slice(value.as_bytes());

    let data_box = build_box(b"data", &data_payload);
    build_box(atom_type, &data_box)
}

fn build_meta_hdlr_box() -> Vec<u8> {
    let mut payload = vec![0u8; 4]; // version + flags
    payload.extend_from_slice(&[0, 0, 0, 0]); // pre_defined
    payload.extend_from_slice(b"mdir"); // handler_type (metadata directory)
    payload.extend_from_slice(b"appl"); // manufacturer
    payload.extend_from_slice(&[0, 0, 0, 0]); // reserved
    payload.extend_from_slice(&[0, 0, 0, 0]); // reserved
    payload.push(0); // name (empty, null-terminated)
    build_box(b"hdlr", &payload)
}

fn format_unix_timestamp(unix_secs: u64) -> String {
    // Simple conversion - days since epoch calculation
    // This is approximate but good enough for metadata
    const SECS_PER_MIN: u64 = 60;
    const SECS_PER_HOUR: u64 = 3600;
    const SECS_PER_DAY: u64 = 86400;

    let days_since_epoch = unix_secs / SECS_PER_DAY;
    let remaining_secs = unix_secs % SECS_PER_DAY;

    let hours = remaining_secs / SECS_PER_HOUR;
    let minutes = (remaining_secs % SECS_PER_HOUR) / SECS_PER_MIN;
    let seconds = remaining_secs % SECS_PER_MIN;

    // Calculate year, month, day from days since 1970-01-01
    // Using a simplified algorithm
    let (year, month, day) = days_to_ymd(days_since_epoch);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    // Simplified algorithm - works for dates from 1970 to ~2100
    let mut remaining_days = days as i64;
    let mut year = 1970u32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u32;
    for &days_in_month in &days_in_months {
        if remaining_days < days_in_month {
            break;
        }
        remaining_days -= days_in_month;
        month += 1;
    }

    let day = (remaining_days + 1) as u32;
    (year, month, day)
}

fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn h264_keyframe() -> Vec<u8> {
        vec![
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11,
            0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0xaa,
            0xbb, 0xcc, 0xdd,
        ]
    }

    #[test]
    fn mp4_writer_error_display_covers_all_variants() {
        let variants = [
            Mp4WriterError::NonIncreasingTimestamp,
            Mp4WriterError::FirstFrameMustBeKeyframe,
            Mp4WriterError::FirstFrameMissingSpsPps,
            Mp4WriterError::FirstFrameMissingSequenceHeader,
            Mp4WriterError::InvalidAdts,
            Mp4WriterError::InvalidAdtsDetailed(Box::new(AdtsValidationError {
                kind: AdtsErrorKind::FrameTooShort,
                severity: ErrorSeverity::Error,
                byte_offset: 5,
                expected: Some("≥7 bytes for ADTS header".to_string()),
                found: Some("5 bytes".to_string()),
                hex_dump: Some("00 01 02 03 04* 05 06 07 08 09 (showing bytes 0-9)".to_string()),
                suggestion: Some("Ensure you're passing complete ADTS frames. Check if the audio data is truncated or corrupted during transmission.".to_string()),
                code_example: None,
                technical_details: Some("ADTS header requires minimum 7 bytes: syncword (2 bytes), MPEG info (1 byte), frame length (3 bytes partial), buffer fullness (2 bytes partial).".to_string()),
                related_errors: Vec::new(),
            })),
            Mp4WriterError::InvalidOpusPacket,
            Mp4WriterError::AudioNotEnabled,
            Mp4WriterError::DurationOverflow,
            Mp4WriterError::AlreadyFinalized,
        ];

        for v in variants {
            let s = format!("{v}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn write_video_with_dts_enforces_first_keyframe_and_codec_config() {
        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        writer.enable_video(VideoCodec::H264);

        let not_keyframe = vec![0x00, 0x00, 0x00, 0x01, 0x41, 0x9a, 0x24, 0x6c];
        assert!(matches!(
            writer.write_video_sample_with_dts(0, 0, &not_keyframe, false),
            Err(Mp4WriterError::FirstFrameMustBeKeyframe)
        ));

        // H.265 requires VPS/SPS/PPS; feed an H.264-ish keyframe and expect config failure.
        let sink = Cursor::new(Vec::<u8>::new());
        let mut hevc = Mp4Writer::new(sink);
        hevc.enable_video(VideoCodec::H265);
        assert!(matches!(
            hevc.write_video_sample_with_dts(0, 0, &h264_keyframe(), true),
            Err(Mp4WriterError::FirstFrameMissingSpsPps)
        ));

        // AV1 requires a Sequence Header OBU.
        let sink = Cursor::new(Vec::<u8>::new());
        let mut av1 = Mp4Writer::new(sink);
        av1.enable_video(VideoCodec::Av1);
        assert!(matches!(
            av1.write_video_sample_with_dts(0, 0, &h264_keyframe(), true),
            Err(Mp4WriterError::FirstFrameMissingSequenceHeader)
        ));
    }

    #[test]
    fn write_video_with_dts_enforces_monotonic_dts_and_duration_bounds() {
        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        writer.enable_video(VideoCodec::H264);
        writer
            .write_video_sample_with_dts(0, 0, &h264_keyframe(), true)
            .unwrap();

        // Non-increasing DTS.
        assert!(matches!(
            writer.write_video_sample_with_dts(3000, 0, &h264_keyframe(), false),
            Err(Mp4WriterError::NonIncreasingTimestamp)
        ));

        // Duration overflow (delta > u32::MAX).
        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        writer.enable_video(VideoCodec::H264);
        writer
            .write_video_sample_with_dts(0, 0, &h264_keyframe(), true)
            .unwrap();
        let big_delta = u64::from(u32::MAX) + 1;
        assert!(matches!(
            writer.write_video_sample_with_dts(big_delta, big_delta, &h264_keyframe(), false),
            Err(Mp4WriterError::DurationOverflow)
        ));

        // Normal delta updates previous sample duration.
        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        writer.enable_video(VideoCodec::H264);
        writer
            .write_video_sample_with_dts(0, 0, &h264_keyframe(), true)
            .unwrap();
        writer
            .write_video_sample_with_dts(3000, 3000, &h264_keyframe(), false)
            .unwrap();
        assert_eq!(writer.video_samples[0].duration, Some(3000));
    }

    #[test]
    fn write_audio_sample_covers_disabled_and_invalid_inputs() {
        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        assert!(matches!(
            writer.write_audio_sample(0, &[0u8; 3]),
            Err(Mp4WriterError::AudioNotEnabled)
        ));

        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        writer.enable_audio(Mp4AudioTrack {
            sample_rate: 48000,
            channels: 2,
            codec: AudioCodec::Aac(AacProfile::Lc),
            flac_streaminfo: None,
            opus_preskip: None,
            aac_asc_override: None,
            opus_config_override: None,
            language: None,
        });
        assert!(matches!(
            writer.write_audio_sample(0, &[0x00, 0x01, 0x02]),
            Err(Mp4WriterError::InvalidAdtsDetailed(_))
        ));

        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        writer.enable_audio(Mp4AudioTrack {
            sample_rate: 48000,
            channels: 2,
            codec: AudioCodec::Opus,
            flac_streaminfo: None,
            opus_preskip: None,
            aac_asc_override: None,
            opus_config_override: None,
            language: None,
        });
        assert!(matches!(
            writer.write_audio_sample(0, &[]),
            Err(Mp4WriterError::InvalidOpusPacket)
        ));
    }

    #[test]
    fn finalize_covers_empty_video_default_config_and_double_finalize() {
        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        writer.enable_video(VideoCodec::H264);
        let video = Mp4VideoTrack {
            width: 640,
            height: 480,
            language: None,
        };

        writer.finalize(Some(&video), None, false).unwrap();
        // Second finalize hits the already-finalized error.
        assert!(writer.finalize(Some(&video), None, false).is_err());
    }

    #[test]
    fn write_rejects_after_finalize() {
        let sink = Cursor::new(Vec::<u8>::new());
        let mut writer = Mp4Writer::new(sink);
        writer.enable_video(VideoCodec::H264);
        let video = Mp4VideoTrack {
            width: 640,
            height: 480,
            language: None,
        };

        writer
            .write_video_sample_with_dts(0, 0, &h264_keyframe(), true)
            .unwrap();
        writer.finalize(Some(&video), None, true).unwrap();

        assert!(matches!(
            writer.write_video_sample_with_dts(3000, 3000, &h264_keyframe(), false),
            Err(Mp4WriterError::AlreadyFinalized)
        ));
    }

    #[test]
    fn aac_profile_validation_accepts_all_supported_profiles() {
        use crate::api::AacProfile;

        let supported_profiles = [
            AacProfile::Lc,
            AacProfile::Main,
            AacProfile::Ssr,
            AacProfile::Ltp,
            AacProfile::He,
            AacProfile::Hev2,
        ];

        for profile in supported_profiles {
            let sink = Cursor::new(Vec::<u8>::new());
            let mut writer = Mp4Writer::new(sink);
            writer.enable_audio(Mp4AudioTrack {
                sample_rate: 48000,
                channels: 2,
                codec: AudioCodec::Aac(profile),
                flac_streaminfo: None,
                opus_preskip: None,
                aac_asc_override: None,
                opus_config_override: None,
                language: None,
            });

            // Create a minimal valid ADTS frame for testing
            // This is a simplified ADTS header that should pass basic validation
            let adts_frame = vec![
                0xFF, 0xF1, // Syncword + MPEG-4 + Layer AAC + protection absent
                0x4C, 0x80, // Profile LC + sample rate 44100 + private bit + channels 2
                0x1F, 0xFC, // Frame length (31 bytes) + buffer fullness
                0x00, 0x00, // Buffer fullness continued + raw data block count
                // Raw AAC data (minimal)
                0x21, 0x00, 0x49, 0x90, 0x02, 0x19, 0x00, 0x23, 0x80,
            ];

            // The profile validation happens in the invariant check
            // If the profile is not supported, it would panic in debug mode
            // In release mode, it would continue but we test that it doesn't fail due to profile
            let result = writer.write_audio_sample(0, &adts_frame);
            // We expect either success or ADTS validation failure, but not profile-related failure
            assert!(!matches!(result, Err(Mp4WriterError::InvalidAdtsDetailed(_)) if false));
        }
    }

    #[test]
    fn adts_to_raw_validates_frame_structure() {
        // Test frame too short
        let short_frame = vec![0xFF, 0xF1, 0x4C];
        let result = adts_to_raw(&short_frame);
        assert!(matches!(
            result,
            Err(AdtsValidationError {
                kind: AdtsErrorKind::FrameTooShort,
                ..
            })
        ));

        // Test invalid syncword
        let bad_sync = vec![
            0xFE, 0xF1, // Invalid syncword
            0x4C, 0x80, 0x1F, 0xFC, 0x00, 0x00, 0x21, 0x00, 0x49, 0x90, 0x02, 0x19, 0x00, 0x23,
            0x80,
        ];
        let result = adts_to_raw(&bad_sync);
        assert!(matches!(
            result,
            Err(AdtsValidationError {
                kind: AdtsErrorKind::MissingSyncword,
                ..
            })
        ));

        // Test invalid MPEG version (MPEG-2)
        let mpeg2_frame = vec![
            0xFF, 0xF9, // MPEG-2 bit set
            0x4C, 0x80, 0x1F, 0xFC, 0x00, 0x00, 0x21, 0x00, 0x49, 0x90, 0x02, 0x19, 0x00, 0x23,
            0x80,
        ];
        let result = adts_to_raw(&mpeg2_frame);
        assert!(matches!(
            result,
            Err(AdtsValidationError {
                kind: AdtsErrorKind::InvalidMpegVersion,
                ..
            })
        ));

        // Test invalid layer (not AAC)
        let non_aac_layer = vec![
            0xFF, 0xF5, // Layer set to 01 (Layer 3)
            0x4C, 0x80, 0x1F, 0xFC, 0x00, 0x00, 0x21, 0x00, 0x49, 0x90, 0x02, 0x19, 0x00, 0x23,
            0x80,
        ];
        let result = adts_to_raw(&non_aac_layer);
        assert!(matches!(
            result,
            Err(AdtsValidationError {
                kind: AdtsErrorKind::InvalidLayer,
                ..
            })
        ));
    }

    #[test]
    fn build_audio_specific_config_standard_rates() {
        // Test standard AAC sample rates
        assert_eq!(build_audio_specific_config(44100, 2), [0x12, 0x10]); // 44100 Hz, stereo
        assert_eq!(build_audio_specific_config(48000, 2), [0x11, 0x90]); // 48000 Hz, stereo
        assert_eq!(build_audio_specific_config(22050, 1), [0x13, 0x88]); // 22050 Hz, mono
        assert_eq!(build_audio_specific_config(8000, 1), [0x15, 0x88]); // 8000 Hz, mono
    }

    #[test]
    fn build_audio_specific_config_edge_cases() {
        // Test non-standard rate (should default to 44100)
        assert_eq!(build_audio_specific_config(12345, 2), [0x12, 0x10]);

        // Test channel limits (max 15 channels)
        assert_eq!(build_audio_specific_config(44100, 16), [0x12, 0x78]); // 15 channels max
        assert_eq!(build_audio_specific_config(44100, 0), [0x12, 0x00]); // 0 channels
    }

    #[test]
    fn build_stts_box_empty_durations() {
        let durations = Vec::new();
        let box_data = build_stts_box(&durations);
        // Box format: length(4) + "stts"(4) + version/flags(4) + entry_count(4)
        assert_eq!(box_data.len(), 16);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 16]); // length = 16
        assert_eq!(&box_data[4..8], b"stts"); // box type
        assert_eq!(box_data[8..12], [0, 0, 0, 0]); // version/flags = 0
        assert_eq!(box_data[12..16], [0, 0, 0, 0]); // entry_count = 0
    }

    #[test]
    fn build_stts_box_single_duration() {
        let durations = vec![3000];
        let box_data = build_stts_box(&durations);
        // Box format: length(4) + "stts"(4) + version/flags(4) + entry_count(4) + sample_count(4) + sample_delta(4)
        assert_eq!(box_data.len(), 24);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 24]); // length = 24
        assert_eq!(&box_data[4..8], b"stts"); // box type
        assert_eq!(box_data[8..12], [0, 0, 0, 0]); // version/flags = 0
        assert_eq!(box_data[12..16], [0, 0, 0, 1]); // entry_count = 1
        assert_eq!(box_data[16..20], [0, 0, 0, 1]); // sample_count = 1
        assert_eq!(box_data[20..24], [0, 0, 0x0B, 0xB8]); // sample_delta = 3000
    }

    #[test]
    fn build_stsc_box_single_chunk() {
        let box_data = build_stsc_box(1, 1);
        // Box format: length(4) + "stsc"(4) + version/flags(4) + entry_count(4) + first_chunk(4) + samples_per_chunk(4) + sample_description_index(4)
        assert_eq!(box_data.len(), 28);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 28]); // length = 28
        assert_eq!(&box_data[4..8], b"stsc"); // box type
        assert_eq!(box_data[8..12], [0, 0, 0, 0]); // version/flags = 0
        assert_eq!(box_data[12..16], [0, 0, 0, 1]); // entry_count = 1
        assert_eq!(box_data[16..20], [0, 0, 0, 1]); // first_chunk = 1
        assert_eq!(box_data[20..24], [0, 0, 0, 1]); // samples_per_chunk = 1
        assert_eq!(box_data[24..28], [0, 0, 0, 1]); // sample_description_index = 1
    }

    #[test]
    fn build_stsz_box_empty_sizes() {
        let sizes = Vec::new();
        let box_data = build_stsz_box(&sizes);
        // Box format: length(4) + "stsz"(4) + version/flags(4) + sample_size(4) + sample_count(4)
        assert_eq!(box_data.len(), 20);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 20]); // length = 20
        assert_eq!(&box_data[4..8], b"stsz"); // box type
        assert_eq!(box_data[8..12], [0, 0, 0, 0]); // version/flags = 0
        assert_eq!(box_data[12..16], [0, 0, 0, 0]); // sample_size = 0
        assert_eq!(box_data[16..20], [0, 0, 0, 0]); // sample_count = 0
    }

    #[test]
    fn build_stsz_box_uniform_sizes() {
        let sizes = vec![1024; 3];
        let box_data = build_stsz_box(&sizes);
        // Box format: length(4) + "stsz"(4) + version/flags(4) + sample_size(4) + sample_count(4) + sizes(4*3)
        assert_eq!(box_data.len(), 32);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 32]); // length = 32
        assert_eq!(&box_data[4..8], b"stsz"); // box type
        assert_eq!(box_data[8..12], [0, 0, 0, 0]); // version/flags = 0
        assert_eq!(box_data[12..16], [0, 0, 0, 0]); // sample_size = 0 (variable)
        assert_eq!(box_data[16..20], [0, 0, 0, 3]); // sample_count = 3
        // Individual sizes
        assert_eq!(box_data[20..24], [0, 0, 0x04, 0x00]); // size[0] = 1024
        assert_eq!(box_data[24..28], [0, 0, 0x04, 0x00]); // size[1] = 1024
        assert_eq!(box_data[28..32], [0, 0, 0x04, 0x00]); // size[2] = 1024
    }

    #[test]
    fn build_stsz_box_variable_sizes() {
        let sizes = vec![100, 200, 300];
        let box_data = build_stsz_box(&sizes);
        // Box format: length(4) + "stsz"(4) + version/flags(4) + sample_size(4) + sample_count(4) + sizes(4*3)
        // Total length: 8 (header) + 16 (fixed fields) + 12 (sizes) = 36? Wait, no: header is 8, payload is 4+4+4+12=24, total 32
        assert_eq!(box_data.len(), 32);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 32]); // length = 32
        assert_eq!(&box_data[4..8], b"stsz"); // box type
        assert_eq!(box_data[8..12], [0, 0, 0, 0]); // version/flags = 0
        assert_eq!(box_data[12..16], [0, 0, 0, 0]); // sample_size = 0 (variable)
        assert_eq!(box_data[16..20], [0, 0, 0, 3]); // sample_count = 3
        assert_eq!(box_data[20..24], [0, 0, 0, 100]); // size[0] = 100
        assert_eq!(box_data[24..28], [0, 0, 0, 200]); // size[1] = 200
        assert_eq!(box_data[28..32], [0, 0, 1, 44]); // size[2] = 300 (0x0000012C)
    }

    #[test]
    fn build_stco_box_single_offset() {
        let offsets = vec![1000];
        let box_data = build_stco_box(&offsets);
        // Box format: length(4) + "stco"(4) + version/flags(4) + entry_count(4) + offsets(4*1)
        assert_eq!(box_data.len(), 20);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 20]); // length = 20
        assert_eq!(&box_data[4..8], b"stco"); // box type
        assert_eq!(box_data[8..12], [0, 0, 0, 0]); // version/flags = 0
        assert_eq!(box_data[12..16], [0, 0, 0, 1]); // entry_count = 1
        assert_eq!(box_data[16..20], [0, 0, 0x03, 0xe8]); // offset[0] = 1000
    }

    #[test]
    fn build_stss_box_single_keyframe() {
        let keyframes = vec![1];
        let box_data = build_stss_box(&keyframes);
        // Box format: length(4) + "stss"(4) + version/flags(4) + entry_count(4) + keyframes(4*1)
        assert_eq!(box_data.len(), 20);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 20]); // length = 20
        assert_eq!(&box_data[4..8], b"stss"); // box type
        assert_eq!(box_data[8..12], [0, 0, 0, 0]); // version/flags = 0
        assert_eq!(box_data[12..16], [0, 0, 0, 1]); // entry_count = 1
        assert_eq!(box_data[16..20], [0, 0, 0, 1]); // keyframe[0] = 1
    }

    #[test]
    fn build_ctts_box_single_offset() {
        let cts_offsets = vec![3000];
        let box_data = build_ctts_box(&cts_offsets);
        // Box format: length(4) + "ctts"(4) + version/flags(4) + entry_count(4) + entries(8*1)
        assert_eq!(box_data.len(), 24);
        assert_eq!(&box_data[0..4], &[0, 0, 0, 24]); // length = 24
        assert_eq!(&box_data[4..8], b"ctts"); // box type
        assert_eq!(box_data[8..12], [1, 0, 0, 0]); // version=1, flags=0
        assert_eq!(box_data[12..16], [0, 0, 0, 1]); // entry_count = 1
        assert_eq!(box_data[16..20], [0, 0, 0, 1]); // sample_count = 1
        assert_eq!(box_data[20..24], [0, 0, 0x0b, 0xb8]); // sample_offset = 3000
    }
}
