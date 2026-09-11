use crate::assert_invariant;
use crate::codec::common::AnnexBNalIter;
use crate::codec::vp9::is_vp9_keyframe;
use crate::fragmented::{FragmentConfig, FragmentedMuxer};
use crate::muxer::mkv::{MkvContainer, MkvWriter, MkvWriterError};
/// Public API definitions for the Muxfin crate.
///
/// This module contains the types and traits that form the public contract
/// for users of the crate.  Concrete implementations live in private
/// modules.  The API defined here intentionally exposes only the
/// capabilities promised by the charter and contract documents.  It does
/// not contain any implementation details.
use crate::muxer::mp4::{
    MEDIA_TIMESCALE, Mp4AudioTrack, Mp4SubtitleTrack, Mp4VideoTrack, Mp4Writer, Mp4WriterError,
};
use crate::muxer::streaming::SeekableStreamingWriter;
use crate::time::{EncodedSample, LanguageCode, Limits, SubtitleCue};
use std::fmt;
use std::io::Write;

/// Enumeration of supported video codecs for the initial version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264/AVC video codec.  Only the AVC Annex B stream format is
    /// currently supported.  B‑frames are not permitted in v0.
    H264,
    /// H.265/HEVC video codec. Annex B stream format with VPS/SPS/PPS.
    /// Requires first keyframe to contain VPS, SPS, and PPS NALs.
    H265,
    /// AV1 video codec. OBU (Open Bitstream Unit) stream format.
    /// Requires first keyframe to contain Sequence Header OBU.
    Av1,
    /// VP9 video codec. Compressed VP9 frames with frame headers.
    /// Requires first keyframe to contain sequence parameters.
    Vp9,
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoCodec::H264 => write!(f, "H.264"),
            VideoCodec::H265 => write!(f, "H.265"),
            VideoCodec::Av1 => write!(f, "AV1"),
            VideoCodec::Vp9 => write!(f, "VP9"),
        }
    }
}

impl std::str::FromStr for VideoCodec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "h264" | "h.264" | "avc" => Ok(VideoCodec::H264),
            "h265" | "h.265" | "hevc" => Ok(VideoCodec::H265),
            "av1" => Ok(VideoCodec::Av1),
            "vp9" => Ok(VideoCodec::Vp9),
            _ => Err(format!("Unknown video codec: {}", s)),
        }
    }
}

/// AAC profile variants supported by Muxfin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacProfile {
    /// AAC Low Complexity (LC) - most common profile.
    Lc,
    /// AAC Main profile - higher quality than LC.
    Main,
    /// AAC Scalable Sample Rate (SSR).
    Ssr,
    /// AAC Long Term Prediction (LTP).
    Ltp,
    /// HE-AAC (High Efficiency AAC) - LC + SBR.
    He,
    /// HE-AAC v2 - HE-AAC + PS (Parametric Stereo).
    Hev2,
}

/// Enumeration of supported audio codecs for the initial version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// AAC (Advanced Audio Coding) with ADTS framing. Supports multiple profiles.
    Aac(AacProfile),
    /// Opus audio codec. Raw Opus packets (no container framing).
    /// Sample rate is always 48kHz per Opus spec.
    Opus,
    /// FLAC (Free Lossless Audio Codec). Native FLAC frames (one frame
    /// per MP4 sample). Requires STREAMINFO via
    /// [`MuxerBuilder::with_flac_streaminfo`].
    Flac,
    /// No audio.  Use this variant when only video is being muxed.
    None,
}

/// Enumeration of supported subtitle codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleCodec {
    /// MP4 Timed Text (`tx3g`) payload.
    MovText,
    /// WebVTT cue inside `vttc` (`wvtt` sample entry).
    WebVtt,
}

/// Caller-supplied decoder configuration (verdict §8).
///
/// The codec parsers remain as convenience extractors
/// (`AvcConfig::extract`, ...), but callers may supply the complete
/// record from their encoder instead of relying on first-keyframe parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoDecoderConfig {
    AvcC(Vec<u8>),
    HvcC(Vec<u8>),
    Av1C(Vec<u8>),
    VpcC(Vec<u8>),
}

/// Input bitstream format for video samples (verdict §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoBitstreamFormat {
    AnnexB,
    LengthPrefixed { length_size: u8 },
    ObuStream,
    RawVp9,
}

/// VP9 chroma subsampling as parsed from the frame header (verdict §8).
/// Never guessed from profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    Cs420Vertical,
    Cs420Colocated,
    Cs422,
    Cs444,
}

/// AAC decoder config: caller-provided ASC required for HE/HEv2 (verdict §8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AacConfig {
    pub audio_specific_config: Vec<u8>,
    pub output_sample_rate: u32,
    pub samples_per_access_unit: u32,
}

/// Complete Opus `dOps` configuration (verdict §8).
/// Re-uses the codec-level struct so CLI/core cannot drift.
pub use crate::codec::opus::OpusConfig;

/// Segment flush policy for fragmented MP4 (verdict §11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentBoundary {
    Manual,
    Duration {
        target: u64,
        require_sync_sample: bool,
    },
}

impl fmt::Display for AudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioCodec::Aac(profile) => write!(f, "AAC-{}", profile),
            AudioCodec::Opus => write!(f, "Opus"),
            AudioCodec::Flac => write!(f, "FLAC"),
            AudioCodec::None => write!(f, "None"),
        }
    }
}

impl fmt::Display for SubtitleCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubtitleCodec::MovText => write!(f, "mov_text"),
            SubtitleCodec::WebVtt => write!(f, "webvtt"),
        }
    }
}

impl fmt::Display for AacProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AacProfile::Lc => write!(f, "LC"),
            AacProfile::Main => write!(f, "Main"),
            AacProfile::Ssr => write!(f, "SSR"),
            AacProfile::Ltp => write!(f, "LTP"),
            AacProfile::He => write!(f, "HE"),
            AacProfile::Hev2 => write!(f, "HEv2"),
        }
    }
}

impl std::str::FromStr for AudioCodec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "aac" | "aac-lc" => Ok(AudioCodec::Aac(AacProfile::Lc)),
            "aac-main" => Ok(AudioCodec::Aac(AacProfile::Main)),
            "aac-ssr" => Ok(AudioCodec::Aac(AacProfile::Ssr)),
            "aac-ltp" => Ok(AudioCodec::Aac(AacProfile::Ltp)),
            "aac-he" => Ok(AudioCodec::Aac(AacProfile::He)),
            "aac-hev2" => Ok(AudioCodec::Aac(AacProfile::Hev2)),
            "opus" => Ok(AudioCodec::Opus),
            "flac" => Ok(AudioCodec::Flac),
            "none" => Ok(AudioCodec::None),
            _ => Err(format!("Unknown audio codec: {}", s)),
        }
    }
}

impl std::str::FromStr for SubtitleCodec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mov_text" | "movtext" | "tx3g" => Ok(SubtitleCodec::MovText),
            "webvtt" | "wvtt" | "vtt" => Ok(SubtitleCodec::WebVtt),
            _ => Err(format!("Unknown subtitle codec: {}", s)),
        }
    }
}

/// Output container selected for muxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContainerFormat {
    /// ISO-BMFF MP4 (`.mp4`). Default; see [`Muxer`].
    #[default]
    Mp4,
    /// Matroska (`.mkv`): all muxfin codecs; see [`MkvMuxer`].
    Matroska,
    /// WebM (`.webm`): VP9/AV1 video and Opus audio only; see [`MkvMuxer`].
    WebM,
}

impl ContainerFormat {
    /// Conventional file extension for this container.
    pub fn extension(self) -> &'static str {
        match self {
            ContainerFormat::Mp4 => "mp4",
            ContainerFormat::Matroska => "mkv",
            ContainerFormat::WebM => "webm",
        }
    }

    /// Whether this format uses the Matroska/WebM muxer backend.
    pub fn is_matroska_family(self) -> bool {
        matches!(self, ContainerFormat::Matroska | ContainerFormat::WebM)
    }
}

impl fmt::Display for ContainerFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainerFormat::Mp4 => write!(f, "MP4"),
            ContainerFormat::Matroska => write!(f, "Matroska"),
            ContainerFormat::WebM => write!(f, "WebM"),
        }
    }
}

impl std::str::FromStr for ContainerFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mp4" | "m4v" | "isobmff" => Ok(ContainerFormat::Mp4),
            "mkv" | "matroska" | "mka" => Ok(ContainerFormat::Matroska),
            "webm" => Ok(ContainerFormat::WebM),
            _ => Err(format!(
                "Unknown container format: {} (expected mp4, mkv, or webm)",
                s
            )),
        }
    }
}

/// High-level muxer configuration intended for simple integrations (e.g. CrabCamera).
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    pub width: u32,
    pub height: u32,
    pub framerate: f64,
    pub audio: Option<AudioTrackConfig>,
    pub metadata: Option<Metadata>,
    pub fast_start: bool,
}

/// Metadata to embed in the MP4 file (title, creation time, etc.)
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    /// Title of the recording (appears in media players)
    pub title: Option<String>,
    /// Creation timestamp in seconds since Unix epoch (1970-01-01)
    pub creation_time: Option<u64>,
    /// Language code (ISO 639-2/T format, e.g., "eng", "spa", "und")
    pub language: Option<String>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_creation_time(mut self, unix_timestamp: u64) -> Self {
        self.creation_time = Some(unix_timestamp);
        self
    }

    /// Set creation time to current system time
    pub fn with_current_time(mut self) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        if let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) {
            self.creation_time = Some(duration.as_secs());
        }
        self
    }

    /// Set language code (ISO 639-2/T format, e.g., "eng", "spa", "und")
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

impl MuxerConfig {
    pub fn new(width: u32, height: u32, framerate: f64) -> Self {
        Self {
            width,
            height,
            framerate,
            audio: None,
            metadata: None,
            fast_start: true, // Default ON for web compatibility
        }
    }

    pub fn with_audio(mut self, codec: AudioCodec, sample_rate: u32, channels: u16) -> Self {
        if codec == AudioCodec::None {
            self.audio = None;
        } else {
            let timescale = match codec {
                AudioCodec::Opus => 48_000,
                _ => sample_rate,
            };
            self.audio = Some(AudioTrackConfig {
                codec,
                sample_rate,
                channels,
                timescale,
                language: LanguageCode::UND,
            });
        }
        self
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_fast_start(mut self, enabled: bool) -> Self {
        self.fast_start = enabled;
        self
    }
}

/// Summary statistics returned when finishing a mux.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MuxerStats {
    pub video_frames: u64,
    pub audio_frames: u64,
    pub subtitle_frames: u64,
    pub duration_secs: f64,
    pub bytes_written: u64,
}

/// Builder for constructing a new muxer instance.
///
/// The builder follows a fluent API pattern: each method returns a
/// modified builder, allowing method chaining.  Only the configuration
/// necessary for the initial v0 release is included.  Additional
/// configuration (such as B‑frame support, fragmented MP4 or other
/// containers) will be added in future slices.
pub struct MuxerBuilder<Writer> {
    /// The underlying writer to which container data will be written.
    writer: Writer,
    /// Optional video configuration.
    video: Option<(VideoCodec, u32, u32, f64)>,
    /// Optional audio configuration.
    audio: Option<(AudioCodec, u32, u16)>,
    /// Optional subtitle configuration.
    subtitle: Option<SubtitleTrackConfig>,
    /// Metadata to embed in the output file.
    metadata: Option<Metadata>,
    /// Whether to enable fast-start (moov before mdat).
    fast_start: bool,
    /// SPS data for fragmented MP4.
    sps: Option<Vec<u8>>,
    /// PPS data for fragmented MP4.
    pps: Option<Vec<u8>>,
    /// VPS data for H.265 fragmented MP4.
    vps: Option<Vec<u8>>,
    /// AV1 sequence header OBU for fragmented MP4.
    av1_sequence_header: Option<Vec<u8>>,
    /// VP9 configuration for fragmented MP4.
    vp9_config: Option<crate::codec::vp9::Vp9Config>,
    /// FLAC STREAMINFO (34 bytes) for FLAC audio tracks.
    flac_streaminfo: Option<Vec<u8>>,
    /// Opus pre-skip override (48 kHz samples) for Opus audio tracks.
    opus_preskip: Option<u16>,
    /// Output container selected for [`MuxerBuilder::build_mkv`].
    container: ContainerFormat,
    /// Caller-supplied video decoder config (verdict §8). When set, the
    /// writer uses it instead of first-keyframe extraction and rejects
    /// mid-stream changes with `DecoderConfigurationChanged`.
    video_decoder_config: Option<VideoDecoderConfig>,
    /// Input bitstream format hint (verdict §8).
    bitstream_format: Option<VideoBitstreamFormat>,
    /// Caller-supplied AAC ASC (required for HE/HEv2, verdict §8).
    aac_config: Option<AacConfig>,
    /// Full Opus dOps config override (verdict §8).
    opus_config: Option<crate::codec::opus::OpusConfig>,
    /// Per-track language overrides (verdict §10).
    video_language: Option<LanguageCode>,
    audio_language: Option<LanguageCode>,
    /// Resource limits (verdict §15).
    limits: Limits,
}

/// Structural validation for a caller-supplied Opus `dOps` config
/// (RFC 7845 §5.1.1). Runs in every `build_*` path so multichannel
/// layouts (family ≥ 1) cannot reach the box builders unchecked.
fn validate_opus_config(config: Option<&crate::codec::opus::OpusConfig>) -> Result<(), MuxerError> {
    if let Some(config) = config {
        config
            .validate()
            .map_err(|e| MuxerError::InvalidOpusConfig {
                reason: e.to_string(),
            })?;
    }
    Ok(())
}

impl<Writer> MuxerBuilder<Writer> {
    /// Create a new builder for the given output writer.
    pub fn new(writer: Writer) -> Self {
        Self {
            writer,
            video: None,
            audio: None,
            subtitle: None,
            metadata: None,
            fast_start: true, // Default ON for web compatibility
            sps: None,
            pps: None,
            vps: None,
            av1_sequence_header: None,
            vp9_config: None,
            flac_streaminfo: None,
            opus_preskip: None,
            container: ContainerFormat::Mp4,
            video_decoder_config: None,
            bitstream_format: None,
            aac_config: None,
            opus_config: None,
            video_language: None,
            audio_language: None,
            limits: Limits::default(),
        }
    }

    /// Configure the video track.
    pub fn video(mut self, codec: VideoCodec, width: u32, height: u32, framerate: f64) -> Self {
        self.video = Some((codec, width, height, framerate));
        self
    }

    /// Configure the audio track.
    pub fn audio(mut self, codec: AudioCodec, sample_rate: u32, channels: u16) -> Self {
        self.audio = Some((codec, sample_rate, channels));
        self
    }

    /// Configure the subtitle track.
    pub fn subtitle(mut self, codec: SubtitleCodec, language: Option<String>) -> Self {
        self.subtitle = Some(SubtitleTrackConfig {
            codec,
            language,
            timescale: 1_000,
        });
        self
    }

    /// Configure subtitle track with explicit timescale (verdict §3).
    pub fn subtitle_with_timescale(
        mut self,
        codec: SubtitleCodec,
        language: Option<String>,
        timescale: u32,
    ) -> Self {
        self.subtitle = Some(SubtitleTrackConfig {
            codec,
            language,
            timescale: timescale.max(1),
        });
        self
    }

    /// Per-track language override validated centrally (verdict §10).
    pub fn with_video_language(mut self, language: &str) -> Result<Self, MuxerError> {
        let _ = LanguageCode::parse(language)?;
        // Stored via metadata for now; per-track wiring lands in build().
        self.metadata.get_or_insert_with(Metadata::default).language = Some(language.to_string());
        Ok(self)
    }

    /// Set metadata to embed in the output file (title, creation time, etc.)
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Enable or disable fast-start mode (moov before mdat).
    /// Default is `true` for web streaming compatibility.
    pub fn with_fast_start(mut self, enabled: bool) -> Self {
        self.fast_start = enabled;
        self
    }

    /// Select the output container for [`MuxerBuilder::build_mkv`].
    ///
    /// `ContainerFormat::Mp4` (the default) is ignored by [`MuxerBuilder::build`],
    /// which always produces MP4. Use [`MuxerBuilder::build_mkv`] to produce
    /// Matroska/WebM: with the default `Mp4` value it yields Matroska, otherwise
    /// the selected Matroska-family container.
    pub fn with_container(mut self, container: ContainerFormat) -> Self {
        self.container = container;
        self
    }

    /// Set SPS (Sequence Parameter Set) data for H.264/H.265 fragmented MP4.
    /// Required for proper fragmented MP4 initialization.
    pub fn with_sps(mut self, sps: Vec<u8>) -> Self {
        self.sps = Some(sps);
        self
    }

    /// Set PPS (Picture Parameter Set) data for H.264/H.265 fragmented MP4.
    /// Required for proper fragmented MP4 initialization.
    pub fn with_pps(mut self, pps: Vec<u8>) -> Self {
        self.pps = Some(pps);
        self
    }

    /// Set VPS (Video Parameter Set) data for H.265 fragmented MP4.
    /// Required for proper H.265 fragmented MP4 initialization.
    pub fn with_vps(mut self, vps: Vec<u8>) -> Self {
        self.vps = Some(vps);
        self
    }

    /// Set AV1 sequence header OBU for fragmented MP4.
    /// Required for proper AV1 fragmented MP4 initialization.
    pub fn with_av1_sequence_header(mut self, sequence_header: Vec<u8>) -> Self {
        self.av1_sequence_header = Some(sequence_header);
        self
    }

    /// Set VP9 configuration for fragmented MP4.
    /// Required for proper VP9 fragmented MP4 initialization.
    pub fn with_vp9_config(mut self, config: crate::codec::vp9::Vp9Config) -> Self {
        self.vp9_config = Some(config);
        self
    }

    /// Supply a complete decoder config record (verdict §8).
    /// When set, extraction helpers become a fallback only.
    pub fn with_video_decoder_config(mut self, config: VideoDecoderConfig) -> Self {
        self.video_decoder_config = Some(config);
        self
    }

    /// Hint the input bitstream format (verdict §8).
    pub fn with_bitstream_format(mut self, format: VideoBitstreamFormat) -> Self {
        self.bitstream_format = Some(format);
        self
    }

    /// Supply AAC AudioSpecificConfig (required for HE/HEv2, verdict §8).
    pub fn with_aac_config(mut self, config: AacConfig) -> Self {
        self.aac_config = Some(config);
        self
    }

    /// Supply full Opus dOps config (verdict §8).
    pub fn with_opus_config(mut self, config: crate::codec::opus::OpusConfig) -> Self {
        self.opus_config = Some(config);
        self
    }

    /// Per-track video language (verdict §10).
    pub fn with_video_language_code(mut self, language: LanguageCode) -> Self {
        self.video_language = Some(language);
        self
    }

    /// Per-track audio language (verdict §10).
    pub fn with_audio_language_code(mut self, language: LanguageCode) -> Self {
        self.audio_language = Some(language);
        self
    }

    /// Override resource limits (verdict §15).
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the FLAC STREAMINFO block (34 raw bytes) for FLAC audio tracks.
    ///
    /// Required when the audio codec is [`AudioCodec::Flac`]: the MP4
    /// `dfLa` box and the Matroska `A_FLAC` CodecPrivate are built from
    /// it. Obtain it from [`crate::demux::FlacStream::streaminfo_raw`]
    /// or any native FLAC file's first metadata block.
    pub fn with_flac_streaminfo(mut self, streaminfo: Vec<u8>) -> Self {
        self.flac_streaminfo = Some(streaminfo);
        self
    }

    /// Override the Opus pre-skip signalled in `dOps`/`OpusHead`.
    ///
    /// Defaults to 312 samples when unset. Set it from
    /// [`crate::demux::OggOpusTrack::pre_skip`] when remuxing Ogg Opus
    /// so players skip exactly the encoder delay.
    pub fn with_opus_preskip(mut self, pre_skip: u16) -> Self {
        self.opus_preskip = Some(pre_skip);
        self
    }

    /// Set creation time for the media file
    pub fn set_create_time(mut self, unix_timestamp: u64) -> Self {
        self.metadata
            .get_or_insert_with(Metadata::default)
            .creation_time = Some(unix_timestamp);
        self
    }

    /// Set language code for the media file
    pub fn set_language(mut self, language: impl Into<String>) -> Self {
        self.metadata.get_or_insert_with(Metadata::default).language = Some(language.into());
        self
    }

    /// Set video track parameters
    pub fn set_video_track(
        mut self,
        codec: VideoCodec,
        width: u32,
        height: u32,
        framerate: f64,
    ) -> Self {
        self.video = Some((codec, width, height, framerate));
        self
    }

    /// Set audio track parameters
    pub fn set_audio_track(mut self, codec: AudioCodec, sample_rate: u32, channels: u16) -> Self {
        self.audio = Some((codec, sample_rate, channels));
        self
    }

    /// Finalise the builder and produce a `Muxer` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if required configuration is missing or invalid.
    pub fn build(self) -> Result<Muxer<Writer>, MuxerError>
    where
        Writer: Write,
    {
        validate_opus_config(self.opus_config.as_ref())?;
        let mut video_track =
            self.video
                .map(|(codec, width, height, framerate)| VideoTrackConfig {
                    codec,
                    width,
                    height,
                    framerate,
                    timescale: 90_000,
                    language: LanguageCode::UND,
                });
        if let (Some(t), Some(lang)) = (video_track.as_mut(), self.video_language) {
            t.language = lang;
        }

        let mut audio_track = self.audio.and_then(|(codec, sample_rate, channels)| {
            if codec == AudioCodec::None {
                None
            } else {
                let timescale = match codec {
                    AudioCodec::Opus => 48_000,
                    _ => sample_rate,
                };
                Some(AudioTrackConfig {
                    codec,
                    sample_rate,
                    channels,
                    timescale,
                    language: LanguageCode::UND,
                })
            }
        });
        if let (Some(t), Some(lang)) = (audio_track.as_mut(), self.audio_language) {
            t.language = lang;
        }

        let subtitle_track = self.subtitle;

        if subtitle_track.is_some() && video_track.is_none() {
            return Err(MuxerError::SubtitleRequiresVideo);
        }

        if video_track.is_none() && audio_track.is_none() && subtitle_track.is_none() {
            return Err(MuxerError::MissingConfig);
        }

        if let Some(audio) = &audio_track
            && audio.codec == AudioCodec::Flac
        {
            let info = self.flac_streaminfo.as_ref().ok_or_else(|| {
                MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "FLAC STREAMINFO must be provided for FLAC audio using with_flac_streaminfo()",
                ))
            })?;
            let parsed = crate::codec::flac::parse_streaminfo(info).ok_or_else(|| {
                MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "FLAC STREAMINFO must be a valid 34-byte STREAMINFO block",
                ))
            })?;
            // The track parameters must agree with STREAMINFO (the MP4
            // sample entry carries the STREAMINFO values).
            if audio.sample_rate != parsed.sample_rate
                || audio.channels != u16::from(parsed.channels)
            {
                return Err(MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "FLAC track parameters ({} Hz, {} ch) disagree with STREAMINFO ({} Hz, {} ch)",
                        audio.sample_rate, audio.channels, parsed.sample_rate, parsed.channels
                    ),
                )));
            }
        }

        let mut writer = Mp4Writer::new(self.writer);
        if let Some(ref video) = video_track {
            if let Some(cfg) = self.video_decoder_config.clone() {
                writer.enable_video_with_config(video.codec, cfg);
            } else {
                writer.enable_video(video.codec);
            }
        }
        if let Some(audio) = &audio_track {
            writer.enable_audio(Mp4AudioTrack {
                sample_rate: audio.sample_rate,
                channels: audio.channels,
                codec: audio.codec,
                flac_streaminfo: self.flac_streaminfo.clone(),
                opus_preskip: self.opus_preskip,
                aac_asc_override: self.aac_config.clone().map(|c| c.audio_specific_config),
                opus_config_override: self.opus_config.clone(),
                language: Some(String::from_utf8_lossy(&audio.language.as_bytes()).into_owned()),
            });
        }
        if let Some(subtitle) = &subtitle_track {
            writer.enable_subtitle(Mp4SubtitleTrack {
                codec: subtitle.codec,
                language: subtitle.language.clone(),
            });
        }

        Ok(Muxer {
            writer,
            video_track,
            audio_track,
            subtitle_track,
            metadata: self.metadata,
            fast_start: self.fast_start,
            limits: self.limits,
            first_video_pts: None,
            last_video_pts: None,
            last_video_dts: None,
            last_audio_pts: None,
            last_subtitle_pts: None,
            video_frame_count: 0,
            audio_frame_count: 0,
            subtitle_frame_count: 0,
            finished: false,
            current_video_pts: 0.0,
            current_audio_pts: 0.0,
        })
    }

    /// Create a seekable-streaming MP4 muxer (§5).
    ///
    /// Unlike [`MuxerBuilder::build`], sample bytes are written to the
    /// builder's writer as they arrive and only per-sample metadata is
    /// retained, so peak RAM is O(metadata) instead of O(media). The writer
    /// must implement [`std::io::Seek`] (files, `Cursor<Vec<u8>>`) because
    /// `finish` seeks back to patch the `mdat` size before appending `moov`.
    ///
    /// The output layout is `ftyp`, `mdat`, `moov` (progressive). For
    /// fast-start `moov`-before-`mdat`, use [`MuxerBuilder::build`].
    /// Validation, bitstream conversion, and timestamp rules are identical
    /// to [`Muxer`]; only the buffering strategy differs.
    pub fn build_streaming_seekable(self) -> Result<StreamingMuxer<Writer>, MuxerError>
    where
        Writer: Write + std::io::Seek,
    {
        validate_opus_config(self.opus_config.as_ref())?;
        let mut video_track =
            self.video
                .map(|(codec, width, height, framerate)| VideoTrackConfig {
                    codec,
                    width,
                    height,
                    framerate,
                    timescale: 90_000,
                    language: LanguageCode::UND,
                });
        if let (Some(t), Some(lang)) = (video_track.as_mut(), self.video_language) {
            t.language = lang;
        }

        let mut audio_track = self.audio.and_then(|(codec, sample_rate, channels)| {
            if codec == AudioCodec::None {
                None
            } else {
                let timescale = match codec {
                    AudioCodec::Opus => 48_000,
                    _ => sample_rate,
                };
                Some(AudioTrackConfig {
                    codec,
                    sample_rate,
                    channels,
                    timescale,
                    language: LanguageCode::UND,
                })
            }
        });
        if let (Some(t), Some(lang)) = (audio_track.as_mut(), self.audio_language) {
            t.language = lang;
        }

        let subtitle_track = self.subtitle;

        if subtitle_track.is_some() && video_track.is_none() {
            return Err(MuxerError::SubtitleRequiresVideo);
        }

        if video_track.is_none() && audio_track.is_none() && subtitle_track.is_none() {
            return Err(MuxerError::MissingConfig);
        }

        if let Some(audio) = &audio_track
            && audio.codec == AudioCodec::Flac
        {
            let info = self.flac_streaminfo.as_ref().ok_or_else(|| {
                MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "FLAC STREAMINFO must be provided for FLAC audio using with_flac_streaminfo()",
                ))
            })?;
            let parsed = crate::codec::flac::parse_streaminfo(info).ok_or_else(|| {
                MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "FLAC STREAMINFO must be a valid 34-byte STREAMINFO block",
                ))
            })?;
            if audio.sample_rate != parsed.sample_rate
                || audio.channels != u16::from(parsed.channels)
            {
                return Err(MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "FLAC track parameters ({} Hz, {} ch) disagree with STREAMINFO ({} Hz, {} ch)",
                        audio.sample_rate, audio.channels, parsed.sample_rate, parsed.channels
                    ),
                )));
            }
        }

        let mut state: Mp4Writer<std::io::Sink> = Mp4Writer::new(std::io::sink());
        if let Some(ref video) = video_track {
            if let Some(cfg) = self.video_decoder_config.clone() {
                state.enable_video_with_config(video.codec, cfg);
            } else {
                state.enable_video(video.codec);
            }
        }
        if let Some(audio) = &audio_track {
            state.enable_audio(Mp4AudioTrack {
                sample_rate: audio.sample_rate,
                channels: audio.channels,
                codec: audio.codec,
                flac_streaminfo: self.flac_streaminfo.clone(),
                opus_preskip: self.opus_preskip,
                aac_asc_override: self.aac_config.clone().map(|c| c.audio_specific_config),
                opus_config_override: self.opus_config.clone(),
                language: Some(String::from_utf8_lossy(&audio.language.as_bytes()).into_owned()),
            });
        }
        if let Some(subtitle) = &subtitle_track {
            state.enable_subtitle(Mp4SubtitleTrack {
                codec: subtitle.codec,
                language: subtitle.language.clone(),
            });
        }

        let video_params = video_track.as_ref().map(|t| Mp4VideoTrack {
            width: t.width,
            height: t.height,
            language: Some(String::from_utf8_lossy(&t.language.as_bytes()).into_owned()),
        });
        let video_codec = video_track.as_ref().map(|t| t.codec);
        let inner =
            SeekableStreamingWriter::begin(self.writer, state, video_params, self.metadata.clone())
                .map_err(MuxerError::Io)?;

        Ok(StreamingMuxer {
            inner,
            video_track,
            audio_track,
            subtitle_track,
            video_codec,
            limits: self.limits,
            first_video_pts: None,
            last_video_pts: None,
            last_video_dts: None,
            last_audio_pts: None,
            last_subtitle_pts: None,
            video_frame_count: 0,
            audio_frame_count: 0,
            subtitle_frame_count: 0,
            video_end_ticks: None,
            audio_end_ticks: None,
            subtitle_end_ticks: None,
            finished: false,
        })
    }

    /// Create a fragmented MP4 muxer.
    ///
    /// This creates a `FragmentedMuxer` with the configuration from this builder.
    /// Supports H.264, H.265, AV1, and VP9 video for fragmented MP4.
    /// Codec-specific parameters must be provided using the appropriate with_() methods.
    /// Only video configuration is supported for fragmented MP4.
    ///
    /// # Errors
    ///
    /// Returns an error if video configuration is missing, unsupported codec,
    /// or required codec parameters are not provided.
    pub fn new_with_fragment(self) -> Result<FragmentedMuxer, MuxerError> {
        // Fragmented MP4 requires video configuration
        let (codec, width, height, _framerate) = self.video.ok_or(MuxerError::MissingConfig)?;

        // Extract codec-specific configuration
        let (sps, pps, vps, av1_sequence_header, vp9_config) = match codec {
            VideoCodec::H264 => {
                let sps = self.sps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "SPS must be provided for H.264 fragmented MP4 using with_sps()",
                    ))
                })?;
                let pps = self.pps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "PPS must be provided for H.264 fragmented MP4 using with_pps()",
                    ))
                })?;
                (sps, pps, None, None, None)
            }
            VideoCodec::H265 => {
                let vps = self.vps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "VPS must be provided for H.265 fragmented MP4 using with_vps()",
                    ))
                })?;
                let sps = self.sps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "SPS must be provided for H.265 fragmented MP4 using with_sps()",
                    ))
                })?;
                let pps = self.pps.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "PPS must be provided for H.265 fragmented MP4 using with_pps()",
                    ))
                })?;
                (sps, pps, Some(vps), None, None)
            }
            VideoCodec::Av1 => {
                let av1_sequence_header = self.av1_sequence_header.ok_or_else(|| MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "AV1 sequence header must be provided for AV1 fragmented MP4 using with_av1_sequence_header()",
                )))?;
                (vec![], vec![], None, Some(av1_sequence_header), None)
            }
            VideoCodec::Vp9 => {
                let vp9_config = self.vp9_config.ok_or_else(|| {
                    MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "VP9 config must be provided for VP9 fragmented MP4 using with_vp9_config()",
                ))
                })?;
                (vec![], vec![], None, None, Some(vp9_config))
            }
        };

        let config = FragmentConfig {
            width,
            height,
            timescale: 90000,           // Standard video timescale
            fragment_duration_ms: 2000, // 2 second fragments
            boundary: SegmentBoundary::Duration {
                target: 180_000, // 2000 ms at 90 kHz
                require_sync_sample: true,
            },
            sps,
            pps,
            vps,
            av1_sequence_header,
            vp9_config,
        };

        Ok(FragmentedMuxer::new(config))
    }

    /// Finalise the builder and produce an [`MkvMuxer`] instance.
    ///
    /// The container comes from [`MuxerBuilder::with_container`]: `Matroska`
    /// or `WebM` are honoured, while the default `Mp4` yields Matroska.
    /// Track validation mirrors [`MuxerBuilder::build`], including
    /// "subtitle requires video".
    ///
    /// ```no_run
    /// use muxfin::api::{ContainerFormat, MuxerBuilder, VideoCodec};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut muxer = MuxerBuilder::new(Vec::<u8>::new())
    ///     .video(VideoCodec::Vp9, 1920, 1080, 30.0)
    ///     .with_container(ContainerFormat::WebM)
    ///     .build_mkv()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if required configuration is missing or invalid.
    pub fn build_mkv(self) -> Result<MkvMuxer<Writer>, MuxerError>
    where
        Writer: Write,
    {
        validate_opus_config(self.opus_config.as_ref())?;
        let container = match self.container {
            ContainerFormat::Mp4 | ContainerFormat::Matroska => MkvContainer::Matroska,
            ContainerFormat::WebM => MkvContainer::WebM,
        };

        let mut video_track =
            self.video
                .map(|(codec, width, height, framerate)| VideoTrackConfig {
                    codec,
                    width,
                    height,
                    framerate,
                    timescale: 90_000,
                    language: LanguageCode::UND,
                });
        if let (Some(t), Some(lang)) = (video_track.as_mut(), self.video_language) {
            t.language = lang;
        }

        let mut audio_track = self.audio.and_then(|(codec, sample_rate, channels)| {
            if codec == AudioCodec::None {
                None
            } else {
                let timescale = match codec {
                    AudioCodec::Opus => 48_000,
                    _ => sample_rate,
                };
                Some(AudioTrackConfig {
                    codec,
                    sample_rate,
                    channels,
                    timescale,
                    language: LanguageCode::UND,
                })
            }
        });
        if let (Some(t), Some(lang)) = (audio_track.as_mut(), self.audio_language) {
            t.language = lang;
        }

        let subtitle_track = self.subtitle;

        if subtitle_track.is_some() && video_track.is_none() {
            return Err(MuxerError::SubtitleRequiresVideo);
        }

        if video_track.is_none() && audio_track.is_none() && subtitle_track.is_none() {
            return Err(MuxerError::MissingConfig);
        }

        if let Some(audio) = &audio_track
            && audio.codec == AudioCodec::Flac
        {
            let parsed = crate::codec::flac::parse_streaminfo(
                self.flac_streaminfo.as_deref().unwrap_or(&[]),
            )
            .ok_or_else(|| {
                MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "valid FLAC STREAMINFO (34 bytes) must be provided for FLAC audio using with_flac_streaminfo()",
                ))
            })?;
            if audio.sample_rate != parsed.sample_rate
                || audio.channels != u16::from(parsed.channels)
            {
                return Err(MuxerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "FLAC track parameters ({} Hz, {} ch) disagree with STREAMINFO ({} Hz, {} ch)",
                        audio.sample_rate, audio.channels, parsed.sample_rate, parsed.channels
                    ),
                )));
            }
        }

        if container == MkvContainer::WebM
            && let Some(audio) = &audio_track
            && audio.codec == AudioCodec::Flac
        {
            return Err(MuxerError::UnsupportedForContainer {
                codec: audio.codec.to_string(),
                container: container.to_string(),
                reason: "WebM only supports Opus audio; use Matroska (.mkv) for FLAC".to_string(),
            });
        }

        let mut writer = MkvWriter::new(self.writer);
        if let Some(ref video) = video_track {
            writer.enable_video(video.codec);
        }
        if let Some(audio) = &audio_track {
            writer.enable_audio(Mp4AudioTrack {
                sample_rate: audio.sample_rate,
                channels: audio.channels,
                codec: audio.codec,
                flac_streaminfo: self.flac_streaminfo.clone(),
                opus_preskip: self.opus_preskip,
                aac_asc_override: self.aac_config.clone().map(|c| c.audio_specific_config),
                opus_config_override: self.opus_config.clone(),
                language: Some(String::from_utf8_lossy(&audio.language.as_bytes()).into_owned()),
            });
        }
        if let Some(subtitle) = &subtitle_track {
            writer.enable_subtitle(Mp4SubtitleTrack {
                codec: subtitle.codec,
                language: subtitle.language.clone(),
            });
        }

        Ok(MkvMuxer {
            writer,
            video_track,
            audio_track,
            subtitle_track,
            metadata: self.metadata,
            limits: self.limits,
            container,
            first_video_pts: None,
            last_video_pts: None,
            last_video_dts: None,
            last_audio_pts: None,
            last_subtitle_pts: None,
            video_frame_count: 0,
            audio_frame_count: 0,
            subtitle_frame_count: 0,
            finished: false,
            current_video_pts: 0.0,
            current_audio_pts: 0.0,
        })
    }
}

/// Configuration for a video track.
#[derive(Debug, Clone)]
pub struct VideoTrackConfig {
    /// Video codec.
    pub codec: VideoCodec,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate (frames per second). Convenience only; timescale is source of truth.
    pub framerate: f64,
    /// Track timescale (default 90_000). Caller-overridable (verdict §3).
    pub timescale: u32,
    /// ISO-639-2/T language (default `und`). Per-track (verdict §10).
    pub language: LanguageCode,
}

/// Configuration for an audio track.
#[derive(Debug, Clone)]
pub struct AudioTrackConfig {
    /// Audio codec.
    pub codec: AudioCodec,
    /// Sample rate (Hz).
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// Track timescale (default: sample rate, 48_000 for Opus). (verdict §3).
    pub timescale: u32,
    /// ISO-639-2/T language (default `und`). Per-track (verdict §10).
    pub language: LanguageCode,
}

/// Configuration for a subtitle track.
#[derive(Debug, Clone)]
pub struct SubtitleTrackConfig {
    /// Subtitle codec.
    pub codec: SubtitleCodec,
    /// Language code (ISO 639-2/T), e.g., "eng".
    pub language: Option<String>,
    /// Track timescale (default 1_000). Caller-overridable (verdict §3).
    pub timescale: u32,
}

/// Opaque muxer type.  Users interact with this type to write frames
/// into the container.  Implementation details are hidden in a private
/// module.
///
/// # Thread Safety
///
/// `Muxer<W>` is `Send` when `W: Send` and `Sync` when `W: Sync`.
/// This means you can safely move a `Muxer<File>` between threads or
/// share a `Muxer<Vec<u8>>` across threads (with appropriate synchronization).
pub struct Muxer<Writer> {
    writer: Mp4Writer<Writer>,
    video_track: Option<VideoTrackConfig>,
    audio_track: Option<AudioTrackConfig>,
    subtitle_track: Option<SubtitleTrackConfig>,
    metadata: Option<Metadata>,
    fast_start: bool,
    first_video_pts: Option<f64>,
    last_video_pts: Option<f64>,
    last_video_dts: Option<f64>,
    last_audio_pts: Option<f64>,
    last_subtitle_pts: Option<f64>,
    video_frame_count: u64,
    audio_frame_count: u64,
    subtitle_frame_count: u64,
    finished: bool,
    current_video_pts: f64,
    current_audio_pts: f64,
    limits: Limits,
}

/// Error type for builder validation and runtime errors.
///
/// All errors include context to help diagnose issues. Error messages are designed
/// to be educational—they explain what went wrong and how to fix it.
#[derive(Debug)]
#[non_exhaustive]
pub enum MuxerError {
    /// Neither video nor audio configuration was provided.
    MissingConfig,
    /// Low-level IO error while writing the container.
    Io(std::io::Error),
    /// The muxer has already been finished.
    AlreadyFinished,
    /// Video `pts` must be non-negative.
    NegativeVideoPts { pts: f64, frame_index: u64 },
    /// Video `dts` must be non-negative.
    NegativeVideoDts { dts: f64, frame_index: u64 },
    /// Video `pts` must be finite (not NaN or Inf).
    InvalidVideoPts { pts: f64, frame_index: u64 },
    /// Video `dts` must be finite (not NaN or Inf).
    InvalidVideoDts { dts: f64, frame_index: u64 },
    /// Audio `pts` must be non-negative.
    NegativeAudioPts { pts: f64, frame_index: u64 },
    /// Audio `pts` must be finite (not NaN or Inf).
    InvalidAudioPts { pts: f64, frame_index: u64 },
    /// Audio was written but no audio track was configured.
    AudioNotConfigured,
    /// Audio sample is empty.
    EmptyAudioFrame { frame_index: u64 },
    /// Subtitle sample is empty.
    EmptySubtitleSample { frame_index: u64 },
    /// Video sample is empty.
    EmptyVideoFrame { frame_index: u64 },
    /// Video timestamps must be strictly increasing.
    NonIncreasingVideoPts {
        prev_pts: f64,
        curr_pts: f64,
        frame_index: u64,
    },
    /// Audio timestamps must be non-decreasing.
    DecreasingAudioPts {
        prev_pts: f64,
        curr_pts: f64,
        frame_index: u64,
    },
    /// Audio may not precede the first video frame.
    AudioBeforeFirstVideo {
        audio_pts: f64,
        first_video_pts: Option<f64>,
    },
    /// Subtitle `pts` must be non-negative.
    NegativeSubtitlePts { pts: f64, frame_index: u64 },
    /// Subtitle `pts` must be finite.
    InvalidSubtitlePts { pts: f64, frame_index: u64 },
    /// Subtitle duration must be positive and finite.
    InvalidSubtitleDuration {
        duration_secs: f64,
        frame_index: u64,
    },
    /// Subtitle track is not configured.
    SubtitleNotConfigured,
    /// Subtitle timestamps must be non-decreasing.
    DecreasingSubtitlePts {
        prev_pts: f64,
        curr_pts: f64,
        frame_index: u64,
    },
    /// Subtitle tracks require a video track for MP4 compatibility.
    SubtitleRequiresVideo,
    /// The first video frame must be a keyframe.
    FirstVideoFrameMustBeKeyframe,
    /// The first video frame must include SPS/PPS (H.264/H.265).
    FirstVideoFrameMissingSpsPps,
    /// The first AV1 keyframe must include a Sequence Header OBU.
    FirstAv1FrameMissingSequenceHeader,
    /// The first VP9 keyframe must include sequence parameters.
    FirstVp9FrameMissingSequenceHeader,
    /// Audio sample is not a valid ADTS frame.
    InvalidAdts { frame_index: u64 },
    /// Audio sample has detailed ADTS validation errors.
    InvalidAdtsDetailed {
        frame_index: u64,
        error: Box<crate::muxer::mp4::AdtsValidationError>,
    },
    /// Audio sample is not a valid Opus packet.
    InvalidOpusPacket { frame_index: u64 },
    /// Caller-supplied Opus `dOps` config failed structural validation.
    InvalidOpusConfig { reason: String },
    /// Audio sample is not a valid FLAC frame.
    InvalidFlacFrame { frame_index: u64 },
    /// DTS must be monotonically increasing.
    NonIncreasingDts {
        prev_dts: f64,
        curr_dts: f64,
        frame_index: u64,
    },
    /// Codec is not allowed in the selected container.
    UnsupportedForContainer {
        /// Human-readable codec description.
        codec: String,
        /// Human-readable container name.
        container: String,
        /// Why it is rejected and what to use instead.
        reason: String,
    },
    /// Sample duration must be greater than zero (verdict §4).
    ZeroDuration,
    /// Timestamp arithmetic overflow (verdict §3).
    TimestampOverflow,
    /// Integer DTS must increase: previous >= current (verdict §7).
    NonIncreasingIntDts { previous: i64, current: i64 },
    /// Composition offset outside signed 32-bit range (verdict §7).
    CompositionOffsetOutOfRange { pts: i64, dts: i64 },
    /// Box field too large for its width (verdict §6).
    FieldTooLarge { field: &'static str, value: u64 },
    /// Container size overflow (64-bit mdat path).
    SizeOverflow,
    /// Decoder configuration changed after track start (verdict §8).
    DecoderConfigurationChanged,
    /// Subtitle sample too large (verdict §9).
    SubtitleTooLarge(usize),
    /// Invalid ISO-639 language code (verdict §10).
    InvalidLanguage,
    /// Subtitle DTS must be non-negative (integer API).
    NegativeSubtitleDts { dts: i64 },
    /// Sample exceeds configured resource limit (verdict §15).
    ResourceLimitExceeded { what: &'static str },
}

impl From<std::io::Error> for MuxerError {
    fn from(err: std::io::Error) -> Self {
        MuxerError::Io(err)
    }
}

impl fmt::Display for MuxerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MuxerError::MissingConfig => {
                write!(
                    f,
                    "missing configuration: call .video(), .audio(), or .subtitle() on MuxerBuilder before .build()"
                )
            }
            MuxerError::SubtitleRequiresVideo => {
                write!(
                    f,
                    "subtitle track requires video track: call .video() before .subtitle()"
                )
            }
            MuxerError::Io(err) => write!(f, "IO error: {}", err),
            MuxerError::AlreadyFinished => {
                write!(
                    f,
                    "muxer already finished: cannot write frames after calling finish()"
                )
            }
            MuxerError::NegativeVideoPts { pts, frame_index } => {
                write!(
                    f,
                    "video frame {} has negative PTS ({:.3}s): timestamps must be >= 0.0",
                    frame_index, pts
                )
            }
            MuxerError::InvalidVideoPts { pts, frame_index } => {
                write!(
                    f,
                    "video frame {} has invalid PTS ({:.3}s): timestamps must be finite (not NaN or Inf)",
                    frame_index, pts
                )
            }
            MuxerError::NegativeVideoDts { dts, frame_index } => {
                write!(
                    f,
                    "video frame {} has negative DTS ({:.3}s): decode timestamps must be >= 0.0",
                    frame_index, dts
                )
            }
            MuxerError::InvalidVideoDts { dts, frame_index } => {
                write!(
                    f,
                    "video frame {} has invalid DTS ({:.3}s): decode timestamps must be finite (not NaN or Inf)",
                    frame_index, dts
                )
            }
            MuxerError::NegativeAudioPts { pts, frame_index } => {
                write!(
                    f,
                    "audio frame {} has negative PTS ({:.3}s): timestamps must be >= 0.0",
                    frame_index, pts
                )
            }
            MuxerError::InvalidAudioPts { pts, frame_index } => {
                write!(
                    f,
                    "audio frame {} has invalid PTS ({:.3}s): timestamps must be finite (not NaN or Inf)",
                    frame_index, pts
                )
            }
            MuxerError::AudioNotConfigured => {
                write!(
                    f,
                    "audio track not configured: call .audio() on MuxerBuilder to enable audio"
                )
            }
            MuxerError::EmptyAudioFrame { frame_index } => {
                write!(
                    f,
                    "audio frame {} is empty: ADTS frames must contain data",
                    frame_index
                )
            }
            MuxerError::EmptySubtitleSample { frame_index } => {
                write!(f, "subtitle sample {} is empty", frame_index)
            }
            MuxerError::EmptyVideoFrame { frame_index } => {
                write!(
                    f,
                    "video frame {} is empty: video samples must contain NAL units",
                    frame_index
                )
            }
            MuxerError::NonIncreasingVideoPts {
                prev_pts,
                curr_pts,
                frame_index,
            } => {
                write!(
                    f,
                    "video frame {} has PTS {:.3}s which is not greater than previous PTS {:.3}s: \
                          video timestamps must strictly increase. For B-frames, use write_video_with_dts()",
                    frame_index, curr_pts, prev_pts
                )
            }
            MuxerError::DecreasingAudioPts {
                prev_pts,
                curr_pts,
                frame_index,
            } => {
                write!(
                    f,
                    "audio frame {} has PTS {:.3}s which is less than previous PTS {:.3}s: \
                          audio timestamps must not decrease",
                    frame_index, curr_pts, prev_pts
                )
            }
            MuxerError::AudioBeforeFirstVideo {
                audio_pts,
                first_video_pts,
            } => match first_video_pts {
                Some(v) => write!(
                    f,
                    "audio PTS {:.3}s arrives before first video PTS {:.3}s: \
                                         write video frames first, or ensure audio PTS >= video PTS",
                    audio_pts, v
                ),
                None => write!(
                    f,
                    "audio frame arrived before any video frame: \
                                       write at least one video frame before writing audio"
                ),
            },
            MuxerError::NegativeSubtitlePts { pts, frame_index } => {
                write!(
                    f,
                    "subtitle sample {} has negative PTS ({:.3}s)",
                    frame_index, pts
                )
            }
            MuxerError::InvalidSubtitlePts { pts, frame_index } => {
                write!(
                    f,
                    "subtitle sample {} has invalid PTS ({:.3}s): timestamps must be finite",
                    frame_index, pts
                )
            }
            MuxerError::InvalidSubtitleDuration {
                duration_secs,
                frame_index,
            } => {
                write!(
                    f,
                    "subtitle sample {} has invalid duration ({:.3}s): duration must be positive and finite",
                    frame_index, duration_secs
                )
            }
            MuxerError::SubtitleNotConfigured => {
                write!(
                    f,
                    "subtitle track not configured: call .subtitle() on MuxerBuilder to enable subtitles"
                )
            }
            MuxerError::DecreasingSubtitlePts {
                prev_pts,
                curr_pts,
                frame_index,
            } => {
                write!(
                    f,
                    "subtitle sample {} has PTS {:.3}s which is less than previous PTS {:.3}s: subtitle timestamps must not decrease",
                    frame_index, curr_pts, prev_pts
                )
            }
            MuxerError::FirstVideoFrameMustBeKeyframe => {
                write!(
                    f,
                    "first video frame must be a keyframe (IDR): \
                          set is_keyframe=true and ensure the frame contains an IDR NAL unit"
                )
            }
            MuxerError::FirstVideoFrameMissingSpsPps => {
                write!(
                    f,
                    "first video frame must contain SPS and PPS NAL units: \
                          prepend SPS (NAL type 7) and PPS (NAL type 8) to the first keyframe"
                )
            }
            MuxerError::FirstAv1FrameMissingSequenceHeader => {
                write!(
                    f,
                    "first AV1 frame must contain a Sequence Header OBU: \
                          ensure the first keyframe includes OBU type 1 (SEQUENCE_HEADER)"
                )
            }
            MuxerError::FirstVp9FrameMissingSequenceHeader => {
                write!(
                    f,
                    "first VP9 frame must contain sequence parameters: \
                          ensure the first keyframe includes VP9 frame header with configuration data"
                )
            }
            MuxerError::InvalidAdts { frame_index } => {
                write!(
                    f,
                    "audio frame {} is not valid ADTS: ensure the frame starts with 0xFFF sync word",
                    frame_index
                )
            }
            MuxerError::InvalidAdtsDetailed { frame_index, error } => {
                write!(
                    f,
                    "audio frame {} ADTS validation failed: {}",
                    frame_index, error
                )
            }
            MuxerError::InvalidOpusPacket { frame_index } => {
                write!(
                    f,
                    "audio frame {} is not a valid Opus packet: ensure the frame has valid TOC byte",
                    frame_index
                )
            }
            MuxerError::InvalidOpusConfig { reason } => {
                write!(f, "invalid Opus dOps config: {}", reason)
            }
            MuxerError::InvalidFlacFrame { frame_index } => {
                write!(
                    f,
                    "audio frame {} is not a valid FLAC frame: ensure the frame starts with the 0xFFF8/0xFFF9 sync",
                    frame_index
                )
            }
            MuxerError::NonIncreasingDts {
                prev_dts,
                curr_dts,
                frame_index,
            } => {
                write!(
                    f,
                    "video frame {} has DTS {:.3}s which is not greater than previous DTS {:.3}s: \
                          DTS (decode timestamps) must strictly increase",
                    frame_index, curr_dts, prev_dts
                )
            }
            MuxerError::UnsupportedForContainer {
                codec,
                container,
                reason,
            } => {
                write!(
                    f,
                    "codec {} is not supported in {}: {}",
                    codec, container, reason
                )
            }
            MuxerError::ZeroDuration => {
                write!(f, "sample duration must be greater than zero")
            }
            MuxerError::TimestampOverflow => write!(f, "timestamp arithmetic overflow"),
            MuxerError::NonIncreasingIntDts { previous, current } => write!(
                f,
                "DTS must increase: previous={}, current={}",
                previous, current
            ),
            MuxerError::CompositionOffsetOutOfRange { pts, dts } => write!(
                f,
                "composition offset is outside signed 32-bit range: pts={}, dts={}",
                pts, dts
            ),
            MuxerError::FieldTooLarge { field, value } => {
                write!(f, "field {} is too large: {}", field, value)
            }
            MuxerError::SizeOverflow => write!(f, "container size overflow"),
            MuxerError::DecoderConfigurationChanged => {
                write!(f, "decoder configuration changed after the track started")
            }
            MuxerError::SubtitleTooLarge(n) => {
                write!(f, "subtitle sample is too large: {} bytes", n)
            }
            MuxerError::InvalidLanguage => write!(f, "invalid ISO-639 language code"),
            MuxerError::NegativeSubtitleDts { dts } => {
                write!(f, "subtitle DTS must be >= 0 (got {})", dts)
            }
            MuxerError::ResourceLimitExceeded { what } => {
                write!(f, "resource limit exceeded: {}", what)
            }
        }
    }
}

impl std::error::Error for MuxerError {}
impl<Writer: Write> Muxer<Writer> {
    /// Write a video frame to the container.
    ///
    /// `pts` is the presentation timestamp in seconds.  Frames must
    /// be supplied in strictly increasing PTS order.  The `data` slice
    /// contains the encoded frame bitstream in Annex B format (for H.264).
    ///
    /// For streams with B-frames (where PTS != DTS), use `write_video_with_dts()` instead.
    pub fn write_video(
        &mut self,
        pts: f64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MuxerError> {
        let frame_index = self.video_frame_count;

        // Reject empty frames - they cause playback issues
        if data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }

        // Validate PTS is finite (not NaN or Inf)
        if !pts.is_finite() {
            return Err(MuxerError::InvalidVideoPts { pts, frame_index });
        }

        // Validate PTS is non-negative
        if pts < 0.0 {
            return Err(MuxerError::NegativeVideoPts { pts, frame_index });
        }

        // Validate PTS is strictly increasing
        if let Some(prev) = self.last_video_pts {
            if pts <= prev {
                return Err(MuxerError::NonIncreasingVideoPts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;

        if self.first_video_pts.is_none() {
            self.first_video_pts = Some(pts);
        }

        self.writer
            .write_video_sample(pts_units, data, is_keyframe)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;

        self.last_video_pts = Some(pts);
        self.video_frame_count += 1;
        Ok(())
    }

    /// Write a video frame with explicit decode timestamp for B-frame support.
    ///
    /// - `pts` is the presentation timestamp in seconds (display order)
    /// - `dts` is the decode timestamp in seconds (decode order)
    ///
    /// For streams with B-frames, PTS and DTS may differ. The only constraint is that
    /// DTS must be strictly monotonically increasing (frames must be fed in decode order).
    ///
    /// Example GOP: I P B B where decode order is I,P,B,B but display order is I,B,B,P
    /// - I: DTS=0, PTS=0
    /// - P: DTS=1, PTS=3 (decoded second, displayed fourth)
    /// - B: DTS=2, PTS=1 (decoded third, displayed second)
    /// - B: DTS=3, PTS=2 (decoded fourth, displayed third)
    pub fn write_video_with_dts(
        &mut self,
        pts: f64,
        dts: f64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }

        let frame_index = self.video_frame_count;

        // Reject empty frames - they cause playback issues
        if data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }

        // Validate PTS is finite (not NaN or Inf)
        if !pts.is_finite() {
            return Err(MuxerError::InvalidVideoPts { pts, frame_index });
        }

        // Validate PTS is non-negative
        if pts < 0.0 {
            return Err(MuxerError::NegativeVideoPts { pts, frame_index });
        }

        // Validate DTS is finite (not NaN or Inf)
        if !dts.is_finite() {
            return Err(MuxerError::InvalidVideoDts { dts, frame_index });
        }

        // Validate DTS is non-negative
        if dts < 0.0 {
            return Err(MuxerError::NegativeVideoDts { dts, frame_index });
        }

        // Note: PTS can be less than DTS for B-frames (displayed before their decode position)
        // This is valid and expected for B-frame streams.

        // Validate DTS is strictly increasing
        if let Some(prev_dts) = self.last_video_dts {
            if dts <= prev_dts {
                return Err(MuxerError::NonIncreasingDts {
                    prev_dts,
                    curr_dts: dts,
                    frame_index,
                });
            }
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;
        let scaled_dts = (dts * MEDIA_TIMESCALE as f64).round();
        let dts_units = scaled_dts as u64;

        if self.first_video_pts.is_none() {
            self.first_video_pts = Some(pts);
        }

        self.writer
            .write_video_sample_with_dts(pts_units, dts_units, data, is_keyframe)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;

        self.last_video_pts = Some(pts);
        self.last_video_dts = Some(dts);
        self.video_frame_count += 1;
        Ok(())
    }

    /// Write a video sample with integer timestamps and explicit duration.
    ///
    /// This is the preferred API (verdict §§3-4): `sample.timing` is in the
    /// video track timescale (default 90_000), `duration` must be > 0, DTS
    /// must strictly increase, and `pts - dts` must fit in `i32` for `ctts`.
    /// Old `f64` methods remain as compatibility shims.
    pub fn write_video_sample(&mut self, sample: EncodedSample<'_>) -> Result<(), MuxerError> {
        use crate::time::{Timescale, rescale};
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let frame_index = self.video_frame_count;
        if sample.data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }
        if sample.timing.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if sample.timing.pts < 0 || sample.timing.dts < 0 {
            return Err(MuxerError::NegativeVideoDts {
                dts: sample.timing.dts as f64,
                frame_index,
            });
        }
        // Composition offset must fit i32 (ctts).
        let _ = sample.timing.composition_offset()?;
        if sample.data.len() > self.limits.max_sample_size {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "video sample",
            });
        }
        let track_ts = self
            .video_track
            .as_ref()
            .map(|t| t.timescale)
            .unwrap_or(90_000);
        let from = Timescale::new(core::num::NonZeroU32::new(track_ts.max(1)).unwrap());
        let to = Timescale::new(core::num::NonZeroU32::new(MEDIA_TIMESCALE).unwrap());
        let pts_i64 = rescale(sample.timing.pts, from, to)?;
        let dts_i64 = rescale(sample.timing.dts, from, to)?;
        let dur_i64 = rescale(i64::from(sample.timing.duration), from, to)?;
        if dur_i64 <= 0 || dur_i64 > i64::from(u32::MAX) {
            return Err(MuxerError::ZeroDuration);
        }
        let pts_u = u64::try_from(pts_i64).map_err(|_| MuxerError::TimestampOverflow)?;
        let dts_u = u64::try_from(dts_i64).map_err(|_| MuxerError::TimestampOverflow)?;
        // Strict DTS increase in media timescale.
        if let Some(prev_f) = self.last_video_dts {
            let prev_u = (prev_f * MEDIA_TIMESCALE as f64).round() as u64;
            if dts_u <= prev_u {
                return Err(MuxerError::NonIncreasingIntDts {
                    previous: prev_u as i64,
                    current: dts_u as i64,
                });
            }
        }
        self.writer
            .write_video_sample_with_dts(pts_u, dts_u, sample.data, sample.is_sync)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;
        self.writer.set_last_video_duration(dur_i64 as u32);
        // Keep f64 mirrors for compat shims.
        let pts_f = pts_u as f64 / MEDIA_TIMESCALE as f64;
        let dts_f = dts_u as f64 / MEDIA_TIMESCALE as f64;
        if self.first_video_pts.is_none() {
            self.first_video_pts = Some(pts_f);
        }
        self.last_video_pts = Some(pts_f);
        self.last_video_dts = Some(dts_f);
        self.video_frame_count += 1;
        Ok(())
    }

    /// Write an audio sample with integer timestamps and explicit duration.
    pub fn write_audio_sample(&mut self, sample: EncodedSample<'_>) -> Result<(), MuxerError> {
        use crate::time::{Timescale, rescale};
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }
        let frame_index = self.audio_frame_count;
        if sample.data.is_empty() {
            return Err(MuxerError::EmptyAudioFrame { frame_index });
        }
        if sample.timing.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if sample.timing.pts < 0 {
            return Err(MuxerError::NegativeAudioPts {
                pts: sample.timing.pts as f64,
                frame_index,
            });
        }
        if sample.data.len() > self.limits.max_sample_size {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "audio sample",
            });
        }
        let track_ts = self
            .audio_track
            .as_ref()
            .map(|t| t.timescale)
            .unwrap_or(48_000);
        let from = Timescale::new(core::num::NonZeroU32::new(track_ts.max(1)).unwrap());
        let to = Timescale::new(core::num::NonZeroU32::new(MEDIA_TIMESCALE).unwrap());
        let pts_i64 = rescale(sample.timing.pts, from, to)?;
        let dur_i64 = rescale(i64::from(sample.timing.duration), from, to)?;
        if dur_i64 <= 0 || dur_i64 > i64::from(u32::MAX) {
            return Err(MuxerError::ZeroDuration);
        }
        let pts_u = u64::try_from(pts_i64).map_err(|_| MuxerError::TimestampOverflow)?;
        // NOTE: verdict §7 permits audio to begin before video; do NOT reject
        // audio-before-video here. Edit lists handle the offset at finalize.
        self.writer
            .write_audio_sample(pts_u, sample.data)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;
        self.writer.set_last_audio_duration(dur_i64 as u32);
        let pts_f = pts_u as f64 / MEDIA_TIMESCALE as f64;
        self.last_audio_pts = Some(pts_f);
        self.audio_frame_count += 1;
        Ok(())
    }

    /// Write a subtitle cue with explicit start/duration (verdict §9).
    ///
    /// `cue.start`/`cue.duration` are in the subtitle track timescale
    /// (default 1_000). The final cue keeps its own duration; it is never
    /// inferred from the next cue.
    pub fn write_subtitle_cue(&mut self, cue: SubtitleCue<'_>) -> Result<(), MuxerError> {
        use crate::time::{Timescale, rescale};
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let track = self
            .subtitle_track
            .clone()
            .ok_or(MuxerError::SubtitleNotConfigured)?;
        let frame_index = self.subtitle_frame_count;
        if cue.text.is_empty() {
            return Err(MuxerError::EmptySubtitleSample { frame_index });
        }
        if cue.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if cue.text.len() > self.limits.max_subtitle_size {
            return Err(MuxerError::SubtitleTooLarge(cue.text.len()));
        }
        let from = Timescale::new(core::num::NonZeroU32::new(track.timescale.max(1)).unwrap());
        let to = Timescale::new(core::num::NonZeroU32::new(MEDIA_TIMESCALE).unwrap());
        let pts_i64 = rescale(cue.start, from, to)?;
        let dur_i64 = rescale(i64::from(cue.duration), from, to)?;
        if pts_i64 < 0 {
            return Err(MuxerError::NegativeSubtitleDts { dts: pts_i64 });
        }
        if dur_i64 <= 0 || dur_i64 > i64::from(u32::MAX) {
            return Err(MuxerError::ZeroDuration);
        }
        let pts_u = u64::try_from(pts_i64).map_err(|_| MuxerError::TimestampOverflow)?;
        let encoded = match track.codec {
            SubtitleCodec::MovText => {
                crate::muxer::mp4::Mp4Writer::<Vec<u8>>::encode_tx3g_sample(cue.text)
                    .map_err(|_| MuxerError::SubtitleTooLarge(cue.text.len()))?
            }
            SubtitleCodec::WebVtt => {
                // Minimal vttc: `vttc` box with `payl` payload.
                let mut vttc_payload = Vec::new();
                let payl = {
                    let mut p = Vec::new();
                    p.extend_from_slice(&(8 + cue.text.len() as u64).to_be_bytes()[4..8]);
                    p.extend_from_slice(b"payl");
                    p.extend_from_slice(cue.text.as_bytes());
                    p
                };
                vttc_payload.extend_from_slice(&payl);
                let mut vttc = Vec::new();
                vttc.extend_from_slice(&(8 + vttc_payload.len() as u64).to_be_bytes()[4..8]);
                vttc.extend_from_slice(b"vttc");
                vttc.extend_from_slice(&vttc_payload);
                vttc
            }
        };
        self.writer
            .write_subtitle_sample(pts_u, dur_i64 as u32, &encoded)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;
        let pts_f = pts_u as f64 / MEDIA_TIMESCALE as f64;
        self.last_subtitle_pts = Some(pts_f);
        self.subtitle_frame_count += 1;
        Ok(())
    }

    /// Convert internal Mp4WriterError to MuxerError with context
    fn convert_mp4_error(&self, err: Mp4WriterError, frame_index: u64) -> MuxerError {
        match err {
            Mp4WriterError::NonIncreasingTimestamp => MuxerError::NonIncreasingVideoPts {
                prev_pts: self.last_video_pts.unwrap_or(0.0),
                curr_pts: 0.0, // We don't have access here, but validation above catches this
                frame_index,
            },
            Mp4WriterError::FirstFrameMustBeKeyframe => MuxerError::FirstVideoFrameMustBeKeyframe,
            Mp4WriterError::FirstFrameMissingSpsPps => MuxerError::FirstVideoFrameMissingSpsPps,
            Mp4WriterError::FirstFrameMissingSequenceHeader => {
                MuxerError::FirstAv1FrameMissingSequenceHeader
            }
            Mp4WriterError::FirstFrameMissingVp9Config => {
                MuxerError::FirstVp9FrameMissingSequenceHeader
            }
            Mp4WriterError::InvalidAdts => MuxerError::InvalidAdts { frame_index },
            Mp4WriterError::InvalidAdtsDetailed(error) => {
                MuxerError::InvalidAdtsDetailed { frame_index, error }
            }
            Mp4WriterError::InvalidOpusPacket => MuxerError::InvalidOpusPacket { frame_index },
            Mp4WriterError::InvalidFlacFrame => MuxerError::InvalidFlacFrame { frame_index },
            Mp4WriterError::MissingFlacStreaminfo => MuxerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "FLAC STREAMINFO must be provided using with_flac_streaminfo()",
            )),
            Mp4WriterError::AudioNotEnabled => MuxerError::AudioNotConfigured,
            Mp4WriterError::SubtitleNotEnabled => MuxerError::SubtitleNotConfigured,
            Mp4WriterError::DurationOverflow => MuxerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "duration overflow",
            )),
            Mp4WriterError::DecoderConfigurationChanged => MuxerError::DecoderConfigurationChanged,
            Mp4WriterError::AlreadyFinalized => MuxerError::AlreadyFinished,
            Mp4WriterError::Io(err) => MuxerError::Io(err),
        }
    }

    /// Write an audio frame to the container.
    ///
    /// `pts` is the presentation timestamp in seconds.  The `data` slice
    /// contains the encoded audio frame (an AAC ADTS frame).
    /// Audio timestamps must be non-decreasing and must not precede the first video frame.
    pub fn write_audio(&mut self, pts: f64, data: &[u8]) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }

        let frame_index = self.audio_frame_count;

        // Validate PTS is finite (not NaN or Inf)
        if !pts.is_finite() {
            return Err(MuxerError::InvalidAudioPts { pts, frame_index });
        }

        // Validate PTS is non-negative
        if pts < 0.0 {
            return Err(MuxerError::NegativeAudioPts { pts, frame_index });
        }

        // Validate frame is not empty
        if data.is_empty() {
            return Err(MuxerError::EmptyAudioFrame { frame_index });
        }

        // Validate PTS is non-decreasing
        if let Some(prev) = self.last_audio_pts {
            if pts < prev {
                return Err(MuxerError::DecreasingAudioPts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }

        // Verdict §7: audio may begin before video; edit lists preserve A/V
        // sync. Do NOT reject audio-before-video here.

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;

        self.writer
            .write_audio_sample(pts_units, data)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;

        self.last_audio_pts = Some(pts);
        self.audio_frame_count += 1;
        Ok(())
    }

    /// Write a subtitle sample to the container.
    ///
    /// `pts` is the presentation timestamp in seconds and `duration` is the sample duration
    /// in seconds. `text` is UTF-8 subtitle payload.
    pub fn write_subtitle(
        &mut self,
        pts: f64,
        duration: f64,
        text: &str,
    ) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.subtitle_track.is_none() {
            return Err(MuxerError::SubtitleNotConfigured);
        }

        let frame_index = self.subtitle_frame_count;

        if !pts.is_finite() {
            return Err(MuxerError::InvalidSubtitlePts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeSubtitlePts { pts, frame_index });
        }

        if !duration.is_finite() || duration <= 0.0 {
            return Err(MuxerError::InvalidSubtitleDuration {
                duration_secs: duration,
                frame_index,
            });
        }

        if text.is_empty() {
            return Err(MuxerError::EmptySubtitleSample { frame_index });
        }

        if let Some(prev) = self.last_subtitle_pts {
            if pts < prev {
                return Err(MuxerError::DecreasingSubtitlePts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;
        let scaled_duration = (duration * MEDIA_TIMESCALE as f64).round().max(1.0);
        let duration_units = scaled_duration as u32;

        self.writer
            .write_subtitle_sample(pts_units, duration_units, text.as_bytes())
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;

        self.last_subtitle_pts = Some(pts);
        self.subtitle_frame_count += 1;
        Ok(())
    }

    /// Simple video encoding method.
    pub fn encode_video(&mut self, data: &[u8], duration_ms: u32) -> Result<(), MuxerError> {
        let pts = self.current_video_pts;
        let is_keyframe = self.is_keyframe(data);
        self.write_video(pts, data, is_keyframe)?;
        self.current_video_pts += duration_ms as f64 / 1000.0;
        Ok(())
    }

    /// Simple audio encoding method.
    pub fn encode_audio(&mut self, data: &[u8], samples: u32) -> Result<(), MuxerError> {
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }
        let sample_rate = self.audio_track.as_ref().unwrap().sample_rate;
        let pts = self.current_audio_pts;
        self.write_audio(pts, data)?;
        self.current_audio_pts += samples as f64 / sample_rate as f64;
        Ok(())
    }

    /// Helper to detect if a video frame is a keyframe.
    fn is_keyframe(&self, data: &[u8]) -> bool {
        let Some(codec) = self.video_track.as_ref().map(|t| t.codec) else {
            return false;
        };
        detect_keyframe(codec, data, self.video_frame_count)
    }

    /// Finalise the container and flush any buffered data.
    ///
    /// In the current slice this writes the `ftyp`/`moov` boxes, resulting
    /// in a minimal MP4 header that can be inspected by the slice 02 tests.
    pub fn finish_in_place(&mut self) -> Result<(), MuxerError> {
        self.finish_in_place_with_stats().map(|_| ())
    }
}

/// Shared keyframe detection for the MP4 ([`Muxer`]) and Matroska
/// ([`MkvMuxer`]) frontends.
fn detect_keyframe(codec: VideoCodec, data: &[u8], video_frame_count: u64) -> bool {
    // INV-100: Video frame data must not be empty
    assert_invariant!(
        !data.is_empty(),
        "INV-100: Video frame data must not be empty",
        "api::is_keyframe"
    );

    match codec {
        VideoCodec::H264 => {
            // Check for IDR NAL (type 5)
            let has_idr = AnnexBNalIter::new(data).any(|nal| (nal[0] & 0x1f) == 5);
            has_idr
        }
        VideoCodec::H265 => {
            // Check for IDR NAL (type 19-21)
            let has_idr = AnnexBNalIter::new(data).any(|nal| {
                let nal_type = (nal[0] >> 1) & 0x3f;
                (19..=21).contains(&nal_type)
            });
            has_idr
        }
        VideoCodec::Av1 => {
            // For AV1, check if it's a key frame (first frame or has key frame flag)
            // Simple heuristic: first frame is keyframe
            let is_key = video_frame_count == 0;

            // INV-103: AV1 first frame must be keyframe
            assert_invariant!(
                is_key || video_frame_count > 0,
                "AV1 first frame must be keyframe",
                "api::is_keyframe::av1"
            );

            is_key
        }
        VideoCodec::Vp9 => {
            // Use VP9 keyframe detection
            let is_key = is_vp9_keyframe(data).unwrap_or(false);

            // INV-104: VP9 keyframe detection must handle invalid frames gracefully
            assert_invariant!(
                is_key || data.len() >= 3,
                "VP9 keyframe detection requires minimum frame size",
                "api::is_keyframe::vp9"
            );

            is_key
        }
    }
}

impl<Writer: Write> Muxer<Writer> {
    /// Finalise the container and return muxing statistics.
    pub fn finish_in_place_with_stats(&mut self) -> Result<MuxerStats, MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let video_params = self.video_track.as_ref().map(|t| Mp4VideoTrack {
            width: t.width,
            height: t.height,
            language: Some(String::from_utf8_lossy(&t.language.as_bytes()).into_owned()),
        });
        self.writer.finalize(
            video_params.as_ref(),
            self.metadata.as_ref(),
            self.fast_start,
        )?;
        self.finished = true;

        let video_frames = self.writer.video_sample_count();
        let audio_frames = self.writer.audio_sample_count();
        let subtitle_frames = self.writer.subtitle_sample_count();
        let duration_ticks = self.writer.max_end_pts().unwrap_or(0);
        let duration_secs = duration_ticks as f64 / MEDIA_TIMESCALE as f64;
        let bytes_written = self.writer.bytes_written();

        Ok(MuxerStats {
            video_frames,
            audio_frames,
            subtitle_frames,
            duration_secs,
            bytes_written,
        })
    }

    pub fn finish(mut self) -> Result<(), MuxerError> {
        self.finish_in_place()
    }

    /// Finalise the container and return muxing statistics.
    pub fn finish_with_stats(mut self) -> Result<MuxerStats, MuxerError> {
        self.finish_in_place_with_stats()
    }

    /// Flush the muxer and finalize the output.
    pub fn flush(self) -> Result<(), MuxerError> {
        self.finish()
    }
}

/// Seekable-streaming MP4 muxer: the bounded-RAM counterpart to [`Muxer`].
///
/// Produced by [`MuxerBuilder::build_streaming_seekable`]. Sample bytes are
/// written to the output as they arrive; only per-sample metadata is kept,
/// so peak RAM is O(metadata) instead of O(media). Requires `W: Write +
/// Seek` (files, `Cursor<Vec<u8>>`).
///
/// Timestamp validation, keyframe requirements, decoder-config rules, and
/// the error vocabulary mirror [`Muxer`]; the output layout is
/// `ftyp`, `mdat`, `moov` (progressive) rather than fast-start.
pub struct StreamingMuxer<Writer> {
    inner: SeekableStreamingWriter<Writer>,
    video_track: Option<VideoTrackConfig>,
    audio_track: Option<AudioTrackConfig>,
    subtitle_track: Option<SubtitleTrackConfig>,
    video_codec: Option<VideoCodec>,
    limits: Limits,
    first_video_pts: Option<f64>,
    last_video_pts: Option<f64>,
    last_video_dts: Option<f64>,
    last_audio_pts: Option<f64>,
    last_subtitle_pts: Option<f64>,
    video_frame_count: u64,
    audio_frame_count: u64,
    subtitle_frame_count: u64,
    video_end_ticks: Option<u64>,
    audio_end_ticks: Option<u64>,
    subtitle_end_ticks: Option<u64>,
    finished: bool,
}

impl<Writer: Write + std::io::Seek> StreamingMuxer<Writer> {
    fn convert_mp4_error(&self, err: Mp4WriterError, frame_index: u64) -> MuxerError {
        match err {
            Mp4WriterError::NonIncreasingTimestamp => MuxerError::NonIncreasingVideoPts {
                prev_pts: self.last_video_pts.unwrap_or(0.0),
                curr_pts: 0.0,
                frame_index,
            },
            Mp4WriterError::FirstFrameMustBeKeyframe => MuxerError::FirstVideoFrameMustBeKeyframe,
            Mp4WriterError::FirstFrameMissingSpsPps => MuxerError::FirstVideoFrameMissingSpsPps,
            Mp4WriterError::FirstFrameMissingSequenceHeader => {
                MuxerError::FirstAv1FrameMissingSequenceHeader
            }
            Mp4WriterError::FirstFrameMissingVp9Config => {
                MuxerError::FirstVp9FrameMissingSequenceHeader
            }
            Mp4WriterError::InvalidAdts => MuxerError::InvalidAdts { frame_index },
            Mp4WriterError::InvalidAdtsDetailed(error) => {
                MuxerError::InvalidAdtsDetailed { frame_index, error }
            }
            Mp4WriterError::InvalidOpusPacket => MuxerError::InvalidOpusPacket { frame_index },
            Mp4WriterError::InvalidFlacFrame => MuxerError::InvalidFlacFrame { frame_index },
            Mp4WriterError::MissingFlacStreaminfo => MuxerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "FLAC STREAMINFO must be provided using with_flac_streaminfo()",
            )),
            Mp4WriterError::AudioNotEnabled => MuxerError::AudioNotConfigured,
            Mp4WriterError::SubtitleNotEnabled => MuxerError::SubtitleNotConfigured,
            Mp4WriterError::DurationOverflow => MuxerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "duration overflow",
            )),
            Mp4WriterError::DecoderConfigurationChanged => MuxerError::DecoderConfigurationChanged,
            Mp4WriterError::AlreadyFinalized => MuxerError::AlreadyFinished,
            Mp4WriterError::Io(err) => MuxerError::Io(err),
        }
    }

    /// Total bytes written so far (headers + streamed samples).
    pub fn bytes_written(&self) -> u64 {
        self.inner.bytes_written()
    }

    /// Push one video sample at media-timescale ticks.
    ///
    /// `duration` is `Some` for the integer API (explicit) and `None` for
    /// the `f64` shims (inferred from the next sample, as in [`Muxer`]).
    fn push_video(
        &mut self,
        pts_u: u64,
        dts_u: u64,
        data: &[u8],
        is_sync: bool,
        duration: Option<u32>,
    ) -> Result<(), MuxerError> {
        let frame_index = self.video_frame_count;
        if self.video_track.is_none() {
            return Err(MuxerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "video track not configured",
            )));
        }
        if self.video_frame_count as usize >= self.limits.max_samples_per_track {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "video samples",
            });
        }
        self.inner
            .write_video_sample_with_dts(pts_u, dts_u, data, is_sync)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;
        if let Some(dur) = duration {
            self.inner.set_last_video_duration(dur);
            self.video_end_ticks = Some(pts_u.saturating_add(u64::from(dur)));
        }
        let pts_f = pts_u as f64 / MEDIA_TIMESCALE as f64;
        let dts_f = dts_u as f64 / MEDIA_TIMESCALE as f64;
        if self.first_video_pts.is_none() {
            self.first_video_pts = Some(pts_f);
        }
        self.last_video_pts = Some(pts_f);
        self.last_video_dts = Some(dts_f);
        self.video_frame_count += 1;
        Ok(())
    }

    /// Write a video sample with integer timestamps and explicit duration.
    ///
    /// Preferred API (verdict §§3-4); mirrors [`Muxer::write_video_sample`].
    pub fn write_video_sample(&mut self, sample: EncodedSample<'_>) -> Result<(), MuxerError> {
        use crate::time::{Timescale, rescale};
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let frame_index = self.video_frame_count;
        if sample.data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }
        if sample.timing.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if sample.timing.pts < 0 || sample.timing.dts < 0 {
            return Err(MuxerError::NegativeVideoDts {
                dts: sample.timing.dts as f64,
                frame_index,
            });
        }
        let _ = sample.timing.composition_offset()?;
        if sample.data.len() > self.limits.max_sample_size {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "video sample",
            });
        }
        let track_ts = self
            .video_track
            .as_ref()
            .map(|t| t.timescale)
            .unwrap_or(90_000);
        let from = Timescale::new(core::num::NonZeroU32::new(track_ts.max(1)).unwrap());
        let to = Timescale::new(core::num::NonZeroU32::new(MEDIA_TIMESCALE).unwrap());
        let pts_i64 = rescale(sample.timing.pts, from, to)?;
        let dts_i64 = rescale(sample.timing.dts, from, to)?;
        let dur_i64 = rescale(i64::from(sample.timing.duration), from, to)?;
        if dur_i64 <= 0 || dur_i64 > i64::from(u32::MAX) {
            return Err(MuxerError::ZeroDuration);
        }
        let pts_u = u64::try_from(pts_i64).map_err(|_| MuxerError::TimestampOverflow)?;
        let dts_u = u64::try_from(dts_i64).map_err(|_| MuxerError::TimestampOverflow)?;
        if let Some(prev_f) = self.last_video_dts {
            let prev_u = (prev_f * MEDIA_TIMESCALE as f64).round() as u64;
            if dts_u <= prev_u {
                return Err(MuxerError::NonIncreasingIntDts {
                    previous: prev_u as i64,
                    current: dts_u as i64,
                });
            }
        }
        self.push_video(
            pts_u,
            dts_u,
            sample.data,
            sample.is_sync,
            Some(dur_i64 as u32),
        )
    }

    /// Write a video frame (`dts == pts`), `f64` compat shim.
    pub fn write_video(
        &mut self,
        pts: f64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let frame_index = self.video_frame_count;
        if data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }
        if !pts.is_finite() {
            return Err(MuxerError::InvalidVideoPts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeVideoPts { pts, frame_index });
        }
        if let Some(prev) = self.last_video_pts {
            if pts <= prev {
                return Err(MuxerError::NonIncreasingVideoPts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }
        let pts_u = (pts * MEDIA_TIMESCALE as f64).round() as u64;
        self.push_video(pts_u, pts_u, data, is_keyframe, None)
    }

    /// Write a video frame with explicit decode timestamp, `f64` compat shim.
    pub fn write_video_with_dts(
        &mut self,
        pts: f64,
        dts: f64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let frame_index = self.video_frame_count;
        if data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }
        if !pts.is_finite() {
            return Err(MuxerError::InvalidVideoPts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeVideoPts { pts, frame_index });
        }
        if !dts.is_finite() {
            return Err(MuxerError::InvalidVideoDts { dts, frame_index });
        }
        if dts < 0.0 {
            return Err(MuxerError::NegativeVideoDts { dts, frame_index });
        }
        if let Some(prev_dts) = self.last_video_dts {
            if dts <= prev_dts {
                return Err(MuxerError::NonIncreasingDts {
                    prev_dts,
                    curr_dts: dts,
                    frame_index,
                });
            }
        }
        let pts_u = (pts * MEDIA_TIMESCALE as f64).round() as u64;
        let dts_u = (dts * MEDIA_TIMESCALE as f64).round() as u64;
        self.push_video(pts_u, dts_u, data, is_keyframe, None)
    }

    /// Push one audio sample at media-timescale ticks.
    fn push_audio(
        &mut self,
        pts_u: u64,
        data: &[u8],
        duration: Option<u32>,
    ) -> Result<(), MuxerError> {
        let frame_index = self.audio_frame_count;
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }
        if self.audio_frame_count as usize >= self.limits.max_samples_per_track {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "audio samples",
            });
        }
        self.inner
            .write_audio_sample(pts_u, data)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;
        if let Some(dur) = duration {
            self.inner.set_last_audio_duration(dur);
            self.audio_end_ticks = Some(pts_u.saturating_add(u64::from(dur)));
        }
        self.last_audio_pts = Some(pts_u as f64 / MEDIA_TIMESCALE as f64);
        self.audio_frame_count += 1;
        Ok(())
    }

    /// Write an audio sample with integer timestamps and explicit duration.
    pub fn write_audio_sample(&mut self, sample: EncodedSample<'_>) -> Result<(), MuxerError> {
        use crate::time::{Timescale, rescale};
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }
        let frame_index = self.audio_frame_count;
        if sample.data.is_empty() {
            return Err(MuxerError::EmptyAudioFrame { frame_index });
        }
        if sample.timing.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if sample.timing.pts < 0 {
            return Err(MuxerError::NegativeAudioPts {
                pts: sample.timing.pts as f64,
                frame_index,
            });
        }
        if sample.data.len() > self.limits.max_sample_size {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "audio sample",
            });
        }
        let track_ts = self
            .audio_track
            .as_ref()
            .map(|t| t.timescale)
            .unwrap_or(48_000);
        let from = Timescale::new(core::num::NonZeroU32::new(track_ts.max(1)).unwrap());
        let to = Timescale::new(core::num::NonZeroU32::new(MEDIA_TIMESCALE).unwrap());
        let pts_i64 = rescale(sample.timing.pts, from, to)?;
        let dur_i64 = rescale(i64::from(sample.timing.duration), from, to)?;
        if dur_i64 <= 0 || dur_i64 > i64::from(u32::MAX) {
            return Err(MuxerError::ZeroDuration);
        }
        let pts_u = u64::try_from(pts_i64).map_err(|_| MuxerError::TimestampOverflow)?;
        self.push_audio(pts_u, sample.data, Some(dur_i64 as u32))
    }

    /// Write an audio frame, `f64` compat shim.
    pub fn write_audio(&mut self, pts: f64, data: &[u8]) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }
        let frame_index = self.audio_frame_count;
        if !pts.is_finite() {
            return Err(MuxerError::InvalidAudioPts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeAudioPts { pts, frame_index });
        }
        if data.is_empty() {
            return Err(MuxerError::EmptyAudioFrame { frame_index });
        }
        if let Some(prev) = self.last_audio_pts {
            if pts < prev {
                return Err(MuxerError::DecreasingAudioPts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }
        let pts_u = (pts * MEDIA_TIMESCALE as f64).round() as u64;
        self.push_audio(pts_u, data, None)
    }

    /// Write a subtitle cue with explicit start/duration (verdict §9).
    pub fn write_subtitle_cue(&mut self, cue: SubtitleCue<'_>) -> Result<(), MuxerError> {
        use crate::time::{Timescale, rescale};
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let track = self
            .subtitle_track
            .clone()
            .ok_or(MuxerError::SubtitleNotConfigured)?;
        let frame_index = self.subtitle_frame_count;
        if cue.text.is_empty() {
            return Err(MuxerError::EmptySubtitleSample { frame_index });
        }
        if cue.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if cue.text.len() > self.limits.max_subtitle_size {
            return Err(MuxerError::SubtitleTooLarge(cue.text.len()));
        }
        if self.subtitle_frame_count as usize >= self.limits.max_samples_per_track {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "subtitle samples",
            });
        }
        let from = Timescale::new(core::num::NonZeroU32::new(track.timescale.max(1)).unwrap());
        let to = Timescale::new(core::num::NonZeroU32::new(MEDIA_TIMESCALE).unwrap());
        let pts_i64 = rescale(cue.start, from, to)?;
        let dur_i64 = rescale(i64::from(cue.duration), from, to)?;
        if pts_i64 < 0 {
            return Err(MuxerError::NegativeSubtitleDts { dts: pts_i64 });
        }
        if dur_i64 <= 0 || dur_i64 > i64::from(u32::MAX) {
            return Err(MuxerError::ZeroDuration);
        }
        let pts_u = u64::try_from(pts_i64).map_err(|_| MuxerError::TimestampOverflow)?;
        let encoded = match track.codec {
            SubtitleCodec::MovText => {
                crate::muxer::mp4::Mp4Writer::<Vec<u8>>::encode_tx3g_sample(cue.text)
                    .map_err(|_| MuxerError::SubtitleTooLarge(cue.text.len()))?
            }
            SubtitleCodec::WebVtt => {
                let mut vttc_payload = Vec::new();
                let payl = {
                    let mut p = Vec::new();
                    p.extend_from_slice(&(8 + cue.text.len() as u64).to_be_bytes()[4..8]);
                    p.extend_from_slice(b"payl");
                    p.extend_from_slice(cue.text.as_bytes());
                    p
                };
                vttc_payload.extend_from_slice(&payl);
                let mut vttc = Vec::new();
                vttc.extend_from_slice(&(8 + vttc_payload.len() as u64).to_be_bytes()[4..8]);
                vttc.extend_from_slice(b"vttc");
                vttc.extend_from_slice(&vttc_payload);
                vttc
            }
        };
        self.inner
            .write_subtitle_sample(pts_u, dur_i64 as u32, &encoded)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;
        self.subtitle_end_ticks = Some(pts_u.saturating_add(dur_i64 as u64));
        self.last_subtitle_pts = Some(pts_u as f64 / MEDIA_TIMESCALE as f64);
        self.subtitle_frame_count += 1;
        Ok(())
    }

    /// Write a subtitle, `f64` compat shim.
    pub fn write_subtitle(
        &mut self,
        pts: f64,
        duration: f64,
        text: &str,
    ) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.subtitle_track.is_none() {
            return Err(MuxerError::SubtitleNotConfigured);
        }
        let frame_index = self.subtitle_frame_count;
        if !pts.is_finite() {
            return Err(MuxerError::InvalidSubtitlePts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeSubtitlePts { pts, frame_index });
        }
        if !duration.is_finite() || duration <= 0.0 {
            return Err(MuxerError::InvalidSubtitleDuration {
                duration_secs: duration,
                frame_index,
            });
        }
        if text.is_empty() {
            return Err(MuxerError::EmptySubtitleSample { frame_index });
        }
        if let Some(prev) = self.last_subtitle_pts {
            if pts < prev {
                return Err(MuxerError::DecreasingSubtitlePts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }
        let pts_u = (pts * MEDIA_TIMESCALE as f64).round() as u64;
        let dur_u = (duration * MEDIA_TIMESCALE as f64).round().max(1.0) as u32;
        let encoded = crate::muxer::mp4::Mp4Writer::<Vec<u8>>::encode_tx3g_sample(text)
            .map_err(|_| MuxerError::SubtitleTooLarge(text.len()))?;
        self.inner
            .write_subtitle_sample(pts_u, dur_u, &encoded)
            .map_err(|e| self.convert_mp4_error(e, frame_index))?;
        self.subtitle_end_ticks = Some(pts_u.saturating_add(u64::from(dur_u)));
        self.last_subtitle_pts = Some(pts);
        self.subtitle_frame_count += 1;
        Ok(())
    }

    /// Finish: patch `mdat`, append `moov`, return statistics.
    pub fn finish_with_stats(mut self) -> Result<MuxerStats, MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        self.finished = true;
        let bytes_written = self
            .inner
            .finish(self.video_codec)
            .map_err(MuxerError::Io)?;
        let duration_ticks = [
            self.video_end_ticks,
            self.audio_end_ticks,
            self.subtitle_end_ticks,
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(0);
        Ok(MuxerStats {
            video_frames: self.video_frame_count,
            audio_frames: self.audio_frame_count,
            subtitle_frames: self.subtitle_frame_count,
            duration_secs: duration_ticks as f64 / MEDIA_TIMESCALE as f64,
            bytes_written,
        })
    }

    /// Finish the container.
    pub fn finish(self) -> Result<(), MuxerError> {
        self.finish_with_stats().map(|_| ())
    }

    /// Flush the muxer and finalize the output.
    pub fn flush(self) -> Result<(), MuxerError> {
        self.finish()
    }
}

/// Matroska/WebM muxer: the parallel counterpart to [`Muxer`] for the
/// Matroska container family.
///
/// Produced by [`MuxerBuilder::build_mkv`]. Timestamp validation, keyframe
/// requirements and error vocabulary mirror [`Muxer`]; only the container
/// encoding differs (EBML via the external `mkv-element` crate instead of
/// ISO-BMFF boxes). `MkvMuxer<W>` is `Send` when `W: Send` and `Sync` when
/// `W: Sync`.
pub struct MkvMuxer<Writer> {
    writer: MkvWriter<Writer>,
    video_track: Option<VideoTrackConfig>,
    audio_track: Option<AudioTrackConfig>,
    subtitle_track: Option<SubtitleTrackConfig>,
    metadata: Option<Metadata>,
    container: MkvContainer,
    first_video_pts: Option<f64>,
    last_video_pts: Option<f64>,
    last_video_dts: Option<f64>,
    last_audio_pts: Option<f64>,
    last_subtitle_pts: Option<f64>,
    video_frame_count: u64,
    audio_frame_count: u64,
    subtitle_frame_count: u64,
    finished: bool,
    current_video_pts: f64,
    current_audio_pts: f64,
    limits: Limits,
}

impl<Writer: Write> MkvMuxer<Writer> {
    /// Which Matroska-family container this muxer writes.
    pub fn container(&self) -> ContainerFormat {
        match self.container {
            MkvContainer::Matroska => ContainerFormat::Matroska,
            MkvContainer::WebM => ContainerFormat::WebM,
        }
    }

    /// Write a video frame to the container.
    ///
    /// `pts` is the presentation timestamp in seconds. Frames must be
    /// supplied in strictly increasing PTS order. The `data` slice uses the
    /// same input formats as [`Muxer::write_video`] (Annex B for
    /// H.264/H.265, OBU stream for AV1, compressed frames for VP9).
    ///
    /// For streams with B-frames (where PTS != DTS), use
    /// [`MkvMuxer::write_video_with_dts`] instead.
    pub fn write_video(
        &mut self,
        pts: f64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MuxerError> {
        let frame_index = self.video_frame_count;

        if data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }
        if !pts.is_finite() {
            return Err(MuxerError::InvalidVideoPts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeVideoPts { pts, frame_index });
        }
        if let Some(prev) = self.last_video_pts {
            if pts <= prev {
                return Err(MuxerError::NonIncreasingVideoPts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;

        if self.first_video_pts.is_none() {
            self.first_video_pts = Some(pts);
        }

        self.writer
            .write_video_sample(pts_units, data, is_keyframe)
            .map_err(|e| self.convert_mkv_error(e, frame_index))?;

        self.last_video_pts = Some(pts);
        self.video_frame_count += 1;
        Ok(())
    }

    /// Write a video frame with explicit decode timestamp for B-frame support.
    ///
    /// Semantics mirror [`Muxer::write_video_with_dts`]: frames are fed in
    /// decode order with strictly increasing DTS, while PTS carries display
    /// order. B-frames are stored as Matroska `BlockGroup`s with an explicit
    /// reference marker.
    pub fn write_video_with_dts(
        &mut self,
        pts: f64,
        dts: f64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }

        let frame_index = self.video_frame_count;

        if data.is_empty() {
            return Err(MuxerError::EmptyVideoFrame { frame_index });
        }
        if !pts.is_finite() {
            return Err(MuxerError::InvalidVideoPts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeVideoPts { pts, frame_index });
        }
        if !dts.is_finite() {
            return Err(MuxerError::InvalidVideoDts { dts, frame_index });
        }
        if dts < 0.0 {
            return Err(MuxerError::NegativeVideoDts { dts, frame_index });
        }
        if let Some(prev_dts) = self.last_video_dts {
            if dts <= prev_dts {
                return Err(MuxerError::NonIncreasingDts {
                    prev_dts,
                    curr_dts: dts,
                    frame_index,
                });
            }
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;
        let scaled_dts = (dts * MEDIA_TIMESCALE as f64).round();
        let dts_units = scaled_dts as u64;

        if self.first_video_pts.is_none() {
            self.first_video_pts = Some(pts);
        }

        self.writer
            .write_video_sample_with_dts(pts_units, dts_units, data, is_keyframe)
            .map_err(|e| self.convert_mkv_error(e, frame_index))?;

        self.last_video_pts = Some(pts);
        self.last_video_dts = Some(dts);
        self.video_frame_count += 1;
        Ok(())
    }

    /// Convert an internal `MkvWriterError` to `MuxerError` with context.
    fn convert_mkv_error(&self, err: MkvWriterError, frame_index: u64) -> MuxerError {
        match err {
            MkvWriterError::NonIncreasingTimestamp => MuxerError::NonIncreasingVideoPts {
                prev_pts: self.last_video_pts.unwrap_or(0.0),
                curr_pts: 0.0,
                frame_index,
            },
            MkvWriterError::FirstFrameMustBeKeyframe => MuxerError::FirstVideoFrameMustBeKeyframe,
            MkvWriterError::FirstFrameMissingSpsPps => MuxerError::FirstVideoFrameMissingSpsPps,
            MkvWriterError::FirstFrameMissingSequenceHeader => {
                MuxerError::FirstAv1FrameMissingSequenceHeader
            }
            MkvWriterError::FirstFrameMissingVp9Config => {
                MuxerError::FirstVp9FrameMissingSequenceHeader
            }
            MkvWriterError::InvalidAdtsDetailed(error) => {
                MuxerError::InvalidAdtsDetailed { frame_index, error }
            }
            MkvWriterError::InvalidOpusPacket => MuxerError::InvalidOpusPacket { frame_index },
            MkvWriterError::InvalidFlacFrame => MuxerError::InvalidFlacFrame { frame_index },
            MkvWriterError::MissingFlacStreaminfo => MuxerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "FLAC STREAMINFO must be provided using with_flac_streaminfo()",
            )),
            MkvWriterError::AudioNotEnabled => MuxerError::AudioNotConfigured,
            MkvWriterError::SubtitleNotEnabled => MuxerError::SubtitleNotConfigured,
            MkvWriterError::DurationOverflow => MuxerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "duration overflow",
            )),
            MkvWriterError::AlreadyFinalized => MuxerError::AlreadyFinished,
            MkvWriterError::UnsupportedForWebM { codec, reason } => {
                MuxerError::UnsupportedForContainer {
                    codec,
                    container: self.container.to_string(),
                    reason,
                }
            }
            MkvWriterError::Encode(message) => MuxerError::Io(std::io::Error::other(message)),
            MkvWriterError::Io(err) => MuxerError::Io(err),
        }
    }

    /// Write an audio frame to the container.
    ///
    /// `pts` is the presentation timestamp in seconds. The `data` slice uses
    /// the same input formats as [`Muxer::write_audio`] (ADTS for AAC, raw
    /// packets for Opus). Audio timestamps must be non-decreasing and must
    /// not precede the first video frame.
    pub fn write_audio(&mut self, pts: f64, data: &[u8]) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }

        let frame_index = self.audio_frame_count;

        if !pts.is_finite() {
            return Err(MuxerError::InvalidAudioPts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeAudioPts { pts, frame_index });
        }
        if data.is_empty() {
            return Err(MuxerError::EmptyAudioFrame { frame_index });
        }
        if let Some(prev) = self.last_audio_pts {
            if pts < prev {
                return Err(MuxerError::DecreasingAudioPts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }
        // Verdict §7: audio may begin before video (edit lists handle offset).

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;

        self.writer
            .write_audio_sample(pts_units, data)
            .map_err(|e| self.convert_mkv_error(e, frame_index))?;

        self.last_audio_pts = Some(pts);
        self.audio_frame_count += 1;
        Ok(())
    }

    /// Write a subtitle sample to the container.
    ///
    /// `pts` is the presentation timestamp in seconds and `duration` is the
    /// sample duration in seconds. `text` is a UTF-8 subtitle payload stored
    /// as `S_TEXT/UTF8`. (WebM rejects subtitles; use Matroska instead.)
    pub fn write_subtitle(
        &mut self,
        pts: f64,
        duration: f64,
        text: &str,
    ) -> Result<(), MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        if self.subtitle_track.is_none() {
            return Err(MuxerError::SubtitleNotConfigured);
        }

        let frame_index = self.subtitle_frame_count;

        if !pts.is_finite() {
            return Err(MuxerError::InvalidSubtitlePts { pts, frame_index });
        }
        if pts < 0.0 {
            return Err(MuxerError::NegativeSubtitlePts { pts, frame_index });
        }
        if !duration.is_finite() || duration <= 0.0 {
            return Err(MuxerError::InvalidSubtitleDuration {
                duration_secs: duration,
                frame_index,
            });
        }
        if text.is_empty() {
            return Err(MuxerError::EmptySubtitleSample { frame_index });
        }
        if let Some(prev) = self.last_subtitle_pts {
            if pts < prev {
                return Err(MuxerError::DecreasingSubtitlePts {
                    prev_pts: prev,
                    curr_pts: pts,
                    frame_index,
                });
            }
        }

        let scaled_pts = (pts * MEDIA_TIMESCALE as f64).round();
        let pts_units = scaled_pts as u64;
        let scaled_duration = (duration * MEDIA_TIMESCALE as f64).round().max(1.0);
        let duration_units = scaled_duration as u32;

        self.writer
            .write_subtitle_sample(pts_units, duration_units, text.as_bytes())
            .map_err(|e| self.convert_mkv_error(e, frame_index))?;

        self.last_subtitle_pts = Some(pts);
        self.subtitle_frame_count += 1;
        Ok(())
    }

    /// Write a video sample with integer timestamps (verdict §§3-4).
    /// Matroska twin of [`Muxer::write_video_sample`]; enforces `limits`.
    pub fn write_video_sample(&mut self, sample: EncodedSample<'_>) -> Result<(), MuxerError> {
        if sample.timing.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if sample.data.len() > self.limits.max_sample_size {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "video sample",
            });
        }
        let _ = sample.timing.composition_offset()?;
        let pts_s = sample.timing.pts as f64 / 90_000.0;
        self.write_video(pts_s, sample.data, sample.is_sync)
    }

    /// Write an audio sample with integer timestamps (verdict §§3-4).
    pub fn write_audio_sample(&mut self, sample: EncodedSample<'_>) -> Result<(), MuxerError> {
        if sample.timing.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if sample.data.len() > self.limits.max_sample_size {
            return Err(MuxerError::ResourceLimitExceeded {
                what: "audio sample",
            });
        }
        let pts_s = sample.timing.pts as f64 / 48_000.0;
        self.write_audio(pts_s, sample.data)
    }

    /// Write a subtitle cue with explicit start/duration (verdict §9).
    pub fn write_subtitle_cue(&mut self, cue: SubtitleCue<'_>) -> Result<(), MuxerError> {
        if cue.duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if cue.text.len() > self.limits.max_subtitle_size {
            return Err(MuxerError::SubtitleTooLarge(cue.text.len()));
        }
        self.write_subtitle(
            cue.start as f64 / 1_000.0,
            cue.duration as f64 / 1_000.0,
            cue.text,
        )
    }

    /// Simple video encoding method.
    pub fn encode_video(&mut self, data: &[u8], duration_ms: u32) -> Result<(), MuxerError> {
        let pts = self.current_video_pts;
        let is_keyframe = self.detect_keyframe(data);
        self.write_video(pts, data, is_keyframe)?;
        self.current_video_pts += duration_ms as f64 / 1000.0;
        Ok(())
    }

    /// Simple audio encoding method.
    pub fn encode_audio(&mut self, data: &[u8], samples: u32) -> Result<(), MuxerError> {
        if self.audio_track.is_none() {
            return Err(MuxerError::AudioNotConfigured);
        }
        let sample_rate = self.audio_track.as_ref().unwrap().sample_rate;
        let pts = self.current_audio_pts;
        self.write_audio(pts, data)?;
        self.current_audio_pts += samples as f64 / sample_rate as f64;
        Ok(())
    }

    /// Helper to detect if a video frame is a keyframe.
    fn detect_keyframe(&self, data: &[u8]) -> bool {
        let Some(codec) = self.video_track.as_ref().map(|t| t.codec) else {
            return false;
        };
        detect_keyframe(codec, data, self.video_frame_count)
    }

    /// Finalise the container and flush any buffered data.
    pub fn finish_in_place(&mut self) -> Result<(), MuxerError> {
        self.finish_in_place_with_stats().map(|_| ())
    }

    /// Finalise the container and return muxing statistics.
    pub fn finish_in_place_with_stats(&mut self) -> Result<MuxerStats, MuxerError> {
        if self.finished {
            return Err(MuxerError::AlreadyFinished);
        }
        let video_params = self.video_track.as_ref().map(|t| Mp4VideoTrack {
            width: t.width,
            height: t.height,
            language: Some(String::from_utf8_lossy(&t.language.as_bytes()).into_owned()),
        });
        self.writer
            .finalize(
                video_params.as_ref(),
                self.metadata.as_ref(),
                self.container,
            )
            .map_err(|e| {
                // `finalize` surfaces writer errors as `io::Error`; recover the
                // structured WebM rejection when present.
                let message = e.to_string();
                if message.contains("not allowed in WebM") {
                    MuxerError::UnsupportedForContainer {
                        codec: "configured codec".to_string(),
                        container: self.container.to_string(),
                        reason: message,
                    }
                } else {
                    MuxerError::Io(e)
                }
            })?;
        self.finished = true;

        let video_frames = self.writer.video_sample_count();
        let audio_frames = self.writer.audio_sample_count();
        let subtitle_frames = self.writer.subtitle_sample_count();
        let duration_ticks = self.writer.max_end_pts().unwrap_or(0);
        let duration_secs = duration_ticks as f64 / MEDIA_TIMESCALE as f64;
        let bytes_written = self.writer.bytes_written();

        Ok(MuxerStats {
            video_frames,
            audio_frames,
            subtitle_frames,
            duration_secs,
            bytes_written,
        })
    }

    pub fn finish(mut self) -> Result<(), MuxerError> {
        self.finish_in_place()
    }

    /// Finalise the container and return muxing statistics.
    pub fn finish_with_stats(mut self) -> Result<MuxerStats, MuxerError> {
        self.finish_in_place_with_stats()
    }

    /// Flush the muxer and finalize the output.
    pub fn flush(self) -> Result<(), MuxerError> {
        self.finish()
    }
}

// Static assertions for thread safety
#[cfg(test)]
mod thread_safety_tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn muxer_is_send_when_writer_is_send() {
        assert_send::<Muxer<std::fs::File>>();
        assert_send::<Muxer<Vec<u8>>>();
    }

    #[test]
    fn muxer_is_sync_when_writer_is_sync() {
        assert_sync::<Muxer<std::fs::File>>();
        assert_sync::<Muxer<Vec<u8>>>();
    }

    #[test]
    fn builder_is_send_sync() {
        assert_send::<MuxerBuilder<std::fs::File>>();
        assert_sync::<MuxerBuilder<std::fs::File>>();
    }

    #[test]
    fn simple_api_works() -> Result<(), MuxerError> {
        let mut buffer = Vec::new();
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
            .build()?;

        // Test video encoding with a valid keyframe
        let video_data = make_h264_keyframe();
        muxer.encode_video(&video_data, 33)?; // 33ms

        // Test audio encoding
        let audio_data = vec![0xff, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc, 0xaa, 0xbb]; // ADTS
        muxer.encode_audio(&audio_data, 1024)?; // 1024 samples

        muxer.finish()?;
        assert!(!buffer.is_empty());
        Ok(())
    }

    /// Helper to create a valid H.264 keyframe with SPS/PPS
    fn make_h264_keyframe() -> Vec<u8> {
        // Minimal valid Annex B H.264 stream with SPS, PPS, and IDR slice
        let mut data = Vec::new();
        // SPS (NAL type 7)
        data.extend_from_slice(&[
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0x95, 0xa8, 0x28, 0x28, 0x28,
        ]);
        // PPS (NAL type 8)
        data.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80]);
        // IDR slice (NAL type 5)
        data.extend_from_slice(&[
            0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
        ]);
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_new_creates_empty_metadata() {
        let metadata = Metadata::new();
        assert!(metadata.title.is_none());
        assert!(metadata.language.is_none());
        assert!(metadata.creation_time.is_none());
    }

    #[test]
    fn metadata_with_title_sets_title() {
        let metadata = Metadata::new().with_title("Test Title");
        assert_eq!(metadata.title, Some("Test Title".to_string()));
    }

    #[test]
    fn metadata_with_language_sets_language() {
        let metadata = Metadata::new().with_language("eng");
        assert_eq!(metadata.language, Some("eng".to_string()));
    }

    #[test]
    fn metadata_with_creation_time_sets_timestamp() {
        let metadata = Metadata::new().with_creation_time(1234567890);
        assert_eq!(metadata.creation_time, Some(1234567890));
    }

    #[test]
    fn metadata_with_current_time_sets_current_timestamp() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metadata = Metadata::new().with_current_time();

        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        assert!(metadata.creation_time.is_some());
        let time = metadata.creation_time.unwrap();
        assert!(time >= before && time <= after);
    }

    #[test]
    fn metadata_chaining_works() {
        let metadata = Metadata::new()
            .with_title("Test Movie")
            .with_language("spa")
            .with_creation_time(1000000000);

        assert_eq!(metadata.title, Some("Test Movie".to_string()));
        assert_eq!(metadata.language, Some("spa".to_string()));
        assert_eq!(metadata.creation_time, Some(1000000000));
    }

    #[test]
    fn muxer_config_new_creates_basic_config() {
        let config = MuxerConfig::new(1920, 1080, 30.0);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.framerate, 30.0);
        assert!(config.audio.is_none());
        assert!(config.metadata.is_none());
        assert!(config.fast_start);
    }

    #[test]
    fn muxer_config_with_audio_sets_audio_config() {
        let config = MuxerConfig::new(1920, 1080, 30.0).with_audio(
            AudioCodec::Aac(AacProfile::Lc),
            48000,
            2,
        );

        assert!(config.audio.is_some());
        let audio = config.audio.unwrap();
        assert!(matches!(audio.codec, AudioCodec::Aac(AacProfile::Lc)));
        assert_eq!(audio.sample_rate, 48000);
        assert_eq!(audio.channels, 2);
    }

    #[test]
    fn muxer_config_with_audio_none_clears_audio() {
        let config = MuxerConfig::new(1920, 1080, 30.0)
            .with_audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
            .with_audio(AudioCodec::None, 0, 0);

        assert!(config.audio.is_none());
    }

    #[test]
    fn muxer_config_with_metadata_sets_metadata() {
        let metadata = Metadata::new().with_title("Test");
        let config = MuxerConfig::new(1920, 1080, 30.0).with_metadata(metadata);

        assert!(config.metadata.is_some());
        assert_eq!(config.metadata.unwrap().title, Some("Test".to_string()));
    }

    #[test]
    fn muxer_config_with_fast_start_sets_fast_start() {
        let config = MuxerConfig::new(1920, 1080, 30.0).with_fast_start(false);

        assert!(!config.fast_start);
    }

    #[test]
    fn muxer_config_chaining_works() {
        let metadata = Metadata::new()
            .with_title("Chained Test")
            .with_language("eng");

        let config = MuxerConfig::new(1280, 720, 24.0)
            .with_audio(AudioCodec::Opus, 48000, 1)
            .with_metadata(metadata)
            .with_fast_start(false);

        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.framerate, 24.0);
        assert!(config.audio.is_some());
        assert!(config.metadata.is_some());
        assert!(!config.fast_start);

        let audio = config.audio.unwrap();
        assert!(matches!(audio.codec, AudioCodec::Opus));

        let metadata = config.metadata.unwrap();
        assert_eq!(metadata.title, Some("Chained Test".to_string()));
        assert_eq!(metadata.language, Some("eng".to_string()));
    }
}
