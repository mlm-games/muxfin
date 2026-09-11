//! Matroska (MKV) / WebM muxing built on external EBML primitives.
//!
//! This module provides [`MkvWriter`], a buffered writer that mirrors
//! [`crate::muxer::mp4::Mp4Writer`]: samples are queued with
//! [`write_video_sample`](MkvWriter::write_video_sample) /
//! [`write_audio_sample`](MkvWriter::write_audio_sample) /
//! [`write_subtitle_sample`](MkvWriter::write_subtitle_sample) and the
//! complete file (EBML header + Segment with Info, Tracks, Clusters and
//! Cues) is emitted by [`finalize`](MkvWriter::finalize).
//!
//! EBML encoding itself is delegated to the external [`mkv_element`] crate
//! (typed elements, pure Rust, no unsafe). Codec configuration bytes
//! (avcC/hvcC/av1C payloads, ADTS handling) are shared with the MP4 muxer.
//!
//! # Timestamps
//!
//! Like the MP4 writer, timestamps arrive in [`MEDIA_TIMESCALE`] ticks
//! (90 kHz). They are converted to Matroska timecodes in milliseconds
//! (`TimestampScale` of 1 000 000 ns). Block timestamps store presentation
//! timestamps; frames fed via `write_video_sample_with_dts` keep decode
//! (feed) order, and B-frames (pts != dts) are written as `BlockGroup`s with
//! a `ReferenceBlock` so decoders see the dependency.
//!
//! # WebM
//!
//! [`MkvContainer::WebM`] writes `DocType "webm"` and enforces the WebM
//! codec whitelist (VP9/AV1 video, Opus audio, no subtitles). Anything else
//! fails with [`MkvWriterError::UnsupportedForWebM`].

use std::fmt;
use std::io::{self, Write};

use bytes::Bytes;
use mkv_element::ClusterBlock;
use mkv_element::io::blocking_impl::WriteTo;
use mkv_element::prelude::*;

use crate::api::{AudioCodec, Metadata, VideoCodec};
use crate::codec::av1::extract_av1_config;
use crate::codec::h264::{annexb_to_avcc, default_avc_config, extract_avc_config};
use crate::codec::h265::{extract_hevc_config, hevc_annexb_to_hvcc};
use crate::codec::opus::is_valid_opus_packet;
use crate::codec::vp9::extract_vp9_config;

use super::mp4::{
    MEDIA_TIMESCALE, Mp4AudioTrack, Mp4SubtitleTrack, Mp4VideoTrack, VideoConfig, adts_to_raw,
    av1c_payload, avcc_payload, hvcc_payload,
};

/// Matroska timestamp scale in nanoseconds (1 ms timecodes).
pub const MKV_TIMESTAMP_SCALE_NS: u64 = 1_000_000;

/// Track number assigned to the video track.
const VIDEO_TRACK_NUMBER: u64 = 1;
/// Track number assigned to the audio track.
const AUDIO_TRACK_NUMBER: u64 = 2;
/// Track number assigned to the subtitle track.
const SUBTITLE_TRACK_NUMBER: u64 = 3;

/// Maximum span of a single Cluster in milliseconds.
///
/// Keeps clusters seek-friendly (~5 s, matching common muxers).
const MAX_CLUSTER_SPAN_MS: u64 = 5000;
/// Maximum number of blocks per Cluster (bounds per-cluster memory).
const MAX_CLUSTER_BLOCKS: usize = 1000;

/// Opus pre-skip in samples at 48 kHz (matches the MP4 `dOps` default).
const OPUS_PRE_SKIP: u16 = 312;
/// Opus seek pre-roll in nanoseconds (80 ms, per Matroska Opus mapping).
const OPUS_SEEK_PRE_ROLL_NS: u64 = 80_000_000;

/// Container flavour written by [`MkvWriter::finalize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MkvContainer {
    /// Matroska (`.mkv`): all muxfin codecs supported.
    #[default]
    Matroska,
    /// WebM (`.webm`): VP9/AV1 video and Opus audio only.
    WebM,
}

impl MkvContainer {
    /// EBML `DocType` string for this container.
    pub fn doc_type(self) -> &'static str {
        match self {
            MkvContainer::Matroska => "matroska",
            MkvContainer::WebM => "webm",
        }
    }

    /// Whether the given video codec is allowed in this container.
    pub fn supports_video(self, codec: VideoCodec) -> bool {
        match self {
            MkvContainer::Matroska => true,
            MkvContainer::WebM => matches!(codec, VideoCodec::Vp9 | VideoCodec::Av1),
        }
    }

    /// Whether the given audio codec is allowed in this container.
    pub fn supports_audio(self, codec: AudioCodec) -> bool {
        match self {
            MkvContainer::Matroska => !matches!(codec, AudioCodec::None),
            MkvContainer::WebM => matches!(codec, AudioCodec::Opus),
        }
    }
}

impl fmt::Display for MkvContainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MkvContainer::Matroska => write!(f, "Matroska"),
            MkvContainer::WebM => write!(f, "WebM"),
        }
    }
}

/// Errors produced while queuing samples or finalising the file.
#[derive(Debug)]
pub enum MkvWriterError {
    /// Video frames must have strictly increasing decode timestamps.
    NonIncreasingTimestamp,
    /// The first frame must be a keyframe containing codec configuration.
    FirstFrameMustBeKeyframe,
    /// The first keyframe must include SPS and PPS NAL units.
    FirstFrameMissingSpsPps,
    /// The first AV1 keyframe must include a Sequence Header OBU.
    FirstFrameMissingSequenceHeader,
    /// The first VP9 keyframe must include valid frame header parameters.
    FirstFrameMissingVp9Config,
    /// Audio sample is not a valid ADTS frame (detailed diagnostics).
    InvalidAdtsDetailed(Box<super::mp4::AdtsValidationError>),
    /// Audio sample is not a valid Opus packet.
    InvalidOpusPacket,
    /// Audio track is not enabled on this writer.
    AudioNotEnabled,
    /// Subtitle track is not enabled on this writer.
    SubtitleNotEnabled,
    /// Computed sample duration overflowed.
    DurationOverflow,
    /// The writer has already been finalised.
    AlreadyFinalized,
    /// Codec is not allowed in the WebM container.
    UnsupportedForWebM {
        /// Human-readable codec description.
        codec: String,
        /// Why it is rejected.
        reason: String,
    },
    /// EBML encoding failed (from the external `mkv-element` crate).
    Encode(String),
    /// Low-level IO error.
    Io(std::io::Error),
}

impl fmt::Display for MkvWriterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MkvWriterError::NonIncreasingTimestamp => write!(f, "timestamps must grow"),
            MkvWriterError::FirstFrameMustBeKeyframe => {
                write!(f, "first frame must be a keyframe")
            }
            MkvWriterError::FirstFrameMissingSpsPps => {
                write!(f, "first frame must contain SPS/PPS")
            }
            MkvWriterError::FirstFrameMissingSequenceHeader => {
                write!(f, "first AV1 frame must contain Sequence Header OBU")
            }
            MkvWriterError::FirstFrameMissingVp9Config => {
                write!(f, "first VP9 frame must contain valid frame header")
            }
            MkvWriterError::InvalidAdtsDetailed(err) => write!(f, "{}", err),
            MkvWriterError::InvalidOpusPacket => write!(f, "invalid Opus packet"),
            MkvWriterError::AudioNotEnabled => write!(f, "audio track not enabled"),
            MkvWriterError::SubtitleNotEnabled => write!(f, "subtitle track not enabled"),
            MkvWriterError::DurationOverflow => write!(f, "sample duration overflow"),
            MkvWriterError::AlreadyFinalized => write!(f, "writer already finalised"),
            MkvWriterError::UnsupportedForWebM { codec, reason } => {
                write!(f, "codec {} is not allowed in WebM: {}", codec, reason)
            }
            MkvWriterError::Encode(err) => write!(f, "EBML encode error: {}", err),
            MkvWriterError::Io(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl std::error::Error for MkvWriterError {}

impl From<mkv_element::Error> for MkvWriterError {
    fn from(err: mkv_element::Error) -> Self {
        match err {
            mkv_element::Error::Io(io_err) => MkvWriterError::Io(io_err),
            other => MkvWriterError::Encode(other.to_string()),
        }
    }
}

/// A single queued sample (shared across video/audio/subtitle).
struct MkvSample {
    /// Presentation timestamp in milliseconds.
    pts_ms: u64,
    /// Decode timestamp in milliseconds (differs from `pts_ms` for B-frames).
    dts_ms: u64,
    /// Converted payload bytes (AVCC/HVCC, raw AAC, raw Opus, UTF-8, ...).
    data: Vec<u8>,
    /// Whether this sample is a keyframe (video) / has no dependencies.
    is_keyframe: bool,
    /// Block duration in milliseconds, if known yet.
    duration_ms: Option<u64>,
}

/// Buffered Matroska writer. See the [module](self) documentation.
pub struct MkvWriter<Writer> {
    writer: Writer,
    video_codec: Option<VideoCodec>,
    video_config: Option<VideoConfig>,
    video_samples: Vec<MkvSample>,
    video_prev_dts: Option<u64>,
    video_prev_dts_ms: Option<u64>,
    video_last_delta_ms: Option<u64>,
    audio_track: Option<Mp4AudioTrack>,
    audio_samples: Vec<MkvSample>,
    audio_prev_ms: Option<u64>,
    audio_last_delta_ms: Option<u64>,
    /// First ADTS header seen (for AAC `AudioSpecificConfig` extraction).
    first_adts_header: Option<[u8; 4]>,
    subtitle_track: Option<Mp4SubtitleTrack>,
    subtitle_samples: Vec<MkvSample>,
    subtitle_prev_ms: Option<u64>,
    finalized: bool,
    bytes_written: u64,
}

impl<Writer: Write> MkvWriter<Writer> {
    /// Wraps the provided writer for Matroska/WebM container output.
    pub fn new(writer: Writer) -> Self {
        Self {
            writer,
            video_codec: None,
            video_config: None,
            video_samples: Vec::new(),
            video_prev_dts: None,
            video_prev_dts_ms: None,
            video_last_delta_ms: None,
            audio_track: None,
            audio_samples: Vec::new(),
            audio_prev_ms: None,
            audio_last_delta_ms: None,
            first_adts_header: None,
            subtitle_track: None,
            subtitle_samples: Vec::new(),
            subtitle_prev_ms: None,
            finalized: bool::default(),
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

    pub(crate) fn max_end_pts(&self) -> Option<u64> {
        fn track_end(samples: &[MkvSample], last_delta_ms: Option<u64>) -> Option<u64> {
            let last = samples.last()?;
            let end_ms = last.dts_ms.checked_add(last_delta_ms.unwrap_or(0))?;
            end_ms
                .checked_mul(u64::from(MEDIA_TIMESCALE))?
                .checked_div(1000)
        }

        let video_end = track_end(&self.video_samples, self.video_last_delta_ms);
        let audio_end = track_end(&self.audio_samples, self.audio_last_delta_ms);
        [video_end, audio_end].into_iter().flatten().max()
    }

    fn write_counted(&mut self, buf: &[u8]) -> io::Result<()> {
        self.bytes_written = self.bytes_written.saturating_add(buf.len() as u64);
        self.writer.write_all(buf)
    }

    pub fn enable_video(&mut self, codec: VideoCodec) {
        self.video_codec = Some(codec);
    }

    pub fn enable_audio(&mut self, track: Mp4AudioTrack) {
        self.audio_track = Some(track);
    }

    pub fn enable_subtitle(&mut self, track: Mp4SubtitleTrack) {
        self.subtitle_track = Some(track);
    }

    /// Queues a video sample for later Cluster emission.
    /// For backward compatibility, dts is assumed equal to pts.
    pub fn write_video_sample(
        &mut self,
        pts: u64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MkvWriterError> {
        self.write_video_sample_with_dts(pts, pts, data, is_keyframe)
    }

    /// Queues a video sample with explicit decode timestamp for B-frames.
    pub fn write_video_sample_with_dts(
        &mut self,
        pts: u64,
        dts: u64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), MkvWriterError> {
        if self.finalized {
            return Err(MkvWriterError::AlreadyFinalized);
        }
        let video_codec = self.video_codec.ok_or_else(|| {
            MkvWriterError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "video track not enabled: enable_video() must have been called before write_video_sample()",
            ))
        })?;

        let pts_ms = ticks_to_ms(pts);
        let dts_ms = ticks_to_ms(dts);

        // DTS must be monotonically increasing (decode order, tick-exact so
        // millisecond quantisation cannot mask a stalled stream).
        if let Some(prev) = self.video_prev_dts {
            if dts <= prev {
                return Err(MkvWriterError::NonIncreasingTimestamp);
            }
            let delta_ms = dts_ms
                .saturating_sub(self.video_prev_dts_ms.unwrap_or(dts_ms))
                .max(1);
            if let Some(last) = self.video_samples.last_mut() {
                last.duration_ms = Some(delta_ms);
            }
            self.video_last_delta_ms = Some(delta_ms);
        } else {
            if !is_keyframe {
                return Err(MkvWriterError::FirstFrameMustBeKeyframe);
            }
            let config = match video_codec {
                VideoCodec::H264 => extract_avc_config(data).map(VideoConfig::Avc),
                VideoCodec::H265 => extract_hevc_config(data).map(VideoConfig::Hevc),
                VideoCodec::Av1 => extract_av1_config(data).map(VideoConfig::Av1),
                VideoCodec::Vp9 => extract_vp9_config(data).map(VideoConfig::Vp9),
            };
            if config.is_none() {
                return Err(match video_codec {
                    VideoCodec::Av1 => MkvWriterError::FirstFrameMissingSequenceHeader,
                    VideoCodec::Vp9 => MkvWriterError::FirstFrameMissingVp9Config,
                    _ => MkvWriterError::FirstFrameMissingSpsPps,
                });
            }
            self.video_config = config;
        }

        // Matroska stores AVC/HEVC length-prefixed (same conversion as MP4);
        // AV1 OBUs and VP9 frames pass through as-is.
        let converted = match video_codec {
            VideoCodec::H264 => annexb_to_avcc(data),
            VideoCodec::H265 => hevc_annexb_to_hvcc(data),
            VideoCodec::Av1 => data.to_vec(),
            VideoCodec::Vp9 => data.to_vec(),
        };

        self.video_samples.push(MkvSample {
            pts_ms,
            dts_ms,
            data: converted,
            is_keyframe,
            duration_ms: None,
        });
        self.video_prev_dts = Some(dts);
        self.video_prev_dts_ms = Some(dts_ms);
        Ok(())
    }

    pub fn write_audio_sample(&mut self, pts: u64, data: &[u8]) -> Result<(), MkvWriterError> {
        if self.finalized {
            return Err(MkvWriterError::AlreadyFinalized);
        }
        let audio_track = self
            .audio_track
            .as_ref()
            .ok_or(MkvWriterError::AudioNotEnabled)?;

        let pts_ms = ticks_to_ms(pts);
        if let Some(prev) = self.audio_prev_ms {
            if pts_ms < prev {
                return Err(MkvWriterError::NonIncreasingTimestamp);
            }
            let delta_ms = pts_ms.saturating_sub(prev).max(1);
            if let Some(last) = self.audio_samples.last_mut() {
                last.duration_ms = Some(delta_ms);
            }
            self.audio_last_delta_ms = Some(delta_ms);
        }

        let sample_data = match audio_track.codec {
            AudioCodec::Aac(_) => {
                let raw = adts_to_raw(data)
                    .map_err(|e| MkvWriterError::InvalidAdtsDetailed(Box::new(e)))?;
                if self.first_adts_header.is_none() && data.len() >= 4 {
                    self.first_adts_header = Some([data[0], data[1], data[2], data[3]]);
                }
                raw.to_vec()
            }
            AudioCodec::Opus => {
                if !is_valid_opus_packet(data) {
                    return Err(MkvWriterError::InvalidOpusPacket);
                }
                data.to_vec()
            }
            AudioCodec::None => {
                return Err(MkvWriterError::AudioNotEnabled);
            }
        };

        self.audio_samples.push(MkvSample {
            pts_ms,
            dts_ms: pts_ms,
            data: sample_data,
            is_keyframe: false,
            duration_ms: None,
        });
        self.audio_prev_ms = Some(pts_ms);
        Ok(())
    }

    pub fn write_subtitle_sample(
        &mut self,
        pts: u64,
        duration: u32,
        data: &[u8],
    ) -> Result<(), MkvWriterError> {
        if self.finalized {
            return Err(MkvWriterError::AlreadyFinalized);
        }
        if self.subtitle_track.is_none() {
            return Err(MkvWriterError::SubtitleNotEnabled);
        }
        let pts_ms = ticks_to_ms(pts);
        let duration_ms = ticks_to_ms(u64::from(duration)).max(1);
        if let Some(prev) = self.subtitle_prev_ms
            && pts_ms < prev
        {
            return Err(MkvWriterError::NonIncreasingTimestamp);
        }
        if data.is_empty() {
            return Err(MkvWriterError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "subtitle sample is empty",
            )));
        }

        self.subtitle_samples.push(MkvSample {
            pts_ms,
            dts_ms: pts_ms,
            data: data.to_vec(),
            is_keyframe: false,
            duration_ms: Some(duration_ms),
        });
        self.subtitle_prev_ms = Some(pts_ms);
        Ok(())
    }

    /// Finalises the file: writes the EBML header and the Segment.
    pub fn finalize(
        &mut self,
        video: Option<&Mp4VideoTrack>,
        metadata: Option<&Metadata>,
        container: MkvContainer,
    ) -> io::Result<()> {
        if self.finalized {
            return Err(io::Error::other("mkv writer already finalised"));
        }
        self.finalized = true;

        self.finalize_inner(video, metadata, container)
            .map_err(|e| match e {
                MkvWriterError::Io(io_err) => io_err,
                other => io::Error::other(other.to_string()),
            })
    }

    fn finalize_inner(
        &mut self,
        video: Option<&Mp4VideoTrack>,
        metadata: Option<&Metadata>,
        container: MkvContainer,
    ) -> Result<(), MkvWriterError> {
        // Resolve the effective video configuration (same fallback as MP4:
        // an H.264 track with no samples still yields a default avcC).
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

        let has_video = video_config.is_some() && video.is_some();
        let has_audio = self.audio_track.is_some();
        let has_subtitle = self.subtitle_track.is_some();

        if video_config.is_some() && video.is_none() {
            return Err(MkvWriterError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "video config extracted but no video track provided",
            )));
        }

        if !has_video && !has_audio && !has_subtitle {
            return Err(MkvWriterError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "no tracks to mux: enable at least one track and write samples",
            )));
        }

        // Enforce the WebM codec whitelist early with actionable errors.
        if container == MkvContainer::WebM {
            if let (Some(codec), true) = (self.video_codec, has_video)
                && !container.supports_video(codec)
            {
                return Err(MkvWriterError::UnsupportedForWebM {
                    codec: codec.to_string(),
                    reason:
                        "WebM only supports VP9 and AV1 video; use Matroska (.mkv) for H.264/H.265"
                            .to_string(),
                });
            }
            if let Some(track) = &self.audio_track
                && has_audio
                && !container.supports_audio(track.codec)
            {
                return Err(MkvWriterError::UnsupportedForWebM {
                    codec: track.codec.to_string(),
                    reason: "WebM only supports Opus audio; use Matroska (.mkv) for AAC"
                        .to_string(),
                });
            }
            if has_subtitle {
                return Err(MkvWriterError::UnsupportedForWebM {
                    codec: "mov_text".to_string(),
                    reason: "WebM subtitles require WebVTT; use Matroska (.mkv) for text subtitles"
                        .to_string(),
                });
            }
        }

        let language = metadata
            .and_then(|m| m.language.as_deref())
            .unwrap_or("und");

        // ---- Tracks -------------------------------------------------------
        let mut entries = Vec::new();
        if let (Some(video_dims), Some(video_config)) = (video, video_config.as_ref())
            && has_video
        {
            let codec = self.video_codec.expect("video codec set with config");
            entries.push(build_video_track(
                video_dims,
                codec,
                video_config,
                language,
            )?);
        }
        if let Some(track) = &self.audio_track
            && has_audio
        {
            entries.push(build_audio_track(track, self.first_adts_header, language)?);
        }
        if let Some(track) = &self.subtitle_track
            && has_subtitle
        {
            entries.push(build_subtitle_track(track, language));
        }

        let tracks = Tracks {
            crc32: None,
            void: None,
            track_entry: entries,
        };

        // ---- Clusters (interleaved in decode order) ------------------------
        let clusters = build_clusters(
            &self.video_samples,
            &self.audio_samples,
            &self.subtitle_samples,
        )?;

        // ---- Info ----------------------------------------------------------
        // Duration: latest sample end across all tracks (informational).
        let end_ms = [
            track_end_ms(&self.video_samples, self.video_last_delta_ms),
            track_end_ms(&self.audio_samples, self.audio_last_delta_ms),
            self.subtitle_samples
                .iter()
                .map(|s| s.pts_ms.saturating_add(s.duration_ms.unwrap_or(0)))
                .max(),
        ]
        .into_iter()
        .flatten()
        .max();
        let info = Info {
            crc32: None,
            void: None,
            segment_uuid: None,
            segment_filename: None,
            prev_uuid: None,
            prev_filename: None,
            next_uuid: None,
            next_filename: None,
            segment_family: Vec::new(),
            chapter_translate: Vec::new(),
            timestamp_scale: TimestampScale(MKV_TIMESTAMP_SCALE_NS),
            duration: end_ms.map(|ms| Duration(ms as f64)),
            date_utc: metadata
                .and_then(|m| m.creation_time)
                .map(unix_to_ebml_date)
                .map(DateUtc),
            title: metadata.and_then(|m| m.title.clone()).map(Title),
            muxing_app: MuxingApp(format!("muxfin-{}", env!("CARGO_PKG_VERSION"))),
            writing_app: WritingApp(format!("muxfin-{}", env!("CARGO_PKG_VERSION"))),
        };

        // ---- Cues (seek index over video-keyframe clusters) ----------------
        // Serialise the front matter first so cue positions are exact
        // segment-relative offsets.
        let info_bytes = encode_element(&info)?;
        let tracks_bytes = encode_element(&tracks)?;
        let mut cluster_blobs: Vec<Vec<u8>> = Vec::with_capacity(clusters.len());
        for cluster in &clusters {
            cluster_blobs.push(encode_element(cluster)?);
        }
        let base = info_bytes.len() as u64 + tracks_bytes.len() as u64;
        let mut offset = base;
        let mut cue_points = Vec::new();
        for (cluster, blob) in clusters.iter().zip(cluster_blobs.iter()) {
            if cluster_starts_with_video_keyframe(cluster) {
                cue_points.push(CuePoint {
                    crc32: None,
                    void: None,
                    cue_time: CueTime(*cluster.timestamp),
                    cue_track_positions: vec![CueTrackPositions {
                        crc32: None,
                        void: None,
                        cue_track: CueTrack(VIDEO_TRACK_NUMBER),
                        cue_cluster_position: CueClusterPosition(offset),
                        cue_relative_position: None,
                        cue_duration: None,
                        cue_block_number: None,
                        cue_codec_state: CueCodecState(0),
                        cue_reference: Vec::new(),
                    }],
                });
            }
            offset += blob.len() as u64;
        }
        let cues = if cue_points.is_empty() {
            None
        } else {
            Some(Cues {
                crc32: None,
                void: None,
                cue_point: cue_points,
            })
        };

        let segment = Segment {
            crc32: None,
            void: None,
            seek_head: Vec::new(),
            info,
            cluster: clusters,
            tracks: Some(tracks),
            cues,
            attachments: None,
            chapters: None,
            tags: Vec::new(),
        };

        let ebml = Ebml {
            crc32: None,
            void: None,
            ebml_version: None,
            ebml_read_version: None,
            ebml_max_id_length: EbmlMaxIdLength(4),
            ebml_max_size_length: EbmlMaxSizeLength(8),
            doc_type: Some(DocType(container.doc_type().to_string())),
            doc_type_version: Some(DocTypeVersion(4)),
            doc_type_read_version: Some(DocTypeReadVersion(2)),
        };

        let mut out = Vec::new();
        ebml.write_to(&mut out)?;
        segment.write_to(&mut out)?;
        self.write_counted(&out).map_err(MkvWriterError::Io)?;
        Ok(())
    }
}

/// 90 kHz ticks to whole milliseconds.
fn ticks_to_ms(ticks: u64) -> u64 {
    ticks.saturating_mul(1000) / u64::from(MEDIA_TIMESCALE)
}

/// Unix seconds to EBML `DateUTC` (nanoseconds since 2001-01-01T00:00:00Z).
fn unix_to_ebml_date(unix_secs: u64) -> i64 {
    const UNIX_TO_EBML_EPOCH_SECS: i64 = 978_307_200;
    let nanos = (unix_secs as i64 - UNIX_TO_EBML_EPOCH_SECS) as i128 * 1_000_000_000i128;
    nanos.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Encode one element (header + body) to bytes.
fn encode_element<T: WriteTo>(element: &T) -> Result<Vec<u8>, MkvWriterError> {
    let mut buf = Vec::new();
    element.write_to(&mut buf)?;
    Ok(buf)
}

/// Minimal EBML vint encoder for track numbers.
fn encode_vint(value: u64) -> Vec<u8> {
    let size = VInt64::encode_size(value);
    VInt64::new(value).as_encoded().to_be_bytes()[8 - size..].to_vec()
}

/// Build a `SimpleBlock`/`Block` payload: track vint + i16 timecode + flags + data.
fn block_payload(track_number: u64, relative_ms: i64, keyframe: bool, data: &[u8]) -> Vec<u8> {
    let mut payload = encode_vint(track_number);
    payload.extend_from_slice(&(relative_ms as i16).to_be_bytes());
    payload.push(if keyframe { 0x80 } else { 0x00 });
    payload.extend_from_slice(data);
    payload
}

/// Matroska `CodecID` for a video codec.
fn video_codec_id(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::H264 => "V_MPEG4/ISO/AVC",
        VideoCodec::H265 => "V_MPEGH/ISO/HEVC",
        VideoCodec::Av1 => "V_AV1",
        VideoCodec::Vp9 => "V_VP9",
    }
}

/// `CodecPrivate` bytes for a video configuration, if the mapping needs any.
///
/// AVC/HEVC/AV1 reuse the MP4 `avcC`/`hvcC`/`av1C` payloads; VP9 (`V_VP9`)
/// carries no private data.
fn video_codec_private(config: &VideoConfig) -> Option<Vec<u8>> {
    match config {
        VideoConfig::Avc(avc) => Some(avcc_payload(avc)),
        VideoConfig::Hevc(hevc) => Some(hvcc_payload(hevc)),
        VideoConfig::Av1(av1) => Some(av1c_payload(av1)),
        VideoConfig::Vp9(_) => None,
    }
}

fn base_track(
    number: u64,
    track_type: u64,
    codec_id: &str,
    codec_private: Option<Vec<u8>>,
    language: &str,
) -> TrackEntry {
    TrackEntry {
        crc32: None,
        void: None,
        track_number: TrackNumber(number),
        track_uid: TrackUid(number),
        track_type: TrackType(track_type),
        flag_enabled: FlagEnabled(1),
        flag_default: FlagDefault(1),
        flag_forced: FlagForced(0),
        flag_hearing_impaired: None,
        flag_visual_impaired: None,
        flag_text_descriptions: None,
        flag_original: None,
        flag_commentary: None,
        flag_lacing: FlagLacing(0),
        default_duration: None,
        default_decoded_field_duration: None,
        max_block_addition_id: MaxBlockAdditionId(0),
        block_addition_mapping: Vec::new(),
        name: None,
        language: Language(language.to_string()),
        language_bcp47: None,
        codec_id: CodecId(codec_id.to_string()),
        codec_private: codec_private.map(Bytes::from).map(CodecPrivate),
        codec_name: None,
        codec_delay: CodecDelay(0),
        seek_pre_roll: SeekPreRoll(0),
        track_translate: Vec::new(),
        video: None,
        audio: None,
        track_operation: None,
        content_encodings: None,
    }
}

fn build_video_track(
    dims: &Mp4VideoTrack,
    codec: VideoCodec,
    config: &VideoConfig,
    language: &str,
) -> Result<TrackEntry, MkvWriterError> {
    let mut entry = base_track(
        VIDEO_TRACK_NUMBER,
        1,
        video_codec_id(codec),
        video_codec_private(config),
        language,
    );
    entry.video = Some(Video {
        crc32: None,
        void: None,
        flag_interlaced: FlagInterlaced(0),
        field_order: FieldOrder(0),
        stereo_mode: StereoMode(0),
        alpha_mode: AlphaMode(0),
        pixel_width: PixelWidth(u64::from(dims.width)),
        pixel_height: PixelHeight(u64::from(dims.height)),
        pixel_crop_bottom: PixelCropBottom(0),
        pixel_crop_top: PixelCropTop(0),
        pixel_crop_left: PixelCropLeft(0),
        pixel_crop_right: PixelCropRight(0),
        display_width: None,
        display_height: None,
        display_unit: DisplayUnit(0),
        uncompressed_fourcc: None,
        colour: None,
        projection: None,
    });
    Ok(entry)
}

/// AudioSpecificConfig derived from an ADTS header (ISO/IEC 14496-3).
///
/// Layout: `AAAAABBB BCCCCDDD` where A = audio object type, B = sample rate
/// index, C = channel configuration.
fn aac_audio_specific_config(adts_header: [u8; 4]) -> [u8; 2] {
    let profile = (adts_header[2] >> 6) & 0x03;
    let sample_rate_idx = (adts_header[2] >> 2) & 0x0F;
    let channels = ((adts_header[2] & 0x01) << 2) | ((adts_header[3] >> 6) & 0x03);
    let audio_object_type = profile + 1;
    [
        (audio_object_type << 3) | (sample_rate_idx >> 1),
        ((sample_rate_idx & 0x01) << 7) | (channels << 3),
    ]
}

/// Default ASC: AAC-LC, 44.1 kHz, stereo (`0x12 0x10`).
const DEFAULT_AAC_ASC: [u8; 2] = [0x12, 0x10];

/// OpusHead `CodecPrivate` for `A_OPUS` (RFC 7845 §5.1, 19 bytes, family 0).
fn opus_head(channels: u8) -> Vec<u8> {
    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead");
    head.push(1); // version
    head.push(channels);
    head.extend_from_slice(&OPUS_PRE_SKIP.to_le_bytes());
    head.extend_from_slice(&48000u32.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes()); // output gain
    head.push(0); // channel mapping family
    head
}

fn build_audio_track(
    track: &Mp4AudioTrack,
    first_adts_header: Option<[u8; 4]>,
    language: &str,
) -> Result<TrackEntry, MkvWriterError> {
    let (codec_id, codec_private, codec_delay, seek_pre_roll) = match track.codec {
        AudioCodec::Aac(_) => {
            let asc = first_adts_header
                .map(aac_audio_specific_config)
                .unwrap_or(DEFAULT_AAC_ASC);
            ("A_AAC", Some(asc.to_vec()), 0, 0)
        }
        AudioCodec::Opus => (
            "A_OPUS",
            Some(opus_head(track.channels.min(8) as u8)),
            u64::from(OPUS_PRE_SKIP) * 1_000_000_000 / 48_000,
            OPUS_SEEK_PRE_ROLL_NS,
        ),
        AudioCodec::None => {
            return Err(MkvWriterError::AudioNotEnabled);
        }
    };

    let mut entry = base_track(AUDIO_TRACK_NUMBER, 2, codec_id, codec_private, language);
    entry.codec_delay = CodecDelay(codec_delay);
    entry.seek_pre_roll = SeekPreRoll(seek_pre_roll);
    entry.audio = Some(Audio {
        crc32: None,
        void: None,
        sampling_frequency: SamplingFrequency(f64::from(track.sample_rate)),
        output_sampling_frequency: None,
        channels: Channels(u64::from(track.channels)),
        bit_depth: None,
        emphasis: Emphasis(0),
    });
    Ok(entry)
}

fn build_subtitle_track(track: &Mp4SubtitleTrack, language: &str) -> TrackEntry {
    let track_language = track.language.as_deref().unwrap_or(language);
    base_track(
        SUBTITLE_TRACK_NUMBER,
        17,
        "S_TEXT/UTF8",
        None,
        track_language,
    )
}

/// One block ready for Cluster assignment.
struct PendingBlock {
    /// Decode timestamp (merge key: keeps B-frame decode order).
    dts_ms: u64,
    /// Presentation timestamp (Block timecode).
    pts_ms: u64,
    track_number: u64,
    is_keyframe: bool,
    is_subtitle: bool,
    data: Vec<u8>,
    duration_ms: Option<u64>,
    /// Original insertion order (stable tie-break).
    order: usize,
}

/// Interleave video/audio/subtitle samples in decode order.
fn collect_blocks(
    video: &[MkvSample],
    audio: &[MkvSample],
    subtitles: &[MkvSample],
) -> Vec<PendingBlock> {
    let mut blocks = Vec::new();
    let mut order = 0usize;
    for s in video {
        blocks.push(PendingBlock {
            dts_ms: s.dts_ms,
            pts_ms: s.pts_ms,
            track_number: VIDEO_TRACK_NUMBER,
            is_keyframe: s.is_keyframe,
            is_subtitle: false,
            data: s.data.clone(),
            duration_ms: s.duration_ms,
            order,
        });
        order += 1;
    }
    for s in audio {
        blocks.push(PendingBlock {
            dts_ms: s.dts_ms,
            pts_ms: s.pts_ms,
            track_number: AUDIO_TRACK_NUMBER,
            is_keyframe: false,
            is_subtitle: false,
            data: s.data.clone(),
            duration_ms: s.duration_ms,
            order,
        });
        order += 1;
    }
    for s in subtitles {
        blocks.push(PendingBlock {
            dts_ms: s.dts_ms,
            pts_ms: s.pts_ms,
            track_number: SUBTITLE_TRACK_NUMBER,
            is_keyframe: false,
            is_subtitle: true,
            data: s.data.clone(),
            duration_ms: s.duration_ms,
            order,
        });
        order += 1;
    }
    blocks.sort_by_key(|a| (a.dts_ms, a.order));
    blocks
}

/// Whether this block needs a `BlockGroup` (B-frame dependency or duration).
fn needs_block_group(block: &PendingBlock) -> bool {
    // Subtitles always need durations; B-frames (pts != dts) and all
    // non-keyframe video use BlockGroups so the reference is explicit.
    block.is_subtitle
        || block.pts_ms != block.dts_ms
        || (block.track_number == VIDEO_TRACK_NUMBER && !block.is_keyframe)
}

fn pending_to_cluster_block(
    block: &PendingBlock,
    cluster_base_ms: u64,
) -> Result<ClusterBlock, MkvWriterError> {
    let relative = (block.pts_ms as i64)
        .checked_sub(cluster_base_ms as i64)
        .ok_or(MkvWriterError::DurationOverflow)?;
    if relative < i16::MIN as i64 || relative > i16::MAX as i64 {
        return Err(MkvWriterError::DurationOverflow);
    }

    if !needs_block_group(block) {
        let payload = block_payload(block.track_number, relative, block.is_keyframe, &block.data);
        return Ok(ClusterBlock::Simple(SimpleBlock(Bytes::from(payload))));
    }

    let payload = block_payload(block.track_number, relative, block.is_keyframe, &block.data);
    Ok(ClusterBlock::Group(BlockGroup {
        crc32: None,
        void: None,
        block: Block(Bytes::from(payload)),
        block_additions: None,
        block_duration: block.duration_ms.map(BlockDuration),
        reference_priority: ReferencePriority(0),
        reference_block: if block.is_keyframe {
            Vec::new()
        } else {
            // Value 0: undecodable alone, exact reference unknown (RFC 9559 §5.1.3.5.7).
            vec![ReferenceBlock(0)]
        },
        codec_state: None,
        discard_padding: None,
    }))
}

/// Group interleaved blocks into Clusters.
///
/// A new Cluster starts at each video keyframe (after the first block),
/// when the 5 s / 1000-block budget is exceeded, or when a relative
/// timecode would overflow `i16`.
fn build_clusters(
    video: &[MkvSample],
    audio: &[MkvSample],
    subtitles: &[MkvSample],
) -> Result<Vec<Cluster>, MkvWriterError> {
    let blocks = collect_blocks(video, audio, subtitles);
    let mut clusters = Vec::new();
    let mut current: Vec<&PendingBlock> = Vec::new();
    let mut base_ms: u64 = 0;

    // Encode the pending blocks as one Cluster at `base_ms`.
    let flush = |current: &mut Vec<&PendingBlock>,
                 base_ms: u64,
                 clusters: &mut Vec<Cluster>|
     -> Result<(), MkvWriterError> {
        if current.is_empty() {
            return Ok(());
        }
        let mut cluster_blocks = Vec::with_capacity(current.len());
        for block in current.drain(..) {
            cluster_blocks.push(pending_to_cluster_block(block, base_ms)?);
        }
        clusters.push(Cluster {
            crc32: None,
            void: None,
            timestamp: Timestamp(base_ms),
            position: None,
            prev_size: None,
            blocks: cluster_blocks,
        });
        Ok(())
    };

    for block in &blocks {
        let need_split = if current.is_empty() {
            base_ms = block.pts_ms;
            false
        } else {
            let video_keyframe_split =
                block.track_number == VIDEO_TRACK_NUMBER && block.is_keyframe;
            let span_split = block.pts_ms.saturating_sub(base_ms) > MAX_CLUSTER_SPAN_MS;
            let count_split = current.len() >= MAX_CLUSTER_BLOCKS;
            let overflow_split = (block.pts_ms as i64) - (base_ms as i64) > i16::MAX as i64
                || (base_ms as i64) - (block.pts_ms as i64) > -(i16::MIN as i64);
            video_keyframe_split || span_split || count_split || overflow_split
        };
        if need_split {
            flush(&mut current, base_ms, &mut clusters)?;
            base_ms = block.pts_ms;
        }
        current.push(block);
    }
    flush(&mut current, base_ms, &mut clusters)?;
    Ok(clusters)
}

/// Latest sample end (decode timestamp + known delta) in milliseconds.
fn track_end_ms(samples: &[MkvSample], last_delta_ms: Option<u64>) -> Option<u64> {
    let last = samples.last()?;
    Some(last.dts_ms.saturating_add(last_delta_ms.unwrap_or(0)))
}

/// Whether a cluster begins with a video keyframe (cue eligibility).
fn cluster_starts_with_video_keyframe(cluster: &Cluster) -> bool {
    // Keyframes are SimpleBlocks with the keyframe flag on track 1, or
    // BlockGroups on track 1 without references. Checking the first block
    // of each cluster is sufficient: clusters always open with the block
    // that triggered the split (a video keyframe) or the first sample.
    let Some(first) = cluster.blocks.first() else {
        return false;
    };
    match first {
        ClusterBlock::Simple(block) => {
            let bytes: &[u8] = &block.0;
            is_track1_keyframe_payload(bytes)
        }
        ClusterBlock::Group(group) => {
            let bytes: &[u8] = &group.block.0;
            is_track1_keyframe_payload(bytes) && group.reference_block.is_empty()
        }
    }
}

/// Inspect a `Block` payload for track 1 + keyframe flag.
fn is_track1_keyframe_payload(payload: &[u8]) -> bool {
    // Track vint (1 byte for our track numbers) + i16 timecode + flags.
    if payload.len() < 4 {
        return false;
    }
    let track_ok = payload[0] == (0x80 | VIDEO_TRACK_NUMBER as u8);
    let keyframe = payload[3] & 0x80 != 0;
    track_ok && keyframe
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_to_ms_converts_90khz() {
        assert_eq!(ticks_to_ms(90_000), 1000);
        assert_eq!(ticks_to_ms(45_000), 500);
        assert_eq!(ticks_to_ms(0), 0);
    }

    #[test]
    fn asc_from_adts_header() {
        // ADTS header for AAC-LC, 44.1 kHz, stereo:
        // profile=1 (LC), sf_idx=4 (44100), channels=2
        let header = [0xFF, 0xF1, 0x50, 0x80];
        assert_eq!(aac_audio_specific_config(header), [0x12, 0x10]);
    }

    #[test]
    fn opus_head_layout() {
        let head = opus_head(2);
        assert_eq!(head.len(), 19);
        assert_eq!(&head[..8], b"OpusHead");
        assert_eq!(head[8], 1); // version
        assert_eq!(head[9], 2); // channels
        assert_eq!(u16::from_le_bytes([head[10], head[11]]), OPUS_PRE_SKIP);
        assert_eq!(
            u32::from_le_bytes([head[12], head[13], head[14], head[15]]),
            48000
        );
    }

    #[test]
    fn unix_to_ebml_date_epoch() {
        // 2001-01-01T00:00:00Z == 978307200 unix
        assert_eq!(unix_to_ebml_date(978_307_200), 0);
        assert_eq!(unix_to_ebml_date(978_307_201), 1_000_000_000);
    }

    #[test]
    fn webm_whitelist() {
        assert!(MkvContainer::WebM.supports_video(VideoCodec::Vp9));
        assert!(MkvContainer::WebM.supports_video(VideoCodec::Av1));
        assert!(!MkvContainer::WebM.supports_video(VideoCodec::H264));
        assert!(!MkvContainer::WebM.supports_video(VideoCodec::H265));
        assert!(MkvContainer::WebM.supports_audio(AudioCodec::Opus));
        assert!(!MkvContainer::WebM.supports_audio(AudioCodec::Aac(crate::api::AacProfile::Lc)));
        assert!(MkvContainer::Matroska.supports_video(VideoCodec::H264));
        assert!(MkvContainer::Matroska.supports_video(VideoCodec::H265));
    }
}
