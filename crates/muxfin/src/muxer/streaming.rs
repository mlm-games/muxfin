//! Seekable-streaming MP4 writer (§5).
//!
//! The buffered [`Mp4Writer`](super::mp4::Mp4Writer) retains every sample's
//! bytes until `finish`, so peak RAM is O(media). This writer instead streams
//! sample bytes to the output as they arrive and retains only per-sample
//! metadata (timestamps, size, file offset, sync flag), so peak RAM is
//! O(metadata) — roughly 40 bytes per sample regardless of sample size.
//!
//! # Layout
//!
//! Because `moov` needs sample tables that are only complete at the end, the
//! streaming layout is `ftyp`, `mdat`, `moov` (progressive download):
//!
//! 1. `begin` writes `ftyp` plus an `mdat` header in fixed 16-byte
//!    largesize form with `largesize = 0` ("extends to end of file").
//! 2. Each accepted sample's converted bytes are appended immediately; its
//!    descriptor (with absolute file offset) is recorded.
//! 3. `finish` seeks back, patches the true `mdat` size into the reserved
//!    16-byte header (header length never changes, so recorded offsets stay
//!    valid), seeks to the end, and appends `moov`.
//!
//! Validation, bitstream conversion, decoder-config extraction/change
//! detection, and timestamp monotonicity are enforced by the shared
//! [`Mp4Writer`](super::mp4::Mp4Writer) `prepare_*` methods, so buffered and
//! streaming paths accept exactly the same inputs. Duration back-patching
//! (previous sample's duration is fixed when the next sample arrives) is
//! applied to the recorded descriptors identically.
//!
//! Use [`MuxerBuilder::build_streaming_seekable`](crate::api::MuxerBuilder::build_streaming_seekable)
//! (fast-start `moov`-before-`mdat` still requires the buffered [`Muxer`](crate::api::Muxer)).

use std::io::{self, Seek, SeekFrom, Write};

use crate::api::{Metadata, VideoCodec};
use crate::codec::h264::default_avc_config;
use crate::muxer::mp4::{
    Mp4AudioTrack, Mp4SubtitleTrack, Mp4VideoTrack, Mp4Writer, Mp4WriterError, SampleMeta,
    SampleTables, VideoConfig, build_audio_only_moov_box, build_ftyp_box, build_moov_box,
};

/// Fixed 16-byte `mdat` header length used for the whole session.
const STREAM_MDAT_HEADER_LEN: u64 = 16;

/// One streamed sample's retained descriptor (no media bytes).
#[derive(Debug, Clone)]
struct StreamEntry {
    meta: SampleMeta,
    offset: u64,
}

/// Seekable-streaming MP4 writer. See the [module](self) documentation.
///
/// The generic validation state lives in an inner `Mp4Writer<io::Sink>`; only
/// converted bytes reach the real output.
pub struct SeekableStreamingWriter<Writer> {
    writer: Writer,
    state: Mp4Writer<io::Sink>,
    video_track: Option<Mp4VideoTrack>,
    metadata: Option<Metadata>,
    mdat_header_pos: u64,
    cursor: u64,
    video_entries: Vec<StreamEntry>,
    audio_entries: Vec<StreamEntry>,
    subtitle_entries: Vec<StreamEntry>,
    finished: bool,
    bytes_written: u64,
}

impl<Writer: Write + Seek> SeekableStreamingWriter<Writer> {
    /// Start a streaming session: writes `ftyp` + placeholder `mdat` header.
    ///
    /// Tracks must already be enabled on `state` (via `enable_video` /
    /// `enable_audio` / `enable_subtitle`); `video_track` carries dimensions
    /// for `moov` construction.
    pub fn begin(
        mut writer: Writer,
        state: Mp4Writer<io::Sink>,
        video_track: Option<Mp4VideoTrack>,
        metadata: Option<Metadata>,
    ) -> io::Result<Self> {
        let ftyp = build_ftyp_box();
        let mdat_header_pos = ftyp.len() as u64;
        writer.write_all(&ftyp)?;
        // Fixed 16-byte largesize header: size=1, type=mdat, largesize=0
        // (0 = "extends to end of file" until patched at finish).
        let mut header = Vec::with_capacity(16);
        header.extend_from_slice(&1u32.to_be_bytes());
        header.extend_from_slice(b"mdat");
        header.extend_from_slice(&0u64.to_be_bytes());
        writer.write_all(&header)?;
        let cursor = mdat_header_pos + STREAM_MDAT_HEADER_LEN;
        let bytes_written = cursor;
        Ok(Self {
            writer,
            state,
            video_track,
            metadata,
            mdat_header_pos,
            cursor,
            video_entries: Vec::new(),
            audio_entries: Vec::new(),
            subtitle_entries: Vec::new(),
            finished: false,
            bytes_written,
        })
    }

    fn ensure_open(&self) -> Result<(), Mp4WriterError> {
        if self.finished {
            return Err(Mp4WriterError::AlreadyFinalized);
        }
        Ok(())
    }

    fn append_bytes(&mut self, bytes: &[u8]) -> io::Result<u64> {
        let offset = self.cursor;
        self.writer.write_all(bytes)?;
        self.cursor = self.cursor.saturating_add(bytes.len() as u64);
        self.bytes_written = self.bytes_written.saturating_add(bytes.len() as u64);
        Ok(offset)
    }

    /// Stream one video sample (decode order). Mirrors
    /// [`Mp4Writer::write_video_sample_with_dts`] semantics.
    pub fn write_video_sample_with_dts(
        &mut self,
        pts: u64,
        dts: u64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), Mp4WriterError> {
        self.ensure_open()?;
        let converted = self
            .state
            .prepare_video_sample(pts, dts, data, is_keyframe)?;
        let size = u32::try_from(converted.len()).map_err(|_| Mp4WriterError::DurationOverflow)?;
        if let Some(last) = self.video_entries.last_mut() {
            last.meta.duration = self.state.video_last_delta();
        }
        let offset = self.append_bytes(&converted).map_err(Mp4WriterError::Io)?;
        self.video_entries.push(StreamEntry {
            meta: SampleMeta {
                pts,
                dts,
                size,
                is_keyframe,
                duration: None,
            },
            offset,
        });
        Ok(())
    }

    /// Stream one video sample with `dts == pts`.
    #[allow(dead_code)]
    pub fn write_video_sample(
        &mut self,
        pts: u64,
        data: &[u8],
        is_keyframe: bool,
    ) -> Result<(), Mp4WriterError> {
        self.write_video_sample_with_dts(pts, pts, data, is_keyframe)
    }

    /// Stream one audio sample. Mirrors [`Mp4Writer::write_audio_sample`].
    pub fn write_audio_sample(&mut self, pts: u64, data: &[u8]) -> Result<(), Mp4WriterError> {
        self.ensure_open()?;
        let converted = self.state.prepare_audio_sample(pts, data)?;
        let size = u32::try_from(converted.len()).map_err(|_| Mp4WriterError::DurationOverflow)?;
        if let Some(last) = self.audio_entries.last_mut() {
            last.meta.duration = self.state.audio_last_delta();
        }
        let offset = self.append_bytes(&converted).map_err(Mp4WriterError::Io)?;
        self.audio_entries.push(StreamEntry {
            meta: SampleMeta {
                pts,
                dts: pts,
                size,
                is_keyframe: false,
                duration: None,
            },
            offset,
        });
        Ok(())
    }

    /// Stream one subtitle sample. Mirrors [`Mp4Writer::write_subtitle_sample`].
    pub fn write_subtitle_sample(
        &mut self,
        pts: u64,
        duration: u32,
        data: &[u8],
    ) -> Result<(), Mp4WriterError> {
        self.ensure_open()?;
        let (bytes, duration) = self.state.prepare_subtitle_sample(pts, duration, data)?;
        let size = u32::try_from(bytes.len()).map_err(|_| Mp4WriterError::DurationOverflow)?;
        let offset = self.append_bytes(&bytes).map_err(Mp4WriterError::Io)?;
        self.subtitle_entries.push(StreamEntry {
            meta: SampleMeta {
                pts,
                dts: pts,
                size,
                is_keyframe: false,
                duration: Some(duration),
            },
            offset,
        });
        Ok(())
    }

    /// Override the last video sample's duration (integer-time API, verdict §4).
    pub fn set_last_video_duration(&mut self, duration: u32) {
        if let Some(last) = self.video_entries.last_mut() {
            last.meta.duration = Some(duration);
        }
        self.state.set_last_video_duration(duration);
    }

    /// Override the last audio sample's duration (integer-time API, verdict §4).
    pub fn set_last_audio_duration(&mut self, duration: u32) {
        if let Some(last) = self.audio_entries.last_mut() {
            last.meta.duration = Some(duration);
        }
        self.state.set_last_audio_duration(duration);
    }

    fn metas(entries: &[StreamEntry]) -> Vec<SampleMeta> {
        entries.iter().map(|e| e.meta.clone()).collect()
    }

    fn offsets(entries: &[StreamEntry]) -> Vec<u64> {
        entries.iter().map(|e| e.offset).collect()
    }

    /// Finish: patch the `mdat` size, append `moov`, return byte count.
    ///
    /// `video_codec` selects the default-config fallback for the
    /// no-video-samples H.264 case, mirroring [`Mp4Writer::finalize`].
    pub fn finish(mut self, video_codec: Option<VideoCodec>) -> io::Result<u64> {
        if self.finished {
            return Err(io::Error::other("streaming writer already finished"));
        }
        self.finished = true;

        let video_config = self.state.video_config().cloned().or_else(|| {
            if self.video_entries.is_empty() {
                match video_codec {
                    Some(VideoCodec::H264) => Some(VideoConfig::Avc(default_avc_config())),
                    _ => None,
                }
            } else {
                None
            }
        });

        let mdat_payload: u64 = self
            .video_entries
            .iter()
            .chain(self.audio_entries.iter())
            .chain(self.subtitle_entries.iter())
            .map(|e| u64::from(e.meta.size))
            .fold(0u64, |a, b| a.saturating_add(b));
        let mdat_total = STREAM_MDAT_HEADER_LEN.saturating_add(mdat_payload);

        // Patch the reserved 16-byte header in place (length unchanged, so
        // recorded chunk offsets stay valid).
        self.writer.seek(SeekFrom::Start(self.mdat_header_pos))?;
        let mut header = Vec::with_capacity(16);
        header.extend_from_slice(&1u32.to_be_bytes());
        header.extend_from_slice(b"mdat");
        header.extend_from_slice(&mdat_total.to_be_bytes());
        self.writer.write_all(&header)?;
        self.writer.seek(SeekFrom::Start(self.cursor))?;

        if let Some(video_config) = video_config {
            let video = self.video_track.as_ref().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "video config extracted but no video track provided",
                )
            })?;
            let audio_present = self.state.audio_track_ref().is_some();
            let subtitle_present = self.state.subtitle_track_ref().is_some();
            let (video_tables, audio_tables, subtitle_tables) = if !audio_present
                && !subtitle_present
            {
                let chunk_offsets = if self.video_entries.is_empty() {
                    Vec::new()
                } else {
                    // Arrival order == per-track decode order, so one
                    // chunk can span all video samples.
                    vec![self.video_entries[0].offset]
                };
                let samples_per_chunk = u32::try_from(self.video_entries.len()).unwrap_or(u32::MAX);
                (
                    SampleTables::from_samples(
                        &Self::metas(&self.video_entries),
                        chunk_offsets,
                        samples_per_chunk,
                        self.state.video_last_delta(),
                    ),
                    None,
                    None,
                )
            } else {
                (
                    SampleTables::from_samples(
                        &Self::metas(&self.video_entries),
                        Self::offsets(&self.video_entries),
                        1,
                        self.state.video_last_delta(),
                    ),
                    Some(SampleTables::from_samples(
                        &Self::metas(&self.audio_entries),
                        Self::offsets(&self.audio_entries),
                        1,
                        self.state.audio_last_delta(),
                    )),
                    Some(SampleTables::from_samples(
                        &Self::metas(&self.subtitle_entries),
                        Self::offsets(&self.subtitle_entries),
                        1,
                        self.state.subtitle_last_delta(),
                    )),
                )
            };
            let audio = audio_tables.as_ref().and_then(|tables| {
                self.state
                    .audio_track_ref()
                    .map(|track| (track as &Mp4AudioTrack, tables))
            });
            let subtitle = subtitle_tables.as_ref().and_then(|tables| {
                self.state
                    .subtitle_track_ref()
                    .map(|track| (track as &Mp4SubtitleTrack, tables))
            });
            let moov = build_moov_box(
                video,
                &video_tables,
                audio,
                subtitle,
                &video_config,
                self.metadata.as_ref(),
            );
            self.writer.write_all(&moov)?;
            self.bytes_written = self.bytes_written.saturating_add(moov.len() as u64);
        } else {
            let audio_track = self.state.audio_track_ref().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "no audio track configured")
            })?;
            let tables = SampleTables::from_samples(
                &Self::metas(&self.audio_entries),
                if self.audio_entries.is_empty() {
                    Vec::new()
                } else {
                    vec![self.audio_entries[0].offset]
                },
                u32::try_from(self.audio_entries.len()).unwrap_or(u32::MAX),
                self.state.audio_last_delta(),
            );
            let moov = build_audio_only_moov_box(audio_track, &tables, self.metadata.as_ref());
            self.writer.write_all(&moov)?;
            self.bytes_written = self.bytes_written.saturating_add(moov.len() as u64);
        }
        Ok(self.bytes_written)
    }

    /// Total bytes written so far (headers + streamed samples).
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}
