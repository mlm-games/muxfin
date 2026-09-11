//! Opus codec support for MP4 muxing.
//!
//! This module provides utilities for working with Opus audio in MP4 containers.
//! Opus in MP4 follows the ISO/IEC 14496-3 Amendment 4 specification, using the
//! `Opus` sample entry and `dOps` (Opus Decoder Configuration) box.
//!
//! # Opus in MP4
//!
//! Key characteristics:
//! - Sample rate is always 48000 Hz (per Opus spec, internal rate is 48kHz)
//! - Timescale should be 48000 for proper timing
//! - Pre-skip samples must be signaled in dOps
//! - Variable frame duration (2.5ms to 60ms)
//!
//! # Frame Duration
//!
//! Opus packets encode their duration in the TOC (Table of Contents) byte.
//! This module can infer frame duration from the TOC or accept user-provided duration.

/// Default Opus sample rate (48kHz, per Opus specification)
pub const OPUS_SAMPLE_RATE: u32 = 48000;

/// Opus decoder configuration for the dOps box.
#[derive(Debug, Clone)]
pub struct OpusConfig {
    /// Opus version (should be 0)
    pub version: u8,
    /// Number of output channels (1-8)
    pub output_channel_count: u8,
    /// Pre-skip samples (samples to discard at start for encoder/decoder delay)
    pub pre_skip: u16,
    /// Original sample rate (for informational purposes only, Opus always decodes at 48kHz)
    pub input_sample_rate: u32,
    /// Output gain in dB (Q7.8 fixed point: value / 256.0 = dB)
    pub output_gain: i16,
    /// Channel mapping family (0 = mono/stereo, 1 = Vorbis order, 2+ = application-defined)
    pub channel_mapping_family: u8,
    /// Stream count (for mapping family >= 1)
    pub stream_count: Option<u8>,
    /// Coupled stream count (for mapping family >= 1)
    pub coupled_count: Option<u8>,
    /// Channel mapping table (for mapping family >= 1)
    pub channel_mapping: Option<Vec<u8>>,
}

impl Default for OpusConfig {
    fn default() -> Self {
        Self {
            version: 0,
            output_channel_count: 2,
            pre_skip: 312, // Common encoder delay
            input_sample_rate: 48000,
            output_gain: 0,
            channel_mapping_family: 0,
            stream_count: None,
            coupled_count: None,
            channel_mapping: None,
        }
    }
}

impl OpusConfig {
    /// Create a mono Opus configuration.
    pub fn mono() -> Self {
        Self {
            output_channel_count: 1,
            ..Default::default()
        }
    }

    /// Create a stereo Opus configuration.
    pub fn stereo() -> Self {
        Self {
            output_channel_count: 2,
            ..Default::default()
        }
    }

    /// Create configuration with custom pre-skip.
    pub fn with_pre_skip(mut self, pre_skip: u16) -> Self {
        self.pre_skip = pre_skip;
        self
    }

    /// Create configuration with a channel count.
    ///
    /// 1-2 channels use mapping family 0 (mono/stereo); 3-6 channels use
    /// mapping family 1 with the RFC 7845 §5.1.1.2 Vorbis-order stream
    /// layout. Counts of 0 or above 6 fall back to a bare family-1 header —
    /// use [`OpusConfig::with_channel_mapping`] with an explicit table for
    /// 7+ channels instead of relying on this fallback.
    pub fn with_channels(mut self, channels: u8) -> Self {
        if let Ok(full) = Self::from_channel_count(channels) {
            full.with_pre_skip(self.pre_skip)
        } else {
            self.output_channel_count = channels;
            if channels > 2 {
                // For > 2 channels, need mapping family 1 or higher
                self.channel_mapping_family = 1;
            }
            self
        }
    }

    /// Three-channel (L/C/R) surround, family 1.
    pub fn three() -> Self {
        Self::from_channel_count(3).expect("3 channels is a defined layout")
    }

    /// Four-channel quad (FL/FR/BL/BR), family 1.
    pub fn quad() -> Self {
        Self::from_channel_count(4).expect("4 channels is a defined layout")
    }

    /// Five-channel surround, family 1.
    pub fn five() -> Self {
        Self::from_channel_count(5).expect("5 channels is a defined layout")
    }

    /// Six-channel 5.1 surround, family 1.
    pub fn surround_51() -> Self {
        Self::from_channel_count(6).expect("5.1 is a defined layout")
    }

    /// Family-1 configuration for 1-6 channels with the RFC 7845 §5.1.1.2
    /// Vorbis-order stream/mapping tables.
    ///
    /// | channels | streams | coupled | mapping |
    /// |----------|---------|---------|---------|
    /// | 1 | — (family 0) | — | — |
    /// | 2 | — (family 0) | — | — |
    /// | 3 | 2 | 1 | `[0, 2, 1]` |
    /// | 4 | 2 | 2 | `[0, 1, 2, 3]` |
    /// | 5 | 3 | 2 | `[0, 4, 1, 2, 3]` |
    /// | 6 | 4 | 2 | `[0, 4, 1, 2, 3, 5]` |
    ///
    /// Returns [`OpusConfigError::UnsupportedChannelCount`] for 0 or 7+.
    pub fn from_channel_count(channels: u8) -> Result<Self, OpusConfigError> {
        let (family, streams, coupled, mapping) = match channels {
            1 | 2 => (0, None, None, None),
            3 => (1, Some(2), Some(1), Some(vec![0, 2, 1])),
            4 => (1, Some(2), Some(2), Some(vec![0, 1, 2, 3])),
            5 => (1, Some(3), Some(2), Some(vec![0, 4, 1, 2, 3])),
            6 => (1, Some(4), Some(2), Some(vec![0, 4, 1, 2, 3, 5])),
            _ => return Err(OpusConfigError::UnsupportedChannelCount(channels)),
        };
        Ok(Self {
            output_channel_count: channels,
            channel_mapping_family: family,
            stream_count: streams,
            coupled_count: coupled,
            channel_mapping: mapping,
            ..Default::default()
        })
    }

    /// Explicit multichannel layout for any family (e.g. 7.1, family 255).
    ///
    /// `mapping` must hold exactly `output_channel_count` entries; use 255
    /// for silent (unmapped) channels per RFC 7845 §5.1.1.
    pub fn with_channel_mapping(
        mut self,
        family: u8,
        stream_count: u8,
        coupled_count: u8,
        mapping: Vec<u8>,
    ) -> Self {
        self.channel_mapping_family = family;
        self.stream_count = Some(stream_count);
        self.coupled_count = Some(coupled_count);
        self.channel_mapping = Some(mapping);
        self
    }

    /// Structural validation for the `dOps` box (RFC 7845 §5.1.1).
    ///
    /// Checks: version is 0 (the only defined version), channel count is
    /// non-zero, family 0 carries no extended mapping and at most 2
    /// channels, family ≥ 1 carries stream/coupled counts with
    /// `coupled <= streams`, a mapping table of exactly
    /// `output_channel_count` entries, and every mapped entry below the
    /// decoded channel count (`streams + coupled`) or 255 (silent).
    pub fn validate(&self) -> Result<(), OpusConfigError> {
        if self.version != 0 {
            return Err(OpusConfigError::UnsupportedVersion(self.version));
        }
        if self.output_channel_count == 0 {
            return Err(OpusConfigError::UnsupportedChannelCount(0));
        }
        if self.channel_mapping_family == 0 {
            if self.output_channel_count > 2 {
                return Err(OpusConfigError::FamilyZeroTooManyChannels(
                    self.output_channel_count,
                ));
            }
            if self.stream_count.is_some()
                || self.coupled_count.is_some()
                || self.channel_mapping.is_some()
            {
                return Err(OpusConfigError::FamilyZeroWithMapping);
            }
            return Ok(());
        }
        let streams = self
            .stream_count
            .ok_or(OpusConfigError::MissingStreamCounts)?;
        let coupled = self
            .coupled_count
            .ok_or(OpusConfigError::MissingStreamCounts)?;
        if coupled > streams {
            return Err(OpusConfigError::CoupledExceedsStreams { streams, coupled });
        }
        let mapping = self
            .channel_mapping
            .as_ref()
            .ok_or(OpusConfigError::MissingMapping)?;
        if mapping.len() != usize::from(self.output_channel_count) {
            return Err(OpusConfigError::MappingLength {
                channels: self.output_channel_count,
                entries: mapping.len(),
            });
        }
        let decoded = u16::from(streams) + u16::from(coupled);
        for &entry in mapping {
            if entry != 255 && u16::from(entry) >= decoded {
                return Err(OpusConfigError::MappingEntryOutOfRange { entry, decoded });
            }
        }
        Ok(())
    }
}

/// Structural errors from [`OpusConfig::validate`] / [`OpusConfig::from_channel_count`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusConfigError {
    /// Only `dOps` version 0 is defined.
    UnsupportedVersion(u8),
    /// No predefined family-1 layout; use `with_channel_mapping`.
    UnsupportedChannelCount(u8),
    /// Mapping family 0 is mono/stereo only.
    FamilyZeroTooManyChannels(u8),
    /// Mapping family 0 must not carry stream counts or a mapping table.
    FamilyZeroWithMapping,
    /// Family ≥ 1 requires stream and coupled counts.
    MissingStreamCounts,
    /// More coupled streams than total streams.
    CoupledExceedsStreams { streams: u8, coupled: u8 },
    /// Family ≥ 1 requires a mapping table.
    MissingMapping,
    /// Mapping table length must equal the channel count.
    MappingLength { channels: u8, entries: usize },
    /// Mapped entry is neither silent (255) nor below `streams + coupled`.
    MappingEntryOutOfRange { entry: u8, decoded: u16 },
}

impl std::fmt::Display for OpusConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpusConfigError::UnsupportedVersion(v) => {
                write!(f, "unsupported Opus dOps version {v} (only 0 is defined)")
            }
            OpusConfigError::UnsupportedChannelCount(n) => write!(
                f,
                "no predefined channel mapping for {n} channels (use with_channel_mapping)"
            ),
            OpusConfigError::FamilyZeroTooManyChannels(n) => {
                write!(f, "mapping family 0 supports at most 2 channels, got {n}")
            }
            OpusConfigError::FamilyZeroWithMapping => write!(
                f,
                "mapping family 0 must not carry stream counts or a mapping table"
            ),
            OpusConfigError::MissingStreamCounts => {
                write!(f, "mapping family >= 1 requires stream and coupled counts")
            }
            OpusConfigError::CoupledExceedsStreams { streams, coupled } => {
                write!(f, "coupled count {coupled} exceeds stream count {streams}")
            }
            OpusConfigError::MissingMapping => {
                write!(f, "mapping family >= 1 requires a channel mapping table")
            }
            OpusConfigError::MappingLength { channels, entries } => write!(
                f,
                "mapping table has {entries} entries for {channels} channels"
            ),
            OpusConfigError::MappingEntryOutOfRange { entry, decoded } => write!(
                f,
                "mapping entry {entry} is outside 0..{decoded} (or 255 for silent)"
            ),
        }
    }
}

impl std::error::Error for OpusConfigError {}

/// Opus frame duration in samples at 48kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusFrameDuration {
    /// 2.5ms = 120 samples
    Ms2_5,
    /// 5ms = 240 samples
    Ms5,
    /// 10ms = 480 samples
    Ms10,
    /// 20ms = 960 samples
    Ms20,
    /// 40ms = 1920 samples
    Ms40,
    /// 60ms = 2880 samples
    Ms60,
}

impl OpusFrameDuration {
    /// Get the duration in samples at 48kHz.
    pub fn samples(self) -> u32 {
        match self {
            OpusFrameDuration::Ms2_5 => 120,
            OpusFrameDuration::Ms5 => 240,
            OpusFrameDuration::Ms10 => 480,
            OpusFrameDuration::Ms20 => 960,
            OpusFrameDuration::Ms40 => 1920,
            OpusFrameDuration::Ms60 => 2880,
        }
    }

    /// Get the duration in seconds.
    pub fn seconds(self) -> f64 {
        self.samples() as f64 / OPUS_SAMPLE_RATE as f64
    }
}

/// Extract frame duration from the Opus TOC byte.
///
/// The TOC byte layout (RFC 6716 Figure 1) is `config(5) | s(1) | c(2)`
/// with `config = toc >> 3`. Durations follow RFC 6716 Table 2:
/// - 0-11 (SILK-only NB/MB/WB): 10, 20, 40, 60 ms cycling `config % 4`
/// - 12-15 (Hybrid SWB/FB): 10, 20 ms cycling `config % 2`
/// - 16-31 (CELT-only NB/WB/SWB/FB): 2.5, 5, 10, 20 ms cycling `config % 4`
///
/// Returns the frame duration for a single frame in the packet.
pub fn opus_frame_duration_from_toc(toc: u8) -> Option<OpusFrameDuration> {
    // Extract config bits (bits 3-7)
    let config = (toc >> 3) & 0x1F;

    if config > 31 {
        return None;
    }

    // Frame size depends on config value
    // See RFC 6716 Section 3.1, Table 2.
    match config {
        // SILK-only modes (NB/MB/WB): 10, 20, 40, 60 ms
        0..=11 => match config % 4 {
            0 => Some(OpusFrameDuration::Ms10),
            1 => Some(OpusFrameDuration::Ms20),
            2 => Some(OpusFrameDuration::Ms40),
            _ => Some(OpusFrameDuration::Ms60),
        },
        // Hybrid modes (SWB/FB): 10, 20 ms
        12..=15 => match config % 2 {
            0 => Some(OpusFrameDuration::Ms10),
            _ => Some(OpusFrameDuration::Ms20),
        },
        // CELT-only modes (NB/WB/SWB/FB): 2.5, 5, 10, 20 ms
        16..=31 => match config % 4 {
            0 => Some(OpusFrameDuration::Ms2_5),
            1 => Some(OpusFrameDuration::Ms5),
            2 => Some(OpusFrameDuration::Ms10),
            _ => Some(OpusFrameDuration::Ms20),
        },
        _ => None,
    }
}

/// Extract the frame count from the Opus packet.
///
/// Opus packets can contain 1, 2, or a variable number of frames.
/// Returns (frame_count, is_vbr) where is_vbr indicates variable bitrate.
pub fn opus_frame_count(packet: &[u8]) -> Option<(u8, bool)> {
    if packet.is_empty() {
        return None;
    }

    let toc = packet[0];
    let code = toc & 0x03;

    if code > 3 {
        return None;
    }

    match code {
        0 => Some((1, false)), // 1 frame
        1 => Some((2, false)), // 2 frames, equal size
        2 => Some((2, true)),  // 2 frames, different sizes
        3 => {
            // Code 3: arbitrary number of frames
            if packet.len() < 2 {
                return None;
            }

            let frame_count_byte = packet[1];
            let is_vbr = (frame_count_byte & 0x80) != 0;
            let count = frame_count_byte & 0x3F;

            if count == 0 {
                return None;
            }

            Some((count, is_vbr))
        }
        _ => None,
    }
}

/// Calculate total sample duration for an Opus packet.
///
/// Returns the total number of samples (at 48kHz) in the packet.
pub fn opus_packet_samples(packet: &[u8]) -> Option<u32> {
    if packet.is_empty() {
        return None;
    }

    let frame_duration = opus_frame_duration_from_toc(packet[0])?;
    let (frame_count, _) = opus_frame_count(packet)?;

    if !(1..=63).contains(&frame_count) {
        return None;
    }

    let samples = frame_duration.samples() * frame_count as u32;

    if samples == 0 {
        return None;
    }

    Some(samples)
}

/// Validate an Opus packet for basic structural correctness.
///
/// Returns true if the packet appears to be a valid Opus packet.
pub fn is_valid_opus_packet(packet: &[u8]) -> bool {
    if packet.is_empty() {
        return false;
    }

    // Check if we can parse the TOC and frame count
    opus_packet_samples(packet).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_config_default() {
        let config = OpusConfig::default();
        assert_eq!(config.version, 0);
        assert_eq!(config.output_channel_count, 2);
        assert_eq!(config.pre_skip, 312);
        assert_eq!(config.input_sample_rate, 48000);
        assert_eq!(config.output_gain, 0);
        assert_eq!(config.channel_mapping_family, 0);
    }

    #[test]
    fn test_opus_config_mono() {
        let config = OpusConfig::mono();
        assert_eq!(config.output_channel_count, 1);
    }

    #[test]
    fn test_opus_config_stereo() {
        let config = OpusConfig::stereo();
        assert_eq!(config.output_channel_count, 2);
    }

    #[test]
    fn test_opus_frame_duration_samples() {
        assert_eq!(OpusFrameDuration::Ms2_5.samples(), 120);
        assert_eq!(OpusFrameDuration::Ms5.samples(), 240);
        assert_eq!(OpusFrameDuration::Ms10.samples(), 480);
        assert_eq!(OpusFrameDuration::Ms20.samples(), 960);
        assert_eq!(OpusFrameDuration::Ms40.samples(), 1920);
        assert_eq!(OpusFrameDuration::Ms60.samples(), 2880);
    }

    #[test]
    fn test_opus_frame_duration_from_toc_silk() {
        // SILK-only NB (config 0-3): 10, 20, 40, 60 ms
        assert_eq!(
            opus_frame_duration_from_toc(0b0000_0000), // config 0
            Some(OpusFrameDuration::Ms10)
        );
        assert_eq!(
            opus_frame_duration_from_toc(0b0001_1000), // config 3
            Some(OpusFrameDuration::Ms60)
        );
        // SILK-only MB (config 4-7): 10, 20, 40, 60 ms
        assert_eq!(
            opus_frame_duration_from_toc(0b0010_0000), // config 4
            Some(OpusFrameDuration::Ms10)
        );
        assert_eq!(
            opus_frame_duration_from_toc(0b0011_1000), // config 7
            Some(OpusFrameDuration::Ms60)
        );
        // SILK-only WB (config 8-11): 10, 20, 40, 60 ms
        assert_eq!(
            opus_frame_duration_from_toc(0b0100_0000), // config 8
            Some(OpusFrameDuration::Ms10)
        );
        assert_eq!(
            opus_frame_duration_from_toc(0b0101_1000), // config 11
            Some(OpusFrameDuration::Ms60)
        );
    }

    #[test]
    fn test_opus_frame_duration_from_toc_celt() {
        // CELT-only NB (config 16-19): 2.5, 5, 10, 20 ms
        assert_eq!(
            opus_frame_duration_from_toc(0b1000_0000), // config 16
            Some(OpusFrameDuration::Ms2_5)
        );
        assert_eq!(
            opus_frame_duration_from_toc(0b1000_1000), // config 17
            Some(OpusFrameDuration::Ms5)
        );
        assert_eq!(
            opus_frame_duration_from_toc(0b1001_0000), // config 18
            Some(OpusFrameDuration::Ms10)
        );
        // CELT-only FB (config 28-31): 2.5, 5, 10, 20 ms
        assert_eq!(
            opus_frame_duration_from_toc(0b1110_0000), // config 28
            Some(OpusFrameDuration::Ms2_5)
        );
        assert_eq!(
            opus_frame_duration_from_toc(0b1111_1000), // config 31
            Some(OpusFrameDuration::Ms20)
        );
    }

    #[test]
    fn test_opus_frame_duration_from_toc_hybrid() {
        // Hybrid SWB/FB (config 12-15): 10, 20 ms
        assert_eq!(
            opus_frame_duration_from_toc(0b0110_0000), // config 12
            Some(OpusFrameDuration::Ms10)
        );
        assert_eq!(
            opus_frame_duration_from_toc(0b0111_1000), // config 15
            Some(OpusFrameDuration::Ms20)
        );
    }

    #[test]
    fn test_opus_frame_count_single() {
        // TOC with code 0 = 1 frame
        let packet = vec![0b0000_0000, 0x01, 0x02, 0x03];
        assert_eq!(opus_frame_count(&packet), Some((1, false)));
    }

    #[test]
    fn test_opus_frame_count_double_equal() {
        // TOC with code 1 = 2 frames, equal size
        let packet = vec![0b0000_0001, 0x01, 0x02, 0x03];
        assert_eq!(opus_frame_count(&packet), Some((2, false)));
    }

    #[test]
    fn test_opus_frame_count_double_different() {
        // TOC with code 2 = 2 frames, different sizes
        let packet = vec![0b0000_0010, 0x01, 0x02, 0x03];
        assert_eq!(opus_frame_count(&packet), Some((2, true)));
    }

    #[test]
    fn test_opus_frame_count_arbitrary() {
        // TOC with code 3 = N frames, count in second byte
        let packet = vec![0b0000_0011, 0b0000_0100]; // 4 frames, CBR
        assert_eq!(opus_frame_count(&packet), Some((4, false)));

        let packet_vbr = vec![0b0000_0011, 0b1000_0100]; // 4 frames, VBR
        assert_eq!(opus_frame_count(&packet_vbr), Some((4, true)));
    }

    #[test]
    fn test_opus_packet_samples() {
        // SILK MB 10ms frame (config=4), 1 frame (code=0)
        // TOC: config=4 (bits 3-7 = 0b00100), s=0, c=0
        // Binary: 0b00100_0_00 = 0x20 = 32
        let packet = vec![0x20, 0x01, 0x02, 0x03];
        assert_eq!(opus_packet_samples(&packet), Some(480));

        // SILK MB 10ms frame (config=4), 2 frames (code=1)
        // TOC: config=4 (bits 3-7 = 0b00100), s=0, c=1
        // Binary: 0b00100_0_01 = 0x21 = 33
        let packet2 = vec![0x21, 0x01, 0x02, 0x03];
        assert_eq!(opus_packet_samples(&packet2), Some(960));
    }

    #[test]
    fn test_opus_functions_handle_bad_input_gracefully() {
        // Empty packet should return None, not panic
        assert_eq!(opus_frame_count(&[]), None);
        assert_eq!(opus_packet_samples(&[]), None);

        // Valid TOC config should work
        let valid_toc = vec![0xFF]; // config=31 (CELT-only FB 20ms), valid
        assert_eq!(
            opus_frame_duration_from_toc(valid_toc[0]),
            Some(OpusFrameDuration::Ms20)
        );
        // But packet is too short for frame count
        assert_eq!(opus_packet_samples(&valid_toc), None);

        // Code 3 with count == 0 should return None
        let invalid_code3 = vec![0x03, 0x00]; // code=3, count=0
        assert_eq!(opus_frame_count(&invalid_code3), None);
        assert_eq!(opus_packet_samples(&invalid_code3), None);
    }

    #[test]
    fn test_is_valid_opus_packet() {
        // Valid: config=4 (SILK MB 10ms), code=0 (1 frame)
        assert!(is_valid_opus_packet(&[0x20, 0x01, 0x02]));
        assert!(!is_valid_opus_packet(&[]));
    }

    #[test]
    fn test_predefined_layouts_validate() {
        // Mono/stereo: family 0, no extended mapping.
        for channels in [1u8, 2] {
            let cfg = OpusConfig::from_channel_count(channels).unwrap();
            assert_eq!(cfg.channel_mapping_family, 0);
            assert!(cfg.validate().is_ok());
        }
        // Family-1 Vorbis-order tables (RFC 7845 §5.1.1.2).
        let layouts: &[(u8, u8, u8, &[u8])] = &[
            (3, 2, 1, &[0, 2, 1]),
            (4, 2, 2, &[0, 1, 2, 3]),
            (5, 3, 2, &[0, 4, 1, 2, 3]),
            (6, 4, 2, &[0, 4, 1, 2, 3, 5]),
        ];
        for (channels, streams, coupled, mapping) in layouts {
            let cfg = OpusConfig::from_channel_count(*channels).unwrap();
            assert_eq!(cfg.channel_mapping_family, 1);
            assert_eq!(cfg.stream_count, Some(*streams));
            assert_eq!(cfg.coupled_count, Some(*coupled));
            assert_eq!(cfg.channel_mapping.as_deref(), Some(*mapping));
            assert!(cfg.validate().is_ok());
        }
        assert_eq!(
            OpusConfig::from_channel_count(0).unwrap_err(),
            OpusConfigError::UnsupportedChannelCount(0)
        );
        assert_eq!(
            OpusConfig::from_channel_count(7).unwrap_err(),
            OpusConfigError::UnsupportedChannelCount(7)
        );
        // Named constructors agree with the table.
        assert_eq!(
            OpusConfig::surround_51().channel_mapping.as_deref(),
            Some([0, 4, 1, 2, 3, 5].as_slice())
        );
        assert_eq!(
            OpusConfig::quad().channel_mapping.as_deref(),
            Some([0, 1, 2, 3].as_slice())
        );
        assert!(OpusConfig::three().validate().is_ok());
        assert!(OpusConfig::five().validate().is_ok());
        // with_channels now resolves full layouts, keeping pre-skip.
        let cfg = OpusConfig::default().with_pre_skip(100).with_channels(6);
        assert_eq!(cfg.pre_skip, 100);
        assert_eq!(
            cfg.channel_mapping.as_deref(),
            Some([0, 4, 1, 2, 3, 5].as_slice())
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_bad_configs() {
        // Family 0 with 6 channels.
        let bad = OpusConfig {
            output_channel_count: 6,
            ..Default::default()
        };
        assert_eq!(
            bad.validate(),
            Err(OpusConfigError::FamilyZeroTooManyChannels(6))
        );
        // Family 0 carrying a mapping table.
        let bad = OpusConfig {
            channel_mapping: Some(vec![0, 1]),
            ..Default::default()
        };
        assert_eq!(bad.validate(), Err(OpusConfigError::FamilyZeroWithMapping));
        // Family 1 without counts.
        let bad = OpusConfig::default().with_channels(6);
        let mut bad = bad;
        bad.stream_count = None;
        assert_eq!(bad.validate(), Err(OpusConfigError::MissingStreamCounts));
        // Coupled exceeds streams.
        let bad = OpusConfig::default().with_channel_mapping(1, 1, 2, vec![0, 1]);
        assert_eq!(
            bad.validate(),
            Err(OpusConfigError::CoupledExceedsStreams {
                streams: 1,
                coupled: 2
            })
        );
        // Mapping length mismatch.
        let bad = OpusConfig::default().with_channel_mapping(1, 2, 1, vec![0, 1, 2]);
        assert_eq!(
            bad.validate(),
            Err(OpusConfigError::MappingLength {
                channels: 2,
                entries: 3
            })
        );
        // Entry outside decoded range (streams + coupled = 3, entry 7 illegal).
        let bad = OpusConfig::default().with_channel_mapping(1, 2, 1, vec![0, 7]);
        assert_eq!(
            bad.validate(),
            Err(OpusConfigError::MappingEntryOutOfRange {
                entry: 7,
                decoded: 3
            })
        );
        // 255 means silent and is always legal.
        let ok = OpusConfig::default().with_channel_mapping(1, 2, 1, vec![0, 255]);
        assert!(ok.validate().is_ok());
        // Only version 0 is defined.
        let bad = OpusConfig {
            version: 1,
            ..Default::default()
        };
        assert_eq!(bad.validate(), Err(OpusConfigError::UnsupportedVersion(1)));
    }
}
