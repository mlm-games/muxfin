//! Integer timestamp primitives.
//!
//! The public `f64`-seconds API is retained as deprecated compatibility
//! shims. New code must use [`Timescale`]/[`SampleTime`]/[`EncodedSample`]
//! so 29.97/59.94 video, 44.1/48 kHz audio, and large timestamps are exact.

use core::num::NonZeroU32;

use crate::api::MuxerError;

/// Track timescale (ticks per second). Always non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Timescale(NonZeroU32);

impl Timescale {
    pub const fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// Convenience for const contexts: `Timescale::const_new(90_000)`.
    pub const fn const_new(value: u32) -> Self {
        // `NonZeroU32::new` is const; panic on zero via unwrap-like trick.
        match NonZeroU32::new(value) {
            Some(v) => Self(v),
            None => panic!("Timescale must be non-zero"),
        }
    }
}

/// Integer-timed sample descriptor. `duration` is explicit and must be > 0:
/// the core muxer never guesses durations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SampleTime {
    pub pts: i64,
    pub dts: i64,
    pub duration: u32,
}

impl SampleTime {
    pub fn new(pts: i64, dts: i64, duration: u32) -> Result<Self, MuxerError> {
        if duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        Ok(Self { pts, dts, duration })
    }

    pub fn composition_offset(self) -> Result<i32, MuxerError> {
        let offset = self
            .pts
            .checked_sub(self.dts)
            .ok_or(MuxerError::TimestampOverflow)?;
        i32::try_from(offset).map_err(|_| MuxerError::CompositionOffsetOutOfRange {
            pts: self.pts,
            dts: self.dts,
        })
    }

    pub fn decode_end(self) -> Result<i64, MuxerError> {
        self.dts
            .checked_add(i64::from(self.duration))
            .ok_or(MuxerError::TimestampOverflow)
    }
}

/// A single encoded access unit with integer timing.
#[derive(Debug)]
pub struct EncodedSample<'a> {
    pub data: &'a [u8],
    pub timing: SampleTime,
    pub is_sync: bool,
}

/// Subtitle cue with explicit start/duration (timescale ticks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubtitleCue<'a> {
    pub start: i64,
    pub duration: u32,
    pub text: &'a str,
}

impl<'a> SubtitleCue<'a> {
    pub fn new(start: i64, duration: u32, text: &'a str) -> Result<Self, MuxerError> {
        if duration == 0 {
            return Err(MuxerError::ZeroDuration);
        }
        if start < 0 {
            return Err(MuxerError::NegativeSubtitleDts { dts: start });
        }
        Ok(Self {
            start,
            duration,
            text,
        })
    }
}

/// Rescale `value` from `from` to `to` with round-to-nearest, checked.
pub(crate) fn rescale(value: i64, from: Timescale, to: Timescale) -> Result<i64, MuxerError> {
    let numerator = i128::from(value)
        .checked_mul(i128::from(to.get()))
        .ok_or(MuxerError::TimestampOverflow)?;
    let denominator = i128::from(from.get());
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    let rounded = if remainder
        .checked_mul(2)
        .ok_or(MuxerError::TimestampOverflow)?
        >= denominator
    {
        quotient
            .checked_add(1)
            .ok_or(MuxerError::TimestampOverflow)?
    } else {
        quotient
    };
    i64::try_from(rounded).map_err(|_| MuxerError::TimestampOverflow)
}

/// Checked `u64 -> u32` for box fields; never use `as u32` in writers.
#[allow(dead_code)]
pub(crate) fn checked_u32(value: u64, field: &'static str) -> Result<u32, MuxerError> {
    u32::try_from(value).map_err(|_| MuxerError::FieldTooLarge { field, value })
}

/// ISO-639-2/T language code (`eng`, `und`, ...), lowercase ASCII.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LanguageCode([u8; 3]);

impl LanguageCode {
    pub fn parse(value: &str) -> Result<Self, MuxerError> {
        let bytes: [u8; 3] = value
            .as_bytes()
            .try_into()
            .map_err(|_| MuxerError::InvalidLanguage)?;
        if !bytes.iter().all(u8::is_ascii_lowercase) {
            return Err(MuxerError::InvalidLanguage);
        }
        Ok(Self(bytes))
    }

    pub const UND: Self = Self(*b"und");

    pub const fn as_bytes(self) -> [u8; 3] {
        self.0
    }

    /// Pack into the 15-bit `mdhd` language field.
    pub fn packed_u16(self) -> u16 {
        self.0
            .iter()
            .fold(0u16, |acc, &c| (acc << 5) | u16::from(c - 0x60))
    }
}

/// Resource limits for untrusted input.
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_tracks: usize,
    pub max_samples_per_track: usize,
    pub max_sample_size: usize,
    pub max_parameter_set_size: usize,
    pub max_metadata_size: usize,
    pub max_subtitle_size: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_tracks: 8,
            max_samples_per_track: 10_000_000,
            max_sample_size: 64 * 1024 * 1024,
            max_parameter_set_size: 1024 * 1024,
            max_metadata_size: 1024 * 1024,
            max_subtitle_size: 65_535,
        }
    }
}

/// Default timescales (verdict §3).
pub const VIDEO_TIMESCALE: Timescale = Timescale::const_new(90_000);
pub const OPUS_TIMESCALE: Timescale = Timescale::const_new(48_000);
pub const SUBTITLE_TIMESCALE: Timescale = Timescale::const_new(1_000);
pub const MOVIE_TIMESCALE: Timescale = Timescale::const_new(1_000);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescale_rounds() {
        let from = Timescale::const_new(90_000);
        let to = Timescale::const_new(1_000);
        assert_eq!(rescale(4_500, from, to).unwrap(), 50);
    }

    #[test]
    fn zero_duration_rejected() {
        assert!(SampleTime::new(0, 0, 0).is_err());
    }

    #[test]
    fn language_parse() {
        assert!(LanguageCode::parse("eng").is_ok());
        assert!(LanguageCode::parse("EN").is_err());
        assert!(LanguageCode::parse("english").is_err());
    }
}
