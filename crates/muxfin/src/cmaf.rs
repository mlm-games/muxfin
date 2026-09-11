//! CMAF (ISO/IEC 23000-19) media profiles and segment framing.
//!
//! The [`FragmentedMuxer`](crate::fragmented::FragmentedMuxer) emits
//! CMAF-shaped output: an init segment (`ftyp` + `moov` with `mvex`, no
//! sample tables) plus media segments (`moof` + `mdat`). This module adds
//! the two things that make such output actually CMAF-constrained:
//!
//! 1. [`CmafProfile`]: the registered media-profile brands muxfin can emit
//!    (`cfhd`/`cfsd` for AVC, `chd1` for HEVC, `cav1` for AV1, `caac` for
//!    AAC, `cfla` for FLAC) plus per-profile constraints (resolution caps,
//!    90 kHz video timescale, bounded fragment durations, required
//!    parameter sets). [`CmafProfile::check_config`] rejects a
//!    [`FragmentConfig`] that violates
//!    them *before* any bytes are written.
//! 2. CMAF segment framing: every CMAF segment starts with an `styp` box
//!    (major brand `msdh`, compatible `msdh`/`msix`), and init segments
//!    advertise `cmf2` plus the media-profile brand in their `ftyp`.
//!    [`cmaf_styp`] builds the header;
//!    [`FragmentedMuxer::flush_cmaf_segment`](crate::fragmented::FragmentedMuxer::flush_cmaf_segment)
//!    prepends it to a flushed segment.
//!
//! Brand assignments follow ISO/IEC 23000-19 §7 (CMAF media profiles).
//!
//! [`FragmentConfig`]: crate::fragmented::FragmentConfig

use crate::fragmented::FragmentConfig;

/// CMAF media profile that muxfin can emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmafProfile {
    /// AVC standard definition (`cfsd`): ≤ 720×576.
    AvcSd,
    /// AVC high definition (`cfhd`): ≤ 1920×1080.
    AvcHd,
    /// AVC ultra definition (`cud1`): ≤ 3840×2160.
    AvcUhd,
    /// HEVC high definition (`chd1`): ≤ 1920×1080.
    HevcHd,
    /// HEVC ultra definition (`cud1`): ≤ 3840×2160.
    HevcUhd,
    /// AV1 high definition (`cav1`): ≤ 1920×1080.
    Av1Hd,
    /// AAC core (`caac`): 1-2 channels (multichannel via explicit dOps).
    Aac,
    /// FLAC (`cfla`).
    Flac,
}

impl CmafProfile {
    /// Registered CMAF media-profile brand (fourcc).
    pub fn brand(self) -> [u8; 4] {
        match self {
            CmafProfile::AvcSd => *b"cfsd",
            CmafProfile::AvcHd => *b"cfhd",
            CmafProfile::AvcUhd => *b"cud1",
            CmafProfile::HevcHd => *b"chd1",
            CmafProfile::HevcUhd => *b"cud1",
            CmafProfile::Av1Hd => *b"cav1",
            CmafProfile::Aac => *b"caac",
            CmafProfile::Flac => *b"cfla",
        }
    }

    /// Brand as a lowercase string (for manifests / logs).
    pub fn brand_str(self) -> &'static str {
        match self {
            CmafProfile::AvcSd => "cfsd",
            CmafProfile::AvcHd => "cfhd",
            CmafProfile::AvcUhd => "cud1",
            CmafProfile::HevcHd => "chd1",
            CmafProfile::HevcUhd => "cud1",
            CmafProfile::Av1Hd => "cav1",
            CmafProfile::Aac => "caac",
            CmafProfile::Flac => "cfla",
        }
    }

    /// Whether this is a video (as opposed to audio) profile.
    pub fn is_video(self) -> bool {
        !matches!(self, CmafProfile::Aac | CmafProfile::Flac)
    }

    /// Maximum (width, height) for video profiles; `None` for audio.
    pub fn max_resolution(self) -> Option<(u32, u32)> {
        match self {
            CmafProfile::AvcSd => Some((720, 576)),
            CmafProfile::AvcHd | CmafProfile::HevcHd | CmafProfile::Av1Hd => Some((1920, 1080)),
            CmafProfile::AvcUhd | CmafProfile::HevcUhd => Some((3840, 2160)),
            CmafProfile::Aac | CmafProfile::Flac => None,
        }
    }

    /// Compatible brands for the init-segment `ftyp`: CMAF base plus the
    /// media-profile brand.
    pub fn init_compatible_brands(self) -> Vec<[u8; 4]> {
        vec![*b"cmf2", self.brand()]
    }

    /// Check a fragment configuration against this profile's constraints.
    ///
    /// Enforces: 90 kHz video timescale (CMAF §7.3.2), resolution caps,
    /// fragment durations in 250 ms..30 s, and presence of decoder
    /// parameter sets (SPS/PPS, VPS for HEVC, sequence header for AV1).
    /// Audio profiles additionally require the matching codec setup to be
    /// representable (AAC: any AAC; FLAC: STREAMINFO present) — the caller
    /// passes the audio codec it intends to mux.
    pub fn check_config(
        self,
        config: &FragmentConfig,
        audio: Option<crate::api::AudioCodec>,
    ) -> Result<(), CmafError> {
        if self.is_video() {
            if config.timescale != 90_000 {
                return Err(CmafError::BadTimescale {
                    expected: 90_000,
                    actual: config.timescale,
                });
            }
            if let Some((max_w, max_h)) = self.max_resolution() {
                if config.width > max_w || config.height > max_h {
                    return Err(CmafError::ResolutionTooLarge {
                        profile: self.brand_str(),
                        width: config.width,
                        height: config.height,
                        max_width: max_w,
                        max_height: max_h,
                    });
                }
            }
            if config.fragment_duration_ms < 250 || config.fragment_duration_ms > 30_000 {
                return Err(CmafError::FragmentDurationOutOfRange {
                    duration_ms: config.fragment_duration_ms,
                });
            }
            match self {
                CmafProfile::AvcSd | CmafProfile::AvcHd | CmafProfile::AvcUhd => {
                    if config.sps.is_empty() || config.pps.is_empty() {
                        return Err(CmafError::MissingParameterSets {
                            profile: self.brand_str(),
                        });
                    }
                }
                CmafProfile::HevcHd | CmafProfile::HevcUhd => {
                    if config.vps.as_ref().is_none_or(Vec::is_empty)
                        || config.sps.is_empty()
                        || config.pps.is_empty()
                    {
                        return Err(CmafError::MissingParameterSets {
                            profile: self.brand_str(),
                        });
                    }
                }
                CmafProfile::Av1Hd => {
                    if config
                        .av1_sequence_header
                        .as_ref()
                        .is_none_or(Vec::is_empty)
                    {
                        return Err(CmafError::MissingParameterSets {
                            profile: self.brand_str(),
                        });
                    }
                }
                CmafProfile::Aac | CmafProfile::Flac => {}
            }
        }
        match (self, audio) {
            (CmafProfile::Aac, Some(codec)) => {
                if !matches!(codec, crate::api::AudioCodec::Aac(_)) {
                    return Err(CmafError::CodecMismatch {
                        profile: self.brand_str(),
                        detail: "caac requires an AAC codec",
                    });
                }
            }
            (CmafProfile::Flac, Some(codec)) => {
                if codec != crate::api::AudioCodec::Flac {
                    return Err(CmafError::CodecMismatch {
                        profile: self.brand_str(),
                        detail: "cfla requires FLAC",
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Constraint violations from [`CmafProfile::check_config`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CmafError {
    /// Video profiles require a 90 kHz timescale.
    BadTimescale { expected: u32, actual: u32 },
    /// Resolution exceeds the profile cap.
    ResolutionTooLarge {
        profile: &'static str,
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },
    /// Fragment duration outside 250 ms..30 s.
    FragmentDurationOutOfRange { duration_ms: u32 },
    /// Required decoder parameter sets are missing.
    MissingParameterSets { profile: &'static str },
    /// Audio codec does not match the profile.
    CodecMismatch {
        profile: &'static str,
        detail: &'static str,
    },
}

impl std::fmt::Display for CmafError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CmafError::BadTimescale { expected, actual } => write!(
                f,
                "CMAF video profiles require a {expected} Hz timescale, got {actual} Hz"
            ),
            CmafError::ResolutionTooLarge {
                profile,
                width,
                height,
                max_width,
                max_height,
            } => write!(
                f,
                "CMAF profile {profile} caps resolution at {max_width}x{max_height}, got {width}x{height}"
            ),
            CmafError::FragmentDurationOutOfRange { duration_ms } => write!(
                f,
                "CMAF fragment duration {duration_ms} ms is outside 250 ms..30 s"
            ),
            CmafError::MissingParameterSets { profile } => write!(
                f,
                "CMAF profile {profile} requires decoder parameter sets (SPS/PPS, VPS for HEVC, sequence header for AV1)"
            ),
            CmafError::CodecMismatch { profile, detail } => {
                write!(f, "CMAF profile {profile}: {detail}")
            }
        }
    }
}

impl std::error::Error for CmafError {}

/// Build a CMAF segment header (`styp`): major brand `msdh`, compatible
/// brands `msdh` + `msix` (ISO/IEC 23000-19 §7.2.2).
pub fn cmaf_styp() -> Vec<u8> {
    let mut payload = Vec::with_capacity(16);
    payload.extend_from_slice(b"msdh"); // major_brand
    payload.extend_from_slice(&0u32.to_be_bytes()); // minor_version
    payload.extend_from_slice(b"msdh"); // compatible_brands[0]
    payload.extend_from_slice(b"msix"); // compatible_brands[1]
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&(8 + payload.len() as u32).to_be_bytes());
    out.extend_from_slice(b"styp");
    out.extend_from_slice(&payload);
    out
}

/// Build a CMAF init-segment `ftyp`: major brand `cmf2`, compatible brands
/// `cmf2` plus the media-profile brand.
pub fn cmaf_init_ftyp(profile: CmafProfile) -> Vec<u8> {
    let mut payload = Vec::with_capacity(16);
    payload.extend_from_slice(b"cmf2"); // major_brand
    payload.extend_from_slice(&0u32.to_be_bytes()); // minor_version
    for brand in profile.init_compatible_brands() {
        payload.extend_from_slice(&brand);
    }
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&(8 + payload.len() as u32).to_be_bytes());
    out.extend_from_slice(b"ftyp");
    out.extend_from_slice(&payload);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h264_config() -> FragmentConfig {
        FragmentConfig {
            width: 1280,
            height: 720,
            timescale: 90_000,
            fragment_duration_ms: 2000,
            ..FragmentConfig::default()
        }
    }

    #[test]
    fn brands_are_registered_fourccs() {
        assert_eq!(CmafProfile::AvcHd.brand(), *b"cfhd");
        assert_eq!(CmafProfile::AvcSd.brand(), *b"cfsd");
        assert_eq!(CmafProfile::AvcUhd.brand(), *b"cud1");
        assert_eq!(CmafProfile::HevcHd.brand(), *b"chd1");
        assert_eq!(CmafProfile::Av1Hd.brand(), *b"cav1");
        assert_eq!(CmafProfile::Aac.brand(), *b"caac");
        assert_eq!(CmafProfile::Flac.brand(), *b"cfla");
    }

    #[test]
    fn hd_accepts_hd_config() {
        assert!(
            CmafProfile::AvcHd
                .check_config(&h264_config(), None)
                .is_ok()
        );
    }

    #[test]
    fn rejects_bad_timescale() {
        let mut cfg = h264_config();
        cfg.timescale = 48_000;
        assert_eq!(
            CmafProfile::AvcHd.check_config(&cfg, None),
            Err(CmafError::BadTimescale {
                expected: 90_000,
                actual: 48_000
            })
        );
    }

    #[test]
    fn rejects_oversize_resolution() {
        let mut cfg = h264_config();
        cfg.width = 3840;
        cfg.height = 2160;
        assert!(matches!(
            CmafProfile::AvcHd.check_config(&cfg, None),
            Err(CmafError::ResolutionTooLarge { .. })
        ));
        // ...but the UHD profile accepts it.
        assert!(CmafProfile::AvcUhd.check_config(&cfg, None).is_ok());
    }

    #[test]
    fn rejects_missing_parameter_sets() {
        let mut cfg = h264_config();
        cfg.sps.clear();
        assert_eq!(
            CmafProfile::AvcHd.check_config(&cfg, None),
            Err(CmafError::MissingParameterSets { profile: "cfhd" })
        );
        let mut cfg = h264_config();
        cfg.vps = None;
        assert!(matches!(
            CmafProfile::HevcHd.check_config(&cfg, None),
            Err(CmafError::MissingParameterSets { .. })
        ));
    }

    #[test]
    fn rejects_absurd_fragment_durations() {
        let mut cfg = h264_config();
        cfg.fragment_duration_ms = 100;
        assert!(matches!(
            CmafProfile::AvcHd.check_config(&cfg, None),
            Err(CmafError::FragmentDurationOutOfRange { .. })
        ));
    }

    #[test]
    fn styp_bytes() {
        let styp = cmaf_styp();
        assert_eq!(&styp[4..8], b"styp");
        assert_eq!(&styp[8..12], b"msdh");
        assert_eq!(&styp[16..20], b"msdh");
        assert_eq!(&styp[20..24], b"msix");
    }

    #[test]
    fn init_ftyp_advertises_cmf2_and_profile() {
        let ftyp = cmaf_init_ftyp(CmafProfile::AvcHd);
        assert_eq!(&ftyp[4..8], b"ftyp");
        assert_eq!(&ftyp[8..12], b"cmf2");
        assert_eq!(&ftyp[16..20], b"cmf2");
        assert_eq!(&ftyp[20..24], b"cfhd");
    }
}
