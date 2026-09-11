//! Container demuxing for audio remuxing.
//!
//! These demuxers extract already-encoded packets (plus exact timestamps)
//! from modern audio containers so they can be muxed into MP4/MKV/WebM
//! without any decode/re-encode step. This preserves muxfin's core
//! invariant: frames arrive already encoded.
//!
//! Supported inputs:
//! - **Ogg Opus** (`.ogg`, `.opus`): packets with granule-position
//!   timestamps, pre-skip handling per RFC 7845.
//! - **Native FLAC** (`.flac`): STREAMINFO plus frames with
//!   sample-accurate timestamps.
//!
//! Vorbis-in-Ogg is detected and rejected with an explicit error: it is
//! a legacy codec and ISO BMFF defines no binding for it, so it cannot
//! be carried in MP4.

pub mod flac;
pub mod ogg;

pub use flac::{FlacFrame, FlacStream, demux_flac};
pub use ogg::{OggOpusPacket, OggOpusTrack, demux_ogg_opus};
