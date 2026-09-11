//! Ogg container demuxer for Opus streams (RFC 7845).
//!
//! Parses Ogg pages, reassembles packets of the single Opus logical
//! stream, and derives exact presentation timestamps from granule
//! positions. Only mapping family 0 (mono/stereo) is supported; other
//! mapping families, Vorbis, multiplexed and chained streams are
//! rejected with explicit errors (fail fast, no guessing).
//!
//! Reference: RFC 7845 (Ogg Encapsulation for the Opus Audio Codec).

use crate::assert_invariant;
use crate::codec::opus::{is_valid_opus_packet, opus_packet_samples};

/// No granule position (`0xFFFF_FFFF_FFFF_FFFF`): no packet completes here.
const GRANULE_NONE: u64 = u64::MAX;
/// Opus operates at a fixed 48 kHz clock for timestamps.
const OPUS_CLOCK: f64 = 48_000.0;

/// A demuxed Opus packet with its presentation timestamp.
#[derive(Clone, Debug, PartialEq)]
pub struct OggOpusPacket {
    /// Presentation timestamp in seconds (pre-skip already removed).
    pub pts: f64,
    /// Raw Opus packet bytes (ready for the MP4/MKV Opus track).
    pub data: Vec<u8>,
}

/// A demuxed Ogg Opus logical stream.
#[derive(Clone, Debug, PartialEq)]
pub struct OggOpusTrack {
    /// Channel count from OpusHead.
    pub channels: u8,
    /// Encoder delay in 48 kHz samples (from OpusHead).
    pub pre_skip: u16,
    /// Informational input sample rate from OpusHead.
    pub input_sample_rate: u32,
    /// Audio packets in decode order with exact PTS.
    pub packets: Vec<OggOpusPacket>,
    /// Stream duration in seconds (last packet end, pre-skip removed).
    pub duration: f64,
}

/// Errors from Ogg demuxing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OggError {
    /// Input is empty.
    Empty,
    /// Truncated page or packet data.
    Truncated { offset: usize },
    /// Bad Ogg capture pattern.
    BadCapture { offset: usize },
    /// Unsupported Ogg stream version.
    BadVersion { version: u8 },
    /// Page CRC mismatch.
    BadChecksum { page: u32 },
    /// Missing BOS page or page-sequence gap.
    BrokenSequence { serial: u32 },
    /// No Opus logical stream found.
    NoOpusStream,
    /// Codec cannot be carried (e.g. Vorbis has no ISO BMFF binding).
    UnsupportedCodec { name: String },
    /// Opus mapping family other than 0 (mono/stereo).
    UnsupportedMapping { family: u8 },
    /// More than one Opus logical stream.
    Multiplexed,
    /// Chained Ogg (second BOS after EOS) is not supported.
    Chained,
    /// Malformed OpusHead packet.
    InvalidOpusHead(String),
    /// Malformed or empty Opus audio packet.
    InvalidPacket { page: u32 },
    /// Page granule disagrees with packet durations (corrupt stream).
    GranuleMismatch { page: u32 },
    /// Stream contains no audio packets.
    NoAudioPackets,
}

impl std::fmt::Display for OggError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OggError::Empty => write!(f, "Ogg input is empty"),
            OggError::Truncated { offset } => {
                write!(f, "truncated Ogg data at offset {}", offset)
            }
            OggError::BadCapture { offset } => {
                write!(f, "bad Ogg capture pattern at offset {}", offset)
            }
            OggError::BadVersion { version } => {
                write!(f, "unsupported Ogg stream version {}", version)
            }
            OggError::BadChecksum { page } => write!(f, "Ogg page {} CRC mismatch", page),
            OggError::BrokenSequence { serial } => {
                write!(f, "broken Ogg page sequence for serial {:#x}", serial)
            }
            OggError::NoOpusStream => write!(f, "no Opus logical stream found in Ogg container"),
            OggError::UnsupportedCodec { name } => write!(
                f,
                "Ogg {} stream cannot be muxed into MP4 (no ISO BMFF binding); use Opus",
                name
            ),
            OggError::UnsupportedMapping { family } => write!(
                f,
                "Opus mapping family {} is not supported (only family 0 mono/stereo)",
                family
            ),
            OggError::Multiplexed => {
                write!(
                    f,
                    "multiplexed Ogg (multiple Opus streams) is not supported"
                )
            }
            OggError::Chained => write!(f, "chained Ogg streams are not supported"),
            OggError::InvalidOpusHead(msg) => write!(f, "invalid OpusHead: {}", msg),
            OggError::InvalidPacket { page } => {
                write!(f, "invalid Opus packet on Ogg page {}", page)
            }
            OggError::NoAudioPackets => write!(f, "Ogg Opus stream contains no audio packets"),
            OggError::GranuleMismatch { page } => write!(
                f,
                "Ogg page {} granule disagrees with Opus packet durations",
                page
            ),
        }
    }
}

impl std::error::Error for OggError {}

/// Raw Ogg page header plus payload slices.
struct Page<'a> {
    header_type: u8,
    granule: u64,
    serial: u32,
    seq: u32,
    /// Reassembled payload bytes of this page (concatenated segments).
    payload: &'a [u8],
    /// Segment lengths (lacing values) for packet reassembly.
    segments: &'a [u8],
    /// Total page length (header + table + payload).
    total_len: usize,
}

/// Ogg CRC-32 (poly 0x04C11DB7, init 0, no reflection, xorout 0).
fn ogg_crc(data: &[u8]) -> u32 {
    // INV-701: the Ogg CRC table must have exactly 256 entries.
    let table = crc_table();
    assert_invariant!(
        table.len() == 256,
        "INV-701: Ogg CRC table must have 256 entries",
        "demux::ogg::ogg_crc"
    );
    let mut crc = 0u32;
    for &byte in data {
        let idx = (((crc >> 24) ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc << 8) ^ table[idx];
    }
    crc
}

fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut r = (i as u32) << 24;
        for _ in 0..8 {
            r = if r & 0x8000_0000 != 0 {
                (r << 1) ^ 0x04C1_1DB7
            } else {
                r << 1
            };
        }
        *entry = r;
    }
    table
}

fn le_u32(data: &[u8]) -> u32 {
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

fn le_u64(data: &[u8]) -> u64 {
    u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ])
}

/// Parse one Ogg page at `offset`. Verifies capture pattern and CRC.
fn parse_page(data: &[u8], offset: usize) -> Result<Page<'_>, OggError> {
    if data.len() < offset + 27 {
        return Err(OggError::Truncated { offset });
    }
    let hdr = &data[offset..offset + 27];
    if &hdr[0..4] != b"OggS" {
        return Err(OggError::BadCapture { offset });
    }
    if hdr[4] != 0 {
        return Err(OggError::BadVersion { version: hdr[4] });
    }
    let header_type = hdr[5];
    let granule = le_u64(&hdr[6..14]);
    let serial = le_u32(&hdr[14..18]);
    let seq = le_u32(&hdr[18..22]);
    let n_segments = usize::from(hdr[26]);
    if data.len() < offset + 27 + n_segments {
        return Err(OggError::Truncated { offset });
    }
    let segments = &data[offset + 27..offset + 27 + n_segments];
    let payload_len: usize = segments.iter().map(|&s| usize::from(s)).sum();
    let total_len = 27 + n_segments + payload_len;
    if data.len() < offset + total_len {
        return Err(OggError::Truncated { offset });
    }

    // Verify checksum with the checksum field zeroed.
    let mut check = Vec::with_capacity(total_len);
    check.extend_from_slice(&data[offset..offset + total_len]);
    check[22..26].fill(0);
    let stored = le_u32(&hdr[22..26]);
    if ogg_crc(&check) != stored {
        return Err(OggError::BadChecksum { page: seq });
    }

    Ok(Page {
        header_type,
        granule,
        serial,
        seq,
        payload: &data[offset + 27 + n_segments..offset + total_len],
        segments,
        total_len,
    })
}

/// Identify a BOS stream by its first packet magic.
fn identify_bos(packet: &[u8]) -> &'static str {
    if packet.starts_with(b"OpusHead") {
        "opus"
    } else if packet.starts_with(b"\x01vorbis") {
        "vorbis"
    } else if packet.starts_with(b"fLaC") || packet.starts_with(b"\x7FFLAC") {
        "flac"
    } else if packet.starts_with(b"\x01theora") || packet.starts_with(b"\x80theora") {
        "theora"
    } else {
        "unknown"
    }
}

/// Parsed OpusHead fields needed for demuxing.
struct OpusHead {
    channels: u8,
    pre_skip: u16,
    input_sample_rate: u32,
}

fn parse_opus_head(packet: &[u8]) -> Result<OpusHead, OggError> {
    if packet.len() < 19 {
        return Err(OggError::InvalidOpusHead(
            "packet shorter than 19 bytes".into(),
        ));
    }
    // Version byte sits after the 8-byte magic ("OpusHead" + version).
    if packet[8] != 1 {
        return Err(OggError::InvalidOpusHead(format!(
            "unsupported OpusHead version {} (expected 1)",
            packet[8]
        )));
    }
    let channels = packet[9];
    if channels == 0 {
        return Err(OggError::InvalidOpusHead("channel count is zero".into()));
    }
    let family = packet[18];
    if family != 0 {
        return Err(OggError::UnsupportedMapping { family });
    }
    if packet.len() != 19 {
        return Err(OggError::InvalidOpusHead(
            "unexpected trailing bytes for mapping family 0".into(),
        ));
    }
    Ok(OpusHead {
        channels,
        pre_skip: u16::from_le_bytes([packet[10], packet[11]]),
        input_sample_rate: le_u32(&packet[12..16]),
    })
}

/// Demux an Ogg Opus stream: returns packets with exact PTS in seconds.
///
/// PTS derives from OpusHead pre-skip plus exact TOC durations.
/// Page granules are verified by differences (loss/duplication
/// tripwire) and bound the stream end (end-trim); their absolute
/// anchor is not trusted, since muxers disagree on whether pre-skip
/// counts (RFC 7845) or not.
pub fn demux_ogg_opus(data: &[u8]) -> Result<OggOpusTrack, OggError> {
    if data.is_empty() {
        return Err(OggError::Empty);
    }

    // ---- Parse all pages -------------------------------------------------
    let mut pages = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let page = parse_page(data, offset)?;
        offset += page.total_len;
        pages.push(page);
    }
    if pages.is_empty() {
        return Err(OggError::Empty);
    }

    // ---- Reassemble packets per serial -----------------------------------
    // Map serial -> (packets, first-packet-complete, eos seen, next seq).
    struct StreamState {
        packets: Vec<(Vec<u8>, usize)>,
        pending: Vec<u8>,
        expected_seq: u32,
        eos: bool,
        bos_seen: bool,
    }
    let mut streams: std::collections::HashMap<u32, StreamState> = std::collections::HashMap::new();
    let mut bos_count: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for (idx, page) in pages.iter().enumerate() {
        if page.header_type & 0x02 != 0 {
            *bos_count.entry(page.serial).or_insert(0) += 1;
            if bos_count[&page.serial] > 1 {
                // Second BOS for the same serial: a chained stream.
                return Err(OggError::Chained);
            }
        }
        let state = streams.entry(page.serial).or_insert(StreamState {
            packets: Vec::new(),
            pending: Vec::new(),
            expected_seq: page.seq,
            eos: false,
            bos_seen: false,
        });
        if state.eos {
            // Data after EOS without a new BOS chain header.
            return Err(OggError::Chained);
        }
        if page.header_type & 0x02 != 0 {
            state.bos_seen = true;
        }
        if page.seq != state.expected_seq {
            return Err(OggError::BrokenSequence {
                serial: page.serial,
            });
        }
        state.expected_seq = state.expected_seq.wrapping_add(1);

        let mut pos = 0;
        for &seg in page.segments {
            let len = usize::from(seg);
            state
                .pending
                .extend_from_slice(&page.payload[pos..pos + len]);
            pos += len;
            if seg < 255 {
                state
                    .packets
                    .push((std::mem::take(&mut state.pending), idx));
            }
        }
        if page.header_type & 0x04 != 0 {
            state.eos = true;
        }
    }
    for (serial, state) in &streams {
        if !state.pending.is_empty() {
            return Err(OggError::Truncated { offset: data.len() });
        }
        if !state.bos_seen {
            return Err(OggError::BrokenSequence { serial: *serial });
        }
        let _ = serial;
    }

    // ---- Identify the Opus stream -----------------------------------------
    let mut opus_serial: Option<u32> = None;
    let mut seen_vorbis = false;
    for (serial, state) in &streams {
        if state.packets.is_empty() {
            continue;
        }
        match identify_bos(&state.packets[0].0) {
            "opus" => {
                if opus_serial.is_some() {
                    return Err(OggError::Multiplexed);
                }
                opus_serial = Some(*serial);
            }
            "vorbis" => seen_vorbis = true,
            _ => {}
        }
    }
    let opus_serial = match opus_serial {
        Some(s) => s,
        None if seen_vorbis => {
            return Err(OggError::UnsupportedCodec {
                name: "vorbis".into(),
            });
        }
        None => return Err(OggError::NoOpusStream),
    };
    let raw_packets = &streams[&opus_serial].packets;

    // ---- Interpret packets ------------------------------------------------
    if raw_packets.is_empty() {
        return Err(OggError::NoAudioPackets);
    }
    let head = parse_opus_head(&raw_packets[0].0)?;
    // Second packet must be OpusTags (zero audio duration).
    let mut audio_start = 1;
    if raw_packets.len() > 1 && raw_packets[1].0.starts_with(b"OpusTags") {
        audio_start = 2;
    }
    if raw_packets.len() <= audio_start {
        return Err(OggError::NoAudioPackets);
    }

    // Durations of audio packets from the TOC.
    let mut durations: Vec<u32> = Vec::with_capacity(raw_packets.len() - audio_start);
    for (bytes, _) in &raw_packets[audio_start..] {
        if !is_valid_opus_packet(bytes) {
            return Err(OggError::InvalidPacket { page: 0 });
        }
        let samples = opus_packet_samples(bytes).ok_or(OggError::InvalidPacket { page: 0 })?;
        durations.push(samples);
    }

    // Per-page packet index ranges for granule verification.
    // page_of_packet[i] = index into `pages` where audio packet i completes.
    let page_of_packet: Vec<usize> = raw_packets[audio_start..]
        .iter()
        .map(|(_, page_idx)| *page_idx)
        .collect();

    // Forward PTS assignment from pre-skip. TOC durations are exact, so
    // this is sample-accurate; page granules only verify integrity (and
    // the EOS granule marks end-trim, which forward assignment honors by
    // keeping every decoded packet).
    let n = durations.len();
    let mut start_sample = vec![0u64; n];
    let mut cursor = u64::from(head.pre_skip);
    for (k, duration) in durations.iter().enumerate() {
        start_sample[k] = cursor;
        cursor += u64::from(*duration);
    }

    // Verify granules by differences: between consecutive pages bearing
    // real granules, the granule delta must equal the summed TOC
    // durations of the packets completing in between. Absolute anchors
    // are deliberately NOT checked: RFC 7845 counts pre-skip samples in
    // granule positions, but muxers in the wild (including dissonia
    // 0.1.x) emit plain decoded-sample counts instead. PTS derives
    // exactly from OpusHead pre-skip + TOC durations either way, so
    // granules serve as a loss/duplication tripwire (differences) and
    // an end bound (EOS trim), not as the clock.
    let mut prev_anchor: Option<(usize, u64)> = None; // (run end idx, granule)
    let mut cumulative = 0u64;
    let mut run_sums: Vec<u64> = Vec::with_capacity(n + 1);
    run_sums.push(0);
    for duration in &durations {
        cumulative += u64::from(*duration);
        run_sums.push(cumulative);
    }
    let mut i = 0;
    while i < n {
        let page = page_of_packet[i];
        let mut j = i;
        while j < n && page_of_packet[j] == page {
            j += 1;
        }
        let granule = pages[page].granule;
        let is_eos = pages[page].header_type & 0x04 != 0;
        if granule != GRANULE_NONE {
            if let Some((prev_end, prev_granule)) = prev_anchor {
                let expected = run_sums[j] - run_sums[prev_end];
                if granule < prev_granule || granule - prev_granule != expected {
                    return Err(OggError::GranuleMismatch {
                        page: pages[page].seq,
                    });
                }
            }
            prev_anchor = Some((j, granule));
            // EOS must not declare more audio than the packets hold
            // (declaring fewer is end-trim and expected).
            let computed_end = u64::from(head.pre_skip) + run_sums[j];
            if is_eos && computed_end < granule {
                return Err(OggError::GranuleMismatch {
                    page: pages[page].seq,
                });
            }
        }
        i = j;
    }

    // ---- Emit packets with PTS --------------------------------------------
    let mut packets = Vec::with_capacity(n);
    for (k, (bytes, _)) in raw_packets[audio_start..].iter().enumerate() {
        // Cursor starts at pre-skip, so starts never underflow it.
        let pts_samples = start_sample[k] - u64::from(head.pre_skip);
        packets.push(OggOpusPacket {
            pts: pts_samples as f64 / OPUS_CLOCK,
            data: bytes.clone(),
        });
    }

    // INV-702: demuxed Opus PTS must be non-decreasing.
    let mut prev = -1.0f64;
    for packet in &packets {
        assert_invariant!(
            packet.pts >= prev,
            "INV-702: Ogg Opus packet PTS must be non-decreasing",
            "demux::ogg::demux_ogg_opus"
        );
        prev = packet.pts;
    }

    let duration = packets
        .last()
        .map(|p| p.pts + u64::from(durations[n - 1]) as f64 / OPUS_CLOCK)
        .unwrap_or(0.0);

    Ok(OggOpusTrack {
        channels: head.channels,
        pre_skip: head.pre_skip,
        input_sample_rate: head.input_sample_rate,
        packets,
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_check_value() {
        // Cross-checked against Python and against real ffmpeg Ogg pages:
        // this is the Ogg CRC-32 (poly 0x04C11DB7, init 0, no reflection)
        // over ASCII "123456789".
        assert_eq!(ogg_crc(b"123456789"), 0x89A1_897F);
    }

    /// Build one Ogg page with correct CRC (test helper).
    pub(crate) fn build_page(
        header_type: u8,
        granule: u64,
        serial: u32,
        seq: u32,
        packets: &[&[u8]],
    ) -> Vec<u8> {
        // Lacing: split each packet into 255-chunks + terminator.
        let mut segments = Vec::new();
        let mut payload = Vec::new();
        for packet in packets {
            let mut rest = *packet;
            // Empty packets still emit one zero lacing value.
            loop {
                let take = rest.len().min(255);
                segments.push(take as u8);
                payload.extend_from_slice(&rest[..take]);
                rest = &rest[take..];
                if take < 255 {
                    break;
                }
            }
        }
        let mut page = Vec::new();
        page.extend_from_slice(b"OggS");
        page.push(0);
        page.push(header_type);
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&seq.to_le_bytes());
        page.extend_from_slice(&[0, 0, 0, 0]); // checksum placeholder
        page.push(segments.len() as u8);
        page.extend_from_slice(&segments);
        page.extend_from_slice(&payload);
        let crc = ogg_crc(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        page
    }

    pub(crate) fn opus_head_packet(channels: u8, pre_skip: u16, input_rate: u32) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(b"OpusHead");
        p.push(1);
        p.push(channels);
        p.extend_from_slice(&pre_skip.to_le_bytes());
        p.extend_from_slice(&input_rate.to_le_bytes());
        p.extend_from_slice(&0i16.to_le_bytes());
        p.push(0); // family 0
        p
    }

    /// Minimal valid Opus packet: TOC 0xF8 = config 31 (CELT-only FB
    /// 20 ms = 960 samples @48k), mono, single frame.
    pub(crate) fn opus_audio_packet() -> Vec<u8> {
        vec![0xF8, 0x00, 0x01, 0x02]
    }

    fn minimal_ogg_opus() -> Vec<u8> {
        let mut data = Vec::new();
        // pre_skip 312; packet durations 960 @48k.
        data.extend(build_page(
            0x02,
            0,
            0x1234,
            0,
            &[&opus_head_packet(2, 312, 48_000)],
        ));
        data.extend(build_page(0x00, 0, 0x1234, 1, &[b"OpusTags...."]));
        // First audio page holds 2 packets: granule = 312 + 2*960 = 2232.
        data.extend(build_page(
            0x00,
            2232,
            0x1234,
            2,
            &[&opus_audio_packet(), &opus_audio_packet()],
        ));
        // Second audio page: 2232 + 960 = 3192, EOS.
        data.extend(build_page(0x04, 3192, 0x1234, 3, &[&opus_audio_packet()]));
        data
    }

    #[test]
    fn test_demux_minimal_stream() {
        let track = demux_ogg_opus(&minimal_ogg_opus()).expect("demux");
        assert_eq!(track.channels, 2);
        assert_eq!(track.pre_skip, 312);
        assert_eq!(track.packets.len(), 3);
        // PTS excludes pre-skip: packet k starts at k*960/48000.
        assert!((track.packets[0].pts - 0.0).abs() < 1e-9);
        assert!((track.packets[1].pts - 0.02).abs() < 1e-9);
        assert!((track.packets[2].pts - 0.04).abs() < 1e-9);
    }

    #[test]
    fn test_rejects_vorbis() {
        let mut data = Vec::new();
        data.extend(build_page(
            0x02,
            0,
            0xABCD,
            0,
            &[b"\x01vorbis................."],
        ));
        assert!(matches!(
            demux_ogg_opus(&data),
            Err(OggError::UnsupportedCodec { .. })
        ));
    }

    #[test]
    fn test_rejects_bad_capture() {
        // 27 zero bytes: long enough to parse, wrong capture pattern.
        assert!(matches!(
            demux_ogg_opus(&[0u8; 64]),
            Err(OggError::BadCapture { .. })
        ));
        assert!(matches!(demux_ogg_opus(&[]), Err(OggError::Empty)));
    }

    #[test]
    fn test_rejects_bad_checksum() {
        let mut data = minimal_ogg_opus();
        data[30] ^= 0xFF; // corrupt payload -> CRC mismatch
        assert!(matches!(
            demux_ogg_opus(&data),
            Err(OggError::BadChecksum { .. })
        ));
    }

    #[test]
    fn test_rejects_mapping_family_1() {
        let mut head = opus_head_packet(6, 312, 48_000);
        head[18] = 1;
        head.extend_from_slice(&[2, 1, 0, 1, 2, 3, 4, 5]);
        let mut data = Vec::new();
        data.extend(build_page(0x02, 0, 0x1234, 0, &[&head]));
        assert!(matches!(
            demux_ogg_opus(&data),
            Err(OggError::UnsupportedMapping { family: 1 })
        ));
    }
}
