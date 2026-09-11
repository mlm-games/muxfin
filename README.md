<p align="center">
  <img src="https://raw.githubusercontent.com/mlm-games/muxfin/main/assets/muxfin-logo.png" alt="Muxfin" width="350"><br>
  <strong>The last mile from encoder to playable MP4.</strong><br><br>
  <a href="https://crates.io/crates/muxfin"><img src="https://img.shields.io/crates/v/muxfin.svg" alt="Crates.io"></a>
  <a href="https://github.com/mlm-games/muxfin/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mlm-games/muxfin" alt="License"></a>
  <a href="https://docs.rs/muxfin"><img src="https://docs.rs/muxfin/badge.svg" alt="Documentation"></a>
    <a href="#license"><img src="https://img.shields.io/crates/l/muxfin.svg" alt="License"></a>
    <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/MSRV-1.98-blue.svg" alt="MSRV"></a>
  <a href="https://github.com/mlm-games/muxfin/actions"><img src="https://github.com/mlm-games/muxfin/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <code>cargo add muxfin</code>
</p>

---

> **Muxfin** takes correctly-timestamped, already-encoded audio/video frames and produces a standards-compliant MP4 — **pure Rust, minimal external dependencies, no FFmpeg.**

<table>
<tr>
<td align="center"><strong>Your Encoder</strong><br><sub>H.264 / HEVC / AV1<br>AAC / Opus / FLAC</sub></td>
<td align="center">➡️</td>
<td align="center"><strong>Muxfin</strong><br><sub>Pure Rust<br>Minimal external deps</sub></td>
<td align="center">➡️</td>
<td align="center"><strong>playable.mp4</strong><br><sub>Standards-compliant<br>Fast-start ready</sub></td>
</tr>
</table>

---

## Why Muxfin Exists

If you're building a recording pipeline in Rust, you know the tradeoffs:

| Approach | Tradeoff |
|----------|----------|
| **FFmpeg CLI/libs** | External binary, GPL licensing concerns, "which build is this?" |
| **GStreamer** | Complex plugin system, C dependencies, heavy runtime |
| **Raw MP4 writing** | ISO-BMFF expertise required (sample tables, interleaving, moov layout) |
| **"Minimal" crates** | Often missing fast-start, strict validation, or production ergonomics |

Muxfin solves **one job cleanly**:

> Take already-encoded frames with correct timestamps → produce a **standards-compliant, immediately-playable MP4** → using **pure Rust**.

Nothing more. Nothing less.

## Installation & Usage

### As a Library
```bash
cargo add muxfin
```

```rust
use muxfin::api::{MuxerBuilder, VideoCodec};

let mut muxer = MuxerBuilder::new(file)
    .video(VideoCodec::H264, 1920, 1080, 30.0)?
    .build()?;

// Write your encoded frames...
muxer.write_video(0.0, &h264_frame, true)?;
muxer.finish()?;
```

### As a CLI Tool
```bash
# Install globally
cargo install muxfin

# Or download pre-built binary from releases
# Then use:
muxfin --help

# Quick examples:
muxfin mux --video frames/ --output output.mp4 --width 1920 --height 1080 --fps 30
muxfin mux --video video.h264 --audio audio.aac --output output.mp4
muxfin validate --video frames/ --audio audio.aac
muxfin info input.mp4
```

The CLI tool accepts raw encoded frames from stdin or files and produces MP4 output.

## Core Invariant

Muxfin enforces a strict contract:

| Your Responsibility | Muxfin's Guarantee |
|:-------------------:|:------------------:|
| ✓ Frames are already encoded | ✓ Valid ISO-BMFF (MP4) |
| ✓ Timestamps are monotonic | ✓ Correct sample tables |
| ✓ DTS provided for B-frames | ✓ Fast-start layout |
| ✓ Codec headers in keyframes | ✓ No post-processing needed |

If input violates the contract, Muxfin **fails fast** with explicit errors—no silent corruption, no guessing.

---

## Features

| Category | Supported | Notes |
|----------|-----------|-------|
| **Video** | H.264/AVC | Annex B format |
| | H.265/HEVC | Annex B with VPS/SPS/PPS |
| | AV1 | OBU stream format |
| | VP9 | Frame header parsing, resolution/bit-depth/color config extraction |
| **Audio** | AAC | All profiles: LC, Main, SSR, LTP, HE, HEv2 |
| | Opus | Raw packets, 48kHz |
| | FLAC | Native frames, `fLaC` + `dfLa` (MP4), `A_FLAC` (MKV) |
| | Demux | Ogg Opus (`.ogg`/`.opus`) and native FLAC (`.flac`) inputs, sample-accurate PTS |
| | Matroska (MKV) | H.264/H.265/AV1/VP9 + AAC/Opus/FLAC + subtitles via `MkvMuxer` |
| | WebM | VP9/AV1 + Opus (whitelist enforced) |
| | B-frames | Explicit PTS/DTS support |
| | Fragmented MP4 | For DASH/HLS streaming |
| | Metadata | Title, creation time, language |
| **Quality** | World-class errors | Detailed diagnostics, hex dumps, JSON output |
| | Production tested | FFmpeg compatibility verified |
| | Comprehensive testing | 80+ tests, property-based validation |

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| 🦀 **Pure Rust** | No unsafe, no FFI, no C bindings |
| 📦 **Minimal deps** | Only essential Rust crates — no external binaries |
| 🧵 **Thread-safe** | `Send + Sync` when writer is |
| ✅ **Well-tested** | Unit, integration, property tests |
| 📜 **Permissive license** | MPL-2.0 |
| 🚨 **Developer-friendly** | Exceptional error messages make debugging 10x faster |

> **Note:** `no_std` is not supported. Muxfin requires `std::io::Write`.

---

## Quick Start

```rust
use muxfin::api::{MuxerBuilder, VideoCodec, AudioCodec, Metadata};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create("recording.mp4")?;
    
    let mut muxer = MuxerBuilder::new(file)
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .audio(AudioCodec::Aac, 48000, 2)
        .with_metadata(Metadata::new().with_title("My Recording"))
        .with_fast_start(true)
        .build()?;

    // Write encoded frames (from your encoder)
    // muxer.write_video(pts_seconds, h264_annex_b_bytes, is_keyframe)?;
    // muxer.write_audio(pts_seconds, aac_adts_bytes)?;

    let stats = muxer.finish_with_stats()?;
    println!("Wrote {} frames, {} bytes", stats.video_frames, stats.bytes_written);
    Ok(())
}
```

<details>
<summary><strong>📹 More Examples: HEVC, AV1, Opus, FLAC, Ogg, Fragmented MP4</strong></summary>

### HEVC/H.265 (4K)

```rust
// Requires VPS, SPS, PPS in first keyframe
let mut muxer = MuxerBuilder::new(file)
    .video(VideoCodec::H265, 3840, 2160, 30.0)
    .build()?;
muxer.write_video(0.0, &hevc_annexb_with_vps_sps_pps, true)?;
```

### AV1

```rust
// Requires Sequence Header OBU in first keyframe
let mut muxer = MuxerBuilder::new(file)
    .video(VideoCodec::Av1, 1920, 1080, 60.0)
    .build()?;
muxer.write_video(0.0, &av1_obu_with_sequence_header, true)?;
```

### Opus Audio

```rust
// Opus always uses 48kHz internally (per spec)
let mut muxer = MuxerBuilder::new(file)
    .video(VideoCodec::H264, 1920, 1080, 30.0)
    .audio(AudioCodec::Opus, 48000, 2)
    .build()?;
muxer.write_audio(0.0, &opus_packet)?;
```

### FLAC Audio (from `.flac` or Ogg-free native frames)

```rust
use muxfin::demux::demux_flac;

// Demux keeps STREAMINFO + sample-accurate frame timestamps.
let flac = demux_flac(&std::fs::read("audio.flac")?)?;

let mut muxer = MuxerBuilder::new(file)
    .audio(
        AudioCodec::Flac,
        flac.streaminfo.sample_rate,
        flac.streaminfo.channels.into(),
    )
    .with_flac_streaminfo(flac.streaminfo_raw.to_vec())
    .build()?;
for frame in &flac.frames {
    muxer.write_audio(frame.pts, &frame.data)?;
}
```

### Ogg Opus input (`.ogg`/`.opus`)

```rust
use muxfin::demux::demux_ogg_opus;

let ogg = demux_ogg_opus(&std::fs::read("audio.ogg")?)?;

let mut muxer = MuxerBuilder::new(file)
    .audio(AudioCodec::Opus, 48_000, ogg.channels.into())
    .with_opus_preskip(ogg.pre_skip)
    .build()?;
for packet in &ogg.packets {
    muxer.write_audio(packet.pts, &packet.data)?;
}
```

### Fragmented MP4 (DASH/HLS)

```rust
use muxfin::codec::vp9::Vp9Config;

// H.264
let sps_bytes = vec![0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11];
let pps_bytes = vec![0x68, 0xce, 0x38, 0x80];

let mut muxer = MuxerBuilder::new(file)
    .video(VideoCodec::H264, 1920, 1080, 30.0)
    .with_sps(sps_bytes)
    .with_pps(pps_bytes)
    .new_with_fragment()?;

// H.265
let vps_bytes = vec![0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00];
let sps_bytes = vec![0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00];
let pps_bytes = vec![0x44, 0x01, 0xc0, 0x73, 0xc0, 0x4c, 0x90];

let mut muxer = MuxerBuilder::new(file)
    .video(VideoCodec::H265, 1920, 1080, 30.0)
    .with_vps(vps_bytes)
    .with_sps(sps_bytes)
    .with_pps(pps_bytes)
    .new_with_fragment()?;

// AV1
let seq_header_bytes = vec![
    0x0A, 0x10, // OBU header + size (example)
    0x00, 0x00, 0x00, 0x00,
];

let mut muxer = MuxerBuilder::new(file)
    .video(VideoCodec::Av1, 1920, 1080, 30.0)
    .with_av1_sequence_header(seq_header_bytes)
    .new_with_fragment()?;

// VP9
let vp9_config = Vp9Config {
    width: 1920,
    height: 1080,
    profile: 0,
    level: 40,
    bit_depth: 8,
    chroma_subsampling: 1,
    video_full_range_flag: 0,
    colour_primaries: 1,
    transfer_characteristics: 1,
    matrix_coefficients: 1,
};

let mut muxer = MuxerBuilder::new(file)
    .video(VideoCodec::Vp9, 1920, 1080, 30.0)
    .with_vp9_config(vp9_config)
    .new_with_fragment()?;

// Get init segment (ftyp + moov)
let init_segment = muxer.init_segment();

// Write frames...
muxer.write_video(0, 0, &frame, true)?;

// Get media segments (moof + mdat)
if let Some(segment) = muxer.flush_segment() {
    // Send segment to client
}
```

### B-Frames with Explicit DTS

```rust
// When encoder produces B-frames, provide both PTS and DTS
muxer.write_video_with_dts(
    pts_seconds,  // Presentation timestamp
    dts_seconds,  // Decode timestamp (for B-frame ordering)
    &frame_data,
    is_keyframe
)?;
```

</details>

---

## Command Line Tool

Muxfin includes a command-line tool for quick testing and development workflows:

```bash
# Install the CLI tool
cargo install muxfin

# Basic video-only muxing
muxfin mux \
  --video keyframes.h264 \
  --width 1920 --height 1080 --fps 30 \
  --output recording.mp4

# Video + audio with metadata
muxfin mux \
  --video stream.h264 \
  --audio stream.aac \
  --video-codec h264 \
  --audio-codec aac-he \
  --width 1920 --height 1080 --fps 30 \
  --sample-rate 44100 --channels 2 \
  --title "My Recording" \
  --language eng \
  --output final.mp4

# JSON output for automation
muxfin mux --json [args...] > stats.json

# Validate input files without muxing
muxfin validate --video input.h264 --audio input.aac

# Get info about supported codecs
muxfin info
```

**Supported Codecs:**
- **Video:** H.264 (AVC), H.265 (HEVC), AV1, VP9
- **Audio:** AAC (all profiles), Opus, FLAC
- **Audio inputs:** raw ADTS/Opus dumps, Ogg Opus (`.ogg`/`.opus`), native FLAC (`.flac`) — containers auto-demux with sample-accurate timestamps, no `--sample-rate`/`--channels` needed

**Supported Containers:** MP4 (default), Matroska (`--format mkv`), WebM (`--format webm`)

**Features:**
- Progress reporting with `--verbose`
- JSON output for CI/CD integration
- Comprehensive error messages
- Fast-start MP4 layout by default
- Metadata support (title, language, creation time)

---

## What Muxfin Is Not

Muxfin is intentionally **focused**. It does **not**:

| Not Supported | Why |
|---------------|-----|
| Encoding/decoding | Use `openh264`, `x264`, `rav1e`, etc. |
| Transcoding | Not a codec library |
| Demuxing/reading MP4 | Write-only by design |
| Timestamp correction | Garbage in = error out |
| Non-MP4/MKV/WebM containers | AVI and legacy formats not supported |
| DRM/encryption | Out of scope |

**Muxfin is the last mile**: encoder output → playable file.

---

## Use Cases

Muxfin is a great fit for:

- 🎥 **Screen recorders** — capture → encode → mux → ship
- 📹 **Camera apps** — webcam/IP camera recording pipelines (e.g., CrabCamera integration)
- 🎬 **Video editors** — export timeline to MP4
- 📡 **Streaming** — generate fMP4 segments for DASH/HLS
- 🏭 **Embedded systems** — single binary, no external deps
- 🔬 **Scientific apps** — deterministic, reproducible output

Probably **not** a fit if you need encoding, demuxing, or legacy codecs (MPEG-2, etc.).

---

## Example: Fast-Start Proof

The `faststart_proof` example demonstrates a structural MP4 invariant:

- Two MP4 files are generated from the same encoded inputs
- One with fast-start enabled, one without
- No external tools are used at any stage

```text
$ cargo run --example faststart_proof --release

output: recording_faststart.mp4
    layout invariant: moov before mdat = YES

output: recording_normal.mp4
    layout invariant: moov before mdat = NO
```

When served over HTTP, the fast-start file can begin playback without waiting for the full download (player behavior varies, but the layout property is deterministic).

This example is intentionally minimal:

- Timestamps are generated in-code
- No B-frames/DTS paths are exercised
- The goal is container layout correctness, not encoding quality

---

## Performance

Muxfin is designed for **minimal overhead**. Muxing should never be your bottleneck.

| Scenario | Time | Throughput |
|----------|------|------------|
| 1000 H.264 frames | 264 µs | **3.7M frames/sec** |
| 1000 H.264 + fast-start | 362 µs | 2.8M frames/sec |
| 1000 video + 1500 audio | 457 µs | 2.2M frames/sec |
| 100 4K frames (~6.5 MB) | 14 ms | **464 MB/sec** |

> **Note:** Benchmarks are based on development hardware. Encoding is typically the bottleneck—muxing overhead is negligible. Run `cargo bench` for your environment (dev-only benchmarks available).AVC

- **Format:** Annex B (start codes: `00 00 00 01` or `00 00 01`)
- **First keyframe must contain:** SPS and PPS NAL units
- **NAL unit types:** IDR (keyframe), non-IDR, SPS, PPS

### H.265/HEVC

- **Format:** Annex B (start codes)
- **First keyframe must contain:** VPS, SPS, and PPS NAL units
- **NAL unit types:** IDR_W_RADL, IDR_N_LP, CRA, VPS, SPS, PPS

### AV1

- **Format:** OBU (Open Bitstream Unit) stream
- **First keyframe must contain:** Sequence Header OBU
- **OBU types:** Sequence Header, Frame, Frame Header, Tile Group

### AAC

- **Format:** ADTS (Audio Data Transport Stream)
- **Header:** 7-byte ADTS header per frame
- **Profiles:** LC-AAC recommended

### Opus

- **Format:** Raw Opus packets (no container)
- **Sample rate:** Always 48000 Hz (Opus specification)
- **Channels:** 1 (mono) or 2 (stereo)

</details>

---

## Documentation

| Resource | Description |
|----------|-------------|
| [📚 API Reference](https://docs.rs/muxfin) | Complete API documentation |
| [📜 Design Charter](docs/charter.md) | Architecture decisions and rationale |
| [📋 API Contract](docs/contract.md) | Input/output guarantees |

---

## FAQ

<details>
<summary><strong>Why not just use FFmpeg?</strong></summary>

FFmpeg is excellent, but:
- External binary dependency (distribution complexity)
- GPL licensing concerns for some builds
- Process orchestration overhead
- "What flags was this built with?" debugging

Muxfin is a single `cargo add` with minimal external dependencies.

</details>

<details>
<summary><strong>Can Muxfin encode video?</strong></summary>

No. Muxfin is **muxing only**. For encoding, use:
- `openh264` — H.264 encoding (BSD)
- `rav1e` — AV1 encoding (BSD)
- `x264`/`x265` — H.264/HEVC (GPL, via FFI)

</details>

<details>
<summary><strong>What if my timestamps are wrong?</strong></summary>

Muxfin will reject non-monotonic timestamps with a clear error. It does not attempt to "fix" broken input — this is by design to ensure predictable output.

</details>

<details>
<summary><strong>Is Muxfin production-ready?</strong></summary>

Yes. Muxfin has an extensive test suite (unit, integration, property-based tests) and is designed for predictable, deterministic behavior.

</details>

---

## License

MPL-2.0.

See [LICENSE](LICENSE) for more info.

---

<p align="center">
  <em>Muxfin is designed to be <strong>boring</strong> in the best way:<br>predictable, strict, fast, and invisible once integrated.</em>
</p>
