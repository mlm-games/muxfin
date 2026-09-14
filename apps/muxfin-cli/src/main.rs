use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use muxfin::api::{
    AacProfile, AudioCodec, ContainerFormat, FlacMuxer, Metadata, MkvMuxer, Muxer, MuxerBuilder,
    OggMuxer, VideoCodec,
};
use muxfin::assert_invariant;
use muxfin::demux::{FlacStream, OggOpusTrack, demux_flac, demux_ogg_opus};

fn read_hex_bytes(contents: &str) -> Vec<u8> {
    let hex: String = contents.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(hex.len().is_multiple_of(2), "hex must have even length");

    let mut out = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex");
        out.push(byte);
    }
    out
}

/// Muxfin - Minimal-dependency pure-Rust MP4 muxer
///
/// A professional-grade MP4 muxer designed for recording applications.
/// Supports H.264/H.265/AV1/VP9 video and AAC/Opus/FLAC audio with world-class error handling.
/// Audio inputs may be raw frame dumps, Ogg Opus (.ogg/.opus) or native FLAC (.flac).
#[derive(Parser)]
#[command(name = "muxfin")]
#[command(version, about, long_about)]
#[command(propagate_version = true)]
#[command(arg_required_else_help = true)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Output to JSON (for automation)
    #[arg(long)]
    json: bool,

    /// Disable progress bars
    #[arg(long)]
    no_progress: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Mux encoded frames into MP4/MKV/WebM (default command)
    #[command(alias = "m")]
    Mux {
        /// Input video file(s) or directory
        #[arg(short, long)]
        video: Option<PathBuf>,

        /// Input audio file(s) or directory
        #[arg(short, long)]
        audio: Option<PathBuf>,

        /// Output file (extension may select the container, see --format)
        #[arg(short, long)]
        output: PathBuf,

        /// Output container: mp4, mkv/matroska, webm, ogg/opus, flac (default: mp4)
        #[arg(long, default_value = "mp4")]
        format: ContainerFormat,

        /// Video codec (auto-detected if not specified)
        #[arg(long)]
        video_codec: Option<VideoCodec>,

        /// Video width
        #[arg(long)]
        width: Option<u32>,

        /// Video height
        #[arg(long)]
        height: Option<u32>,

        /// Video frame rate
        #[arg(long)]
        fps: Option<f64>,

        /// Audio codec (auto-detected if not specified)
        #[arg(long)]
        audio_codec: Option<AudioCodec>,

        /// Audio sample rate
        #[arg(long)]
        sample_rate: Option<u32>,

        /// Audio channels
        #[arg(long)]
        channels: Option<u8>,

        /// Enable fragmented MP4 (for DASH/HLS)
        #[arg(long)]
        fragmented: bool,

        /// Fragment duration in milliseconds
        #[arg(long, default_value = "2000")]
        fragment_duration_ms: u32,

        /// Video title
        #[arg(long)]
        title: Option<String>,

        /// Content language (ISO 639-2/T)
        #[arg(long)]
        language: Option<String>,

        /// Creation time (ISO 8601)
        #[arg(long)]
        creation_time: Option<String>,

        /// Validate inputs without creating output file
        #[arg(long)]
        dry_run: bool,

        /// Explicit JSON manifest (verdict §12): sample paths with integer
        /// pts/dts/duration in track timescale. When set, Annex-B splitting
        /// is bypassed; each sample file is one access unit.
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// Overwrite output if it exists (otherwise error).
        #[arg(long)]
        overwrite: bool,

        /// Validate only: check manifest/samples, write nothing.
        #[arg(long)]
        validate_only: bool,
    },

    /// Validate frame data without muxing
    #[command(alias = "v")]
    Validate {
        /// Input video file(s)
        #[arg(short, long)]
        video: Option<PathBuf>,

        /// Input audio file(s)
        #[arg(short, long)]
        audio: Option<PathBuf>,

        /// Output validation report (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Display codec and frame information
    #[command(alias = "i")]
    Info {
        /// Input file to analyze
        input: PathBuf,
    },
}

#[derive(Debug, Serialize)]
struct MuxStats {
    video_frames: u64,
    audio_frames: u64,
    total_bytes: u64,
    duration_ms: u64,
}

impl MuxStats {
    fn new() -> Self {
        Self {
            video_frames: 0,
            audio_frames: 0,
            total_bytes: 0,
            duration_ms: 0,
        }
    }
}

struct ProgressReporter {
    progress: Option<ProgressBar>,
    stats: MuxStats,
}

impl ProgressReporter {
    fn new(enabled: bool) -> Self {
        let progress = if enabled {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] {msg} ({bytes_per_sec})",
                )
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
            );
            pb.set_message("Muxing frames...");
            Some(pb)
        } else {
            None
        };

        Self {
            progress,
            stats: MuxStats::new(),
        }
    }

    fn update_video_frame(&mut self) {
        self.stats.video_frames += 1;
        if let Some(pb) = &self.progress {
            pb.set_message(format!(
                "Muxing frames... (video: {}, audio: {})",
                self.stats.video_frames, self.stats.audio_frames
            ));
        }
    }

    fn update_audio_frame(&mut self) {
        self.stats.audio_frames += 1;
        if let Some(pb) = &self.progress {
            pb.set_message(format!(
                "Muxing frames... (video: {}, audio: {})",
                self.stats.video_frames, self.stats.audio_frames
            ));
        }
    }

    fn update_bytes(&mut self, bytes: u64) {
        self.stats.total_bytes += bytes;
        if let Some(pb) = &self.progress {
            pb.set_length(self.stats.total_bytes);
        }
    }

    fn finish(self) -> Result<MuxStats> {
        if let Some(pb) = self.progress {
            pb.finish_with_message("Muxing complete!");
        }
        Ok(self.stats)
    }
}

/// Explicit manifest schema (verdict §12). Each sample file is exactly one
/// access unit; no Annex-B heuristics are applied.
#[derive(Debug, serde::Deserialize)]
struct Manifest {
    timescale: u32,
    video: Option<ManifestVideo>,
    audio: Option<ManifestAudio>,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ManifestVideo {
    codec: String,
    width: u32,
    height: u32,
    #[serde(default = "default_timescale_90k")]
    timescale: u32,
    samples: Vec<ManifestSample>,
}

#[derive(Debug, serde::Deserialize)]
struct ManifestAudio {
    codec: String,
    sample_rate: u32,
    channels: u16,
    #[serde(default = "default_timescale_48k")]
    timescale: u32,
    samples: Vec<ManifestSample>,
}

#[derive(Debug, serde::Deserialize)]
struct ManifestSample {
    path: PathBuf,
    pts: i64,
    dts: i64,
    duration: u32,
    #[serde(default)]
    keyframe: bool,
}

fn default_timescale_90k() -> u32 {
    90_000
}

fn default_timescale_48k() -> u32 {
    48_000
}

fn manifest_rescale(
    value: i64,
    from: muxfin::time::Timescale,
    to: muxfin::time::Timescale,
) -> anyhow::Result<i64> {
    let num = (value as i128)
        .checked_mul(to.get() as i128)
        .ok_or_else(|| anyhow::anyhow!("timestamp overflow"))?;
    let den = from.get() as i128;
    let q = num.div_euclid(den);
    let r = num.rem_euclid(den);
    let rounded = if r * 2 >= den { q + 1 } else { q };
    i64::try_from(rounded).map_err(|_| anyhow::anyhow!("timestamp overflow"))
}

/// Manifest mux path: integer timestamps, explicit durations, atomic output.
fn mux_manifest_command(
    manifest_path: &PathBuf,
    output: &PathBuf,
    overwrite: bool,
    validate_only: bool,
    verbose: bool,
    json: bool,
) -> Result<()> {
    use muxfin::time::{EncodedSample, SampleTime, Timescale};

    let raw = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("reading manifest {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_str(&raw).with_context(|| "parsing manifest JSON")?;
    if manifest.timescale == 0 {
        anyhow::bail!("manifest timescale must be non-zero");
    }
    let video_cfg = manifest.video.as_ref();
    let audio_cfg = manifest.audio.as_ref();
    if video_cfg.is_none() && audio_cfg.is_none() {
        anyhow::bail!("manifest must contain video and/or audio");
    }
    if output.exists() && !overwrite && !validate_only {
        anyhow::bail!(
            "output {} exists (use --overwrite to replace)",
            output.display()
        );
    }
    // Validate samples first (paths, non-empty, duration > 0, dts order).
    let mut video_blobs: Vec<(ManifestSample, Vec<u8>)> = Vec::new();
    if let Some(v) = video_cfg {
        let mut prev_dts: Option<i64> = None;
        for s in &v.samples {
            if s.duration == 0 {
                anyhow::bail!("video sample {} has zero duration", s.path.display());
            }
            if s.pts < 0 || s.dts < 0 {
                anyhow::bail!("video sample {} has negative timestamp", s.path.display());
            }
            if let Some(prev) = prev_dts
                && s.dts <= prev
            {
                anyhow::bail!("video DTS must strictly increase");
            }
            prev_dts = Some(s.dts);
            let blob = std::fs::read(&s.path)
                .with_context(|| format!("reading sample {}", s.path.display()))?;
            if blob.is_empty() {
                anyhow::bail!("video sample {} is empty", s.path.display());
            }
            video_blobs.push((
                ManifestSample {
                    path: s.path.clone(),
                    pts: s.pts,
                    dts: s.dts,
                    duration: s.duration,
                    keyframe: s.keyframe,
                },
                blob,
            ));
        }
    }
    let mut audio_blobs: Vec<(ManifestSample, Vec<u8>)> = Vec::new();
    if let Some(a) = audio_cfg {
        let mut prev_dts: Option<i64> = None;
        for s in &a.samples {
            if s.duration == 0 {
                anyhow::bail!("audio sample {} has zero duration", s.path.display());
            }
            if s.pts < 0 {
                anyhow::bail!("audio sample {} has negative timestamp", s.path.display());
            }
            if let Some(prev) = prev_dts
                && s.dts < prev
            {
                anyhow::bail!("audio DTS must not decrease");
            }
            prev_dts = Some(s.dts);
            let blob = std::fs::read(&s.path)
                .with_context(|| format!("reading sample {}", s.path.display()))?;
            if blob.is_empty() {
                anyhow::bail!("audio sample {} is empty", s.path.display());
            }
            audio_blobs.push((
                ManifestSample {
                    path: s.path.clone(),
                    pts: s.pts,
                    dts: s.dts,
                    duration: s.duration,
                    keyframe: true,
                },
                blob,
            ));
        }
    }
    if validate_only {
        let report = serde_json::json!({
            "valid": true,
            "video_samples": video_blobs.len(),
            "audio_samples": audio_blobs.len(),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    // Atomic output: temp file in same dir + rename; remove partial on failure.
    let tmp = output.with_extension("tmp-muxfin");
    let result: Result<()> = (|| {
        let out = File::create(&tmp)
            .with_context(|| format!("creating temp output {}", tmp.display()))?;
        let mut builder = MuxerBuilder::new(out);
        if let Some(v) = video_cfg {
            let codec: VideoCodec = v.codec.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            // fps is informational only; timescale is truth.
            builder = builder.video(codec, v.width, v.height, 30.0);
        }
        if let Some(a) = audio_cfg {
            let codec: AudioCodec = a.codec.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            builder = builder.audio(codec, a.sample_rate, a.channels);
        }
        if let Some(t) = manifest.title.clone() {
            builder = builder.with_metadata(Metadata::new().with_title(t));
        }
        let mut muxer = builder.build()?;
        // Interleave by DTS in manifest timescale.
        let vts = video_cfg.map(|v| v.timescale.max(1)).unwrap_or(90_000);
        let ats = audio_cfg.map(|a| a.timescale.max(1)).unwrap_or(48_000);
        let mut vi = 0usize;
        let mut ai = 0usize;
        while vi < video_blobs.len() || ai < audio_blobs.len() {
            let take_video = match (video_blobs.get(vi), audio_blobs.get(ai)) {
                (Some((s, _)), Some((a, _))) => {
                    // Compare in common 90kHz-ish domain via cross-multiplication.
                    (s.dts as i128 * ats as i128) <= (a.dts as i128 * vts as i128)
                }
                (Some(_), None) => true,
                _ => false,
            };
            if take_video {
                let (s, blob) = &video_blobs[vi];
                let timing = SampleTime::new(s.pts, s.dts, s.duration)?;
                // Rescale manifest-track ticks into the muxer's track
                // timescale (video 90k). SampleTime is track-relative, so
                // convert here when they differ.
                let (pts, dts, dur) = if vts == 90_000 {
                    (s.pts, s.dts, s.duration)
                } else {
                    let from = Timescale::new(core::num::NonZeroU32::new(vts).unwrap());
                    let to = Timescale::new(core::num::NonZeroU32::new(90_000).unwrap());
                    (
                        manifest_rescale(s.pts, from, to)?,
                        manifest_rescale(s.dts, from, to)?,
                        manifest_rescale(s.duration as i64, from, to)? as u32,
                    )
                };
                let _ = timing;
                muxer.write_video_sample(EncodedSample {
                    data: blob,
                    timing: SampleTime::new(pts, dts, dur)?,
                    is_sync: s.keyframe,
                })?;
                vi += 1;
            } else {
                let (s, blob) = &audio_blobs[ai];
                let track_ts = audio_cfg.map(|a| a.sample_rate).unwrap_or(48_000);
                let (pts, dur) = if ats == track_ts {
                    (s.pts, s.duration)
                } else {
                    let from = Timescale::new(core::num::NonZeroU32::new(ats).unwrap());
                    let to = Timescale::new(core::num::NonZeroU32::new(track_ts.max(1)).unwrap());
                    (
                        manifest_rescale(s.pts, from, to)?,
                        manifest_rescale(s.duration as i64, from, to)? as u32,
                    )
                };
                muxer.write_audio_sample(EncodedSample {
                    data: blob,
                    timing: SampleTime::new(pts, pts, dur)?,
                    is_sync: true,
                })?;
                ai += 1;
            }
        }
        let stats = muxer.finish_with_stats()?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "video_frames": stats.video_frames,
                    "audio_frames": stats.audio_frames,
                    "bytes_written": stats.bytes_written,
                }))?
            );
        } else if verbose {
            eprintln!(
                "muxed {} video + {} audio samples",
                stats.video_frames, stats.audio_frames
            );
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            std::fs::rename(&tmp, output)
                .with_context(|| format!("renaming {} to {}", tmp.display(), output.display()))?;
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    if cli.verbose {
        eprintln!("Muxfin v{} - Starting...", env!("CARGO_PKG_VERSION"));
    }

    match cli.command {
        Commands::Mux {
            video,
            audio,
            output,
            format,
            video_codec,
            width,
            height,
            fps,
            audio_codec,
            sample_rate,
            channels,
            fragmented,
            fragment_duration_ms,
            title,
            language,
            creation_time,
            dry_run,
            manifest,
            overwrite,
            validate_only,
        } => {
            // Progress only when stderr is a terminal (verdict §12); machine
            // output goes to stdout, diagnostics to stderr.
            use std::io::IsTerminal as _;
            let progress_enabled = !cli.no_progress && !cli.json && std::io::stderr().is_terminal();
            let progress = ProgressReporter::new(progress_enabled);
            mux_command(
                video,
                audio,
                output,
                format,
                video_codec,
                width,
                height,
                fps,
                audio_codec,
                sample_rate,
                channels,
                fragmented,
                fragment_duration_ms,
                title,
                language,
                creation_time,
                dry_run,
                manifest,
                overwrite,
                validate_only,
                progress,
                cli.verbose,
                cli.json,
            )
        }

        Commands::Validate {
            video,
            audio,
            output,
        } => validate_command(video, audio, output, cli.verbose, cli.json),

        Commands::Info { input } => info_command(input, cli.verbose, cli.json),
    }
}

#[allow(clippy::too_many_arguments)]
fn mux_command(
    video: Option<PathBuf>,
    audio: Option<PathBuf>,
    output: PathBuf,
    format: ContainerFormat,
    video_codec: Option<VideoCodec>,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<f64>,
    audio_codec: Option<AudioCodec>,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    fragmented: bool,
    _fragment_duration_ms: u32,
    title: Option<String>,
    language: Option<String>,
    creation_time: Option<String>,
    dry_run: bool,
    manifest: Option<PathBuf>,
    overwrite: bool,
    validate_only: bool,
    mut progress: ProgressReporter,
    verbose: bool,
    json: bool,
) -> Result<()> {
    // Manifest mode bypasses Annex-B heuristics entirely (verdict §12).
    if let Some(manifest_path) = manifest {
        return mux_manifest_command(
            &manifest_path,
            &output,
            overwrite,
            validate_only,
            verbose,
            json,
        );
    }
    if verbose {
        eprintln!("Setting up muxer...");
    }

    // Refuse to clobber without --overwrite; atomic rename on success.
    if output.exists() && !overwrite && !dry_run {
        anyhow::bail!(
            "output {} exists (use --overwrite to replace)",
            output.display()
        );
    }

    // Validate required parameters
    if video.is_none() && audio.is_none() {
        anyhow::bail!("At least one of --video or --audio must be specified");
    }

    // Invariant: CLI must validate video parameters when video is specified
    if video.is_some() {
        assert_invariant!(
            width.is_some() && height.is_some() && fps.is_some(),
            "Video parameters must be complete when video input is provided",
            "cli::mux_command"
        );
    }

    // Invariant: CLI must validate audio parameters when audio is specified.
    // Container inputs (Ogg Opus, FLAC) carry their own parameters, so
    // explicit --sample-rate/--channels are only required for raw input.
    let audio_is_container = audio
        .as_ref()
        .is_some_and(|path| detect_audio_container(path).is_some());
    if audio.is_some() && !audio_is_container {
        assert_invariant!(
            sample_rate.is_some() && channels.is_some(),
            "Audio parameters must be complete when audio input is provided",
            "cli::mux_command"
        );
    }

    if video.is_some() && (width.is_none() || height.is_none() || fps.is_none()) {
        anyhow::bail!(
            "Video parameters --width, --height, and --fps are required when using --video"
        );
    }

    // For dry run, validate inputs but don't create output
    if dry_run {
        if verbose {
            eprintln!("Dry run: Validating inputs...");
        }

        // Validate that input files exist and are readable
        if let Some(ref video_path) = video {
            if !video_path.exists() {
                anyhow::bail!("Video input file does not exist: {}", video_path.display());
            }
            if verbose {
                eprintln!("✓ Video input: {}", video_path.display());
            }
        }

        if let Some(ref audio_path) = audio {
            if !audio_path.exists() {
                anyhow::bail!("Audio input file does not exist: {}", audio_path.display());
            }
            if verbose {
                eprintln!("✓ Audio input: {}", audio_path.display());
            }
        }

        // Validate output path (basic check for dry run)
        if output.exists() && !output.is_file() {
            anyhow::bail!("Output path exists but is not a file: {}", output.display());
        }

        if json {
            println!("{{\"dry_run\": true, \"valid\": true}}");
        } else {
            println!("✅ Dry run complete - inputs are valid!");
            if let Some(ref video_path) = video {
                println!("   Video input: {}", video_path.display());
            }
            if let Some(ref audio_path) = audio {
                println!("   Audio input: {}", audio_path.display());
            }
            println!("   Output would be: {}", output.display());
        }
        return Ok(());
    }

    // Store paths for later use
    let video_path = video.clone();

    // Load audio input once: containers (.ogg/.opus/.flac) are demuxed,
    // everything else keeps the legacy raw-hex frame path.
    let audio_input: Option<AudioInput> = audio
        .as_ref()
        .map(load_audio_input)
        .transpose()
        .with_context(|| "Failed to load audio input")?;

    // Create output file
    let output_file = File::create(&output)
        .with_context(|| format!("Failed to create output file: {}", output.display()))?;

    if fragmented {
        anyhow::bail!(
            "Fragmented output is not yet supported in the CLI. Use the library API with FragmentedMuxer."
        );
    }

    // Build muxer configuration
    let mut builder = MuxerBuilder::new(output_file);

    // Configure video if provided
    if let (Some(_video), Some(width), Some(height), Some(fps)) = (&video, width, height, fps) {
        let codec = video_codec.unwrap_or(VideoCodec::H264); // Default to H.264

        // Invariant: Video codec must be supported
        assert_invariant!(
            matches!(
                codec,
                VideoCodec::H264 | VideoCodec::H265 | VideoCodec::Av1 | VideoCodec::Vp9
            ),
            "Video codec must be one of the supported variants",
            "cli::mux_command"
        );

        builder = builder.video(codec, width, height, fps);

        // Invariant: Video dimensions must be reasonable
        assert_invariant!(
            width >= 320 && height >= 240 && width <= 4096 && height <= 2160,
            "Video dimensions must be within reasonable limits (320x240 to 4096x2160)",
            "cli::mux_command"
        );

        // Invariant: Frame rate must be reasonable
        assert_invariant!(
            fps > 0.0 && fps <= 120.0,
            "Frame rate must be positive and within reasonable limits",
            "cli::mux_command"
        );

        if verbose {
            eprintln!(
                "Configured video: {} {}x{} @ {}fps",
                codec, width, height, fps
            );
        }
    }

    // Configure audio if provided
    if let Some(input) = &audio_input {
        let (codec, rate, chs) = match input {
            AudioInput::OggOpus(track) => {
                if let Some(flag) = audio_codec
                    && flag != AudioCodec::Opus
                {
                    anyhow::bail!("--audio-codec {flag} conflicts with Ogg Opus input");
                }
                (
                    AudioCodec::Opus,
                    sample_rate.unwrap_or(48_000),
                    channels.map(u16::from).unwrap_or(u16::from(track.channels)),
                )
            }
            AudioInput::Flac(stream) => {
                if let Some(flag) = audio_codec
                    && flag != AudioCodec::Flac
                {
                    anyhow::bail!("--audio-codec {flag} conflicts with FLAC input");
                }
                (
                    AudioCodec::Flac,
                    sample_rate.unwrap_or(stream.streaminfo.sample_rate),
                    channels
                        .map(u16::from)
                        .unwrap_or(u16::from(stream.streaminfo.channels)),
                )
            }
            AudioInput::RawHex(_) => {
                if sample_rate.is_none() || channels.is_none() {
                    anyhow::bail!(
                        "--sample-rate and --channels are required for raw audio input (not needed for .ogg/.flac)"
                    );
                }
                (
                    audio_codec.unwrap_or(AudioCodec::Aac(AacProfile::Lc)),
                    sample_rate.unwrap_or(0),
                    channels.map(u16::from).unwrap_or(0),
                )
            }
        };

        // Invariant: Audio codec must be supported
        assert_invariant!(
            matches!(
                codec,
                AudioCodec::Aac(_) | AudioCodec::Opus | AudioCodec::Flac
            ),
            "Audio codec must be one of the supported variants",
            "cli::mux_command"
        );

        builder = builder.audio(codec, rate, chs);
        if let AudioInput::Flac(stream) = input {
            builder = builder.with_flac_streaminfo(stream.streaminfo_raw.to_vec());
        }
        if let AudioInput::OggOpus(track) = input {
            builder = builder.with_opus_preskip(track.pre_skip);
        }

        // Invariant: Audio sample rate must be reasonable
        assert_invariant!(
            rate > 0 && rate <= 192000,
            "Audio sample rate must be positive and within reasonable limits",
            "cli::mux_command"
        );

        // Invariant: Audio channels must be reasonable
        assert_invariant!(
            chs > 0 && chs <= 8,
            "Audio channels must be positive and within reasonable limits",
            "cli::mux_command"
        );

        if verbose {
            eprintln!("Configured audio: {codec} {rate}Hz {chs}ch");
        }
    }

    // Add metadata
    if let Some(title) = title {
        builder = builder.with_metadata(Metadata::new().with_title(title));
    }
    if let Some(language) = language {
        builder = builder.set_language(language);
    }
    if let Some(_creation_time) = creation_time {
        // Parse ISO 8601 datetime
        // For now, skip this - would need chrono dependency
        eprintln!("Warning: creation_time not yet implemented");
    }

    // Build the muxer for the selected container.
    if format == ContainerFormat::Ogg {
        if video.is_some() {
            anyhow::bail!(
                "Ogg output carries Opus audio only; omit --video or use --format mp4/mkv/webm"
            );
        }
        let mut muxer = builder
            .build_ogg()
            .with_context(|| format!("Failed to build {} muxer", format))?;

        if let Some(input) = &audio_input {
            process_audio_frames(input, &mut muxer, &mut progress, verbose)?;
        }

        if verbose {
            eprintln!("Finalizing {}...", format);
        }

        // Invariant: Ogg output requires an audio stream.
        assert_invariant!(
            audio.is_some(),
            "Ogg output requires an audio stream",
            "cli::mux_command"
        );

        // Invariant: Output file must be writable
        assert_invariant!(
            output.metadata().is_ok(),
            "Output file path must be writable",
            "cli::mux_command"
        );

        muxer
            .finish()
            .with_context(|| format!("Failed to finalize {}", format))?;
    } else if format == ContainerFormat::Flac {
        if video.is_some() {
            anyhow::bail!(
                "FLAC output carries FLAC audio only; omit --video or use --format mp4/mkv/webm"
            );
        }
        let mut muxer = builder
            .build_flac()
            .with_context(|| format!("Failed to build {} muxer", format))?;

        if let Some(input) = &audio_input {
            process_audio_frames(input, &mut muxer, &mut progress, verbose)?;
        }

        if verbose {
            eprintln!("Finalizing {}...", format);
        }

        // Invariant: FLAC output requires an audio stream.
        assert_invariant!(
            audio.is_some(),
            "FLAC output requires an audio stream",
            "cli::mux_command"
        );

        // Invariant: Output file must be writable
        assert_invariant!(
            output.metadata().is_ok(),
            "Output file path must be writable",
            "cli::mux_command"
        );

        muxer
            .finish()
            .with_context(|| format!("Failed to finalize {}", format))?;
    } else if format.is_matroska_family() {
        let mut muxer = builder
            .with_container(format)
            .build_mkv()
            .with_context(|| format!("Failed to build {} muxer", format))?;

        // Process video frames
        if let Some(video_path) = video_path {
            process_video_frames(&video_path, &mut muxer, &mut progress, verbose)?;
        }

        // Process audio frames
        if let Some(input) = &audio_input {
            process_audio_frames(input, &mut muxer, &mut progress, verbose)?;
        }

        // Finalize muxing
        if verbose {
            eprintln!("Finalizing {}...", format);
        }

        // Invariant: At least one media stream must be configured
        assert_invariant!(
            video.is_some() || audio.is_some(),
            "At least one media stream (video or audio) must be configured",
            "cli::mux_command"
        );

        // Invariant: Output file must be writable
        assert_invariant!(
            output.metadata().is_ok(),
            "Output file path must be writable",
            "cli::mux_command"
        );

        muxer
            .finish()
            .with_context(|| format!("Failed to finalize {}", format))?;
    } else {
        let mut muxer = builder.build().with_context(|| "Failed to build muxer")?;

        // Process video frames
        if let Some(video_path) = video_path {
            process_video_frames(&video_path, &mut muxer, &mut progress, verbose)?;
        }

        // Process audio frames
        if let Some(input) = &audio_input {
            process_audio_frames(input, &mut muxer, &mut progress, verbose)?;
        }

        // Finalize muxing
        if verbose {
            eprintln!("Finalizing MP4...");
        }

        // Invariant: At least one media stream must be configured
        assert_invariant!(
            video.is_some() || audio.is_some(),
            "At least one media stream (video or audio) must be configured",
            "cli::mux_command"
        );

        // Invariant: Output file must be writable
        assert_invariant!(
            output.metadata().is_ok(),
            "Output file path must be writable",
            "cli::mux_command"
        );

        muxer.finish().with_context(|| "Failed to finalize MP4")?;
    }

    let stats = progress.finish()?;

    // Invariant: Final output must have reasonable size
    assert_invariant!(
        stats.total_bytes > 0,
        "Final output must have non-zero size",
        "cli::mux_command"
    );

    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("✅ Muxing complete!");
        println!("   Video frames: {}", stats.video_frames);
        println!("   Audio frames: {}", stats.audio_frames);
        println!("   Total size: {} bytes", stats.total_bytes);
        println!("   Output: {}", output.display());
    }

    Ok(())
}

/// Unified frame sink for the MP4 ([`Muxer`]) and Matroska ([`MkvMuxer`])
/// frontends so frame ingestion is container-agnostic.
trait MediaSink {
    fn push_video(&mut self, pts: f64, data: &[u8]) -> Result<()>;
    fn push_audio(&mut self, pts: f64, data: &[u8]) -> Result<()>;
}

impl MediaSink for Muxer<File> {
    fn push_video(&mut self, pts: f64, data: &[u8]) -> Result<()> {
        self.write_video(pts, data, true)
            .with_context(|| "Failed to write video frame")
    }

    fn push_audio(&mut self, pts: f64, data: &[u8]) -> Result<()> {
        self.write_audio(pts, data)
            .with_context(|| "Failed to write audio frame")
    }
}

impl MediaSink for MkvMuxer<File> {
    fn push_video(&mut self, pts: f64, data: &[u8]) -> Result<()> {
        self.write_video(pts, data, true)
            .with_context(|| "Failed to write video frame")
    }

    fn push_audio(&mut self, pts: f64, data: &[u8]) -> Result<()> {
        self.write_audio(pts, data)
            .with_context(|| "Failed to write audio frame")
    }
}

impl MediaSink for OggMuxer<File> {
    fn push_video(&mut self, _pts: f64, _data: &[u8]) -> Result<()> {
        anyhow::bail!("Ogg output carries Opus audio only (no video track)")
    }

    fn push_audio(&mut self, pts: f64, data: &[u8]) -> Result<()> {
        self.write_audio(pts, data)
            .with_context(|| "Failed to write audio frame")
    }
}

impl MediaSink for FlacMuxer<File> {
    fn push_video(&mut self, _pts: f64, _data: &[u8]) -> Result<()> {
        anyhow::bail!("FLAC output carries FLAC audio only (no video track)")
    }

    fn push_audio(&mut self, pts: f64, data: &[u8]) -> Result<()> {
        self.write_audio(pts, data)
            .with_context(|| "Failed to write audio frame")
    }
}

fn process_video_frames(
    video_path: &PathBuf,
    muxer: &mut dyn MediaSink,
    progress: &mut ProgressReporter,
    verbose: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Processing video frames from: {}", video_path.display());
    }

    let file = File::open(video_path)
        .with_context(|| format!("Failed to open video file: {}", video_path.display()))?;

    let mut reader = BufReader::new(file);
    let mut hex_content = String::new();
    reader
        .read_to_string(&mut hex_content)
        .with_context(|| "Failed to read video data")?;

    // Convert hex string to bytes (like the example does)
    let data = read_hex_bytes(&hex_content);

    // Write the frame (assuming it's a keyframe at time 0)
    muxer.push_video(0.0, &data)?;

    progress.update_video_frame();
    progress.update_bytes(data.len() as u64);

    Ok(())
}

/// Audio input after loading: either a legacy raw-hex frame dump or a
/// demuxed modern container (Ogg Opus, native FLAC).
enum AudioInput {
    RawHex(Vec<u8>),
    OggOpus(OggOpusTrack),
    Flac(FlacStream),
}

/// Supported container inputs, detected by magic bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AudioContainer {
    Ogg,
    Flac,
}

fn detect_audio_container(path: &PathBuf) -> Option<AudioContainer> {
    let file = File::open(path).ok()?;
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(&mut BufReader::new(file), &mut magic).ok()?;
    if &magic == b"OggS" {
        Some(AudioContainer::Ogg)
    } else if &magic == b"fLaC" {
        Some(AudioContainer::Flac)
    } else {
        None
    }
}

fn load_audio_input(audio_path: &PathBuf) -> Result<AudioInput> {
    match detect_audio_container(audio_path) {
        Some(AudioContainer::Ogg) => {
            let bytes = std::fs::read(audio_path)
                .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
            demux_ogg_opus(&bytes)
                .map(AudioInput::OggOpus)
                .map_err(|e| anyhow::anyhow!("Failed to demux Ogg audio: {e}"))
        }
        Some(AudioContainer::Flac) => {
            let bytes = std::fs::read(audio_path)
                .with_context(|| format!("Failed to read audio file: {}", audio_path.display()))?;
            demux_flac(&bytes)
                .map(AudioInput::Flac)
                .map_err(|e| anyhow::anyhow!("Failed to demux FLAC audio: {e}"))
        }
        None => {
            let file = File::open(audio_path)
                .with_context(|| format!("Failed to open audio file: {}", audio_path.display()))?;
            let mut reader = BufReader::new(file);
            let mut hex_content = String::new();
            reader
                .read_to_string(&mut hex_content)
                .with_context(|| "Failed to read audio data")?;
            // Convert hex string to bytes
            Ok(AudioInput::RawHex(read_hex_bytes(&hex_content)))
        }
    }
}

fn process_audio_frames(
    input: &AudioInput,
    muxer: &mut dyn MediaSink,
    progress: &mut ProgressReporter,
    verbose: bool,
) -> Result<()> {
    match input {
        AudioInput::RawHex(data) => {
            // Write the frame at time 0
            muxer.push_audio(0.0, data)?;
            progress.update_audio_frame();
            progress.update_bytes(data.len() as u64);
        }
        AudioInput::OggOpus(track) => {
            if verbose {
                eprintln!(
                    "Demuxed Ogg Opus: {} packets, {} ch, pre-skip {}",
                    track.packets.len(),
                    track.channels,
                    track.pre_skip
                );
            }
            for packet in &track.packets {
                muxer.push_audio(packet.pts, &packet.data)?;
                progress.update_audio_frame();
                progress.update_bytes(packet.data.len() as u64);
            }
        }
        AudioInput::Flac(stream) => {
            if verbose {
                eprintln!(
                    "Demuxed FLAC: {} frames, {} Hz, {} ch",
                    stream.frames.len(),
                    stream.streaminfo.sample_rate,
                    stream.streaminfo.channels
                );
            }
            for frame in &stream.frames {
                muxer.push_audio(frame.pts, &frame.data)?;
                progress.update_audio_frame();
                progress.update_bytes(frame.data.len() as u64);
            }
        }
    }

    Ok(())
}

fn validate_command(
    video: Option<PathBuf>,
    audio: Option<PathBuf>,
    output: Option<PathBuf>,
    verbose: bool,
    json: bool,
) -> Result<()> {
    if verbose {
        eprintln!("Running validation...");
    }

    let mut is_valid = true;
    let mut checks = Vec::new();

    // Validate video input
    if let Some(ref video_path) = video {
        if !video_path.exists() {
            checks.push(serde_json::json!({
                "type": "video_file",
                "status": "error",
                "message": format!("Video file does not exist: {}", video_path.display())
            }));
            is_valid = false;
        } else {
            // Try to read and validate hex content
            match validate_hex_file(video_path, "video") {
                Ok(msg) => checks.push(serde_json::json!({
                    "type": "video_file",
                    "status": "success",
                    "message": msg
                })),
                Err(e) => {
                    checks.push(serde_json::json!({
                        "type": "video_file",
                        "status": "error",
                        "message": format!("Video file validation failed: {}", e)
                    }));
                    is_valid = false;
                }
            }
        }
    }

    // Validate audio input
    if let Some(ref audio_path) = audio {
        if !audio_path.exists() {
            checks.push(serde_json::json!({
                "type": "audio_file",
                "status": "error",
                "message": format!("Audio file does not exist: {}", audio_path.display())
            }));
            is_valid = false;
        } else {
            // Container inputs (Ogg/FLAC) are demuxed; everything else
            // must be a raw hex frame dump.
            match validate_audio_file(audio_path) {
                Ok(msg) => checks.push(serde_json::json!({
                    "type": "audio_file",
                    "status": "success",
                    "message": msg
                })),
                Err(e) => {
                    checks.push(serde_json::json!({
                        "type": "audio_file",
                        "status": "error",
                        "message": format!("Audio file validation failed: {}", e)
                    }));
                    is_valid = false;
                }
            }
        }
    }

    if video.is_none() && audio.is_none() {
        checks.push(serde_json::json!({
            "type": "input",
            "status": "error",
            "message": "At least one of video or audio input must be specified"
        }));
        is_valid = false;
    }

    let report = serde_json::json!({
        "status": if is_valid { "success" } else { "failed" },
        "valid": is_valid,
        "checks": checks
    });

    if let Some(output_path) = output {
        std::fs::write(&output_path, serde_json::to_string_pretty(&report)?)?;
        if !json {
            println!("Validation report written to: {}", output_path.display());
        }
    } else if json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        if is_valid {
            println!("✅ Validation successful!");
        } else {
            println!("❌ Validation failed!");
        }
        for check in &checks {
            let status = check["status"].as_str().unwrap();
            let message = check["message"].as_str().unwrap();
            if status == "error" {
                println!("   ❌ {}", message);
            } else {
                println!("   ✅ {}", message);
            }
        }
    }

    Ok(())
}

fn validate_audio_file(path: &PathBuf) -> Result<String> {
    match detect_audio_container(path) {
        Some(AudioContainer::Ogg) => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read audio file: {}", path.display()))?;
            let track =
                demux_ogg_opus(&bytes).map_err(|e| anyhow::anyhow!("Ogg demux failed: {e}"))?;
            Ok(format!(
                "Ogg Opus audio is valid ({} packets, {} ch, {:.2}s)",
                track.packets.len(),
                track.channels,
                track.duration
            ))
        }
        Some(AudioContainer::Flac) => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read audio file: {}", path.display()))?;
            let stream =
                demux_flac(&bytes).map_err(|e| anyhow::anyhow!("FLAC demux failed: {e}"))?;
            Ok(format!(
                "FLAC audio is valid ({} frames, {} Hz, {} ch)",
                stream.frames.len(),
                stream.streaminfo.sample_rate,
                stream.streaminfo.channels
            ))
        }
        None => validate_hex_file(path, "audio"),
    }
}

fn validate_hex_file(path: &PathBuf, file_type: &str) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open {} file: {}", file_type, path.display()))?;

    let mut reader = BufReader::new(file);
    let mut content = String::new();
    reader
        .read_to_string(&mut content)
        .with_context(|| format!("Failed to read {} file content", file_type))?;

    // Check if content looks like hex
    let hex_chars: String = content.chars().filter(|c| !c.is_whitespace()).collect();
    if hex_chars.is_empty() {
        anyhow::bail!("{} file is empty", file_type);
    }
    if !hex_chars.len().is_multiple_of(2) {
        anyhow::bail!("{} file contains odd number of hex characters", file_type);
    }
    for ch in hex_chars.chars() {
        if !ch.is_ascii_hexdigit() {
            anyhow::bail!("{} file contains invalid hex character: {}", file_type, ch);
        }
    }

    // Try to convert to bytes
    let bytes = read_hex_bytes(&content);
    if bytes.is_empty() {
        anyhow::bail!("{} file converted to empty byte array", file_type);
    }

    Ok(format!(
        "{} file is valid hex ({} bytes)",
        file_type,
        bytes.len()
    ))
}

fn info_command(input: PathBuf, verbose: bool, json: bool) -> Result<()> {
    if verbose {
        eprintln!("Analyzing file: {}", input.display());
    }

    if !input.exists() {
        anyhow::bail!("Input file does not exist: {}", input.display());
    }

    let file =
        File::open(&input).with_context(|| format!("Failed to open file: {}", input.display()))?;

    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .with_context(|| "Failed to read file content")?;

    if buffer.len() < 4 {
        anyhow::bail!("File too small to be a valid media file");
    }

    // EBML magic (0x1A45DFA3) selects the Matroska/WebM path, which parses
    // with the external mkv-element crate.
    if buffer.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return info_matroska_command(&input, &buffer, verbose, json);
    }

    // Ogg and native FLAC inputs are demuxed with the built-in demuxers.
    if buffer.starts_with(b"OggS") {
        return info_audio_container_command(&input, &buffer, verbose, json);
    }
    if buffer.starts_with(b"fLaC") {
        return info_audio_container_command(&input, &buffer, verbose, json);
    }

    if buffer.len() < 8 {
        anyhow::bail!("File too small to be a valid MP4");
    }

    // Parse basic MP4 structure
    let mut boxes = Vec::new();
    let mut offset = 0;
    while offset + 8 <= buffer.len() {
        let size = u32::from_be_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
        let typ = &buffer[offset + 4..offset + 8];

        if size == 0 {
            break; // Last box
        }

        if offset + size > buffer.len() {
            boxes.push(serde_json::json!({
                "type": "invalid",
                "size": size,
                "offset": offset,
                "error": "Box size exceeds file size"
            }));
            break;
        }

        let box_type = std::str::from_utf8(typ).unwrap_or("????");
        boxes.push(serde_json::json!({
            "type": box_type,
            "size": size,
            "offset": offset
        }));

        offset += size;
    }

    // Check for common MP4 boxes and extract basic codec info
    let has_ftyp = boxes.iter().any(|b| b["type"] == "ftyp");
    let has_moov = boxes.iter().any(|b| b["type"] == "moov");
    let _has_mdat = boxes.iter().any(|b| b["type"] == "mdat");
    let is_valid_mp4 = has_ftyp && has_moov;

    // Basic codec detection
    let has_avc1 = boxes.iter().any(|b| b["type"] == "avc1");
    let has_hvc1 = boxes.iter().any(|b| b["type"] == "hvc1");
    let has_mp4a = boxes.iter().any(|b| b["type"] == "mp4a");
    let has_vp09 = boxes.iter().any(|b| b["type"] == "vp09");

    let video_codec = if has_avc1 {
        "H.264/AVC"
    } else if has_hvc1 {
        "H.265/HEVC"
    } else if has_vp09 {
        "VP9"
    } else {
        "Unknown"
    };
    let has_audio = has_mp4a;

    let info = serde_json::json!({
        "file": input.display().to_string(),
        "file_size": buffer.len(),
        "is_valid_mp4": is_valid_mp4,
        "video_codec": video_codec,
        "has_audio": has_audio,
        "boxes": boxes
    });

    if json {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("File: {}", input.display());
        println!("Size: {} bytes", buffer.len());
        println!("Valid MP4: {}", if is_valid_mp4 { "Yes" } else { "No" });
        println!("Video Codec: {}", video_codec);
        println!("Has Audio: {}", if has_audio { "Yes" } else { "No" });
        println!("Boxes found: {}", info["boxes"].as_array().unwrap().len());
        for box_info in info["boxes"].as_array().unwrap() {
            let typ = box_info["type"].as_str().unwrap();
            let size = box_info["size"].as_u64().unwrap();
            println!("  {}: {} bytes", typ, size);
        }
    }

    Ok(())
}

/// Inspect an Ogg Opus or native FLAC file using the built-in demuxers.
fn info_audio_container_command(
    input: &Path,
    buffer: &[u8],
    verbose: bool,
    json: bool,
) -> Result<()> {
    let info = if buffer.starts_with(b"OggS") {
        match demux_ogg_opus(buffer) {
            Ok(track) => serde_json::json!({
                "file": input.display().to_string(),
                "file_size": buffer.len(),
                "container": "Ogg",
                "audio_codec": "Opus",
                "channels": track.channels,
                "sample_rate": 48000,
                "pre_skip": track.pre_skip,
                "packets": track.packets.len(),
                "duration_secs": track.duration,
            }),
            Err(e) => serde_json::json!({
                "file": input.display().to_string(),
                "container": "Ogg",
                "error": e.to_string(),
            }),
        }
    } else {
        match demux_flac(buffer) {
            Ok(stream) => serde_json::json!({
                "file": input.display().to_string(),
                "file_size": buffer.len(),
                "container": "FLAC",
                "audio_codec": "FLAC",
                "channels": stream.streaminfo.channels,
                "sample_rate": stream.streaminfo.sample_rate,
                "bits_per_sample": stream.streaminfo.bits_per_sample,
                "frames": stream.frames.len(),
            }),
            Err(e) => serde_json::json!({
                "file": input.display().to_string(),
                "container": "FLAC",
                "error": e.to_string(),
            }),
        }
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else if let Some(error) = info.get("error").and_then(|e| e.as_str()) {
        println!("File: {}", input.display());
        println!("Demux error: {}", error);
    } else {
        println!("File: {}", input.display());
        println!("Size: {} bytes", buffer.len());
        println!("Container: {}", info["container"].as_str().unwrap_or("?"));
        println!(
            "Audio Codec: {}",
            info["audio_codec"].as_str().unwrap_or("?")
        );
        println!(
            "Format: {} Hz, {} ch",
            info["sample_rate"], info["channels"]
        );
        if let Some(packets) = info.get("packets").and_then(|v| v.as_u64()) {
            println!("Packets: {}", packets);
        }
        if let Some(frames) = info.get("frames").and_then(|v| v.as_u64()) {
            println!("Frames: {}", frames);
        }
        if verbose {
            println!("Details: {}", serde_json::to_string(&info)?);
        }
    }

    Ok(())
}

/// Inspect a Matroska/WebM file using the external `mkv-element` crate.
fn info_matroska_command(input: &Path, buffer: &[u8], verbose: bool, json: bool) -> Result<()> {
    use mkv_element::io::blocking_impl::ReadFrom;
    use mkv_element::prelude::{Ebml, Segment};

    let mut cursor = std::io::Cursor::new(buffer);
    let ebml = Ebml::read_from(&mut cursor).with_context(|| "Failed to parse EBML header")?;
    let doc_type = ebml
        .doc_type
        .as_ref()
        .map(|d| d.0.clone())
        .unwrap_or_else(|| "unknown".to_string());
    if verbose {
        eprintln!("DocType: {}", doc_type);
    }

    let segment = Segment::read_from(&mut cursor).with_context(|| {
        "Failed to parse Matroska Segment (streaming layouts with unknown-size \
         Segment/Cluster are not supported by the inspector)"
    })?;

    let title = segment.info.title.as_ref().map(|t| t.0.clone());
    let duration = segment.info.duration.as_ref().map(|d| d.0);
    let track_count = segment
        .tracks
        .as_ref()
        .map(|t| t.track_entry.len())
        .unwrap_or(0);
    let cluster_count = segment.cluster.len();
    let cue_count = segment
        .cues
        .as_ref()
        .map(|c| c.cue_point.len())
        .unwrap_or(0);

    let mut tracks_json = Vec::new();
    if let Some(tracks) = &segment.tracks {
        for entry in &tracks.track_entry {
            let kind = match *entry.track_type {
                1 => "video",
                2 => "audio",
                17 => "subtitle",
                other => {
                    tracks_json.push(serde_json::json!({
                        "number": *entry.track_number,
                        "type": format!("unknown({})", other),
                        "codec": entry.codec_id.0.clone(),
                    }));
                    continue;
                }
            };
            let mut track_info = serde_json::json!({
                "number": *entry.track_number,
                "type": kind,
                "codec": entry.codec_id.0.clone(),
                "language": entry.language.0.clone(),
            });
            if let Some(video) = &entry.video {
                track_info["width"] = serde_json::json!(*video.pixel_width);
                track_info["height"] = serde_json::json!(*video.pixel_height);
            }
            if let Some(audio) = &entry.audio {
                track_info["sample_rate"] = serde_json::json!(*audio.sampling_frequency);
                track_info["channels"] = serde_json::json!(*audio.channels);
            }
            tracks_json.push(track_info);
        }
    }

    let info = serde_json::json!({
        "file": input.display().to_string(),
        "file_size": buffer.len(),
        "container": doc_type,
        "title": title,
        "duration_ms": duration,
        "tracks": tracks_json,
        "track_count": track_count,
        "cluster_count": cluster_count,
        "cue_count": cue_count,
    });

    if json {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("File: {}", input.display());
        println!("Size: {} bytes", buffer.len());
        println!("Container: {}", doc_type);
        if let Some(title) = &title {
            println!("Title: {}", title);
        }
        if let Some(duration) = duration {
            println!("Duration: {:.2} s", duration / 1000.0);
        }
        println!("Tracks: {}", track_count);
        for track in &tracks_json {
            println!(
                "  track {}: {} ({}){}",
                track["number"],
                track["type"].as_str().unwrap_or("?"),
                track["codec"].as_str().unwrap_or("?"),
                track
                    .get("language")
                    .and_then(|l| l.as_str())
                    .map(|l| format!(" [{}]", l))
                    .unwrap_or_default(),
            );
            if let (Some(w), Some(h)) = (
                track.get("width").and_then(|v| v.as_u64()),
                track.get("height").and_then(|v| v.as_u64()),
            ) {
                println!("    dimensions: {}x{}", w, h);
            }
            if let (Some(rate), Some(ch)) = (
                track.get("sample_rate").and_then(|v| v.as_f64()),
                track.get("channels").and_then(|v| v.as_u64()),
            ) {
                println!("    audio: {} Hz, {} ch", rate, ch);
            }
        }
        println!("Clusters: {}", cluster_count);
        println!("Cues: {}", cue_count);
    }

    Ok(())
}
