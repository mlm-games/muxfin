//! ASS/SSA subtitle helpers for Matroska (`S_TEXT/SSA`, `S_TEXT/ASS`).
//!
//! Per the Matroska codec mapping the `[Script Info]` and `[V4 Styles]` /
//! `[V4+ Styles]` sections are stored in the track's CodecPrivate, while
//! each `Dialogue:` event is stored in its own Block with the payload
//!
//! ```text
//! ReadOrder, Layer, Style, Name, MarginL, MarginR, MarginV, Effect, Text
//! ```
//!
//! (`Layer` is ASS-only; SSA events carry an empty layer field.) The
//! `ReadOrder` field lets the decoder reorder streamed samples back into
//! file order. This module builds both halves from a plain ASS/SSA script
//! without a full subtitle parser dependency.

/// Extract the CodecPrivate for an ASS/SSA track from a full script.
///
/// Returns the `[Script Info]` section plus the `[V4 Styles]` / `[V4+
/// Styles]` section (headers included), normalised to `\n` line endings.
/// Returns `None` when the script carries neither styles section.
pub fn codec_private_from_script(script: &str) -> Option<Vec<u8>> {
    let normalised = script.replace("\r\n", "\n").replace('\r', "\n");
    let mut info: Vec<&str> = Vec::new();
    let mut styles: Vec<&str> = Vec::new();
    let mut current: Option<bool> = None; // false = info, true = styles

    for line in normalised.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let name = trimmed[1..trimmed.len() - 1].to_ascii_lowercase();
            current = match name.as_str() {
                "script info" => {
                    info.push(line);
                    Some(false)
                }
                "v4 styles" | "v4+ styles" | "v4++ styles" => {
                    styles.push(line);
                    Some(true)
                }
                _ => None,
            };
            continue;
        }
        match current {
            Some(false) => info.push(line),
            Some(true) => styles.push(line),
            None => {}
        }
    }

    while info.last().is_some_and(|l| l.trim().is_empty()) {
        info.pop();
    }
    while styles.last().is_some_and(|l| l.trim().is_empty()) {
        styles.pop();
    }

    if styles.is_empty() {
        return None;
    }
    let mut out = info.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(&styles.join("\n"));
    out.push('\n');
    Some(out.into_bytes())
}

/// A single `Dialogue:` event with the fields the Matroska Block needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssEvent {
    /// ASS layer (empty for SSA).
    pub layer: String,
    /// Style name.
    pub style: String,
    /// Speaker / actor name.
    pub name: String,
    /// Margin overrides.
    pub margin_l: String,
    /// Margin overrides.
    pub margin_r: String,
    /// Margin overrides.
    pub margin_v: String,
    /// Effect specification.
    pub effect: String,
    /// Event text (override tags and `\N` escapes preserved verbatim).
    pub text: String,
}

/// Parse the `Dialogue:` lines of a script into events.
///
/// Malformed lines are skipped; `Format:` column order is honoured when a
/// `[Events]` format line is present, otherwise the canonical
/// `Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect,
/// Text` order is assumed.
pub fn parse_dialogue_events(script: &str) -> Vec<AssEvent> {
    let normalised = script.replace("\r\n", "\n").replace('\r', "\n");
    // Column positions from the [Events] Format: line, if present.
    let mut columns: Option<Vec<String>> = None;
    let mut events = Vec::new();
    for line in normalised.lines() {
        let trimmed = line.trim();
        if trimmed.to_ascii_lowercase().starts_with("format:") {
            columns = Some(
                trimmed["Format:".len()..]
                    .split(',')
                    .map(|c| c.trim().to_ascii_lowercase())
                    .collect(),
            );
            continue;
        }
        let Some(payload) = trimmed
            .strip_prefix("Dialogue:")
            .or_else(|| trimmed.strip_prefix("dialogue:"))
        else {
            continue;
        };
        // Split into 10 fields (Text swallows the remainder).
        let fields: Vec<&str> = payload.splitn(10, ',').collect();
        if fields.len() < 10 {
            continue;
        }
        let get = |key: &str, fallback: usize| -> &str {
            if let Some(cols) = &columns {
                if let Some(pos) = cols.iter().position(|c| c == key) {
                    return fields.get(pos).map_or("", |f| f.trim());
                }
            }
            fields.get(fallback).map_or("", |f| f.trim())
        };
        events.push(AssEvent {
            layer: get("layer", 0).to_string(),
            style: get("style", 3).to_string(),
            name: get("name", 4).to_string(),
            margin_l: get("marginl", 5).to_string(),
            margin_r: get("marginr", 6).to_string(),
            margin_v: get("marginv", 7).to_string(),
            effect: get("effect", 8).to_string(),
            text: fields[9].to_string(),
        });
    }
    events
}

/// Encode one ASS/SSA Block payload for `read_order`.
///
/// `read_order` is the 1-based Dialogue index in file order. Newlines in
/// `Text` are preserved; callers pass `\N`-escaped text through verbatim
/// (no `\n` conversion — Matroska stores the ASS text as-is).
pub fn encode_block_payload(read_order: u32, event: &AssEvent) -> Vec<u8> {
    format!(
        "{},{},{},{},{},{},{},{},{}",
        read_order,
        event.layer,
        event.style,
        event.name,
        event.margin_l,
        event.margin_r,
        event.margin_v,
        event.effect,
        event.text
    )
    .into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCRIPT: &str = "[Script Info]\nTitle: test\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname\nStyle: Default,Arial\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\b1}Hi\\Nthere\n";

    #[test]
    fn codec_private_keeps_info_and_styles() {
        let private = codec_private_from_script(SCRIPT).expect("styles present");
        let text = String::from_utf8(private).unwrap();
        assert!(text.contains("[Script Info]"));
        assert!(text.contains("[V4+ Styles]"));
        assert!(!text.contains("[Events]"));
        assert!(!text.contains("Dialogue:"));
    }

    #[test]
    fn codec_private_none_without_styles() {
        assert_eq!(codec_private_from_script("[Script Info]\nTitle: x\n"), None);
    }

    #[test]
    fn dialogue_parse_and_block_layout() {
        let events = parse_dialogue_events(SCRIPT);
        assert_eq!(events.len(), 1);
        let payload = encode_block_payload(1, &events[0]);
        assert_eq!(payload, b"1,0,Default,,0,0,0,,{\\b1}Hi\\Nthere");
    }
}
