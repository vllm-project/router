//! Conservative model-output detection for HTTP load and history observation.

use serde_json::Value;

use super::sse::{sse_lines, FrameScanner};

const MAX_OBSERVABLE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy)]
pub(super) enum HistoryOutput {
    Completion,
    ChatDelta,
    ChatMessage,
}

enum Observation {
    Ignore,
    Output,
    Terminal,
}

impl HistoryOutput {
    pub(super) fn has_output(self, body: &[u8]) -> bool {
        body.len() <= MAX_OBSERVABLE_BYTES
            && serde_json::from_slice(body)
                .is_ok_and(|value| matches!(self.inspect(&value), Observation::Output))
    }

    fn inspect(self, value: &Value) -> Observation {
        if value.get("error").is_some_and(|error| !error.is_null())
            || value.get("object").and_then(Value::as_str) == Some("error")
        {
            return Observation::Terminal;
        }
        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            return Observation::Ignore;
        };
        let mut output = false;
        for choice in choices {
            let reason = choice.get("finish_reason").and_then(Value::as_str);
            if matches!(reason, Some("abort" | "error")) {
                return Observation::Terminal;
            }
            output |= self.choice_has_output(choice);
        }
        if output {
            Observation::Output
        } else {
            Observation::Ignore
        }
    }

    fn choice_has_output(self, choice: &Value) -> bool {
        // Reasoning (including reasoning_content), tools, roles and prompt
        // metadata are not evidence of generated text on these routes.
        let text = match self {
            Self::Completion => choice.get("text"),
            Self::ChatDelta => choice.get("delta").and_then(|delta| delta.get("content")),
            Self::ChatMessage => choice
                .get("message")
                .and_then(|message| message.get("content")),
        };
        text.and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty())
    }
}

pub(super) struct CompletionOutputObserver {
    event: Vec<u8>,
    scanner: FrameScanner,
    first_event: bool,
    stopped: bool,
    history: Option<HistoryOutput>,
}

impl CompletionOutputObserver {
    pub(super) fn new(disabled: bool) -> Self {
        Self {
            event: Vec::new(),
            scanner: FrameScanner::default(),
            first_event: true,
            stopped: disabled,
            history: None,
        }
    }

    pub(super) fn for_history(output: HistoryOutput) -> Self {
        Self {
            history: Some(output),
            ..Self::new(false)
        }
    }

    /// Return true only once, after a complete SSE event proves model output.
    /// No EOF flush: an unterminated event is not evidence. Oversized events
    /// permanently disable detection for this attempt without affecting bytes.
    /// The bound counts wire bytes, including CRLF delimiters.
    pub(super) fn observe(&mut self, mut chunk: &[u8]) -> bool {
        if self.stopped {
            return false;
        }
        while !chunk.is_empty() {
            let scan_len = chunk.len().min(MAX_OBSERVABLE_BYTES - self.event.len() + 1);
            let end = self.scanner.frame_end(&chunk[..scan_len]);
            let consumed = end.unwrap_or(scan_len);
            if consumed > MAX_OBSERVABLE_BYTES - self.event.len() {
                self.event.clear();
                self.stopped = true;
                return false;
            }
            self.event.extend_from_slice(&chunk[..consumed]);
            chunk = &chunk[consumed..];
            if end.is_some() {
                let event = if std::mem::take(&mut self.first_event) {
                    self.event
                        .strip_prefix(b"\xef\xbb\xbf")
                        .unwrap_or(&self.event)
                } else {
                    &self.event
                };
                let observation = inspect_event(event, self.history);
                self.event.clear();
                if !matches!(observation, Observation::Ignore) {
                    self.stopped = true;
                    return matches!(observation, Observation::Output);
                }
            }
        }
        false
    }
}

fn inspect_event(event: &[u8], history: Option<HistoryOutput>) -> Observation {
    let mut data = Vec::new();
    let mut event_type = None;
    for (line, _) in sse_lines(event) {
        if let Some(value) = line.strip_prefix(b"data:") {
            if !data.is_empty() {
                data.push(b'\n');
            }
            data.extend_from_slice(value.strip_prefix(b" ").unwrap_or(value));
        } else if let Some(value) = line.strip_prefix(b"event:") {
            event_type = Some(value.strip_prefix(b" ").unwrap_or(value));
        }
    }
    if history.is_some()
        && (matches!(event_type, Some(b"error" | b"abort")) || data.trim_ascii() == b"[DONE]")
    {
        return Observation::Terminal;
    }
    if event_type == Some(b"error".as_slice()) {
        return Observation::Ignore;
    }
    let Ok(value) = serde_json::from_slice::<Value>(&data) else {
        return Observation::Ignore;
    };
    if let Some(history) = history {
        return history.inspect(&value);
    }
    // Keep load's existing Completion-only frame predicate, including its
    // distinct handling of error fields and terminal events.
    if value.get("error").is_none()
        && value
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                choices
                    .iter()
                    .any(|choice| HistoryOutput::Completion.choice_has_output(choice))
            })
    {
        Observation::Output
    } else {
        Observation::Ignore
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn assert_output_at(event: &[u8], complete_at: usize) {
        for (mode, history) in [("load", None), ("history", Some(HistoryOutput::Completion))] {
            let observer = || match history {
                Some(output) => CompletionOutputObserver::for_history(output),
                None => CompletionOutputObserver::new(false),
            };
            for split in 0..=event.len() {
                let mut observer = observer();
                assert_eq!(
                    observer.observe(&event[..split]),
                    split >= complete_at,
                    "{mode}, split {split}"
                );
                assert_eq!(
                    observer.observe(&event[split..]),
                    split < complete_at,
                    "{mode}, split {split}"
                );
                assert!(!observer.observe(event), "{mode}: repeated output");
            }
            let mut observer = observer();
            for (index, byte) in event.iter().enumerate() {
                assert_eq!(
                    observer.observe(&[*byte]),
                    index + 1 == complete_at,
                    "{mode}, byte {index}"
                );
            }
        }
    }

    #[test]
    fn bare_cr_proves_complete_output_across_every_split() {
        let event = b"data: {\"choices\":\rdata: [{\"text\":\"output\"}]}\r\r";
        assert_output_at(event, event.len());
    }

    #[test]
    fn one_initial_bom_proves_output_across_every_split() {
        for ending in ["\n", "\r\n", "\r"] {
            for blank_line in ["", ending] {
                let event = format!(
                    "\u{feff}{blank_line}data: {{\"choices\":[{{\"text\":\"output\"}}]}}{ending}{ending}"
                );
                let complete_at = event.len() - usize::from(ending == "\r\n");
                assert_output_at(event.as_bytes(), complete_at);
            }
        }
    }

    #[test]
    fn later_or_duplicate_bom_is_not_a_stream_prefix() {
        for prefix in ["\u{feff}\u{feff}", ": first line\n\u{feff}", "\n\u{feff}"] {
            let event = format!("{prefix}data: {{\"choices\":[{{\"text\":\"not output\"}}]}}\r\r");
            for split in 0..=event.len() {
                let mut observer = CompletionOutputObserver::new(false);
                assert!(!observer.observe(&event.as_bytes()[..split]));
                assert!(!observer.observe(&event.as_bytes()[split..]));
            }
        }
    }

    #[test]
    fn output_requires_complete_lf_or_crlf_event_across_every_split() {
        for event in [
            b"data: {\"choices\":[{\"text\":\"output\"}]}\n\n".as_slice(),
            b"data: {\"choices\":[{\"text\":\"output\"}]}\r\n\r\n".as_slice(),
            b"data: {\"choices\":\r\ndata: [{\"text\":\"output\"}]}\n\r\n".as_slice(),
        ] {
            // The final CR completes a CRLF delimiter before its optional LF.
            assert_output_at(event, event.len() - usize::from(event.ends_with(b"\r\n")));
        }
    }

    #[test]
    fn comments_metadata_and_invalid_events_cannot_prove_output() {
        let mut observer = CompletionOutputObserver::new(false);
        for event in [
            ": comment containing data: [DONE]\n\n",
            "\n\n",
            "data:\r\n\r\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"text\":\"\"}]}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n",
            "data: {\"error\":{},\"choices\":[{\"text\":\"not output\"}]}\n\n",
            "event: error\r\ndata: {\"choices\":[{\"text\":\"not output\"}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"chat, not completion\"}}]}\n\n",
            "data: not-json\n\n",
            "data: {\"choices\":[{\"text\":42}]}\n\n",
        ] {
            assert!(!observer.observe(event.as_bytes()), "{event:?}");
        }
        assert!(observer.observe(b"data: {\"choices\":[{\"text\":\"yes\"}]}\n\n"));
    }

    #[test]
    fn disabled_and_unterminated_events_cannot_prove_output() {
        let event = b"data: {\"choices\":[{\"text\":\"prompt echo\"}]}\n\n";
        assert!(!CompletionOutputObserver::new(true).observe(event));
        let mut observer = CompletionOutputObserver::new(false);
        assert!(!observer.observe(&event[..event.len() - 1]));
        assert!(!observer.stopped);
    }

    #[test]
    fn oversized_event_disables_detection_before_buffer_growth() {
        let pending = vec![b'x'; MAX_OBSERVABLE_BYTES];
        for mut observer in [
            CompletionOutputObserver::new(false),
            CompletionOutputObserver::for_history(HistoryOutput::Completion),
        ] {
            assert!(!observer.observe(&pending));
            assert_eq!(observer.event.len(), MAX_OBSERVABLE_BYTES);
            assert!(
                !observer.observe(b"x\n\ndata: {\"choices\":[{\"text\":\"false positive\"}]}\n\n")
            );
            assert!(observer.event.is_empty());
            assert!(observer.stopped);
            assert!(!observer.observe(b"data: {\"choices\":[{\"text\":\"still disabled\"}]}\n\n"));
        }
    }

    #[test]
    fn history_requires_route_specific_generated_text() {
        for (case, json, expected) in [
            (
                "final completion text",
                r#"{"error":null,"choices":[{"text":" ","finish_reason":"stop"}]}"#,
                [true, true, false, false],
            ),
            (
                "chat delta",
                r#"{"choices":[{"delta":{"content":"generated"}}]}"#,
                [false, false, true, false],
            ),
            (
                "chat message",
                r#"{"choices":[{"message":{"content":"generated"},"finish_reason":"length"}]}"#,
                [false, false, false, true],
            ),
            (
                "wrong types and empty text",
                r#"{"choices":[{"text":"","delta":{"content":42},"message":{"content":[]}}]}"#,
                [false; 4],
            ),
            (
                "role, reasoning, tools and prompt metadata",
                r#"{"choices":[{"prompt_token_ids":[1],"delta":{"role":"assistant","reasoning":"hidden","tool_calls":[{}]},"message":{"role":"assistant","reasoning_content":"hidden","tool_calls":[{}]}}]}"#,
                [false; 4],
            ),
            (
                "usage only",
                r#"{"choices":[],"usage":{"completion_tokens":1}}"#,
                [false; 4],
            ),
            (
                "incomplete JSON",
                r#"{"choices":[{"text":"partial"}"#,
                [false; 4],
            ),
            (
                "abort takes precedence over text in any choice",
                r#"{"choices":[{"text":"not output","delta":{"content":"not output"},"message":{"content":"not output"}},{"finish_reason":"abort"}]}"#,
                [false; 4],
            ),
            (
                "error object takes precedence over text",
                r#"{"object":"error","choices":[{"text":"not output","delta":{"content":"not output"},"message":{"content":"not output"}}]}"#,
                [false; 4],
            ),
            (
                "non-null error field takes precedence over text",
                r#"{"error":{},"choices":[{"text":"not output","delta":{"content":"not output"},"message":{"content":"not output"}}]}"#,
                [false; 4],
            ),
        ] {
            let event = format!("data: {json}\n\n");
            for ((output, stream, mode), expected) in [
                (HistoryOutput::Completion, true, "completion stream"),
                (HistoryOutput::Completion, false, "completion body"),
                (HistoryOutput::ChatDelta, true, "chat stream"),
                (HistoryOutput::ChatMessage, false, "chat body"),
            ]
            .into_iter()
            .zip(expected)
            {
                let observed = if stream {
                    CompletionOutputObserver::for_history(output).observe(event.as_bytes())
                } else {
                    output.has_output(json.as_bytes())
                };
                assert_eq!(observed, expected, "{case}: {mode}");
            }
        }
    }

    #[test]
    fn history_stops_at_terminal_frames_before_later_output() {
        let output = "data: {\"choices\":[{\"text\":\"too late\"}]}\n\n";
        for terminal in [
            "data: \t[DONE] \t\n\n",
            "event: error\ndata: {\"choices\":[{\"text\":\"not output\"}]}\n\n",
            "event: abort\n\n",
            "data: {\"error\":{}}\n\n",
            "data: {\"object\":\"error\"}\n\n",
            "data: {\"choices\":[{\"text\":\"not output\",\"finish_reason\":\"abort\"}]}\n\n",
            "data: {\"choices\":[{\"text\":\"not output\",\"finish_reason\":\"error\"}]}\n\n",
        ] {
            let mut observer = CompletionOutputObserver::for_history(HistoryOutput::Completion);
            assert!(!observer.observe(b": router-stages\n\ndata: not-json\n\n"));
            assert!(!observer.stopped);
            assert!(
                !observer.observe(format!("{terminal}{output}").as_bytes()),
                "{terminal:?}"
            );
            assert!(!observer.observe(output.as_bytes()), "{terminal:?}");
        }
        // A normal finish belongs to one choice, not the entire response.
        let mut observer = CompletionOutputObserver::for_history(HistoryOutput::Completion);
        assert!(!observer.observe(
            b"data: {\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}]}\n\n"
        ));
        assert!(observer.observe(b"data: {\"choices\":[{\"index\":1,\"text\":\"generated\"}]}\n\n"));
    }

    #[test]
    fn history_bounds_complete_json_before_parsing() {
        let mut body = br#"{"choices":[{"text":"generated"}]}"#.to_vec();
        body.resize(MAX_OBSERVABLE_BYTES, b' ');
        assert!(HistoryOutput::Completion.has_output(&body));
        body.push(b' ');
        assert!(!HistoryOutput::Completion.has_output(&body));
    }
}
