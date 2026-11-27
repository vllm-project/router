//! Conservative model-output detection for proxied Completion SSE streams,
//! and for any OpenAI-style generation stream that holds a peer fetch slot.

use super::sse::{sse_lines, FrameScanner};

const MAX_EVENT_BYTES: usize = 1024 * 1024;

/// What a complete event must carry to count as model output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputRule {
    /// A Completion choice with non-empty `text`.
    CompletionText,
    /// A Completion choice with non-empty `text`, or a chat choice whose
    /// `delta` has non-empty `content`, `reasoning_content` or `reasoning`, or
    /// a non-empty `tool_calls` list. A role-only delta, a usage-only event, a
    /// finish without text and SSE comments (heartbeats) are not output.
    GeneratedOutput,
}

pub(super) struct CompletionOutputObserver {
    event: Vec<u8>,
    scanner: FrameScanner,
    first_event: bool,
    stopped: bool,
    rule: OutputRule,
}

impl CompletionOutputObserver {
    pub(super) fn new(disabled: bool) -> Self {
        Self::with_rule(disabled, OutputRule::CompletionText)
    }

    /// Detects generated output on a Completion or chat stream. Used to free a
    /// peer fetch source slot once prefill (and so the fetch) is over; a stream
    /// in a shape this does not recognize keeps the slot until the body ends.
    pub(super) fn generated_output() -> Self {
        Self::with_rule(false, OutputRule::GeneratedOutput)
    }

    fn with_rule(disabled: bool, rule: OutputRule) -> Self {
        Self {
            event: Vec::new(),
            scanner: FrameScanner::default(),
            first_event: true,
            stopped: disabled,
            rule,
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
            let scan_len = chunk.len().min(MAX_EVENT_BYTES - self.event.len() + 1);
            let end = self.scanner.frame_end(&chunk[..scan_len]);
            let consumed = end.unwrap_or(scan_len);
            if consumed > MAX_EVENT_BYTES - self.event.len() {
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
                let output = event_has_output(event, self.rule);
                self.event.clear();
                if output {
                    self.stopped = true;
                    return true;
                }
            }
        }
        false
    }
}

fn event_has_output(event: &[u8], rule: OutputRule) -> bool {
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
    if event_type == Some(b"error".as_slice()) {
        return false;
    }
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&data) else {
        return false;
    };
    value.get("error").is_none()
        && value
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|choices| choices.iter().any(|choice| choice_has_output(choice, rule)))
}

fn choice_has_output(choice: &serde_json::Value, rule: OutputRule) -> bool {
    let non_empty_str = |value: Option<&serde_json::Value>| {
        value
            .and_then(serde_json::Value::as_str)
            .is_some_and(|text| !text.is_empty())
    };
    if non_empty_str(choice.get("text")) {
        return true;
    }
    if rule == OutputRule::CompletionText {
        return false;
    }
    let Some(delta) = choice.get("delta") else {
        return false;
    };
    ["content", "reasoning_content", "reasoning"]
        .iter()
        .any(|field| non_empty_str(delta.get(*field)))
        || delta
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|calls| !calls.is_empty())
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn assert_output_at(event: &[u8], complete_at: usize) {
        for split in 0..=event.len() {
            let mut observer = CompletionOutputObserver::new(false);
            assert_eq!(
                observer.observe(&event[..split]),
                split >= complete_at,
                "split {split}"
            );
            assert_eq!(
                observer.observe(&event[split..]),
                split < complete_at,
                "split {split}"
            );
            assert!(!observer.observe(event), "output transition is idempotent");
        }
        let mut observer = CompletionOutputObserver::new(false);
        for (index, byte) in event.iter().enumerate() {
            assert_eq!(
                observer.observe(&[*byte]),
                index + 1 == complete_at,
                "byte {index}"
            );
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
    fn generated_output_counts_chat_deltas_but_not_heartbeats_or_metadata() {
        for event in [
            ": keepalive\n\n",
            "event: ping\ndata: {}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n",
            "data: {\"error\":{},\"choices\":[{\"delta\":{\"content\":\"no\"}}]}\n\n",
            "event: error\ndata: {\"choices\":[{\"delta\":{\"content\":\"no\"}}]}\n\n",
            "data: [DONE]\n\n",
        ] {
            let mut observer = CompletionOutputObserver::generated_output();
            assert!(!observer.observe(event.as_bytes()), "{event:?}");
        }
        for event in [
            "data: {\"choices\":[{\"text\":\"completion\"}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"chat\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0}]}}]}\n\n",
        ] {
            let mut observer = CompletionOutputObserver::generated_output();
            assert!(observer.observe(event.as_bytes()), "{event:?}");
        }
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
        let mut observer = CompletionOutputObserver::new(false);
        assert!(!observer.observe(&vec![b'x'; MAX_EVENT_BYTES]));
        assert_eq!(observer.event.len(), MAX_EVENT_BYTES);
        assert!(!observer.observe(b"x\n\ndata: {\"choices\":[{\"text\":\"false positive\"}]}\n\n"));
        assert!(observer.event.is_empty());
        assert!(observer.stopped);
        assert!(!observer.observe(b"data: {\"choices\":[{\"text\":\"still disabled\"}]}\n\n"));
    }
}
