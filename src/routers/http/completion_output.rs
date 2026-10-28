//! Conservative model-output detection for proxied Completion SSE streams.

use super::sse::{sse_lines, FrameScanner};

const MAX_EVENT_BYTES: usize = 1024 * 1024;

pub(super) struct CompletionOutputObserver {
    event: Vec<u8>,
    scanner: FrameScanner,
    first_event: bool,
    stopped: bool,
}

impl CompletionOutputObserver {
    pub(super) fn new(disabled: bool) -> Self {
        Self {
            event: Vec::new(),
            scanner: FrameScanner::default(),
            first_event: true,
            stopped: disabled,
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
                let output = event_has_output(event);
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

fn event_has_output(event: &[u8]) -> bool {
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
            .is_some_and(|choices| {
                choices.iter().any(|choice| {
                    choice
                        .get("text")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|text| !text.is_empty())
                })
            })
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
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
