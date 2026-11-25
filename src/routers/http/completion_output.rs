//! Bounded HTTP output observation and Completion load detection.

use std::ops::ControlFlow;

use serde_json::Value;

use super::sse::{sse_lines, FrameScanner};

const MAX_OBSERVABLE_BYTES: usize = 1024 * 1024;

pub(super) fn observe_body_with(
    body: &[u8],
    predicate: impl FnOnce(Option<&[u8]>, &[u8]) -> ControlFlow<bool>,
) -> bool {
    body.len() <= MAX_OBSERVABLE_BYTES && matches!(predicate(None, body), ControlFlow::Break(true))
}

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

    pub(super) fn observe(&mut self, chunk: &[u8]) -> bool {
        self.observe_with(chunk, inspect_completion_load)
    }

    /// Return true only once, after a complete SSE event proves model output.
    /// No EOF flush: an unterminated event is not evidence. Oversized events
    /// permanently disable detection for this attempt without affecting bytes.
    /// The bound counts wire bytes, including CRLF delimiters.
    pub(super) fn observe_with(
        &mut self,
        mut chunk: &[u8],
        mut predicate: impl FnMut(Option<&[u8]>, &[u8]) -> ControlFlow<bool>,
    ) -> bool {
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
                let observation = inspect_event(event, &mut predicate);
                self.event.clear();
                if let ControlFlow::Break(output) = observation {
                    self.stopped = true;
                    return output;
                }
            }
        }
        false
    }
}

fn inspect_event(
    event: &[u8],
    predicate: impl FnOnce(Option<&[u8]>, &[u8]) -> ControlFlow<bool>,
) -> ControlFlow<bool> {
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
    predicate(event_type, &data)
}

fn inspect_completion_load(event_type: Option<&[u8]>, data: &[u8]) -> ControlFlow<bool> {
    if event_type == Some(b"error".as_slice()) {
        return ControlFlow::Continue(());
    }
    let Ok(value) = serde_json::from_slice::<Value>(data) else {
        return ControlFlow::Continue(());
    };
    if value.get("error").is_none()
        && value
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                choices.iter().any(|choice| {
                    choice
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.is_empty())
                })
            })
    {
        ControlFlow::Break(true)
    } else {
        ControlFlow::Continue(())
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn observe_callback(observer: &mut CompletionOutputObserver, chunk: &[u8]) -> bool {
        observer.observe_with(chunk, |event, data| {
            assert!(event.is_none());
            let expected = br#""text":"output""#;
            if data.windows(expected.len()).any(|win| win == expected) {
                ControlFlow::Break(true)
            } else {
                ControlFlow::Continue(())
            }
        })
    }

    fn assert_output_at(event: &[u8], complete_at: usize) {
        for (mode, observe) in [
            (
                "load",
                CompletionOutputObserver::observe
                    as fn(&mut CompletionOutputObserver, &[u8]) -> bool,
            ),
            ("callback", observe_callback),
        ] {
            for split in 0..=event.len() {
                let mut observer = CompletionOutputObserver::new(false);
                assert_eq!(
                    observe(&mut observer, &event[..split]),
                    split >= complete_at,
                    "{mode}, split {split}"
                );
                assert_eq!(
                    observe(&mut observer, &event[split..]),
                    split < complete_at,
                    "{mode}, split {split}"
                );
                assert!(!observe(&mut observer, event), "{mode}: repeated output");
            }
            let mut observer = CompletionOutputObserver::new(false);
            for (index, byte) in event.iter().enumerate() {
                assert_eq!(
                    observe(&mut observer, &[*byte]),
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
        let mut observer = CompletionOutputObserver::new(false);
        let unexpected = |_: Option<&[u8]>, _: &[u8]| panic!("oversized event reached callback");
        assert!(!observer.observe_with(&pending, unexpected));
        assert_eq!(observer.event.len(), MAX_OBSERVABLE_BYTES);
        assert!(!observer.observe_with(b"x\n\ndata: too late\n\n", unexpected));
        assert!(observer.event.is_empty());
        assert!(observer.stopped);
        assert!(!observer.observe_with(b"data: still disabled\n\n", unexpected));
    }

    #[test]
    fn callback_can_stop_without_evidence_and_skip_later_events() {
        let mut observer = CompletionOutputObserver::new(false);
        let inspect = |event: Option<&[u8]>, data: &[u8]| match (event, data) {
            (None, b"wait") => ControlFlow::Continue(()),
            (Some(b"stop"), b"first\nsecond") => ControlFlow::Break(false),
            _ => panic!("stopped observer inspected another event"),
        };
        assert!(!observer.observe_with(b"data: wait\n\n", inspect));
        assert!(!observer.stopped);
        assert!(!observer.observe_with(
            b"event: stop\r\ndata: first\r\ndata: second\r\n\r\ndata: too late\n\n",
            inspect,
        ));
        assert!(observer.stopped);
        assert!(!observer.observe_with(b"data: still stopped\n\n", inspect));
    }

    #[test]
    fn complete_body_observation_is_bounded_before_callback() {
        let mut body = vec![b' '; MAX_OBSERVABLE_BYTES];
        assert!(observe_body_with(&body, |event, bytes| {
            assert!(event.is_none());
            assert_eq!(bytes, body.as_slice());
            ControlFlow::Break(true)
        }));
        body.push(b' ');
        assert!(!observe_body_with(&body, |_, _| panic!(
            "oversized body reached callback"
        )));
    }
}
