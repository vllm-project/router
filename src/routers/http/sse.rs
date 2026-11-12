//! SSE framing shared by local response consumers; payload semantics stay local.

#[derive(Default)]
pub(super) struct FrameScanner {
    line_has_bytes: bool,
    pub(super) skip_lf: bool,
}

impl FrameScanner {
    // CR, LF and CRLF all terminate lines. Process CR immediately and remember
    // to ignore its optional LF, including when it arrives in a later chunk.
    pub(super) fn frame_end(&mut self, bytes: &[u8]) -> Option<usize> {
        let mut offset = 0;
        while offset < bytes.len() {
            let byte = bytes[offset];
            offset += 1;
            if std::mem::take(&mut self.skip_lf) && byte == b'\n' {
                continue;
            }
            if byte == b'\r' || byte == b'\n' {
                let blank = !self.line_has_bytes;
                self.line_has_bytes = false;
                self.skip_lf = byte == b'\r';
                if blank {
                    // Include an available LF; the PD wrapper forwards a split
                    // optional LF separately without rewriting its bytes.
                    if self.skip_lf && bytes.get(offset) == Some(&b'\n') {
                        self.skip_lf = false;
                        offset += 1;
                    }
                    return Some(offset);
                }
            } else {
                self.line_has_bytes = true;
            }
        }
        None
    }
}

/// Split complete frame lines while retaining the original line terminators.
pub(super) fn sse_lines(mut bytes: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    std::iter::from_fn(move || {
        if bytes.is_empty() {
            return None;
        }
        let end = bytes
            .iter()
            .position(|byte| *byte == b'\r' || *byte == b'\n')
            .unwrap_or(bytes.len());
        let (line, rest) = bytes.split_at(end);
        let ending_len = if rest.starts_with(b"\r\n") {
            2
        } else {
            usize::from(!rest.is_empty())
        };
        let (ending, tail) = rest.split_at(ending_len);
        bytes = tail;
        Some((line, ending))
    })
}

/// Observe explicit SSE failures without changing or delaying forwarded bytes.
#[derive(Default)]
pub(super) struct ErrorObserver {
    scanner: FrameScanner,
    frame: Vec<u8>,
    oversized: bool,
}

impl ErrorObserver {
    const MAX_FRAME_BYTES: usize = 1024 * 1024;

    pub(super) fn observe(&mut self, mut chunk: &[u8]) -> bool {
        while !chunk.is_empty() {
            let end = self.scanner.frame_end(chunk);
            let consumed = end.unwrap_or(chunk.len());
            if !self.oversized {
                if consumed > Self::MAX_FRAME_BYTES - self.frame.len() {
                    self.frame.clear();
                    self.oversized = true;
                } else {
                    self.frame.extend_from_slice(&chunk[..consumed]);
                }
            }
            chunk = &chunk[consumed..];
            if end.is_some() {
                let failed = !self.oversized && self.frame_failed();
                self.frame.clear();
                self.oversized = false;
                if failed {
                    return true;
                }
            }
        }
        false
    }

    fn frame_failed(&self) -> bool {
        // Avoid parsing ordinary token events; escaped JSON keys still need inspection.
        if !self.frame.windows(5).any(|bytes| bytes == b"error") && !self.frame.contains(&b'\\') {
            return false;
        }
        let mut data = Vec::new();
        let mut event = None;
        for (line, _) in sse_lines(
            self.frame
                .strip_prefix(b"\xef\xbb\xbf")
                .unwrap_or(&self.frame),
        ) {
            if let Some(value) = line.strip_prefix(b"event:") {
                event = Some(value.strip_prefix(b" ").unwrap_or(value));
            } else if let Some(value) = line.strip_prefix(b"data:") {
                if !data.is_empty() {
                    data.push(b'\n');
                }
                data.extend_from_slice(value.strip_prefix(b" ").unwrap_or(value));
            }
        }
        event == Some(b"error".as_slice())
            || serde_json::from_slice::<serde_json::Value>(&data).is_ok_and(|value| {
                value.get("error").is_some_and(|error| !error.is_null())
                    || value.get("type").and_then(serde_json::Value::as_str) == Some("error")
            })
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[test]
    fn error_observer_handles_fragmented_frames_and_ignores_content() {
        for frame in [
            b"event: error\r\ndata: {}\r\n\r\n".as_slice(),
            b"data: {\"error\":\ndata: \"failed\"}\n\n",
            b"data: {\"type\":\"error\"}\r\r",
            b"data: {\"\\u0065rror\":\"failed\"}\n\n",
        ] {
            for split in 0..=frame.len() {
                let mut observer = ErrorObserver::default();
                let first = observer.observe(&frame[..split]);
                assert!(first || observer.observe(&frame[split..]), "split {split}");
            }
        }
        let mut observer = ErrorObserver::default();
        assert!(
            !observer.observe(b"data: {\"choices\":[{\"text\":\"error\"}]}\n\ndata: [DONE]\n\n")
        );
        assert!(!observer.observe(b"data: {\"error\":null,\"choices\":[]}\n\n"));
        assert!(!observer.observe(b"data: {\"error\":\"unterminated\"}"));
    }

    #[test]
    fn oversized_frames_do_not_hide_later_errors() {
        let mut observer = ErrorObserver::default();
        assert!(!observer.observe(&vec![b'x'; ErrorObserver::MAX_FRAME_BYTES + 1]));
        assert!(observer.observe(b"\n\nevent: error\n\n"));
    }
}
