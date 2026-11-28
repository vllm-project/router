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

/// Parse the fields of a complete SSE event. Multiple data lines are joined
/// with a newline as specified by the event stream format.
pub(super) fn event_fields(event: &[u8]) -> (Option<&[u8]>, Vec<u8>) {
    let mut data = Vec::new();
    let mut has_data = false;
    let mut kind = None;
    for (line, _) in sse_lines(event) {
        let field = |prefix| {
            line.strip_prefix(prefix)
                .map(|value| value.strip_prefix(b" ").unwrap_or(value))
        };
        if let Some(value) = field(b"data:".as_slice()) {
            if has_data {
                data.push(b'\n');
            }
            data.extend_from_slice(value);
            has_data = true;
        } else if let Some(value) = field(b"event:".as_slice()) {
            kind = Some(value);
        }
    }
    (kind, data)
}
