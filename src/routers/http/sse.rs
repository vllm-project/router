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
