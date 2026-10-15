use serde_json::Value;

#[derive(Default)]
pub(super) struct FirstTokenObserver {
    line: Vec<u8>,
    data: Vec<u8>,
    finished: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Observation {
    Pending,
    Output,
    End,
}

impl FirstTokenObserver {
    pub(super) fn observe(&mut self, bytes: &[u8]) -> Observation {
        if self.finished {
            return Observation::End;
        }
        for part in bytes.split_inclusive(|byte| *byte == b'\n') {
            if self.line.len() + self.data.len() + part.len() > 1024 * 1024 {
                self.finished = true;
                self.line.clear();
                self.data.clear();
                return Observation::End;
            }
            self.line.extend_from_slice(part);
            if part.last() != Some(&b'\n') {
                continue;
            }
            let mut line = std::mem::take(&mut self.line);
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                let data = std::mem::take(&mut self.data);
                let observation = classify(&data);
                if observation != Observation::Pending {
                    self.finished = true;
                    return observation;
                }
            } else if let Some(data) = line.strip_prefix(b"data:") {
                if !self.data.is_empty() {
                    self.data.push(b'\n');
                }
                self.data
                    .extend_from_slice(data.strip_prefix(b" ").unwrap_or(data));
            }
        }
        Observation::Pending
    }
}

fn nonempty(value: Option<&Value>) -> bool {
    value.and_then(Value::as_str).is_some_and(|s| !s.is_empty())
}

fn function_output(value: &Value) -> bool {
    nonempty(value.get("name")) || nonempty(value.get("arguments"))
}

fn classify(data: &[u8]) -> Observation {
    if data == b"[DONE]" {
        return Observation::End;
    }
    let Ok(value) = serde_json::from_slice::<Value>(data) else {
        return Observation::Pending;
    };
    if value.get("error").is_some_and(|error| !error.is_null()) {
        return Observation::End;
    }
    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return Observation::Pending;
    };
    for choice in choices {
        if nonempty(choice.get("text"))
            || choice
                .get("token_ids")
                .and_then(Value::as_array)
                .is_some_and(|ids| !ids.is_empty())
        {
            return Observation::Output;
        }
        if let Some(delta) = choice.get("delta") {
            if ["content", "reasoning", "reasoning_content", "refusal"]
                .iter()
                .any(|field| nonempty(delta.get(field)))
                || delta.get("function_call").is_some_and(function_output)
                || delta
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .is_some_and(|calls| {
                        calls
                            .iter()
                            .any(|call| call.get("function").is_some_and(function_output))
                    })
            {
                return Observation::Output;
            }
        }
    }
    Observation::Pending
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_generated_content_across_supported_stream_formats() {
        for data in [
            r#"{"choices":[{"text":"hello"}]}"#,
            r#"{"choices":[{"token_ids":[17],"text":""}]}"#,
            r#"{"choices":[{"delta":{"content":"hello"}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"thinking"}}]}"#,
            r#"{"choices":[{"delta":{"reasoning":"thinking"}}]}"#,
            r#"{"choices":[{"delta":{"refusal":"cannot comply"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{"}}]}}]}"#,
            r#"{"choices":[{"delta":{"function_call":{"name":"lookup"}}}]}"#,
        ] {
            let mut observer = FirstTokenObserver::default();
            assert_eq!(
                observer.observe(format!("data: {data}\n\n").as_bytes()),
                Observation::Output,
                "{data}",
            );
        }
    }

    #[test]
    fn handles_multiline_events_without_sampling_metadata() {
        let mut observer = FirstTokenObserver::default();
        assert_eq!(
            observer.observe(b"event: message\nid: 1\n: keepalive\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":0}}\n\n"),
            Observation::Pending,
        );
        assert_eq!(
            observer.observe(b"data: {\"choices\":\n"),
            Observation::Pending
        );
        assert_eq!(
            observer.observe(b"data: [{\"text\":\"hello\"}]}\n\n"),
            Observation::Output,
        );
    }

    #[test]
    fn terminators_and_split_errors_disable_later_sampling() {
        for event in ["data: {\"error\":\"failed\"}\n\n", "data: [DONE]\n\n"] {
            let mut observer = FirstTokenObserver::default();
            for byte in &event.as_bytes()[..event.len() - 1] {
                assert_eq!(observer.observe(&[*byte]), Observation::Pending);
            }
            assert_eq!(observer.observe(b"\n"), Observation::End);
            assert_eq!(
                observer.observe(b"data: {\"choices\":[{\"text\":\"late\"}]}\n\n"),
                Observation::End,
            );
        }
    }

    #[test]
    fn one_finished_choice_does_not_end_other_choices() {
        let finished =
            b"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
        let output = b"data: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n";
        let mut observer = FirstTokenObserver::default();
        assert_eq!(observer.observe(finished), Observation::Pending);
        assert_eq!(observer.observe(output), Observation::Output);

        let mut observer = FirstTokenObserver::default();
        assert_eq!(
            observer.observe(&[finished.as_slice(), output.as_slice()].concat()),
            Observation::Output
        );
    }

    #[test]
    fn oversized_partial_events_disable_sampling() {
        let mut observer = FirstTokenObserver::default();
        assert_eq!(
            observer.observe(&vec![b'x'; 1024 * 1024 + 1]),
            Observation::End
        );
        assert_eq!(
            observer.observe(b"data: {\"choices\":[{\"text\":\"late\"}]}\n\n"),
            Observation::End,
        );
    }
}
