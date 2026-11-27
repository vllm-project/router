//! Observe public OpenAI HTTP response bodies without changing their framing.

use std::time::Instant;

use bytes::Bytes;
use metrics::histogram;
use serde_json::Value;

use super::sse::{event_fields, FrameScanner};

const BODY_LIMIT: usize = 1024 * 1024;
pub(super) type Usage = (Option<u64>, Option<u64>);

pub(super) enum Observer {
    Sse(Sse),
    Json(Option<Vec<u8>>),
}

impl Observer {
    pub(super) fn new(route: &'static str, sse: bool, echo: bool) -> Option<Self> {
        matches!(
            route,
            "/v1/completions" | "/v1/chat/completions" | "/v1/responses"
        )
        .then(|| {
            if sse {
                Self::Sse(Sse::new(route, echo))
            } else {
                Self::Json(Some(Vec::new()))
            }
        })
    }

    pub(super) fn observe(&mut self, bytes: &Bytes, started: Instant, at: Instant) {
        match self {
            Self::Sse(sse) => sse.observe(bytes, started, at),
            Self::Json(body) => {
                if let Some(buf) = body {
                    if bytes.len() <= BODY_LIMIT - buf.len() {
                        buf.extend_from_slice(bytes);
                    } else {
                        *body = None;
                    }
                }
            }
        }
    }

    pub(super) fn terminal(&self) -> bool {
        matches!(self, Self::Sse(sse) if sse.terminal || sse.failed)
    }

    pub(super) fn finish(self, route: &'static str, hidden_usage: Option<Usage>) {
        let observed = match self {
            Self::Sse(sse) => sse
                .terminal
                .then_some(sse)
                .filter(|sse| !sse.failed)
                .and_then(|sse| sse.usage.or(hidden_usage)),
            Self::Json(body) => body
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .filter(|value| !is_error(value) && !bad_response_status(value))
                .and_then(|value| value.get("usage").and_then(usage)),
        };
        if let Some((input, output)) = observed {
            if let Some(input) = input {
                histogram!("vllm_router_input_tokens", "route" => route).record(input as f64);
            }
            if let Some(output) = output {
                histogram!("vllm_router_output_tokens", "route" => route).record(output as f64);
            }
        }
    }
}

pub(super) fn usage(value: &Value) -> Option<Usage> {
    let input = value
        .get("prompt_tokens")
        .or_else(|| value.get("input_tokens"))
        .and_then(Value::as_u64);
    let output = value
        .get("completion_tokens")
        .or_else(|| value.get("output_tokens"))
        .and_then(Value::as_u64);
    (input.is_some() || output.is_some()).then_some((input, output))
}

fn is_error(value: &Value) -> bool {
    value.get("error").is_some_and(|err| !err.is_null())
        || value.get("object").and_then(Value::as_str) == Some("error")
}

fn bad_response_status(value: &Value) -> bool {
    matches!(
        value.get("status").and_then(Value::as_str),
        Some("failed" | "cancelled" | "incomplete")
    )
}

pub(super) struct Sse {
    route: &'static str,
    echo: bool,
    scanner: FrameScanner,
    event: Vec<u8>,
    oversized: bool,
    first_event: bool,
    first_output: bool,
    last_single: Option<(Instant, u64)>,
    reset_interval: bool,
    terminal: bool,
    failed: bool,
    usage: Option<Usage>,
}

#[derive(Clone, Copy)]
enum Output {
    None,
    Uncounted,
    Single(u64),
}

impl Sse {
    fn new(route: &'static str, echo: bool) -> Self {
        Self {
            route,
            echo,
            scanner: FrameScanner::default(),
            event: Vec::new(),
            oversized: false,
            first_event: true,
            first_output: false,
            last_single: None,
            reset_interval: false,
            terminal: false,
            failed: false,
            usage: None,
        }
    }

    fn observe(&mut self, mut bytes: &[u8], started: Instant, at: Instant) {
        let mut batch = Output::None;
        self.reset_interval = false;
        while !bytes.is_empty() {
            let end = self.scanner.frame_end(bytes);
            let len = end.unwrap_or(bytes.len());
            if !self.oversized && len <= BODY_LIMIT - self.event.len() {
                self.event.extend_from_slice(&bytes[..len]);
            } else {
                self.oversized = true;
                self.event.clear();
            }
            bytes = &bytes[len..];
            if end.is_some() {
                let first_event = std::mem::take(&mut self.first_event);
                if std::mem::take(&mut self.oversized) {
                    // The omitted event could hide output or a failure.
                    self.failed = true;
                    self.last_single = None;
                    continue;
                }
                let event = std::mem::take(&mut self.event);
                let fields = if first_event {
                    event.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&event)
                } else {
                    &event
                };
                let output = self.event(fields);
                self.event = event;
                self.event.clear();
                batch = match (batch, output) {
                    (output, Output::None) | (Output::None, output) => output,
                    _ => Output::Uncounted,
                };
            }
        }
        if !matches!(batch, Output::None) {
            if !self.first_output {
                self.first_output = true;
                histogram!("vllm_router_ttft_seconds", "route" => self.route, "stream" => "true")
                    .record(at.duration_since(started).as_secs_f64());
            }
            let current = match (batch, self.reset_interval) {
                (Output::Single(index), false) => Some((at, index)),
                _ => None,
            };
            if let (Some((previous, old_index)), Some((now, index))) = (self.last_single, current) {
                if old_index == index && now > previous {
                    histogram!("vllm_router_itl_seconds", "route" => self.route)
                        .record(now.duration_since(previous).as_secs_f64());
                }
            }
            self.last_single = current;
        }
    }

    fn event(&mut self, event: &[u8]) -> Output {
        let (kind, data) = event_fields(event);
        let failed = |kind| {
            matches!(
                kind,
                "error" | "response.failed" | "response.incomplete" | "response.cancelled"
            )
        };
        if kind
            .and_then(|kind| std::str::from_utf8(kind).ok())
            .is_some_and(failed)
        {
            self.failed = true;
            return Output::None;
        }
        if data.is_empty() {
            return Output::None;
        }
        if data == b"[DONE]" && self.route != "/v1/responses" {
            self.terminal = true;
            return Output::None;
        }
        let Ok(value) = serde_json::from_slice::<Value>(&data) else {
            // A malformed data event is not successful stream evidence.
            self.failed = true;
            return Output::None;
        };
        let event_type = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if is_error(&value) || failed(event_type) {
            self.failed = true;
            return Output::None;
        }
        if self.failed || self.terminal {
            return Output::None;
        }
        if self.route == "/v1/responses" {
            if event_type == "response.completed" {
                let response = value.get("response");
                self.failed = response.is_some_and(|r| bad_response_status(r) || is_error(r));
                self.terminal = true;
                self.usage = response.and_then(|r| r.get("usage")).and_then(usage);
            } else if matches!(
                event_type,
                "response.output_text.delta"
                    | "response.refusal.delta"
                    | "response.reasoning_text.delta"
                    | "response.reasoning_summary_text.delta"
                    | "response.function_call_arguments.delta"
            ) && nonempty(value.get("delta"))
                && !self.echo
            {
                return Output::Uncounted;
            }
            return Output::None;
        }
        let Some(choices) = value.get("choices").and_then(Value::as_array) else {
            return Output::None;
        };
        // Only the terminal usage-only chunk is authoritative. Do not take
        // running totals embedded in choice frames or combine snapshots.
        if choices.is_empty() {
            if let Some(usage) = value.get("usage").and_then(usage) {
                self.usage = Some(usage);
            }
        }
        if !choices.is_empty() {
            self.usage = None;
        }
        if self.echo {
            return Output::None;
        }
        let mut generated = choices.iter().filter(|choice| {
            if self.route == "/v1/completions" {
                nonempty(choice.get("text"))
            } else {
                let Some(delta) = choice.get("delta") else {
                    return false;
                };
                ["content", "reasoning_content", "reasoning", "refusal"]
                    .iter()
                    .any(|key| nonempty(delta.get(*key)))
                    || nonempty(delta.pointer("/function_call/arguments"))
                    || delta
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .is_some_and(|tools| {
                            tools
                                .iter()
                                .any(|tool| nonempty(tool.pointer("/function/arguments")))
                        })
            }
        });
        let Some(choice) = generated.next() else {
            // An explicitly timed token without a visible delta must not be
            // bridged by an interval between the surrounding output events.
            if choices.iter().any(|choice| {
                choice
                    .get("token_ids")
                    .and_then(Value::as_array)
                    .is_some_and(|ids| !ids.is_empty())
            }) {
                self.last_single = None;
                self.reset_interval = true;
            }
            return Output::None;
        };
        if choices.len() == 1 && generated.next().is_none() {
            if let (Some(index), true) = (
                choice.get("index").and_then(Value::as_u64),
                single_token(choice, self.route),
            ) {
                return Output::Single(index);
            }
        }
        Output::Uncounted
    }
}

fn nonempty(value: Option<&Value>) -> bool {
    value.and_then(Value::as_str).is_some_and(|s| !s.is_empty())
}

fn single_token(choice: &Value, route: &str) -> bool {
    if let Some(ids) = choice.get("token_ids").and_then(Value::as_array) {
        return ids.len() == 1 && ids[0].is_u64();
    }
    if route == "/v1/completions" {
        choice
            .pointer("/logprobs/tokens")
            .and_then(Value::as_array)
            .is_some_and(|tokens| tokens.len() == 1 && tokens[0].is_string())
    } else {
        // Chat logprobs for visible content or refusal, but never infer
        // token counts for a combined reasoning/tool/content event from them.
        let Some(delta) = choice.get("delta") else {
            return false;
        };
        if [
            "reasoning",
            "reasoning_content",
            "function_call",
            "tool_calls",
        ]
        .iter()
        .any(|key| delta.get(*key).is_some_and(|v| !v.is_null()))
        {
            return false;
        }
        ["content", "refusal"]
            .into_iter()
            .filter_map(|key| choice.get("logprobs")?.get(key)?.as_array())
            .flatten()
            .take(2)
            .count()
            == 1
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn setup(route: &'static str, echo: bool) -> Sse {
        Sse::new(route, echo)
    }

    #[test]
    fn classifies_chat_output_without_counting_metadata_or_ambiguous_tokens() {
        let mut sse = setup("/v1/chat/completions", false);
        assert!(matches!(
            sse.event(br#"data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#),
            Output::None
        ));
        assert!(matches!(sse.event(br#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call","function":{"name":"lookup","arguments":""}}]}}]}"#), Output::None));
        assert!(matches!(sse.event(br#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"{"}}]}}]}"#), Output::Uncounted));
        assert!(matches!(sse.event(br#"data: {"choices":[{"index":0,"delta":{"reasoning_content":"think"},"token_ids":[12]}]}"#), Output::Single(0)));
        assert!(matches!(sse.event(br#"data: {"choices":[{"index":0,"delta":{"content":"many words"},"token_ids":[13,14]}]}"#), Output::Uncounted));
        assert!(matches!(sse.event(br#"data: {"choices":[{"index":0,"delta":{"content":"yes"},"logprobs":{"content":[{"token":"yes"}]}}]}"#), Output::Single(0)));
    }

    #[test]
    fn skips_echo_and_uses_responses_terminal_usage() {
        let mut sse = setup("/v1/completions", true);
        assert!(matches!(sse.event(br#"data: {"choices":[{"index":0,"text":"echo and maybe output","token_ids":[7]}]}"#), Output::None));
        let mut sse = setup("/v1/responses", false);
        assert!(matches!(
            sse.event(br#"data: {"type":"response.output_item.added","item":{"type":"message"}}"#),
            Output::None
        ));
        assert!(matches!(
            sse.event(br#"data: {"type":"response.output_text.delta","delta":"hello"}"#),
            Output::Uncounted
        ));
        sse.event(br#"data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":1}}}"#);
        assert!(sse.terminal);
        assert_eq!(sse.usage, Some((Some(3), Some(1))));
        sse.event(b"event: response.failed");
        assert!(sse.failed);
    }
}
