//! Example OnRequest middleware for vLLM Router.
//!
//! Demo behavior (not product logic):
//! - Reject(N) when the body contains `__wasm_reject_N__` (e.g. `__wasm_reject_403__`)
//! - Reject(400) when the body contains `__wasm_reject__`
//! - Infinite loop when the body contains `__wasm_loop__` (for host timeout tests)
//! - Otherwise Modify: set `x-wasm-middleware: example` and leave the body unchanged

#![allow(clippy::missing_safety_doc)]

wit_bindgen::generate!({
    path: "../../wit",
    world: "middleware",
});

use exports::vllm::router_middleware::on_request::{Guest, Request};
use vllm::router_middleware::types::{Action, Header, ModifyAction};

struct Component;

impl Guest for Component {
    fn handle(req: Request) -> Action {
        if body_contains_marker(&req.body, b"__wasm_loop__") {
            // Intentional spin for host epoch-deadline tests. Do not use in production plugins.
            loop {}
        }

        if let Some(status) = parse_reject_status(&req.body) {
            return Action::Reject(status);
        }

        Action::Modify(ModifyAction {
            headers_set: vec![Header {
                name: "x-wasm-middleware".to_string(),
                value: b"example".to_vec(),
            }],
            headers_add: Vec::new(),
            headers_remove: Vec::new(),
            body_replace: None,
        })
    }
}

fn body_contains_marker(body: &[u8], marker: &[u8]) -> bool {
    body.windows(marker.len()).any(|window| window == marker)
}

/// Parse `__wasm_reject__` (defaults to 400) or `__wasm_reject_NNN__`.
fn parse_reject_status(body: &[u8]) -> Option<u16> {
    const PREFIX: &[u8] = b"__wasm_reject_";
    const BARE: &[u8] = b"__wasm_reject__";

    if let Some(start) = find_subslice(body, PREFIX) {
        let rest = &body[start + PREFIX.len()..];
        let digits: Vec<u8> = rest.iter().copied().take_while(|b| b.is_ascii_digit()).collect();
        if rest.get(digits.len()..)
            .is_some_and(|tail| tail.starts_with(b"__"))
            && !digits.is_empty()
        {
            if let Ok(text) = std::str::from_utf8(&digits) {
                if let Ok(status) = text.parse::<u16>() {
                    return Some(status);
                }
            }
        }
    }

    if body_contains_marker(body, BARE) {
        return Some(400);
    }
    None
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

export!(Component);
