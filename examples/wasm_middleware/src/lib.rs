//! Example OnRequest middleware for vLLM Router.
//!
//! Demo behavior (not product logic):
//! - Reject(400) when the body contains the ASCII marker `__wasm_reject__`
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
        if body_contains_marker(&req.body, b"__wasm_reject__") {
            return Action::Reject(400);
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

export!(Component);
