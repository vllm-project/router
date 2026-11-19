//! Active-mode peer KV fetch: the request mutation.
//!
//! The `kv_aware` policy decides (plan, source address, in-flight cap); this
//! module only writes the decision into the JSON body sent to the destination
//! engine. The router sends the source's side-channel address and nothing
//! else: no block hashes, no `remote_block_ids`, no `remote_request_id`. The
//! destination engine hashes its own prompt and learns the rest from the
//! source's reply.

use serde_json::{json, Map, Value};

use crate::policies::PeerEndpoint;

/// What happened when peer fetch params were applied to a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PeerFetchApply {
    /// The params were added.
    Applied,
    /// The client already sent `kv_transfer_params`; the body is untouched.
    ClientParamsPresent,
    /// The body is not a JSON object; it is untouched.
    InvalidBody,
}

impl PeerFetchApply {
    /// Value of the `outcome` label on `vllm_router_kv_transfer_requests_total`.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Applied => "sent",
            Self::ClientParamsPresent => "client_params_present",
            Self::InvalidBody => "invalid_body",
        }
    }
}

/// The `kv_transfer_params` object for a fetch from `source`.
fn peer_fetch_params(source: &PeerEndpoint) -> Value {
    json!({
        "do_remote_prefill": true,
        "peer_lookup": true,
        "remote_engine_id": &*source.engine_id,
        "remote_host": &*source.host,
        "remote_port": source.port,
        "tp_size": source.tp_size,
        "dcp_size": source.dcp_size,
        "pp_size": source.pp_size,
    })
}

/// Add `kv_transfer_params` for a fetch from `source` to `body`, unless the
/// client supplied its own (an explicit `null` counts as absent, as it does for
/// the engine). Pure: callers pass a fresh copy of the client's body for each
/// attempt, so a retry never inherits another destination's params.
pub(super) fn apply_peer_fetch(body: &mut Value, source: &PeerEndpoint) -> PeerFetchApply {
    let Some(object): Option<&mut Map<String, Value>> = body.as_object_mut() else {
        return PeerFetchApply::InvalidBody;
    };
    if object
        .get("kv_transfer_params")
        .is_some_and(|params| !params.is_null())
    {
        return PeerFetchApply::ClientParamsPresent;
    }
    object.insert("kv_transfer_params".to_owned(), peer_fetch_params(source));
    PeerFetchApply::Applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn endpoint() -> PeerEndpoint {
        PeerEndpoint {
            engine_id: Arc::from("engine-a"),
            host: Arc::from("10.0.0.7"),
            port: 5601,
            tp_size: 2,
            dcp_size: 1,
            pp_size: 1,
        }
    }

    #[test]
    fn adds_exactly_the_documented_fields() {
        let mut body = json!({"prompt": [1, 2, 3], "max_tokens": 1});
        assert_eq!(
            apply_peer_fetch(&mut body, &endpoint()),
            PeerFetchApply::Applied
        );
        assert_eq!(
            body,
            json!({
                "prompt": [1, 2, 3],
                "max_tokens": 1,
                "kv_transfer_params": {
                    "do_remote_prefill": true,
                    "peer_lookup": true,
                    "remote_engine_id": "engine-a",
                    "remote_host": "10.0.0.7",
                    "remote_port": 5601,
                    "tp_size": 2,
                    "dcp_size": 1,
                    "pp_size": 1,
                }
            })
        );
        let params = body["kv_transfer_params"].as_object().unwrap();
        for forbidden in ["block_hashes", "remote_block_ids", "remote_request_id"] {
            assert!(!params.contains_key(forbidden));
        }
    }

    #[test]
    fn client_params_are_never_overwritten() {
        let original = json!({"prompt": "hi", "kv_transfer_params": {"x": 1}});
        let mut body = original.clone();
        assert_eq!(
            apply_peer_fetch(&mut body, &endpoint()),
            PeerFetchApply::ClientParamsPresent
        );
        assert_eq!(body, original);
    }

    #[test]
    fn null_client_params_count_as_absent() {
        let mut body = json!({"prompt": "hi", "kv_transfer_params": null});
        assert_eq!(
            apply_peer_fetch(&mut body, &endpoint()),
            PeerFetchApply::Applied
        );
        assert_eq!(body["kv_transfer_params"]["peer_lookup"], json!(true));
    }

    #[test]
    fn non_object_bodies_are_left_alone() {
        for original in [json!([1, 2]), json!("x"), Value::Null] {
            let mut body = original.clone();
            assert_eq!(
                apply_peer_fetch(&mut body, &endpoint()),
                PeerFetchApply::InvalidBody
            );
            assert_eq!(body, original);
        }
    }

    #[test]
    fn each_attempt_starts_from_the_original_body() {
        let original = json!({"prompt": [1], "max_tokens": 1});
        let mut other = endpoint();
        other.engine_id = Arc::from("engine-b");

        let mut first = original.clone();
        apply_peer_fetch(&mut first, &endpoint());
        let mut second = original.clone();
        apply_peer_fetch(&mut second, &other);

        assert_eq!(original.get("kv_transfer_params"), None);
        assert_eq!(first["kv_transfer_params"]["remote_engine_id"], "engine-a");
        assert_eq!(second["kv_transfer_params"]["remote_engine_id"], "engine-b");
    }
}
