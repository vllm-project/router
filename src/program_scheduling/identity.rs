//! Canonical Program identity parsing at the Router boundary.
//!
//! The parser preserves the existing AgentInfer Router precedence: canonical
//! `vllm_xargs.agentic_context`, then `agent_hint`, then supported framework
//! headers. Request and trace identifiers never become Program identifiers.

use super::ScheduleError;
use crate::policies::{hash_key, RequestHeaders};
use http::HeaderMap;
use serde_json::{Map as JsonMap, Value as JsonValue};

const MAX_IDENTITY_COMPONENT_BYTES: usize = 256;
const MAX_PLACEMENT_HASH_KEY_BYTES: usize = 512;
const MAX_AGENTIC_CONTEXT_BYTES: usize = 16 * 1024;
const CLAUDE_SESSION_HEADER: &str = "x-claude-code-session-id";
const CLAUDE_AGENT_HEADER: &str = "x-claude-code-agent-id";
const CODEX_SESSION_HEADER: &str = "session-id";
const CODEX_THREAD_HEADER: &str = "thread-id";
const GENERIC_SESSION_HEADER: &str = "x-session-id";

/// Stable scheduling identity for one Program generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramIdentity {
    model_pool: String,
    program_id: String,
    /// Exact key consumed by the standalone Consistent Hash policy.
    placement_hash_key: String,
    /// Stable task/session group used by Program-scoped scheduling.
    placement_key: String,
    expected_resume: bool,
}

impl ProgramIdentity {
    pub(crate) fn for_rebinding(
        reference: &super::ProgramRef,
        placement_hash_key: String,
        placement_key: String,
        expected_resume: bool,
    ) -> Self {
        Self {
            model_pool: reference.model_pool().to_string(),
            program_id: reference.program_id().to_string(),
            placement_hash_key,
            placement_key,
            expected_resume,
        }
    }

    /// Resolve the existing Program identity contract at the Router boundary.
    ///
    /// `headers` contains the framework HTTP headers, `request` is the
    /// serialized generation request, and `model_pool` scopes identical
    /// Program IDs belonging to different served models.
    pub fn from_request(
        headers: Option<&HeaderMap>,
        request: Option<&JsonValue>,
        model_pool: Option<&str>,
    ) -> Result<Option<Self>, ScheduleError> {
        let request_object = request.and_then(JsonValue::as_object);
        let identity = if let Some(context) = canonical_context(request_object)? {
            canonical_identity(&context, headers, model_pool)?
        } else if let Some(identity) = agent_hint_identity(request_object, model_pool)? {
            Some(identity)
        } else {
            let (session_id, actor_id) = framework_identity_headers(headers)?;
            match session_id {
                Some(session_id) => {
                    let actor_id = normalized_agent_component(Some(&session_id), actor_id)?;
                    let expected_resume = actor_id
                        .as_deref()
                        .is_none_or(|actor_id| actor_id == "lead");
                    let program_id = scoped_program_id(&session_id, actor_id.as_deref());
                    Some(Self::build(
                        program_id,
                        session_id,
                        expected_resume,
                        model_pool,
                    )?)
                }
                None => None,
            }
        };
        let Some(identity) = identity else {
            return Ok(None);
        };
        identity
            .with_placement_hash_key(consistent_hash_request_key(headers, request))
            .map(Some)
    }

    fn build(
        program_id: String,
        placement_key: String,
        expected_resume: bool,
        model_pool: Option<&str>,
    ) -> Result<Self, ScheduleError> {
        let model_pool = model_pool.unwrap_or("default").to_string();
        for (name, value) in [
            ("program_id", program_id.as_str()),
            ("placement key", placement_key.as_str()),
            ("model pool", model_pool.as_str()),
        ] {
            if value.len() > MAX_IDENTITY_COMPONENT_BYTES {
                return Err(ScheduleError::InvalidIdentity(format!(
                    "{name} exceeds {MAX_IDENTITY_COMPONENT_BYTES} bytes"
                )));
            }
        }
        Ok(Self {
            model_pool,
            program_id,
            placement_hash_key: placement_key.clone(),
            placement_key,
            expected_resume,
        })
    }

    fn with_placement_hash_key(
        mut self,
        placement_hash_key: String,
    ) -> Result<Self, ScheduleError> {
        if placement_hash_key.len() > MAX_PLACEMENT_HASH_KEY_BYTES {
            return Err(ScheduleError::InvalidIdentity(format!(
                "placement hash key exceeds {MAX_PLACEMENT_HASH_KEY_BYTES} bytes"
            )));
        }
        self.placement_hash_key = placement_hash_key;
        Ok(self)
    }

    /// Model pool that scopes this Program ID.
    pub fn model_pool(&self) -> &str {
        &self.model_pool
    }

    /// Stable Program ID across requests in one live generation.
    pub fn program_id(&self) -> &str {
        &self.program_id
    }

    /// Stable key used by Program initial-binding policies.
    pub fn placement_hash_key(&self) -> &str {
        &self.placement_hash_key
    }

    /// Task/session group used by Program-scoped scheduling.
    pub fn placement_key(&self) -> &str {
        &self.placement_key
    }

    /// Whether upstream expects another model request after this round.
    pub fn expected_resume(&self) -> bool {
        self.expected_resume
    }
}

fn consistent_hash_request_key(headers: Option<&HeaderMap>, request: Option<&JsonValue>) -> String {
    if let Some(headers) = headers {
        for name in [
            CLAUDE_SESSION_HEADER,
            CODEX_SESSION_HEADER,
            GENERIC_SESSION_HEADER,
        ] {
            if let Ok(Some(value)) = read_nonempty_header(headers, name) {
                return format!("header:{name}:{value}");
            }
        }
    }
    let request_headers = headers.map(|headers| {
        headers
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_lowercase(), value.to_string()))
            })
            .collect::<RequestHeaders>()
    });
    if let Some(key) = request_headers
        .as_ref()
        .and_then(hash_key::extract_hash_key_from_headers)
    {
        return key;
    }
    let request_text = request.and_then(|request| serde_json::to_string(request).ok());
    hash_key::extract_hash_key(request_text.as_deref(), None)
}

fn canonical_identity(
    context: &JsonMap<String, JsonValue>,
    headers: Option<&HeaderMap>,
    model_pool: Option<&str>,
) -> Result<Option<ProgramIdentity>, ScheduleError> {
    let (header_session_id, header_agent_id) = framework_identity_headers(headers)?;
    let explicit_program_id = context_string(context, &["program_id"])?;
    let body_task_id = context_string(context, &["task_id"])?;
    let session_id = context_string(context, &["session_id"])?.or(header_session_id);
    let task_id = if body_task_id.is_some()
        || (explicit_program_id.is_some() && context.contains_key("task_id"))
    {
        body_task_id
    } else {
        session_id.clone()
    };

    let mut agent_id = context_string(context, &["agent_id"])?.or(header_agent_id);
    if agent_id.is_none() && task_id.is_some() {
        agent_id = Some("lead".to_string());
    }
    agent_id = normalized_agent_component(task_id.as_deref(), agent_id)?;
    let agent_role = context_string(context, &["agent_role"])?.or_else(|| {
        agent_id.as_ref().map(|agent_id| {
            if agent_id == "lead" {
                "lead".to_string()
            } else {
                "subagent".to_string()
            }
        })
    });
    let program_id =
        explicit_program_id.or_else(|| match (task_id.as_deref(), agent_id.as_deref()) {
            (Some(task_id), Some(agent_id)) => Some(scoped_program_id(task_id, Some(agent_id))),
            _ => None,
        });
    let Some(program_id) = program_id else {
        return Ok(None);
    };
    let placement_key = task_id
        .clone()
        .or(session_id)
        .unwrap_or_else(|| program_id.clone());

    let blocks_parent = context_bool(context, &["blocks_parent", "blocking_parent"])?
        .unwrap_or(agent_role.as_deref() == Some("subagent"));
    let expected_resume = context_bool(context, &["expected_resume"])?
        .unwrap_or_else(|| !(blocks_parent && agent_role.as_deref() == Some("subagent")));
    ProgramIdentity::build(program_id, placement_key, expected_resume, model_pool).map(Some)
}

fn normalized_agent_component(
    task_id: Option<&str>,
    agent_id: Option<String>,
) -> Result<Option<String>, ScheduleError> {
    let Some(agent_id) = agent_id else {
        return Ok(None);
    };
    let Some(task_id) = task_id else {
        return Ok(Some(agent_id));
    };
    let prefix = format!("{task_id}:");
    let Some(component) = agent_id.strip_prefix(&prefix) else {
        return Ok(Some(agent_id));
    };
    if component.is_empty() {
        return Err(ScheduleError::InvalidIdentity(
            "agent_id must include an agent component".to_string(),
        ));
    }
    Ok(Some(component.to_string()))
}

fn canonical_context(
    request: Option<&JsonMap<String, JsonValue>>,
) -> Result<Option<JsonMap<String, JsonValue>>, ScheduleError> {
    let Some(raw_xargs) = request.and_then(|request| request.get("vllm_xargs")) else {
        return Ok(None);
    };
    if raw_xargs.is_null() {
        return Ok(None);
    }
    let xargs = raw_xargs.as_object().ok_or_else(|| {
        ScheduleError::InvalidIdentity("vllm_xargs must be an object".to_string())
    })?;
    let Some(raw_context) = xargs.get("agentic_context") else {
        return Ok(None);
    };
    if raw_context.is_null() {
        return Ok(None);
    }
    let decoded = match raw_context {
        JsonValue::String(value) => {
            if value.len() > MAX_AGENTIC_CONTEXT_BYTES {
                return Err(ScheduleError::InvalidIdentity(
                    "vllm_xargs.agentic_context must contain a bounded JSON object".to_string(),
                ));
            }
            serde_json::from_str::<JsonValue>(value).map_err(|_| {
                ScheduleError::InvalidIdentity(
                    "vllm_xargs.agentic_context must contain a bounded JSON object".to_string(),
                )
            })?
        }
        value => value.clone(),
    };
    decoded.as_object().cloned().map(Some).ok_or_else(|| {
        ScheduleError::InvalidIdentity("vllm_xargs.agentic_context must be an object".to_string())
    })
}

fn agent_hint_identity(
    request: Option<&JsonMap<String, JsonValue>>,
    model_pool: Option<&str>,
) -> Result<Option<ProgramIdentity>, ScheduleError> {
    let Some(raw_hint) = request.and_then(|request| request.get("agent_hint")) else {
        return Ok(None);
    };
    if raw_hint.is_null() {
        return Ok(None);
    }
    let hint = raw_hint.as_object().ok_or_else(|| {
        ScheduleError::InvalidIdentity("agent_hint must be an object".to_string())
    })?;
    let session_id = context_string(hint, &["session_id"])?.ok_or_else(|| {
        ScheduleError::InvalidIdentity("agent_hint.session_id is required".to_string())
    })?;
    let parent_session_id = context_string(hint, &["parent_session_id"])?;
    let blocks_parent = context_bool(hint, &["blocks_parent"])?.unwrap_or(true);
    if blocks_parent && parent_session_id.is_none() {
        return Err(ScheduleError::InvalidIdentity(
            "blocking agent_hint requires parent_session_id".to_string(),
        ));
    }
    let expected_resume = context_bool(hint, &["expected_resume"])?.unwrap_or(!blocks_parent);
    ProgramIdentity::build(session_id.clone(), session_id, expected_resume, model_pool).map(Some)
}

fn framework_identity_headers(
    headers: Option<&HeaderMap>,
) -> Result<(Option<String>, Option<String>), ScheduleError> {
    let Some(headers) = headers else {
        return Ok((None, None));
    };
    for (session_header, agent_header) in [
        (CLAUDE_SESSION_HEADER, Some(CLAUDE_AGENT_HEADER)),
        (CODEX_SESSION_HEADER, Some(CODEX_THREAD_HEADER)),
        (GENERIC_SESSION_HEADER, None),
    ] {
        let Some(session_id) = read_nonempty_header(headers, session_header)? else {
            continue;
        };
        let agent_id = match agent_header {
            Some(name) => read_nonempty_header(headers, name)?,
            None => None,
        };
        return Ok((Some(session_id), agent_id));
    }
    Ok((None, None))
}

fn read_nonempty_header(headers: &HeaderMap, name: &str) -> Result<Option<String>, ScheduleError> {
    headers
        .get(name)
        .map(|value| {
            value
                .to_str()
                .map(|value| value.trim().to_string())
                .map_err(|_| ScheduleError::InvalidIdentity(format!("Invalid {name} header")))
        })
        .transpose()
        .map(|value| value.filter(|value| !value.is_empty()))
}

fn context_string(
    context: &JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Result<Option<String>, ScheduleError> {
    for key in keys {
        let Some(value) = context.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let normalized = match value {
            JsonValue::String(value) => value.trim().to_string(),
            JsonValue::Number(value) if value.is_i64() || value.is_u64() => value.to_string(),
            _ => {
                return Err(ScheduleError::InvalidIdentity(format!(
                    "{key} must be a string-compatible identifier"
                )))
            }
        };
        if normalized.is_empty() {
            return Err(ScheduleError::InvalidIdentity(format!(
                "{key} must not be empty"
            )));
        }
        return Ok(Some(normalized));
    }
    Ok(None)
}

fn context_bool(
    context: &JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Result<Option<bool>, ScheduleError> {
    for key in keys {
        let Some(value) = context.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        return value
            .as_bool()
            .map(Some)
            .ok_or_else(|| ScheduleError::InvalidIdentity(format!("{key} must be a boolean")));
    }
    Ok(None)
}

fn scoped_program_id(session_id: &str, actor_id: Option<&str>) -> String {
    let actor_id = actor_id.unwrap_or("lead");
    let prefix = format!("{session_id}:");
    if actor_id.starts_with(&prefix) {
        actor_id.to_string()
    } else {
        format!("{prefix}{actor_id}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_existing_program_identity_contracts() {
        let canonical = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "program_id": "task-a:researcher",
                "expected_resume": false
            }}
        });
        let identity = ProgramIdentity::from_request(None, Some(&canonical), Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.model_pool(), "model");
        assert_eq!(identity.program_id(), "task-a:researcher");
        assert!(!identity.expected_resume());

        let mut codex_headers = HeaderMap::new();
        codex_headers.insert(CODEX_SESSION_HEADER, "session-a".parse().unwrap());
        codex_headers.insert(CODEX_THREAD_HEADER, "worker".parse().unwrap());
        let identity = ProgramIdentity::from_request(Some(&codex_headers), None, Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "session-a:worker");
        assert_eq!(identity.placement_hash_key(), "header:session-id:session-a");
        assert!(!identity.expected_resume());

        let mut claude_headers = HeaderMap::new();
        claude_headers.insert(CLAUDE_SESSION_HEADER, "session-c".parse().unwrap());
        claude_headers.insert(CLAUDE_AGENT_HEADER, "researcher".parse().unwrap());
        let identity = ProgramIdentity::from_request(Some(&claude_headers), None, Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "session-c:researcher");
        assert_eq!(
            identity.placement_hash_key(),
            "header:x-claude-code-session-id:session-c"
        );
        assert_eq!(identity.placement_key(), "session-c");
        assert!(!identity.expected_resume());

        let mut generic_headers = HeaderMap::new();
        generic_headers.insert(GENERIC_SESSION_HEADER, "session-d".parse().unwrap());
        let identity = ProgramIdentity::from_request(Some(&generic_headers), None, Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "session-d:lead");
        assert_eq!(
            identity.placement_hash_key(),
            "header:x-session-id:session-d"
        );
        assert!(identity.expected_resume());
    }

    #[test]
    fn body_contract_precedence_and_defaults_match_current_behavior() {
        let encoded_context = serde_json::json!({
            "vllm_xargs": {"agentic_context": serde_json::json!({
                "program_id": "encoded-program",
                "task_id": null,
                "expected_resume": true
            }).to_string()}
        });
        let identity = ProgramIdentity::from_request(None, Some(&encoded_context), Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "encoded-program");

        let inferred_blocking_subagent = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "task_id": "task-b", "agent_id": "researcher"
            }}
        });
        let identity =
            ProgramIdentity::from_request(None, Some(&inferred_blocking_subagent), Some("model"))
                .unwrap()
                .unwrap();
        assert_eq!(identity.program_id(), "task-b:researcher");
        assert_eq!(identity.placement_key(), "task-b");
        assert!(!identity.expected_resume());

        let explicit_resume = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "task_id": "task-b",
                "agent_id": "researcher",
                "expected_resume": true
            }}
        });
        let identity = ProgramIdentity::from_request(None, Some(&explicit_resume), Some("model"))
            .unwrap()
            .unwrap();
        assert!(identity.expected_resume());

        let agent_hint = serde_json::json!({
            "agent_hint": {
                "session_id": "child-session",
                "parent_session_id": "parent-session",
                "blocks_parent": true
            }
        });
        let identity = ProgramIdentity::from_request(None, Some(&agent_hint), Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "child-session");
        assert!(!identity.expected_resume());
    }

    #[test]
    fn request_and_trace_ids_do_not_create_program_identity() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "request-1".parse().unwrap());
        headers.insert("x-trace-id", "trace-1".parse().unwrap());
        assert!(
            ProgramIdentity::from_request(Some(&headers), None, Some("model"))
                .unwrap()
                .is_none()
        );

        let unsupported = serde_json::json!({"session_params": {"session_id": "task-a"}});
        assert!(
            ProgramIdentity::from_request(None, Some(&unsupported), Some("model"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn malformed_recognized_contract_is_rejected() {
        let malformed = serde_json::json!({
            "vllm_xargs": {"agentic_context": {"task_id": [], "agent_id": "worker"}}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&malformed), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));

        let blocking_root = serde_json::json!({
            "agent_hint": {"session_id": "root", "blocks_parent": true}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&blocking_root), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));
    }
}
