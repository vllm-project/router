//! Stable request identity shared by routing and Program scheduling.
//!
//! Scheduling relationships, hints, and lifecycle constraints belong to the
//! scheduling consumer, not to this parser.

use http::HeaderMap;
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::io::{self, Write};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct IdentityError(pub String);

pub(crate) const MAX_IDENTITY_COMPONENT_BYTES: usize = 256;
pub(crate) const MAX_AGENTIC_CONTEXT_BYTES: usize = 16 * 1024;
pub(crate) const CLAUDE_SESSION_HEADER: &str = "x-claude-code-session-id";
pub(crate) const CLAUDE_AGENT_HEADER: &str = "x-claude-code-agent-id";
pub(crate) const CLAUDE_PARENT_AGENT_HEADER: &str = "x-claude-code-parent-agent-id";
pub(crate) const CODEX_SESSION_HEADER: &str = "session-id";
pub(crate) const CODEX_THREAD_HEADER: &str = "thread-id";
pub(crate) const GENERIC_SESSION_HEADER: &str = "x-session-id";

/// Provenance of a stable identity. Affinity routing preserves the source
/// kind so identical identifiers from different contracts never share a
/// binding scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IdentitySource {
    /// Canonical `vllm_xargs.agentic_context` body contract.
    CanonicalContext,
    /// `agent_hint` body contract.
    AgentHint,
    /// Supported framework session headers.
    FrameworkHeader,
}

impl IdentitySource {
    pub fn as_str(self) -> &'static str {
        match self {
            IdentitySource::CanonicalContext => "agentic_context",
            IdentitySource::AgentHint => "agent_hint",
            IdentitySource::FrameworkHeader => "framework_header",
        }
    }
}

/// Identity and its protocol components, without scheduling metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StableIdentity {
    pub source: IdentitySource,
    pub program_id: String,
    pub task_id: Option<String>,
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
}

impl StableIdentity {
    pub fn placement_key(&self) -> &str {
        self.task_id
            .as_deref()
            .or(self.session_id.as_deref())
            .unwrap_or(&self.program_id)
    }
}

pub(crate) fn resolve_stable_identity(
    headers: Option<&HeaderMap>,
    request: Option<&JsonValue>,
) -> Result<Option<StableIdentity>, IdentityError> {
    let request = request.and_then(JsonValue::as_object);
    let context = canonical_context(request)?;
    identity_from_parts(headers, request, context.as_ref())
}

/// Accept an already decoded context so scheduling parses it only once.
pub(crate) fn identity_from_parts(
    headers: Option<&HeaderMap>,
    request: Option<&JsonMap<String, JsonValue>>,
    context: Option<&JsonMap<String, JsonValue>>,
) -> Result<Option<StableIdentity>, IdentityError> {
    let identity = if let Some(context) = context {
        let (header_session, header_agent, _) = framework_identity_headers(headers)?;
        let program_id = context_string(context, &["program_id"])?;
        let body_task = context_string(context, &["task_id"])?;
        let session_id = context_string(context, &["session_id"])?.or(header_session);
        let task_id =
            if body_task.is_some() || (program_id.is_some() && context.contains_key("task_id")) {
                body_task
            } else {
                session_id.clone()
            };
        let agent_id = context_string(context, &["agent_id"])?
            .or(header_agent)
            .or_else(|| task_id.as_ref().map(|_| "lead".to_string()));
        let agent_id = normalized_agent_component(task_id.as_deref(), agent_id)?;
        let program_id = program_id.or_else(|| match (task_id.as_deref(), agent_id.as_deref()) {
            (Some(task), Some(agent)) => Some(scoped_program_id(task, Some(agent))),
            _ => None,
        });
        program_id.map(|program_id| StableIdentity {
            source: IdentitySource::CanonicalContext,
            program_id,
            task_id,
            session_id,
            agent_id,
        })
    } else if let Some(hint) = request
        .and_then(|request| request.get("agent_hint"))
        .filter(|hint| !hint.is_null())
    {
        let hint = hint
            .as_object()
            .ok_or_else(|| IdentityError("agent_hint must be an object".to_string()))?;
        let session_id = context_string(hint, &["session_id"])?
            .ok_or_else(|| IdentityError("agent_hint.session_id is required".to_string()))?;
        Some(StableIdentity {
            source: IdentitySource::AgentHint,
            program_id: session_id.clone(),
            task_id: None,
            session_id: Some(session_id),
            agent_id: None,
        })
    } else {
        let (session_id, agent_id, _) = framework_identity_headers(headers)?;
        match session_id {
            Some(session_id) => {
                let agent_id = normalized_agent_component(Some(&session_id), agent_id)?;
                Some(StableIdentity {
                    source: IdentitySource::FrameworkHeader,
                    program_id: scoped_program_id(&session_id, agent_id.as_deref()),
                    task_id: Some(session_id.clone()),
                    session_id: Some(session_id),
                    agent_id,
                })
            }
            None => None,
        }
    };
    if let Some(identity) = &identity {
        for (name, value) in [
            ("program_id", identity.program_id.as_str()),
            ("placement key", identity.placement_key()),
        ] {
            if value.len() > MAX_IDENTITY_COMPONENT_BYTES {
                return Err(IdentityError(format!(
                    "{name} exceeds {MAX_IDENTITY_COMPONENT_BYTES} bytes"
                )));
            }
        }
    }
    Ok(identity)
}

pub(crate) fn normalized_agent_component(
    task_id: Option<&str>,
    agent_id: Option<String>,
) -> Result<Option<String>, IdentityError> {
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
        return Err(IdentityError(
            "agent_id must include an agent component".to_string(),
        ));
    }
    Ok(Some(component.to_string()))
}

pub(crate) fn canonical_context(
    request: Option<&JsonMap<String, JsonValue>>,
) -> Result<Option<JsonMap<String, JsonValue>>, IdentityError> {
    let Some(raw_xargs) = request.and_then(|request| request.get("vllm_xargs")) else {
        return Ok(None);
    };
    if raw_xargs.is_null() {
        return Ok(None);
    }
    let xargs = raw_xargs
        .as_object()
        .ok_or_else(|| IdentityError("vllm_xargs must be an object".to_string()))?;
    let Some(raw_context) = xargs.get("agentic_context") else {
        return Ok(None);
    };
    if raw_context.is_null() {
        return Ok(None);
    }
    let context = match raw_context {
        JsonValue::String(value) => {
            if value.len() > MAX_AGENTIC_CONTEXT_BYTES {
                return Err(bounded_agentic_context_error());
            }
            let decoded = serde_json::from_str::<JsonValue>(value)
                .map_err(|_| bounded_agentic_context_error())?;
            decoded
                .as_object()
                .cloned()
                .ok_or_else(bounded_agentic_context_error)?
        }
        JsonValue::Object(context) => {
            ensure_bounded_agentic_context(raw_context)?;
            context.clone()
        }
        _ => {
            return Err(IdentityError(
                "vllm_xargs.agentic_context must be an object".to_string(),
            ))
        }
    };
    Ok(Some(context))
}

fn bounded_agentic_context_error() -> IdentityError {
    IdentityError("vllm_xargs.agentic_context must contain a bounded JSON object".to_string())
}

fn ensure_bounded_agentic_context(context: &JsonValue) -> Result<(), IdentityError> {
    let mut writer = BoundedJsonWriter::default();
    serde_json::to_writer(&mut writer, context).map_err(|_| bounded_agentic_context_error())
}

#[derive(Default)]
struct BoundedJsonWriter {
    written: usize,
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(total) = self.written.checked_add(buffer.len()) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "agentic context size overflow",
            ));
        };
        if total > MAX_AGENTIC_CONTEXT_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "agentic context exceeds size limit",
            ));
        }
        self.written = total;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

type FrameworkIdentityHeaders = (Option<String>, Option<String>, Option<String>);

pub(crate) fn framework_identity_headers(
    headers: Option<&HeaderMap>,
) -> Result<FrameworkIdentityHeaders, IdentityError> {
    let Some(headers) = headers else {
        return Ok((None, None, None));
    };
    for (session_header, agent_header, parent_header) in [
        (
            CLAUDE_SESSION_HEADER,
            Some(CLAUDE_AGENT_HEADER),
            Some(CLAUDE_PARENT_AGENT_HEADER),
        ),
        (CODEX_SESSION_HEADER, Some(CODEX_THREAD_HEADER), None),
        (GENERIC_SESSION_HEADER, None, None),
    ] {
        let Some(session_id) = read_nonempty_header(headers, session_header)? else {
            continue;
        };
        let agent_id = match agent_header {
            Some(name) => read_nonempty_header(headers, name)?,
            None => None,
        };
        let parent_agent_id = match parent_header {
            Some(name) => read_nonempty_header(headers, name)?,
            None => None,
        };
        return Ok((Some(session_id), agent_id, parent_agent_id));
    }
    Ok((None, None, None))
}

pub(crate) fn read_nonempty_header(
    headers: &HeaderMap,
    name: &str,
) -> Result<Option<String>, IdentityError> {
    headers
        .get(name)
        .map(|value| {
            value
                .to_str()
                .map(|value| value.trim().to_string())
                .map_err(|_| IdentityError(format!("Invalid {name} header")))
        })
        .transpose()
        .map(|value| value.filter(|value| !value.is_empty()))
}

pub(crate) fn context_string(
    context: &JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Result<Option<String>, IdentityError> {
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
                return Err(IdentityError(format!(
                    "{key} must be a string-compatible identifier"
                )))
            }
        };
        if normalized.is_empty() {
            return Err(IdentityError(format!("{key} must not be empty")));
        }
        return Ok(Some(normalized));
    }
    Ok(None)
}

pub(crate) fn scoped_program_id(session_id: &str, actor_id: Option<&str>) -> String {
    let actor_id = actor_id.unwrap_or("lead");
    let prefix = format!("{session_id}:");
    if actor_id.starts_with(&prefix) {
        actor_id.to_string()
    } else {
        format!("{prefix}{actor_id}")
    }
}
