//! Shared hash key extraction for consistent hashing policies
//!
//! Extracts routing keys from HTTP headers and request bodies.
//! Used by both ConsistentHashPolicy and RendezvousHashPolicy.

use super::RequestHeaders;
use crate::policies::ConsistentHashPolicy;
use tracing::debug;

/// HTTP header names to check for session ID (case-insensitive, checked in order)
pub(crate) const SESSION_HEADER_NAMES: &[&str] = &[
    "x-session-id",
    "x-user-id",
    "x-tenant-id",
    "x-correlation-id", // per-session — check before per-request
    "x-request-id",
    "x-trace-id",
];

/// Extract hash key with priority: HTTP headers > body fields > request content hash
///
/// Priority order:
/// 1. HTTP Headers: x-session-id, x-user-id, x-tenant-id, x-correlation-id, x-request-id, x-trace-id
/// 2. Body: session_params.session_id (nested)
/// 3. Body: user field (OpenAI format)
/// 4. Body: session_id (legacy)
/// 5. Body: user_id (legacy)
/// 6. Fallback: hash of request body (long) or raw text (short)
pub(crate) fn extract_hash_key(
    request_text: Option<&str>,
    headers: Option<&RequestHeaders>,
) -> String {
    // 1. First priority: HTTP headers
    if let Some(hdrs) = headers {
        if let Some(key) = extract_hash_key_from_headers(hdrs) {
            return key;
        }
    }

    // 2. Second priority: Body fields
    if let Some(key) = extract_hash_key_from_body(request_text) {
        return key;
    }

    // 3. Final fallback: hash of request body
    let text = request_text.unwrap_or("");
    if text.len() > 100 {
        format!("request_hash:{:016x}", ConsistentHashPolicy::fbi_hash(text))
    } else {
        format!("request:{}", text)
    }
}

/// Extract hash key from HTTP headers
pub(crate) fn extract_hash_key_from_headers(headers: &RequestHeaders) -> Option<String> {
    for header_name in SESSION_HEADER_NAMES {
        if let Some(value) = headers.get(*header_name) {
            if !value.is_empty() {
                debug!(
                    "Hash key extraction: found session key in header '{}': {}",
                    header_name, value
                );
                return Some(format!("header:{}:{}", header_name, value));
            }
        }
    }
    None
}

/// Extract hash key from request body fields
///
/// Priority: session_params.session_id > user > session_id > user_id
pub(crate) fn extract_hash_key_from_body(request_text: Option<&str>) -> Option<String> {
    let text = request_text.unwrap_or("");
    if text.is_empty() {
        return None;
    }

    // 1. Try to extract session_params.session_id first (highest priority in body)
    if let Some(session_id) = extract_nested_field_value(text, "session_params", "session_id") {
        debug!(
            "Hash key extraction: found session_params.session_id: {}",
            session_id
        );
        return Some(format!("session:{}", session_id));
    }

    // 2. Try to extract direct user field (from OpenAI ChatCompletion/Completion requests)
    if let Some(user) = extract_field_value(text, "user") {
        debug!("Hash key extraction: found user field: {}", user);
        return Some(format!("user:{}", user));
    }

    // 3. Fallback: try legacy session_id field
    if let Some(session_id) = extract_field_value(text, "session_id") {
        return Some(format!("session:{}", session_id));
    }

    // 4. Fallback: try legacy user_id field
    if let Some(user_id) = extract_field_value(text, "user_id") {
        return Some(format!("user:{}", user_id));
    }

    None
}

/// Extract nested field value like session_params.session_id from JSON text
pub(crate) fn extract_nested_field_value(
    text: &str,
    parent_field: &str,
    child_field: &str,
) -> Option<String> {
    if let Some(parent_start) = find_field_start(text, parent_field) {
        if let Some(obj_start) = text[parent_start..].find('{') {
            let obj_start_pos = parent_start + obj_start;
            if let Some(obj_content) = extract_json_object(&text[obj_start_pos..]) {
                return extract_field_value(&obj_content, child_field);
            }
        }
    }
    None
}

/// Find the start position after the colon of a field in JSON text
pub(crate) fn find_field_start(text: &str, field_name: &str) -> Option<usize> {
    let patterns = [format!("\"{}\"", field_name), format!("'{}'", field_name)];

    for pattern in &patterns {
        if let Some(field_pos) = text.find(pattern) {
            let after_field = &text[field_pos + pattern.len()..];
            for (i, ch) in after_field.char_indices() {
                if ch == ':' {
                    return Some(field_pos + pattern.len() + i + 1);
                } else if !ch.is_whitespace() {
                    break;
                }
            }
        }
    }
    None
}

/// Extract JSON object content (simple brace matching)
pub(crate) fn extract_json_object(text: &str) -> Option<String> {
    if !text.starts_with('{') {
        return None;
    }

    let mut brace_count = 0;
    let mut end_pos = 0;

    for (i, ch) in text.char_indices() {
        match ch {
            '{' => brace_count += 1,
            '}' => {
                brace_count -= 1;
                if brace_count == 0 {
                    end_pos = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }

    if brace_count == 0 && end_pos > 0 {
        Some(text[0..end_pos].to_string())
    } else {
        None
    }
}

/// Extract field value from JSON-like text (simple parser)
///
/// Supports double-quoted, single-quoted, and unquoted values.
pub(crate) fn extract_field_value(text: &str, field_name: &str) -> Option<String> {
    let patterns = [
        format!("\"{}\"", field_name),
        format!("'{}'", field_name),
        field_name.to_string(),
    ];

    for pattern in &patterns {
        if let Some(field_pos) = text.find(pattern) {
            let after_field = &text[field_pos + pattern.len()..];

            // Skip whitespace and look for colon
            let mut colon_pos = None;
            for (i, ch) in after_field.char_indices() {
                if ch == ':' {
                    colon_pos = Some(i);
                    break;
                } else if !ch.is_whitespace() {
                    break;
                }
            }

            if let Some(colon_idx) = colon_pos {
                let after_colon = &after_field[colon_idx + 1..];
                let trimmed = after_colon.trim_start();

                // Extract quoted string (double or single quotes)
                if trimmed.starts_with('"') {
                    if let Some(stripped) = trimmed.strip_prefix('"') {
                        if let Some(end_quote) = stripped.find('"') {
                            return Some(stripped[..end_quote].to_string());
                        }
                    }
                } else if trimmed.starts_with('\'') {
                    if let Some(stripped) = trimmed.strip_prefix('\'') {
                        if let Some(end_quote) = stripped.find('\'') {
                            return Some(stripped[..end_quote].to_string());
                        }
                    }
                } else {
                    // Unquoted value - extract until delimiter
                    let end_pos = trimmed
                        .find(&[',', ' ', '}', ']', '\n', '\r', '\t'][..])
                        .unwrap_or(trimmed.len());
                    if end_pos > 0 {
                        return Some(trimmed[..end_pos].to_string());
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // === extract_field_value tests ===

    #[test]
    fn test_extract_field_value_double_quoted() {
        let text = r#"{"session_id": "abc123", "prompt": "hello"}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_single_quoted() {
        let text = r#"{'session_id': 'def456', 'prompt': 'world'}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("def456".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_unquoted() {
        let text = r#"{"count": 42, "name": "test"}"#;
        assert_eq!(extract_field_value(text, "count"), Some("42".to_string()));
    }

    #[test]
    fn test_extract_field_value_missing() {
        let text = r#"{"other": "val"}"#;
        assert_eq!(extract_field_value(text, "session_id"), None);
    }

    #[test]
    fn test_extract_field_value_no_space_after_colon() {
        let text = r#"{"session_id":"compact_value"}"#;
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("compact_value".to_string())
        );
    }

    #[test]
    fn test_extract_field_value_multiple_fields() {
        let text = r#"{"user": "bob", "prompt": "hi", "session_id": "sess1"}"#;
        assert_eq!(extract_field_value(text, "user"), Some("bob".to_string()));
        assert_eq!(
            extract_field_value(text, "session_id"),
            Some("sess1".to_string())
        );
    }

    // === extract_nested_field_value tests ===

    #[test]
    fn test_extract_nested_field_value() {
        let text = r#"{"session_params": {"session_id": "nested123"}, "prompt": "hi"}"#;
        assert_eq!(
            extract_nested_field_value(text, "session_params", "session_id"),
            Some("nested123".to_string())
        );
    }

    #[test]
    fn test_extract_nested_field_value_missing_parent() {
        let text = r#"{"prompt": "hi"}"#;
        assert_eq!(
            extract_nested_field_value(text, "session_params", "session_id"),
            None
        );
    }

    #[test]
    fn test_extract_nested_field_value_missing_child() {
        let text = r#"{"session_params": {"other": "val"}, "prompt": "hi"}"#;
        assert_eq!(
            extract_nested_field_value(text, "session_params", "session_id"),
            None
        );
    }

    // === extract_json_object tests ===

    #[test]
    fn test_extract_json_object_simple() {
        let text = r#"{"key": "value"} trailing"#;
        assert_eq!(
            extract_json_object(text),
            Some(r#"{"key": "value"}"#.to_string())
        );
    }

    #[test]
    fn test_extract_json_object_nested() {
        let text = r#"{"outer": {"inner": "val"}} trailing"#;
        assert_eq!(
            extract_json_object(text),
            Some(r#"{"outer": {"inner": "val"}}"#.to_string())
        );
    }

    #[test]
    fn test_extract_json_object_not_object() {
        assert_eq!(extract_json_object("not json"), None);
    }

    #[test]
    fn test_extract_json_object_unclosed() {
        assert_eq!(extract_json_object("{unclosed"), None);
    }

    // === extract_hash_key_from_headers tests ===

    #[test]
    fn test_header_extraction_priority() {
        let mut headers = HashMap::new();
        headers.insert("x-request-id".to_string(), "req-1".to_string());
        headers.insert("x-session-id".to_string(), "sess-1".to_string());

        // x-session-id has higher priority than x-request-id
        let key = extract_hash_key_from_headers(&headers).unwrap();
        assert_eq!(key, "header:x-session-id:sess-1");
    }

    #[test]
    fn test_header_extraction_skips_empty() {
        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "".to_string());
        headers.insert("x-user-id".to_string(), "user-1".to_string());

        let key = extract_hash_key_from_headers(&headers).unwrap();
        assert_eq!(key, "header:x-user-id:user-1");
    }

    #[test]
    fn test_header_extraction_no_match() {
        let mut headers = HashMap::new();
        headers.insert("x-custom-header".to_string(), "val".to_string());
        assert_eq!(extract_hash_key_from_headers(&headers), None);
    }

    // === extract_hash_key_from_body tests ===

    #[test]
    fn test_body_extraction_session_params_priority() {
        // session_params.session_id takes priority over direct session_id
        let text = r#"{"session_params": {"session_id": "nested"}, "session_id": "direct"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "session:nested");
    }

    #[test]
    fn test_body_extraction_user_field() {
        let text = r#"{"user": "alice", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "user:alice");
    }

    #[test]
    fn test_body_extraction_legacy_session_id() {
        let text = r#"{"session_id": "legacy123", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "session:legacy123");
    }

    #[test]
    fn test_body_extraction_legacy_user_id() {
        let text = r#"{"user_id": "uid456", "prompt": "hi"}"#;
        let key = extract_hash_key_from_body(Some(text)).unwrap();
        assert_eq!(key, "user:uid456");
    }

    #[test]
    fn test_body_extraction_empty() {
        assert_eq!(extract_hash_key_from_body(None), None);
        assert_eq!(extract_hash_key_from_body(Some("")), None);
    }

    #[test]
    fn test_body_extraction_no_known_fields() {
        let text = r#"{"prompt": "hello", "model": "llama"}"#;
        assert_eq!(extract_hash_key_from_body(Some(text)), None);
    }

    // === extract_hash_key (top-level) tests ===

    #[test]
    fn test_hash_key_headers_over_body() {
        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "from-header".to_string());
        let body = r#"{"session_id": "from-body"}"#;

        let key = extract_hash_key(Some(body), Some(&headers));
        assert_eq!(key, "header:x-session-id:from-header");
    }

    #[test]
    fn test_hash_key_fallback_short_text() {
        let key = extract_hash_key(Some("short"), None);
        assert_eq!(key, "request:short");
    }

    #[test]
    fn test_hash_key_fallback_long_text() {
        let long_text = "x".repeat(200);
        let key = extract_hash_key(Some(&long_text), None);
        assert!(key.starts_with("request_hash:"));
        assert_eq!(key.len(), "request_hash:".len() + 16); // 16 hex chars

        // Same long text should produce same hash
        let key2 = extract_hash_key(Some(&long_text), None);
        assert_eq!(key, key2);
    }

    #[test]
    fn test_hash_key_fallback_none() {
        let key = extract_hash_key(None, None);
        assert_eq!(key, "request:");
    }

    // === find_field_start tests ===

    #[test]
    fn test_find_field_start_double_quoted() {
        let text = r#"{"field": "value"}"#;
        let pos = find_field_start(text, "field");
        assert!(pos.is_some());
        // Should point to after the colon
        let after = &text[pos.unwrap()..];
        assert!(after.trim_start().starts_with('"'));
    }

    #[test]
    fn test_find_field_start_single_quoted() {
        let text = r#"{'field': 'value'}"#;
        let pos = find_field_start(text, "field");
        assert!(pos.is_some());
    }

    #[test]
    fn test_find_field_start_missing() {
        let text = r#"{"other": "value"}"#;
        assert_eq!(find_field_start(text, "field"), None);
    }
}
