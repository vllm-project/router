use axum::body::Body;
use axum::extract::Request;
use axum::http::HeaderMap;

pub use crate::otel_http::TRACE_HEADER_NAMES;

/// Copy request headers to a Vec of name-value string pairs
/// Used for forwarding headers to backend workers
pub fn copy_request_headers(req: &Request<Body>) -> Vec<(String, String)> {
    forwarded_request_headers(req.headers())
}

/// Filter the client's headers down to what may be relayed to a worker.
///
/// Hop-by-hop fields describe this connection only and must not be passed on
/// (RFC 9110 §7.6.1), nor may a field that the request's own `Connection`
/// names. `content-encoding` is deliberately kept: the request body is relayed
/// as-is, so dropping it would make the worker misread a compressed body.
fn forwarded_request_headers(headers: &HeaderMap) -> Vec<(String, String)> {
    let connection_fields = connection_named_fields(headers);

    headers
        .iter()
        .filter_map(|(name, value)| {
            let name_str = name.as_str();
            if !should_forward_request_header(name_str)
                || connection_fields
                    .iter()
                    .any(|field| field.eq_ignore_ascii_case(name_str))
            {
                return None;
            }
            // Convert header value to string, skipping non-UTF8 headers
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

/// Field names that a `Connection` header points at.
///
/// RFC 9110 §7.6.1 lets a message name extra hop-by-hop fields in its own
/// `Connection` header, and those fields must not be forwarded either. The
/// header may appear more than once and each value is a comma-separated list,
/// so every occurrence is split and trimmed. Callers compare the names
/// case-insensitively.
fn connection_named_fields(headers: &HeaderMap) -> Vec<&str> {
    headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .collect()
}

/// Convert headers from reqwest Response to axum HeaderMap
/// Filters out hop-by-hop headers that shouldn't be forwarded
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    // Connection can name additional hop-by-hop fields to remove.
    let connection_fields = connection_named_fields(reqwest_headers);

    for (name, value) in reqwest_headers.iter() {
        // Skip hop-by-hop headers that shouldn't be forwarded
        let name_str = name.as_str();
        if should_forward_response_header(name_str)
            && !connection_fields
                .iter()
                .any(|field| field.eq_ignore_ascii_case(name_str))
        {
            // The original name and value are already valid, so we can just clone them
            headers.append(name.clone(), value.clone());
        }
    }

    headers
}

/// Hop-by-hop fields, which belong to a single connection and must never be
/// relayed (RFC 9110 §7.6.1).
fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name,
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

/// Determine if a header should be forwarded from backend to client
fn should_forward_response_header(name: &str) -> bool {
    !is_hop_by_hop(name) &&
        name != "content-encoding" && // Let axum/hyper handle encoding
        name != "host" // Should not forward the backend's host header
}

/// Determine if a header should be forwarded from client to backend
///
/// Unlike the response direction, `content-encoding` has to survive: the body
/// is relayed untouched, so the worker still needs it to read the body. `host`
/// is dropped so the worker sees the host of the URL the router actually
/// dialed rather than the router's own hostname.
fn should_forward_request_header(name: &str) -> bool {
    !is_hop_by_hop(name) && name != "host"
}

/// Propagate OpenTelemetry trace headers to a reqwest RequestBuilder
///
/// When OTel is enabled: actively injects the current span's trace context,
/// making the router's span the parent of the backend request's span.
/// When OTel is disabled: passively forwards existing trace headers from
/// the incoming request.
pub fn propagate_trace_headers(
    request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
) -> reqwest::RequestBuilder {
    crate::otel_http::propagate_trace_headers(request, headers)
}

/// Propagate specific headers from incoming request to outgoing reqwest RequestBuilder
///
/// This is a general-purpose helper for selectively forwarding headers by name.
/// Only headers whose names match the provided list (case-insensitive) are propagated.
///
/// # Arguments
/// * `request` - The reqwest RequestBuilder to add headers to
/// * `headers` - Optional incoming headers to check
/// * `header_names` - List of header names to propagate (matched case-insensitively)
///
/// # Returns
/// The RequestBuilder with matching headers added
pub fn propagate_headers(
    mut request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
    header_names: &[&str],
) -> reqwest::RequestBuilder {
    if let Some(h) = headers {
        for &name in header_names {
            if let Some(value) = h.get(name) {
                request = request.header(name, value);
            }
        }
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preserve_repeated_set_cookie_headers() {
        let mut headers = HeaderMap::new();
        headers.append("set-cookie", "a=1; Path=/".parse().unwrap());
        headers.append("set-cookie", "b=2; Path=/".parse().unwrap());

        let result = preserve_response_headers(&headers);
        let cookies: Vec<_> = result
            .get_all("set-cookie")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();

        assert_eq!(cookies, ["a=1; Path=/", "b=2; Path=/"]);
    }

    #[test]
    fn test_preserve_response_headers_skips_connection_named_fields() {
        let mut headers = HeaderMap::new();
        // Connection may be repeated, its value is a comma-separated list, and
        // the field names in it are case-insensitive and may be padded.
        headers.append("connection", "keep-alive, X-Backend-Hint".parse().unwrap());
        headers.append(
            "connection",
            " X-Internal-Trace ,x-Chain-Only".parse().unwrap(),
        );
        headers.insert("x-backend-hint", "backend-only".parse().unwrap());
        headers.insert("x-internal-trace", "private".parse().unwrap());
        headers.insert("x-chain-only", "chained".parse().unwrap());
        headers.insert("x-backend-hint-extra", "public".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());

        let result = preserve_response_headers(&headers);

        assert!(result.get("connection").is_none());
        assert!(result.get("x-backend-hint").is_none());
        assert!(result.get("x-internal-trace").is_none());
        assert!(result.get("x-chain-only").is_none());
        // A name that only shares a prefix with a listed one must survive.
        assert_eq!(result["x-backend-hint-extra"], "public");
        assert_eq!(result["content-type"], "application/json");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_preserve_response_headers_filters_hop_by_hop_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.append("connection", "keep-alive".parse().unwrap());
        headers.append("connection", "close".parse().unwrap());
        headers.insert("transfer-encoding", "chunked".parse().unwrap());

        let result = preserve_response_headers(&headers);

        assert_eq!(result.len(), 1);
        assert_eq!(result["content-type"], "application/json");
    }

    #[test]
    fn test_forwarded_request_headers_strip_hop_by_hop_and_connection_named_fields() {
        let mut headers = HeaderMap::new();
        // Connection may be repeated, its value is a comma-separated list, and
        // the field names in it are case-insensitive and may be padded.
        headers.append("connection", "keep-alive, X-Client-Hint".parse().unwrap());
        headers.append(
            "connection",
            " X-Internal-Trace ,x-Chain-Only".parse().unwrap(),
        );
        headers.insert("keep-alive", "timeout=5".parse().unwrap());
        headers.insert("transfer-encoding", "chunked".parse().unwrap());
        headers.insert("host", "router.local".parse().unwrap());
        headers.insert("x-client-hint", "internal".parse().unwrap());
        headers.insert("x-internal-trace", "trace-1".parse().unwrap());
        headers.insert("x-chain-only", "chained".parse().unwrap());
        headers.insert("x-client-hint-extra", "keep-me".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("content-encoding", "gzip".parse().unwrap());
        headers.insert("authorization", "Bearer token".parse().unwrap());

        let forwarded = forwarded_request_headers(&headers);
        let names: Vec<&str> = forwarded.iter().map(|(name, _)| name.as_str()).collect();

        for dropped in [
            "connection",
            "keep-alive",
            "transfer-encoding",
            "host",
            "x-client-hint",
            "x-internal-trace",
            "x-chain-only",
        ] {
            assert!(
                !names.contains(&dropped),
                "{dropped} must not reach the worker, got {names:?}"
            );
        }
        // A name that only shares a prefix with a listed one must survive.
        assert!(names.contains(&"x-client-hint-extra"));
        // End-to-end fields survive. content-encoding especially: the body is
        // relayed as-is, so the worker still needs it to read the body.
        assert!(names.contains(&"content-type"));
        assert!(names.contains(&"content-encoding"));
        assert!(names.contains(&"authorization"));
        assert_eq!(forwarded.len(), 4);
        assert_eq!(
            forwarded
                .iter()
                .find(|(name, _)| name == "authorization")
                .map(|(_, value)| value.as_str()),
            Some("Bearer token")
        );
    }
}
