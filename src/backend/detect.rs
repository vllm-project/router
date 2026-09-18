//! Worker URL scheme → connection mode and tonic URI.
//!
//! `grpc://host:port` is rewritten to h2c `http://host:port` for tonic.
//! That is still gRPC, not the worker's OpenAI `--port`. Optional
//! `@dp_rank` suffix (`grpc://host:port@2`) is stripped before connect.
//!
//! A worker pool is all-`http(s)://` or all-`grpc(s)://`. Mixed schemes
//! fail at init / `add_worker` (gRPC is `token_ids` only; HTTP proxies text).

use crate::core::ConnectionMode;

/// Homogeneous worker pool. `None` means no workers registered yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerPoolKind {
    Http,
    Grpc,
}

/// True when the operator marked this worker as a vLLM rust gRPC endpoint.
pub fn is_grpc_url(url: &str) -> bool {
    url.starts_with("grpc://") || url.starts_with("grpcs://")
}

/// Classify a worker URL list. Empty → `Ok(None)`. Mixed schemes → `Err`.
pub fn classify_worker_urls(urls: &[String]) -> Result<Option<WorkerPoolKind>, String> {
    let mut kind: Option<WorkerPoolKind> = None;
    for url in urls {
        let next = if is_grpc_url(url) {
            WorkerPoolKind::Grpc
        } else {
            WorkerPoolKind::Http
        };
        match kind {
            None => kind = Some(next),
            Some(existing) if existing != next => {
                return Err(format!(
                    "mixed worker schemes are not supported \
                     (grpc:// is token_ids-only; http:// proxies text). \
                     got {url} in a {existing:?} pool"
                ));
            }
            Some(_) => {}
        }
    }
    Ok(kind)
}

/// Infer connection mode from the worker URL. No extra CLI flag.
pub fn connection_mode_from_url(url: &str) -> ConnectionMode {
    if is_grpc_url(url) {
        ConnectionMode::Grpc
    } else {
        ConnectionMode::Http
    }
}

/// Strip optional `@dp_rank` suffix used by intra-node DP URLs.
pub fn strip_dp_suffix(url: &str) -> &str {
    url.rsplit_once('@')
        .and_then(|(prefix, rank)| rank.parse::<u32>().ok().map(|_| prefix))
        .unwrap_or(url)
}

/// DP rank encoded as `grpc://host:port@rank`, if present.
pub fn parse_dp_rank(url: &str) -> Option<u32> {
    url.rsplit_once('@')
        .and_then(|(_, rank)| rank.parse::<u32>().ok())
}

/// Tonic connect URI (`http://host:port` or `https://host:port`).
pub fn grpc_connect_uri(url: &str) -> Result<String, String> {
    let base = strip_dp_suffix(url);
    if let Some(rest) = base.strip_prefix("grpc://") {
        Ok(format!("http://{rest}"))
    } else if let Some(rest) = base.strip_prefix("grpcs://") {
        Ok(format!("https://{rest}"))
    } else {
        Err(format!("not a grpc worker URL: {url}"))
    }
}

/// Host:port extracted from a `grpc://` / `grpcs://` URL (no probe).
pub fn grpc_socket_addr(url: &str) -> Result<String, String> {
    let uri = grpc_connect_uri(url)?;
    uri.strip_prefix("http://")
        .or_else(|| uri.strip_prefix("https://"))
        .map(str::to_string)
        .ok_or_else(|| format!("failed to parse grpc address from {url}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_explicit_grpc_scheme() {
        assert!(is_grpc_url("grpc://127.0.0.1:50051"));
        assert!(is_grpc_url("grpcs://worker:50051"));
        assert!(!is_grpc_url("http://127.0.0.1:8000"));
        assert!(!is_grpc_url("https://127.0.0.1:8000"));
    }

    #[test]
    fn maps_scheme_to_connection_mode() {
        assert_eq!(
            connection_mode_from_url("grpc://10.0.0.1:50051"),
            ConnectionMode::Grpc
        );
        assert_eq!(
            connection_mode_from_url("http://10.0.0.1:8000"),
            ConnectionMode::Http
        );
    }

    #[test]
    fn strips_dp_rank_and_builds_tonic_uri() {
        assert_eq!(
            grpc_connect_uri("grpc://192.168.1.10:50051@2").unwrap(),
            "http://192.168.1.10:50051"
        );
        assert_eq!(parse_dp_rank("grpc://192.168.1.10:50051@2"), Some(2));
        assert_eq!(parse_dp_rank("grpc://192.168.1.10:50051"), None);
        assert_eq!(
            grpc_socket_addr("grpc://127.0.0.1:50051").unwrap(),
            "127.0.0.1:50051"
        );
    }

    #[test]
    fn classifies_homogeneous_pools() {
        assert_eq!(classify_worker_urls(&[]).unwrap(), None);
        assert_eq!(
            classify_worker_urls(&["http://a:8000".into()]).unwrap(),
            Some(WorkerPoolKind::Http)
        );
        assert_eq!(
            classify_worker_urls(&["https://a:8000".into(), "http://b:8001".into(),]).unwrap(),
            Some(WorkerPoolKind::Http)
        );
        assert_eq!(
            classify_worker_urls(&["grpc://a:50051".into(), "grpc://b:50051@0".into(),]).unwrap(),
            Some(WorkerPoolKind::Grpc)
        );
    }

    #[test]
    fn rejects_mixed_schemes() {
        let err =
            classify_worker_urls(&["http://a:8000".into(), "grpc://a:50051".into()]).unwrap_err();
        assert!(err.contains("mixed"));
        assert!(err.contains("grpc://a:50051"));
    }
}
