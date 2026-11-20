//! Worker credentials and the gRPC destinations authorized to receive them.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use tonic::metadata::AsciiMetadataValue;

use super::{is_grpc_url, strip_dp_suffix};

#[derive(Clone, Default)]
pub(crate) struct WorkerAuth(Option<Arc<Credential>>);

struct Credential {
    key: Arc<str>,
    grpc_workers: HashSet<String>,
    grpc_bearer: Option<AsciiMetadataValue>,
}

impl fmt::Debug for WorkerAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerAuth").finish_non_exhaustive()
    }
}

impl WorkerAuth {
    /// Only operator-configured gRPC transports receive the shared key. DP
    /// ranks use the same transport, and registration never extends this set.
    pub(crate) fn new<'a>(
        key: Option<&str>,
        configured_urls: impl Iterator<Item = &'a str>,
    ) -> Result<Self, String> {
        let Some(key) = key else {
            return Ok(Self::default());
        };
        let grpc_workers: HashSet<_> = configured_urls
            .filter(|url| is_grpc_url(url))
            .map(|url| strip_dp_suffix(url).trim_end_matches('/').to_owned())
            .collect();
        let grpc_bearer = if grpc_workers.is_empty() {
            None
        } else {
            Some(bearer_metadata(key)?)
        };
        Ok(Self(Some(Arc::new(Credential {
            key: Arc::from(key),
            grpc_workers,
            grpc_bearer,
        }))))
    }

    pub(crate) fn http_key(&self) -> Option<&str> {
        self.0.as_ref().map(|credential| credential.key.as_ref())
    }

    pub(crate) fn key_for(&self, url: &str) -> Option<Arc<str>> {
        let credential = self.0.as_ref()?;
        if is_grpc_url(url) && self.grpc_bearer_for(url).is_none() {
            return None;
        }
        Some(credential.key.clone())
    }

    pub(super) fn grpc_bearer_for(&self, url: &str) -> Option<&AsciiMetadataValue> {
        let credential = self.0.as_deref()?;
        let bearer = credential.grpc_bearer.as_ref()?;
        credential
            .grpc_workers
            .contains(strip_dp_suffix(url).trim_end_matches('/'))
            .then_some(bearer)
    }
}

fn bearer_metadata(key: &str) -> Result<AsciiMetadataValue, &'static str> {
    const INVALID: &str = "worker API key is invalid for ASCII gRPC authorization metadata";
    if !key.is_ascii() {
        return Err(INVALID);
    }
    let mut value: AsciiMetadataValue = format!("Bearer {key}").parse().map_err(|_| INVALID)?;
    value.set_sensitive(true);
    Ok(value)
}

pub(super) fn set_bearer_auth<T>(
    request: &mut tonic::Request<T>,
    key: Option<&str>,
) -> Result<(), &'static str> {
    set_bearer_metadata(request, key.map(bearer_metadata).transpose()?);
    Ok(())
}

/// Clone the prevalidated metadata for an RPC and its cancellation guard.
pub(super) fn set_bearer_metadata<T>(
    request: &mut tonic::Request<T>,
    bearer: Option<AsciiMetadataValue>,
) {
    if let Some(value) = bearer {
        request.metadata_mut().insert("authorization", value);
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn grpc_credentials_require_a_configured_transport_and_valid_metadata() {
        let url = "grpc://worker:50051";
        let auth = WorkerAuth::new(Some("test-key"), [url].into_iter()).unwrap();
        let mut request = tonic::Request::new(());
        set_bearer_auth(
            &mut request,
            auth.key_for("grpc://worker:50051@2").as_deref(),
        )
        .unwrap();
        assert!(request
            .metadata()
            .get("authorization")
            .unwrap()
            .is_sensitive());
        assert!(auth.key_for("grpc://worker:50052").is_none());
        for key in ["invalid\nkey", "non-ascii-ü"] {
            assert!(WorkerAuth::new(Some(key), [url].into_iter()).is_err());
        }
    }
}
