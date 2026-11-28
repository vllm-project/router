use url::{Host, Url};

// Worker identity is URL-syntax canonical, not inferred physical-host identity:
// keep mapped IPv6 distinct here, while publisher comparison folds it in
// canonical_endpoint. Returned ownership keys remain the exact registry strings.
pub(crate) fn canonical_worker(value: &str) -> Result<String, String> {
    let parsed = Url::parse(value).map_err(|error| format!("invalid worker URL: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || !matches!(parsed.path(), "" | "/")
    {
        return Err("KV workers must be plain HTTP(S) origins without credentials or paths".into());
    }
    if parsed.port() == Some(0) {
        return Err("KV worker port must be nonzero".into());
    }
    display_host(&parsed)?;
    Ok(parsed.to_string().trim_end_matches('/').to_string())
}

pub(crate) fn canonical_endpoint(value: &str) -> Result<String, String> {
    let parsed = Url::parse(value).map_err(|error| format!("invalid KV endpoint: {error}"))?;
    if parsed.scheme() != "tcp"
        || parsed.host().is_none()
        || parsed.port().is_none_or(|port| port == 0)
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || !matches!(parsed.path(), "" | "/")
    {
        return Err("KV endpoint must be tcp://HOST:PORT without credentials or paths".into());
    }
    Ok(format!(
        "tcp://{}:{}",
        display_host(&parsed)?,
        parsed.port().unwrap()
    ))
}

fn display_host(parsed: &Url) -> Result<String, String> {
    // tcp has opaque hosts. Reuse the special-host parser for deterministic
    // numeric IPv4 canonicalization, without resolving DNS aliases.
    let host = Host::parse(parsed.host_str().ok_or("missing endpoint host")?)
        .map_err(|error| format!("invalid endpoint host: {error}"))?;
    let host = match host {
        Host::Ipv6(address) => address
            .to_ipv4_mapped()
            .map_or(Host::Ipv6(address), Host::Ipv4),
        host => host,
    };
    match host {
        Host::Ipv6(address) if address.is_unspecified() => {
            Err("KV connect host cannot be unspecified".into())
        }
        Host::Ipv4(address) if address.is_unspecified() => {
            Err("KV connect host cannot be unspecified".into())
        }
        Host::Domain(domain) if domain == "*" => Err("KV connect host cannot be a wildcard".into()),
        Host::Ipv6(address) => Ok(format!("[{address}]")),
        Host::Ipv4(address) => Ok(address.to_string()),
        Host::Domain(domain) => Ok(domain.to_ascii_lowercase()),
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn canonical_worker_keeps_the_http_origin_and_rejects_invalid_connect_hosts() {
        for (input, normalized) in [
            ("HTTP://WORKER:80/", "http://worker"),
            ("https://WORKER:443", "https://worker"),
            ("http://127.1:8000/", "http://127.0.0.1:8000"),
            (
                "http://[::ffff:127.0.0.1]:8000/",
                "http://[::ffff:7f00:1]:8000",
            ),
        ] {
            assert_eq!(canonical_worker(input).unwrap(), normalized);
        }
        for input in [
            "http://worker:0",
            "http://user:pass@worker:8000",
            "http://worker:8000/base",
            "http://worker:8000?rank=0",
            "http://worker:8000#rank",
            "http://0.0.0.0:8000",
            "http://[::]:8000",
            "grpc://worker:8000",
        ] {
            assert!(canonical_worker(input).is_err(), "{input}");
        }
    }

    #[test]
    fn canonical_endpoint_folds_equivalent_connect_addresses() {
        for (input, normalized) in [
            ("tcp://EVENTS.Example:5557/", "tcp://events.example:5557"),
            ("tcp://127.1:5557", "tcp://127.0.0.1:5557"),
            ("tcp://[::ffff:127.0.0.1]:5557", "tcp://127.0.0.1:5557"),
            ("tcp://[0:0:0:0:0:0:0:1]:5557", "tcp://[::1]:5557"),
        ] {
            assert_eq!(canonical_endpoint(input).unwrap(), normalized);
        }
        for input in [
            "tcp://*:5557",
            "tcp://[::]:5557",
            "tcp://0.0.0.0:5557",
            "tcp://publisher",
            "tcp://publisher:0",
            "tcp://publisher:5557/path",
            "tcp://publisher:5557?rank=1",
            "tcp://publisher:5557#rank",
            "tcp://user:pass@publisher:5557",
            "http://publisher:5557",
        ] {
            assert!(canonical_endpoint(input).is_err(), "{input}");
        }
    }
}

/// One event source for one (worker, rank).
#[derive(Clone, Debug)]
pub struct EventSource {
    pub source: crate::kv_index::SourceId,
    pub dp_rank: u32,
    pub pub_endpoint: String,
    pub replay_endpoint: Option<String>,
    pub topic: String,
    /// SUB receive high-water mark; None leaves the ZMQ default.
    pub hwm: Option<i32>,
}
