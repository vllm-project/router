use std::collections::{HashMap, HashSet};

use url::{Host, Url};

/// Parse an explicit independently-addressable worker/publisher mapping.
pub fn parse_endpoint_mapping(value: &str) -> Result<(String, String), String> {
    let (worker, endpoint) = value
        .split_once('=')
        .ok_or("KV endpoint mapping must be WORKER_HTTP_URL=tcp://HOST:PORT")?;
    Ok((canonical_worker(worker)?, canonical_endpoint(endpoint)?))
}

pub fn resolve_endpoints(
    worker_urls: &[String],
    mappings: &[(String, String)],
    fallback_port: u16,
) -> Result<Vec<(String, String)>, String> {
    if fallback_port == 0 || worker_urls.is_empty() {
        return Err("KV events require static workers and a nonzero fallback port".into());
    }
    let mut overrides = HashMap::new();
    for (worker, endpoint) in mappings {
        let worker = canonical_worker(worker)?;
        if overrides
            .insert(worker.clone(), canonical_endpoint(endpoint)?)
            .is_some()
        {
            return Err(format!("duplicate KV endpoint mapping for {worker}"));
        }
    }
    let mut workers_seen = HashSet::new();
    let mut endpoints_seen = HashSet::new();
    let mut resolved = Vec::with_capacity(worker_urls.len());
    for worker in worker_urls {
        let canonical = canonical_worker(worker)?;
        if !workers_seen.insert(canonical.clone()) {
            return Err(format!("duplicate static worker URL: {canonical}"));
        }
        let endpoint = if let Some(endpoint) = overrides.remove(&canonical) {
            endpoint
        } else {
            let parsed = Url::parse(&canonical).map_err(|error| error.to_string())?;
            let host = display_host(&parsed)?;
            format!("tcp://{host}:{fallback_port}")
        };
        if !endpoints_seen.insert(endpoint.clone()) {
            return Err(format!(
                "workers cannot share KV endpoint {endpoint}; configure explicit per-worker endpoints"
            ));
        }
        // Keep the registry's exact URL as the ownership key.
        resolved.push((worker.clone(), endpoint));
    }
    if let Some(unknown) = overrides.keys().next() {
        return Err(format!(
            "KV endpoint mapping refers to an unconfigured worker: {unknown}"
        ));
    }
    Ok(resolved)
}

fn canonical_worker(value: &str) -> Result<String, String> {
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
    Ok(parsed.to_string().trim_end_matches('/').to_string())
}

fn canonical_endpoint(value: &str) -> Result<String, String> {
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
    match parsed.host().ok_or("missing endpoint host")? {
        Host::Ipv6(address) => Ok(format!("[{address}]")),
        Host::Ipv4(address) => Ok(address.to_string()),
        // `tcp` is a non-special URL scheme: url::Url's opaque host
        // parser preserves ASCII case, but DNS names are case-insensitive.
        Host::Domain(domain) => Ok(domain.to_ascii_lowercase()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_same_host_workers_and_legacy_fallback() {
        let workers = vec!["http://host:8000".into(), "http://host:8001".into()];
        let mappings = vec![
            parse_endpoint_mapping("http://host:8000/=tcp://host:5557").unwrap(),
            parse_endpoint_mapping("http://host:8001=tcp://host:5558").unwrap(),
        ];
        assert_eq!(
            resolve_endpoints(&workers, &mappings, 5557).unwrap(),
            vec![
                (workers[0].clone(), "tcp://host:5557".into()),
                (workers[1].clone(), "tcp://host:5558".into()),
            ]
        );
        assert!(resolve_endpoints(&workers, &[], 5557).is_err());
        assert_eq!(
            resolve_endpoints(&["http://[::1]:8000".into()], &[], 5557).unwrap()[0].1,
            "tcp://[::1]:5557"
        );
    }

    #[test]
    fn rejects_duplicate_unknown_and_ambiguous_mappings() {
        let workers = vec!["http://host:8000".into()];
        let mapping = parse_endpoint_mapping("http://host:8000=tcp://host:5557").unwrap();
        assert!(resolve_endpoints(&workers, &[mapping.clone(), mapping], 5557).is_err());
        let unknown = parse_endpoint_mapping("http://other:8000=tcp://other:5557").unwrap();
        assert!(resolve_endpoints(&workers, &[unknown], 5557).is_err());
        for bad in [
            "http://host/path=tcp://host:5557",
            "http://host=tcp://host",
            "http://host=ipc:///tmp/publisher",
            "http://host=tcp://host:0",
        ] {
            assert!(parse_endpoint_mapping(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn rejects_shared_endpoint_hidden_by_dns_case() {
        let workers: Vec<String> = vec!["http://host:8000".into(), "http://host:8001".into()];
        let mappings = vec![
            (workers[0].clone(), "tcp://HOST.Example:5557".into()),
            (workers[1].clone(), "tcp://host.example:5557".into()),
        ];
        let error = resolve_endpoints(&workers, &mappings, 5557).unwrap_err();
        assert!(error.contains("cannot share KV endpoint tcp://host.example:5557"));
    }
}
