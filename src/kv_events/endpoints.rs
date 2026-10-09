use std::collections::{HashMap, HashSet};

use url::{Host, Url};

/// Parse an explicit independently-addressable worker/publisher mapping.
pub fn parse_endpoint_mapping(value: &str) -> Result<(String, String), String> {
    let (worker, endpoint) = value
        .split_once('=')
        .ok_or("KV endpoint mapping must be WORKER_HTTP_URL=tcp://HOST:PORT; for example: http://worker:8000=tcp://publisher:5557")?;
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
            canonical_endpoint(&format!("tcp://{host}:{fallback_port}"))?
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
    display_host(&parsed)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipv4_mapped_explicit_publishers_cannot_hide_shared_endpoints() {
        let workers: Vec<String> = vec!["http://worker:8000".into(), "http://worker:8001".into()];
        for host in [
            "[::ffff:127.0.0.1]",
            "[::ffff:7f00:1]",
            "[0:0:0:0:0:ffff:7f00:1]",
        ] {
            assert_eq!(
                parse_endpoint_mapping(&format!("{}=tcp://{host}:5557", workers[0])).unwrap(),
                (workers[0].clone(), "tcp://127.0.0.1:5557".into()),
                "{host}"
            );
            let mappings = vec![
                (workers[0].clone(), format!("tcp://{host}:5557")),
                (workers[1].clone(), "tcp://127.0.0.1:5557".into()),
            ];
            assert_eq!(
                resolve_endpoints(&workers, &mappings, 5558),
                Err("workers cannot share KV endpoint tcp://127.0.0.1:5557; configure explicit per-worker endpoints".into()),
                "{host}"
            );
        }
    }

    #[test]
    fn ipv4_mapped_explicit_and_fallback_publishers_cannot_share_endpoints() {
        for host in [
            "[::ffff:127.0.0.1]",
            "[::ffff:7f00:1]",
            "[0:0:0:0:0:ffff:7f00:1]",
        ] {
            for (explicit_host, fallback_host) in [(host, "127.0.0.1"), ("127.0.0.1", host)] {
                let workers = vec![
                    "http://worker:8000".into(),
                    format!("http://{fallback_host}:8001"),
                ];
                let mappings = vec![(workers[0].clone(), format!("tcp://{explicit_host}:5557"))];
                assert_eq!(
                    resolve_endpoints(&workers, &mappings, 5557),
                    Err("workers cannot share KV endpoint tcp://127.0.0.1:5557; configure explicit per-worker endpoints".into()),
                    "explicit: {explicit_host}, fallback: {fallback_host}"
                );
            }
        }
    }

    #[test]
    fn ipv4_mapped_unspecified_hosts_are_rejected_at_each_public_boundary() {
        for host in ["[::ffff:0.0.0.0]", "[::ffff:0:0]"] {
            assert_eq!(
                parse_endpoint_mapping(&format!("http://worker:8000=tcp://{host}:5557")),
                Err("KV connect host cannot be unspecified".into()),
                "parse: {host}"
            );
            let mappings = vec![("http://worker:8000".into(), format!("tcp://{host}:5557"))];
            assert_eq!(
                resolve_endpoints(&["http://worker:8000".into()], &mappings, 5557),
                Err("KV connect host cannot be unspecified".into()),
                "raw mapping: {host}"
            );
            assert_eq!(
                resolve_endpoints(&[format!("http://{host}:8000")], &[], 5557),
                Err("KV connect host cannot be unspecified".into()),
                "fallback: {host}"
            );
        }
    }

    #[test]
    fn ipv4_mapped_distinct_publisher_ports_are_accepted() {
        let workers: Vec<String> = vec![
            "http://127.0.0.1:8000".into(),
            "http://[::ffff:127.0.0.1]:8001".into(),
        ];
        let mappings = vec![
            (workers[0].clone(), "tcp://[::ffff:7f00:1]:5557".into()),
            (workers[1].clone(), "tcp://127.0.0.1:5558".into()),
        ];
        let expected = vec![
            (workers[0].clone(), "tcp://127.0.0.1:5557".into()),
            (workers[1].clone(), "tcp://127.0.0.1:5558".into()),
        ];
        assert_eq!(
            resolve_endpoints(&workers, &mappings, 5559).unwrap(),
            expected
        );
        assert_eq!(
            resolve_endpoints(&workers, &mappings[..1], 5558).unwrap(),
            expected
        );
    }

    #[test]
    fn ipv4_mapped_normalization_preserves_native_and_compatible_ipv6() {
        let workers: Vec<String> = vec![
            "http://127.0.0.1:8000".into(),
            "http://[::1]:8001".into(),
            "http://[::127.0.0.1]:8002".into(),
        ];
        let mappings = vec![
            parse_endpoint_mapping("http://127.0.0.1:8000=tcp://127.0.0.1:5557").unwrap(),
            parse_endpoint_mapping("http://[::1]:8001=tcp://[::1]:5557").unwrap(),
            parse_endpoint_mapping("http://[::127.0.0.1]:8002=tcp://[::127.0.0.1]:5557").unwrap(),
        ];
        let expected = vec![
            (workers[0].clone(), "tcp://127.0.0.1:5557".into()),
            (workers[1].clone(), "tcp://[::1]:5557".into()),
            (workers[2].clone(), "tcp://[::7f00:1]:5557".into()),
        ];
        assert_eq!(
            resolve_endpoints(&workers, &mappings, 5558).unwrap(),
            expected
        );
        assert_eq!(resolve_endpoints(&workers, &[], 5557).unwrap(), expected);
    }

    #[test]
    fn ipv4_mapped_publishers_preserve_the_exact_registry_ownership_key() {
        let workers = vec!["HTTP://[0:0:0:0:0:FFFF:7F00:1]:8000/".to_string()];
        let mapping =
            parse_endpoint_mapping("http://[::ffff:127.0.0.1]:8000=tcp://[::FFFF:127.0.0.1]:5557/")
                .unwrap();
        let expected = vec![(workers[0].clone(), "tcp://127.0.0.1:5557".into())];
        assert_eq!(
            resolve_endpoints(&workers, &[mapping], 5558).unwrap(),
            expected
        );
        assert_eq!(resolve_endpoints(&workers, &[], 5557).unwrap(), expected);
    }

    #[test]
    fn numeric_ipv4_aliases_cannot_hide_shared_publishers() {
        let workers: Vec<String> = vec![
            "http://127.0.0.1:8000".into(),
            "http://127.0.0.1:8001".into(),
        ];
        for host in [
            "127.1",
            "2130706433",
            "0177.0.0.1",
            "0x7f000001",
            "127.0.0.1",
        ] {
            let mapping = (workers[0].clone(), format!("tcp://{host}:5557"));
            let explicit = (workers[1].clone(), "tcp://127.0.0.1:5557".into());
            assert!(
                resolve_endpoints(&workers, &[mapping.clone(), explicit], 5558).is_err(),
                "explicit: {host}"
            );
            assert!(
                resolve_endpoints(&workers, &[mapping], 5557).is_err(),
                "fallback: {host}"
            );
        }
        let registry = vec!["HTTP://127.1:8000/".to_string()];
        let mappings = vec![("http://127.0.0.1:8000".into(), "tcp://127.1:5557".into())];
        assert_eq!(
            resolve_endpoints(&registry, &mappings, 5557).unwrap(),
            vec![(registry[0].clone(), "tcp://127.0.0.1:5557".into())]
        );
    }

    #[test]
    fn connect_endpoints_reject_wildcard_and_unspecified_hosts() {
        for host in ["*", "0.0.0.0", "[::]", "0", "0x0"] {
            assert!(
                parse_endpoint_mapping(&format!("http://worker:8000=tcp://{host}:5557")).is_err(),
                "explicit: {host}"
            );
            let mappings = vec![("http://worker:8000".into(), format!("tcp://{host}:5557"))];
            assert!(resolve_endpoints(&["http://worker:8000".into()], &mappings, 5557).is_err());
            assert!(
                resolve_endpoints(&[format!("http://{host}:8000")], &[], 5557).is_err(),
                "fallback: {host}"
            );
        }
    }

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

    #[test]
    fn overrides_normalize_without_rewriting_the_registry_ownership_key() {
        let workers = vec!["HTTP://HOST:80/".to_string()];
        let mapping = parse_endpoint_mapping("http://host=tcp://EVENTS.Example:5558/").unwrap();
        assert_eq!(
            resolve_endpoints(&workers, &[mapping], 5557).unwrap(),
            vec![(workers[0].clone(), "tcp://events.example:5558".to_string())]
        );
    }

    #[test]
    fn rejects_invalid_origins_empty_pools_and_normalized_duplicates() {
        for mapping in [
            "ftp://host=tcp://host:5557",
            "http://user:pass@host=tcp://host:5557",
            "http://host?query=1=tcp://host:5557",
            "http://host#fragment=tcp://host:5557",
            "http://host=tcp://user@host:5557",
            "http://host=tcp://host:5557/path",
            "http://host=tcp://host:5557?query=1",
            "http://host=tcp://[::1]:0",
        ] {
            assert!(parse_endpoint_mapping(mapping).is_err(), "{mapping}");
        }
        assert!(resolve_endpoints(&[], &[], 5557).is_err());
        assert!(resolve_endpoints(&["http://host".into()], &[], 0).is_err());
        let duplicate_workers = vec!["HTTP://HOST:80/".into(), "http://host".into()];
        assert!(resolve_endpoints(&duplicate_workers, &[], 5557).is_err());
        let workers = vec!["http://host:8000".into(), "http://host:8001".into()];
        let mappings = vec![
            parse_endpoint_mapping("http://host:8000=tcp://[::1]:5557").unwrap(),
            parse_endpoint_mapping("http://host:8001=tcp://[0:0:0:0:0:0:0:1]:5557").unwrap(),
        ];
        assert!(resolve_endpoints(&workers, &mappings, 5557).is_err());
    }
}
