use vllm_router_rs::kv_events::{parse_endpoint_mapping, resolve_endpoints};

const UNSPECIFIED_HOSTS: &[&str] = &[
    "0.0.0.0",
    "0",
    "0x0",
    "[::]",
    "[::ffff:0.0.0.0]",
    "[::ffff:0:0]",
];

#[test]
fn parser_rejects_unspecified_workers_with_valid_publishers() {
    for scheme in ["http", "https"] {
        for host in UNSPECIFIED_HOSTS {
            let mapping = format!("{scheme}://{host}:8000=tcp://publisher:5557");
            assert_eq!(
                parse_endpoint_mapping(&mapping),
                Err("KV connect host cannot be unspecified".into()),
                "{mapping}"
            );
        }
    }
}

#[test]
fn resolver_rejects_unspecified_workers_with_raw_valid_overrides() {
    for scheme in ["http", "https"] {
        for host in UNSPECIFIED_HOSTS {
            let workers = vec![format!("{scheme}://{host}:8000")];
            let mappings = vec![(workers[0].clone(), "tcp://publisher:5557".into())];
            assert_eq!(
                resolve_endpoints(&workers, &mappings, 5558),
                Err("KV connect host cannot be unspecified".into()),
                "{}",
                workers[0]
            );
        }
    }
}

#[test]
fn parser_rejects_wildcard_workers_with_valid_publishers() {
    for scheme in ["http", "https"] {
        let mapping = format!("{scheme}://*:8000=tcp://publisher:5557");
        assert_eq!(
            parse_endpoint_mapping(&mapping),
            Err("KV connect host cannot be a wildcard".into()),
            "{mapping}"
        );
    }
}

#[test]
fn resolver_rejects_wildcard_workers_with_raw_valid_overrides() {
    for scheme in ["http", "https"] {
        let workers = vec![format!("{scheme}://*:8000")];
        let mappings = vec![(workers[0].clone(), "tcp://publisher:5557".into())];
        assert_eq!(
            resolve_endpoints(&workers, &mappings, 5558),
            Err("KV connect host cannot be a wildcard".into()),
            "{}",
            workers[0]
        );
    }
}

#[test]
fn fallback_still_rejects_unspecified_and_wildcard_workers() {
    for scheme in ["http", "https"] {
        for host in UNSPECIFIED_HOSTS {
            assert_eq!(
                resolve_endpoints(&[format!("{scheme}://{host}:8000")], &[], 5557),
                Err("KV connect host cannot be unspecified".into()),
                "{scheme}://{host}"
            );
        }
        assert_eq!(
            resolve_endpoints(&[format!("{scheme}://*:8000")], &[], 5557),
            Err("KV connect host cannot be a wildcard".into())
        );
    }
}

#[test]
fn valid_workers_preserve_registry_keys_with_overrides_and_fallback() {
    for (worker, fallback) in [
        ("HTTP://EXAMPLE.COM:80/", "tcp://example.com:5557"),
        ("HTTPS://EXAMPLE.COM:443/", "tcp://example.com:5557"),
        ("http://127.1:8000/", "tcp://127.0.0.1:5557"),
        ("http://[::1]:8000/", "tcp://[::1]:5557"),
        (
            "HTTP://[0:0:0:0:0:FFFF:7F00:1]:8000/",
            "tcp://127.0.0.1:5557",
        ),
    ] {
        let workers = vec![worker.to_string()];
        let mapping = parse_endpoint_mapping(&format!("{worker}=tcp://publisher:5558")).unwrap();
        assert_eq!(
            resolve_endpoints(&workers, &[mapping], 5557).unwrap(),
            vec![(worker.to_string(), "tcp://publisher:5558".into())]
        );
        assert_eq!(
            resolve_endpoints(&workers, &[], 5557).unwrap(),
            vec![(worker.to_string(), fallback.into())]
        );
    }
}
