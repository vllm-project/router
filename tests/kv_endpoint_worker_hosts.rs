#![cfg(all(feature = "cpu", feature = "ready"))]

use vllm_router_rs::kv_index::{parse_endpoint_mapping, resolve_endpoints};

#[test]
fn valid_workers_preserve_registry_keys_with_overrides_and_fallback() {
    for (worker, fallback) in [
        ("http://example.com", "tcp://example.com:5557"),
        ("https://example.com", "tcp://example.com:5557"),
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

#[test]
fn parser_rejects_worker_port_zero_with_valid_publisher() {
    for scheme in ["http", "https"] {
        assert_eq!(
            parse_endpoint_mapping(&format!("{scheme}://worker:0=tcp://publisher:5557")),
            Err("KV worker port must be nonzero".into())
        );
    }
}

#[test]
fn resolver_rejects_worker_port_zero_with_raw_override() {
    for scheme in ["http", "https"] {
        let worker = format!("{scheme}://worker:0");
        assert_eq!(
            resolve_endpoints(
                std::slice::from_ref(&worker),
                &[(worker.clone(), "tcp://publisher:5557".into())],
                5558,
            ),
            Err("KV worker port must be nonzero".into())
        );
    }
}

#[test]
fn fallback_rejects_worker_port_zero() {
    for scheme in ["http", "https"] {
        assert_eq!(
            resolve_endpoints(&[format!("{scheme}://worker:0")], &[], 5557),
            Err("KV worker port must be nonzero".into())
        );
    }
}

#[test]
fn unknown_worker_diagnostic_chooses_smallest_name_in_both_input_orders() {
    let workers = vec!["http://configured:8000".to_string()];
    let a = parse_endpoint_mapping("http://a-unknown:8000=tcp://publisher-a:5557").unwrap();
    let z = parse_endpoint_mapping("http://z-unknown:8000=tcp://publisher-z:5557").unwrap();
    for mappings in [[a.clone(), z.clone()], [z, a]] {
        assert_eq!(
            resolve_endpoints(&workers, &mappings, 5558),
            Err(
                "KV endpoint mapping refers to an unconfigured worker: http://a-unknown:8000"
                    .into()
            )
        );
    }
}
