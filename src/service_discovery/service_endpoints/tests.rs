#![cfg(all(test, feature = "cpu", feature = "ready"))]

use super::*;
use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions};
use serde_json::{json, Value};

fn service_json() -> Value {
    json!({
        "metadata": {
            "name": "chat", "namespace": "inference", "uid": "svc-1",
            "annotations": {
                "router.vllm.ai/model-id": "model-a",
                "router.vllm.ai/inference-port": "http"
            }
        },
        "spec": {"selector": {"app": "vllm"}, "ports": [{"name": "http", "port": 80}]}
    })
}

fn service() -> Service {
    serde_json::from_value(service_json()).unwrap()
}

fn endpoint(uid: &str, ip: &str) -> Endpoint {
    serde_json::from_value(json!({
        "addresses": [ip],
        "targetRef": {"apiVersion": "v1", "kind": "Pod", "namespace": "inference", "uid": uid}
    }))
    .unwrap()
}

fn slice(name: &str, port: u16, endpoints: Vec<Endpoint>) -> EndpointSlice {
    serde_json::from_value(json!({
        "metadata": {
            "name": name, "namespace": "inference",
            "labels": {
                "kubernetes.io/service-name": "chat",
                "endpointslice.kubernetes.io/managed-by": "endpointslice-controller.k8s.io"
            },
            "ownerReferences": [{"apiVersion": "v1", "kind": "Service", "name": "chat", "uid": "svc-1"}]
        },
        "addressType": "IPv4", "ports": [{"name": "http", "port": port}], "endpoints": endpoints
    })).unwrap()
}

fn resolve(slices: &[EndpointSlice]) -> EndpointResolution {
    resolve_endpoints(
        &parse_service(&service()).unwrap(),
        slices,
        AddressFamily::Ipv4,
    )
}

fn expected_endpoint(uid: &str, address: &str, model_id: &str) -> DiscoveredEndpoint {
    DiscoveredEndpoint {
        service: ServiceBinding {
            namespace: "inference".into(),
            name: "chat".into(),
            uid: "svc-1".into(),
            model_id: model_id.into(),
            port_name: "http".into(),
        },
        pod_uid: uid.into(),
        address: address.parse().unwrap(),
    }
}

fn assert_diagnostics(result: &EndpointResolution, expected: &[(&str, &str)]) {
    let actual: Vec<_> = result
        .diagnostics
        .iter()
        .map(|error| (error.resource.as_str(), error.field.as_str()))
        .collect();
    assert_eq!(actual, expected);
}

mod service_contract {
    use super::*;

    #[test]
    fn requires_service_name_namespace_and_uid() {
        for field in ["name", "namespace", "uid"] {
            for value in [Value::Null, json!("")] {
                let mut input = service_json();
                input["metadata"][field] = value;
                let error = parse_service(&serde_json::from_value(input).unwrap()).unwrap_err();
                assert_eq!(error.field, format!("metadata.{field}"));
            }
        }
    }

    #[test]
    fn requires_model_id_and_named_port_annotations() {
        for key in [MODEL_ID_ANNOTATION, INFERENCE_PORT_ANNOTATION] {
            for value in [None, Some(""), Some(" ")] {
                let mut input = service();
                let annotations = input.metadata.annotations.as_mut().unwrap();
                match value {
                    Some(value) => {
                        annotations.insert(key.into(), value.into());
                    }
                    None => {
                        annotations.remove(key);
                    }
                }
                let error = parse_service(&input).unwrap_err();
                assert_eq!(error.resource, "Service inference/chat");
                assert_eq!(error.field, format!("metadata.annotations[{key}]"));
            }
        }
    }

    #[test]
    fn requires_a_selector_and_rejects_publish_not_ready_addresses() {
        for (field, value) in [
            ("selector", Value::Null),
            ("selector", json!({})),
            ("publishNotReadyAddresses", json!(true)),
        ] {
            let mut input = service_json();
            input["spec"][field] = value;
            let error = parse_service(&serde_json::from_value(input).unwrap()).unwrap_err();
            assert_eq!(error.resource, "Service inference/chat");
            assert_eq!(error.field, format!("spec.{field}"));
        }
        let mut input = service();
        input.spec = None;
        assert_eq!(parse_service(&input).unwrap_err().field, "spec");
    }

    #[test]
    fn requires_exactly_one_valid_named_tcp_service_port() {
        for ports in [
            json!([]),
            json!([{"name": "metrics", "port": 80}]),
            json!([{"name": "http", "port": 80}, {"name": "http", "port": 81}]),
            json!([{"name": "http", "port": 80, "protocol": "UDP"}]),
            json!([{"name": "http", "port": 0}]),
            json!([{"name": "http", "port": -1}]),
            json!([{"name": "http", "port": 65536}]),
        ] {
            let mut input = service_json();
            input["spec"]["ports"] = ports.clone();
            let error = parse_service(&serde_json::from_value(input).unwrap()).unwrap_err();
            assert_eq!(error.field, "spec.ports", "{ports}");
        }
    }
}

mod endpoint_resolution {
    use super::*;

    #[test]
    fn returns_pod_addresses_and_the_declared_model_and_service_identity() {
        for model_id in ["model-a", "model-b"] {
            let mut input = service();
            input
                .metadata
                .annotations
                .as_mut()
                .unwrap()
                .insert(MODEL_ID_ANNOTATION.into(), model_id.into());
            let slices = [slice(
                "a",
                8000,
                vec![endpoint("pod-1", "10.0.0.1"), endpoint("pod-2", "10.0.0.2")],
            )];
            let result = resolve_endpoints(
                &parse_service(&input).unwrap(),
                &slices,
                AddressFamily::Ipv4,
            );
            assert_eq!(
                result.endpoints,
                vec![
                    expected_endpoint("pod-1", "10.0.0.1:8000", model_id),
                    expected_endpoint("pod-2", "10.0.0.2:8000", model_id),
                ]
            );
            assert_diagnostics(&result, &[]);
        }
    }

    #[test]
    fn uses_each_slices_named_backend_port_and_keeps_distinct_ports() {
        let mut a = slice("a", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
        let mut metrics = a.ports.as_ref().unwrap()[0].clone();
        metrics.name = Some("metrics".into());
        metrics.port = Some(9090);
        a.ports.as_mut().unwrap().insert(0, metrics);
        let b = slice("b", 9000, vec![endpoint("pod-1", "10.0.0.1")]);
        let result = resolve(&[a, b]);
        assert_eq!(
            result.endpoints,
            vec![
                expected_endpoint("pod-1", "10.0.0.1:8000", "model-a"),
                expected_endpoint("pod-1", "10.0.0.1:9000", "model-a"),
            ]
        );
        assert_diagnostics(&result, &[]);
    }

    #[test]
    fn merges_slices_into_a_sorted_deduplicated_list_regardless_of_input_order() {
        let a = slice(
            "a",
            8000,
            vec![endpoint("pod-a", "10.0.0.2"), endpoint("pod-z", "10.0.0.1")],
        );
        let b = slice(
            "b",
            8000,
            vec![endpoint("pod-m", "10.0.0.3"), endpoint("pod-z", "10.0.0.1")],
        );
        let result = resolve(&[a.clone(), b.clone()]);
        assert_eq!(
            result.endpoints,
            vec![
                expected_endpoint("pod-z", "10.0.0.1:8000", "model-a"),
                expected_endpoint("pod-a", "10.0.0.2:8000", "model-a"),
                expected_endpoint("pod-m", "10.0.0.3:8000", "model-a"),
            ]
        );
        assert_diagnostics(&result, &[]);
        assert_eq!(result, resolve(&[b, a]));
    }

    #[test]
    fn accepts_empty_placeholders_without_ports_and_preserves_other_endpoints() {
        let valid = slice("valid", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
        let expected = resolve(std::slice::from_ref(&valid));
        for ports in [None, Some(vec![])] {
            let mut empty = slice("empty", 8000, vec![]);
            empty.ports = ports;
            assert_eq!(resolve(&[empty.clone()]), EndpointResolution::default());
            assert_eq!(resolve(&[empty, valid.clone()]), expected);
        }
    }

    #[test]
    fn removing_one_duplicate_slice_keeps_the_remaining_endpoint() {
        let a = slice("a", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
        let b = slice("b", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
        let expected = vec![expected_endpoint("pod-1", "10.0.0.1:8000", "model-a")];
        assert_eq!(resolve(&[a, b.clone()]).endpoints, expected);
        assert_eq!(resolve(&[b]).endpoints, expected);
    }

    #[test]
    fn requires_exactly_one_valid_named_tcp_slice_port() {
        for ports in [
            json!([]),
            json!([{"name": "metrics", "port": 8000}]),
            json!([{"name": "http", "port": 8000}, {"name": "http", "port": 9000}]),
            json!([{"name": "http", "port": 8000, "protocol": "UDP"}]),
            json!([{"name": "http"}]),
            json!([{"name": "http", "port": 0}]),
            json!([{"name": "http", "port": -1}]),
            json!([{"name": "http", "port": 65536}]),
        ] {
            let mut input = slice("invalid-port", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
            input.ports = Some(serde_json::from_value(ports.clone()).unwrap());
            let result = resolve(&[input]);
            assert!(result.endpoints.is_empty(), "{ports}");
            assert_diagnostics(
                &result,
                &[("EndpointSlice inference/invalid-port", "ports")],
            );
        }
    }
}

mod readiness {
    use super::*;

    #[test]
    fn filters_all_27_nullable_condition_combinations() {
        let eligible = [
            (None, None, None),
            (None, None, Some(false)),
            (None, Some(true), None),
            (None, Some(true), Some(false)),
            (Some(true), None, None),
            (Some(true), None, Some(false)),
            (Some(true), Some(true), None),
            (Some(true), Some(true), Some(false)),
        ];
        for ready in [None, Some(false), Some(true)] {
            for serving in [None, Some(false), Some(true)] {
                for terminating in [None, Some(false), Some(true)] {
                    let mut target = endpoint("pod-1", "10.0.0.1");
                    target.conditions = Some(EndpointConditions {
                        ready,
                        serving,
                        terminating,
                    });
                    let result = resolve(&[slice("a", 8000, vec![target])]);
                    let expected = if eligible.contains(&(ready, serving, terminating)) {
                        vec![expected_endpoint("pod-1", "10.0.0.1:8000", "model-a")]
                    } else {
                        vec![]
                    };
                    assert_eq!(
                        result.endpoints, expected,
                        "{ready:?}/{serving:?}/{terminating:?}"
                    );
                    assert_diagnostics(&result, &[]);
                }
            }
        }
    }

    #[test]
    fn conflicting_readiness_excludes_only_the_conflicting_target_in_either_order() {
        let a = slice(
            "a",
            8000,
            vec![endpoint("pod-1", "10.0.0.1"), endpoint("pod-2", "10.0.0.2")],
        );
        let mut unready = endpoint("pod-1", "10.0.0.1");
        unready.conditions = Some(EndpointConditions {
            ready: Some(false),
            ..Default::default()
        });
        let b = slice("b", 8000, vec![unready]);
        let result = resolve(&[a.clone(), b.clone()]);
        assert_eq!(
            result.endpoints,
            vec![expected_endpoint("pod-2", "10.0.0.2:8000", "model-a")]
        );
        assert_diagnostics(
            &result,
            &[
                ("EndpointSlice inference/a:endpoints[0]", "conditions"),
                ("EndpointSlice inference/b:endpoints[0]", "conditions"),
            ],
        );
        assert_eq!(result, resolve(&[b, a]));
    }
}

mod slice_lifecycle {
    use super::*;

    fn deleting_slice(uid: &str, ready: bool) -> EndpointSlice {
        let mut target = endpoint(uid, "10.0.0.1");
        target.conditions = Some(EndpointConditions {
            ready: Some(ready),
            ..Default::default()
        });
        let mut deleting = slice("deleting", 8000, vec![target]);
        deleting.metadata.deletion_timestamp =
            Some(serde_json::from_value(json!("2026-11-25T00:00:00Z")).unwrap());
        deleting.metadata.finalizers = Some(vec!["example.com/cleanup".into()]);
        deleting
    }

    #[test]
    fn skips_deleting_slices_even_when_their_endpoints_are_ready() {
        let result = resolve(&[deleting_slice("old-pod", true)]);
        assert!(result.endpoints.is_empty());
        assert_diagnostics(&result, &[]);
    }

    #[test]
    fn deleting_slice_does_not_suppress_a_live_replacement_in_either_order() {
        let live = slice("live", 8000, vec![endpoint("new-pod", "10.0.0.1")]);
        for ready in [false, true] {
            let deleting = deleting_slice("old-pod", ready);
            let result = resolve(&[deleting.clone(), live.clone()]);
            assert_eq!(
                result.endpoints,
                vec![expected_endpoint("new-pod", "10.0.0.1:8000", "model-a")]
            );
            assert_diagnostics(&result, &[]);
            assert_eq!(result, resolve(&[live.clone(), deleting]));
        }
    }

    #[test]
    fn deleting_slice_does_not_create_a_readiness_conflict_in_either_order() {
        let deleting = deleting_slice("pod-1", false);
        let live = slice("live", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
        let result = resolve(&[deleting.clone(), live.clone()]);
        assert_eq!(
            result.endpoints,
            vec![expected_endpoint("pod-1", "10.0.0.1:8000", "model-a")]
        );
        assert_diagnostics(&result, &[]);
        assert_eq!(result, resolve(&[live, deleting]));
    }
}

mod ownership_and_pod_identity {
    use super::*;

    #[test]
    fn empty_slices_still_validate_ownership_controller_and_address_type() {
        let mut empty = slice("empty", 8000, vec![]);
        empty.ports = None;
        let mut stale = empty.clone();
        stale.metadata.owner_references.as_mut().unwrap()[0].uid = "old-service".into();
        let mut custom = empty.clone();
        custom
            .metadata
            .labels
            .as_mut()
            .unwrap()
            .insert(MANAGED_BY_LABEL.into(), "custom-controller".into());
        let mut fqdn = empty;
        fqdn.address_type = "FQDN".into();

        for (input, field) in [
            (stale, "metadata.ownerReferences"),
            (
                custom,
                "metadata.labels[endpointslice.kubernetes.io/managed-by]",
            ),
            (fqdn, "addressType"),
        ] {
            let result = resolve(&[input]);
            assert!(result.endpoints.is_empty());
            assert_diagnostics(&result, &[("EndpointSlice inference/empty", field)]);
        }
    }

    #[test]
    fn rejects_missing_or_stale_service_owners() {
        for uid in [None, Some("old-service")] {
            let mut input = slice("stale", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
            match uid {
                Some(uid) => input.metadata.owner_references.as_mut().unwrap()[0].uid = uid.into(),
                None => input.metadata.owner_references = None,
            }
            let result = resolve(&[input]);
            assert!(result.endpoints.is_empty());
            assert_diagnostics(
                &result,
                &[("EndpointSlice inference/stale", "metadata.ownerReferences")],
            );
        }
    }

    #[test]
    fn rejects_slices_from_unsupported_controllers() {
        let mut input = slice("custom", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
        input
            .metadata
            .labels
            .as_mut()
            .unwrap()
            .insert(MANAGED_BY_LABEL.into(), "custom-controller".into());
        let result = resolve(&[input]);
        assert!(result.endpoints.is_empty());
        assert_diagnostics(
            &result,
            &[(
                "EndpointSlice inference/custom",
                "metadata.labels[endpointslice.kubernetes.io/managed-by]",
            )],
        );
    }

    #[test]
    fn ignores_slices_for_other_namespaces_or_services() {
        let mut other_namespace = slice("a", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
        other_namespace.metadata.namespace = Some("other".into());
        let mut other_service = slice("b", 8000, vec![endpoint("pod-2", "10.0.0.2")]);
        other_service
            .metadata
            .labels
            .as_mut()
            .unwrap()
            .insert(SERVICE_NAME_LABEL.into(), "other".into());
        assert_eq!(
            resolve(&[other_namespace, other_service]),
            EndpointResolution::default()
        );
    }

    #[test]
    fn requires_a_pod_uid_in_the_service_namespace() {
        for (field, value) in [
            ("uid", Value::Null),
            ("uid", json!("")),
            ("kind", json!("Node")),
            ("namespace", json!("other")),
            ("apiVersion", json!("apps/v1")),
        ] {
            let mut target = serde_json::to_value(endpoint("pod-1", "10.0.0.1")).unwrap();
            target["targetRef"][field] = value;
            let result = resolve(&[slice(
                "invalid-pod",
                8000,
                vec![serde_json::from_value(target).unwrap()],
            )]);
            assert!(result.endpoints.is_empty(), "{field}");
            assert_diagnostics(
                &result,
                &[(
                    "EndpointSlice inference/invalid-pod",
                    "endpoints[0].targetRef",
                )],
            );
        }
    }

    #[test]
    fn accepts_implicit_pod_namespace_and_api_version() {
        let mut target = endpoint("pod-1", "10.0.0.1");
        let reference = target.target_ref.as_mut().unwrap();
        reference.namespace = None;
        reference.api_version = None;
        let result = resolve(&[slice("a", 8000, vec![target])]);
        assert_eq!(
            result.endpoints,
            vec![expected_endpoint("pod-1", "10.0.0.1:8000", "model-a")]
        );
        assert_diagnostics(&result, &[]);
    }

    #[test]
    fn preserves_the_new_pod_uid_when_an_address_is_reused_between_snapshots() {
        for uid in ["old-pod", "new-pod"] {
            let result = resolve(&[slice("a", 8000, vec![endpoint(uid, "10.0.0.1")])]);
            assert_eq!(
                result.endpoints,
                vec![expected_endpoint(uid, "10.0.0.1:8000", "model-a")]
            );
            assert_diagnostics(&result, &[]);
        }
    }

    #[test]
    fn excludes_addresses_claimed_by_different_pod_uids_but_keeps_other_targets() {
        let a = slice(
            "a",
            8000,
            vec![
                endpoint("old-pod", "10.0.0.1"),
                endpoint("unrelated", "10.0.0.2"),
            ],
        );
        let b = slice("b", 8000, vec![endpoint("new-pod", "10.0.0.1")]);
        let result = resolve(&[a.clone(), b.clone()]);
        assert_eq!(
            result.endpoints,
            vec![expected_endpoint("unrelated", "10.0.0.2:8000", "model-a")]
        );
        assert_diagnostics(
            &result,
            &[
                ("EndpointSlice inference/a:endpoints[0]", "targetRef.uid"),
                ("EndpointSlice inference/b:endpoints[0]", "targetRef.uid"),
            ],
        );
        assert_eq!(result, resolve(&[b, a]));
    }
}

mod address_families {
    use super::*;

    #[test]
    fn selects_one_family_and_formats_ipv6_with_brackets() {
        let v4 = slice("v4", 8000, vec![endpoint("pod-1", "10.0.0.1")]);
        let mut v6 = slice("v6", 8000, vec![endpoint("pod-1", "2001:db8::1")]);
        v6.address_type = "IPv6".into();
        let binding = parse_service(&service()).unwrap();
        for (family, address) in [
            (AddressFamily::Ipv4, "10.0.0.1:8000"),
            (AddressFamily::Ipv6, "[2001:db8::1]:8000"),
        ] {
            let result = resolve_endpoints(&binding, &[v4.clone(), v6.clone()], family);
            assert_eq!(
                result.endpoints,
                vec![expected_endpoint("pod-1", address, "model-a")]
            );
            assert_eq!(
                format!("http://{}", result.endpoints[0].address),
                format!("http://{address}")
            );
            assert_diagnostics(&result, &[]);
        }
    }

    #[test]
    fn reports_invalid_or_wrong_family_addresses_without_dropping_valid_targets() {
        let input = slice(
            "mixed",
            8000,
            vec![
                endpoint("pod-1", "not-an-ip"),
                endpoint("pod-2", "::1"),
                endpoint("pod-3", "10.0.0.3"),
            ],
        );
        let result = resolve(&[input]);
        assert_eq!(
            result.endpoints,
            vec![expected_endpoint("pod-3", "10.0.0.3:8000", "model-a")]
        );
        assert_diagnostics(
            &result,
            &[
                ("EndpointSlice inference/mixed", "endpoints[0].addresses[0]"),
                ("EndpointSlice inference/mixed", "endpoints[1].addresses[0]"),
            ],
        );
    }

    #[test]
    fn rejects_fqdn_slices() {
        let mut input = slice("fqdn", 8000, vec![endpoint("pod-1", "example.com")]);
        input.address_type = "FQDN".into();
        let result = resolve(&[input]);
        assert!(result.endpoints.is_empty());
        assert_diagnostics(&result, &[("EndpointSlice inference/fqdn", "addressType")]);
    }
}
