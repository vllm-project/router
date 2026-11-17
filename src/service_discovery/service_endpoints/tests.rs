#![cfg(all(test, feature = "cpu", feature = "ready"))]

use super::*;
use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions};
use serde_json::{json, Value};

fn service() -> Service {
    serde_json::from_value(json!({
        "metadata": {
            "name": "chat", "namespace": "inference", "uid": "svc-1",
            "annotations": {
                "router.vllm.ai/model-id": "model-a",
                "router.vllm.ai/inference-port": "http"
            }
        },
        "spec": {"selector": {"app": "vllm"}, "ports": [{"name": "http", "port": 80}]}
    }))
    .unwrap()
}

fn endpoint(uid: &str, ip: &str) -> Endpoint {
    serde_json::from_value(json!({
        "addresses": [ip],
        "targetRef": {"apiVersion": "v1", "kind": "Pod", "namespace": "inference", "uid": uid}
    }))
    .unwrap()
}

fn slice(name: &str) -> EndpointSlice {
    serde_json::from_value(json!({
        "metadata": {
            "name": name, "namespace": "inference",
            "labels": {
                "kubernetes.io/service-name": "chat",
                "endpointslice.kubernetes.io/managed-by": "endpointslice-controller.k8s.io"
            },
            "ownerReferences": [{"apiVersion": "v1", "kind": "Service", "name": "chat", "uid": "svc-1"}]
        },
        "addressType": "IPv4", "ports": [{"name": "http", "port": 8000}], "endpoints": []
    })).unwrap()
}

fn resolve(slices: &[EndpointSlice]) -> EndpointResolution {
    resolve_endpoints(
        &parse_service(&service()).unwrap(),
        slices,
        AddressFamily::Ipv4,
    )
}

#[test]
fn resolves_backend_port_and_preserves_model_and_service_identity() {
    let mut a = slice("a");
    a.endpoints.push(endpoint("pod-1", "10.0.0.1"));
    let result = resolve(&[a]);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.endpoints.len(), 1);
    let target = &result.endpoints[0];
    assert_eq!(target.address, "10.0.0.1:8000".parse().unwrap());
    assert_eq!(target.pod_uid, "pod-1");
    assert_eq!(target.service, parse_service(&service()).unwrap());

    let mut other_model = service();
    other_model
        .metadata
        .annotations
        .as_mut()
        .unwrap()
        .insert(MODEL_ID_ANNOTATION.into(), "model-b".into());
    assert_eq!(parse_service(&other_model).unwrap().model_id, "model-b");
}

#[test]
fn rejects_invalid_service_fields_with_resource_and_field() {
    let cases: &[(&str, Value, &str)] = &[
        ("/metadata/name", json!(""), "metadata.name"),
        ("/metadata/namespace", Value::Null, "metadata.namespace"),
        ("/metadata/uid", Value::Null, "metadata.uid"),
        (
            "/metadata/annotations/router.vllm.ai~1model-id",
            json!(" "),
            "metadata.annotations[router.vllm.ai/model-id]",
        ),
        (
            "/metadata/annotations/router.vllm.ai~1inference-port",
            json!(""),
            "metadata.annotations[router.vllm.ai/inference-port]",
        ),
        (
            "/metadata/annotations",
            Value::Null,
            "metadata.annotations[router.vllm.ai/model-id]",
        ),
        ("/spec", Value::Null, "spec"),
        ("/spec/selector", json!({}), "spec.selector"),
        (
            "/spec/publishNotReadyAddresses",
            json!(true),
            "spec.publishNotReadyAddresses",
        ),
        ("/spec/ports", json!([]), "spec.ports"),
        ("/spec/ports/0/name", json!("grpc"), "spec.ports"),
        ("/spec/ports/0/port", json!(0), "spec.ports"),
        ("/spec/ports/0/port", json!(65536), "spec.ports"),
    ];
    for (pointer, value, field) in cases {
        let mut input = serde_json::to_value(service()).unwrap();
        // Optional fields may be absent after serialization.
        if *pointer == "/spec/publishNotReadyAddresses" {
            input["spec"]["publishNotReadyAddresses"] = value.clone();
        } else {
            *input.pointer_mut(pointer).unwrap() = value.clone();
        }
        let error = parse_service(&serde_json::from_value(input).unwrap()).unwrap_err();
        assert_eq!(error.field, *field, "{pointer}");
        assert!(error.resource.starts_with("Service "), "{pointer}");
        assert!(error.resource.contains('/'), "{pointer}");
    }
    let mut input = service();
    let ports = input.spec.as_mut().unwrap().ports.as_mut().unwrap();
    ports.push(ports[0].clone());
    assert_eq!(parse_service(&input).unwrap_err().field, "spec.ports");
    let mut input = service();
    input.spec.as_mut().unwrap().ports.as_mut().unwrap()[0].protocol = Some("UDP".into());
    assert_eq!(parse_service(&input).unwrap_err().field, "spec.ports");
}

#[test]
fn duplicate_slices_are_order_independent_and_one_copy_can_be_removed() {
    let mut a = slice("a");
    a.endpoints = vec![endpoint("pod-1", "10.0.0.1"), endpoint("pod-2", "10.0.0.2")];
    let mut b = a.clone();
    b.metadata.name = Some("b".into());
    assert_eq!(
        resolve(&[a.clone(), b.clone()]),
        resolve(&[b.clone(), a.clone()])
    );
    assert_eq!(resolve(&[a, b.clone()]), resolve(&[b]));
}

#[test]
fn same_pod_on_different_ports_remains_distinct() {
    let mut a = slice("a");
    a.endpoints.push(endpoint("pod-1", "10.0.0.1"));
    let mut b = a.clone();
    b.metadata.name = Some("b".into());
    b.ports.as_mut().unwrap()[0].port = Some(9000);
    let result = resolve(&[a, b]);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result
            .endpoints
            .iter()
            .map(|e| e.address.port())
            .collect::<Vec<_>>(),
        vec![8000, 9000]
    );
}

#[test]
fn checks_all_nullable_condition_combinations() {
    for ready in [None, Some(false), Some(true)] {
        for serving in [None, Some(false), Some(true)] {
            for terminating in [None, Some(false), Some(true)] {
                let mut a = slice("a");
                let mut target = endpoint("pod-1", "10.0.0.1");
                target.conditions = Some(EndpointConditions {
                    ready,
                    serving,
                    terminating,
                });
                a.endpoints.push(target);
                let expected = ready.unwrap_or(true)
                    && serving.unwrap_or(true)
                    && !terminating.unwrap_or(false);
                let result = resolve(&[a]);
                assert!(result.diagnostics.is_empty());
                assert_eq!(
                    result.endpoints.len(),
                    usize::from(expected),
                    "{ready:?}/{serving:?}/{terminating:?}"
                );
            }
        }
    }
}

#[test]
fn conflicting_readiness_excludes_only_the_affected_target() {
    let mut a = slice("a");
    a.endpoints = vec![endpoint("pod-1", "10.0.0.1"), endpoint("pod-2", "10.0.0.2")];
    let mut b = slice("b");
    let mut unready = endpoint("pod-1", "10.0.0.1");
    unready.conditions = Some(EndpointConditions {
        ready: Some(false),
        ..Default::default()
    });
    b.endpoints.push(unready);
    let result = resolve(&[a.clone(), b.clone()]);
    assert_eq!(result.endpoints.len(), 1);
    assert_eq!(result.endpoints[0].pod_uid, "pod-2");
    assert_eq!(result.diagnostics.len(), 2);
    assert!(result
        .diagnostics
        .iter()
        .all(|error| error.field == "conditions"));
    assert_eq!(result, resolve(&[b, a]));
}

#[test]
fn rejects_stale_or_missing_owner_and_unsupported_controller() {
    let mut a = slice("a");
    a.endpoints.push(endpoint("pod-1", "10.0.0.1"));
    for owners in [None, Some(vec![]), {
        let mut owners = a.metadata.owner_references.clone().unwrap();
        owners[0].uid = "old-service".into();
        Some(owners)
    }] {
        let mut invalid = a.clone();
        invalid.metadata.owner_references = owners;
        let result = resolve(&[invalid]);
        assert!(result.endpoints.is_empty());
        assert_eq!(result.diagnostics[0].field, "metadata.ownerReferences");
    }
    a.metadata
        .labels
        .as_mut()
        .unwrap()
        .insert(MANAGED_BY_LABEL.into(), "custom-controller".into());
    assert!(resolve(&[a]).diagnostics[0].field.contains("managed-by"));
}

#[test]
fn ignores_slices_for_other_namespaces_and_services() {
    let mut a = slice("a");
    a.endpoints.push(endpoint("pod-1", "10.0.0.1"));
    let mut b = a.clone();
    a.metadata.namespace = Some("other".into());
    b.metadata
        .labels
        .as_mut()
        .unwrap()
        .insert(SERVICE_NAME_LABEL.into(), "other".into());
    assert_eq!(resolve(&[a, b]), EndpointResolution::default());
}

#[test]
fn rejects_invalid_slice_ports() {
    for port in [None, Some(0), Some(-1), Some(65536)] {
        let mut a = slice("a");
        a.ports.as_mut().unwrap()[0].port = port;
        assert_eq!(resolve(&[a]).diagnostics[0].field, "ports");
    }
    for ports in [None, Some(vec![]), {
        let port = slice("a").ports.unwrap()[0].clone();
        Some(vec![port.clone(), port])
    }] {
        let mut a = slice("a");
        a.ports = ports;
        assert_eq!(resolve(&[a]).diagnostics[0].field, "ports");
    }
    let mut a = slice("a");
    a.ports.as_mut().unwrap()[0].protocol = Some("UDP".into());
    assert_eq!(resolve(&[a]).diagnostics[0].field, "ports");
}

#[test]
fn requires_a_same_namespace_pod_uid_but_accepts_implicit_namespace() {
    for (field, value) in [
        ("uid", Value::Null),
        ("uid", json!("")),
        ("kind", json!("Node")),
        ("namespace", json!("other")),
        ("apiVersion", json!("apps/v1")),
    ] {
        let mut a = slice("a");
        let mut target = serde_json::to_value(endpoint("pod-1", "10.0.0.1")).unwrap();
        target["targetRef"][field] = value;
        a.endpoints.push(serde_json::from_value(target).unwrap());
        let result = resolve(&[a]);
        assert!(result.endpoints.is_empty());
        assert_eq!(result.diagnostics[0].field, "endpoints[0].targetRef");
    }
    let mut a = slice("a");
    let mut target = endpoint("pod-1", "10.0.0.1");
    let reference = target.target_ref.as_mut().unwrap();
    reference.namespace = None;
    reference.api_version = None;
    a.endpoints.push(target);
    assert_eq!(resolve(&[a]).endpoints.len(), 1);
}

#[test]
fn handles_dual_stack_and_reports_invalid_addresses() {
    let mut v4 = slice("v4");
    v4.endpoints.push(endpoint("pod-1", "10.0.0.1"));
    let mut v6 = slice("v6");
    v6.address_type = "IPv6".into();
    v6.endpoints.push(endpoint("pod-1", "2001:db8::1"));
    let binding = parse_service(&service()).unwrap();
    let result = resolve_endpoints(&binding, &[v4.clone(), v6.clone()], AddressFamily::Ipv6);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.endpoints.len(), 1);
    assert_eq!(
        format!("http://{}", result.endpoints[0].address),
        "http://[2001:db8::1]:8000"
    );
    assert_eq!(resolve(&[v4.clone(), v6]), resolve(&[v4]));
    let mut invalid = slice("invalid");
    invalid.endpoints = vec![
        endpoint("pod-1", "not-an-ip"),
        endpoint("pod-2", "::1"),
        endpoint("pod-3", "10.0.0.3"),
    ];
    let result = resolve(&[invalid.clone()]);
    assert_eq!(result.endpoints.len(), 1);
    assert_eq!(result.diagnostics.len(), 2);
    invalid.address_type = "FQDN".into();
    assert_eq!(resolve(&[invalid]).diagnostics[0].field, "addressType");
}

#[test]
fn detects_address_reuse_without_losing_pod_identity() {
    let mut old = slice("old");
    old.endpoints.push(endpoint("old-pod", "10.0.0.1"));
    let mut new = slice("new");
    new.endpoints.push(endpoint("new-pod", "10.0.0.1"));
    assert_eq!(resolve(&[old.clone()]).endpoints[0].pod_uid, "old-pod");
    assert_eq!(resolve(&[new.clone()]).endpoints[0].pod_uid, "new-pod");
    let conflicting = resolve(&[old.clone(), new.clone()]);
    assert!(conflicting.endpoints.is_empty());
    assert_eq!(conflicting.diagnostics.len(), 2);
    assert!(conflicting
        .diagnostics
        .iter()
        .all(|error| error.field == "targetRef.uid"));
    assert_eq!(conflicting, resolve(&[new, old]));
}
