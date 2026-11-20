//! Resolve a selected Service and a complete EndpointSlice snapshot without API calls.
//! The metadata keys below are proposed by #530; no discovery watcher uses them yet.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, SocketAddr};

use k8s_openapi::api::{core::v1::Service, discovery::v1::EndpointSlice};

pub const MODEL_ID_ANNOTATION: &str = "router.vllm.ai/model-id";
pub const INFERENCE_PORT_ANNOTATION: &str = "router.vllm.ai/inference-port";
const SERVICE_NAME_LABEL: &str = "kubernetes.io/service-name";
const MANAGED_BY_LABEL: &str = "endpointslice.kubernetes.io/managed-by";
const ENDPOINTSLICE_CONTROLLER: &str = "endpointslice-controller.k8s.io";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceBinding {
    pub namespace: String,
    pub name: String,
    pub uid: String,
    pub model_id: String,
    pub port_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressFamily {
    Ipv4,
    Ipv6,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredEndpoint {
    pub service: ServiceBinding,
    pub pod_uid: String,
    pub address: SocketAddr,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, thiserror::Error)]
#[error("{resource}: {field}: {message}")]
pub struct DiscoveryError {
    pub resource: String,
    pub field: String,
    pub message: String,
}

impl DiscoveryError {
    fn new(resource: &str, field: &str, message: &str) -> Self {
        Self {
            resource: resource.into(),
            field: field.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct EndpointResolution {
    pub endpoints: Vec<DiscoveredEndpoint>,
    pub diagnostics: Vec<DiscoveryError>,
}

fn required<'a>(
    value: Option<&'a str>,
    resource: &str,
    field: &str,
) -> Result<&'a str, DiscoveryError> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| DiscoveryError::new(resource, field, "required nonempty value"))
}

fn tcp_port(port: Option<i32>, protocol: Option<&str>) -> Option<u16> {
    if protocol.unwrap_or("TCP") != "TCP" {
        return None;
    }
    port.and_then(|port| u16::try_from(port).ok())
        .filter(|port| *port != 0)
}

/// The caller selects Services by namespace and label before invoking this parser.
pub fn parse_service(service: &Service) -> Result<ServiceBinding, DiscoveryError> {
    let metadata = &service.metadata;
    let resource = format!(
        "Service {}/{}",
        metadata.namespace.as_deref().unwrap_or("?"),
        metadata.name.as_deref().unwrap_or("?")
    );
    let name = required(metadata.name.as_deref(), &resource, "metadata.name")?;
    let namespace = required(
        metadata.namespace.as_deref(),
        &resource,
        "metadata.namespace",
    )?;
    let uid = required(metadata.uid.as_deref(), &resource, "metadata.uid")?;
    let annotation = |key| {
        required(
            metadata
                .annotations
                .as_ref()
                .and_then(|a| a.get(key))
                .map(String::as_str),
            &resource,
            &format!("metadata.annotations[{key}]"),
        )
    };
    let model_id = annotation(MODEL_ID_ANNOTATION)?;
    let port_name = annotation(INFERENCE_PORT_ANNOTATION)?;
    let spec = service
        .spec
        .as_ref()
        .ok_or_else(|| DiscoveryError::new(&resource, "spec", "required for Service discovery"))?;
    if spec
        .selector
        .as_ref()
        .is_none_or(|selector| selector.is_empty())
    {
        return Err(DiscoveryError::new(
            &resource,
            "spec.selector",
            "only selector-backed Services are supported",
        ));
    }
    if spec.publish_not_ready_addresses == Some(true) {
        return Err(DiscoveryError::new(
            &resource,
            "spec.publishNotReadyAddresses",
            "must not override endpoint readiness",
        ));
    }
    let ports: Vec<_> = spec
        .ports
        .iter()
        .flatten()
        .filter(|p| p.name.as_deref() == Some(port_name))
        .collect();
    if ports.len() != 1 || tcp_port(Some(ports[0].port), ports[0].protocol.as_deref()).is_none() {
        return Err(DiscoveryError::new(
            &resource,
            "spec.ports",
            "inference-port must name exactly one valid TCP port",
        ));
    }
    Ok(ServiceBinding {
        namespace: namespace.into(),
        name: name.into(),
        uid: uid.into(),
        model_id: model_id.into(),
        port_name: port_name.into(),
    })
}

/// Resolve endpoints from the complete current EndpointSlice set for a Service.
/// Do not pass only the Slice changed by a watch event.
///
/// Report invalid resources and conflicting targets in diagnostics,
/// excluding them from the result while retaining unrelated valid endpoints.
pub fn resolve_endpoints(
    service: &ServiceBinding,
    slices: &[EndpointSlice],
    family: AddressFamily,
) -> EndpointResolution {
    // Retain ineligible observations until duplicate conflicts have been checked.
    let mut observations: BTreeMap<(String, SocketAddr), BTreeSet<(String, bool)>> =
        BTreeMap::new();
    let mut diagnostics = BTreeSet::new();
    for slice in slices {
        let labels = slice.metadata.labels.as_ref();
        if slice.metadata.namespace.as_deref() != Some(&service.namespace)
            || labels.and_then(|l| l.get(SERVICE_NAME_LABEL)) != Some(&service.name)
        {
            continue;
        }
        let resource = format!(
            "EndpointSlice {}/{}",
            service.namespace,
            slice.metadata.name.as_deref().unwrap_or("?")
        );
        let error = |field: &str, message: &str| DiscoveryError::new(&resource, field, message);
        let owners = slice
            .metadata
            .owner_references
            .as_deref()
            .unwrap_or_default();
        if !owners.iter().any(|owner| {
            owner.api_version == "v1"
                && owner.kind == "Service"
                && owner.name == service.name
                && owner.uid == service.uid
        }) {
            diagnostics.insert(error(
                "metadata.ownerReferences",
                "no owner matching the selected Service name and UID",
            ));
            continue;
        }
        if labels
            .and_then(|l| l.get(MANAGED_BY_LABEL))
            .map(String::as_str)
            != Some(ENDPOINTSLICE_CONTROLLER)
        {
            diagnostics.insert(error(
                "metadata.labels[endpointslice.kubernetes.io/managed-by]",
                "only Kubernetes EndpointSlice controller output is supported",
            ));
            continue;
        }
        match (family, slice.address_type.as_str()) {
            (AddressFamily::Ipv4, "IPv4") | (AddressFamily::Ipv6, "IPv6") => (),
            (_, "IPv4" | "IPv6") => continue,
            _ => {
                diagnostics.insert(error("addressType", "only IPv4 and IPv6 are supported"));
                continue;
            }
        }
        if slice.endpoints.is_empty() {
            continue;
        }
        let ports: Vec<_> = slice
            .ports
            .iter()
            .flatten()
            .filter(|p| p.name.as_deref() == Some(&service.port_name))
            .collect();
        let port = if ports.len() == 1 {
            tcp_port(ports[0].port, ports[0].protocol.as_deref())
        } else {
            None
        };
        let Some(port) = port else {
            diagnostics.insert(error(
                "ports",
                "inference-port must name exactly one valid TCP port",
            ));
            continue;
        };
        for (index, endpoint) in slice.endpoints.iter().enumerate() {
            let field = format!("endpoints[{index}].targetRef");
            let Some(target) = endpoint.target_ref.as_ref().filter(|target| {
                target.kind.as_deref() == Some("Pod")
                    && target.api_version.as_deref().is_none_or(|v| v == "v1")
                    && target
                        .namespace
                        .as_deref()
                        .is_none_or(|ns| ns == service.namespace)
                    && target
                        .uid
                        .as_ref()
                        .is_some_and(|uid| !uid.trim().is_empty())
            }) else {
                diagnostics.insert(error(
                    &field,
                    "expected a Pod reference with a UID in the Service namespace",
                ));
                continue;
            };
            let conditions = endpoint.conditions.as_ref();
            let eligible = conditions.and_then(|c| c.ready).unwrap_or(true)
                && conditions.and_then(|c| c.serving).unwrap_or(true)
                && !conditions.and_then(|c| c.terminating).unwrap_or(false);
            for (address_index, address) in endpoint.addresses.iter().enumerate() {
                let ip = address.parse::<IpAddr>().ok().filter(|ip| {
                    matches!(
                        (family, ip),
                        (AddressFamily::Ipv4, IpAddr::V4(_)) | (AddressFamily::Ipv6, IpAddr::V6(_))
                    )
                });
                let Some(ip) = ip else {
                    diagnostics.insert(error(
                        &format!("endpoints[{index}].addresses[{address_index}]"),
                        "expected an IP address matching addressType",
                    ));
                    continue;
                };
                let source = format!("{resource}:endpoints[{index}]");
                observations
                    .entry((target.uid.clone().unwrap(), SocketAddr::new(ip, port)))
                    .or_default()
                    .insert((source, eligible));
            }
        }
    }

    let mut socket_owners: BTreeMap<SocketAddr, BTreeSet<&str>> = BTreeMap::new();
    for (uid, address) in observations.keys() {
        socket_owners.entry(*address).or_default().insert(uid);
    }
    let mut endpoints = Vec::new();
    for ((pod_uid, address), sources) in &observations {
        let conflicting_uid = socket_owners[address].len() > 1;
        let conflicting_conditions = sources.iter().any(|(_, eligible)| *eligible)
            && sources.iter().any(|(_, eligible)| !eligible);
        if conflicting_uid || conflicting_conditions {
            for (source, _) in sources {
                diagnostics.insert(DiscoveryError::new(
                    source,
                    if conflicting_uid {
                        "targetRef.uid"
                    } else {
                        "conditions"
                    },
                    if conflicting_uid {
                        "address is claimed by different Pod UIDs"
                    } else {
                        "duplicate target has conflicting readiness conditions"
                    },
                ));
            }
        } else if sources.iter().all(|(_, eligible)| *eligible) {
            endpoints.push(DiscoveredEndpoint {
                service: service.clone(),
                pod_uid: pod_uid.clone(),
                address: *address,
            });
        }
    }
    endpoints.sort_unstable_by_key(|endpoint| endpoint.address);
    EndpointResolution {
        endpoints,
        diagnostics: diagnostics.into_iter().collect(),
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests;
