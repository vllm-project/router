//! Per-worker peer KV transfer registrations for `kv_aware`.
//!
//! The directory holds what the router knows about each worker's transfer
//! capability and side-channel address. It has no lock of its own: callers
//! hold `KvAwarePolicy::selection_lock` for registration mutations (the
//! `store`, `register_transfer_capability`, `apply_lookup` and
//! `drop_engine_owned` methods), exactly as they did before it was extracted.
//! `pin_trusted_host` is the exception: it writes only `trusted_hosts` and is
//! called without the lock, also as before. Readers go through the map's own
//! synchronization and see whole `Arc` records.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use super::peer_fetch::{PeerEndpoint, TransferCapability};
use crate::kv_index::PeerLookupRecord;

/// Everything the router knows about a worker's peer KV transfer ability,
/// replaced as a whole so a reader sees one consistent registration.
#[derive(Debug, Clone)]
pub(super) struct PeerRegistration {
    /// Changes only when the contents change; from `next_peer_epoch`. The
    /// grant compares it with the planned registration's to detect a change.
    pub(super) epoch: u64,
    pub(super) capability: Option<TransferCapability>,
    pub(super) endpoint: Option<PeerEndpoint>,
    /// `capability` came from an engine record, not static config. Only such
    /// a capability is dropped when re-discovery finds no record or the
    /// worker is unregistered.
    pub(super) engine_owned: bool,
}

#[derive(Debug)]
pub(super) struct PeerDirectory {
    /// Peer KV transfer registration per worker URL. Writers replace the whole
    /// `Arc` under `selection_lock`; readers clone it and see one consistent
    /// capability/endpoint pair. A capability without an endpoint can be
    /// planned but not fetched.
    pub(super) peer_regs: DashMap<String, Arc<PeerRegistration>>,
    /// Side-channel hosts the operator pinned per worker URL (normalized).
    /// Static config only: engine records never set one, and a pin outlives
    /// unregister like the static capability it came with.
    pub(super) trusted_hosts: DashMap<String, Arc<str>>,
    /// Source of `PeerRegistration::epoch`; starts at 1.
    pub(super) next_peer_epoch: AtomicU64,
}

impl PeerDirectory {
    pub(super) fn new() -> Self {
        Self {
            peer_regs: DashMap::new(),
            trusted_hosts: DashMap::new(),
            next_peer_epoch: AtomicU64::new(1),
        }
    }

    /// Replace `worker_url`'s registration with the given contents. Both parts
    /// empty removes it; unchanged contents keep the existing `Arc` and epoch.
    /// The caller holds `selection_lock`.
    pub(super) fn store(
        &self,
        worker_url: &str,
        capability: Option<TransferCapability>,
        endpoint: Option<PeerEndpoint>,
        engine_owned: bool,
    ) {
        if capability.is_none() && endpoint.is_none() {
            self.peer_regs.remove(worker_url);
            return;
        }
        if let Some(current) = self.peer_regs.get(worker_url) {
            if current.capability == capability
                && current.endpoint == endpoint
                && current.engine_owned == engine_owned
            {
                return;
            }
        }
        self.peer_regs.insert(
            worker_url.to_owned(),
            Arc::new(PeerRegistration {
                epoch: self.next_peer_epoch.fetch_add(1, Ordering::AcqRel),
                capability,
                endpoint,
                engine_owned,
            }),
        );
    }

    /// Declare what a worker can do for KV transfer. Replaced on re-register;
    /// the worker's reported endpoint is kept.
    /// Survives `drop_engine_owned`: only engine-reported capabilities are
    /// dropped when a worker is removed. The caller holds `selection_lock`.
    pub(super) fn register_transfer_capability(
        &self,
        worker_url: &str,
        capability: TransferCapability,
    ) {
        let endpoint = self
            .peer_regs
            .get(worker_url)
            .and_then(|reg| reg.endpoint.clone());
        // An explicit declaration is never engine-sourced.
        self.store(worker_url, Some(capability), endpoint, false);
    }

    /// Apply (or clear, with `None`) the engine's validated peer-lookup record
    /// for a registered worker (the caller has checked it is registered and
    /// holds `selection_lock`).
    ///
    /// A static capability for the worker wins over the engine record (an
    /// operator override); the address is still stored. With `None`, only an
    /// engine-sourced capability is removed.
    pub(super) fn apply_lookup(&self, worker_url: &str, record: Option<&PeerLookupRecord>) {
        let current = self.peer_regs.get(worker_url).map(|reg| Arc::clone(&reg));
        let (current_capability, engine_owned) = match current.as_deref() {
            Some(reg) => (reg.capability.clone(), reg.engine_owned),
            None => (None, false),
        };
        let Some(record) = record else {
            let capability = if engine_owned {
                None
            } else {
                current_capability
            };
            self.store(worker_url, capability, None, false);
            return;
        };
        let endpoint = PeerEndpoint::from(record);
        if current_capability.is_some() && !engine_owned {
            self.store(worker_url, current_capability, Some(endpoint), false);
            return;
        }
        let capability = TransferCapability::from_engine_record(record);
        self.store(worker_url, Some(capability), Some(endpoint), true);
    }

    /// Drop what the engine reported for a removed worker, keeping a static
    /// capability (operator config outlives the worker) but not its address.
    /// The caller holds `selection_lock`.
    pub(super) fn drop_engine_owned(&self, worker_url: &str) {
        let kept = self
            .peer_regs
            .get(worker_url)
            .map(|reg| Arc::clone(&reg))
            .filter(|reg| !reg.engine_owned)
            .and_then(|reg| reg.capability.clone());
        self.store(worker_url, kept, None, false);
    }

    /// Where `worker_url` accepts peer KV fetches, if its engine reported a
    /// valid record. A capability without an address can be planned but not
    /// fetched; active mode requires this on the source.
    pub(super) fn peer_endpoint(&self, worker_url: &str) -> Option<PeerEndpoint> {
        self.peer_regs
            .get(worker_url)
            .and_then(|reg| reg.endpoint.clone())
    }

    /// The current registration, cloned so no map ref is held.
    pub(super) fn registration(&self, worker_url: &str) -> Option<Arc<PeerRegistration>> {
        self.peer_regs.get(worker_url).map(|reg| Arc::clone(&reg))
    }

    /// The current registration's epoch, `None` when there is none.
    pub(super) fn epoch_of(&self, worker_url: &str) -> Option<u64> {
        self.peer_regs.get(worker_url).map(|reg| reg.epoch)
    }

    /// Pin the side-channel host the operator allows for `worker_url`.
    pub(super) fn pin_trusted_host(&self, worker_url: String, host: Arc<str>) {
        self.trusted_hosts.insert(worker_url, host);
    }

    /// Whether `endpoint` may be handed to a destination as the fetch source
    /// for `source_url`. The engine's reported address is not verified, so it
    /// must be the host the operator pinned for that worker or, without a pin,
    /// the host of the worker URL the operator registered.
    pub(super) fn endpoint_is_trusted(&self, source_url: &str, endpoint: &PeerEndpoint) -> bool {
        let reported = super::peer_fetch::normalize_host(&endpoint.host);
        match self.trusted_hosts.get(source_url) {
            Some(pin) => *reported == **pin,
            None => super::peer_fetch::host_of(source_url).is_some_and(|host| host == reported),
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn capability(fingerprint: &str) -> TransferCapability {
        TransferCapability {
            can_export: true,
            can_import: true,
            connector: Arc::from("nixl"),
            connector_version: Arc::from("1"),
            layout_fingerprint: Arc::from(fingerprint),
        }
    }

    fn record(fingerprint: &str, port: u16) -> PeerLookupRecord {
        PeerLookupRecord {
            engine_id: "engine".into(),
            side_channel_host: "10.0.0.5".into(),
            side_channel_port: port,
            tp_size: 2,
            dcp_size: 1,
            pp_size: 1,
            connector: "nixl".into(),
            connector_version: "1".into(),
            layout_fingerprint: fingerprint.into(),
        }
    }

    #[test]
    fn store_keeps_the_epoch_for_identical_contents_and_removes_empty() {
        let dir = PeerDirectory::new();
        dir.store("http://a", Some(capability("x")), None, false);
        let epoch = dir.epoch_of("http://a").unwrap();
        dir.store("http://a", Some(capability("x")), None, false);
        assert_eq!(dir.epoch_of("http://a"), Some(epoch));
        dir.store("http://a", Some(capability("y")), None, false);
        assert!(dir.epoch_of("http://a").unwrap() > epoch);
        dir.store("http://a", None, None, false);
        assert!(dir.registration("http://a").is_none());
    }

    #[test]
    fn static_capability_wins_over_a_lookup_record_but_keeps_its_address() {
        let dir = PeerDirectory::new();
        dir.register_transfer_capability("http://a", capability("static"));
        dir.apply_lookup("http://a", Some(&record("engine", 5600)));
        let reg = dir.registration("http://a").unwrap();
        assert!(!reg.engine_owned);
        assert_eq!(
            &*reg.capability.as_ref().unwrap().layout_fingerprint,
            "static"
        );
        assert_eq!(dir.peer_endpoint("http://a").unwrap().port, 5600);

        // A lookup with no record clears the address, not the static capability.
        dir.apply_lookup("http://a", None);
        assert!(dir.registration("http://a").unwrap().capability.is_some());
        assert!(dir.peer_endpoint("http://a").is_none());
    }

    #[test]
    fn engine_owned_entries_are_dropped_but_static_ones_survive() {
        let dir = PeerDirectory::new();
        dir.apply_lookup("http://e", Some(&record("engine", 5600)));
        assert!(dir.registration("http://e").unwrap().engine_owned);
        dir.drop_engine_owned("http://e");
        assert!(dir.registration("http://e").is_none());

        dir.register_transfer_capability("http://s", capability("static"));
        dir.apply_lookup("http://s", Some(&record("engine", 5601)));
        dir.drop_engine_owned("http://s");
        let reg = dir.registration("http://s").unwrap();
        assert!(reg.capability.is_some() && reg.endpoint.is_none());
    }

    #[test]
    fn a_declaration_after_an_engine_record_replaces_it_for_good() {
        let dir = PeerDirectory::new();
        dir.apply_lookup("http://a", Some(&record("engine", 5600)));
        dir.register_transfer_capability("http://a", capability("static"));
        dir.apply_lookup("http://a", None);
        let reg = dir
            .registration("http://a")
            .expect("static capability kept");
        assert_eq!(
            reg.capability.as_ref().map(|c| &*c.layout_fingerprint),
            Some("static")
        );
        assert!(!reg.engine_owned);
        // No engine record any more, so no address either.
        assert!(reg.endpoint.is_none());
    }

    #[test]
    fn reapplying_an_identical_record_keeps_the_epoch() {
        let dir = PeerDirectory::new();
        dir.apply_lookup("http://a", Some(&record("fp", 5600)));
        let epoch = dir.epoch_of("http://a").expect("registered");
        dir.apply_lookup("http://a", Some(&record("fp", 5600)));
        assert_eq!(dir.epoch_of("http://a"), Some(epoch));
        dir.apply_lookup("http://a", Some(&record("fp", 5601)));
        assert!(dir.epoch_of("http://a").expect("registered") > epoch);
    }

    #[test]
    fn reported_host_must_match_the_pin_or_the_worker_url_host() {
        let dir = PeerDirectory::new();
        dir.apply_lookup("http://10.0.0.5:8000", Some(&record("fp", 5600)));
        let endpoint = dir.peer_endpoint("http://10.0.0.5:8000").unwrap();
        assert!(dir.endpoint_is_trusted("http://10.0.0.5:8000", &endpoint));
        assert!(!dir.endpoint_is_trusted("http://10.0.0.6:8000", &endpoint));
        dir.pin_trusted_host("http://10.0.0.6:8000".into(), Arc::from("10.0.0.5"));
        assert!(dir.endpoint_is_trusted("http://10.0.0.6:8000", &endpoint));
    }
}
