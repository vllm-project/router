//! Per-base capacity and pressure, refreshed off the request path.
//!
//! Kept apart from the KV event index on purpose: a deployment that publishes no event source still publishes these gauges, and admission must work there. The table owns its own scrape and never consults event subscriptions.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use futures_util::StreamExt;

use crate::kv_index::capacity::EngineSample;

/// How long a sample stays usable before a refresh is due.
const STALE_AFTER: Duration = Duration::from_secs(2);
/// Bound on one scrape, so an unresponsive engine cannot delay a decision.
const SCRAPE_TIMEOUT: Duration = Duration::from_millis(500);

/// What admission decided about one candidate rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityVerdict {
    /// Nothing is known about this rank; the caller must not invent a ceiling.
    Unknown,
    /// The rank has room for the request.
    Fits,
    /// The rank is full, queued on capacity, or would exceed its ceiling.
    Full,
}

#[derive(Default)]
struct Entry {
    engines: BTreeMap<u32, EngineSample>,
    checked: Option<Instant>,
}

/// Latest samples per base URL, refreshed on demand.
pub struct CapacityTable {
    max_kv_usage: f64,
    entries: Mutex<HashMap<String, Entry>>,
    /// Bases being scraped right now. Without it, concurrent admissions that both see a
    /// cold table both scrape, which multiplies the load on a busy pool at startup.
    in_flight: Mutex<std::collections::HashSet<String>>,
}

impl std::fmt::Debug for CapacityTable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CapacityTable")
            .field("max_kv_usage", &self.max_kv_usage)
            .finish()
    }
}

impl CapacityTable {
    pub fn new(max_kv_usage: f64) -> Self {
        Self {
            max_kv_usage: max_kv_usage.clamp(0.0, 1.0),
            entries: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(std::collections::HashSet::new()),
        }
    }

    pub fn max_kv_usage(&self) -> f64 {
        self.max_kv_usage
    }

    /// Scrape every base that is due, without holding the map across the network. A failed scrape keeps the previous sample only until it goes stale; after that the rank reads as `Unknown` rather than as healthy.
    pub async fn refresh(&self, client: &reqwest::Client, bases: &[String]) {
        let due: Vec<String> = {
            let entries = self.entries.lock().expect("capacity table");
            let mut in_flight = self.in_flight.lock().expect("capacity in flight");
            bases
                .iter()
                .filter(|base| {
                    let stale = entries
                        .get(*base)
                        .and_then(|entry| entry.checked)
                        .is_none_or(|checked| checked.elapsed() >= STALE_AFTER);
                    // Claim the base here so a concurrent admission does not scrape it too.
                    stale && in_flight.insert((*base).clone())
                })
                .cloned()
                .collect()
        };
        if due.is_empty() {
            return;
        }
        let client = client.clone();
        let scrapes: Vec<(String, Option<String>)> = futures_util::stream::iter(due)
            .map(|base| {
                let client = client.clone();
                async move {
                    let url = format!("{}/metrics", base.trim_end_matches('/'));
                    let body = tokio::time::timeout(SCRAPE_TIMEOUT, async {
                        let response = client.get(&url).send().await.ok()?;
                        response.text().await.ok()
                    })
                    .await
                    .ok()
                    .flatten();
                    (base, body)
                }
            })
            .buffer_unordered(8)
            .collect()
            .await;
        let mut entries = self.entries.lock().expect("capacity table");
        let mut in_flight = self.in_flight.lock().expect("capacity in flight");
        for (base, body) in scrapes {
            in_flight.remove(&base);
            match body {
                Some(text) => {
                    entries.insert(
                        base,
                        Entry {
                            engines: crate::kv_index::capacity::parse_metrics(&text),
                            checked: Some(Instant::now()),
                        },
                    );
                }
                None => {
                    if entries
                        .get(&base)
                        .and_then(|entry| entry.checked)
                        .is_some_and(|checked| checked.elapsed() >= STALE_AFTER)
                    {
                        entries.remove(&base);
                    }
                }
            }
        }
        // A base that left discovery is never a candidate again, so its row would live
        // forever. A stale row already reads as `Unknown`, so dropping it here only saves
        // memory and one later scrape.
        entries.retain(|_, entry| {
            entry
                .checked
                .is_some_and(|checked| checked.elapsed() < STALE_AFTER)
        });
    }

    /// Verdict for one candidate rank.
    pub fn verdict(
        &self,
        base: &str,
        rank: Option<u32>,
        add_tokens: Option<u64>,
        add_sequences: u64,
    ) -> CapacityVerdict {
        let entries = self.entries.lock().expect("capacity table");
        let Some(entry) = entries.get(base) else {
            return CapacityVerdict::Unknown;
        };
        if entry
            .checked
            .is_none_or(|checked| checked.elapsed() >= STALE_AFTER)
        {
            return CapacityVerdict::Unknown;
        }
        let Some(rank) = rank else {
            return self.unranked_verdict(entry, add_tokens, add_sequences);
        };
        let Some(sample) = entry.engines.get(&rank) else {
            return CapacityVerdict::Unknown;
        };
        if self.sample_full(sample, add_tokens, add_sequences) {
            CapacityVerdict::Full
        } else {
            CapacityVerdict::Fits
        }
    }

    /// Without a rank, a base is only usable when every rank it reports fits: picking one
    /// rank's sample would let a full sibling through.
    fn unranked_verdict(
        &self,
        entry: &Entry,
        add_tokens: Option<u64>,
        add_sequences: u64,
    ) -> CapacityVerdict {
        if entry.engines.is_empty() {
            return CapacityVerdict::Unknown;
        }
        if entry
            .engines
            .values()
            .any(|sample| self.sample_full(sample, add_tokens, add_sequences))
        {
            return CapacityVerdict::Full;
        }
        CapacityVerdict::Fits
    }

    fn sample_full(
        &self,
        sample: &EngineSample,
        add_tokens: Option<u64>,
        add_sequences: u64,
    ) -> bool {
        if sample.is_saturated(self.max_kv_usage) {
            return true;
        }
        if sample.fits_sequences(add_sequences) == Some(false) {
            return true;
        }
        add_tokens.is_some() && sample.fits_tokens(add_tokens.unwrap_or(0), 95) == Some(false)
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_index::capacity::DEFAULT_MAX_KV_USAGE;

    fn table_with(body: &str) -> CapacityTable {
        let table = CapacityTable::new(DEFAULT_MAX_KV_USAGE);
        {
            let mut entries = table.entries.lock().unwrap();
            entries.insert(
                "http://w0".to_string(),
                Entry {
                    engines: crate::kv_index::capacity::parse_metrics(body),
                    checked: Some(Instant::now()),
                },
            );
        }
        table
    }

    #[test]
    fn unknown_base_is_not_a_ceiling() {
        let table = CapacityTable::new(DEFAULT_MAX_KV_USAGE);
        assert_eq!(
            table.verdict("http://nope", None, Some(10), 1),
            CapacityVerdict::Unknown
        );
    }

    #[test]
    fn full_rank_is_refused_and_healthy_sibling_is_not() {
        let table = table_with(
            "vllm:kv_cache_usage_perc{engine=\"0\"} 0.20\nvllm:kv_cache_usage_perc{engine=\"1\"} 0.98\n",
        );
        assert_eq!(
            table.verdict("http://w0", Some(0), None, 1),
            CapacityVerdict::Fits
        );
        assert_eq!(
            table.verdict("http://w0", Some(1), None, 1),
            CapacityVerdict::Full
        );
        assert_eq!(
            table.verdict("http://w0", None, None, 1),
            CapacityVerdict::Full
        );
    }

    #[test]
    fn capacity_ceiling_rejects_a_request_that_does_not_fit() {
        let table = table_with(
            "vllm:cache_config_info{engine=\"0\",block_size=\"16\",num_gpu_blocks=\"100\"} 1\nvllm:kv_cache_usage_perc{engine=\"0\"} 0.10\n",
        );
        assert_eq!(
            table.verdict("http://w0", Some(0), Some(1400), 1),
            CapacityVerdict::Fits
        );
        assert_eq!(
            table.verdict("http://w0", Some(0), Some(1600), 1),
            CapacityVerdict::Full
        );
    }

    #[test]
    fn stale_sample_reads_as_unknown() {
        let table = CapacityTable::new(DEFAULT_MAX_KV_USAGE);
        {
            let mut entries = table.entries.lock().unwrap();
            entries.insert(
                "http://w0".to_string(),
                Entry {
                    engines: BTreeMap::new(),
                    checked: Some(Instant::now() - STALE_AFTER - Duration::from_secs(1)),
                },
            );
        }
        assert_eq!(
            table.verdict("http://w0", Some(0), None, 1),
            CapacityVerdict::Unknown
        );
    }
}
