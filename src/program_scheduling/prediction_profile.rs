//! Versioned offline seeds and bounded online overlays for shadow prediction.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;

pub(crate) const PROFILE_SCHEMA_VERSION: &str = "1";

pub(crate) fn validate_profile_json(
    raw: &str,
    return_time_bounds: &[f64],
    output_length_bounds: &[usize],
    max_keys: usize,
) -> Result<(), String> {
    let profile: PredictionProfileFile =
        serde_json::from_str(raw).map_err(|error| error.to_string())?;
    let mut store = PredictionProfileStore::empty(
        return_time_bounds.to_vec(),
        output_length_bounds.to_vec(),
        max_keys,
        1,
    );
    store.load_profile(profile)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PredictionSource {
    #[default]
    None,
    Offline,
    Online,
    Blended,
}

impl PredictionSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Offline => "offline",
            Self::Online => "online",
            Self::Blended => "blended",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct ReturnTimeKey {
    pub workflow_profile_id: Option<String>,
    pub profile_version: Option<String>,
    pub model_pool: Option<String>,
    pub agent_role: Option<String>,
    pub spawn_reason: Option<String>,
    pub blocks_parent: Option<bool>,
    pub expected_resume: Option<bool>,
    pub step_bucket: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct OutputLengthKey {
    pub workflow_profile_id: Option<String>,
    pub profile_version: Option<String>,
    pub model_pool: Option<String>,
    #[serde(alias = "agent_role")]
    pub profile_role: Option<String>,
}

#[derive(Debug, Clone)]
struct Histogram {
    offline: Vec<f64>,
    online: Vec<u64>,
    offline_no_return: f64,
    online_no_return: u64,
    max_observed: f64,
    recent: VecDeque<f64>,
}

impl Histogram {
    fn new(bucket_count: usize) -> Self {
        Self {
            offline: vec![0.0; bucket_count],
            online: vec![0; bucket_count],
            offline_no_return: 0.0,
            online_no_return: 0,
            max_observed: 0.0,
            recent: VecDeque::new(),
        }
    }

    fn from_seed(
        counts: &[f64],
        no_return: f64,
        max_observed: f64,
        bucket_count: usize,
    ) -> Option<Self> {
        if counts.len() != bucket_count
            || counts
                .iter()
                .any(|count| !count.is_finite() || *count < 0.0)
            || !no_return.is_finite()
            || no_return < 0.0
            || !max_observed.is_finite()
            || max_observed < 0.0
        {
            return None;
        }
        Some(Self {
            offline: counts.to_vec(),
            online: vec![0; bucket_count],
            offline_no_return: no_return,
            online_no_return: 0,
            max_observed: max_observed.max(0.0),
            recent: VecDeque::new(),
        })
    }

    fn observe(&mut self, value: f64, bounds: &[f64], recent_limit: usize) {
        let index = bounds
            .iter()
            .position(|upper| value <= *upper)
            .unwrap_or(bounds.len());
        self.online[index] = self.online[index].saturating_add(1);
        self.max_observed = self.max_observed.max(value);
        self.recent.push_back(value);
        while self.recent.len() > recent_limit {
            self.recent.pop_front();
        }
    }

    fn observe_no_return(&mut self) {
        self.online_no_return = self.online_no_return.saturating_add(1);
    }

    fn effective_counts(&self, prior_weight: f64) -> Vec<f64> {
        self.offline
            .iter()
            .zip(&self.online)
            .map(|(offline, online)| offline * prior_weight + *online as f64)
            .collect()
    }

    fn effective_no_return(&self, prior_weight: f64) -> f64 {
        self.offline_no_return * prior_weight + self.online_no_return as f64
    }

    fn effective_support(&self, prior_weight: f64) -> f64 {
        self.effective_counts(prior_weight).iter().sum::<f64>()
            + self.effective_no_return(prior_weight)
    }

    fn offline_support(&self) -> f64 {
        self.offline.iter().sum::<f64>() + self.offline_no_return
    }

    fn online_support(&self) -> usize {
        self.online.iter().sum::<u64>() as usize + self.online_no_return as usize
    }

    fn source(&self) -> PredictionSource {
        match (self.offline_support() > 0.0, self.online_support() > 0) {
            (true, true) => PredictionSource::Blended,
            (true, false) => PredictionSource::Offline,
            (false, true) => PredictionSource::Online,
            (false, false) => PredictionSource::None,
        }
    }

    fn quantile(&self, bounds: &[f64], quantile: f64, prior_weight: f64) -> Option<f64> {
        let counts = self.effective_counts(prior_weight);
        let total = counts.iter().sum::<f64>();
        if total <= 0.0 {
            return None;
        }
        let target = total * quantile;
        let mut cumulative = 0.0;
        for (index, count) in counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return Some(
                    bounds
                        .get(index)
                        .copied()
                        .unwrap_or(self.max_observed.max(*bounds.last().unwrap_or(&1.0))),
                );
            }
        }
        Some(self.max_observed.max(*bounds.last().unwrap_or(&1.0)))
    }

    fn recent_stable(&self, quantile: f64, window: usize, tolerance: f64) -> bool {
        if self.offline_support() > 0.0 && self.recent.len() < window {
            return true;
        }
        if self.recent.len() < window || window < 2 {
            return false;
        }
        let start = self.recent.len() - window;
        let values = self.recent.iter().skip(start).copied().collect::<Vec<_>>();
        let midpoint = values.len() / 2;
        let first = sample_quantile(&values[..midpoint], quantile);
        let second = sample_quantile(&values[midpoint..], quantile);
        match (first, second) {
            (Some(first), Some(second)) => (second - first).abs() / first.max(1.0) <= tolerance,
            _ => false,
        }
    }
}

fn sample_quantile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted.get(index).copied()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ReturnTimeProfileEntry {
    pub key: ReturnTimeKey,
    pub bucket_counts: Vec<f64>,
    #[serde(default)]
    pub no_return_count: f64,
    #[serde(default)]
    pub max_observed_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OutputLengthProfileEntry {
    pub key: OutputLengthKey,
    pub bucket_counts: Vec<f64>,
    #[serde(default)]
    pub max_observed_tokens: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PredictionProfileFile {
    pub schema_version: String,
    #[serde(default)]
    pub generated_at: Option<String>,
    pub return_time_bucket_bounds_seconds: Vec<f64>,
    pub output_length_bucket_bounds_tokens: Vec<usize>,
    #[serde(default)]
    pub return_time_profiles: Vec<ReturnTimeProfileEntry>,
    #[serde(default)]
    pub output_length_profiles: Vec<OutputLengthProfileEntry>,
}

#[derive(Debug, Clone)]
pub(crate) struct ReturnPrediction {
    pub probability: f64,
    pub suggested_ttl_seconds: f64,
    pub support: usize,
    pub online_support: usize,
    pub backoff_level: usize,
    pub source: PredictionSource,
}

#[derive(Debug, Clone)]
pub(crate) struct OutputPrediction {
    pub tokens: usize,
    pub p50: usize,
    pub p95: usize,
    pub p99: usize,
    pub support: usize,
    pub online_support: usize,
    pub backoff_level: usize,
    pub source: PredictionSource,
    pub mature: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct PredictionProfileStore {
    return_time_bounds: Vec<f64>,
    output_length_bounds: Vec<f64>,
    return_time: HashMap<ReturnTimeKey, Histogram>,
    output_length: HashMap<OutputLengthKey, Histogram>,
    max_keys: usize,
    recent_limit: usize,
    revision: u64,
    snapshotted_revision: u64,
}

impl PredictionProfileStore {
    pub(crate) fn empty(
        return_time_bounds: Vec<f64>,
        output_length_bounds: Vec<usize>,
        max_keys: usize,
        recent_limit: usize,
    ) -> Self {
        Self {
            return_time_bounds,
            output_length_bounds: output_length_bounds
                .into_iter()
                .map(|value| value as f64)
                .collect(),
            return_time: HashMap::new(),
            output_length: HashMap::new(),
            max_keys,
            recent_limit,
            revision: 0,
            snapshotted_revision: 0,
        }
    }

    pub(crate) fn load_path(&mut self, path: &Path) -> Result<(), String> {
        let raw = fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let profile: PredictionProfileFile = serde_json::from_str(&raw)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        self.load_profile(profile)
    }

    fn load_profile(&mut self, profile: PredictionProfileFile) -> Result<(), String> {
        if profile.schema_version != PROFILE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported prediction profile schema '{}'",
                profile.schema_version
            ));
        }
        if profile.return_time_bucket_bounds_seconds != self.return_time_bounds
            || profile
                .output_length_bucket_bounds_tokens
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>()
                != self.output_length_bounds
        {
            return Err("prediction profile bucket bounds do not match Router config".to_string());
        }
        for entry in profile.return_time_profiles {
            if self.return_time.len() >= self.max_keys {
                break;
            }
            let histogram = Histogram::from_seed(
                &entry.bucket_counts,
                entry.no_return_count,
                entry.max_observed_seconds,
                self.return_time_bounds.len() + 1,
            )
            .ok_or_else(|| "invalid return-time profile entry".to_string())?;
            self.return_time.insert(entry.key, histogram);
        }
        for entry in profile.output_length_profiles {
            if self.output_length.len() >= self.max_keys {
                break;
            }
            let histogram = Histogram::from_seed(
                &entry.bucket_counts,
                0.0,
                entry.max_observed_tokens,
                self.output_length_bounds.len() + 1,
            )
            .ok_or_else(|| "invalid output-length profile entry".to_string())?;
            self.output_length.insert(entry.key, histogram);
        }
        Ok(())
    }

    pub(crate) fn observe_return(&mut self, keys: &[ReturnTimeKey], seconds: f64) {
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        for key in keys {
            if !self.return_time.contains_key(key) && self.return_time.len() >= self.max_keys {
                continue;
            }
            self.return_time
                .entry(key.clone())
                .or_insert_with(|| Histogram::new(self.return_time_bounds.len() + 1))
                .observe(seconds, &self.return_time_bounds, self.recent_limit);
        }
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn observe_no_return(&mut self, keys: &[ReturnTimeKey]) {
        for key in keys {
            if !self.return_time.contains_key(key) && self.return_time.len() >= self.max_keys {
                continue;
            }
            self.return_time
                .entry(key.clone())
                .or_insert_with(|| Histogram::new(self.return_time_bounds.len() + 1))
                .observe_no_return();
        }
        self.revision = self.revision.wrapping_add(1);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_return(
        &self,
        keys: &[ReturnTimeKey],
        elapsed: f64,
        horizon: f64,
        min_support: usize,
        prior_weight: f64,
        cache_miss_cost: f64,
        max_ttl: f64,
    ) -> Option<ReturnPrediction> {
        for (backoff_level, key) in keys.iter().enumerate() {
            let Some(histogram) = self.return_time.get(key) else {
                continue;
            };
            let support = histogram.effective_support(prior_weight).round() as usize;
            if support < min_support {
                continue;
            }
            let counts = histogram.effective_counts(prior_weight);
            let no_return = histogram.effective_no_return(prior_weight);
            let mut survivors = no_return;
            let mut future_returns = 0.0;
            for (index, count) in counts.iter().enumerate() {
                let upper = self
                    .return_time_bounds
                    .get(index)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                if upper > elapsed {
                    survivors += count;
                }
                if upper > elapsed && upper <= elapsed + horizon {
                    future_returns += count;
                }
            }
            let probability = if survivors > 0.0 {
                future_returns / survivors
            } else {
                0.0
            };
            let suggested_ttl_seconds = best_ttl(
                &counts,
                no_return,
                &self.return_time_bounds,
                cache_miss_cost,
                max_ttl,
            );
            return Some(ReturnPrediction {
                probability,
                suggested_ttl_seconds,
                support,
                online_support: histogram.online_support(),
                backoff_level,
                source: histogram.source(),
            });
        }
        None
    }

    pub(crate) fn observe_output(&mut self, keys: &[OutputLengthKey], tokens: usize) {
        for key in keys {
            if !self.output_length.contains_key(key) && self.output_length.len() >= self.max_keys {
                continue;
            }
            self.output_length
                .entry(key.clone())
                .or_insert_with(|| Histogram::new(self.output_length_bounds.len() + 1))
                .observe(tokens as f64, &self.output_length_bounds, self.recent_limit);
        }
        self.revision = self.revision.wrapping_add(1);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_output(
        &self,
        keys: &[OutputLengthKey],
        quantile: f64,
        min_support: usize,
        tail_samples: usize,
        prior_weight: f64,
        stability_window: usize,
        stability_tolerance: f64,
    ) -> Option<OutputPrediction> {
        let tail_support =
            (tail_samples as f64 / (1.0 - quantile).max(f64::EPSILON)).ceil() as usize;
        let required_support = min_support.max(tail_support);
        for (backoff_level, key) in keys.iter().enumerate() {
            let Some(histogram) = self.output_length.get(key) else {
                continue;
            };
            let support = histogram.effective_support(prior_weight).round() as usize;
            if support < required_support {
                continue;
            }
            let p50 = histogram
                .quantile(&self.output_length_bounds, 0.50, prior_weight)?
                .ceil() as usize;
            let p95 = histogram
                .quantile(&self.output_length_bounds, 0.95, prior_weight)?
                .ceil() as usize;
            let p99 = histogram
                .quantile(&self.output_length_bounds, 0.99, prior_weight)?
                .ceil() as usize;
            let tokens = histogram
                .quantile(&self.output_length_bounds, quantile, prior_weight)?
                .ceil() as usize;
            let mature = support >= required_support
                && histogram.recent_stable(quantile, stability_window, stability_tolerance);
            return Some(OutputPrediction {
                tokens,
                p50,
                p95,
                p99,
                support,
                online_support: histogram.online_support(),
                backoff_level,
                source: histogram.source(),
                mature,
            });
        }
        None
    }

    pub(crate) fn snapshot(&self) -> Option<(u64, PredictionProfileFile)> {
        if self.revision == self.snapshotted_revision {
            return None;
        }
        Some((
            self.revision,
            PredictionProfileFile {
                schema_version: PROFILE_SCHEMA_VERSION.to_string(),
                generated_at: None,
                return_time_bucket_bounds_seconds: self.return_time_bounds.clone(),
                output_length_bucket_bounds_tokens: self
                    .output_length_bounds
                    .iter()
                    .map(|value| *value as usize)
                    .collect(),
                return_time_profiles: self
                    .return_time
                    .iter()
                    .filter(|(_, histogram)| histogram.online_support() > 0)
                    .map(|(key, histogram)| ReturnTimeProfileEntry {
                        key: key.clone(),
                        bucket_counts: histogram.online.iter().map(|value| *value as f64).collect(),
                        no_return_count: histogram.online_no_return as f64,
                        max_observed_seconds: histogram.max_observed,
                    })
                    .collect(),
                output_length_profiles: self
                    .output_length
                    .iter()
                    .filter(|(_, histogram)| histogram.online_support() > 0)
                    .map(|(key, histogram)| OutputLengthProfileEntry {
                        key: key.clone(),
                        bucket_counts: histogram.online.iter().map(|value| *value as f64).collect(),
                        max_observed_tokens: histogram.max_observed,
                    })
                    .collect(),
            },
        ))
    }

    pub(crate) fn mark_snapshotted(&mut self, revision: u64) {
        self.snapshotted_revision = self.snapshotted_revision.max(revision);
    }

    pub(crate) fn return_key_count(&self) -> usize {
        self.return_time.len()
    }

    pub(crate) fn output_key_count(&self) -> usize {
        self.output_length.len()
    }
}

fn best_ttl(
    counts: &[f64],
    no_return: f64,
    bounds: &[f64],
    cache_miss_cost: f64,
    max_ttl: f64,
) -> f64 {
    let total = counts.iter().sum::<f64>() + no_return;
    if total <= 0.0 || cache_miss_cost <= 0.0 || max_ttl <= 0.0 {
        return 0.0;
    }
    let mut best = (0.0, 0.0);
    for candidate in std::iter::once(0.0).chain(
        bounds
            .iter()
            .copied()
            .filter(|bound| *bound <= max_ttl)
            .chain(std::iter::once(max_ttl)),
    ) {
        let mut hit_probability = 0.0;
        let mut retained_cost = no_return / total * candidate;
        for (index, count) in counts.iter().enumerate() {
            let upper = bounds.get(index).copied().unwrap_or(max_ttl.max(candidate));
            let probability = count / total;
            if upper <= candidate {
                hit_probability += probability;
            }
            retained_cost += probability * upper.min(candidate);
        }
        let utility = cache_miss_cost * hit_probability - retained_cost;
        if utility >= best.1 {
            best = (candidate, utility);
        }
    }
    best.0
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    fn output_key() -> OutputLengthKey {
        OutputLengthKey {
            workflow_profile_id: Some("workflow".into()),
            profile_version: Some("1".into()),
            model_pool: Some("model".into()),
            profile_role: Some("engineer".into()),
        }
    }

    #[test]
    fn legacy_output_profile_agent_role_deserializes_as_profile_role() {
        let key: OutputLengthKey = serde_json::from_value(serde_json::json!({
            "workflow_profile_id": "workflow",
            "profile_version": "1",
            "model_pool": "model",
            "agent_role": "engineer",
            "input_bucket": 1024
        }))
        .unwrap();

        assert_eq!(key.profile_role.as_deref(), Some("engineer"));
    }

    #[test]
    fn online_output_profile_becomes_mature_and_predicts_quantile() {
        let mut store = PredictionProfileStore::empty(vec![1.0, 2.0], vec![64, 128, 256], 16, 20);
        let key = output_key();
        for value in 1..=20 {
            store.observe_output(std::slice::from_ref(&key), value * 5);
        }
        let prediction = store
            .predict_output(&[key], 0.5, 10, 1, 1.0, 10, 10.0)
            .unwrap();
        assert_eq!(prediction.tokens, 64);
        assert!(prediction.mature);
        assert_eq!(prediction.source, PredictionSource::Online);
    }

    #[test]
    fn return_probability_conditions_on_elapsed_time() {
        let mut store = PredictionProfileStore::empty(vec![1.0, 2.0, 5.0, 10.0], vec![64], 16, 20);
        let key = ReturnTimeKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: Some("model".into()),
            agent_role: None,
            spawn_reason: None,
            blocks_parent: None,
            expected_resume: None,
            step_bucket: None,
        };
        for seconds in [1.0, 2.0, 5.0, 5.0, 10.0] {
            store.observe_return(std::slice::from_ref(&key), seconds);
        }
        store.observe_no_return(std::slice::from_ref(&key));
        let prediction = store
            .predict_return(&[key], 2.0, 3.0, 1, 1.0, 10.0, 10.0)
            .unwrap();
        assert!((prediction.probability - 0.5).abs() < 1e-9);
    }

    #[test]
    fn offline_seed_and_online_overlay_are_blended_without_mutating_seed() {
        let key = output_key();
        let profile = PredictionProfileFile {
            schema_version: PROFILE_SCHEMA_VERSION.to_string(),
            generated_at: None,
            return_time_bucket_bounds_seconds: vec![1.0, 2.0],
            output_length_bucket_bounds_tokens: vec![64, 128, 256],
            return_time_profiles: Vec::new(),
            output_length_profiles: vec![OutputLengthProfileEntry {
                key: key.clone(),
                bucket_counts: vec![10.0, 0.0, 0.0, 0.0],
                max_observed_tokens: 64.0,
            }],
        };
        let mut store = PredictionProfileStore::empty(vec![1.0, 2.0], vec![64, 128, 256], 16, 4);
        store.load_profile(profile).unwrap();
        store.observe_output(std::slice::from_ref(&key), 200);
        let prediction = store
            .predict_output(&[key], 0.5, 1, 1, 1.0, 4, 10.0)
            .unwrap();
        assert_eq!(prediction.source, PredictionSource::Blended);
        assert_eq!(prediction.online_support, 1);
        assert_eq!(prediction.support, 11);
        assert_eq!(prediction.p50, 64);
    }

    #[test]
    fn snapshot_contains_only_online_overlay_and_round_trips() {
        let key = output_key();
        let mut store = PredictionProfileStore::empty(vec![1.0, 2.0], vec![64, 128, 256], 16, 4);
        store.observe_output(std::slice::from_ref(&key), 80);
        store.observe_output(std::slice::from_ref(&key), 80);
        let (_, snapshot) = store.snapshot().unwrap();
        let encoded = serde_json::to_string(&snapshot).unwrap();
        let decoded: PredictionProfileFile = serde_json::from_str(&encoded).unwrap();
        let mut restored = PredictionProfileStore::empty(vec![1.0, 2.0], vec![64, 128, 256], 16, 4);
        restored.load_profile(decoded).unwrap();
        let prediction = restored
            .predict_output(&[key], 0.5, 1, 1, 1.0, 4, 10.0)
            .unwrap();
        assert_eq!(prediction.source, PredictionSource::Offline);
        assert_eq!(prediction.tokens, 128);
    }

    #[test]
    fn mismatched_bucket_contract_is_rejected() {
        let profile = PredictionProfileFile {
            schema_version: PROFILE_SCHEMA_VERSION.to_string(),
            generated_at: None,
            return_time_bucket_bounds_seconds: vec![1.0],
            output_length_bucket_bounds_tokens: vec![64],
            return_time_profiles: Vec::new(),
            output_length_profiles: Vec::new(),
        };
        let mut store = PredictionProfileStore::empty(vec![1.0, 2.0], vec![64], 16, 4);
        assert!(store.load_profile(profile).is_err());
    }

    #[test]
    fn output_lookup_reports_backoff_and_tail_quantiles() {
        let exact = output_key();
        let global = OutputLengthKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: None,
            profile_role: None,
        };
        let mut store = PredictionProfileStore::empty(vec![1.0], vec![64, 128, 256, 512], 16, 20);
        for _ in 0..18 {
            store.observe_output(std::slice::from_ref(&global), 60);
        }
        store.observe_output(std::slice::from_ref(&global), 200);
        store.observe_output(std::slice::from_ref(&global), 400);
        let prediction = store
            .predict_output(&[exact, global], 0.95, 10, 1, 1.0, 20, 10.0)
            .unwrap();
        assert_eq!(prediction.backoff_level, 1);
        assert_eq!(prediction.p50, 64);
        assert_eq!(prediction.p95, 256);
        assert_eq!(prediction.p99, 512);
    }

    #[test]
    fn sparse_exact_output_profile_backs_off_to_supported_role_profile() {
        let exact = output_key();
        let mut role = exact.clone();
        role.workflow_profile_id = None;
        role.profile_version = None;
        let mut store = PredictionProfileStore::empty(vec![1.0], vec![64, 128, 256], 16, 20);
        store.observe_output(std::slice::from_ref(&exact), 200);
        for _ in 0..20 {
            store.observe_output(std::slice::from_ref(&role), 60);
        }

        let prediction = store
            .predict_output(&[exact, role], 0.95, 10, 1, 1.0, 20, 10.0)
            .unwrap();

        assert_eq!(prediction.backoff_level, 1);
        assert_eq!(prediction.tokens, 64);
    }

    #[test]
    fn snapshot_acknowledgement_preserves_newer_observations() {
        let key = output_key();
        let mut store = PredictionProfileStore::empty(vec![1.0], vec![64, 128], 16, 4);
        store.observe_output(std::slice::from_ref(&key), 60);
        let (revision, _) = store.snapshot().unwrap();
        store.observe_output(std::slice::from_ref(&key), 70);
        store.mark_snapshotted(revision);
        assert!(store.snapshot().is_some());
    }

    #[test]
    fn profile_key_growth_is_bounded() {
        let mut store = PredictionProfileStore::empty(vec![1.0], vec![64], 1, 4);
        let first = output_key();
        let mut second = output_key();
        second.profile_role = Some("reviewer".into());
        store.observe_output(&[first], 10);
        store.observe_output(&[second], 20);
        assert_eq!(store.output_key_count(), 1);
    }
}
