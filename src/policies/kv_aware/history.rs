//! Advisory history tie-breaking for KV-aware selection.

/// Refine only the scorer's exact tie. Credit is from the same placement
/// estimate used for charging; missing evidence never becomes a measured zero.
pub(super) fn prefer(
    ordinary: usize,
    credit: impl Iterator<Item = Option<bool>>,
    candidates: impl Iterator<Item = (usize, usize)>,
) -> (usize, &'static str) {
    let mut unknown = false;
    for credited in credit {
        match credited {
            Some(true) => return (ordinary, "positive_credit"),
            None => unknown = true,
            Some(false) => {}
        }
    }
    if unknown {
        return (ordinary, "missing_evidence");
    }
    let mut longest = 0;
    let mut tied = Vec::new();
    let mut count = 0;
    for (index, length) in candidates {
        count += 1;
        if length > longest {
            longest = length;
            tied.clear();
        }
        if length > 0 && length == longest {
            tied.push(index);
        }
    }
    if count < 2 {
        return (ordinary, "no_tie");
    }
    if tied.contains(&ordinary) {
        return (ordinary, "eligible_ordinary");
    }
    match super::pick_tied(&tied) {
        Some(index) => (index, "override"),
        None => (ordinary, "no_overlap"),
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn completion_history_keeps_ordinary_with_equal_positive_overlap() {
        for (name, ordinary, candidates, expected) in [
            (
                "equal_first",
                0,
                vec![(0, 32), (1, 32)],
                (0, "eligible_ordinary"),
            ),
            (
                "equal_second",
                1,
                vec![(0, 32), (1, 32)],
                (1, "eligible_ordinary"),
            ),
            (
                "longest",
                1,
                vec![(0, 16), (1, 32), (2, 0)],
                (1, "eligible_ordinary"),
            ),
            (
                "strictly_better",
                0,
                vec![(0, 16), (1, 32)],
                (1, "override"),
            ),
        ] {
            let actual = prefer(
                ordinary,
                candidates.iter().map(|_| Some(false)),
                candidates.iter().copied(),
            );
            assert_eq!(actual, expected, "{name}");
        }
    }

    #[test]
    fn completion_history_preference_never_treats_unknown_as_zero() {
        for (name, credits, candidates, expected) in [
            (
                "zero",
                vec![Some(false), Some(false)],
                vec![(0, 0), (1, 32)],
                (1, "override"),
            ),
            (
                "unknown",
                vec![Some(false), None],
                vec![(0, 0), (1, 32)],
                (0, "missing_evidence"),
            ),
            (
                "positive_outside_tie",
                vec![Some(false), Some(false), Some(true)],
                vec![(0, 0), (1, 32)],
                (0, "positive_credit"),
            ),
            ("singleton", vec![Some(false)], vec![(1, 32)], (0, "no_tie")),
            (
                "no_overlap",
                vec![Some(false), Some(false)],
                vec![(0, 0), (1, 0)],
                (0, "no_overlap"),
            ),
        ] {
            assert_eq!(
                prefer(0, credits.into_iter(), candidates.into_iter()),
                expected,
                "{name}"
            );
        }
    }

    #[test]
    fn completion_history_requires_named_identity_and_one_reported_rank() {
        use crate::core::{BasicWorker, Worker, WorkerType};
        use crate::kv_index::source::test_support::{current_registration, full_binding};
        use crate::kv_index::{CacheKey, RoutableTarget, WorkerKvInfo};
        use crate::policies::{KvAwareOps, KvAwarePolicy};
        let kv = KvAwarePolicy::default();
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorker::new("http://w".into(), WorkerType::Regular));
        let provider = Arc::new(full_binding(
            RoutableTarget {
                instance_id: worker.base_url().into(),
                dp_rank: 0,
            },
            16,
        ));
        provider.begin_empty(0);
        for (case, model, ranks, current, expected) in [
            (
                "eligible",
                "model",
                vec![0],
                Some(current_registration()),
                true,
            ),
            (
                "url_identity",
                "http://w",
                vec![0],
                Some(current_registration()),
                false,
            ),
            (
                "multiple_ranks",
                "model",
                vec![0, 1],
                Some(current_registration()),
                false,
            ),
            (
                "unknown_ranks",
                "model",
                vec![],
                Some(current_registration()),
                false,
            ),
            ("no_owner", "model", vec![0], None, false),
        ] {
            kv.register_history_worker(
                &worker,
                Some(&WorkerKvInfo {
                    ranks,
                    cache_key: Some(CacheKey {
                        model: Arc::from(model),
                        block_size: 16,
                    }),
                    ..Default::default()
                }),
                Some(provider.clone()),
                current,
            );
            let captured = kv
                .capture_completion_history(std::slice::from_ref(&worker), model)
                .unwrap();
            assert_eq!(
                captured
                    .registration_id(worker.as_ref(), &kv.binding_for_worker(worker.as_ref()))
                    .is_some(),
                expected,
                "{case}"
            );
        }
    }
}
