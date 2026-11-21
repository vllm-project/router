//! Sparse multiple-choice knapsack: one target at most per waiting Program.
//! Costs and capacities are supplied by the caller, not estimated here.

use std::collections::BTreeMap;

// Exact DP on conservative quantized costs; optional frontier cap falls back to BFD.

#[derive(Clone, Debug)]
pub(super) struct OptionValue {
    pub target: usize,
    pub tokens: usize,
    pub value: (u128, u128),
    pub index: usize,
}

#[derive(Clone, Debug, Default)]
struct Selection {
    value: (u128, u128),
    indices: Vec<usize>,
}

impl Selection {
    fn better_than(&self, other: &Self) -> bool {
        self.value > other.value || (self.value == other.value && self.indices < other.indices)
    }
}

/// Return selected option indices; `Err` means the exact frontier exceeded its
/// work bound. Each capacity is (available tokens, available Program slots).
pub(super) fn solve(
    programs: &[Vec<OptionValue>],
    capacities: &[(usize, usize)],
    config: &crate::program_scheduling::ProgramSchedulerConfig,
) -> Vec<usize> {
    if config.resume_solver_strategy == crate::config::types::ResumeSolverStrategy::Greedy {
        let result = greedy_objective(programs, capacities);
        tracing::debug!(
            event = "resume_solver_decision",
            solver = "greedy_objective",
            selected = result.len()
        );
        return result;
    }
    let q = config.dp_capacity_quantum;
    let quantized: Vec<Vec<OptionValue>> = programs
        .iter()
        .map(|opts| {
            opts.iter()
                .map(|o| {
                    let mut o = o.clone();
                    o.tokens = o.tokens / q + usize::from(o.tokens % q != 0);
                    o
                })
                .collect()
        })
        .collect();
    let caps: Vec<_> = capacities
        .iter()
        .map(|&(tokens, slots)| (tokens / q, slots))
        .collect();
    match select(
        &quantized,
        &caps,
        config.dp_state_budget.unwrap_or(usize::MAX),
    ) {
        Ok(result) => {
            tracing::debug!(
                event = "resume_solver_decision",
                solver = "dp",
                quantum = q,
                selected = result.len()
            );
            result
        }
        Err(()) => {
            let result = bfd(programs, capacities);
            tracing::warn!(event="resume_solver_fallback", reason="state_budget", solver="bfd",
                state_budget=?config.dp_state_budget, candidates=programs.len(), selected=result.len());
            result
        }
    }
}

// Select globally across eligible session/target pairs by the caller's complete
// lexicographic objective. Capacity only controls feasibility, never ranking.
// Values are fixed during this scheduler snapshot; decreasing capacities mean
// a single sorted traversal is equivalent to repeatedly choosing the best pair.
fn greedy_objective(programs: &[Vec<OptionValue>], capacities: &[(usize, usize)]) -> Vec<usize> {
    let mut pairs: Vec<_> = programs
        .iter()
        .enumerate()
        .flat_map(|(p, options)| options.iter().map(move |o| (p, o)))
        .collect();
    pairs.sort_by_key(|(p, o)| (std::cmp::Reverse(o.value), *p, o.target, o.index));
    let mut remaining = capacities.to_vec();
    let mut assigned = vec![false; programs.len()];
    let mut selected = Vec::new();
    for (p, o) in pairs {
        if assigned[p] {
            continue;
        }
        let Some((tokens, slots)) = remaining.get_mut(o.target) else {
            continue;
        };
        if *slots == 0 || o.tokens > *tokens {
            continue;
        }
        *tokens -= o.tokens;
        *slots -= 1;
        assigned[p] = true;
        selected.push(o.index);
    }
    selected
}

// Best fit decreasing: descending token cost; choose the feasible eligible target
// with least residual tokens. Values and stable indices break ties. Never rebind
// beyond the options supplied by the scheduler; never select a Program twice.
fn bfd(programs: &[Vec<OptionValue>], capacities: &[(usize, usize)]) -> Vec<usize> {
    let mut order: Vec<_> = (0..programs.len()).collect();
    order.sort_by_key(|&p| {
        (
            std::cmp::Reverse(programs[p].iter().map(|o| o.tokens).max().unwrap_or(0)),
            p,
        )
    });
    let mut remaining = capacities.to_vec();
    let mut selected = Vec::new();
    for p in order {
        let chosen = programs[p]
            .iter()
            .filter(|o| {
                remaining
                    .get(o.target)
                    .is_some_and(|&(t, s)| s > 0 && o.tokens <= t)
            })
            .min_by_key(|o| {
                (
                    remaining[o.target].0 - o.tokens,
                    std::cmp::Reverse(o.value),
                    o.target,
                    o.index,
                )
            });
        if let Some(o) = chosen {
            remaining[o.target].0 -= o.tokens;
            remaining[o.target].1 -= 1;
            selected.push(o.index);
        }
    }
    selected.sort_unstable();
    selected
}

fn select(
    programs: &[Vec<OptionValue>],
    capacities: &[(usize, usize)],
    max_states: usize,
) -> Result<Vec<usize>, ()> {
    let mut frontier = BTreeMap::from([(
        vec![(0usize, 0usize); capacities.len()],
        Selection::default(),
    )]);
    for options in programs {
        let mut next = frontier.clone(); // Skipping this Program is always possible.
        for (used, selection) in &frontier {
            for option in options {
                let Some(&(token_limit, slot_limit)) = capacities.get(option.target) else {
                    continue;
                };
                let Some(tokens) = used[option.target].0.checked_add(option.tokens) else {
                    continue;
                };
                let count = used[option.target].1 + 1;
                if tokens > token_limit || count > slot_limit {
                    continue;
                }
                let mut key = used.clone();
                key[option.target] = (tokens, count);
                let mut candidate = selection.clone();
                candidate.value.0 += option.value.0;
                candidate.value.1 += option.value.1;
                candidate.indices.push(option.index);
                match next.get_mut(&key) {
                    Some(current) if candidate.better_than(current) => *current = candidate,
                    None => {
                        next.insert(key, candidate);
                    }
                    _ => {}
                }
                if next.len() > max_states {
                    return Err(());
                }
            }
        }
        frontier = next;
    }
    let mut best = Selection::default();
    for selection in frontier.into_values() {
        if selection.better_than(&best) {
            best = selection;
        }
    }
    Ok(best.indices)
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn option(index: usize, target: usize, tokens: usize, cached: u128) -> OptionValue {
        OptionValue {
            index,
            target,
            tokens,
            value: (cached, tokens as u128),
        }
    }

    #[test]
    fn agentx_sized_128_candidates_two_replicas_make_progress() {
        let programs: Vec<_> = (0..128)
            .map(|i| {
                (0..2)
                    .map(|target| {
                        option(
                            i * 2 + target,
                            target,
                            200_000 + ((i + target) * 7919) % 180_000,
                            ((i + target) * 1709) as u128,
                        )
                    })
                    .collect()
            })
            .collect();
        let caps = [(1_132_746, 128), (1_132_746, 128)];
        for budget in [Some(16), Some(50_000)] {
            let cfg = crate::program_scheduling::ProgramSchedulerConfig {
                dp_state_budget: budget,
                ..Default::default()
            };
            let selected = solve(&programs, &caps, &cfg);
            assert!(!selected.is_empty());
            for (target, capacity) in caps.iter().enumerate() {
                let tokens: usize = programs
                    .iter()
                    .flatten()
                    .filter(|o| o.target == target && selected.contains(&o.index))
                    .map(|o| o.tokens)
                    .sum();
                assert!(tokens <= capacity.0);
            }
        }
    }

    #[test]
    fn quantum_is_conservative_and_default_frontier_is_bounded() {
        let cfg = crate::program_scheduling::ProgramSchedulerConfig::default();
        assert_eq!(cfg.dp_capacity_quantum, 512);
        assert_eq!(cfg.dp_state_budget, Some(50_000));
        let programs = vec![vec![option(0, 0, 513, 4)], vec![option(1, 0, 512, 3)]];
        assert_eq!(solve(&programs, &[(1023, 2)], &cfg), vec![1]);
        assert_eq!(solve(&programs, &[(1024, 2)], &cfg), vec![0]);
    }

    #[test]
    fn budget_exhaustion_uses_complete_raw_bfd_not_partial_dp() {
        let cfg = crate::program_scheduling::ProgramSchedulerConfig {
            dp_state_budget: Some(1),
            dp_capacity_quantum: 1,
            ..Default::default()
        };
        let programs = vec![
            vec![option(0, 0, 8, 8)],
            vec![option(1, 0, 5, 5)],
            vec![option(2, 0, 5, 5)],
        ];
        assert_eq!(solve(&programs, &[(10, 3)], &cfg), vec![0]);
        let unlimited = crate::program_scheduling::ProgramSchedulerConfig {
            dp_state_budget: None,
            ..cfg.clone()
        };
        assert_eq!(solve(&programs, &[(10, 3)], &unlimited), vec![1, 2]);
        let greedy = crate::program_scheduling::ProgramSchedulerConfig {
            resume_solver_strategy: crate::config::types::ResumeSolverStrategy::Greedy,
            ..unlimited
        };
        assert_eq!(solve(&programs, &[(10, 3)], &greedy), vec![0]);
    }

    #[test]
    fn greedy_prefers_objective_over_largest_cost_and_budget_fallback_stays_bfd() {
        let programs = vec![
            vec![option(0, 0, 10, 1)],
            vec![option(1, 0, 4, 9)],
            vec![option(2, 0, 6, 8)],
        ];
        let config = crate::program_scheduling::ProgramSchedulerConfig {
            resume_solver_strategy: crate::config::types::ResumeSolverStrategy::Greedy,
            dp_state_budget: Some(1),
            ..Default::default()
        };
        assert_eq!(solve(&programs, &[(10, 2)], &config), vec![1, 2]);
        let config = crate::program_scheduling::ProgramSchedulerConfig {
            resume_solver_strategy: crate::config::types::ResumeSolverStrategy::Dp,
            dp_capacity_quantum: 1,
            ..config
        };
        assert_eq!(solve(&programs, &[(10, 2)], &config), vec![0]);
    }

    #[test]
    fn greedy_ranks_pairs_globally_and_uses_feasible_alternative_without_duplication() {
        let programs = vec![
            vec![option(0, 0, 6, 10), option(1, 1, 6, 8)],
            vec![option(2, 0, 5, 11), option(3, 1, 5, 1)],
        ];
        assert_eq!(greedy_objective(&programs, &[(6, 1), (6, 1)]), vec![2, 1]);
    }

    #[test]
    fn greedy_compares_full_vector_then_stable_identity_and_skips_infeasible_pairs() {
        let programs = vec![
            vec![option(0, 0, 9, 5)],
            vec![option(1, 0, 4, 5)],
            vec![option(2, 0, 4, 5)],
            vec![option(3, 0, 10, 100)],
        ];
        assert_eq!(greedy_objective(&programs, &[(9, 1)]), vec![0]);
        assert_eq!(greedy_objective(&programs, &[(8, 2)]), vec![1, 2]);
    }

    #[test]
    fn bfd_respects_bindings_slots_and_best_fit() {
        let programs = vec![
            vec![option(0, 0, 6, 1), option(1, 1, 6, 1)],
            vec![option(2, 0, 5, 2)],
            vec![option(3, 1, 1, 1)],
        ];
        assert_eq!(bfd(&programs, &[(10, 1), (6, 1)]), vec![1, 2]);
        assert!(bfd(&programs, &[(0, 2), (0, 2)]).is_empty());
    }

    #[test]
    fn more_than_old_state_limit_is_solved_without_budget() {
        let programs: Vec<_> = (0..15).map(|i| vec![option(i, 0, 1 << i, 1)]).collect();
        assert!(select(&programs, &[(32767, 15)], 16384).is_err());
        assert_eq!(
            select(&programs, &[(32767, 15)], usize::MAX).unwrap().len(),
            15
        );
    }

    #[test]
    fn greedy_and_fallback_never_exceed_raw_budgets() {
        for seed in 0..50 {
            let programs: Vec<_> = (0..12)
                .map(|p| {
                    (0..2)
                        .map(|t| option(p * 2 + t, t, (seed + p * 3 + t) % 11, (p % 4) as u128))
                        .collect()
                })
                .collect();
            let selected = bfd(&programs, &[(15, 3), (20, 4)]);
            let mut used = [(0, 0); 2];
            for opts in &programs {
                let chosen: Vec<_> = opts
                    .iter()
                    .filter(|o| selected.contains(&o.index))
                    .collect();
                assert!(chosen.len() <= 1);
                for o in chosen {
                    used[o.target].0 += o.tokens;
                    used[o.target].1 += 1;
                }
            }
            assert!(used[0].0 <= 15 && used[0].1 <= 3 && used[1].0 <= 20 && used[1].1 <= 4);
        }
    }

    #[test]
    fn selects_subset_instead_of_greedy_largest_cache() {
        let programs = vec![
            vec![option(0, 0, 8, 8)],
            vec![option(1, 0, 5, 5)],
            vec![option(2, 0, 5, 5)],
        ];
        assert_eq!(select(&programs, &[(10, 3)], usize::MAX), Ok(vec![1, 2]));
        assert_eq!(select(&programs, &[(10, 1)], usize::MAX), Ok(vec![0]));
        assert_eq!(select(&programs, &[(4, 3)], usize::MAX), Ok(vec![]));
        assert_eq!(select(&[], &[(10, 3)], usize::MAX), Ok(vec![]));
    }

    #[test]
    fn target_choice_does_not_duplicate_a_program() {
        let programs = vec![
            vec![option(0, 0, 5, 5), option(1, 1, 5, 4)],
            vec![option(2, 0, 5, 5)],
        ];
        assert_eq!(
            select(&programs, &[(5, 1), (5, 1)], usize::MAX),
            Ok(vec![1, 2])
        );
    }

    #[test]
    fn fixed_bindings_keep_replica_budgets_independent() {
        let programs = vec![
            vec![option(0, 0, 8, 8)],
            vec![option(1, 1, 9, 9)],
            vec![option(2, 0, 5, 5)],
            vec![option(3, 0, 5, 5)],
            vec![option(4, 1, 5, 5)],
        ];
        assert_eq!(
            select(&programs, &[(10, 2), (9, 2)], usize::MAX),
            Ok(vec![1, 2, 3])
        );
    }

    #[test]
    fn lexicographic_value_and_ties_are_deterministic() {
        let programs = vec![
            vec![option(0, 0, 5, 1)],
            vec![option(1, 0, 10, 0)],
            vec![option(2, 0, 5, 1)],
        ];
        assert_eq!(select(&programs, &[(10, 1)], usize::MAX), Ok(vec![0]));
        let programs = vec![vec![option(0, 0, 5, 1)], vec![option(1, 0, 6, 1)]];
        assert_eq!(select(&programs, &[(6, 1)], usize::MAX), Ok(vec![1]));
        assert_eq!(select(&programs, &[(6, 1)], 1), Err(()));
    }

    #[test]
    fn matches_exhaustive_small_multiple_target_problems() {
        fn brute(programs: &[Vec<OptionValue>], caps: &[(usize, usize)]) -> (u128, u128) {
            let Some((options, rest)) = programs.split_first() else {
                return (0, 0);
            };
            let mut best = brute(rest, caps);
            for o in options {
                if o.tokens <= caps[o.target].0 && caps[o.target].1 > 0 {
                    let mut remaining = caps.to_vec();
                    remaining[o.target].0 -= o.tokens;
                    remaining[o.target].1 -= 1;
                    let value = brute(rest, &remaining);
                    best = best.max((value.0 + o.value.0, value.1 + o.value.1));
                }
            }
            best
        }
        for seed in 0..20 {
            let programs: Vec<_> = (0..5)
                .map(|p| {
                    (0..2)
                        .map(|t| {
                            option(
                                p * 2 + t,
                                t,
                                (seed + p * 3 + t) % 7,
                                ((seed + p + t * 2) % 5) as u128,
                            )
                        })
                        .collect()
                })
                .collect();
            for capacity in 0..10 {
                let caps = [(capacity, 2), (capacity + 1, 3)];
                let selected = select(&programs, &caps, usize::MAX).unwrap();
                let value = programs
                    .iter()
                    .flatten()
                    .filter(|o| selected.contains(&o.index))
                    .fold((0, 0), |v, o| (v.0 + o.value.0, v.1 + o.value.1));
                assert_eq!(value, brute(&programs, &caps));
            }
        }
    }
}
