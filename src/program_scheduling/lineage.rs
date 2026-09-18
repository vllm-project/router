//! Program parent/root relationship tracking for multi-level agent trees.
//!
//! The index is a directed union-find: each Program points toward a declared
//! parent or root, and path compression keeps repeated root lookup bounded.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LineageKey {
    model_pool: String,
    program_id: String,
}

impl LineageKey {
    fn new(model_pool: &str, program_id: &str) -> Self {
        Self {
            model_pool: model_pool.to_string(),
            program_id: program_id.to_string(),
        }
    }
}

/// Model-scoped parent forest used to resolve the root Program lazily.
#[derive(Debug, Default)]
pub(crate) struct ProgramLineage {
    parents: HashMap<LineageKey, LineageKey>,
}

impl ProgramLineage {
    /// Record explicit parent/root facts and return the currently resolved root.
    pub(crate) fn observe(
        &mut self,
        model_pool: &str,
        program_id: &str,
        parent_program_id: Option<&str>,
        root_program_id: Option<&str>,
    ) -> String {
        let program = LineageKey::new(model_pool, program_id);
        self.parents
            .entry(program.clone())
            .or_insert_with(|| program.clone());
        if let Some(parent_id) = parent_program_id.filter(|parent| *parent != program_id) {
            let parent = LineageKey::new(model_pool, parent_id);
            self.parents
                .entry(parent.clone())
                .or_insert_with(|| parent.clone());
            if !self.would_cycle(&program, &parent) {
                self.parents.insert(program.clone(), parent);
            }
        }
        if let Some(root_id) = root_program_id.filter(|root| *root != program_id) {
            let root = LineageKey::new(model_pool, root_id);
            self.parents
                .entry(root.clone())
                .or_insert_with(|| root.clone());
            if !self.would_cycle(&program, &root) {
                self.parents.insert(program.clone(), root);
            }
        }
        self.root(model_pool, program_id)
    }

    /// Resolve one Program's root and compress the traversed parent path.
    pub(crate) fn root(&mut self, model_pool: &str, program_id: &str) -> String {
        let start = LineageKey::new(model_pool, program_id);
        self.parents
            .entry(start.clone())
            .or_insert_with(|| start.clone());
        let mut path = Vec::new();
        let mut current = start;
        while let Some(parent) = self.parents.get(&current).cloned() {
            if parent == current {
                break;
            }
            path.push(current);
            current = parent;
        }
        for node in path {
            self.parents.insert(node, current.clone());
        }
        current.program_id
    }

    /// Resolve the current root without mutating the forest.
    pub(crate) fn root_readonly(&self, model_pool: &str, program_id: &str) -> String {
        let start = LineageKey::new(model_pool, program_id);
        let mut current = start;
        for _ in 0..=self.parents.len() {
            let Some(parent) = self.parents.get(&current) else {
                break;
            };
            if parent == &current {
                break;
            }
            current = parent.clone();
        }
        current.program_id
    }

    fn would_cycle(&self, child: &LineageKey, proposed_parent: &LineageKey) -> bool {
        let mut current = proposed_parent.clone();
        for _ in 0..=self.parents.len() {
            if &current == child {
                return true;
            }
            let Some(parent) = self.parents.get(&current) else {
                return false;
            };
            if parent == &current {
                return false;
            }
            current = parent.clone();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_multi_level_parent_chain_and_explicit_root() {
        let mut lineage = ProgramLineage::default();
        assert_eq!(lineage.observe("m", "lead", None, None), "lead");
        assert_eq!(lineage.observe("m", "child", Some("lead"), None), "lead");
        assert_eq!(
            lineage.observe("m", "grandchild", Some("child"), None),
            "lead"
        );
        assert_eq!(lineage.observe("m", "external", None, Some("lead")), "lead");
        assert_eq!(lineage.root_readonly("m", "grandchild"), "lead");
        assert_eq!(
            lineage.observe("m", "late-child", Some("late-parent"), None),
            "late-parent"
        );
        assert_eq!(
            lineage.observe("m", "late-parent", Some("lead"), None),
            "lead"
        );
        assert_eq!(lineage.root_readonly("m", "late-child"), "lead");
    }
}
