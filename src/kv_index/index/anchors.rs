//! Source-scoped positions for one sequence hash.

use std::collections::HashMap;

use super::{Anchor, Fast, RoutableTarget};
use std::sync::Arc;

/// A live block, or an ancestor required by a live block.
#[derive(Default)]
pub(super) struct PathRecord {
    pub(super) position: Option<Anchor>,
    pub(super) parent: Option<Arc<str>>,
    pub(super) children: usize,
}

/// Most hashes belong to one source; allocate a map only when they are shared.
#[derive(Default)]
pub(super) enum AnchorSet {
    #[default]
    Empty,
    One(Arc<RoutableTarget>, PathRecord),
    Many(HashMap<Arc<RoutableTarget>, PathRecord, Fast>),
}

impl AnchorSet {
    pub(super) fn get_or_insert(&mut self, target: &Arc<RoutableTarget>) -> &mut PathRecord {
        match self {
            Self::Empty => *self = Self::One(target.clone(), PathRecord::default()),
            Self::One(existing, _) if existing != target => {
                let Self::One(old, value) = std::mem::take(self) else {
                    unreachable!()
                };
                let mut entries = HashMap::default();
                entries.insert(old, value);
                *self = Self::Many(entries);
            }
            _ => {}
        }
        match self {
            Self::One(_, value) => value,
            Self::Many(entries) => entries.entry(target.clone()).or_default(),
            Self::Empty => unreachable!(),
        }
    }

    pub(super) fn get(&self, target: &RoutableTarget) -> Option<&PathRecord> {
        match self {
            Self::One(key, value) if key.as_ref() == target => Some(value),
            Self::Many(entries) => entries.get(target),
            _ => None,
        }
    }

    pub(super) fn get_mut(&mut self, target: &RoutableTarget) -> Option<&mut PathRecord> {
        match self {
            Self::One(key, value) if key.as_ref() == target => Some(value),
            Self::Many(entries) => entries.get_mut(target),
            _ => None,
        }
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&RoutableTarget, &PathRecord)> {
        let one = match self {
            Self::One(key, value) => Some((key, value)),
            _ => None,
        };
        let many = match self {
            Self::Many(entries) => Some(entries.iter()),
            _ => None,
        };
        one.into_iter()
            .chain(many.into_iter().flatten())
            .map(|(key, value)| (key.as_ref(), value))
    }

    pub(super) fn remove(&mut self, target: &RoutableTarget) {
        match self {
            Self::One(key, _) if key.as_ref() == target => *self = Self::Empty,
            Self::Many(entries) => {
                entries.remove(target);
                if entries.len() == 1 {
                    let (key, value) = std::mem::take(entries).into_iter().next().unwrap();
                    *self = Self::One(key, value);
                } else if entries.is_empty() {
                    *self = Self::Empty;
                }
            }
            _ => {}
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::kv_index::index::TreeNode;
    use parking_lot::RwLock;
    use std::sync::Arc;

    #[test]
    fn shared_anchor_returns_to_single_source_after_retirement() {
        let target = |name| {
            Arc::new(RoutableTarget {
                instance_id: Arc::from(name),
                dp_rank: 0,
            })
        };
        let first = target("first");
        let second = target("second");
        let node = Arc::new(RwLock::new(TreeNode::root()));
        let mut anchors = AnchorSet::default();
        anchors.get_or_insert(&first).position = Some(Anchor {
            node: node.clone(),
            offset: 1,
        });
        anchors.get_or_insert(&second).position = Some(Anchor {
            node: node.clone(),
            offset: 2,
        });
        anchors
            .get_mut(&second)
            .unwrap()
            .position
            .as_mut()
            .unwrap()
            .offset = 3;
        assert_eq!(anchors.iter().count(), 2);
        anchors.remove(&first);
        assert!(matches!(anchors, AnchorSet::One(..)));
        assert_eq!(
            anchors
                .get(&second)
                .unwrap()
                .position
                .as_ref()
                .unwrap()
                .offset,
            3
        );
        anchors.remove(&second);
        assert!(anchors.is_empty());
    }
}
