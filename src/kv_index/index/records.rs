//! Canonical cache-fact keys and namespace handling.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;

use crate::kv_hash::SequenceHash;
use crate::kv_index::{ResidencyOwner, StorageTier};

fn sparse_digest(hash: &SequenceHash) -> [u8; 66] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut key = [0; 66];
    key[0] = b'b';
    key[1] = b':';
    for (i, byte) in hash.iter().copied().enumerate() {
        key[2 + 2 * i] = HEX[usize::from(byte >> 4)];
        key[3 + 2 * i] = HEX[usize::from(byte & 0x0f)];
    }
    key
}

pub(super) fn sparse_key(namespace: Option<&str>, hash: &SequenceHash) -> Arc<str> {
    let key = sparse_digest(hash);
    let digest = std::str::from_utf8(&key).expect("digest uses ASCII hex");
    match namespace {
        None => Arc::from(digest),
        Some(name) => qualified_seq(name, digest),
    }
}

pub(crate) fn qualified_seq(namespace: &str, hash: &str) -> Arc<str> {
    Arc::from(format!("n:{}:{}:{}", namespace.len(), namespace, hash))
}

/// Strip only a well-formed internal namespace, preserving arbitrary wire text.
pub(super) fn unqualified_seq(hash: &str) -> &str {
    namespaced_seq(hash).map_or(hash, |(_, hash)| hash)
}

pub(super) fn namespaced_seq(hash: &str) -> Option<(&str, &str)> {
    let rest = hash.strip_prefix("n:")?;
    let (len, rest) = rest.split_once(':')?;
    let len = len.parse::<usize>().ok()?;
    Some((rest.get(..len)?, rest.get(len..)?.strip_prefix(':')?))
}

/// Canonical full byte hashes are stored without hex encoding. Other wire
/// identifiers remain exact strings, including uppercase and adapter keys.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FactKey {
    Digest(SequenceHash),
    Text(Arc<str>),
}

/// Value of a lowercase hex digit, 0xFF for anything else. Uppercase is not canonical.
const NIBBLE: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut i = 0;
    while i < 10 {
        t[b'0' as usize + i] = i as u8;
        i += 1;
    }
    let mut j = 0;
    while j < 6 {
        t[b'a' as usize + j] = 10 + j as u8;
        j += 1;
    }
    t
};

impl FactKey {
    pub(crate) fn from_seq(seq: &Arc<str>) -> Self {
        let bytes = seq.as_bytes();
        if bytes.len() == 66 && bytes.starts_with(b"b:") {
            let mut digest = [0; 32];
            // Invalid digits are 0xFF, so any bit in the high nibble flags the whole key.
            let mut invalid = 0u8;
            for (i, byte) in digest.iter_mut().enumerate() {
                let hi = NIBBLE[bytes[2 + i * 2] as usize];
                let lo = NIBBLE[bytes[3 + i * 2] as usize];
                invalid |= hi | lo;
                *byte = (hi << 4) | (lo & 0x0f);
            }
            if invalid & 0xF0 != 0 {
                return Self::Text(seq.clone());
            }
            return Self::Digest(digest);
        }
        Self::Text(seq.clone())
    }

    pub(super) fn to_seq(&self) -> Arc<str> {
        match self {
            Self::Digest(hash) => sparse_key(None, hash),
            Self::Text(seq) => seq.clone(),
        }
    }
}

/// One fact per owner and sequence hash, with residency tiers as bits.
pub(super) type ResidencyRecords =
    DashMap<ResidencyOwner, HashMap<FactKey, Residency, foldhash::fast::RandomState>>;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Residency {
    pub(super) tiers: TierBits,
}

impl Residency {
    pub(super) fn observe(&mut self, tier: StorageTier) {
        self.tiers.add(tier);
    }

    pub(super) fn withdraw(&mut self, tier: Option<StorageTier>) {
        let mask = tier.map_or(0b1111, TierBits::bit);
        self.tiers.0 &= !mask;
    }

    pub(super) fn contains(self, tier: StorageTier) -> bool {
        self.tiers.contains(tier)
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct TierBits(pub(crate) u8);

impl TierBits {
    pub(crate) fn bit(tier: StorageTier) -> u8 {
        1 << tier.rank()
    }

    pub(crate) fn contains(self, tier: StorageTier) -> bool {
        self.0 & Self::bit(tier) != 0
    }

    pub(super) fn add(&mut self, tier: StorageTier) {
        self.0 |= Self::bit(tier);
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[test]
    fn canonical_digest_decoding_preserves_every_byte_and_rejects_other_text() {
        let hash: SequenceHash = std::array::from_fn(|i| (i * 7) as u8);
        let canonical = sparse_key(None, &hash);
        assert_eq!(FactKey::from_seq(&canonical), FactKey::Digest(hash));
        for pos in 0..64 {
            for byte in 0..=127u8 {
                let mut text = canonical.as_bytes().to_vec();
                text[pos + 2] = byte;
                let seq: Arc<str> = Arc::from(std::str::from_utf8(&text).unwrap());
                let expected = if byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) {
                    let digit = if byte.is_ascii_digit() {
                        byte - b'0'
                    } else {
                        byte - b'a' + 10
                    };
                    let mut expected = hash;
                    let slot = &mut expected[pos / 2];
                    *slot = if pos % 2 == 0 {
                        (*slot & 0x0f) | (digit << 4)
                    } else {
                        (*slot & 0xf0) | digit
                    };
                    FactKey::Digest(expected)
                } else {
                    FactKey::Text(seq.clone())
                };
                assert_eq!(FactKey::from_seq(&seq), expected, "pos={pos} byte={byte}");
            }
        }
        for text in [
            "".to_owned(),
            "b:".into(),
            "i:42".into(),
            "b:00".into(),
            format!("b:{}", "AB".repeat(32)),
            format!("b:{}é", "0".repeat(62)),
            format!("b:{}中", "0".repeat(61)),
            format!("b:{}🦀", "0".repeat(60)),
            qualified_seq("适配:a", &canonical).to_string(),
        ] {
            let seq: Arc<str> = Arc::from(text);
            assert_eq!(FactKey::from_seq(&seq), FactKey::Text(seq.clone()));
        }
    }
}
