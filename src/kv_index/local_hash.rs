use sha2::{Digest, Sha256};

/// sha256 of each block's token-id bytes, seeded by `lora_name`.
/// The seed is re-applied per block (no parent chaining here) so identical
/// tokens under different adapters land on different nodes.
// ponytail: cache_salt/MM isolation deferred — needs raw msgpack capture of
// per-block extra_keys tuples (custom newtype or rmpv); add when cost model
// scores cross-salt collisions. Until then lora_name alone distinguishes adapters.
pub fn local_hashes(token_ids: &[u32], block_size: u32, lora_name: Option<&str>) -> Vec<[u8; 32]> {
    let bs = block_size as usize;
    if bs == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(token_ids.len().div_ceil(bs));
    for chunk in token_ids.chunks(bs) {
        let mut hasher = Sha256::new();
        if let Some(l) = lora_name {
            hasher.update(l.as_bytes());
        }
        for &tid in chunk {
            hasher.update(tid.to_le_bytes());
        }
        out.push(hasher.finalize().into());
    }
    out
}
