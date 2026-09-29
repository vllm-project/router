# Review notes for the first static Regular KV contribution

This document concerns the first native-CLI Qwen3 Dense slice. It does not
include the later Python render bridge, automatic Worker capabilities,
exact-history, Hybrid/MTP, PD, load-slack, or prepared-input features.

## A forwarding correction made during publication review

The original `624d840` HTTP entrypoint parsed requests into `serde_json::Value`
and dispatched them through `RequestBuilder::json`. This retained otherwise
unknown fields but did **not** retain the original bytes. Map order, Unicode
escapes, whitespace, duplicate keys, and number spelling could change.
The existing tests compared decoded JSON values, so they did not establish
byte-identical forwarding.

The reviewed patch keeps the bounded original `Bytes` next to the inspection
view. The KV HTTP Completion/Chat path forwards that same buffer on the first
attempt and retries. The non-KV path again uses the upstream typed `Json<T>`
extractor, including its duplicate-field and rejection behavior. Internal typed
calls without an HTTP buffer retain their prior serialization behavior.

New/updated tests check actual backend bytes, unsupported Chat extensions,
retry destinations and bytes, body limits, JSON media-type/syntax rejection,
and non-KV typed duplicate-field rejection. No hash, index, selector, or
subscription algorithm is changed by this correction.

## Validation identities must remain separate

The supplied author ledger records the following for **the original candidate**:

- Source: `624d8408b2610046db40e48f6d512b3f0741fc02`.
- Base: `bc16b190f8875a275287dd70ce5b4c9e54373dd6`.
- Tree: `b37ad4e478e4f75b22e817adeb4ca7ae481060fb`.
- Native SHA-256: `02788bb0c7e5355cd1f513ed8c8053cefb6931de9982e1a6d49c3fbcdfe3119c`.
- Rust: 900 passed, 0 failed, 4 ignored; one explicitly skipped external
  OpenAI health test. 29 suites, including rustdoc.
- fmt/check/strict Clippy and a native release build: recorded PASS.
- Qwen3-0.6B, vLLM 0.29.0, one GPU/two independent DP=1 Workers: recorded
  10/10 CUDA correctness cases.
- Qwen3-1.7B/4B: CPU metadata/tokenizer checks only.

The publication-review export contains the Git bundle, diff, ledger and evidence
index, not the indexed raw binaries and every execution log. The reviewer
verified the export checksums, Git objects, and ledger consistency; that is not
an independent re-run or a rehash of absent native artifacts.

**The forwarding patch is a new candidate.** Original 900/10 results must not be
relabeled as results for it. At patch preparation time this review environment
had no Cargo/Rust toolchain; its Rust tests/build were NOT RUN. A new validation
receipt must identify the checked commit/tree and actual commands before this
patch is represented as tested. CUDA reproduction is separate; see
[kv_aware.md](kv_aware.md).

## Focused checks for the reviewed forwarding change

Run in the actual contribution checkout, with the locked toolchain/dependencies:

```sh
cargo fmt --all --check
cargo check --locked --all-targets
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --lib server::kv_ingress_tests::
cargo test --locked --lib routers::http::router::tests::kv_
cargo test --locked --lib prompt_tokens::tests::
cargo test --locked --lib kv_events::
cargo test --locked --lib kv_index::
cargo test --locked --lib policies::kv_aware::
cargo test --locked --test api_endpoints_test
python3 scripts/kv_aware_cuda_validate.py self-check
```

Asset-backed ignored tests need the documented public tokenizer files. The full
suite also has upstream WASM guest and public tokenizer prerequisites; the
source-only bundle is not a vendored offline dependency distribution. Neither
missing assets nor zero matching tests should be reported as a model parity pass.

## Shared-interface decisions before merge

RFC #295 proposes a staged contribution within #294, not a competing full KV
architecture. This first implementation has a flat device-only index and a
borrowed-token compatibility method. It does not implement #294's two-hash,
owner/tier/provider interfaces or reconciled EXACT trust state.

Coordinate the token boundary with the request-scoped context proposed in #297;
its open status is not permission to copy or replace that work. Likewise, decide
with the M1/M2/M4 owners whether this narrowly scoped foundation can land first,
or whether the identity/read boundary must adapt before merge. Do not invent
maintainer acceptance or silently relax index safety to claim conformity.

Observed event gaps/disconnects purge the affected view. This is not replay or
an authoritative snapshot. The local generation is not an attested remote boot
identity. The operator must keep the explicitly configured Worker cohort
compatible. Neither matched block count nor routing affinity is a guarantee of
instantaneous residency, saved Prefill work, or throughput improvement.
