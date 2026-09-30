# Pinned Qwen3 text profile fixtures

All prompts in `prompts.json` are newly authored synthetic examples with no user
or production data. They are distributed under this repository's Apache-2.0
license. Expected rendered strings cover the restricted, text-only branch of
the model's chat template and are independently checked by the Python oracle.
`python_oracle.json` contains actual outputs from the canonical, SHA-verified
tokenizer on Linux ARM64, using Transformers 4.57.6, tokenizers 0.22.2 and
Jinja2 3.1.6. It is an oracle fixture, not evidence that a candidate native
binary or vLLM worker has passed the test.

The model assets and chat template are from
[Qwen/Qwen3-0.6B at c1899de289a04d12100db370d81485cdf75e47ca](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca),
licensed Apache-2.0 by Qwen. Model assets are not vendored here. The small Rust
renderer adapts only the template branches described below; human review must
confirm attribution and license requirements before publication.

| Asset | SHA256 |
| --- | --- |
| tokenizer.json | aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4 |
| tokenizer_config.json | d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101 |
| vocab.json | ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 |
| merges.txt | 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 |

## Fixed verification corpus, not a runtime revision allowlist

This corpus fixes Qwen3-0.6B tokenizer/template bytes so results can be reproduced.
Production routing instead validates the configured local Qwen3 Dense metadata
and tokenizer semantics; it does not require this model size, name, or revision.
The standard template fingerprint enables the restricted Chat renderer. Unknown
templates disable Chat affinity while preserving exact Completion support.

The finite 0.6B hardware matrix uses immutable model assets (including the
explicitly recorded ModelScope snapshot described in `docs/kv_aware.md`), with
no adapter, cache salt, prompt truncation, or custom template. Worker aliases
must resolve to those actual assets. Local metadata does not prove worker
configuration equivalence; the live token checks must establish that separately.

- Completion: one string (honoring boolean `add_special_tokens`, default true)
  or one nonempty array of nonnegative token IDs. Batches fall back.
- Chat: optional first system message; alternating user and assistant text;
  final user message; generation prompt enabled. Only `role` and string
  `content` fields are accepted per message. Empty strings and whitespace are
  preserved. `chat_template_kwargs` may contain only boolean `enable_thinking`.
- Tools, reasoning fields/markers, multimodal content arrays, unknown message
  fields, final-assistant continuation, custom templates/kwargs and unsupported
  top-level fields fall back. The original JSON is still forwarded unchanged.

## Reproduction

Use an isolated Linux environment with the versions in
`scripts/kv_qwen3_oracle.requirements.txt`. This script needs no GPU and makes no
network requests. Use a complete
download of the exact revision, never an unpinned model identifier:

```sh
python scripts/kv_qwen3_oracle.py --tokenizer-directory /path/to/pinned/snapshot

# Recheck all five Chat and four Completion cases against the checked-in oracle:
python scripts/kv_qwen3_oracle.py --tokenizer-directory /path/to/pinned/snapshot \
  --check tests/fixtures/kv_qwen3/python_oracle.json
```

This prints the actual Python oracle token IDs and dependency versions. Save
that output as a test artifact, set `QWEN3_TOKENIZER_PATH` to the pinned JSON and
optionally `QWEN3_ORACLE_PATH` to the artifact (otherwise the checked-in oracle
is used), then run the ignored Rust test
`prompt_tokens::tests::pinned_python_oracle_token_ids` in the prescribed Linux
environment. Bind both output and native test binary to the candidate SHA.

`--template-only` can verify rendered strings from the pinned vocabulary,
merges, and template when tokenizer.json is unavailable. Its output is marked
`template-only`, has no token-ID goldens, and is **not** evidence of native
tokenizer parity or vLLM CPU/CUDA behavior.
