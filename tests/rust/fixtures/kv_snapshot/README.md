# Captured publisher wire

The five `publisher-57789-*.json` files encode exact multipart byte arrays
captured from the proposed vLLM publisher snapshot extension. They originate
from the prior producer interop capture, not a real B32 Worker acceptance run.
The synthetic public B16 token prefix is 100 through 131.

- `seed`: live full-attention Store.
- `baseline`: complete cut with one resident reference per block.
- `two-references`: complete cut with two resident references per block.
- `remove`: live Remove for one reference per block.
- `removed`: complete cut with no resident references.

The Rust socket peer replays the captured cut/payloads, supplies its own
contiguous transport sequence, and adds disjoint positive marker prefixes after
Remove. This exercises production socket acquisition, reference normalization,
fenced installation and actual HTTP admission/release without installing vLLM.
It is not a stock Worker or hardware performance test. The publisher profile and
strict request opcode are experimental; upstream acceptance is not implied.
