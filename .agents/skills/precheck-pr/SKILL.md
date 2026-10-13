---
name: precheck-pr
description: Contributor self-review for vllm-project/router. Use before implementing a contribution and before opening a PR to establish human intent, decide whether a PR is needed, and verify scoped evidence. Does not post or change files.
---

# Contributor pre-check

## Contributor intent gate

Run this gate before implementing a contribution and again before submission
if its intent or scope has changed. It applies to contributor self-review,
not independent review of somebody else's PR or automated review selection.

The human contributor must explicitly answer these four questions:

1. What outcome do you want to achieve?
2. What is the highest-leverage variable for that outcome?
3. How will this work move that variable?
4. Whose customer or user needs does this address, and what are those needs?

Reuse explicit human answers already supplied for this task or in its issue;
do not ask the contributor to repeat them. An agent must not invent answers,
infer approval from a diff, or fill gaps with its own rationale. Ask the human
for any missing or materially ambiguous answers and wait before implementing
or submitting the contribution. Read-only investigation may continue to make
the questions concrete.

Before submission, verify that the PR description answers all four required
questions in the [PR template](../../../.github/PULL_REQUEST_TEMPLATE.md),
or provides public-safe references to the human's explicit answers. Blank
sections, unchanged placeholders, and agent-written guesses do not satisfy
the gate. Missing or ambiguous answers block submission and readiness for
review; ask the human and wait. This is a contributor workflow requirement,
not a claim that GitHub or CI mechanically enforces the template.

Once the answers are clear, ask whether achieving the outcome actually needs
a PR. Prefer an existing capability, configuration change, documentation or
support answer, or no change when that meets the need. Documentation changes
may themselves need a PR. If no PR is needed, stop the submission workflow and
explain the smaller next step. If a PR is needed, connect its narrow scope and
validation plan to the human's stated outcome and variable.

For a bug fix, establish an actual failure: record expected versus observed
behavior, the affected revision and environment, and supporting evidence such
as a minimal reproducer, failing regression test, or observed failure trace.
Do not treat a plausible diagnosis or an unexecuted reproducer as verified
failure evidence. If the failure is not established, report the evidence gap
and continue investigation rather than claiming a confirmed fix.

Keep the answers and their source in the self-review report. Do not copy
private task conversations into a public issue or PR; share an intent summary
only within the audience and information-sharing scope the human authorized.

## Pre-submit self-review

1. Read [CONTRIBUTING.md](../../../CONTRIBUTING.md) and the repository's
   development instructions. Freeze the target base and proposed head revision.
2. Inspect the complete proposed diff. Keep changes focused on the agreed
   outcome and preserve unrelated work.
3. Run the applicable repository checks and the regression or measurement
   that demonstrates the intended outcome. Record commands, revisions,
   environment, expected and actual results, and checks that could not run.
4. Report intent completeness, PR necessity, bug evidence when applicable,
   changed scope, validation, and remaining blockers. Missing human intent
   blocks implementation/submission; missing evidence must not become a pass.

Stop after the contributor report. This skill does not modify files, open
PRs, post comments, or authorize publication. Maintainer review and existing
repository checks remain separate requirements.
