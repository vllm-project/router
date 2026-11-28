# Buildkite CI/CD Configuration

This directory contains the Buildkite pipeline configurations for the vLLM Router project.

Formatting, Clippy, Black, Ruff, and spelling are not Buildkite jobs. They run
from [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) locally and in the
GitHub Actions `pre-commit` workflow:

```bash
pip install pre-commit && pre-commit install
```

## Pipeline Files

`pipeline.yml` runs a lightweight CPU bootstrap, following
[vllm-omni's diff-aware bootstrap](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/common/scripts/upload_pipeline.py).
When every changed path is under `docs/`, ends in `.md`, or is
`mkdocs.yml`, it succeeds without uploading build or test jobs. This includes
documentation additions and deletions. Mixed code/documentation changes run CI.
For PRs (including `merge-test`), the bootstrap fetches the target branch and
compares its merge base with `HEAD`. Merge queues compare against Buildkite's
recorded queue base commit. Default-branch pushes always run full CI because
the bootstrap has no verified pre-push SHA to compare the complete push range.
It does not fall back to `HEAD^1`, which could miss earlier code changes in a
multi-commit push. Skipping post-merge pushes requires a verified push range
from the webhook before it can be enabled safely.
Renames include both the old and new paths. Empty or unresolved diffs run CI.
Scheduled builds and `NIGHTLY=1` always run CI.
PR labels `ready`, `merge-test`, and `nightly-test` do not override docs-only
skipping; they select test levels only when the full pipeline is uploaded.
GitHub Actions quality/security checks and release publishing
remain enabled. Unlike Omni, pytest skip-mark changes do not qualify.

Otherwise the bootstrap uploads `bootstrap-upload-steps.yml`, which builds the release
binary and wheel, pushes the NVIDIA CI image, then uploads L1, L2, and L3.
Those uploads depend on the build step, not on each other. Jobs in each uploaded file depend on that upload step. The L1 and L3 NVIDIA P/D jobs, and the L3 gRPC perf job, also wait for `image-build`. `NIGHTLY=1` on this test pipeline selects tests. `NIGHTLY=1`
on `release-pipeline.yml` (the `router-release` pipeline) publishes images.

| File | Level | When it is uploaded | What runs |
| --- | --- | --- | --- |
| `test-ready.yml` | L1 ready | non-default branch labeled `ready` | Rust and Python unit/integration tests, one NVIDIA P/D completion, and kv_aware e2e on 2× L4 (`ROCM_ONLY != 1` on the GPU jobs) |
| `test-merge.yml` | L2 merge | default branch (except `NIGHTLY=1`), its merge queue, or any other branch labeled `merge-test` | one-mode xGMI smoke and RDMA `SMOKE_ONLY` |
| `test-nightly.yml` | L3 nightly | default branch with `NIGHTLY=1`, or any other branch labeled `nightly-test` | Rust and Python unit/integration tests (skipped when the PR also has `ready`), full NVIDIA GSM8K (`ROCM_ONLY != 1`), gRPC prefix-KV perf on 2× H100 (`ROCM_ONLY != 1`), router_policy TraceLab c16 on 4× L4 (`ROCM_ONLY != 1`), L1 prefix-encoding perf on 1× L4 with Qwen3-0.6B (`ROCM_ONLY != 1`, 24 waves), full xGMI matrix, full RDMA, kv_aware e2e on 2× MI300X (scheduled builds only) |
| `release-pipeline.yml` | release | tag / release pipeline | wheels and images only |

A pull request labeled `ready` uploads `test-ready.yml`, including forks,
unless the bootstrap skips it as documentation-only. The same exception applies
to `merge-test` and merge-queue L2 uploads. Default-branch push L2 uploads
remain enabled.
`test-merge.yml` has no per-job `if`. In `test-ready.yml` and `test-nightly.yml`, a job whose gate is narrower than the upload keeps its own `if`.

GPU machine presets are in [`ci_mirror_hardwares.yml`](ci_mirror_hardwares.yml).
A step names one with `mirror_hardwares`. `scripts/upload_pipeline.py` expands
that name into `agents` and `plugins` before Buildkite upload. A preset that
injects `VLLM_CI_HF_TOKEN` also exports that value as `HF_TOKEN` in the job
shell. A build-level `HF_TOKEN` would override a pod env entry of the same name.

| Name | Machine |
| --- | --- |
| `l4_2` | EKS `l4-k8s`, same `l4x4` pool, pod requests 2× L4, same CI image |
| `l4_4` | EKS `l4-k8s`, 4× L4, image `public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:router-$BUILDKITE_COMMIT` (router_policy perf) |
| `l4_1` | Same L4 pool and image, 1× L4 |
| `h100_2` | `mithril-h100-pool`, 2× H100 SXM, us-west-2 pull-through cache of the router CI image, GPU limit 2 |
| `mi300_2` | `router_rocm_mi300_2`, same-host xGMI Docker devices |
| `mi300_rdma` | `router_rocm_mi300_2` coordinator, no Docker plugin |

CPU steps keep `medium_cpu_queue_premerge` inline. Their command lives in the Docker plugin, so they do not use a preset. Dependency sets live in `ci_apt_packages.conf`, `requirements/`, and `scripts/install_packages.sh`. `rust` and `nvidia_pd` install apt packages. `router-bin` downloads the release binary and checks it. `lm-eval` installs the ROCm accuracy harness; `rocm-kv-aware` installs the pinned pytest/requests pair without building the router. A sourced `python` call downloads the binary and installs `.[dev]` into `/tmp/venv`. `python-wheel` uses that apt set and venv for `python -m build`.

### Rust coverage artifacts

The Rust unit and integration jobs in `test-nightly.yml`
run through [`scripts/rust_coverage.sh`](scripts/rust_coverage.sh), using
[`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov/tree/v0.6.21)
0.6.21 and pinned `nightly-2026-04-15` with its `llvm-tools-preview` component. Each job
runs its existing `cpu,ready` test targets once with coverage instrumentation.
Instrumentation is limited to the host target so nested WASM component builds
do not inherit unsupported coverage flags.
This adds tool installation and instrumented compilation time to those jobs.
`test-ready.yml` runs ordinary Rust tests without coverage instrumentation.
The nightly Rust jobs keep their existing scheduling condition, so a PR
also labeled `ready` skips them and does not produce Rust coverage artifacts.

Each job uploads one machine-readable LCOV file from the mounted checkout:

| Artifact | Unit job | Integration job |
| --- | --- | --- |
| LCOV data | `target/coverage/rust-unit/lcov.info` | `target/coverage/rust-integration/lcov.info` |

Reports exclude `tests/` and `benches/` files as well as inline test modules
and test-only helpers marked with `#[cfg_attr(coverage_nightly, coverage(off))]`.
The nightly toolchain enables this unstable attribute; ordinary stable builds
do not enable it. Add the same marker to new test modules and test-only helpers.
The existing `check-mark` pre-commit hook checks each test-only module,
function and impl declaration under `src/` for this marker, including
file-based `mod tests;` declarations.
These are separate coverage
measurements for each test group; their percentages must not be added or
averaged to obtain total coverage. Python tests, doctests, ignored tests, and
nightly-only branch coverage are outside this measurement. No minimum
coverage threshold is enforced.
Each run cleans workspace coverage artifacts so reports cannot include stale
binaries from another toolchain or test group; dependency builds remain cached.

Tests retain a failing job status when they fail. Report generation is
attempted after test failures, but compilation failures or missing profiles
may prevent reports from being generated. Report generation failures also
fail the job.
The previous report is removed before tool installation. A new report is
written to a temporary file and renamed to `lcov.info` only after generation
succeeds, so failed runs cannot publish stale or partially written reports.
The completed report is set to mode `0644` before publication so the host
Buildkite agent can read it even when the container runs as root.

To reproduce with the same Rust image and dependency set as CI, run from the
repository root (requires Docker and network access):

```bash
docker run --rm -v "$PWD:/workdir" -w /workdir rust:1.95.0-bookworm \
  bash -euc 'bash .buildkite/scripts/install_packages.sh rust; bash .buildkite/scripts/rust_coverage.sh unit'
docker run --rm -v "$PWD:/workdir" -w /workdir rust:1.95.0-bookworm \
  bash -euc 'bash .buildkite/scripts/install_packages.sh rust; bash .buildkite/scripts/rust_coverage.sh integration'
```

On a Linux development host with Rust, rustup, and the `rust` apt dependency
set already installed, use `bash .buildkite/scripts/rust_coverage.sh unit`
or `bash .buildkite/scripts/rust_coverage.sh integration`. The script installs
the pinned nightly toolchain, precompiled coverage tool from GitHub releases,
and LLVM component, so it also
needs network access.

### `release-pipeline.yml`
Release pipeline for tagged versions, nightly Docker images, and versioned
Docker releases.

- **Git tags (`v*`):** build wheels, smoke-test, publish to PyPI
- **`NIGHTLY=1`:** build and push `vllm/vllm-router:nightly` (multi-arch)
- **Manual / API (no `NIGHTLY`):** fill **Provide Release version here**
  (`X.Y.Z`), unblock the two Docker gates, and publish
  `vllm/vllm-router:vX.Y.Z` plus `:latest` (and per-arch tags)

## P/D Disaggregation Test

The P/D (Prefill/Decode) disaggregation test validates the router's ability to coordinate separate prefill and decode vLLM instances.

### How It Works

The test launches:
1. **Prefill instance(s)**: vLLM servers configured for prefill operations
2. **Decode instance(s)**: vLLM servers configured for decode operations
3. **Router**: Coordinates requests between prefill and decode instances

See `scripts/llama3.1/` for example setup scripts showing the proper configuration.

### Test Script

Location: `tests/python/e2e/pd_disagg_vllm/run_accuracy_test.sh`

The test script:
- Launches prefill and decode vLLM instances in Docker containers
- Starts the router with `--pd-disaggregation` flag
- Runs accuracy validation tests
- Cleans up containers on exit

### CI Configuration

A `ready` label runs one completion from `test-ready.yml`
(`SKIP_LM_EVAL=true`, `NUM_SANITY_REQUESTS=1`) when `ROCM_ONLY != 1`.
Full GSM8K stays in `test-nightly.yml`. Both include forks.

**Key features:**
- Requires 4 GPUs (runs on `l4-k8s`)
- 60 minute timeout
- Automatic retry (up to 2 attempts) for flaky failures
- Manual retry option available

### ROCm MoRI coverage

The dedicated `router_rocm_mi300_2` queue covers two MoRI lanes:

- `run_moriio_accuracy_test.sh` uses two GPUs on one host with the xGMI backend.
- `run_moriio_rdma_accuracy_test.sh` uses one GPU on each of two hosts with the
  RDMA backend.

The hosts restore the digest-pinned ROCm image before each job
(https://github.com/vllm-project/router/issues/363). L2 xGMI is one
`read_mode` plus `LM_EVAL_LIMIT=20`, and runs for a default-branch push, the
default-branch merge queue, or any pull request labeled `merge-test`. L3
xGMI is the read+write matrix. It runs for `NIGHTLY=1`, any pull request
labeled `nightly-test`, or a manual/API build with `RUN_ROCM_XGMI=1`. These
are the L2/L3 triggers, not the old "every trusted PR" condition.

L2 RDMA (`test-merge.yml`) sets `SMOKE_ONLY=true` and `read_mode=true`. Both
L2 jobs run whenever that file is uploaded. L3 RDMA runs both directions for
scheduled `NIGHTLY=1`, a pull request labeled `nightly-test`, or a manual/API
build with `RUN_ROCM_RDMA=1`.

ROCm-only scheduled builds should set both `NIGHTLY=1` and `ROCM_ONLY=1`.
`ROCM_ONLY=1` skips the NVIDIA four-GPU step, the H100 gRPC perf job, and the
L4 router_policy perf job while retaining the CPU build that supplies the router
artifact to the ROCm jobs.

The L3 matrices use a digest-pinned ROCm vLLM image and run health, sanity,
and 500-example GSM8K accuracy checks. The RDMA job is coordinator-owned and
controls the second host through the strict `vllm-router-rocm-worker` SSH alias.

### Environment Variables

The test script supports these environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_DOCKER_IMAGE` | `vllm/vllm-openai:latest` | vLLM Docker image |
| `MODEL_NAMES` | `meta-llama/Llama-3.2-1B-Instruct` | Model to test |
| `GPU_MEMORY_UTILIZATION` | `0.6` | GPU memory utilization (0.0-1.0) |
| `KV_BUFFER_DEVICE` | `cuda` | KV buffer device (cuda/cpu) |
| `DECODER_KV_LAYOUT` | `HND` | KV layout (HND/NHD) |
| `NUM_PREFILL_INSTANCES` | `1` | Number of prefill instances |
| `NUM_DECODE_INSTANCES` | `1` | Number of decode instances |
| `PREFILLER_TP_SIZE` | `1` | Tensor parallel size for prefill |
| `DECODER_TP_SIZE` | `1` | Tensor parallel size for decode |
| `NUM_SANITY_REQUESTS` | `20` | Completions before GSM8K |
| `SKIP_LM_EVAL` | `false` | Skip GSM8K |

### Artifacts

On test completion (success or failure), artifacts are collected:

- `/tmp/router.log`: Router process logs

### Running Locally

To run the test locally:

```bash
cd tests/python/e2e/pd_disagg_vllm

# Run with defaults
bash ./run_accuracy_test.sh

# Run with custom configuration
export GPU_MEMORY_UTILIZATION=0.8
export PREFILLER_TP_SIZE=2
export DECODER_TP_SIZE=2
bash ./run_accuracy_test.sh
```

**Requirements:**
- 4+ GPUs
- Docker with GPU support
- vLLM router binary in PATH

### Debugging Failed Tests

1. **Check router logs**: Download `/tmp/router.log` from Buildkite artifacts
2. **Check container logs**: The test script outputs logs on failure
3. **Verify GPU availability**: Run `nvidia-smi` to check GPU status
4. **Retry**: Use manual retry if failure appears infrastructure-related
5. **Run locally**: Reproduce the issue with the same configuration

## Additional Pipeline Steps

### Build
The `build` group in `bootstrap-upload-steps.yml` creates release artifacts:
- Rust binary (`target/release/vllm-router`)
- Python wheels and source distribution

Each step has a premerge variant and a nightly variant. Premerge builds
skip thin LTO and raise `codegen-units` to 16 for faster CI turnaround.
Nightly builds (triggered by `NIGHTLY=1` or the `nightly-test` label)
keep the repository release profile so that perf baselines match the
published binary. Only one variant of each pair runs per build.

L1, L2, and L3 uploads depend on this group. L2 and L3 do not depend on the L1 upload.

### Tests
Comprehensive test suite:
- Rust unit tests
- Rust integration tests
- Python tests with coverage

### ROCm KV-aware e2e

`rocm-kv-aware-e2e` in `test-nightly.yml` reuses the L4 functional test on
two MI300X GPUs on one host. It supports `ROCM_ONLY=1` and does not require
the NVIDIA image build.
The pinned ROCm image supplies `vllm-rs`; the router binary comes from this
build. See the [test configuration](../tests/python/e2e/function/kv_aware/config.py)
for the prefix-reuse scenario and worker settings; this is not a P/D test.
This lane runs only on scheduled `NIGHTLY=1` builds, not PR-triggered L3
builds, to avoid adding long lease waits to the shared PR queue.

Router installation and Python dependency installation each have a ten-minute
bound; pytest has a separate 30-minute bound. On forced termination, container
teardown remains the final cleanup boundary for workers. Worker logs
are collected under `target/ci-logs/kv-aware-rocm`; pytest output is in the
Buildkite console. See [shared host leases](#shared-host-leases-and-timeouts)
for the overall job timeout.

### Benchmarks
Not part of the pipeline. Run them locally with `scripts/run_benchmarks.py`
(criterion) or the router overhead harness; see
`docs/benchmarks/router_overhead.md`.

### Docker Build
Builds Docker image for the router.

## Agent Queues

- `medium_cpu_queue_premerge`: CPU-only tasks (builds, lints, unit tests)
- `l4-k8s`: NVIDIA L4 GPU tests on EKS (4-GPU P/D disaggregation; 4-GPU router_policy c16)
- `mithril-h100-pool`: NVIDIA H100 SXM tests (2-GPU gRPC prefix-KV perf)
- `router_rocm_mi300_2`: MI300X MoRI tests and the single-host, two-GPU KV-aware functional lane (`spawn=1`)
- `default`: General purpose queue

### Shared host leases and timeouts

The `router_rocm_mi300_2` hosts are shared. Their `pre-command` hook invokes
`vllm-router-rocm-hold-cluster-lease` after the agent accepts the job, so lease
acquisition counts toward Buildkite's step timeout. The hook can wait up to
360 minutes; jobs have previously timed out here before their test command ran.

The MoRI and KV-aware lanes allow 450 minutes: 360 for the lease plus 90 for
setup and execution. MoRI commands have a separate 89-minute limit covering
setup and the harness, followed by up to one minute before forcibly terminating
the local process group. Container and remote-worker cleanup remains the
harness's responsibility. KV-aware commands retain their shorter per-phase
limits. Automatic retries are disabled; inspect failures before retrying
manually. Each manual retry starts a fresh, independently bounded attempt.

## Adding New Tests

To add a new test step:

1. Choose appropriate location in pipeline
2. Define test command and dependencies
3. Set `mirror_hardwares` to a name in `ci_mirror_hardwares.yml` (GPU) or an agent queue (CPU)
4. Add artifact collection
5. Consider retry logic for flaky tests
6. Update this documentation

Example:
```yaml
- label: ":test_tube: New Test"
  command: |
    # Test commands here
  agents:
    queue: "medium_cpu_queue_premerge"
  depends_on: "build"
  artifact_paths:
    - "test-results/**/*"
```

## Environment Variables

Buildkite provides these built-in variables:

- `BUILDKITE_COMMIT`: Git commit SHA
- `BUILDKITE_TAG`: Git tag (for releases)
- `BUILDKITE_BRANCH`: Git branch name
- `BUILDKITE_BUILD_NUMBER`: Build number

## Additional Resources

- [Buildkite Documentation](https://buildkite.com/docs)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Project README](../README.md)
- [Example P/D Setup Scripts](../scripts/llama3.1/)

## Docker Hub release credentials

Docker publishing and nightly cleanup use the Buildkite secret
`ROUTER_DOCKERHUB_TOKEN`, mapped to `RELEASE_DOCKERHUB_TOKEN` only on Docker jobs.
Agents must support step-level secrets (Buildkite agent 3.106.0 or newer).
Configure an organization access token for `vllm/vllm-router` and use `vllm`
as the Docker Hub username. Image Push grants image pull/push; Tag Admin
grants tag listing/deletion for nightly cleanup. Do not grant access to
other repositories or organization management.

Restrict the secret's access policy to the immutable `router-release` pipeline
ID, main or an approved version ref, and the CPU and ARM release queues. A job's `secrets` entry
controls injection; the access policy controls whether other jobs may
retrieve the secret. Protect release refs and pipeline settings as well.

The cleanup script uses Docker Hub's `/v2/auth/token` exchange and
namespace-scoped tag API. Shell tracing stays disabled to avoid exposing
credentials or bearer tokens. The legacy S3 `DOCKERHUB_TOKEN` variable is
not used by these Docker jobs. After validating the replacement, remove
obsolete S3 copies and revoke shared tokens only after all consumers have
migrated. Never commit token values to this repository.
