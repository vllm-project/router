# Buildkite CI/CD Configuration

This directory contains the Buildkite pipeline configurations for the vLLM Router project.

Formatting, Clippy, Black, Ruff, and spelling are not Buildkite jobs. They run
from [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) locally and in the
GitHub Actions `pre-commit` workflow:

```bash
pip install pre-commit && pre-commit install
```

## Pipeline Files

`pipeline.yml` builds the release binary and wheel, pushes the NVIDIA CI image, then uploads L1, L2, and L3.
Those uploads depend on the build step, not on each other. Jobs in each uploaded file depend on that upload step. The L1 and L3 NVIDIA P/D jobs, and the L3 gRPC perf job, also wait for `image-build`. `NIGHTLY=1` on this test pipeline selects tests. `NIGHTLY=1`
on `release-pipeline.yml` (the `router-release` pipeline) publishes images.

| File | Level | When it is uploaded | What runs |
| --- | --- | --- | --- |
| `test-ready.yml` | L1 ready | non-default branch labeled `ready` | Rust and Python unit/integration tests, one NVIDIA P/D completion (`ROCM_ONLY != 1` on that job) |
| `test-merge.yml` | L2 merge | default branch (except `NIGHTLY=1`), its merge queue, or any other branch labeled `merge-test` | one-mode xGMI smoke and RDMA `SMOKE_ONLY` |
| `test-nightly.yml` | L3 nightly | default branch with `NIGHTLY=1`, or any other branch labeled `nightly-test` | Rust and Python unit/integration tests (skipped when the PR also has `ready`), full NVIDIA GSM8K (`ROCM_ONLY != 1`), gRPC prefix-KV perf on 2× H100 (`ROCM_ONLY != 1`), full xGMI matrix, full RDMA |
| `release-pipeline.yml` | release | tag / release pipeline | wheels and images only |

A pull request labeled `ready` uploads `test-ready.yml`, including forks.
`test-merge.yml` has no per-job `if`. In `test-ready.yml` and `test-nightly.yml`, a job whose gate is narrower than the upload keeps its own `if`.

GPU machine presets are in [`ci_mirror_hardwares.yml`](ci_mirror_hardwares.yml).
A step names one with `mirror_hardwares`. `scripts/upload_pipeline.py` expands
that name into `agents` and `plugins` before Buildkite upload.

| Name | Machine |
| --- | --- |
| `l4_4` | EKS `l4-k8s`, 4× L4, image `public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:router-$BUILDKITE_COMMIT` |
| `h100_2` | `mithril-h100-pool`, 2× H100 SXM, same router CI image |
| `h100_4` | `mithril-h100-pool`, 4× H100 SXM, same router CI image (router_policy perf) |
| `mi300_2` | `router_rocm_mi300_2`, same-host xGMI Docker devices |
| `mi300_rdma` | `router_rocm_mi300_2` coordinator, no Docker plugin |

CPU steps keep `medium_cpu_queue_premerge` inline. Their command lives in the Docker plugin, so they do not use a preset. Dependency sets live in `ci_apt_packages.conf` and `scripts/install_packages.sh`. `rust` and `nvidia_pd` install apt packages. `router-bin` downloads the release binary and checks it. `lm-eval` installs the ROCm accuracy harness. A sourced `python` call downloads the binary and installs `.[dev]` into `/tmp/venv`. `python-wheel` uses that apt set and venv for `python -m build`.

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
H100 router_policy perf job while retaining the CPU build that supplies the router
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
| `NUM_SANITY_REQUESTS` | `20` | Completions before GSM8K. L1 sets `1` |
| `SKIP_LM_EVAL` | `false` | Skip full GSM8K. L1 sets `true` |

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
The `build` group in `pipeline.yml` creates release artifacts:
- Rust binary (`target/release/vllm-router`)
- Python wheels and source distribution

L1, L2, and L3 uploads depend on this group. L2 and L3 do not depend on the L1 upload.

### Tests
Comprehensive test suite:
- Rust unit tests
- Rust integration tests
- Python tests with coverage

### Benchmarks
Not part of the pipeline. Run them locally with `scripts/run_benchmarks.py`
(criterion) or the router overhead harness; see
`docs/benchmarks/router_overhead.md`.

### Docker Build
Builds Docker image for the router.

## Agent Queues

- `medium_cpu_queue_premerge`: CPU-only tasks (builds, lints, unit tests)
- `l4-k8s`: NVIDIA L4 GPU tests on EKS (4-GPU P/D disaggregation)
- `mithril-h100-pool`: NVIDIA H100 SXM tests (2-GPU gRPC prefix-KV perf; 4-GPU router_policy c16)
- `router_rocm_mi300_2`: dedicated two-node MI300X MoRI tests (`spawn=1`)
- `default`: General purpose queue

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
