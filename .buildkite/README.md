# Buildkite CI/CD Configuration

This directory contains the Buildkite pipeline configurations for the vLLM Router project.

## Pipeline Files

### `pipeline.yml`
Main CI/CD pipeline that runs on all commits and pull requests. Includes:

- **Fast Checks**: Code formatting and linting (Rust, Python)
- **Build**: Release builds for Rust binary and Python wheels
- **Tests**: Comprehensive test suite (unit, integration, Python)
- **P/D Disaggregation Test**: GPU-based integration test for prefill/decode disaggregation
- **Benchmarks**: Optional performance benchmarks
- **Docker Build**: Container image creation

### `release-pipeline.yml`
Release pipeline triggered on version tags (e.g., `v1.2.3`). Handles:

- Building release artifacts
- Publishing to PyPI
- Building and pushing Docker images
- Creating GitHub releases

## P/D Disaggregation Test

The P/D (Prefill/Decode) disaggregation test validates the router's ability to coordinate separate prefill and decode vLLM instances.

### How It Works

The test launches:
1. **Prefill instance(s)**: vLLM servers configured for prefill operations
2. **Decode instance(s)**: vLLM servers configured for decode operations
3. **Router**: Coordinates requests between prefill and decode instances

See `scripts/llama3.1/` for example setup scripts showing the proper configuration.

### Test Script

Location: `py_test/e2e/pd_disagg_vllm/run_accuracy_test.sh`

The test script:
- Launches prefill and decode vLLM instances in Docker containers
- Starts the router with `--pd-disaggregation` flag
- Runs accuracy validation tests
- Cleans up containers on exit

### CI Configuration

The test runs in the pipeline at `.buildkite/pipeline.yml:97-132`:

```yaml
- label: ":satellite: P/D Disaggregation Test (4 GPUs)"
  timeout_in_minutes: 30
  retry:
    automatic:
      - exit_status: "*"
        limit: 2
```

**Key features:**
- Requires 4 GPUs (runs on `gpu_4_queue`)
- 30 minute timeout
- Automatic retry (up to 2 attempts) for flaky failures
- Manual retry option available

### ROCm MoRI coverage

The dedicated `router_rocm_mi300_2` queue adds two MI300X matrices, each with
`read_mode=true` and `read_mode=false`:

- `run_moriio_accuracy_test.sh` uses two GPUs on one host with the xGMI backend.
- `run_moriio_rdma_accuracy_test.sh` uses one GPU on each of two hosts with the
  RDMA backend.

The xGMI group runs for same-repository pull requests, the default branch, and
merge-queue builds. Trusted manual/API builds can set `RUN_ROCM_XGMI=1`. The
RDMA matrix runs for scheduled builds with `NIGHTLY=1`, or trusted manual/API
builds with `RUN_ROCM_RDMA=1`. Group-level filtering prevents Docker plugin
hooks from running for untrusted fork pull requests.

ROCm-only scheduled builds should set both `NIGHTLY=1` and `ROCM_ONLY=1`.
`ROCM_ONLY=1` skips the unrelated NVIDIA four-GPU P/D step while retaining the
CPU build that supplies the router artifact to both ROCm matrices.

Both matrices use a digest-pinned ROCm vLLM image and run health, sanity, and
500-example GSM8K accuracy checks. The RDMA job is coordinator-owned and
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

### Artifacts

On test completion (success or failure), artifacts are collected:

- `/tmp/router.log`: Router process logs

### Running Locally

To run the test locally:

```bash
cd py_test/e2e/pd_disagg_vllm

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

### Fast Checks
Runs in parallel for quick feedback:
- Rust format check (`cargo fmt`)
- Clippy linting (`cargo clippy`)
- Python format check (black, ruff)

### Build
Creates release artifacts:
- Rust binary (`target/release/vllm-router`)
- Python wheels and source distribution

### Tests
Comprehensive test suite:
- Rust unit tests
- Rust integration tests
- Python tests with coverage

### Benchmarks
Optional manual trigger for performance benchmarks.

### Docker Build
Builds Docker image for the router.

## Agent Queues

- `cpu_queue_premerge`: CPU-only tasks (builds, lints, unit tests)
- `gpu_4_queue`: GPU tests requiring 4+ GPUs
- `router_rocm_mi300_2`: dedicated two-node MI300X MoRI tests (`spawn=1`)
- `default`: General purpose queue

## Adding New Tests

To add a new test step:

1. Choose appropriate location in pipeline
2. Define test command and dependencies
3. Specify agent queue
4. Add artifact collection
5. Consider retry logic for flaky tests
6. Update this documentation

Example:
```yaml
- label: ":test_tube: New Test"
  command: |
    # Test commands here
  agents:
    queue: "cpu_queue_premerge"
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
