#!/usr/bin/env bash
# Build and push the router NVIDIA CI image to the shared vLLM public ECR.
# Same registry as vllm-omni and afd-plugin. The tag is router-specific so
# it does not replace another project's image for the same commit SHA.
#
# Expects BUILDKITE_COMMIT from the Buildkite agent.
# The CPU build step must already have uploaded target/release/vllm-router.

set -euo pipefail

ECR_NAMESPACE="public.ecr.aws/q9t5s3a7"
REGISTRY="${ECR_NAMESPACE}/vllm-ci-test-repo"
REGION="us-east-1"
DOCKERFILE="docker/Dockerfile.ci"
BUILDER_NAME="vllm-router-builder"

if [ -z "${BUILDKITE_COMMIT:-}" ]; then
  echo "ERROR: BUILDKITE_COMMIT is not set" >&2
  exit 1
fi

IMAGE_TAG="router-${BUILDKITE_COMMIT}"
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"
echo "Image: ${REGISTRY}:${IMAGE_TAG}"

buildkite-agent artifact download "target/release/vllm-router" .
chmod +x target/release/vllm-router

aws ecr-public get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_NAMESPACE}"

# Cache key covers the image inputs. The router- prefix keeps this cache
# off the deps-cache-* tags used by vllm-omni and afd-plugin.
DEP_FILES="${DOCKERFILE} .buildkite/ci_apt_packages.conf .buildkite/scripts/install_packages.sh"
CACHE_KEY="$(cat ${DEP_FILES} | sha256sum | cut -c1-16)"
CACHE_TAG="router-deps-cache-${CACHE_KEY}"
echo "Cache key: ${CACHE_TAG}"

docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1 \
  || docker buildx create --name "${BUILDER_NAME}" --driver docker-container
docker buildx use "${BUILDER_NAME}"

docker buildx build --push --progress=plain \
  --cache-from "type=registry,ref=${REGISTRY}:${CACHE_TAG}" \
  --cache-to "type=registry,ref=${REGISTRY}:${CACHE_TAG},mode=max,compression=zstd" \
  --file "${DOCKERFILE}" \
  -t "${REGISTRY}:${IMAGE_TAG}" \
  .
