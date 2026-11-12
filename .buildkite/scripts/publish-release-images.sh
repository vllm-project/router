#!/bin/bash
# Publish multi-arch Docker images to DockerHub.
#
# Usage:
#   publish-release-images.sh            # release mode (default)
#   publish-release-images.sh release
#   publish-release-images.sh nightly
#
# Expects per-arch images already pushed by the matching build steps:
#   nightly:  vllm/vllm-router:nightly-{x86_64,aarch64}
#   release:  vllm/vllm-router:v${RELEASE_VERSION}-{x86_64,aarch64}
#
# Nightly tags:  nightly, nightly-YYYYMMDD-<sha7>
# Release tags:  latest, v${VERSION} (+ per-arch)

set -euo pipefail

MODE="${1:-release}"
DOCKERHUB_REPO="vllm/vllm-router"
COMMIT="${BUILDKITE_COMMIT:-}"
PRIMARY_TAGS=()
EXTRA_MANIFEST_TAGS=()
ARCH_TAG_BASE=""
SRC_X86=""
SRC_ARM=""

case "${MODE}" in
  release)
    RELEASE_VERSION=$(buildkite-agent meta-data get release-version --default "" | sed 's/^v//')
    if [ -z "${RELEASE_VERSION}" ]; then
      echo "ERROR: release-version metadata not set."
      echo "Fill the 'Provide Release version here' input step (X.Y.Z)."
      exit 1
    fi
    if ! [[ "${RELEASE_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]]; then
      echo "ERROR: invalid release version '${RELEASE_VERSION}'. Expected X.Y.Z"
      exit 1
    fi
    ARCH_TAG_BASE="v${RELEASE_VERSION}"
    PRIMARY_TAGS=("latest" "v${RELEASE_VERSION}")
    SRC_X86="${DOCKERHUB_REPO}:v${RELEASE_VERSION}-x86_64"
    SRC_ARM="${DOCKERHUB_REPO}:v${RELEASE_VERSION}-aarch64"
    ;;
  nightly)
    if [ -z "${COMMIT}" ]; then
      echo "ERROR: BUILDKITE_COMMIT is not set"
      exit 1
    fi
    ARCH_TAG_BASE="nightly"
    PRIMARY_TAGS=("nightly")
    EXTRA_MANIFEST_TAGS=("nightly-$(date +%Y%m%d)-${COMMIT:0:7}")
    SRC_X86="${DOCKERHUB_REPO}:nightly-x86_64"
    SRC_ARM="${DOCKERHUB_REPO}:nightly-aarch64"
    ;;
  *)
    echo "Usage: $0 {release|nightly}"
    exit 2
    ;;
esac

echo "========================================"
echo "Publishing ${MODE} images to ${DOCKERHUB_REPO}"
echo "  Commit: ${COMMIT:-n/a}"
echo "  Source: ${SRC_X86} ${SRC_ARM}"
echo "  Primary tags: ${PRIMARY_TAGS[*]}"
if [ ${#EXTRA_MANIFEST_TAGS[@]} -gt 0 ]; then
  echo "  Extra manifests: ${EXTRA_MANIFEST_TAGS[*]}"
fi
echo "========================================"

docker pull "${SRC_X86}"
docker pull "${SRC_ARM}"

publish_manifest() {
  local tag="$1"
  local arch_tag_base="$2"
  docker manifest rm "${DOCKERHUB_REPO}:${tag}" || true
  docker manifest create \
    "${DOCKERHUB_REPO}:${tag}" \
    "${DOCKERHUB_REPO}:${arch_tag_base}-x86_64" \
    "${DOCKERHUB_REPO}:${arch_tag_base}-aarch64"
  docker manifest push "${DOCKERHUB_REPO}:${tag}"
}

for tag in "${PRIMARY_TAGS[@]}"; do
  if [ "${tag}" != "${ARCH_TAG_BASE}" ]; then
    docker tag "${SRC_X86}" "${DOCKERHUB_REPO}:${tag}-x86_64"
    docker tag "${SRC_ARM}" "${DOCKERHUB_REPO}:${tag}-aarch64"
    docker push "${DOCKERHUB_REPO}:${tag}-x86_64"
    docker push "${DOCKERHUB_REPO}:${tag}-aarch64"
  fi
  publish_manifest "${tag}" "${tag}"
done

for tag in "${EXTRA_MANIFEST_TAGS[@]}"; do
  publish_manifest "${tag}" "${ARCH_TAG_BASE}"
done

echo ""
echo "Successfully published ${MODE} images to ${DOCKERHUB_REPO}"
