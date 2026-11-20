#!/usr/bin/env bash
# Install a named CI dependency set.
#
#   bash .buildkite/scripts/install_packages.sh rust
#   bash .buildkite/scripts/install_packages.sh nvidia_pd
#   bash .buildkite/scripts/install_packages.sh router-bin
#   bash .buildkite/scripts/install_packages.sh lm-eval
#   source .buildkite/scripts/install_packages.sh python
#   source .buildkite/scripts/install_packages.sh python-wheel
#
# rust and nvidia_pd install apt packages from ci_apt_packages.conf.
# router-bin downloads target/release/vllm-router and checks that it runs.
# lm-eval installs the ROCm accuracy harness; vLLM and MoRI stay in the image.
# A sourced python call downloads the binary, creates /tmp/venv, and runs
# pip install -e .[dev]. python-wheel uses the same apt set and venv, then
# installs the build frontend. setuptools, wheel, and setuptools-rust come
# from pyproject.toml during that build. Sourcing keeps the venv on PATH.

set -euo pipefail

fetch_router_binary() {
  buildkite-agent artifact download "target/release/vllm-router" .
  chmod +x target/release/vllm-router
  ldd target/release/vllm-router
  target/release/vllm-router --version
}

require_sourced() {
  if [ "${script_path}" = "$0" ]; then
    echo "source install_packages.sh ${name} so the venv stays active" >&2
    exit 1
  fi
}

activate_venv() {
  python3 -m venv /tmp/venv
  # shellcheck disable=SC1091
  source /tmp/venv/bin/activate
  pip install -U pip
}

if [ "$#" -ne 1 ] || [ -z "${1}" ]; then
  echo "usage: install_packages.sh <rust|nvidia_pd|router-bin|lm-eval|python|python-wheel>" >&2
  exit 1
fi

name="$1"
script_path="${BASH_SOURCE[0]}"

if [ "${name}" = "router-bin" ]; then
  fetch_router_binary
  exit 0
fi

if [ "${name}" = "lm-eval" ]; then
  python3 -m pip install --no-cache-dir \
    requests \
    openai \
    'lm-eval[api]==0.4.12'
  exit 0
fi

apt_name="${name}"
if [ "${name}" = "python-wheel" ]; then
  apt_name="python"
fi

root="$(cd "$(dirname "${script_path}")/.." && pwd)"
config="${root}/ci_apt_packages.conf"
packages=""

while read -r key rest; do
  if [ -z "${key}" ] || [ "${key#\#}" != "${key}" ]; then
    continue
  fi
  if [ "${key}" = "${apt_name}" ]; then
    packages="${rest}"
    break
  fi
done < "${config}"

if [ -z "${packages}" ]; then
  echo "unknown package set: ${name}" >&2
  echo "known sets:" >&2
  while read -r key rest; do
    if [ -n "${key}" ] && [ "${key#\#}" = "${key}" ]; then
      echo "  ${key}" >&2
    fi
  done < "${config}"
  echo "  router-bin"
  echo "  lm-eval"
  echo "  python-wheel"
  exit 1
fi

echo "Installing apt set ${apt_name}: ${packages}"
export DEBIAN_FRONTEND=noninteractive
apt-get update
# shellcheck disable=SC2086
apt-get install -y ${packages}

if [ "${name}" = "python" ]; then
  require_sourced
  fetch_router_binary
  activate_venv
  pip install -e .[dev]
elif [ "${name}" = "python-wheel" ]; then
  require_sourced
  activate_venv
  pip install build
fi
