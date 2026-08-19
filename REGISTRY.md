# vLLM Router Docker Registry

This document documents the container registry used for vLLM Router images.

## Registry Location

```
registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token
```

## Image Tags

| Tag | Description | Platforms |
|-----|-------------|-----------|
| `latest` | Latest release (multi-arch manifest list) | amd64, arm64 |
| `0.1.2` | Version 0.1.2 (multi-arch manifest list) | amd64, arm64 |
| `0.1.2-amd64` | Version 0.1.2 for AMD64 only | amd64 |
| `0.1.2-arm64` | Version 0.1.2 for ARM64 only | arm64 |
| `0.1.1` | Previous version (multi-arch manifest list) | amd64, arm64 |
| `0.1.0` | Initial version (multi-arch manifest list) | amd64, arm64 |

## Pull Commands

```bash
# Pull multi-arch image (automatically selects correct platform)
docker pull registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2

# Pull latest version
docker pull registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:latest

# Pull specific platform
docker pull registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2-amd64
docker pull registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2-arm64
```

## Build & Push Commands

```bash
# Build multi-arch image for both amd64 and arm64
cd /path/to/vllm-router-auth
docker buildx build --platform linux/amd64,linux/arm64 \
    -t registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2 \
    --push -f Dockerfile.router .

# Create platform-specific tags
docker buildx imagetools create \
    -t registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2-amd64 \
    registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2@sha256:<amd64-digest>

docker buildx imagetools create \
    -t registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2-arm64 \
    registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2@sha256:<arm64-digest>

# Update latest tag
docker buildx imagetools create \
    -t registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:latest \
    registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2
```

## Kubernetes Deployment

Example deployment snippet:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-router
spec:
  template:
    spec:
      containers:
      - name: router
        image: registry.jsc.fz-juelich.de/kaas/rke2-clusters/blablador/vllm-router-token:0.1.2
        ports:
        - containerPort: 8080
```

## Version History

- **0.1.2** - Current: Health endpoints (`/health`, `/health_generate`) now accessible without authentication
- **0.1.1** - Previous release
- **0.1.0** - Initial release