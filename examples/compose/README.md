# Optional prompt logging with Compose

This example builds the repository's Rust router image and passes an explicit
prompt logging switch through its startup command. Use an image built from a
revision containing this feature; older images do not recognize the option.

From this directory:

```bash
cp .env.example .env
```

Edit `.env` and set `WORKER_URL` to an actual worker URL reachable from inside
the container. A worker running on another machine can use that machine's LAN
address. `localhost` refers to the router container itself. Configure a
shared container network if the worker is in another Compose project.

The default is `ROUTER_LOG_USER_PROMPTS=false`. Set it to `true` to enable
logging. Both values are passed as `--log-user-prompts=true/false`; this
variable is used by Compose for interpolation, not read directly by Router.
An invalid value is rejected by the router CLI.

Docker Compose:

```bash
docker compose up -d --build
docker compose logs -f router
```

Podman Compose (requires an installed Compose provider):

```bash
podman compose up -d --build
podman compose logs -f router
```

After changing the switch in `.env`, recreate the router container with the
applicable command:

```bash
docker compose up -d --force-recreate router
podman compose up -d --force-recreate router
```

Enabled requests emit one INFO event on `vllm_router_rs::prompt`, including
the request ID and all user-message text. System/assistant messages and media
payloads are excluded. User text is not truncated or redacted. Normal log
filters still apply; keep `ROUTER_LOG_LEVEL=info` to see these events. The
switch does not enable logging for Python's separate `--mini-lb` debug server.

Container engines collect the existing console output, so prompt logs appear
in `compose logs` without a bind mount. Logs are also written to files if you
add the existing `--log-dir` option and configure a writable log volume.

To stop and remove the example containers, run `docker compose down` or
`podman compose down` from this directory.
