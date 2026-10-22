# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Load agentaware case/runner modules (pytest import bootstrap)."""

from __future__ import annotations

from _siblings import import_sibling_module

_case_config = import_sibling_module("case_config")
_replay_runner = import_sibling_module("replay_runner")

RouterReplayE2EConfig = _case_config.RouterReplayE2EConfig
run_router_replay = _replay_runner.run_router_replay
