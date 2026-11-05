"""The prompt logging switch takes an explicit boolean in both CLI namespaces."""

import argparse

import pytest
from vllm_router.router_args import RouterArgs

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.mark.parametrize("use_prefix", [False, True])
@pytest.mark.parametrize("value", [None, "true", "false"])
def test_prompt_logging_switch(use_prefix, value):
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=use_prefix)
    prefix = "router-" if use_prefix else ""
    argv = [] if value is None else [f"--{prefix}log-user-prompts={value}"]
    args = RouterArgs.from_cli_args(
        parser.parse_args(argv), use_router_prefix=use_prefix
    )
    assert args.log_user_prompts is (value == "true")
    assert RouterArgs().log_user_prompts is False


@pytest.mark.parametrize("value", ["invalid", "1", "0", "TRUE", ""])
def test_prompt_logging_rejects_invalid_boolean(value):
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([f"--log-user-prompts={value}"])


def test_prompt_logging_requires_a_value():
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--log-user-prompts"])
