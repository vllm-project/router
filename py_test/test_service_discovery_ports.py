import argparse
import contextlib
import io
import unittest

from vllm_router.router_args import RouterArgs
from vllm_router_rs import PolicyType, Router


class TestServiceDiscoveryPorts(unittest.TestCase):
    def test_cli_ports(self):
        for prefix in (False, True):
            flag = (
                "--router-service-discovery-port"
                if prefix
                else "--service-discovery-port"
            )
            parser = argparse.ArgumentParser()
            RouterArgs.add_cli_args(parser, use_router_prefix=prefix)
            cases = [
                ([], 80),
                ([flag, "8000"], 8000),
                ([flag, "8000", "8001"], [8000, 8001]),
                ([flag, "8000", flag, "8001"], [8000, 8001]),
            ]
            for argv, expected in cases:
                with self.subTest(prefix=prefix, argv=argv):
                    args = RouterArgs.from_cli_args(
                        parser.parse_args(argv), use_router_prefix=prefix
                    )
                    self.assertEqual(args.service_discovery_port, expected)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(
                io.StringIO()
            ):
                parser.parse_args([flag])

    def test_python_accepts_scalar_and_list(self):
        for ports in (8000, [8000], [8000, 8001], [8000, 8000]):
            with self.subTest(ports=ports):
                Router(
                    worker_urls=[],
                    policy=PolicyType.RoundRobin,
                    service_discovery_port=ports,
                )

    def test_python_rejects_empty_and_invalid_ports(self):
        for ports in ([], 0, -1, 65536, [8000, 0], [8000, -1], [8000, 65536]):
            with self.subTest(ports=ports), self.assertRaises(
                (ValueError, TypeError, OverflowError)
            ):
                Router(
                    worker_urls=[],
                    policy=PolicyType.RoundRobin,
                    service_discovery_port=ports,
                )
