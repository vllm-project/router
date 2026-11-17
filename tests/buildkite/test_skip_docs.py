"""CPU-only bootstrap regression tests: python -m unittest discover -s tests/buildkite."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".buildkite" / "scripts"))

import skip_docs  # noqa: E402


class BootstrapRoutingTests(unittest.TestCase):
    def test_nightly_label_bypasses_docs_detection(self):
        root = Path(__file__).resolve().parents[2]
        pipeline = yaml.safe_load((root / ".buildkite/pipeline.yml").read_text())
        docs_step, nightly_step = pipeline["steps"]
        # Buildkite evaluates these complementary gates before running commands.
        label_gate = 'build.pull_request.labels includes "nightly-test"'
        self.assertEqual(docs_step["if"], f"!({label_gate})")
        self.assertEqual(nightly_step["if"], label_gate)
        self.assertIn("skip_docs.py", docs_step["command"])
        self.assertEqual(
            nightly_step["command"].split(),
            [
                "buildkite-agent",
                "pipeline",
                "upload",
                ".buildkite/bootstrap-upload-steps.yml",
            ],
        )
        self.assertTrue((root / nightly_step["command"].split()[-1]).is_file())


class DocsOnlyPRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-b", "main")
        self.git("config", "user.name", "CI Test")
        self.git("config", "user.email", "ci@example.com")
        self.write("README.md", "initial\n")
        self.write("src/main.rs", "fn main() {}\n")
        self.commit()
        # A local target ref models the upstream repository without network access.
        self.git("remote", "add", "origin", str(self.root))
        self.git("checkout", "-b", "feature")
        self.env = {
            "BUILDKITE_PULL_REQUEST": "12",
            "BUILDKITE_PULL_REQUEST_BASE_BRANCH": "main",
        }
        self.root_patch = patch.object(skip_docs, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.root, stderr=subprocess.PIPE
        )

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-m", "test")

    def test_documentation_add_modify_delete(self):
        self.write("docs/image.svg", "<svg/>\n")
        self.write("guide.md", "guide\n")
        self.write("mkdocs.yml", "site_name: router\n")
        (self.root / "README.md").unlink()
        self.commit()
        self.assertTrue(skip_docs.docs_only_change(self.env))

    def test_mixed_change(self):
        self.write("README.md", "updated\n")
        self.write("src/main.rs", 'fn main() { println!("hi"); }\n')
        self.commit()
        self.assertFalse(skip_docs.docs_only_change(self.env))

    def test_code_renamed_into_docs(self):
        self.git("mv", "src/main.rs", "example.md")
        self.commit()
        self.assertFalse(skip_docs.docs_only_change(self.env))

    def test_empty_diff(self):
        self.assertFalse(skip_docs.docs_only_change(self.env))

    def test_push_and_nightly(self):
        self.write("README.md", "updated\n")
        self.commit()
        for overrides in ({"BUILDKITE_PULL_REQUEST": "false"}, {"NIGHTLY": "1"}):
            with self.subTest(overrides=overrides):
                self.assertFalse(skip_docs.docs_only_change({**self.env, **overrides}))

    def test_unknown_base(self):
        self.env["BUILDKITE_PULL_REQUEST_BASE_BRANCH"] = "missing"
        self.assertFalse(skip_docs.docs_only_change(self.env))

    def main_env(self, **overrides):
        return {
            "BUILDKITE_PULL_REQUEST": "false",
            "BUILDKITE_BRANCH": "main",
            "BUILDKITE_PIPELINE_DEFAULT_BRANCH": "main",
            **overrides,
        }

    def test_main_squash_docs(self):
        self.git("checkout", "main")
        self.write("README.md", "squashed documentation\n")
        self.commit()
        self.assertTrue(skip_docs.docs_only_change(self.main_env()))

    def test_main_merge_includes_all_pr_commits(self):
        self.write("src/main.rs", "// code change in earlier commit\n")
        self.commit()
        self.write("README.md", "final docs commit\n")
        self.commit()
        self.git("checkout", "main")
        self.git("merge", "--no-ff", "feature", "-m", "Merge PR")
        self.assertFalse(skip_docs.docs_only_change(self.main_env()))

    def test_main_docs_merge(self):
        self.write("README.md", "documentation\n")
        self.commit()
        self.git("checkout", "main")
        self.git("merge", "--no-ff", "feature", "-m", "Merge PR")
        self.assertTrue(skip_docs.docs_only_change(self.main_env()))

    def test_merge_queue_docs_and_code(self):
        base = self.git("rev-parse", "HEAD").decode().strip()
        env = {
            "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
            "BUILDKITE_MERGE_QUEUE_BASE_COMMIT": base,
        }
        self.write("README.md", "documentation\n")
        self.commit()
        self.assertTrue(skip_docs.docs_only_change(env))
        self.write("src/main.rs", "// queue contains code too\n")
        self.commit()
        self.assertFalse(skip_docs.docs_only_change(env))

    def test_merge_queue_missing_base_runs(self):
        self.write("README.md", "documentation\n")
        self.commit()
        self.assertFalse(
            skip_docs.docs_only_change({"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"})
        )

    def test_nightly_and_schedule_on_main_run(self):
        self.git("checkout", "main")
        self.write("README.md", "documentation\n")
        self.commit()
        for overrides in ({"NIGHTLY": "1"}, {"BUILDKITE_SOURCE": "schedule"}):
            with self.subTest(overrides=overrides):
                self.assertFalse(skip_docs.docs_only_change(self.main_env(**overrides)))

    def test_merge_test_docs_pr_skips(self):
        self.write("README.md", "documentation\n")
        self.commit()
        self.assertTrue(
            skip_docs.docs_only_change(
                {**self.env, "BUILDKITE_PULL_REQUEST_LABELS": "merge-test"}
            )
        )

    def test_merge_base_excludes_target_only_changes(self):
        self.git("checkout", "main")
        self.write("src/main.rs", "// upstream change\n")
        self.commit()
        self.git("checkout", "feature")
        self.write("README.md", "updated\n")
        self.commit()
        self.assertTrue(skip_docs.docs_only_change(self.env))

    def test_upload_only_when_needed(self):
        for docs_only in (True, False):
            with self.subTest(docs_only=docs_only), patch.object(
                skip_docs, "docs_only_change", return_value=docs_only
            ), patch.object(skip_docs.subprocess, "run") as upload:
                skip_docs.main()
                self.assertEqual(upload.call_count, int(not docs_only))


if __name__ == "__main__":
    unittest.main()
