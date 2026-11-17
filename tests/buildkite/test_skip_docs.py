"""CPU-only bootstrap regression tests: python -m unittest discover -s tests/buildkite."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".buildkite" / "scripts"))

import skip_docs  # noqa: E402


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
        self.assertTrue(skip_docs.docs_only_pr(self.env))

    def test_mixed_change(self):
        self.write("README.md", "updated\n")
        self.write("src/main.rs", 'fn main() { println!("hi"); }\n')
        self.commit()
        self.assertFalse(skip_docs.docs_only_pr(self.env))

    def test_code_renamed_into_docs(self):
        self.git("mv", "src/main.rs", "example.md")
        self.commit()
        self.assertFalse(skip_docs.docs_only_pr(self.env))

    def test_empty_diff(self):
        self.assertFalse(skip_docs.docs_only_pr(self.env))

    def test_push_and_nightly(self):
        self.write("README.md", "updated\n")
        self.commit()
        for overrides in ({"BUILDKITE_PULL_REQUEST": "false"}, {"NIGHTLY": "1"}):
            with self.subTest(overrides=overrides):
                self.assertFalse(skip_docs.docs_only_pr({**self.env, **overrides}))

    def test_unknown_base(self):
        self.env["BUILDKITE_PULL_REQUEST_BASE_BRANCH"] = "missing"
        self.assertFalse(skip_docs.docs_only_pr(self.env))

    def test_merge_base_excludes_target_only_changes(self):
        self.git("checkout", "main")
        self.write("src/main.rs", "// upstream change\n")
        self.commit()
        self.git("checkout", "feature")
        self.write("README.md", "updated\n")
        self.commit()
        self.assertTrue(skip_docs.docs_only_pr(self.env))

    def test_upload_only_when_needed(self):
        for docs_only in (True, False):
            with self.subTest(docs_only=docs_only), patch.object(
                skip_docs, "docs_only_pr", return_value=docs_only
            ), patch.object(skip_docs.subprocess, "run") as upload:
                skip_docs.main()
                self.assertEqual(upload.call_count, int(not docs_only))


if __name__ == "__main__":
    unittest.main()
