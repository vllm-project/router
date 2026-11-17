"""CPU bootstrap regression tests: pytest tests/python/buildkite -m "ready and cpu"."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".buildkite" / "scripts"))
import skip_docs  # noqa: E402

pytestmark = [pytest.mark.ready, pytest.mark.cpu]


@pytest.mark.parametrize(
    "path", [".buildkite/pipeline.yml", "tests/python/buildkite/test_skip_docs.py"]
)
def test_ci_and_test_changes_are_not_docs(path):
    assert not skip_docs.is_doc(path)


class TestBootstrapRouting:
    def test_all_labels_use_docs_detection(self):
        root = Path(__file__).resolve().parents[3]
        pipeline = yaml.safe_load((root / ".buildkite/pipeline.yml").read_text())
        assert len(pipeline["steps"]) == 1
        docs_step = pipeline["steps"][0]
        assert "if" not in docs_step
        assert docs_step["command"] == "python3 .buildkite/scripts/skip_docs.py"


class TestDocsOnlyChanges:
    @pytest.fixture(autouse=True)
    def setup_repo(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.git("init", "-b", "main")
        self.git("config", "user.name", "CI Test")
        self.git("config", "user.email", "ci@example.com")
        self.write("README.md", "initial\n")
        self.write("src/main.rs", "fn main() {}\n")
        self.commit()
        self.git("remote", "add", "origin", str(self.root))
        self.git("checkout", "-b", "feature")
        self.env = {
            "BUILDKITE_PULL_REQUEST": "12",
            "BUILDKITE_PULL_REQUEST_BASE_BRANCH": "main",
        }
        monkeypatch.setattr(skip_docs, "ROOT", self.root)

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
        assert skip_docs.docs_only_change(self.env)

    def test_mixed_change(self):
        self.write("README.md", "updated\n")
        self.write("src/main.rs", 'fn main() { println!("hi"); }\n')
        self.commit()
        assert not skip_docs.docs_only_change(self.env)

    def test_code_renamed_into_docs(self):
        self.git("mv", "src/main.rs", "example.md")
        self.commit()
        assert not skip_docs.docs_only_change(self.env)

    def test_empty_diff(self):
        assert not skip_docs.docs_only_change(self.env)

    def test_push_and_nightly(self):
        self.write("README.md", "updated\n")
        self.commit()
        for overrides in ({"BUILDKITE_PULL_REQUEST": "false"}, {"NIGHTLY": "1"}):
            assert not skip_docs.docs_only_change({**self.env, **overrides})

    def test_unknown_base(self):
        self.env["BUILDKITE_PULL_REQUEST_BASE_BRANCH"] = "missing"
        assert not skip_docs.docs_only_change(self.env)

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
        assert skip_docs.docs_only_change(self.main_env())

    def test_main_merge_includes_all_pr_commits(self):
        self.write("src/main.rs", "// code change in earlier commit\n")
        self.commit()
        self.write("README.md", "final docs commit\n")
        self.commit()
        self.git("checkout", "main")
        self.git("merge", "--no-ff", "feature", "-m", "Merge PR")
        assert not skip_docs.docs_only_change(self.main_env())

    def test_main_docs_merge(self):
        self.write("README.md", "documentation\n")
        self.commit()
        self.git("checkout", "main")
        self.git("merge", "--no-ff", "feature", "-m", "Merge PR")
        assert skip_docs.docs_only_change(self.main_env())

    def test_merge_queue_docs_and_code(self):
        base = self.git("rev-parse", "HEAD").decode().strip()
        env = {
            "BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main",
            "BUILDKITE_MERGE_QUEUE_BASE_COMMIT": base,
        }
        self.write("README.md", "documentation\n")
        self.commit()
        assert skip_docs.docs_only_change(env)
        self.write("src/main.rs", "// queue contains code too\n")
        self.commit()
        assert not skip_docs.docs_only_change(env)

    def test_merge_queue_missing_base_runs(self):
        self.write("README.md", "documentation\n")
        self.commit()
        assert not skip_docs.docs_only_change(
            {"BUILDKITE_MERGE_QUEUE_BASE_BRANCH": "main"}
        )

    def test_nightly_and_schedule_on_main_run(self):
        self.git("checkout", "main")
        self.write("README.md", "documentation\n")
        self.commit()
        for overrides in ({"NIGHTLY": "1"}, {"BUILDKITE_SOURCE": "schedule"}):
            assert not skip_docs.docs_only_change(self.main_env(**overrides))

    def test_all_test_labels_docs_pr_skips(self):
        self.write("README.md", "documentation\n")
        self.commit()
        for labels in (
            "ready",
            "merge-test",
            "nightly-test",
            "ready,merge-test,nightly-test",
        ):
            assert skip_docs.docs_only_change(
                {**self.env, "BUILDKITE_PULL_REQUEST_LABELS": labels}
            )

    def test_merge_base_excludes_target_only_changes(self):
        self.git("checkout", "main")
        self.write("src/main.rs", "// upstream change\n")
        self.commit()
        self.git("checkout", "feature")
        self.write("README.md", "updated\n")
        self.commit()
        assert skip_docs.docs_only_change(self.env)

    @pytest.mark.parametrize("docs_only", [True, False])
    def test_upload_only_when_needed(self, docs_only, monkeypatch):
        uploads = []
        monkeypatch.setattr(skip_docs, "docs_only_change", lambda env: docs_only)
        monkeypatch.setattr(
            skip_docs.subprocess, "run", lambda *args, **kwargs: uploads.append(args)
        )
        skip_docs.main()
        assert len(uploads) == int(not docs_only)
