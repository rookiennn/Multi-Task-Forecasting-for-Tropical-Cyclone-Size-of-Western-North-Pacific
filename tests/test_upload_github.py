"""Offline integration checks for the PowerShell Git upload helper.

Run with: python -m unittest discover -s tests -p test_upload_github.py -v
Set TCSF_TEST_POWERSHELL to a PowerShell executable to select a specific engine.
All repositories and remotes are disposable local directories, with no network.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_SCRIPT = PROJECT_ROOT / "upload_github.ps1"
GIT = shutil.which("git")
POWERSHELL = (
    shutil.which(os.environ["TCSF_TEST_POWERSHELL"])
    if os.environ.get("TCSF_TEST_POWERSHELL")
    else shutil.which("pwsh") or shutil.which("powershell")
)


@unittest.skipUnless(GIT and POWERSHELL, "Git and PowerShell are required for upload integration tests.")
class UploadGitHubTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tcsf-upload-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = self.base / "project-\u6d4b\u8bd5"
        self.project.mkdir()
        self.caller = self.base / "caller"
        self.caller.mkdir()
        self.remote = self.base / "remote.git"
        self.other_remote = self.base / "other-remote.git"

        config = self.base / "isolated.gitconfig"
        config.write_text(
            "[user]\n"
            "    name = Offline Upload Test\n"
            "    email = upload-test@example.invalid\n"
            "[init]\n"
            "    defaultBranch = main\n"
            "[core]\n"
            "    autocrlf = false\n",
            encoding="utf-8",
        )
        self.environment = os.environ.copy()
        for key in list(self.environment):
            if key in {
                "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
            } or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
                self.environment.pop(key)
        self.environment.update(
            GIT_CONFIG_GLOBAL=str(config),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_TERMINAL_PROMPT="0",
            PYTHONUTF8="1",
        )

        (self.project / ".gitignore").write_text(
            ".venv/\ndata/\noutputs/\n__pycache__/\n", encoding="utf-8"
        )
        (self.project / "main.py").write_text("print('initial fixture')\n", encoding="utf-8")
        for directory in (".venv", "data", "outputs"):
            path = self.project / directory
            path.mkdir()
            (path / "must_not_upload.txt").write_text("Ignored fixture data.\n", encoding="utf-8")
        (self.project / "model").mkdir()
        (self.project / "model" / "dummy.pth").write_bytes(b"offline test checkpoint fixture\n")
        self.run_git("init", "--bare", self.remote)
        self.run_git("init", "--bare", self.other_remote)

    def run_process(self, arguments, expect_success=True):
        result = subprocess.run(
            list(map(str, arguments)),
            cwd=self.caller,
            env=self.environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
        details = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        if expect_success:
            self.assertEqual(result.returncode, 0, details)
        else:
            self.assertNotEqual(result.returncode, 0, details)
        return result

    def run_git(self, *arguments, repository=None, expect_success=True):
        command = [GIT]
        if repository is not None:
            command += ["-C", repository]
        command += list(arguments)
        return self.run_process(command, expect_success=expect_success)

    def upload(self, repository=None, dry_run=False, expect_success=True,
               script=UPLOAD_SCRIPT, include_project=True, message="Upload offline fixture"):
        command = [
            POWERSHELL, "-NoLogo", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-File", script,
            "-RepositoryUrl", repository or self.remote,
            "-CommitMessage", message,
        ]
        # Leave -Branch unspecified to test its documented main default.
        if include_project:
            command += ["-ProjectPath", self.project]
        if dry_run:
            command += ["-DryRun"]
        return self.run_process(command, expect_success=expect_success)

    def remote_refs(self, remote):
        result = self.run_git("for-each-ref", "--format=%(refname)", repository=remote)
        return result.stdout.splitlines()

    def remote_head(self, remote):
        return self.run_git("rev-parse", "refs/heads/main", repository=remote).stdout.strip()

    def test_dry_run_changes_neither_project_nor_remote(self):
        self.upload(dry_run=True)
        self.assertFalse((self.project / ".git").exists())
        self.assertEqual(self.remote_refs(self.remote), [])

        # A copied script also checks the default ProjectPath from a Unicode
        # script directory while the caller's working directory is elsewhere.
        copied_script = self.project / "upload_github.ps1"
        shutil.copyfile(UPLOAD_SCRIPT, copied_script)
        self.upload(dry_run=True, script=copied_script, include_project=False)
        self.assertFalse((self.project / ".git").exists())
        self.assertEqual(self.remote_refs(self.remote), [])
        self.assertEqual(self.remote_refs(self.other_remote), [])

    def test_upload_respects_ignore_and_supports_repeat_updates(self):
        self.upload()
        uploaded = set(
            self.run_git(
                "ls-tree", "-r", "--name-only", "refs/heads/main", repository=self.remote
            ).stdout.splitlines()
        )
        self.assertEqual(uploaded, {".gitignore", "main.py", "model/dummy.pth"})
        first_head = self.remote_head(self.remote)

        self.upload()
        self.assertEqual(self.remote_head(self.remote), first_head)

        changed_source = "print('updated fixture')\n"
        (self.project / "main.py").write_text(changed_source, encoding="utf-8")
        self.upload(message="Update offline fixture")
        updated_head = self.remote_head(self.remote)
        self.assertNotEqual(updated_head, first_head)
        self.assertEqual(
            self.run_git("show", "refs/heads/main:main.py", repository=self.remote).stdout,
            changed_source,
        )
        self.run_git("merge-base", "--is-ancestor", first_head, updated_head, repository=self.remote)
        self.assertEqual(self.remote_refs(self.other_remote), [])

    def test_origin_mismatch_aborts_before_staging_or_pushing(self):
        self.run_git("init", "--initial-branch=main", repository=self.project)
        self.run_git("add", "--all", repository=self.project)
        self.run_git("commit", "-m", "Create fixture baseline", repository=self.project)
        self.run_git("remote", "add", "origin", self.remote, repository=self.project)
        self.run_git("push", "origin", "HEAD:refs/heads/main", repository=self.project)
        original_head = self.remote_head(self.remote)
        (self.project / "must_remain_unstaged.txt").write_text("Do not stage.\n", encoding="utf-8")

        result = self.upload(repository=self.other_remote, expect_success=False)
        self.assertIn("origin", (result.stdout + result.stderr).lower())
        self.assertEqual(
            self.run_git("diff", "--cached", "--name-only", repository=self.project).stdout.strip(),
            "",
        )
        self.assertIn(
            "?? must_remain_unstaged.txt",
            self.run_git("status", "--porcelain", repository=self.project).stdout,
        )
        self.assertEqual(
            self.run_git("rev-parse", "HEAD", repository=self.project).stdout.strip(), original_head
        )
        self.assertEqual(
            Path(self.run_git("remote", "get-url", "origin", repository=self.project).stdout.strip()),
            self.remote,
        )
        self.assertEqual(self.remote_head(self.remote), original_head)
        self.assertEqual(self.remote_refs(self.other_remote), [])


if __name__ == "__main__":
    unittest.main()
