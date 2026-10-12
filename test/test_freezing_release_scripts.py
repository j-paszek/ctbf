"""Bounded shell-runner tests. No network, scientific data or real Git mutation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "freeze_release"


@pytest.fixture
def release_workspace(tmp_path):
    ctbf = tmp_path / "ctbf checkout"
    bio = tmp_path / "bio checkout"
    work = tmp_path / "release workspace"
    for directory in (ctbf, bio):
        directory.mkdir()
        subprocess.run(["git", "-c", "init.defaultBranch=main", "init", str(directory)],
                       check=True, capture_output=True)
    env = os.environ.copy()
    env.update(CTBF_ROOT=str(ctbf), CTBFBIO_ROOT=str(bio), CTBF_RELEASE_WORK=str(work),
               CTBF_PYTHON=sys.executable, CTBF_RELEASE_TAG="v1.2.0", CTBFBIO_RELEASE_TAG="v1.0.0",
               CTBF_GITHUB="fixture/ctbf", CTBFBIO_GITHUB="fixture/ctbfbio")
    return ctbf, bio, work, env


def run_step(fixture, step, *args, env=None):
    ctbf, bio, work, default_env = fixture
    return subprocess.run(["bash", str(SCRIPTS / f"step{step}.sh"), *args],
                          cwd=bio, env=env or default_env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)


@pytest.mark.parametrize("script", sorted(SCRIPTS.glob("*.sh")), ids=lambda p: p.name)
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def test_setup_twice_is_independent_and_preserves_old_workspace(release_workspace):
    ctbf, bio, work, env = release_workspace
    (work / "audit").mkdir(parents=True)
    sentinel = work / "audit" / "untracked-code.txt"
    sentinel.write_text("existing inventory\n")
    first = run_step(release_workspace, 1)
    assert first.returncode == 0, first.stdout
    identity = (work / "release-config.tsv").read_bytes()
    second = run_step(release_workspace, 1)
    assert second.returncode == 0, second.stdout
    assert "complete: step1" in second.stdout
    assert sentinel.read_text() == "existing inventory\n"
    assert (work / "release-config.tsv").read_bytes() == identity
    logs = list((work / "logs").glob("step1-*.log.*"))
    assert len(logs) == 2
    assert all("complete: step1" in log.read_text() for log in logs)
    assert len(list((work / "audit").glob("untracked-code.*"))) == 3


def test_missing_python_has_persistent_failure_log(release_workspace):
    env = dict(release_workspace[-1], CTBF_PYTHON="/nonexistent/ctbf-python")
    result = run_step(release_workspace, 1, env=env)
    assert result.returncode != 0
    assert "Configured Python is unavailable" in result.stdout
    assert "FAILED: step1" in result.stdout
    log = next((release_workspace[2] / "logs").glob("step1-*"))
    assert "FAILED: step1" in log.read_text()
    assert "complete: step1" not in log.read_text()


def test_changed_identity_fails_without_overwriting(release_workspace):
    assert run_step(release_workspace, 1).returncode == 0
    identity = release_workspace[2] / "release-config.tsv"
    before = identity.read_bytes()
    env = dict(release_workspace[-1], CTBF_GITHUB="fixture/different")
    result = run_step(release_workspace, 1, env=env)
    assert result.returncode != 0
    assert "Workspace belongs to different" in result.stdout
    assert identity.read_bytes() == before


def test_independent_prerequisite_step_reports_missing_file(release_workspace):
    assert run_step(release_workspace, 1).returncode == 0
    result = run_step(release_workspace, 3)
    assert result.returncode != 0
    assert "README.md" in result.stdout
    assert "FAILED: step3" in result.stdout


@pytest.mark.parametrize("step", (9, 12, 13))
def test_mutating_release_steps_require_explicit_flag(release_workspace, step):
    assert run_step(release_workspace, 1).returncode == 0
    result = run_step(release_workspace, step)
    assert result.returncode != 0
    assert "Usage:" in result.stdout
    assert not (release_workspace[2] / "audit" / "selected-commits.tsv").exists()


def test_source_preview_does_not_change_git_index(release_workspace):
    assert run_step(release_workspace, 1).returncode == 0
    result = run_step(release_workspace, 7)
    assert result.returncode == 0, result.stdout
    assert "Preview only" in result.stdout
    assert "examples/quickstart.py" in result.stdout
    assert "test/test_quickstart.py" in result.stdout
    assert "presentation/quickstart-v2/trimmed_true_tree.html" in result.stdout
    assert "presentation/quickstart-v2/M2.html" in result.stdout
    assert "presentation/quickstart-v2/M7.html" in result.stdout
    assert "local files preserved" in result.stdout
    assert "nj.html" in result.stdout
    assert not (release_workspace[0] / ".git" / "index").exists()


@pytest.mark.parametrize("skip_worktree", (False, True))
def test_embedded_library_exclusion_preserves_local_copies_and_other_files(release_workspace, skip_worktree):
    ctbf = release_workspace[0]
    obsolete = ("nj.html", "reconstructed.html", "simulated_tree.html", "true_tree.html")
    for name in (*obsolete, "keep.txt"):
        (ctbf / name).write_text(f"preserve {name}\n")
    subprocess.run(["git", "add", "--", *obsolete, "keep.txt"], cwd=ctbf, check=True)
    if skip_worktree:
        # The owner's repository is NOT a sparse checkout, but these entries
        # have individual skip-worktree bits. Preserve an unrelated bit too.
        subprocess.run(["git", "update-index", "--skip-worktree", "--",
                        "nj.html", "true_tree.html", "keep.txt"], cwd=ctbf, check=True)
        assert subprocess.run(["git", "config", "--get", "core.sparseCheckout"],
                              cwd=ctbf, capture_output=True).returncode == 1

    def call_helper(name):
        return subprocess.run(["bash", "-c", 'source "$1"; "$2"', "check",
                               str(SCRIPTS / "common.sh"), name], cwd=ctbf,
                              text=True, capture_output=True, timeout=10)

    blocked = call_helper("require_no_tracked_embedded_html")
    assert blocked.returncode != 0 and "still tracks" in blocked.stderr
    removed = call_helper("exclude_legacy_embedded_html_from_index")
    assert removed.returncode == 0, removed.stderr
    assert call_helper("exclude_legacy_embedded_html_from_index").returncode == 0
    assert call_helper("require_no_tracked_embedded_html").returncode == 0
    assert all((ctbf / name).read_text() == f"preserve {name}\n" for name in obsolete)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ctbf, text=True)
    assert tracked.strip() == "keep.txt"
    assert (ctbf / "keep.txt").read_text() == "preserve keep.txt\n"
    if skip_worktree:
        flags = subprocess.check_output(["git", "ls-files", "-v", "--", "keep.txt"],
                                        cwd=ctbf, text=True)
        assert flags.strip() == "S keep.txt"
        assert subprocess.run(["git", "config", "--get", "core.sparseCheckout"],
                              cwd=ctbf, capture_output=True).returncode == 1


def fake_command(fixture, name):
    ctbf, bio, work, env = fixture
    directory = work / "fake-bin"
    directory.mkdir()
    command = directory / name
    command.write_text(f"#!/bin/bash\nprintf 'simulated {name} failure\\n' >&2\nexit 22\n")
    command.chmod(0o755)
    return dict(env, PATH=str(directory) + os.pathsep + env["PATH"])


def prepare_adopted_licences(fixture):
    source_root = SCRIPTS.parents[1]
    for name in ("LICENSE", "LICENSE-DATA", "LICENSESCOPE.md", "DATA_ATTRIBUTION.md", "THIRD_PARTY_NOTICES.md"):
        (fixture[0] / name).write_bytes((source_root / name).read_bytes())


def test_adopted_packet_preserves_old_candidates_without_network(release_workspace):
    assert run_step(release_workspace, 1).returncode == 0
    prepare_adopted_licences(release_workspace)
    candidates = release_workspace[2] / "licence-candidates"
    for name in ("LICENSE-MIT.txt", "LICENSE-DATA-CC0.txt"):
        (candidates / name).write_text("owner edited candidate\n")
    result = run_step(release_workspace, 2, env=fake_command(release_workspace, "curl"))
    assert result.returncode == 0, result.stdout
    assert "CC BY 4.0" in result.stdout
    assert "simulated curl" not in result.stdout
    packet = release_workspace[2] / "approved-licences"
    assert (packet / "LICENSE-DATA").read_bytes() == (release_workspace[0] / "LICENSE-DATA").read_bytes()
    assert not (packet / "LICENSE-DATA-CC0.txt").exists()
    assert (candidates / "LICENSE-DATA-CC0.txt").read_text() == "owner edited candidate\n"
    repeated = run_step(release_workspace, 2)
    assert repeated.returncode == 0, repeated.stdout
    assert "Preserved identical adopted notice" in repeated.stdout


def test_command_failure_is_not_swallowed_and_has_no_completed_copy(release_workspace):
    assert run_step(release_workspace, 1).returncode == 0
    prepare_adopted_licences(release_workspace)
    result = run_step(release_workspace, 2, env=fake_command(release_workspace, "cp"))
    assert result.returncode == 22
    assert "simulated cp failure" in result.stdout
    assert "FAILED: step2 (exit 22)" in result.stdout
    assert not (release_workspace[2] / "approved-licences" / "LICENSE").exists()


def test_adopted_packet_rejects_difference_without_overwriting(release_workspace):
    assert run_step(release_workspace, 1).returncode == 0
    prepare_adopted_licences(release_workspace)
    installed = run_step(release_workspace, 2)
    assert installed.returncode == 0, installed.stdout
    licence_path = release_workspace[2] / "approved-licences" / "LICENSE"
    licence_path.write_text("different text\n")
    different = run_step(release_workspace, 2)
    assert different.returncode != 0
    assert "Existing approved packet differs" in different.stdout
    assert licence_path.read_text() == "different text\n"
    assert "MIT License" in (release_workspace[0] / "LICENSE").read_text()


def test_old_candidate_install_option_is_retired(release_workspace):
    assert run_step(release_workspace, 1).returncode == 0
    result = run_step(release_workspace, 2, "--install-approved-mit")
    assert result.returncode != 0
    assert "candidate-install options are retired" in result.stdout
    assert not (release_workspace[0] / "LICENSE").exists()


def test_workspace_symlink_is_rejected_without_target_write(release_workspace):
    ctbf, bio, work, env = release_workspace
    work.symlink_to(bio, target_is_directory=True)
    result = run_step(release_workspace, 1)
    assert result.returncode != 0
    assert "must not be a symlink" in result.stdout
    assert not (bio / "logs").exists()


def test_worktree_overlap_is_rejected(release_workspace):
    ctbf, bio, work, env = release_workspace
    result = run_step(release_workspace, 1, env=dict(env, CTBF_RELEASE_WORK=str(ctbf / "release")))
    assert result.returncode != 0
    assert "outside both repositories" in result.stdout
    assert not (ctbf / "release").exists()
