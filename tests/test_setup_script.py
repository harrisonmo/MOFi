"""Exercise setup orchestration with real Bash and offline Git/uv stand-ins."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PIN = "df014447f0812dd4f55ed60359b71a981c3eeb60"
GIT = shutil.which("git")
if os.name == "nt":
    candidate = Path(GIT).parents[1] / "bin/bash.exe" if GIT else Path("missing")
    BASH = str(candidate) if candidate.is_file() else None
else:
    BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH, reason="Setup checks require Bash (Git Bash on Windows)")

GIT_STAND_IN = r'''#!/usr/bin/env bash
set -eu
printf 'git %s\n' "$*" >> "$MOFI_TEST_LOG"
if [[ "${1:-}" == -C ]]; then shift 2; fi
case "${1:-} ${2:-}" in
    'clone --origin') mkdir -p "${@: -1}/.git" ;;
    'rev-parse --show-toplevel') printf '%s\n' "$MOFI_TEST_BUILDER" ;;
    'status --porcelain') printf '%s' "${MOFI_TEST_DIRTY:-}" ;;
    'remote get-url')
        if [[ "$3" == origin ]]; then
            printf '%s\n' "${MOFI_TEST_ORIGIN:-https://github.com/harrisonmo/Crystal-Builder.git}"
        elif [[ "${MOFI_TEST_NO_UPSTREAM:-0}" == 1 ]]; then exit 2
        else printf '%s\n' 'https://github.com/JulesOpp/Crystal-Builder.git'; fi ;;
    'cat-file -e') [[ "${MOFI_TEST_MISSING_COMMIT:-0}" == 0 ]] ;;
    'rev-parse HEAD') printf '%s\n' "${MOFI_TEST_HEAD:-df014447f0812dd4f55ed60359b71a981c3eeb60}" ;;
    'remote add'|'checkout --detach'|'fetch --no-tags') ;;
    *) printf 'Unexpected fake git command\n' >&2; exit 3 ;;
esac
'''
UV_STAND_IN = r'''#!/usr/bin/env bash
set -eu
printf 'uv %s\n' "$*" >> "$MOFI_TEST_LOG"
if [[ "${1:-}" == venv ]]; then
    folder="${@: -1}"
    mkdir -p "$folder/bin" "$folder/Scripts"
    printf 'home = synthetic\n' > "$folder/pyvenv.cfg"
    printf '#!/bin/sh\nexit 0\n' > "$folder/bin/python"
    cp "$folder/bin/python" "$folder/Scripts/python.exe"
    chmod +x "$folder/bin/python"
fi
'''


def executable(path, content):
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


@pytest.fixture
def installation(tmp_path):
    root = tmp_path / "MOFi checkout with spaces"
    root.mkdir()
    shutil.copyfile(ROOT / "setup.sh", root / "setup.sh")
    (root / "pyproject.toml").write_text("# Synthetic project\n", encoding="utf-8")
    (root / "uv.lock").write_text("# Synthetic lock\n", encoding="utf-8")
    package = root / "packages/mofi-tools"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text("# Synthetic package\n", encoding="utf-8")
    tools = tmp_path / "fake-tools"
    tools.mkdir()
    executable(tools / "git", GIT_STAND_IN)
    executable(tools / "uv", UV_STAND_IN)
    log = tmp_path / "commands.log"
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
               MOFI_TEST_LOG=log.as_posix(), MOFI_TEST_BUILDER=(root / "Crystal-Builder").as_posix(),
               MOFI_TEST_BIN=tools.as_posix())
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    return root, tools, log, env


def run(installation, *arguments, **settings):
    root, _, _, env = installation
    # Git Bash constructs its own PATH at startup. Prepend the stand-ins inside
    # Bash after converting the Windows path, and refuse to use the real Git.
    # Source the script so a second Git Bash startup cannot rewrite PATH again.
    wrapper = '''test_bin="$(cd "$MOFI_TEST_BIN" && pwd -P)"
export PATH="$test_bin:$PATH"
hash -r
[[ "$(command -v git)" == "$test_bin/git" ]] || exit 91
script="$1"
shift
source "$script" "$@"'''
    return subprocess.run([BASH, "-c", wrapper, "setup-test", (root / "setup.sh").as_posix(), *arguments],
                          cwd=root.parent, env={**env, **settings},
                          capture_output=True, text=True, encoding="utf-8", timeout=30)


def existing_builder(installation):
    (installation[0] / "Crystal-Builder/.git").mkdir(parents=True)


def test_help_and_dry_run_do_not_write_or_invoke_installation(installation):
    assert run(installation, "--help").returncode == 0
    result = run(installation, "--dry-run", "--with-mcp", "--extra", "api")
    assert result.returncode == 0, result.stderr
    assert "Dry run complete" in result.stdout
    assert "--frozen" in result.stdout and PIN in result.stdout
    root, _, log, _ = installation
    assert not log.exists()
    assert not (root / "Crystal-Builder").exists()
    assert not (root / ".venv-mcp").exists()


@pytest.mark.parametrize("arguments", [
    ("--extra",), ("--extra", "unknown"), ("--unknown",),
    ("--extra", "gnn", "--extra", "llm"),
])
def test_invalid_options_fail_before_commands(installation, arguments):
    assert run(installation, *arguments).returncode != 0
    assert not installation[2].exists()


def test_new_install_pins_dependency_and_selects_all_extras(installation):
    result = run(installation, "--extra", "api", "--extra", "gnn",
                 MOFI_TEST_NO_UPSTREAM="1", MOFI_TEST_HEAD="other", MOFI_TEST_MISSING_COMMIT="1")
    assert result.returncode == 0, result.stderr
    log = installation[2].read_text(encoding="utf-8")
    assert "clone --origin origin https://github.com/harrisonmo/Crystal-Builder.git" in log
    assert "remote add upstream https://github.com/JulesOpp/Crystal-Builder.git" in log
    assert f"fetch --no-tags origin {PIN}" in log
    assert f"checkout --detach {PIN}" in log
    assert "uv sync --frozen --python 3.12 --extra api --extra gnn" in log
    assert "reset" not in log and "clean -" not in log


def test_clean_pinned_checkout_is_reused(installation):
    existing_builder(installation)
    result = run(installation)
    assert result.returncode == 0, result.stderr
    log = installation[2].read_text(encoding="utf-8")
    assert "uv sync --frozen" in log
    assert not any(word in log for word in ("clone --", "checkout --", "fetch --", "remote add"))


@pytest.mark.parametrize("settings", [
    {"MOFI_TEST_DIRTY": " M xtal/core.py"},
    {"MOFI_TEST_ORIGIN": "https://example.invalid/unrelated.git"},
])
def test_existing_dependency_changes_are_preserved(installation, settings):
    existing_builder(installation)
    result = run(installation, **settings)
    assert result.returncode != 0
    log = installation[2].read_text(encoding="utf-8")
    assert "uv " not in log and "checkout --" not in log and "reset" not in log


def test_non_repository_dependency_folder_is_not_overwritten(installation):
    (installation[0] / "Crystal-Builder").mkdir()
    result = run(installation)
    assert result.returncode != 0
    assert not installation[2].exists()


def test_mcp_uses_a_separate_environment(installation):
    result = run(installation, "--with-mcp")
    assert result.returncode == 0, result.stderr
    log = installation[2].read_text(encoding="utf-8")
    assert "uv venv --python 3.12" in log
    assert "uv pip install --python" in log and ".venv-mcp" in log
    assert "packages/mofi-tools" in log


def test_incompatible_workspace_environment_is_not_recreated(installation):
    environment = installation[0] / ".venv"
    environment.mkdir()
    (environment / "pyvenv.cfg").write_text("# Other platform\n", encoding="utf-8")
    result = run(installation)
    assert result.returncode != 0
    assert "separate UV_PROJECT_ENVIRONMENT" in result.stderr
    assert not installation[2].exists()


@pytest.mark.skipif(os.name == "nt", reason="Git Bash uses the separate PowerShell uv installer")
def test_missing_uv_bootstraps_locally_without_network(installation):
    root, tools, log, env = installation
    template = root.parent / "uv-template"
    shutil.copyfile(tools / "uv", template)
    (tools / "uv").unlink()
    for command in ("bash", "dirname", "uname", "mktemp", "rm", "sh", "env", "mkdir", "chmod", "cp"):
        (tools / command).symlink_to(shutil.which(command))
    installer = root.parent / "synthetic-installer.sh"
    executable(installer, '#!/bin/sh\nmkdir -p "$UV_UNMANAGED_INSTALL"\ncp "$MOFI_TEST_UV_TEMPLATE" "$UV_UNMANAGED_INSTALL/uv"\nchmod +x "$UV_UNMANAGED_INSTALL/uv"\n')
    executable(tools / "curl", '#!/usr/bin/env bash\ncp "$MOFI_TEST_INSTALLER" "${@: -1}"\n')
    env.update(PATH=str(tools), MOFI_TEST_UV_TEMPLATE=str(template), MOFI_TEST_INSTALLER=str(installer))
    result = run(installation)
    assert result.returncode == 0, result.stderr
    assert (root / ".tools/uv/uv").is_file()
    assert "uv sync --frozen" in log.read_text(encoding="utf-8")
