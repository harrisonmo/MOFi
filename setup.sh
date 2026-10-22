#!/usr/bin/env bash
# Install the locked MOFi workspace without running models or simulations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BUILDER="$ROOT/Crystal-Builder"
BUILDER_URL="https://github.com/harrisonmo/Crystal-Builder.git"
UPSTREAM_URL="https://github.com/JulesOpp/Crystal-Builder.git"
BUILDER_COMMIT="df014447f0812dd4f55ed60359b71a981c3eeb60"
DRY_RUN=0
WITH_MCP=0
HAS_GNN=0
HAS_LLM=0
SYNC_ARGS=(sync --frozen --python 3.12)

usage() {
    cat <<'USAGE'
Usage: bash setup.sh [--extra NAME] [--with-mcp] [--dry-run]

Install the pinned Crystal-Builder dependency and locked Python 3.12 workspace.
An existing uv installation is reused; otherwise uv is installed locally in
.tools/uv without changing shell profiles. Git and network access are required.

  --extra NAME  Add api, valency, aria, gui, gnn, openmm, gcmc or llm; repeatable.
  --with-mcp    Install the portable MCP package in a separate .venv-mcp.
  --dry-run     Print the installation commands without changing files or using the network.
  -h, --help    Show this help.

gnn and llm cannot be installed together. Pass all desired extras on each run;
uv sync makes the workspace environment match the requested set.
USAGE
}

fail() { printf 'Setup stopped: %s\n' "$*" >&2; exit 1; }
run() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extra)
            [[ $# -ge 2 ]] || fail "--extra needs a name"
            case "$2" in
                api|valency|aria|gui|gnn|openmm|gcmc|llm) ;;
                *) fail "Unknown extra: $2" ;;
            esac
            [[ "$2" != gnn ]] || HAS_GNN=1
            [[ "$2" != llm ]] || HAS_LLM=1
            SYNC_ARGS+=(--extra "$2")
            shift 2 ;;
        --with-mcp) WITH_MCP=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1; use --help" ;;
    esac
done
[[ "$HAS_GNN" == 0 || "$HAS_LLM" == 0 ]] || fail "gnn and llm extras conflict"
command -v git >/dev/null 2>&1 || fail "Install Git before running setup"
[[ -f "$ROOT/pyproject.toml" && -f "$ROOT/uv.lock" ]] || fail "Run the setup.sh from a complete MOFi checkout"
cd "$ROOT"

# Preflight existing checkouts and environments before any installation writes.
if [[ -e "$BUILDER" ]]; then
    [[ -e "$BUILDER/.git" ]] || fail "Crystal-Builder exists but is not an independent Git checkout"
    BUILDER_TOP="$(git -C "$BUILDER" rev-parse --show-toplevel)"
    [[ "$(cd "$BUILDER_TOP" && pwd -P)" == "$BUILDER" ]] || fail "Crystal-Builder is not the dependency repository root"
    [[ -z "$(git -C "$BUILDER" status --porcelain)" ]] || fail "Crystal-Builder has local changes; commit or preserve them before setup"
    ORIGIN="$(git -C "$BUILDER" remote get-url origin)"
    case "$ORIGIN" in
        https://github.com/harrisonmo/Crystal-Builder|https://github.com/harrisonmo/Crystal-Builder.git|git@github.com:harrisonmo/Crystal-Builder.git) ;;
        *) fail "Crystal-Builder origin must reference harrisonmo/Crystal-Builder" ;;
    esac
    if UPSTREAM="$(git -C "$BUILDER" remote get-url upstream 2>/dev/null)"; then
        case "$UPSTREAM" in
            https://github.com/JulesOpp/Crystal-Builder|https://github.com/JulesOpp/Crystal-Builder.git|git@github.com:JulesOpp/Crystal-Builder.git) ;;
            *) fail "Crystal-Builder upstream must reference JulesOpp/Crystal-Builder" ;;
        esac
    fi
fi

PLATFORM="$(uname -s)"
WORKSPACE_ENV="${UV_PROJECT_ENVIRONMENT:-$ROOT/.venv}"
if [[ -d "$WORKSPACE_ENV" ]]; then
    [[ -f "$WORKSPACE_ENV/pyvenv.cfg" ]] || fail "The workspace environment directory is not a virtual environment"
    case "$PLATFORM" in
        MINGW*|MSYS*|CYGWIN*) [[ -f "$WORKSPACE_ENV/Scripts/python.exe" ]] || fail "Use a separate UV_PROJECT_ENVIRONMENT for Windows" ;;
        *) [[ -f "$WORKSPACE_ENV/bin/python" ]] || fail "Use a separate UV_PROJECT_ENVIRONMENT for Linux/macOS/WSL" ;;
    esac
fi
MCP_ENV="$ROOT/.venv-mcp"
if [[ "$WITH_MCP" == 1 ]]; then
    [[ -f "$ROOT/packages/mofi-tools/pyproject.toml" ]] || fail "The MCP package source is missing"
    if [[ -e "$MCP_ENV" ]]; then
        [[ -f "$MCP_ENV/pyvenv.cfg" ]] || fail ".venv-mcp exists but is not a virtual environment"
        case "$PLATFORM" in
            MINGW*|MSYS*|CYGWIN*) [[ -f "$MCP_ENV/Scripts/python.exe" ]] || fail "The MCP environment belongs to a different platform" ;;
            *) [[ -f "$MCP_ENV/bin/python" ]] || fail "The MCP environment belongs to a different platform" ;;
        esac
    fi
fi

UV_DIR="$ROOT/.tools/uv"
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [[ -x "$UV_DIR/uv" ]]; then
    UV="$UV_DIR/uv"
elif [[ -x "$UV_DIR/uv.exe" ]]; then
    UV="$UV_DIR/uv.exe"
else
    UV="$UV_DIR/uv"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'Would install uv from https://astral.sh/uv into %s (no shell profile changes).\n' "$UV_DIR"
    else
        case "$PLATFORM" in
            MINGW*|MSYS*|CYGWIN*)
                command -v powershell.exe >/dev/null 2>&1 || fail "Install uv or make PowerShell available in Git Bash"
                command -v cygpath >/dev/null 2>&1 || fail "Install uv or use a Git Bash environment with cygpath"
                env MOFI_UV_INSTALL_DIR="$(cygpath -w "$UV_DIR")" powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
                    '$env:UV_UNMANAGED_INSTALL = $env:MOFI_UV_INSTALL_DIR; Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression'
                UV="$UV_DIR/uv.exe" ;;
            *)
                INSTALLER="$(mktemp)"
                trap 'rm -f -- "$INSTALLER"' EXIT
                if command -v curl >/dev/null 2>&1; then
                    curl -LsSf https://astral.sh/uv/install.sh -o "$INSTALLER"
                elif command -v wget >/dev/null 2>&1; then
                    wget -q https://astral.sh/uv/install.sh -O "$INSTALLER"
                else
                    fail "Install curl, wget or uv before running setup"
                fi
                env UV_UNMANAGED_INSTALL="$UV_DIR" sh "$INSTALLER" ;;
        esac
        [[ -x "$UV" ]] || fail "uv installation did not create the expected executable"
    fi
fi

printf 'Preparing pinned Crystal-Builder dependency...\n'
if [[ ! -e "$BUILDER" ]]; then
    run git clone --origin origin "$BUILDER_URL" "$BUILDER"
fi
if [[ "$DRY_RUN" == 1 && ! -e "$BUILDER" ]]; then
    run git -C "$BUILDER" remote add upstream "$UPSTREAM_URL"
    run git -C "$BUILDER" checkout --detach "$BUILDER_COMMIT"
else
    if ! git -C "$BUILDER" remote get-url upstream >/dev/null 2>&1; then
        run git -C "$BUILDER" remote add upstream "$UPSTREAM_URL"
    fi
    if ! git -C "$BUILDER" cat-file -e "$BUILDER_COMMIT^{commit}" 2>/dev/null; then
        run git -C "$BUILDER" fetch --no-tags origin "$BUILDER_COMMIT"
    fi
    if [[ "$(git -C "$BUILDER" rev-parse HEAD)" != "$BUILDER_COMMIT" ]]; then
        run git -C "$BUILDER" checkout --detach "$BUILDER_COMMIT"
    fi
fi

printf 'Installing the locked Python 3.12 workspace...\n'
run "$UV" "${SYNC_ARGS[@]}"
if [[ "$WITH_MCP" == 1 ]]; then
    if [[ ! -e "$MCP_ENV" ]]; then run "$UV" venv --python 3.12 "$MCP_ENV"; fi
    case "$PLATFORM" in
        MINGW*|MSYS*|CYGWIN*) MCP_PYTHON="$MCP_ENV/Scripts/python.exe" ;;
        *) MCP_PYTHON="$MCP_ENV/bin/python" ;;
    esac
    run "$UV" pip install --python "$MCP_PYTHON" "$ROOT/packages/mofi-tools"
fi

if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Dry run complete. No installation commands were executed.\n'
else
    printf 'Setup complete. No models, scientific calculations or publishing commands were run.\n'
    printf 'Run workspace commands with: %q run --no-sync ...\n' "$UV"
    if [[ "$WITH_MCP" == 1 ]]; then
        printf 'Inspect MCP tools with: %q -m mofi_tools catalog\n' "$MCP_PYTHON"
    fi
fi
