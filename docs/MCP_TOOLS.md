# MCP tools and Python packaging

MOFi supplies a stdio MCP server and a separate Python distribution project,
`packages/mofi-tools/`. The distribution name is **mofi-tools**, version
**0.1.0**, and the public MCP identity is **io.github.harrisonmo/mofi**. The
unrelated [mofi package on PyPI](https://pypi.org/project/mofi/) is not MOFi.
Version **0.1.0** is published on [PyPI](https://pypi.org/project/mofi-tools/0.1.0/)
and active in the [MCP Registry](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.harrisonmo%2Fmofi/versions/latest).
Publication is manually triggered; setup never uploads packages.

## Install and inspect

Use Python 3.12 or newer to install a published release:

```bash
python -m pip install mofi-tools==0.1.0
mofi-tools --version
mofi-tools catalog
mofi-tools query --arguments '{"name_contains":"QMOF"}'
```

This installs only the portable catalog and MCP adapter dependencies. It does
not change the root uv project or install any simulator, checkpoint or model
backend. `mofi-mcp` and `python -m mofi_tools serve` start the same stdio server.
The package is independent of the current working directory.

For source installation, run `python -m pip install ./packages/mofi-tools` from
the cloned repository. With uv, start the standalone MCP profile using:

```bash
uvx --from mofi-tools==0.1.0 mofi-tools serve
```

The standalone profile provides:

| Interface | Purpose |
|---|---|
| `mof_database_catalog` | Filter the 26-row, byte-pinned database survey |
| `mofi_tool_catalog` | Inspect available tools, missing engines and guidance |
| `mofi://catalog` | Read the tool catalog as an MCP JSON resource |

The CSV is source-selection metadata. Its names, links, coverage, size and
license fields have not been verified against the external sources. It contains
no downloadable database structures or scientific reference labels.

## Connect the scientific workspace

Prepare the existing root science environment and pinned Crystal-Builder
checkout as described in [Environment](ENVIRONMENT.md). Then launch:

```bash
mofi-mcp --workspace /absolute/path/to/MOFi \
  --science-python /absolute/path/to/science-environment/bin/python \
  --hub-config configs/science-hub.json --session runs/mcp-client
```

On Windows, the science interpreter normally ends in `.venv/Scripts/python.exe`.
All operator paths are resolved explicitly; they are never tool arguments.
The adapter invokes `rsi.science.hub_bridge` in that environment. It preserves
the existing schemas, allowed structures and methods, scheduler, budget ledger,
scientific checks and hashed evidence. A changed hub binding requires a fresh
session. Configure limits before exposing scientific tools to a client.

Available tools depend on installed engines and libraries: DFT/EMT, methane
GCMC, Zeo++, molecular ORCA, OpenMM, database catalog and budget status. Missing
engines are reported in the catalog and are not advertised as callable tools.
Crystal-Builder, ARIA and engine binaries are separate installations, not PyPI
dependencies of this lightweight adapter. This release uses a local science
interpreter; it does not automatically start the WSL launcher.

Use a separate session directory for Codex, Claude Code and Claude Desktop.
`mofi_tool_catalog` is an initial registration snapshot; `science_budget_status`
returns live usage. A client disconnect or timeout does not prove that an engine
stopped or refund a reservation. Inspect the persistent hub ledger after an
interruption. Scientific wall-time limits remain owned by the existing hub.

## Register with Codex

Use the Python interpreter containing the installed adapter:

```bash
codex mcp add mofi -- /absolute/path/to/mcp-python -m mofi_tools serve
codex mcp get mofi
```

To enable scientific tools, append `--workspace`, `--science-python`,
`--hub-config` and `--session` as above. For long calculations, set suitable
`startup_timeout_sec` and `tool_timeout_sec` in `[mcp_servers.mofi]`; these
client timeouts do not replace the scientific budget. Reload the client or
start a new session after registration. See the
[official Codex MCP configuration](https://developers.openai.com/codex/mcp).

## Register with Claude

Claude Code, at user scope:

```bash
claude mcp add --scope user --transport stdio mofi -- /absolute/path/to/mcp-python -m mofi_tools serve
claude mcp get mofi
```

For Claude Desktop, merge an `mofi` entry into `mcpServers` in the existing
desktop configuration, retaining all other entries:

```json
{
  "mcpServers": {
    "mofi": {
      "command": "/absolute/path/to/mcp-python",
      "args": ["-m", "mofi_tools", "serve"]
    }
  }
}
```

Append the workspace settings for scientific tools and use a distinct session
path per client. Restart Claude Desktop to load its updated configuration.
See [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

## Build and verify without publishing

```bash
python scripts/sync_mcp_catalog.py --check
python -m pip install build twine pytest
python -m build packages/mofi-tools --outdir /path/to/private-build-output
python -m twine check /path/to/private-build-output/*
python -m pytest -c packages/mofi-tools/pyproject.toml packages/mofi-tools/tests -q
python scripts/verify_mcp_installation.py --python /path/to/installed-wheel-python
```

The wheel and source distribution include the portable parser, MCP adapter,
CSV, hash annotations, tool schema and Apache notices. They exclude the source
workspace, credentials, environments, run records, trained models and scientific
engine binaries. The asset synchronization check prevents the small packaged
catalog snapshot from drifting from the workspace source. Install and exercise
the built wheel outside the repository before publication.

## Public releases

The manually dispatched [release workflow](../.github/workflows/release.yml)
builds and tests the distribution, checks metadata and starts the installed wheel
over MCP from outside the source tree. Only `main` can run a release.

Before the first PyPI upload, the account owner must add a pending GitHub Trusted
Publisher under PyPI account settings with these exact fields:

| Field | Value |
|---|---|
| PyPI project | `mofi-tools` |
| GitHub owner | `harrisonmo` |
| Repository | `MOFi` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

Choose `validate` to build without publishing, or `pypi-and-mcp` to upload the
validated artifacts and then publish `server.json` to the official MCP Registry.
The PyPI job uses Trusted Publishing and the MCP job uses GitHub OIDC; neither
needs a stored upload token. `mcp-only` retries registry publication after a
successful PyPI upload. Registry validation checks the ownership marker embedded
in the package description. The publisher binary is pinned to version 1.8.1 and
its SHA-256 is checked before execution.

For later versions, update `pyproject.toml`, `mofi_tools.__version__`, `server.json`
and the versioned installation examples together. Run
`python scripts/check_release_metadata.py` before dispatch. Published package
files cannot be replaced; use a new version for package changes. A rerun skips
already uploaded PyPI artifacts, while MCP versions must be checked before
retrying publication. See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
and [MCP publishing](https://modelcontextprotocol.io/registry/quickstart).
