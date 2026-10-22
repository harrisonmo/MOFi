# MOFi tools

`mofi-tools` provides a local Model Context Protocol server for MOFi's bounded
scientific tools. The unrelated PyPI package named `mofi` is not this project.

<!-- mcp-name: io.github.harrisonmo/mofi -->

## Install from PyPI

```bash
python -m pip install mofi-tools==0.1.0
mofi-tools catalog
mofi-mcp
```

With uv, run the stdio server using `uvx --from mofi-tools==0.1.0 mofi-tools serve`.
The public MCP Registry identity is `io.github.harrisonmo/mofi`; its package
metadata is maintained in the repository's `server.json`.

## Install from source

Python 3.12 or newer is required:

```bash
python -m pip install ./packages/mofi-tools
mofi-tools catalog
mofi-tools query --arguments '{"name_contains":"QMOF"}'
mofi-mcp
```

The standalone profile provides `mof_database_catalog`, `mofi_tool_catalog`, and
the `mofi://catalog` resource. It bundles the byte-pinned CSV and reviewed sidecar.
The table is unverified survey metadata, not MOF structures or reference labels.
No model login, model call, engine installation or GPU is needed for this profile.

## Scientific workspace

Point the installed server at a separately prepared MOFi source checkout:

```bash
mofi-mcp --workspace /path/to/MOFi \
  --science-python /path/to/science-environment/bin/python \
  --hub-config configs/science-hub.json --session runs/mcp-client
```

Only configured and available hub tools are advertised. The adapter reuses the
hub's schemas, scientific validation, scheduler, budget reservations and evidence
ledgers. The science interpreter needs the workspace's scientific dependencies;
Crystal-Builder, ARIA and external engines are not bundled in this package.
Give each client a separate session directory. Tool calls may run calculations
within the operator's configured budget; catalog inspection makes no model calls.

Standard output carries MCP messages only. Diagnostics and engine output use
standard error. Native jobs keep the hub's existing interruption accounting; a
client timeout does not prove that a calculation stopped or refund its budget.

See [MCP setup and packaging](https://github.com/harrisonmo/MOFi/blob/main/docs/MCP_TOOLS.md)
for Codex, Claude Code, Claude Desktop, building and release steps.

Original code uses Apache-2.0; see LICENSE and NOTICE. Scientific dependencies
retain their own licenses. No credentials, model weights, engine binaries,
development records or scientific run output are part of the distribution.
