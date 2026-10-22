# Repository tools

| Catalog | Execution interface | Scope |
|---|---|---|
| `q2/`, `q2b/` | Versioned expression/contract runtimes | Frozen tool-evolution experiments |
| `science/methane-gcmc-v1.json` | `rsi.science.tool_registry.ScientificToolRegistry` | Native methane uptake and working capacity |
| `science/zeopp-pores-v1.json` | `rsi.zeopp.registry.ZeoppToolRegistry` | Zeo++ pore geometry, pocket blocking and grid cross-check ([usage](../docs/ZEOPP_PORES.md)) |
| `science/zeopp-pores-v2.json`, `methane-gcmc-v2.json`, `dft-calculation-v1.json`, `orca-calculation-v1.json`, `openmm-minimize-v1.json`, `science-budget-status-v1.json` | `rsi.science.hub.ScienceToolHub` | One catalog; the model chooses structures, methods and parameters within bounds; all jobs run through `rsi.hpc` ([usage](../docs/SCIENCE_HUB.md)) |

Discover registered scientific tool definitions and approved tasks:

```bash
uv run --no-sync python -m rsi.science.tools list
```

The tool name is **`methane_gcmc_capacity`**, version 1. Its only model-supplied
argument is `task_id`; the controller owns structure/force-field paths and
budgets. The default registration is `configs/scientific-tools.json`. See
[GCMC usage](../docs/GCMC_METHANE.md#registered-scientific-tool) for dispatch,
structured results and persistent session limits. This is a repository tool
interface for an agent controller, not a globally installed Codex/MCP plugin.

The unified hub lists every scientific tool with model-chosen parameters:

```bash
python -m rsi.science.hub list
```
