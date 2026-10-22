# MOFi

**Materials Optimization through Feedback-Driven Intelligence**

MOFi connects AI research agents to materials construction, simulation,
literature tools, and independently evaluated feedback loops. It provides
bounded execution and reproducible evidence for autonomous materials research.

The Evolution Loop proposes changes to research strategies, tools and model
recipes. The Research Loop applies a versioned candidate in a scientific
environment. An independent evaluator and controller own scoring, budgets and
promotion. See [Architecture](docs/ARCHITECTURE.md).

## Components

| Path | Purpose |
|---|---|
| `rsi/science/` | Scientific tool hub, bounded agent loop and WSL bridge |
| `rsi/hpc/` | Local and Slurm scheduling, resource admission and ledgers |
| `rsi/dft/`, `rsi/orca/`, `rsi/gcmc/`, `rsi/zeopp/` | Scientific engine adapters |
| `rsi/backends/`, `rsi/literature/`, `rsi/aria_mcp/` | Model and advisory tool connectors |
| `rsi/micro_mlip/`, `rsi/jarzynski/`, `rsi/llm/` | Experimental model and training components |
| `rsi/`, `workflows/`, `tools/` | Reusable controllers, workflow policies and tool contracts |
| `configs/`, `examples/`, `benchmarks/`, `datasets/catalogs/` | Bounded settings and small functional inputs |
| `tests/`, `scripts/`, `validation/` | Regression tests, preparation, execution and analysis utilities |

## Install

Python 3.12 and uv are required. Crystal-Builder is a separate, pinned dependency:

```powershell
git clone https://github.com/harrisonmo/MOFi.git
cd MOFi
git clone https://github.com/harrisonmo/Crystal-Builder.git Crystal-Builder
git -C Crystal-Builder checkout df014447f0812dd4f55ed60359b71a981c3eeb60
uv sync --frozen
```

Optional stacks are uv extras: `api`, `valency`, `aria`, `gui`, `gnn`, `openmm`,
`gcmc`, and `llm`. `gnn` and `llm` require different PyTorch builds and cannot be
installed together. See [Environment](docs/ENVIRONMENT.md).

## Use

Inspect the configured tool catalog without making a model call:

```powershell
uv run --no-sync python -m rsi.science.hub list --config configs/science-hub.json --session runs/catalog
```

Run a bounded research task using the existing Codex CLI login:

```powershell
uv run --no-sync python -m rsi.science.agent --agent-config configs/science-agent.json --run runs/my-task --task "Describe the configured MOF database catalog and cite its source metadata."
```

The second command makes model calls. Configure the target and budget first;
the controller records usage and scientific costs. Set up optional engines
before requesting their tools. Catalog metadata is unverified survey information.

## Test

PowerShell, using temporary files outside the repository:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
uv run --no-sync pytest -q --basetemp "$env:TEMP/mofi-pytest"
```

Tests use synthetic inputs and local engine surrogates; optional dependency
tests may skip. No paid model call or full training campaign is required.
The [component guide](docs/COMPONENTS.md) distinguishes reusable functionality
from entry points that require prior campaign evidence.

## Documentation and contribution

See the [documentation index](docs/README.md) and [contributing guide](CONTRIBUTING.md).
Development records, historical experiment outputs and trained weights are
maintained separately. This repository does not assert scientific improvement
from the presence of an implementation or from passing software tests.

Developers: Qiankun Mo, Yi Cao, Xingfei Wei and Junsung Lim.

Original code and documentation use [Apache License 2.0](LICENSE). Attribution
is retained in [NOTICE](NOTICE). Crystal-Builder is MIT-licensed and separately
distributed; see [third-party notices](THIRD_PARTY_NOTICES.md).
