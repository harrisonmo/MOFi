# Functional components and input boundaries

The supported portable entry points are the science hub, bounded science agent,
HPC scheduler, scientific adapter CLIs and optional MCP connectors. Use each
module's `--help` for arguments and the adjacent usage guides for prerequisites.
`configs/science-agent.json` selects GPT-6 Astra through the supported Codex CLI.
Provider SDK backends are opt-in. Authentication remains local.

Reusable workflow, screening, budget, evaluation and model code remains under
the existing `rsi` namespace. GNN training/preparation programs and experimental
micro-MLIP/QLoRA programs remain available under `scripts/` and their packages.
Real training requires independently admitted data, hardware and compute budgets.
Run builders and admission steps before launching a campaign.

Some versioned M/Q/Blind-G1 entry points, historical auditors and campaign
orchestrators require externally supplied prior run directories, evidence
manifests or frozen policy registries. These inputs are deliberately not bundled.
Their integrity checks remain intact. Do not treat a clean clone as a portable
reproduction of earlier campaigns or remove a frozen-hash check to make it run.
Generate new inputs with the appropriate builders or use an independently
provided evidence archive. Start new experiments with fresh output directories.

The test suite retains implementation contracts and synthetic examples. Assertions
whose only purpose was to validate excluded historical results are maintained
with the separate development archive. Optional engine/SDK tests skip when their
dependencies are absent. No historical campaign result is a release fixture.
