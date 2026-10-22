# Environment

The project uses Python 3.12 and uv. Clone Crystal-Builder alongside the source
as `Crystal-Builder/` at commit `df014447f0812dd4f55ed60359b71a981c3eeb60`,
then run `uv sync --frozen`. The virtual project imports `rsi` from the working
tree; Crystal-Builder is installed editable from its independent checkout.

| Extra | Purpose |
|---|---|
| `api` | Optional provider SDK backends |
| `valency` | MCP literature client |
| `aria` | Pinned ARIA MCP reasoning server |
| `gui` | Crystal-Builder desktop GUI |
| `gnn` | GNN/Jarzynski stack, PyTorch CUDA 12.8 build |
| `openmm` | OpenMM science engine |
| `gcmc` | RASPA3 Python bindings |
| `llm` | Experimental QLoRA stack, PyTorch CUDA 12.4 build |

Pass all desired extras together, for example `uv sync --frozen --extra api
--extra gnn`. The `gnn` and `llm` extras conflict. After selecting extras use
`uv run --no-sync ...` to avoid changing the selected environment implicitly.
Use a separate Linux environment for WSL; never execute the Windows `.venv`
inside Linux. GPAW, CP2K, Zeo++ and ORCA need separate installation.

The two root requirement files remain compatibility inputs for source-fingerprint
and environment checks; `pyproject.toml` and `uv.lock` are the installation inputs.

## WSL

Set `SCIENCE_HUB_PYTHON` to the Linux science Python executable. Optionally set
`SCIENCE_HUB_CP2K_ENV`, `XTAL_ZEOPP`, `RSI_ORCA` and `CP2K_DATA_DIR` for engines
installed outside PATH. `scripts/science-hub-wsl.sh` adds the source/dependency
to PYTHONPATH. The Windows launcher uses the root `.venv` and keeps model
authentication on Windows. It never copies authentication into WSL.

For the separate RASPA CLI wrappers, set `RASPA3_WSL_HOME` or
`RASPA3_WSL_PYTHON`. `ZEOPP_WSL_BINARY` selects the Linux Zeo++ binary for the
Windows Zeo++ wrapper. The optional `*_WSL_DISTRO` variables default to Ubuntu.
Executable paths are deployment settings; no personal home paths are supplied.
