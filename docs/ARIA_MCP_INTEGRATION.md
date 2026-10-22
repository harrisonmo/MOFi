# ARIA MCP integration

## Purpose

MOFi can expose ARIA as an optional reasoning tool group alongside scientific
calculators and Valency literature retrieval. ARIA contributes hypotheses,
Processing → Structure → Property explanations, inverse-design suggestions and
knowledge-graph diagnostics. Its generated answers and confidence scores remain
advisory; MOFi's scientific engines and independent evaluator assess outcomes.

The connector uses the official MCP SDK's **stdio** client and the unmodified ARIA
MCP server at revision `cff9080f57360b1e89509adb3f298824eeed809d`:
[ARIA source](https://github.com/yicao-elina/ARIA/tree/cff9080f57360b1e89509adb3f298824eeed809d).
It is separate from Valency's OAuth/HTTP transport and credentials.

## Interfaces

| Model-facing function | ARIA operation | Requirements |
|---|---|---|
| `aria__predict` | Predict from material, processing and target property | Ollama model and cached embedding model |
| `aria__design` | Suggest processing given material, target and constraints | Ollama model and cached embedding model |
| `aria__diagnose` | Inspect knowledge-graph quality | Installed server, graph and cached embedding model; no Ollama required |
| `aria__explain` | Return ARIA's full-mode explanation | Explicit external-search opt-in plus inference dependencies |

Model-facing schemas come from a saved, real server discovery. The controller
removes `kg`, `model` and `mode` from those schemas, rejects extra top-level fields,
and injects its own choices. The model cannot select a filesystem graph path,
server executable, model or mode through a tool call.

The connector also works around the pinned upstream forward method's unused
`target_property` parameter by including the target in the processing dictionary
that reaches routing. ARIA is not modified. Its upstream `aria_search` forward
path attaches literature metadata after answer generation; this integration does
not claim that Valency retrieval automatically changes ARIA's prompts.

## Installation and registration

From the MOFi repository root, with the `aria` extra of the uv project (see
[Python environment](ENVIRONMENT.md)):

```bash
uv sync --extra aria
uv run --no-sync python -m rsi.aria_mcp discover --output runs/aria/catalog.json
uv run --no-sync python -m rsi.aria_mcp list --session runs/aria-session-01
```

The `aria` extra in `pyproject.toml` pins the VCS source and MCP SDK 1.30.0
(formerly `requirements-aria.txt`). The connector checks
installed VCS provenance and fingerprints the installed ARIA Python/JSON files.
A wheel with missing VCS provenance is refused. Discovering starts a fresh local
server and lists its schemas, without invoking prediction or design. Re-discovery
requires a new output filename; its stderr filename must also be unused. Update
`catalog_path` if selecting a different catalog, then start a fresh session.

The shipped `configs/aria-mcp.json` enables predict, design and diagnose in `aria`
mode. With `kg_path: null`, the pinned package's **2D-material demo graph** is used.
That graph is not a MOF knowledge base. Read its scope before interpreting answers.
To use MOFs, supply a reviewed ARIA-format graph through the controller's `kg_path`
(relative to the config file, or absolute), and start a new session.

## Call the tools

Graph diagnostics require the cached `all-MiniLM-L6-v2` embedding model for the
upstream semantic-diversity stage. They do not require Ollama. If that cache is
absent, full `diagnose` fails in offline mode; it is not a purely structural check.
See the no-Ollama verification.

Graph diagnostics:

```bash
python -m rsi.aria_mcp call --session runs/aria-session-01 \
  --tool diagnose --arguments-file examples/aria/diagnose.json
python -m rsi.aria_mcp budget --session runs/aria-session-01
```

Before inference, install/start Ollama and explicitly download the configured
model and embedding model. These commands are operator setup, not automatically
executed by the connector:

```bash
ollama serve
# In another terminal:
ollama pull qwen2:7b
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Both the ARIA factory's reachability check and the child CLI use the local Ollama
service on port 11434. Arbitrary inherited `OLLAMA_HOST` is not forwarded. Embedding
loading in the server uses offline mode, so uncached weights fail rather than
triggering an implicit download. Inference examples use the bundled graph's domain:

```bash
python -m rsi.aria_mcp call --session runs/aria-session-01 \
  --tool predict --arguments-file examples/aria/predict-demo.json
python -m rsi.aria_mcp call --session runs/aria-session-01 \
  --tool design --arguments-file examples/aria/design-demo.json
python -m rsi.aria_mcp export --session runs/aria-session-01 \
  --call-id call-0001 --output runs/aria-evidence.json
```

For `explain`, use a separate operator config with `allow_external_search: true`
and add `explain` to `allowed_tools`. Supply that config via `--config` on all
commands and use a new session. Then `examples/aria/explain-demo.json` supplies the
forward example. ARIA full mode performs its own searches against OpenAlex and
Semantic Scholar. Those searches do not pass through Valency or its quota ledger;
this is why the operation is disabled by default. The same opt-in is required
for `aria_search` and `aria_full` as controller-selected modes.

## Compose with the harness

```python
from rsi.aria_mcp.core import AriaTools
from rsi.literature.valency import ValencyTools
from rsi.literature.grounding import ResearchToolHub
from rsi.science.hub import ScienceToolHub

science = ScienceToolHub("configs/science-hub.json", "runs/study-01/science")
aria = AriaTools("configs/aria-mcp.json", "runs/study-01/aria")
# Optional; requires a separately configured Valency catalog and login:
# literature = ValencyTools("configs/valency.json", "runs/study-01/literature")
hub = ResearchToolHub(science, aria=aria)
# Or: hub = ResearchToolHub(science, literature, aria=aria)
tools = hub.catalog()["tools"]
# Supply tools to the agent controller's function-call interface.
# Route each selected function through hub.call(name, arguments).
```

`AriaTools` also exposes `await acall(name, arguments)` for async controllers.
`catalog()` and `budget()` are local reads. The backend's structured-only legacy
campaigns do not automatically acquire function calling or new external context;
new controllers must route these calls explicitly. The scientific registry,
GCMC implementation, frozen benchmark policies and evaluator are unchanged.

The Evolution Loop may propose improvements using ARIA results; the Research Loop
may request explanations or hypotheses. Both consume returned content as untrusted
advisory data. Neither may substitute an ARIA answer for a GCMC measurement,
energy/force reference or independent acceptance score. Valency literature can
support a reviewed graph, but graph extraction/curation is not automated here.

## Accounting and evidence

- Defaults: 10 attempts, 180 seconds of MCP waiting per attempt, 1800 seconds of
  cumulative reserved waiting allowance, and 262144 returned bytes per attempt.
- Every invocation starts a fresh pinned stdio server. The call and full time
  reservation persist before execution. Failed/interrupted attempts consume both;
  no retries or refunds are made by the connector.
- Upstream ARIA can make several internal LLM calls and retry internally. Their
  counts, tokens and actual Ollama CPU/GPU cost are unknown in this interface and
  are explicitly recorded as null. This is a tool-attempt allowance, not an
  enforced model-token or GPU budget. Freeze and instrument those costs before a
  matched-budget model/RSI experiment.
- The timeout cancels the MCP wait and invokes SDK child-process cleanup, which
  may add shutdown time. Killing the MCP server does not guarantee cancellation of
  work already queued in the independent Ollama daemon. No hard compute or total
  wall-clock containment claim is made.
- The response cap is checked after SDK receipt; it is not a streaming memory cap.
- The ledger binds the config, catalog, graph, installed source, SDK version and
  connector sources. Reopening cannot reset its budget or accept changed bindings.
- Every call snapshots the graph, saves requested/effective arguments, raw MCP
  results, parsed ARIA output, error status, evidence hashes and server stderr.
  ARIA's `{ "error": ... }` application results are failures even if MCP itself
  reports `isError: false`.
- Export rechecks request, result and graph hashes. Only successful results can
  be exported, and the packet is marked `scientific_validation: false`.
- The configured model name is recorded; mutable Ollama model tags and embedding
  caches are not equivalent to frozen weight digests. Admit those artifacts for a
  reproducible scientific comparison.
- Stdio is a local process connection, not an isolation sandbox. No general shell,
  filesystem or model-sampling tools are granted through the MCP client.
