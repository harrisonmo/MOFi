# Valency Bond integration

## Scope and status

MOFi can use Valency Bond as a literature source alongside its scientific tools.
The optional connector implements the official MCP Python SDK's Streamable HTTP
client and OAuth 2.0/PKCE flow. Valency supplies papers, metadata and citations;
MOFi owns scientific execution, compute budgets, evaluation and promotion.

The checked-in config exposes **zero tools** initially. Live tool names and schemas
are discovered after login; none are guessed. All new behavior is opt-in. Existing
frozen experiments and their scientific tools remain separate.

## Where it fits

1. The controller logs in and snapshots the available Bond tool catalog.
2. The operator reviews read-only literature tools and selects them in the config.
3. An agent can select an approved literature function and supply its arguments.
4. MOFi validates arguments, reserves a call, invokes Bond, and stores the result.
5. The controller exports selected successful results as a fixed literature packet.
6. Evolution and Research agents can consume the same packet through the existing
   structured-decision backend interface. Scientific tools execute independently.
7. The independent evaluator uses its registered scores and references. Retrieved
   literature cannot change its labels, acceptance rules or cost accounting.

Literature is useful for proposing hypotheses, selecting published methods and
explaining failures. It is not proof of valid force-field parameters, converged
adsorption capacity, physical stability or improved predictive accuracy. The
corpus includes preprints; do not label every result peer reviewed. This connector
makes no claim about a general real-time materials dataset API.

## Install and log in later

From the repository root, with the `valency` extra of the uv project (see
[Python environment](ENVIRONMENT.md)):

```bash
uv sync --extra valency
uv run --no-sync python -m rsi.literature login
uv run --no-sync python -m rsi.literature discover --output runs/valency/catalog.json
```

`login` opens the system browser for explicit Valency consent and listens only on
`127.0.0.1:8765/callback`, for at most 300 seconds. The official SDK performs dynamic
client registration, PKCE, state verification and token refresh. No password or API
key is requested by MOFi. Later unattended requests never open a login browser.

Credentials are saved to `.valency/oauth.json`, separately from experiment evidence;
`.valency/` is ignored by Git. The file is atomically written with private file mode
on POSIX. On Windows, use an account-private checkout and appropriate filesystem
ACLs. Do not share or commit this directory. `--auth-file` selects a different
operator-owned private location. Remove the local file to forget credentials;
this does not revoke consent at Valency.

The discovery snapshot is immutable for a session. For later discovery use a new
output path and a new session, and update `catalog_path` accordingly. Catalog
listing/initialization uses provider protocol requests; these may count toward
Valency's quota even though they do not consume MOFi's `max_calls` tool budget.

## Select tools and make a bounded call

Inspect `runs/valency/catalog.json`, including each tool's input schema and
annotations. Add exact remote names of reviewed read-only search/detail/citation
tools to `allowed_tools` in `configs/valency.json`. Do not select publishing,
submission, account mutation, messaging or other write operations. Explicit
`readOnlyHint: false` tools are rejected; absent hints require operator review.

The model sees names prefixed with `valency__`; upstream JSON Schemas are kept
unchanged with `strict: false` rather than relabeled as strict schemas.

```bash
python -m rsi.literature list --session runs/valency-session-01
python -m rsi.literature budget --session runs/valency-session-01
```

Create `runs/valency-query.json` with the exact argument structure from the actual
selected tool's discovered schema. Suggested research question: published methods
for methane working capacity in MOFs at 298 K, 65 to 5.8 bar, including force-field
choices and GCMC convergence. No specific tool name or query parameter is assumed.
Then replace `REMOTE_TOOL_NAME` below with that discovered name:

```bash
python -m rsi.literature call --session runs/valency-session-01 \
  --tool REMOTE_TOOL_NAME --arguments-file runs/valency-query.json
python -m rsi.literature export --session runs/valency-session-01 \
  --call-id call-0001 --output runs/valency-context.json
```

The CLI prints the exported packet's SHA-256. A failed call cannot be exported as
successful literature context. Queries are sent to Valency; do not put private
structures, unpublished results or credentials into a query unless their sharing
is authorized for that task.

## Use literature with the current decision interface

For controllers already using `create_backend`, add the following optional section
to their backend config, using the actual packet path and hash printed by export:

```json
{
  "literature": {
    "evidence_path": "runs/valency-context.json",
    "evidence_sha256": "REPLACE_WITH_EXPORTED_SHA256",
    "max_context_bytes": 65536
  }
}
```

This is an extra section, not a complete model config. `create_backend(config)`
returns a `GroundedBackend` around the configured CLI or provider API backend.
Its `run(prompt, schema, output_dir)` interface is unchanged. It checks the packet's
hash on every call, rejects oversized context rather than truncating it, and copies
the packet and provenance into each model output directory. The provider's prompt
log includes the context; existing model token accounting includes it. There are
no implicit literature queries during model decisions. All returned literature is
explicitly marked as untrusted external data, not instructions.

Existing controllers that directly instantiate their own backend do not suddenly
use the factory or receive literature. They may explicitly inject a wrapped backend:

```python
from rsi.backends.factory import create_backend
from rsi.experiment import Experiment

backend = create_backend(config)
experiment = Experiment(config, output="runs/grounded-research-01", backend=backend)
# Use the controller's registered decision/research operations and budgets.
```

The frozen legacy `Experiment.run` entry point has its own CLI-specific manifest
assumptions; it is not upgraded here. Use the backend-injection decision seam or a
new controller with `backend_identity(backend)`. Local model transports can also be
wrapped explicitly with `GroundedBackend`; this does not turn a model into a
network-enabled agent or grant it filesystem access.

## Combine scientific and literature function catalogs

New controllers that already execute model-selected tool calls can compose both:

```python
from rsi.science.hub import ScienceToolHub
from rsi.literature.valency import ValencyTools
from rsi.literature.grounding import ResearchToolHub

science = ScienceToolHub("configs/science-hub.json", "runs/study-01/science")
literature = ValencyTools("configs/valency.json", "runs/study-01/literature")
hub = ResearchToolHub(science, literature)
tools = hub.catalog()["tools"]
# Pass tools to the agent controller's model tool interface.
# Route the returned function name and JSON arguments through:
# result = hub.call(name, arguments)
```

This supplies the catalog/dispatch seam, not a new automatically launched agent
campaign. The legacy structured-only backends still produce their registered
JSON decisions. Async controllers should await `literature.acall` for literature
requests and dispatch scientific jobs through their existing scheduler path.

## Budgets, provenance and limits

- Defaults: 20 tool attempts per session, 60 seconds per attempt including connection,
  at most 262144 bytes of serialized returned data, and 65536 bytes of arguments.
- A reservation is persisted before remote execution. Failure, cancellation and
  interruption consume the attempt. Calls are not automatically retried or refunded.
- The response-size cap is checked after SDK receipt, not a network streaming or
  process memory limit. Oversized results retain their digest and byte count but
  are not exposed as successful context.
- The session ledger binds the config, catalog, connector sources, shared ledger
  helpers and installed MCP version. Changed bindings require a new session.
- Evidence contains timestamped arguments, tool identity, raw MCP content and
  structured results, result hashes and elapsed time. DOI/URL fields are preserved
  as returned; the connector does not invent citations or independently verify them.
- Discovery and login are explicit setup operations; tools/list requests are not
  part of the persistent tool-attempt allowance. Provider quota, pricing and actual
  backend billable usage are not inferred from that allowance.
- Literature requests use their own network-call ledger, separate from HPC
  CPU/GPU accounting. Scientific jobs retain the existing HPC budget.
- Benchmark blindness still matters: choose and freeze the same admissible packet
  for comparison arms before searching or proposing changes. Do not expose held-out
  labels, private benchmark results, provider candidates or prohibited research
  history through queries. Fresh literature is a changed information condition.
- Valency Hub publishing and agent reviews are outside this integration.
