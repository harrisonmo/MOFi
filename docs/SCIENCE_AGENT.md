# Agentic tool-calling loop over the science hub


`rsi/science/agent.py` turns a natural-language task into model-chosen calls on the
[science hub](SCIENCE_HUB.md) and ends with a cited final report:

```
task text -> [model step: one action] -> controller -> ScienceToolHub.call -> envelope -> next model step -> ... -> final_report
```

## Design

- **Backend-neutral.** Each step is one stateless structured decision through the existing
  `run(prompt, schema, output_dir)` contract, so the loop works with the default Codex CLI
  (existing login) and with the `openai_api` / `anthropic_api` backends from
  [API backends](API_BACKENDS.md). Provider-native function calling is not used; the model never
  executes anything and the Codex CLI stays in its read-only, no-tools mode.
- **Step schema.** `reasoning`, `action` (`call_tool` or `final_report`), `tool` (enum of the hub's
  available tools), `arguments_json` (a JSON object string, because tool parameter schemas differ),
  and the report fields `answer`, `evidence_call_ids`, `caveats`.
- **Prompt.** Fixed instructions, the task, the hub catalog (tool definitions with enums and
  "Allowed" ranges, structures with atom counts, unavailable tools with reasons, guidance, budget) and
  the history of actions with their result envelopes, each clipped to `max_observation_chars`.
  The last allowed step tells the model to report.
- **Controller checks.** Unparsable or non-object arguments return an `ActionError` observation and
  never reach the hub; the hub rejects schema and bound violations before creating anything. A report
  is `completed` only if it has an answer and cites at least one call ID, all of which are successful
  tool calls of this run; otherwise `invalid_report`. Other terminal states: `model_error` (any
  backend status other than `ok`), `step_limit`, `token_limit`.
- **Isolated model workspace.** The Codex CLI reads `AGENTS.md` files from its working directory up
  to the repository root, so each model call runs in a fresh scratch directory under the system
  temporary folder (or the optional `model_scratch` config key). The controller refuses a scratch
  root that has `AGENTS.md` or `.git` in it or in any parent folder. After the call it copies the
  call's records to `model/step-NN/` and deletes the scratch directory. `rsi/backends/codex_cli.py`
  is unchanged, because the Q2B/Q2C runners fingerprint it.
- **Ownership.** Backend, hub config, HPC budget and agent limits are controller settings
  (`configs/science-agent.json`, `configs/science-hub.json`), never model inputs.

## Run records

`python -m rsi.science.agent --task "..." --run runs/<name>` writes:

| Path | Content |
|---|---|
| `manifest.json` | task, agent config and hash, hub config hash, credential-free backend identity, hub binding, workspace and Crystal-Builder commits, agent source hash |
| `model/step-NN/` | the backend's own per-call record (prompt, schema, events, final, result) |
| `hub/` | the hub session: `hub-ledger.json`, `call-NNNN/`, `hpc/` |
| `agent-run.json` | status, every step (model status, usage, prompt hash, reasoning, action, tool, arguments, observation), the final report with `unsupported_citations`, usage totals and the hub budget |

Zeo++ on Windows needs `XTAL_ZEOPP` pointing at `scripts/zeopp-wsl.cmd` and `ZEOPP_WSL_BINARY` set
(see [Zeo++ tool](ZEOPP_PORES.md)); otherwise the tool is listed as unavailable and the model is told
why.
