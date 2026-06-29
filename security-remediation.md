# Task: remediate security findings in agent-assistant

You are working in a fresh clone of `agent-assistant`. A whole-system security
review (memory feature + core, model adapters, MCP layer, subagents,
telemetry) produced the findings below. This project runs in **hardened,
locked-down containers** in production: read-only rootfs, `cap_drop: [ALL]`,
`no-new-privileges`, an `internal: true` (no internet route) network for MCP
servers, and a default-deny tool-permission allowlist.

**Already fixed — do NOT redo:** `.env` baked into the Docker image (resolved
via `.dockerignore`).

This supersedes any earlier `memory-security-remediation.md`.

## Read first (orient before changing anything)

`README.md` (esp. "Containers / hardening notes"); `agent/core/loop.py`,
`agent/core/entrypoint.py`, `agent/core/events.py`, `agent/core/memory.py`,
`agent/core/interfaces.py`; `agent/memory/*`; `agent/models/*`; `agent/mcp/*`;
`agent/agents/*`; `agent/observability/*`; `agent/config.py`,
`agent/composition.py`; `deploy/compose.yaml`, `deploy/agent.container.toml`,
`Dockerfile`; `agent.toml`, `agents/*.toml`; `scripts/consolidate.py`;
`evals/spec.py`, `evals/scorers.py`, `evals/bridge.py`, `evals/cases/*.jsonl`.

## Non-negotiable constraints (every change)

1. **`make check` must stay green**: ruff, `ruff format --check`, pyright
   (strict), pytest at **≥90% coverage**. Run it before declaring done.
2. **Core purity**: `agent/core/*` must not import `agent/composition.py`,
   `agent/config.py`, any provider SDK, or anything under `agent/memory/`. New
   core-facing surface = a Protocol + plain data type only; concrete impls live
   outside core, wired in `agent/composition.py`.
3. **AI reasons, a deterministic layer executes** — never let a model free-write
   durable memory or pick a filesystem path / destination host unchecked.
4. **Evals are data**: for behavioural findings (injection, gating, routing)
   add `evals/cases/*.jsonl` (+ replay cassettes if needed) exercising the fix
   through the unchanged `run_agent` path. Match existing case style.
5. Match surrounding style/typing/docstrings. Keep diffs minimal and focused.

Two cross-cutting root causes to keep in mind: (a) **credentials/data sent to a
config/env-controlled destination with no allowlist**; (b) **a model-controlled
value flowing into a trusted position**. Several findings are instances of these
— fixing the pattern is better than patching one call site.

Each section below is an independently shippable PR. Do them in order; 1 and 2
are highest value.

---

## PR 1 — Model-adapter credential scoping (HIGH) + robustness

**1a (HIGH). Credentials broadcast to every `base_url`.**
`agent/models/openai_compat.py:53-61`. The ALFA/OpenAI key and
`x-alfa-authorization` are set as `default_headers` on the `AsyncOpenAI` client,
so they ride on every request to whatever `base_url` is configured — including
local llama-server/vLLM endpoints. Also: `Authorization: f"Bearer sk-{api_key}"`
is malformed when the key already starts with `sk-` or is `None`
(`composition.py:40` can pass `None`); `x-alfa-authorization` falls back to the
literal `"empty"`.
**Fix:** build headers conditionally — attach `Authorization` only when
`api_key` is set, value `f"Bearer {api_key}"` (no `sk-` prefix); attach
`x-alfa-authorization` only when `api_secret` is set; scope the ALFA header so it
is only sent to the ALFA host. Validate `base_url` against a trusted-host
allowlist (config-driven) before sending any credential.
**Acceptance:** tests prove (i) no `Authorization`/`x-alfa-*` header is sent when
the corresponding secret is absent; (ii) the key is passed verbatim;
(iii) a non-allowlisted `base_url` is rejected (or credentials withheld).

**1b (MEDIUM). Unbounded model-output accumulation (DoS).**
`agent/models/prompted_tools.py:74,88`, `openai_compat.py:79-128`. Response text
and tool-arg fragments accumulate with no cap; a misbehaving local model can
drive memory growth and a regex pass over an arbitrarily large buffer.
**Fix:** enforce a hard char/byte ceiling on accumulated stream content and abort
the stream when exceeded. **Acceptance:** test that an over-long stream aborts.

**1c (LOW). Uncaught `json.loads` on model tool args.**
`openai_compat.py:127`, `anthropic.py:109` raise `JSONDecodeError` mid-stream on
malformed args (single-turn DoS). `prompted_tools.py` already catches — leave it.
**Fix:** wrap so a bad payload becomes a tool-error result, not a crash.

---

## PR 2 — Telemetry hardening (HIGH)

Context (verified): `OtelSink` exports only metadata (token counts, model id,
tool names, ids, a SHA-256 messages digest) — no prompts/args/results. The risk
is the destination and defaults, not the payload.

**2a (HIGH). Fail-open export default.** `agent/config.py:85` —
`OtelConfig.enabled: bool = True`. Any config built without `agent.toml`'s
`enabled=false` exports by default. **Fix:** default `enabled = False`; require
explicit opt-in.

**2b (HIGH). Unvalidated env-driven endpoint + auth over plaintext.**
`agent/config.py:86-87`, `agent/observability/otel.py:23-26`.
`AGENT_OTEL__ENDPOINT` and `AGENT_OTEL__HEADERS__AUTHORIZATION` are env-settable
with no scheme/host validation; prod compose uses plaintext `http://`.
**Fix:** require `https`/TLS for any non-loopback endpoint and refuse to send the
auth header to a plaintext non-loopback host; optionally validate the endpoint
host against an allowlist. **Acceptance:** tests for the reject paths.

**2c (MEDIUM). OTLP credential in broadly-inherited container env.**
`deploy/compose.yaml:35-36`. Readable via `/proc/self/environ`, inherited by
spawned subprocesses. **Fix:** deliver via a file/secret mount read at startup,
not a plain env var. Document in README.

**2d (LOW). `Error` span event ships free-text `message`.**
`agent/observability/sink.py:264`. Safe today (only emitter is content-free),
but the one OTel path that could exfiltrate a future interpolated string.
**Fix:** truncate/constrain `Error.message` before attaching; add a test/comment
asserting no content is interpolated.

---

## PR 3 — Hardened deployment for the memory MCP servers (HIGH)

`deploy/agent.container.toml` has no `[memory]` section (memory off in the
container) and `deploy/compose.yaml` has no `markdown-rag` / `obsidian-mcp-guard`
services or vault volume. Naively enabling memory breaks hardening: writable
vault vs `read_only`; a RAG embedding API vs the no-egress `mcp-internal` net;
no hardening on the two third-party servers (highest risk — they touch the FS).
**Do:** add hardened compose services for both memory servers mirroring
`mcp-echo-clock` (`read_only`, `tmpfs:[/tmp]`, `cap_drop:[ALL]`,
`no-new-privileges`, non-root, on `mcp-internal`); mount **one** writable vault
volume scoped to the vault path, on the memory servers only (agent stays
read-only); add the `streamable_http` `[memory]` wiring to
`agent.container.toml`. **Resolve the embedding-egress decision explicitly**
(prefer local/in-container embeddings so the server stays on `mcp-internal`;
document any deliberate egress exception in the README).
**Open question to surface, not guess:** how `markdown-rag` (sibling repo
`../markdown-rag`) and `obsidian-mcp-guard` (`uvx`) become container images.
**Acceptance:** `docker compose up` runs with memory enabled, agent still
read-only, memory servers hardened on `mcp-internal`, egress posture documented.

---

## PR 4 — Memory write-path & recall safety

**4a (MEDIUM). Path traversal via model-controlled `target_fact_id`.**
`agent/memory/consolidation.py:215-226`. `fact_id` from the consolidator model
is interpolated into `f"{semantic_folder}/{fact_id}.md"`; `../../etc/x` escapes
the folder. (Episodic path uses a server `uuid` — safe; match that.)
**Fix:** validate `target_fact_id` against a strict pattern (UUID or
`^[A-Za-z0-9_-]+$`) and/or assert the resolved path stays within
`semantic_folder`; reject otherwise. Don't rely on `obsidian-mcp-guard`.
**Acceptance:** test that traversal/absolute/empty ids never write outside
`semantic_folder`.

**4b (MEDIUM). Recalled memory injected into the system prompt incl. untrusted
content.** `agent/core/loop.py:82-88` appends recalled memories (which can carry
`TOOL_OUTPUT` provenance via `RagMemoryProvider`) to the system prompt.
**Fix:** move recalled memories into a clearly-delimited lower-trust
context/user message; consider excluding/quarantining `TOOL_OUTPUT`-provenance
episodic content from recall (`agent/memory/provider.py`). Keep the provenance
label. Keep core pure. **Acceptance:** a `prompt_injection` eval where a recalled
memory contains an embedded instruction shows the agent does not follow it.

**4c (MEDIUM). Memory I/O bypasses the permission allowlist and audit trail.**
`agent/memory/store.py:143,183`, `sink.py`. Memory reads/writes call the MCP
connection directly, outside `AllowlistPolicy`/`PermissionDecided`.
**Fix:** emit transcript/audit events for memory writes (add event type(s) in
`agent/core/events.py`, keep core pure); add a config switch to disable writes
(recall-only). **Acceptance:** writes emit an event (tested); write-disable
honoured (tested).

**4d (LOW). No redaction before persistence.** `agent/memory/distiller.py`
writes user goal + tool outputs verbatim into durable notes.
**Fix:** add a deterministic redaction pass (mask bearer tokens/API keys/emails)
or restrict distillation to an allowlist of fields. **Acceptance:** test a token/
email is redacted in the persisted record.

---

## PR 5 — MCP routing & server exposure

**5a (MEDIUM). Tool-name collision mis-routes across servers, bypassing the
allowlist.** `agent/mcp/registry.py:30-36`. Flat `_tool_to_server` dict, no
collision check; last server connected wins, and `AllowRule` keys on
`(server, tool)`. **Fix:** raise `ValueError` on duplicate `tool.name` in
`connect()` (mirror `CompositeToolRegistry`'s `seen` check) so collisions fail
loudly at startup. **Acceptance:** test that two servers sharing a tool name
fail at construction.

**5b (MEDIUM). Unauthenticated `0.0.0.0` MCP servers, DNS-rebinding disabled, no
defense-in-depth.** `mcp_servers/_runtime.py:26-38`. Sole control is the
internal network. **Fix:** don't default `MCP_HOST` to `0.0.0.0` (require it set
explicitly for HTTP transport) and/or add an env-injected shared-secret/bearer
check; at minimum re-enable rebinding protection with an explicit `allowed_hosts`
allowlist of the expected service DNS names. **Acceptance:** server rejects
requests without the secret / from non-allowlisted hosts (tested).

**5c (LOW). No input-size bound on bundled tools.**
`mcp_servers/wordcount/server.py`, `echo_clock/server.py`. **Fix:** bound
`len(text)` (Pydantic `Field(max_length=...)` or explicit guard returning an
error). **Acceptance:** over-long input returns an error, not OOM.

**5d (LOW). Full `os.environ` to stdio MCP subprocesses.**
`agent/mcp/client.py:65-69` hands all API keys to third-party stdio servers
(`markdown-rag`, `uvx obsidian-mcp-guard`). **Fix:** add an optional
`env`/`env_passthrough` field to `MCPServerConfig` (`agent/config.py`); default
to a minimal set (`HOME`, `PATH`) plus declared extras. **Acceptance:** test a
stdio server receives only its declared vars, not unrelated secrets.

---

## PR 6 — Subagent provenance & budget

**6a (MEDIUM). Downward prompt-injection across the subagent boundary.**
`agent/agents/subagent_tools.py:182-187`. The parent model's `task` argument is
injected as the child's user turn tagged `Provenance.USER_STATED`, so tainted
content the parent ingested arrives as a trusted instruction in the child.
(Return-path provenance composition is correct; only the downward path lacks
tagging.) **Fix:** thread the parent's effective/inbound provenance into the
delegated `Task` so a tainted parent context produces a `TOOL_OUTPUT`-derived
(untrusted) child turn. **Acceptance:** eval/test showing a delegated task built
from tainted parent content is not treated as trusted in the child.

**6b (LOW). Loose budget; no explicit fan-out count cap.**
`agent/agents/subagent_tools.py:151,199`. `budget.consume(allocated)` charges the
child's full `max_steps` regardless of actual usage (conservative, not an
exhaustion risk, but kills honest children early); there's no fan-out *count*
cap independent of steps. **Fix:** charge actual steps used from the child
result; add an explicit per-tree max-subagent-invocations counter.
**Acceptance:** test honest children aren't over-charged; invocation cap enforced.

**6c (LOW, docs). Child perms not constrained to parent's.** Child `run_agent`
runs under its own TOML allowlist, not an intersection with the parent's. This
is operator-controlled (not model-driven escalation) but unenforced.
**Fix:** document as an explicit trust assumption, and/or optionally assert
child-perms ⊆ parent-perms at registry build time.

---

## Verified clean (no action — recorded so coverage is auditable)
- Skills/agents loading: fixed single-level globs, `yaml.safe_load`, no
  model-influenced path traversal; `_resolve_dir` only handles operator config.
- Subagent depth + cycle guards: sound, fail-closed.
- Registry agent-name lookup: names come only from trusted TOML, not model output.
- `CompositeToolRegistry` rejects tool-name shadowing at construction.
- No `eval`/`exec`/`pickle`/`yaml.load` in scope; no secret logging at
  startup/`--help`; CI workflow references no secrets.
- Tool calls from the prompted-text parser still pass `permissions.evaluate` —
  the parser cannot bypass the gate (schema-validation of parsed args would be
  defense-in-depth only).
- OTel spans carry only metadata (no prompts/args/results); `messages_digest` is
  a SHA-256 prefix.

## When done
- `make check` green (ruff, format-check, pyright strict, pytest ≥90%).
- New eval cases pass through the unchanged `run_agent` path.
- Summarise each PR and its acceptance evidence; keep PRs scoped one-per-section.
