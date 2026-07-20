# Market Analysis Agent — Design

**Date:** 2026-07-19
**Status:** Approved (pending user review of this document)
**Context:** Technical test "Développeur IA — Agent d'analyse de marché e-commerce". Coding scope = steps 1–3 (architecture, tools, tests); steps 4–7 are theory questions answered in the README. Evaluation: agent architecture 25%, code quality 25%, LLM integration 25%, innovation/extensibility 25%.

> **Amendment (2026-07-20):** After the initial build, SSE progress streaming and LLM cost estimation were removed to keep the MVP lean — clarity over surface area. The async job model (`202` + polling, plus a `?wait=true` sync convenience) is unchanged; progress is followed by polling `GET /{id}`. The API input was likewise narrowed from a free-form `query` to a `product` field (product name only): the planner always plans the complete analysis, and the `analyses` request field is the sole narrowing mechanism. The Makefile command menu was dropped too — the README documents the raw `uv`/`docker` commands directly. The mock platforms were switched from the French trio (Amazon/Cdiscount/Fnac, EUR) to the North American market (Amazon/Best Buy/Walmart, CAD). The **README reflects the shipped system**; this document records the original design and is kept as-is below.

## 1. Goal

A containerized FastAPI service exposing a market-analysis agent: given a product/market query, an LLM-planned LangGraph pipeline collects (mock) product data, analyzes sentiment and price trends in parallel, synthesizes a structured business report, and passes it through an LLM-as-judge quality gate. Runs end-to-end with **zero API keys** (mock LLM provider); any real provider is one env var away.

**Deliverable language:** README + theory answers in French; code, comments, identifiers in English.

## 2. Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | **LangGraph** | User preference: don't rebuild the orchestration loop. Provider-agnostic via LangChain model abstraction. Differentiation comes from graph design + README justification, not framework novelty. |
| Rejected: Claude Agent SDK | — | Anthropic-only in headless mode (`ANTHROPIC_API_KEY` required; no OAuth headless). Conflicts with the provider-agnostic requirement. (Verified: pip package bundles the CLI binary, so Docker would have been simple — the lock-in was the decider, not packaging.) |
| Rejected: native loop | — | Strongest orchestration showcase but highest effort; user explicitly declined rebuilding fundamentals. Both rejections become README content (choix justifié). |
| Orchestration shape | **Hybrid: LLM plans, graph controls** | Planner LLM emits a structured plan → conditional routing → parallel analysis fan-out → LLM synthesis → LLM-as-judge with one revision loop. "The LLM makes the decisions; the graph guarantees the engineering." |
| Provider strategy | Env-driven factory + OpenAI-compatible fallback + **mock provider** | Evaluator runs with any key or none. Structured outputs validated against Pydantic schemas everywhere. |
| Data strategy | Deterministic seeded mock adapters | Test explicitly blesses mock data. Reproducible demos for any query. Clean `PlatformAdapter` seam documented for real data (post-MVP user wish). |
| API pattern | Async job + polling + SSE, with `?wait=true` sync convenience | Production signal (feeds theory step 6) while keeping one-line curl demos. |

## 3. Architecture

### 3.1 System

```
client ──► FastAPI (api/) ──► AnalysisService ──► LangGraph StateGraph (agent/)
                 │                    │                    │
                 │              job registry         nodes call tools (tools/)
                 │              (in-memory)          LLM via factory (llm/)
                 └── SSE events ◄─── per-run event stream
```

One process. Each request creates an `AnalysisState`, runs the compiled graph async, streams node-level progress events, stores the result in an in-memory job registry (documented limitation; Redis/Postgres path argued in theory step 4).

### 3.2 The graph

```
                 ┌─────────┐
   request ────► │ planner │  LLM → structured AnalysisPlan
                 └────┬────┘
                      ▼
                ┌──────────┐
                │ collect  │  scraper tool → offers, prices, reviews, history
                └────┬─────┘
            ┌────────┴────────┐   conditional fan-out per plan
            ▼                 ▼
     ┌───────────┐     ┌──────────┐
     │ sentiment │     │  trends  │    PARALLEL branches
     └─────┬─────┘     └────┬─────┘
           └───────┬────────┘          fan-in via state reducers
                   ▼
            ┌────────────┐
            │ synthesize │  LLM → structured MarketReport
            └─────┬──────┘
                  ▼
             ┌─────────┐   score < threshold AND revision_count == 0
             │  judge  │ ────────────► synthesize (critique injected)
             └────┬────┘
                  ▼  pass or already revised
                 END
```

- **planner** (LLM): interprets the query → `AnalysisPlan {analyses_needed, platforms, rationale}`. The agentic decision point: "compare iPhone 16 prices" skips sentiment; "why are customers unhappy with X" requires it. When the request pins `analyses` explicitly, the planner treats it as a constraint and plans the rest (platforms, scope).
- **collect** (no LLM): runs platform adapters, always. Hard failure of *all* adapters → analysis fails; partial failure → degraded run.
- **sentiment** (LLM): structured extraction over collected reviews — distribution, praises, complaints, themes, representative quotes.
- **trends** (deterministic + LLM): stats over 30-day price/popularity history (min/max/avg, volatility, regression slope, competitor gaps, positioning percentile) + short LLM interpretation. Arithmetic stays out of the LLM.
- **synthesize** (LLM): compiles `MarketReport` — executive summary, price analysis, sentiment/trend findings, prioritized recommendations, confidence, caveats.
- **judge** (LLM): scores rubric {grounding in tool data, completeness vs plan, actionability}. Below threshold → one revision loop max, critique injected into synthesis context. Skippable via env (`JUDGE_ENABLED=false`), default on.

If the plan requests neither sentiment nor trends, collect routes straight to synthesize.

### 3.3 State (sketch)

```python
class AnalysisState(TypedDict):
    request: AnalysisRequest
    plan: AnalysisPlan | None
    collected: CollectedData | None
    sentiment: SentimentInsights | None
    trends: TrendInsights | None
    report: MarketReport | None
    judge_verdict: JudgeVerdict | None
    revision_count: int
    errors: Annotated[list[AnalysisError], operator.add]   # accumulating reducer
    usage: Annotated[list[LLMUsage], operator.add]         # per-call tokens/cost
```

All inner models are Pydantic v2. Parallel branches write disjoint keys (`sentiment`, `trends`) and append to reducer-managed lists, so fan-in is conflict-free.

## 4. Components

### 4.1 Tools (`tools/`) — all four suggested tools

Tools are plain typed Python callables (LLM-optional); graph nodes are thin wrappers. Tool = capability, node = orchestration step — this separation is a README talking point.

1. **Scraper** — `PlatformAdapter` ABC: `fetch(query) → PlatformData {offers, prices, ratings, reviews, price_history, popularity}`. Three mocks: Amazon, Cdiscount, Fnac (French platforms as a local touch). Data generated deterministically, seeded by `hash(product_query)` — realistic dispersion, any query works, same query = same data. Real-data adapters (post-MVP): implement the same ABC.
2. **Sentiment analyzer** — one focused prompt, structured output. Prompt-engineering showcase.
3. **Trend analyzer** — pure-Python stats + short LLM interpretation.
4. **Report generator** — `MarketReport` Pydantic model + Markdown renderer. JSON = API contract; rendered `.md` = "exemple de rapport" deliverable, one committed in `examples/`.

### 4.2 LLM layer (`llm/`)

- Config (pydantic-settings): `LLM_PROVIDER` (mock|groq|deepseek|openrouter|ollama|openai|anthropic), `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL` (optional override).
- Factory: named providers via first-class LangChain integrations; OpenAI-compatible `base_url` adapter covers the long tail. Exact package pins resolved at planning (background research in flight).
- **Structured output everywhere**: Pydantic schema binding with validation + one corrective re-ask on parse failure (cheap-model resilience).
- **Mock provider**: deterministic fake LLM returning schema-valid planner/sentiment/synthesis/judge outputs (lightly varied by query seed). Powers zero-key demos AND all unit tests.
- Per-call usage capture → aggregated cost estimate in response metadata.

### 4.3 API (`api/`)

| Endpoint | Behavior |
|---|---|
| `POST /api/v1/analyses` | Submit `{query, platforms?, analyses? ("auto" \| list), language?}` → `202 {id, status}`. `?wait=true` blocks and returns the full result. `analyses: "auto"` (default) lets the planner decide; an explicit list constrains the planner (it still selects platforms and scope). `language` defaults to `fr`. |
| `GET /api/v1/analyses/{id}` | Status → full result: report + metadata `{provider, model, duration_ms, tokens, cost_estimate_usd, judge {score, passed, revised}, degraded, caveats}`. |
| `GET /api/v1/analyses/{id}/events` | SSE progress stream from LangGraph event streaming (node started/finished, terminal event). |
| `GET /api/v1/analyses` | History list (in-memory). |
| `GET /api/v1/analyses/{id}/report.md` | Rendered Markdown report. |
| `GET /health` | Liveness + active provider/model (no secrets). |

Swagger/OpenAPI at `/docs` serves as the "exemples de requêtes" deliverable, plus curl examples in the README.

### 4.4 Job registry

In-memory `dict[id, AnalysisJob]` guarded by a lock; statuses `queued|running|done|failed`; per-job event buffer feeding SSE (snapshot + live). Explicitly documented as the single-process MVP choice; the Redis/Postgres upgrade is theory-step-4 content.

## 5. Error handling

1. **API layer**: Pydantic 422s with precise messages; 404 unknown id; startup fail-fast validation of LLM config (except mock).
2. **Graph layer**: node wrappers convert tool failures into typed `AnalysisError` entries in state; run continues degraded; report carries `caveats` + reduced `confidence`. Only total collection failure → `failed`.
3. **LLM layer**: per-call timeout; retry with exponential backoff on transient errors (rate limit, 5xx); structured-output corrective re-ask (1); per-analysis global timeout.
4. **Logging**: structured JSON logs w/ `analysis_id`, node, duration, tokens — lived evidence for theory step 5.

## 6. Testing (~30 focused tests, all offline, mock LLM)

- **Tools**: adapter determinism + schema validity; trends math on known series; sentiment parse from scripted output; renderer output.
- **Orchestration**: planner routing honors the plan (branches skipped correctly); parallel fan-in merges cleanly; judge triggers exactly one revision; degradation path produces caveats not crashes.
- **API**: `?wait=true` end-to-end on mock; 422/404; SSE event-order smoke.
- `pytest` + `pytest-asyncio`; no network; seconds to run.

## 7. Packaging & DX

- `pyproject.toml` + `uv` lockfile (plain-pip path documented too); `ruff` lint/format; type hints throughout; Python 3.13.
- Multi-stage `Dockerfile` (`python:3.13-slim`, non-root, healthcheck); `docker-compose.yml` with `LLM_PROVIDER=mock` default → `docker compose up` just works.
- `Makefile`: install / dev / test / lint / run / docker-build / docker-up / demo (`make demo` submits, streams SSE, prints report).
- `.env.example` with a commented block per provider; `examples/` with sample request + committed generated report.

### Repo layout

```
market-analysis-intelligence/
├── src/market_agent/
│   ├── api/          # FastAPI app, routes, request/response schemas, SSE
│   ├── agent/        # state.py, nodes/, graph.py (StateGraph assembly)
│   ├── tools/        # scraper/ (adapters), sentiment.py, trends.py, report.py
│   ├── llm/          # settings-driven factory, mock provider, usage capture
│   └── core/         # config, logging, errors
├── tests/
├── examples/
├── docs/design.md    # this document
├── Dockerfile  docker-compose.yml  Makefile  .env.example  README.md
```

## 8. README plan (French)

1. Pitch + architecture diagram (mermaid) 2. Démarrage rapide (zéro clé → docker compose up; puis config provider réel) 3. Choix techniques justifiés (LangGraph vs natif vs Claude Agent SDK — argued from today's real analysis; pattern hybride; structured outputs; mock-first) 4. API + exemples curl 5. Outils + seam vers données réelles 6. Tests 7. **Étapes 4–7** (architecture de données, monitoring, scaling, amélioration continue) — each grounded in what the code already does 8. Limites & évolutions.

## 9. Scope ladder

- **MVP (must):** 6-node graph, 4 tools, mock provider, API (POST/GET/wait), Docker, core tests, full French README incl. theory.
- **Standout (planned):** SSE streaming, judge loop, cost metadata, Markdown report rendering, seeded realistic mocks.
- **Stretch (time permitting):** TTL cache on collect (ties to theory 6), one real-data adapter behind env flag (user wish: "see if we can connect to real data").
- **Out:** real marketplace scraping (ToS/fragility), auth, external DB, deployment, UI.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Cheap providers flaky at structured output | JSON-mode + schema validation + corrective re-ask; mock provider guarantees the demo path. |
| Judge loop adds cost/latency | Capped at 1 revision; env-skippable; default on. |
| SSE blocked by some proxies | Polling endpoint always available. |
| LangGraph/LangChain version drift | Exact pins + lockfile at plan time (research task in flight; verify Python 3.13 wheels). |
| Evaluator has zero keys | Mock provider is the default path; README leads with it. |
