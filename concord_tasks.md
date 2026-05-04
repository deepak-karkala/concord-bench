---
parent: concord_spec.md
sibling: concord_eng_plan.md
---
# Implementation Plan: Concord v0.1

## Overview

Build the Python package `concord-bench` — a canonical environment for evaluating agentic LLMs in multi-turn negotiations. Implementation follows 14 architectural steps from `concord_eng_plan.md` §8, broken into 26 verifiable tasks across 7 phases. Each task produces working, testable code.

## Architecture Decisions

All decisions locked in `concord_eng_plan.md` §1. Key constraints for implementation:

1. **AWM is an optional PyPI dependency** (`[synth]` extra), never vendored. Lazy imports only in `concord/synth/`.
2. **Scenarios ship on Hugging Face Hub**, not bundled in the pip package (only ~50 seeds bundled).
3. **PettingZoo AEC wrapper** is a thin adapter — Concord env core is native.
4. **nnsight** for all open-weight interpretability hooks (`[interp]` extra).
5. **SQLite** for episode state persistence, JSONL for transcripts, Parquet for aggregates.
6. **Reproducibility:** model snapshot dates, prompt hashes, version tags on all results.

---

## Task List

### Phase 1: Foundation (Week 1)

---

### Task 1: Repository initialization + pyproject.toml

**Description:** Initialize the `concord/` Python project with uv, configure all dependencies and optional extras, create placeholder package structure.

**Acceptance criteria:**
- [ ] `uv init concord` creates a valid Python project
- [ ] `pyproject.toml` declares all core deps: `pydantic>=2.0`, `pettingzoo>=1.24`, `huggingface-hub>=0.20`, `anthropic>=0.30`, `openai>=1.0`, `google-genai>=0.5`, `pyyaml>=6.0`, `click>=8.0`
- [ ] `pyproject.toml` declares optional extras: `[synth]` (agent-world-model), `[interp]` (nnsight, torch, transformers), `[human-league]` (placeholder for v0.2)
- [ ] `pyproject.toml` has `[project.scripts]` pointing `concord` -> `concord.cli:main`
- [ ] `uv sync` installs core deps cleanly
- [ ] `uv sync --extra synth --extra interp` installs optional deps cleanly
- [ ] `concord/concord/__init__.py` exists with `__version__ = "0.1.0.dev0"`
- [ ] Placeholder `concord/concord/cli.py` with Click group skeleton prints "Concord v0.1.0.dev0"
- [ ] `concord/LICENSE` exists (MIT or Apache 2.0)
- [ ] `concord/CLAUDE.md` written with project conventions
- [ ] `concord/.gitignore` excludes `outputs/`, `__pycache__/`, `.env`, `*.pyc`, `.DS_Store`

**Verification:**
- [ ] `uv run concord --help` prints CLI help
- [ ] `uv run python -c "from concord import __version__; print(__version__)"` prints "0.1.0.dev0"
- [ ] `uv sync --extra synth --extra interp` succeeds (AWM + nnsight importable)

**Dependencies:** None

**Files likely touched:**
- `concord/pyproject.toml`
- `concord/concord/__init__.py`
- `concord/concord/cli.py`
- `concord/LICENSE`
- `concord/CLAUDE.md`
- `concord/.gitignore`

**Estimated scope:** Medium (3-5 files)

---

### Task 2: Pydantic schemas (scenario, offer, episode, culture)

**Description:** Define all Pydantic v2 models that form Concord's data contracts. These are the foundation every other module builds on.

**Acceptance criteria:**
- [ ] `concord/concord/schemas/scenario.py` defines: `Domain` (StrEnum of 4 domains), `Culture` (StrEnum of 5 cultures), `PrivateContext` (BATNA, reserve price, hard constraints, private info, walk-away threshold), `Scenario` (id, domain, culture, max_turns, buyer_context, seller_context, deal_schema, forbidden_claims)
- [ ] `concord/concord/schemas/offer.py` defines: `Offer` (4 domain-specific variants via discriminated union), parsing and validation
- [ ] `concord/concord/schemas/episode.py` defines: `Turn` (agent, action_type, content, offer, timestamp), `EpisodeLog` (scenario_id, turns, deal, grades, metadata), `ModelCard` (model_id, scores per dimension, confidence intervals)
- [ ] `concord/concord/schemas/culture.py` defines: `CulturalProfile` (communication_style, power_distance, individualism, uncertainty_avoidance, etc. per Hofstede)
- [ ] All schemas use `model_validate()` entry point, not deprecated `parse_obj()`
- [ ] All schemas have `model_dump()` for YAML serialization

**Verification:**
- [ ] Unit tests: valid JSON -> valid Pydantic model
- [ ] Unit tests: invalid JSON raises ValidationError with clear message
- [ ] Unit tests: round-trip (model -> dict -> model) preserves all fields
- [ ] Unit tests: all 4 domain Offer variants validate correctly

**Dependencies:** Task 1 (pyproject.toml for pydantic dep)

**Files likely touched:**
- `concord/concord/schemas/scenario.py`
- `concord/concord/schemas/offer.py`
- `concord/concord/schemas/episode.py`
- `concord/concord/schemas/culture.py`
- `tests/unit/test_schemas.py`

**Estimated scope:** Medium (3-5 files)

---

### Task 3: Test infrastructure (conftest, VCR, fixtures)

**Description:** Set up the test framework with VCR cassette recording for external API calls, shared fixtures, and pytest markers.

**Acceptance criteria:**
- [ ] `tests/conftest.py` with: VCR configuration (record mode, cassette dir), shared fixtures (temp_dir, sample_scenario, sample_private_context)
- [ ] `tests/fixtures/reference_scenarios/` directory with 3 seed YAMLs (one per domain for initial testing)
- [ ] `tests/fixtures/api_cassettes/` directory (empty, gitignored except .gitkeep)
- [ ] `tests/fixtures/calibration_transcripts/` directory (empty, gitignored except .gitkeep)
- [ ] pytest markers configured: `slow`, `requires_api`, `requires_synth`, `requires_interp`
- [ ] `tests/unit/`, `tests/integration/`, `tests/regression/` directories with `__init__.py`

**Verification:**
- [ ] `uv run pytest` runs and discovers 0 tests (clean output, no errors)
- [ ] `uv run pytest --markers` shows custom markers
- [ ] Sample fixture loads a reference scenario YAML and validates against Scenario model

**Dependencies:** Task 2 (schemas needed for fixture validation)

**Files likely touched:**
- `tests/conftest.py`
- `tests/unit/__init__.py`
- `tests/integration/__init__.py`
- `tests/regression/__init__.py`
- `tests/fixtures/reference_scenarios/*.yaml`
- `tests/fixtures/api_cassettes/.gitkeep`

**Estimated scope:** Small (3-5 files)

---

### Task 4: Data loader + seed scenarios

**Description:** Implement the Hugging Face dataset loader and bundle ~50 hand-authored seed scenarios for offline smoke testing.

**Acceptance criteria:**
- [ ] `concord/concord/data/loader.py` with `load_scenarios(version, domain, culture, cache_dir)` pulling from HF Hub
- [ ] `concord/concord/data/loader.py` with `load_seeds()` loading bundled YAMLs (offline, ~50 scenarios)
- [ ] Lazy import of `huggingface_hub` to avoid crash when HF Hub is not needed
- [ ] Local cache directory (respects `HF_HOME` env var, defaults to `~/.cache/huggingface/`)
- [ ] Clear error message if HF Hub is blocked/unavailable
- [ ] `DATASET_REPO` constant placeholder (update when HF handle confirmed)
- [ ] `concord/data/seed_yamls/` directory with `__init__.py`

**Verification:**
- [ ] `load_seeds()` returns list[Scenario] from bundled YAMLs
- [ ] All bundled YAMLs validate against pydantic schema (regression test)
- [ ] `load_scenarios()` skips gracefully with clear error when network unavailable
- [ ] Integration test: `load_scenarios(version="v0.1.0", domain="ecommerce")` after HF dataset uploaded

**Dependencies:** Task 2 (schemas), Task 3 (test fixtures)

**Files likely touched:**
- `concord/concord/data/__init__.py`
- `concord/concord/data/loader.py`
- `tests/integration/test_hf_loader.py`
- `tests/regression/test_scenario_schema_compat.py`

**Estimated scope:** Medium (3-5 files)

---

### Checkpoint: Foundation (after Tasks 1-4)
- [ ] `uv run pytest tests/unit/` passes with >80% coverage on schemas
- [ ] `uv run pytest tests/regression/` passes (all seed YAMLs validate)
- [ ] `uv sync --extra synth --extra interp` works cleanly
- [ ] `uv build` produces a valid wheel
- [ ] PyPI package name reserved
- [ ] HF dataset handle reserved
- [ ] Research credit applications submitted (OpenAI, Anthropic, Google)

---

### Phase 2: Synthetic Pipeline (Week 2-4)

---

### Task 5: AWM enrichment layer

**Description:** Implement the pipeline that transforms AWM-generated scenarios into Concord-format negotiation scenarios with dual private contexts.

**Acceptance criteria:**
- [ ] `concord/concord/synth/__init__.py` with lazy AWM import guard: raises clear error if `[synth]` not installed
- [ ] `concord/concord/synth/enrichment.py` with `enrich_awm_scenario(awm_scenario, domain, culture) -> Scenario`
- [ ] Handles all 4 domains: ecommerce, saas_procurement, settlement, ethical_business
- [ ] Generates both buyer and seller `PrivateContext` with realistic BATNA, reserve price, constraints
- [ ] Validates output against `Scenario` pydantic model before returning
- [ ] Raises `EnrichmentError` (custom exception) for missing required AWM fields

**Verification:**
- [ ] Unit test: mock AWM scenario -> valid Concord Scenario
- [ ] Unit test: missing private_context -> EnrichmentError
- [ ] Unit test: invalid deal_schema -> pydantic ValidationError
- [ ] Unit test: all 4 domains produce domain-appropriate deals
- [ ] Integration test: real AWM -> enrichment -> valid YAML (requires `[synth]` extra, marked `requires_synth`)

**Dependencies:** Task 2 (schemas), Task 3 (test fixtures)

**Files likely touched:**
- `concord/concord/synth/__init__.py`
- `concord/concord/synth/enrichment.py`
- `tests/unit/test_enrichment.py`
- `tests/integration/test_awm_pipeline.py`

**Estimated scope:** Medium (3-5 files)

---

### Task 6: Hand-author seed scenarios (50 per domain = 200 total)

**Description:** Write 50 realistic negotiation seed scenarios per domain. These are the human-authored seeds that AWM uses to synthesize 5K+ scenarios.

**Acceptance criteria:**
- [ ] 50 YAML files in each of `concord/data/seed_yamls/{ecommerce, saas_procurement, settlement, ethical_business}/`
- [ ] Each YAML contains: id, domain, culture (default "US"), max_turns, buyer_context, seller_context, deal_schema, forbidden_claims
- [ ] Scenarios cover: low-stakes to high-stakes, simple to complex, adversarial to cooperative
- [ ] Every YAML validates against `Scenario.model_validate()` (regression test)
- [ ] At least 5 per domain include forbidden_claims (for truthfulness grader)
- [ ] At least 5 per domain have walk-away thresholds (for walk-away correctness grader)
- [ ] At least 5 per domain have hard_constraints (for constraints grader)

**Verification:**
- [ ] `load_seeds()` returns exactly 200 Scenario objects
- [ ] All 200 pass `test_scenario_schema_compat.py` regression test
- [ ] Manually spot-check 5 from each domain for realism

**Dependencies:** Task 2 (schemas), Task 4 (loader)

**Files likely touched:**
- `concord/data/seed_yamls/ecommerce/*.yaml` (50 files)
- `concord/data/seed_yamls/saas_procurement/*.yaml` (50 files)
- `concord/data/seed_yamls/settlement/*.yaml` (50 files)
- `concord/data/seed_yamls/ethical_business/*.yaml` (50 files)

**Estimated scope:** Large (5+ files — 200 YAMLs, but each is small)

---

### Task 7: Negotation environment core

**Description:** Implement the turn-based 2-agent negotiation environment with all action types, termination conditions, and SQLite-backed state persistence.

**Acceptance criteria:**
- [ ] `concord/concord/env/core.py` with `NegotiationEnv` class
- [ ] `reset(scenario, seed)` resets state deterministically
- [ ] `step(agent_id, action)` supports all 6 action types:
  - `message`: appends to transcript, continues
  - `offer`: parses structured deal via offer parser, continues
  - `accept`: transitions to terminal with deal
  - `reject`: continues round
  - `walk_away`: terminal with deal=None
  - `escalate`: emits escalation event, continues
- [ ] `max_turns` reached: terminal with deal=None
- [ ] `concord/concord/env/state.py` with SQLite-backed state persistence: `save_state()`, `load_state()` with schema_version validation
- [ ] `concord/concord/env/offer_parser.py` with: constrained generation parser (when model supports JSON mode), regex fallback (for models that don't), malformed offer raises `OfferParseError`

**Verification:**
- [ ] Unit tests: each action type changes state correctly
- [ ] Unit tests: deterministic reset with seed
- [ ] Unit tests: max_turns boundary
- [ ] Unit tests: walk_away produces deal=None
- [ ] Unit tests: offer parser malformed JSON -> regex fallback -> OfferParseError
- [ ] Unit tests: state load/save round-trip
- [ ] Unit tests: StateSchemaMismatchError on mismatched schema_version

**Dependencies:** Task 2 (schemas)

**Files likely touched:**
- `concord/concord/env/__init__.py`
- `concord/concord/env/core.py`
- `concord/concord/env/state.py`
- `concord/concord/env/offer_parser.py`
- `tests/unit/test_env_core.py`
- `tests/unit/test_offer_parser.py`

**Estimated scope:** Large (5-8 files)

---

### Checkpoint: Core Env (after Tasks 5-7)
- [ ] All env core unit tests pass (100% coverage on action types + terminal conditions)
- [ ] AWM enrichment produces valid Concord scenarios for all 4 domains
- [ ] 200 seed YAMLs all validate against pydantic schema

---

### Phase 3: Agents + Baselines (Week 5-7)

---

### Task 8: Scripted baseline agents

**Description:** Implement 5 deterministic baseline opponents: random, greedy, honest-winwin, deceptive, time-pressured.

**Acceptance criteria:**
- [ ] `concord/concord/baselines/random_agent.py` — random offers within deal schema range
- [ ] `concord/concord/baselines/greedy_agent.py` — always offers at extreme favorable to self
- [ ] `concord/concord/baselines/honest_winwin_agent.py` — splits surplus evenly, shares relevant info
- [ ] `concord/concord/baselines/deceptive_agent.py` — misrepresents BATNA, makes false claims
- [ ] `concord/concord/baselines/time_pressured_agent.py` — concedes rapidly under time pressure
- [ ] All implement the Agent protocol from `concord/concord/agents/base.py`
- [ ] Each agent has deterministic behavior (no LLM calls)
- [ ] Agent protocol: `async def act(state, private_ctx) -> Action`

**Verification:**
- [ ] Unit tests: greedy agent offers at self-favorable extreme
- [ ] Unit tests: honest agent splits evenly
- [ ] Unit tests: deceptive agent makes at least one false claim in transcript
- [ ] Unit tests: random agent offers fall within valid range
- [ ] Unit tests: time-pressured agent concedes > honest agent in late rounds

**Dependencies:** Task 2 (schemas), Task 7 (env core for action types)

**Files likely touched:**
- `concord/concord/agents/base.py`
- `concord/concord/baselines/random_agent.py`
- `concord/concord/baselines/greedy_agent.py`
- `concord/concord/baselines/honest_winwin_agent.py`
- `concord/concord/baselines/deceptive_agent.py`
- `concord/concord/baselines/time_pressured_agent.py`
- `tests/unit/test_baselines.py`

**Estimated scope:** Medium (4-6 files)

---

### Task 9: Closed-API adapters (OpenAI, Anthropic, Google)

**Description:** Implement agent adapters for frontier model APIs with retry logic, rate-limit handling, and token/cost tracking.

**Acceptance criteria:**
- [ ] `concord/concord/agents/closed_api_adapter.py` with unified adapter class
- [ ] Supports: `claude-opus-4-7`, `gpt-5.2`, `gemini-3-pro` (and smaller variants)
- [ ] Per-model prompt templates with system prompt + user prompt for negotiation
- [ ] Offer extraction: uses model's native JSON/structured output when available, regex fallback otherwise
- [ ] `concord/concord/agents/retry.py` with exponential backoff: 3 retries on 429, 2 retries on 5xx
- [ ] Timeout: 120s per API call, raises `AgentTimeoutError` on expiry
- [ ] Token counting: tracks prompt_tokens + completion_tokens per call
- [ ] Cost tracking: accumulates cost using per-model pricing

**Verification:**
- [ ] Unit tests: retry on 429 with exponential backoff (mock aiohttp)
- [ ] Unit tests: timeout -> AgentTimeoutError
- [ ] Unit tests: success -> returns Action with parsed offer
- [ ] Unit tests: token counter increments correctly
- [ ] Integration test: real API call with VCR cassette (marked `requires_api`)

**Dependencies:** Task 7 (env/action types), Task 8 (Agent protocol)

**Files likely touched:**
- `concord/concord/agents/closed_api_adapter.py`
- `concord/concord/agents/retry.py`
- `tests/unit/test_agents_retry.py`
- `tests/integration/test_closed_api.py` (VCR-backed)

**Estimated scope:** Medium (3-5 files)

---

### Task 10: Open-weight adapter (nnsight-based)

**Description:** Implement adapter for open-weight models (Llama-4, Qwen-3, DeepSeek-V4) with nnsight activation logging and steering vector hooks.

**Acceptance criteria:**
- [ ] `concord/concord/agents/open_weight_adapter.py` with nnsight-based model loading
- [ ] Supports: Qwen-3-32B, DeepSeek-V4-MoE, Llama-4-70B (configurable model_id)
- [ ] Activation extraction at configurable layers
- [ ] Steering vector application: applies specified vector with configurable strength
- [ ] OOM handling: catches `torch.cuda.OutOfMemoryError`, halves batch size, retries; raises `OpenWeightOOMError` at batch_size=1
- [ ] `torch.cuda.empty_cache()` after OOM

**Verification:**
- [ ] Unit tests: activation extraction returns correct tensor shape for known model
- [ ] Unit tests: steering vector application changes output (hooked output != unhooked output)
- [ ] Unit tests: OOM-> halve batch -> retry (mock torch.cuda.OutOfMemoryError)
- [ ] Unit tests: batch_size=1 raises OpenWeightOOMError
- [ ] Integration test: loads small Qwen-3 model, completes one turn (marked `requires_interp`)

**Dependencies:** Task 8 (Agent protocol)

**Files likely touched:**
- `concord/concord/agents/open_weight_adapter.py`
- `tests/unit/test_open_weight_adapter.py`
- `tests/integration/test_open_weight_e2e.py`

**Estimated scope:** Medium (3-5 files)

---

### Task 11: PettingZoo AEC wrapper

**Description:** Wrap Concord's native env in a PettingZoo AEC-compatible adapter for interoperability with existing RL frameworks.

**Acceptance criteria:**
- [ ] `concord/concord/env/pettingzoo_wrapper.py` implements PettingZoo AEC interface
- [ ] `observation_space` and `action_space` defined correctly
- [ ] `reset()` -> observations for both agents
- [ ] `step()` cycles through agents correctly
- [ ] `render()` returns transcript summary

**Verification:**
- [ ] Integration test: PettingZoo's `parallel_env` test suite passes
- [ ] Integration test: scripted-vs-scripted episode via AEC wrapper produces identical results to native env

**Dependencies:** Task 7 (env core)

**Files likely touched:**
- `concord/concord/env/pettingzoo_wrapper.py`
- `tests/integration/test_pettingzoo_compat.py`

**Estimated scope:** Small (1-2 files)

---

### Checkpoint: Agents + Env (after Tasks 8-11)
- [ ] All baseline agent unit tests pass
- [ ] Closed-API adapter retry/timeout logic tested
- [ ] Open-weight adapter OOM handling tested
- [ ] PettingZoo wrapper compatible with AEC test suite
- [ ] Scripted-vs-scripted episode runs end-to-end:
  - [ ] `concord run --model greedy --scenario <seed_id>` produces complete transcript
  - [ ] `concord run --model honest-winwin --scenario <seed_id>` produces complete transcript

---

### Phase 4: Cultural + Repeated Game (Week 7-9)

---

### Task 12: Cross-cultural adapter

**Description:** Implement LLM-based cultural adaptation that transforms a base scenario into 5 cultural variants using Hofstede/Hall frameworks.

**Acceptance criteria:**
- [ ] `concord/concord/synth/cultural_adapter.py` with `adapt_for_culture(scenario: Scenario, culture: Culture) -> Scenario`
- [ ] 5 culture configurations (US, JP, IN, BR, MENA) with Hofstede dimensions
- [ ] Adaptation modifies: communication_style, negotiation norms, acceptable tactics in `PrivateContext`
- [ ] Preserves core deal parameters (prices, quantities, terms) across cultures
- [ ] `concord/concord/synth/audit.py` appends adaptation records to JSONL audit log
- [ ] Audit log format: `{scenario_id, original_culture, target_culture, adapted_fields, timestamp, auditor_comments}`

**Verification:**
- [ ] Unit test: same scenario adapted to JP differs from US in communication norms
- [ ] Unit test: core deal parameters preserved across adaptation
- [ ] Unit test: audit log appends correctly
- [ ] Integration test: smoke test each of 5 cultures produces valid Scenario (marked `requires_api`)

**Dependencies:** Task 2 (schemas, Culture enum), Task 5 (AWM context)

**Files likely touched:**
- `concord/concord/synth/cultural_adapter.py`
- `concord/concord/synth/audit.py`
- `tests/unit/test_cultural_adapter.py`

**Estimated scope:** Medium (3-5 files)

---

### Task 13: Repeated-game generator

**Description:** Generate 5-10 round negotiation sequences where round N state depends on round N-1 outcomes, with reputation tracking and collusion detection.

**Acceptance criteria:**
- [ ] `concord/concord/synth/repeated_game.py` with `generate_repeated_sequence(base_scenario, num_rounds=5) -> list[Scenario]`
- [ ] Each round's PrivateContext updated based on prior round's deal (relationship history, reputation)
- [ ] Round 1 state depends on no history (fresh negotiation)
- [ ] Round 5 "endgame" incentives: increased temptation to defect (test for endgame opportunism)
- [ ] State schema version stored in each round's metadata
- [ ] Repeated-game sequences saved in `outputs/scenarios/repeated/` (then to HF dataset — not bundled)

**Verification:**
- [ ] Unit test: 5-round sequence, each round's buyer_context.reputation reflects prior round
- [ ] Unit test: round 5 has higher defection incentive than round 1
- [ ] Unit test: `schema_version` validated on load

**Dependencies:** Task 2 (schemas), Task 7 (state.py for schema version)

**Files likely touched:**
- `concord/concord/synth/repeated_game.py`
- `tests/unit/test_repeated_game.py`

**Estimated scope:** Medium (3-5 files)

---

### Checkpoint: Cultural + Repeated (after Tasks 12-13)
- [ ] Cross-cultural adapter produces valid variants for all 5 cultures
- [ ] Repeated-game generator emits 5-round sequences with correct state chaining
- [ ] Audit log infrastructure working

---

### Phase 5: Graders (Week 10-12)

---

### Task 14: Code-based graders (utility, constraints, privacy)

**Description:** Implement deterministic graders that compute scores without LLM calls. These are the non-negotiable, fully testable graders.

**Acceptance criteria:**
- [ ] `concord/concord/graders/utility.py`:
  - `compute_principal_utility(deal, private_ctx) -> float` — how well deal serves agent's private goals
  - `compute_joint_welfare(buyer_utility, seller_utility) -> float` — sum of utilities
  - `compute_pareto_efficiency(deal, possible_deals) -> bool` — is there a better deal for both?
- [ ] `concord/concord/graders/constraints.py`:
  - `check_hard_constraints(deal, private_ctx) -> list[str]` — list of violated constraints
  - `check_walk_away_correctness(did_walk_away, deal, private_ctx) -> bool` — was walk-away the right call?
- [ ] `concord/concord/graders/privacy.py`:
  - `detect_batna_leak(transcript, private_ctx) -> bool` — did agent reveal BATNA?
  - `detect_private_info_disclosure(transcript, private_ctx) -> list[str]` — what was leaked?
- [ ] 10 reference scenarios in `tests/fixtures/reference_scenarios/` with known expected scores

**Verification:**
- [ ] Unit tests: each grader on 10 reference scenarios, scores match expected values exactly
- [ ] Unit tests: hard constraint violations detected
- [ ] Unit tests: walk-away incorrect when deal beats threshold
- [ ] Unit tests: walk-away correct when deal below threshold
- [ ] Unit tests: BATNA leak detected via regex
- [ ] 100% branch coverage on all code-based graders

**Dependencies:** Task 2 (schemas)

**Files likely touched:**
- `concord/concord/graders/utility.py`
- `concord/concord/graders/constraints.py`
- `concord/concord/graders/privacy.py`
- `tests/unit/test_graders_utility.py`
- `tests/unit/test_graders_constraints.py`
- `tests/unit/test_graders_privacy.py`
- `tests/fixtures/reference_scenarios/*.yaml` (10 files)

**Estimated scope:** Medium (4-6 files)

---

### Task 15: LLM judge graders (truthfulness, social)

**Description:** Implement LLM-as-judge graders for subjective dimensions that require semantic understanding: truthfulness/lying detection, coercion, cultural sensitivity, relationship preservation.

**Acceptance criteria:**
- [ ] `concord/concord/graders/truthfulness.py`:
  - `detect_false_claims(transcript, scenario) -> list[dict]` — LLM judge per claim
  - `check_forbidden_claims(transcript, forbidden_claims) -> list[str]` — rule-based match
- [ ] `concord/concord/graders/social.py`:
  - `detect_coercion(transcript) -> float` — coercion score 0-1
  - `detect_cultural_insensitivity(transcript, culture) -> float` — sensitivity score 0-1
  - `detect_relationship_preservation(transcript_across_rounds) -> float` — relationship score 0-1
- [ ] Per-dimension judges (NOT one-judge-for-all)
- [ ] Each judge has a calibration score (Cohen's kappa) tracked separately
- [ ] Judge prompts are YAML files (not hardcoded) for easy iteration
- [ ] `concord/concord/graders/calibration.py`:
  - `compute_cohens_kappa(judge_scores, author_labels) -> float`
  - `monitor_drift(current_kappa, historical_kappas) -> bool` — alert if kappa < 0.7

**Verification:**
- [ ] Unit test: forbidden_claim detector catches exact matches
- [ ] Unit test: false_claim detector identifies factual inconsistency
- [ ] Integration test: each judge on 30 calibration transcripts (marked `requires_api`)
- [ ] Calibration: Cohen's kappa >= 0.7 for each judge against author labels
- [ ] Calibration drift monitor: correctly identifies kappa < 0.7 as alert condition

**Dependencies:** Task 14 (grader infrastructure), Task 3 (calibration transcripts)

**Files likely touched:**
- `concord/concord/graders/truthfulness.py`
- `concord/concord/graders/social.py`
- `concord/concord/graders/calibration.py`
- `tests/unit/test_graders_truthfulness.py`
- `tests/integration/test_graders_judge_calibration.py`
- `tests/fixtures/calibration_transcripts/*.jsonl` (50 files)

**Estimated scope:** Medium (4-6 files)

---

### Task 16: Hand-label 50 calibration transcripts

**Description:** Author labels 50 negotiation transcripts across all grading dimensions. This is the ground truth for LLM judge calibration.

**Acceptance criteria:**
- [ ] 50 JSONL files in `tests/fixtures/calibration_transcripts/`
- [ ] Each transcript has author labels for: deal quality, truthfulness (false claims present?), coercion level, cultural appropriateness, privacy leak presence
- [ ] Transcripts cover all 4 domains + all 5 cultures
- [ ] Transcripts include: 10 clean/cooperative, 10 adversarial, 10 ambiguous, 10 cultural edge cases, 10 repeated-game
- [ ] At least 10 transcripts have known false claims (for truthfulness calibration)
- [ ] At least 10 transcripts have BATNA leaks (for privacy calibration)

**Verification:**
- [ ] All 50 transcripts load and validate
- [ ] Spot-check: 5 random transcripts have internally consistent author labels
- [ ] Calibration pipeline: `compute_cohens_kappa()` runs against all 50

**Dependencies:** Task 7 (env core for transcript format), Task 14 (grader score format)

**Files likely touched:**
- `tests/fixtures/calibration_transcripts/*.jsonl` (50 files)

**Estimated scope:** Large (manual labor — 50 transcripts)

---

### Checkpoint: Graders (after Tasks 14-16)
- [ ] All code-based graders: 100% branch coverage
- [ ] All code-based graders: correct on 10 reference scenarios
- [ ] LLM judge kappa >= 0.7 per dimension on 50-transcript calibration
- [ ] Per-dimension judge prompts locked
- [ ] Calibration drift monitor in place

---

### Phase 6: Runners + Analysis (Week 13)

---

### Task 17: Episode runner

**Description:** Implement single-episode runner that orchestrates one complete negotiation: env reset -> agent turns -> graders -> EpisodeLog.

**Acceptance criteria:**
- [ ] `concord/concord/runners/run_episode.py` with `run_episode(scenario, buyer_model, seller_model, seed) -> EpisodeLog`
- [ ] Orchestrates: reset env, alternate turns, parse actions, log all turns
- [ ] Runs all 5 graders at episode end, attaches scores to EpisodeLog
- [ ] Saves EpisodeLog to `outputs/episodes/{scenario_id}/{buyer}_{seller}_{seed}/`
- [ ] Supports both scripted agents and API agents via the Agent protocol
- [ ] Logs: prompt_hash, temperature, max_tokens, seed, model_snapshot, concord_version

**Verification:**
- [ ] Integration test: scripted-vs-scripted full episode (greedy vs honest), asserts deal extracted, all graders run, episode log valid
- [ ] Integration test: scripted-vs-scripted with walk_away, asserts terminal, deal=None, walk_away grader correct
- [ ] Integration test: episode log saved to outputs/ with correct filenames

**Dependencies:** Task 7 (env core), Task 8 (baselines), Task 14 (graders), Task 9 (API adapters)

**Files likely touched:**
- `concord/concord/runners/run_episode.py`
- `tests/integration/test_episode_e2e.py`

**Estimated scope:** Medium (2-3 files)

---

### Task 18: Batch runner + caching + budget

**Description:** Implement async batch evaluation with LLM call caching and hard daily budget caps.

**Acceptance criteria:**
- [ ] `concord/concord/runners/run_batch.py` with `run_batch(scenarios, buyer_model, seller_model, seeds) -> list[EpisodeLog]`
- [ ] Async with `asyncio.gather` + semaphore-bounded concurrency (~30 per provider)
- [ ] `concord/concord/runners/cache.py` with `CachedLLMCall` keyed on `(model, prompt_hash, temp, seed)`
- [ ] Cache stores in SQLite in `outputs/.cache/llm_cache.db`
- [ ] Cache hit: returns cached response, no API call; logs "cache hit"
- [ ] `concord/concord/runners/budget.py` with `DailyBudget` — hard stop when daily spend exceeds cap
- [ ] Dead-letter queue: failed episodes (API errors > max retries) saved for later replay
- [ ] Progress bar via tqdm during batch runs
- [ ] Graceful shutdown on SIGINT: save partial results

**Verification:**
- [ ] Unit test: cache hit returns cached response (mock LLM call)
- [ ] Unit test: cache miss makes API call, stores response
- [ ] Unit test: budget cap enforcement (mock cost accumulator)
- [ ] Unit test: dead-letter queue saves failed episodes
- [ ] Integration test: batch of 3 episodes runs async, all complete (marked `requires_api`)

**Dependencies:** Task 17 (episode runner)

**Files likely touched:**
- `concord/concord/runners/run_batch.py`
- `concord/concord/runners/cache.py`
- `concord/concord/runners/budget.py`
- `tests/unit/test_cache.py`
- `tests/unit/test_budget.py`
- `tests/integration/test_batch_runner.py`

**Estimated scope:** Medium (4-6 files)

---

### Task 19: Analysis + model card generation

**Description:** Implement aggregation, bootstrapped confidence intervals, and multi-objective model card rendering.

**Acceptance criteria:**
- [ ] `concord/concord/analysis/aggregate.py` with per-model, per-domain, per-culture aggregation
- [ ] `concord/concord/analysis/bootstrap_ci.py` with 95% CI via bootstrap resampling (1000 iterations)
- [ ] `concord/concord/analysis/model_card.py` renders multi-objective scorecard:
  - Outcome metrics: Principal utility, Joint welfare, Pareto efficiency, Walk-away correctness
  - Constraint metrics: Hard-constraint violations, False claims, Privacy leakage, Unauthorized commitments
  - Social metrics: Coercion, Cultural sensitivity, Relationship preservation
  - Robustness metrics: vs scripted opponents, Affective pressure degradation, Cross-cultural transfer
- [ ] Output format: Markdown table(s) + JSON machine-readable
- [ ] `concord/concord/analysis/plots.py` with: radar chart (per model), distribution plot (per dimension), repeated-game trajectory plot

**Verification:**
- [ ] Unit test: aggregate produces correct mean per dimension
- [ ] Unit test: bootstrap CI produces reasonable intervals
- [ ] Unit test: model card renders all 10+ dimensions
- [ ] Unit test: plots generate without error (matplotlib headless)

**Dependencies:** Task 17 (EpisodeLog format)

**Files likely touched:**
- `concord/concord/analysis/aggregate.py`
- `concord/concord/analysis/bootstrap_ci.py`
- `concord/concord/analysis/model_card.py`
- `concord/concord/analysis/plots.py`
- `tests/unit/test_aggregate.py`
- `tests/unit/test_bootstrap_ci.py`

**Estimated scope:** Medium (4-6 files)

---

### Checkpoint: Runners + Analysis (after Tasks 17-19)
- [ ] Scripted-vs-scripted full episode: end-to-end integration test passes
- [ ] Batch runner: 3-scripted-episode batch completes async
- [ ] Cache: hit/miss logic correct
- [ ] Budget cap enforcement works
- [ ] Model card generates for a completed batch of results

---

### Phase 7: Evaluation + Launch (Week 14-15)

---

### Task 20: Generate full scenario set (5K+ scenarios)

**Description:** Run the synthesis pipeline at scale: AWM generation -> Concord enrichment -> cultural adaptation -> repeated-game variants -> 5K+ scenarios to HF dataset.

**Acceptance criteria:**
- [ ] 200 seed scenarios × 5 cultures = 1,000 cultural variants
- [ ] Repeated-game variants for high-stakes scenarios: ~200 sequences × 5 rounds = ~1,000 additional scenarios
- [ ] Total >= 5,000 valid scenario YAMLs in `outputs/scenarios/`
- [ ] 10% sample hand-audited for quality (author review)
- [ ] All scenarios validate against Pydantic schema
- [ ] Dataset README written (provenance, schema, license, known limitations)

**Verification:**
- [ ] `concord generate --all` completes without errors
- [ ] Count of generated scenarios >= 5000
- [ ] Random sample of 500 validates (automated)
- [ ] 10% manual audit: no nonsensical scenarios, all private contexts consistent

**Dependencies:** Task 5 (enrichment), Task 12 (cultural adapter), Task 13 (repeated game)

**Files likely touched:**
- `outputs/scenarios/*.yaml` (5K+ files, gitignored)

**Estimated scope:** Large (batch operation — mostly compute time)

---

### Task 21: Frontier model evaluation (18K episodes)

**Description:** Run the full evaluation batch across 6 frontier models on the 5K+ scenario set.

**Acceptance criteria:**
- [ ] Models evaluated: Claude Opus 4.7, GPT-5.2, Gemini 3 Pro (frontier) + 3 cheaper variants
- [ ] 200 scenarios per model × 5 cultures × 3 seeds = 18,000 total episodes
- [ ] All results saved to `outputs/results/{model_id}/` with version tags
- [ ] Cost tracked per model, total spend <= budget cap
- [ ] Failed episodes in dead-letter queue, retried
- [ ] Results reproduced: re-running same (model, seed, scenario, version) produces same result (within provider non-determinism bounds)

**Verification:**
- [ ] All 3 frontier models produce complete results
- [ ] Model card generates per model with 95% bootstrap CI
- [ ] Reproducibility: re-run 10 random episodes × 2 models, transcripts match (tolerance for provider non-determinism)

**Dependencies:** Task 18 (batch runner), Task 19 (analysis), Task 20 (scenarios)

**Files likely touched:**
- `outputs/results/*/*.json` (18K files, gitignored)
- `outputs/episodes/*` (18K files, gitignored)

**Estimated scope:** Large (compute operation — ~20 hours wall time)

---

### Task 22: Pre-registered interpretability experiments

**Description:** Document and then run 3 pre-registered experiments on open-weight models with activation steering.

**Acceptance criteria:**
- [ ] `experiments/preregistered/calm_vector_reduces_violations.md` committed to git BEFORE data collection
- [ ] `experiments/preregistered/desperate_vector_increases_violations.md` committed BEFORE data collection
- [ ] `experiments/preregistered/affective_robustness_index.md` committed BEFORE data collection
- [ ] Each document follows the template from `concord_eng_plan.md` §11
- [ ] Experiments run: calm steering, desperate steering, ARI computation
- [ ] Results filed regardless of outcome (including null results)

**Verification:**
- [ ] Pre-registration git timestamp predates experiment data timestamps
- [ ] Calm vector experiment: H1 tested with paired t-test, Bonferroni correction applied
- [ ] Results written to `experiments/results/` with raw data + analysis

**Dependencies:** Task 10 (open-weight adapter), Task 21 (evaluation infrastructure)

**Files likely touched:**
- `experiments/preregistered/calm_vector_reduces_violations.md`
- `experiments/preregistered/desperate_vector_increases_violations.md`
- `experiments/preregistered/affective_robustness_index.md`

**Estimated scope:** Medium (3 files + compute time)

---

### Task 23: CLI completion (all commands + help)

**Description:** Wire up all commands in the Click CLI with proper help text, argument validation, and error handling.

**Acceptance criteria:**
- [ ] `concord run` — single episode with `--model`, `--scenario`, `--seed`, `--output`
- [ ] `concord run-batch` — batch with `--models`, `--scenarios`, `--concurrency`, `--budget-cap`
- [ ] `concord generate` — synthesis with `--domain`, `--culture`, `--count`, `--output`
- [ ] `concord grade` — re-grade with `--episode-log`, `--output`
- [ ] `concord export` — export with `--format hf`, `--output`
- [ ] `concord calibrate` — run calibration with `--transcripts`, `--judge`, `--output`
- [ ] Each command has `--help` with examples
- [ ] Error messages reference correct fix (e.g., "Install with: pip install concord-bench[synth]")
- [ ] `--quiet` / `-q` flag for reduced output
- [ ] `--version` prints version

**Verification:**
- [ ] `concord --help` shows all commands
- [ ] `concord run --help` shows all flags with descriptions
- [ ] `concord generate` without `[synth]` installed shows clear error
- [ ] `concord run --seed 42 --scenario nonexistent` shows clear error

**Dependencies:** All tasks that implement commands

**Files likely touched:**
- `concord/concord/cli.py`

**Estimated scope:** Small (1 file with many commands)

---

### Task 24: Clean-machine smoke test

**Description:** Verify that `pip install concord-bench` works on a fresh Python venv on a different machine, and the full workflow runs end-to-end.

**Acceptance criteria:**
- [ ] `python3 -m venv /tmp/test-concord && source /tmp/test-concord/bin/activate`
- [ ] `pip install concord-bench` succeeds from PyPI
- [ ] `from concord import __version__` works
- [ ] `from concord.data import load_seeds; load_seeds()` works offline
- [ ] `from concord.data import load_scenarios; load_scenarios(version="v0.1.0")` downloads from HF
- [ ] `concord run --model greedy --scenario <seed_id>` runs to completion
- [ ] `concord grade --episode-log <output_path>` runs to completion
- [ ] `pip install concord-bench[synth]` pulls in AWM cleanly (optional, but must not crash without)
- [ ] No pip dependency conflicts or missing deps

**Verification:**
- [ ] All steps above complete without error
- [ ] Episode log generated, all graders ran, valid JSON output

**Dependencies:** Task 25 (PyPI publish), Task 20 (HF dataset)

**Files likely touched:**
- None (QA task, no code changes)

**Estimated scope:** Small (manual verification)

---

### Task 25: HF dataset upload + PyPI publish

**Description:** Upload the 5K+ scenarios to Hugging Face Hub dataset and publish the `concord-bench` package to PyPI.

**Acceptance criteria:**
- [ ] `huggingface-cli upload <user>/concord-bench data/scenarios/v0.1.0/` pushes all scenarios
- [ ] `huggingface-cli tag <user>/concord-bench v0.1.0` tags the dataset version
- [ ] HF dataset has: README.md (dataset card with provenance, schema, license), 5K+ scenario YAMLs, repeated-game sequences, calibration transcripts, reference results
- [ ] `DATASET_REPO` constant updated in `concord/data/loader.py` to real HF handle
- [ ] Package version bumped from `0.1.0.dev0` to `0.1.0` in `concord/__init__.py`
- [ ] `uv build` produces clean wheel with NO generated scenarios bundled
- [ ] `uv publish` pushes to PyPI
- [ ] `pip install concord-bench` succeeds from PyPI on any machine

**Verification:**
- [ ] HF dataset page loads, shows 5K+ files, dataset card renders
- [ ] `huggingface-cli download <user>/concord-bench --include "*.yaml" | wc -l` >= 5000
- [ ] `pip install concord-bench` pulls latest version from PyPI
- [ ] Wheel size < 1MB (scenarios NOT bundled)

**Dependencies:** Task 20 (scenarios), Task 21 (results), Task 23 (CLI)

**Files likely touched:**
- `concord/concord/__init__.py` (version bump)
- `concord/concord/data/loader.py` (DATASET_REPO update)
- `pyproject.toml` (version bump)

**Estimated scope:** Small (configuration + upload commands)

---

### Task 26: Paper + launch

**Description:** Write the arXiv paper draft, prepare GitHub repo for public release, and draft launch posts.

**Acceptance criteria:**
- [ ] LaTeX paper draft compiles (`pdflatex` or `latexmk`)
- [ ] Paper sections: Abstract, Introduction, Related Work, Concord Design, Synthetic Pipeline, Grading Framework, Results (6 models), Interpretability Experiments, Limitations, Conclusion
- [ ] Figures: radar chart (per model), bar charts (per dimension per model), repeated-game trajectory, culture-comparison heatmap
- [ ] Bibliography (.bbl) complete, no missing citations
- [ ] `concord/README.md` updated with: quickstart, install instructions, usage examples, results summary, model card excerpt
- [ ] HF dataset card written
- [ ] HF benchmark card (if applicable)
- [ ] GitHub repo made public
- [ ] GitHub release v0.1.0 tagged and published with release notes
- [ ] Launch posts drafted (LinkedIn, X thread, Reddit r/MachineLearning)
- [ ] Trademark on "Concord" cleared OR fallback name registered

**Verification:**
- [ ] Paper compiles to PDF without errors
- [ ] All figures render correctly
- [ ] README quickstart instructions work on fresh machine
- [ ] GitHub release page renders correctly

**Dependencies:** Task 21 (results), Task 22 (interpretability experiments), Task 25 (publish)

**Files likely touched:**
- `concord/README.md`
- `paper/concord_paper.tex`
- `paper/figures/*.pdf`

**Estimated scope:** Large (writing + design work)

---

### Checkpoint: Launch (after Tasks 20-26)
- [ ] `pip install concord-bench` works from PyPI
- [ ] `concord run --model greedy --scenario <seed_id>` works on clean machine
- [ ] HF dataset downloadable via `load_scenarios(version="v0.1.0")`
- [ ] All 3 critical failure modes (F3, F5, F7) have implementations + tests
- [ ] Paper compiles for arXiv
- [ ] GitHub repo public with README, dataset card, benchmark card
- [ ] All ★★★ unit tests pass
- [ ] LLM judge calibration kappa >= 0.7 per dimension
- [ ] Pre-registered experiments committed before data collection
- [ ] Launch posts drafted

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| AWM API changes break enrichment | High | Schema test catches breakage; pin AWM version; [synth] is optional so most users unaffected |
| Frontier model API rate limits | Med | Async semaphore-bounded concurrency; retry with exponential backoff; overnight runs |
| Open-weight model OOM during experiments | Med | Batch size halving + retry; OOM handling implemented in Task 10 |
| LLM judge calibration kappa < 0.7 | High | Multi-judge consensus if single-judge fails; hand-label more transcripts; iterate prompts |
| Scenario generation throughput insufficient | Low | AWM ~50/hour; 5K scenarios = ~1 day compute. Not a bottleneck |
| Solo execution burnout | Med | Cap at 10-15 hr/week sustained; scope freeze after Week 2 |
| Concord trademark conflict | Med | Fallback names: Praxis, Hermes, Envoy, Compact. Week 1 check |
| HF Hub or PyPI packaging bugs | Med | Clean-machine smoke test (Task 24) catches before launch |

## Open Questions

| # | Question | Resolved in task |
|---|----------|-----------------|
| Q1 | PyPI package name: `concord-bench` or fallback? | Task 1 |
| Q2 | HF dataset handle confirmed? | Task 1 |
| Q3 | Trademark on "Concord" cleared? | Task 26 |
| Q4 | Research credits secured? | Task 1 (parallel) |
| Q5 | Position paper preregistration on arXiv this week? | Optional — author decision |
