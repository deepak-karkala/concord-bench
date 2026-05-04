---
parent: concord_ceo_plan.md
sibling: concord_eng_plan.md
---
# Spec: Concord v0.1

## Objective

Concord is the canonical environment for evaluating and training agentic LLMs as principal-aligned representatives in multi-turn negotiations under private goals, ethical constraints, social pressure, and cross-cultural norms.

**What:** A Python library + CLI that provides (a) a synthetic negotiation scenario pipeline, (b) a turn-based 2-agent negotiation environment, (c) multi-objective graders, and (d) reproducible evaluation runners for frontier and open-weight models.

**Who:** ML researchers evaluating frontier models; AI safety researchers probing constraint alignment; interpretability researchers using white-box hooks; model developers training prosocial agents.

**Success criteria (v0.1):**
- `pip install concord-bench` works from PyPI on a fresh Python venv
- `from concord.data import load_scenarios; load_scenarios(version="v0.1.0")` returns 5K+ validated scenarios from HF Hub
- `concord run --model claude-opus-4-7 --scenario <id>` runs end-to-end on a fresh machine
- All ★★★ unit tests pass (100% branch coverage on graders)
- LLM judge calibration kappa >= 0.7 per dimension on 50-transcript holdout set
- Frontier eval batch completed: 18K episodes across 6 models, results in `outputs/results/`
- Pre-registered interpretability experiments committed before data collection
- All 3 critical failure modes (F3 open-weight OOM, F5 calibration drift, F7 repeated-game state corruption) have implemented mitigations + tests
- Paper draft compiles for arXiv
- GitHub repo public with dataset card, benchmark card, README

## Tech Stack

| Layer | Choice | Version |
|-------|--------|---------|
| Language | Python | >=3.11 |
| Package manager | uv | latest |
| Schema | Pydantic | >=2.0 |
| Env wrapper | PettingZoo AEC | >=1.24 |
| Scenario synthesis | agent-world-model (AWM) | >=0.1.0 (optional `[synth]` extra) |
| Interpretability | nnsight + torch + transformers | optional `[interp]` extra |
| Scenario distribution | Hugging Face Hub dataset | huggingface-hub >=0.20 |
| LLM SDKs | anthropic, openai, google-genai | latest |
| CLI | Click | >=8.0 |
| Config | PyYAML | >=6.0 |
| Testing | pytest + pytest-asyncio | latest |
| Linting | ruff | latest |
| Type checking | mypy | latest |

## Commands

```bash
# Development
uv sync --extra synth --extra interp    # Install all deps including optional
uv run pytest                           # Run all tests
uv run pytest tests/unit/               # Unit tests only
uv run pytest tests/integration/        # Integration tests only
uv run ruff check concord/ tests/       # Lint
uv run mypy concord/                    # Type check
uv build                                # Build wheel

# Concord CLI (post-install)
concord run --model <id> --scenario <id>                    # Single episode
concord run-batch --models <ids> --scenarios <split>        # Batch evaluation
concord generate --domain <d> --culture <c> --count N       # Synthetic generation
concord grade --episode-log <path>                          # Re-grade saved episode
concord export --format hf                                   # Export to HF dataset format
```

## Project Structure

```
concord/                                 # GitHub repo root
├── pyproject.toml                       # Declares extras: [synth], [interp]
├── README.md
├── CLAUDE.md
├── LICENSE                              # MIT or Apache 2.0
├── concord_ceo_plan.md
├── concord_eng_plan.md
├── concord_spec.md                      # This file
├── concord_tasks.md                     # Task breakdown
├── test_plan.md
├── ideation/                            # Original idea docs + feedback
│
├── concord/                             # Python package (published to PyPI)
│   ├── __init__.py
│   ├── schemas/                         # Pydantic models
│   │   ├── scenario.py                  # Scenario, PrivateContext, CulturalProfile
│   │   ├── offer.py                     # Offer (4 domain variants)
│   │   ├── episode.py                   # EpisodeLog, Transcript, ModelCard
│   │   └── culture.py                   # CulturalProfile enum (US, JP, IN, BR, MENA)
│   │
│   ├── synth/                           # Synthesis pipeline (lazy AWM imports)
│   │   ├── __init__.py                  # Lazy import guard for [synth] extra
│   │   ├── enrichment.py                # AWM scenario -> Concord scenario YAML
│   │   ├── cultural_adapter.py          # LLM-based 5-culture adaptation
│   │   ├── repeated_game.py             # Round-sequence variant generation
│   │   └── audit.py                     # Native-speaker audit log writer
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                    # HF Hub pull + local cache
│   │   └── seed_yamls/                  # Bundled hand-authored seeds (~50)
│   │       ├── ecommerce/
│   │       ├── saas_procurement/
│   │       ├── settlement/
│   │       └── ethical_business/
│   │
│   ├── env/
│   │   ├── core.py                      # NegotiationEnv (turn-based, 2-agent)
│   │   ├── pettingzoo_wrapper.py        # AEC adapter
│   │   ├── offer_parser.py              # Constrained gen + regex fallback
│   │   └── state.py                     # SQLite-backed episode state
│   │
│   ├── agents/
│   │   ├── base.py                      # Agent protocol
│   │   ├── closed_api_adapter.py        # OpenAI/Anthropic/Google
│   │   ├── open_weight_adapter.py       # nnsight-based, activation logging
│   │   └── retry.py                     # Backoff, rate-limit handling
│   │
│   ├── graders/
│   │   ├── utility.py                   # Principal utility, joint welfare, Pareto
│   │   ├── constraints.py               # Hard-constraint, walk-away correctness
│   │   ├── truthfulness.py              # False-claim detector (LLM judge)
│   │   ├── privacy.py                   # BATNA leak, private-info disclosure
│   │   ├── social.py                    # Coercion, cultural-sensitivity, relationship
│   │   └── calibration.py               # Cohen's kappa, drift monitoring
│   │
│   ├── baselines/
│   │   ├── random_agent.py
│   │   ├── greedy_agent.py
│   │   ├── honest_winwin_agent.py
│   │   ├── deceptive_agent.py
│   │   └── time_pressured_agent.py
│   │
│   ├── runners/
│   │   ├── run_episode.py               # Single episode
│   │   ├── run_batch.py                 # Async batch (semaphore-bounded)
│   │   ├── cache.py                     # LLM call cache
│   │   └── budget.py                    # Hard daily cap
│   │
│   ├── analysis/
│   │   ├── aggregate.py                 # Per-model, per-domain, per-culture
│   │   ├── bootstrap_ci.py              # 95% CI via bootstrap resampling
│   │   ├── model_card.py                # Multi-objective scorecard renderer
│   │   └── plots.py                     # Radar, distribution, repeated-game
│   │
│   └── cli.py                           # Click CLI entry point
│
├── experiments/
│   └── preregistered/                   # Pre-registered hypotheses
│       ├── calm_vector_reduces_violations.md
│       ├── desperate_vector_increases_violations.md
│       └── affective_robustness_index.md
│
├── tests/
│   ├── conftest.py                      # Fixtures, VCR cassettes
│   ├── unit/
│   │   ├── test_schemas.py
│   │   ├── test_enrichment.py
│   │   ├── test_offer_parser.py
│   │   ├── test_graders_utility.py
│   │   ├── test_graders_constraints.py
│   │   ├── test_graders_truthfulness.py
│   │   └── test_agents_retry.py
│   ├── integration/
│   │   ├── test_episode_e2e.py
│   │   ├── test_awm_pipeline.py
│   │   ├── test_hf_loader.py
│   │   └── test_pettingzoo_compat.py
│   ├── fixtures/
│   │   ├── reference_scenarios/         # 10 known-good scenarios
│   │   ├── calibration_transcripts/     # 50 hand-labeled transcripts
│   │   └── api_cassettes/               # VCR-recorded API responses
│   └── regression/
│       └── test_scenario_schema_compat.py
│
└── outputs/                             # Generated artifacts (gitignored)
    ├── scenarios/
    ├── episodes/
    └── results/
```

Files at end of v0.1: ~80 source + ~50 test. Scenarios on HF Hub, not in git.

## Code Style

**Naming:** `snake_case` for files, functions, variables. `PascalCase` for Pydantic models and classes. Constants in `UPPER_SNAKE_CASE`.

**Typing:** All public functions have type annotations. Pydantic v2 models for all structured data. Use `str | None` (3.10+ union syntax), not `Optional[str]`.

**Example:**

```python
# concord/concord/schemas/scenario.py
from pydantic import BaseModel
from enum import StrEnum

class Domain(str, Enum):
    ECOMMERCE = "ecommerce"
    SAAS_PROCUREMENT = "saas_procurement"
    SETTLEMENT = "settlement"
    ETHICAL_BUSINESS = "ethical_business"

class PrivateContext(BaseModel):
    """Agent's private knowledge: BATNA, reserve price, hard constraints."""
    batna: float
    reserve_price: float | None = None
    hard_constraints: list[str] = []
    private_info: list[str] = []
    walk_away_threshold: float | None = None

class Scenario(BaseModel):
    """A complete negotiation scenario with public + private contexts."""
    id: str
    domain: Domain
    culture: str
    max_turns: int = 10
    buyer_context: PrivateContext
    seller_context: PrivateContext
    deal_schema: dict  # Structured offer format
    forbidden_claims: list[str] = []
```

**Lazy imports for optional deps:**

```python
# concord/concord/synth/__init__.py
_AWM_AVAILABLE: bool | None = None

def _check_awm() -> None:
    global _AWM_AVAILABLE
    if _AWM_AVAILABLE is not None:
        return
    try:
        import awm  # noqa: F401
        _AWM_AVAILABLE = True
    except ModuleNotFoundError:
        _AWM_AVAILABLE = False
        raise ModuleNotFoundError(
            "Concord scenario generation requires the [synth] extra. "
            "Install with: pip install concord-bench[synth]"
        )
```

**Docstrings:** Google-style for public API. No docstrings on private helpers.

**Async:** All LLM calls and batch runners use `asyncio`. Semaphore-bounded concurrency.

**Error handling:** Custom exception hierarchy under `ConcordError`. Pydantic `model_validate()` for input validation. All external API calls wrapped in retry with exponential backoff.

## Testing Strategy

**Framework:** pytest + pytest-asyncio. VCR.py for API call recording.

**Test levels:**

| Level | Location | Scope | Legend |
|-------|----------|-------|--------|
| Unit | `tests/unit/` | Single function/class, mocked deps | ★★★ behavior + edge + error |
| Integration | `tests/integration/` | Multi-module, real deps | ★★ happy path |
| Regression | `tests/regression/` | Schema compatibility, calibration drift | ★ smoke check |

**Coverage targets:**
- Graders: 100% branch coverage (non-negotiable)
- Env core: 100% for all action types + terminal conditions
- Agents: all retry/error paths covered
- Schemas: all validation paths covered

**Calibration set:** 50 hand-labeled transcripts. Used to compute Cohen's kappa for each LLM judge. Kappa >= 0.7 required before locking judges.

**Fixture strategy:**
- `conftest.py`: shared fixtures (VCR setup, temp SQLite, reference scenarios)
- `fixtures/reference_scenarios/`: 10 known-good scenario YAMLs for grader tests
- `fixtures/calibration_transcripts/`: 50 hand-labeled transcripts
- `fixtures/api_cassettes/`: VCR-recorded API responses

**Granularity:**
- Use pytest markers: `@pytest.mark.slow`, `@pytest.mark.requires_api`, `@pytest.mark.requires_synth`, `@pytest.mark.requires_interp`
- Slow/integration tests gated behind markers; CI runs unit tests + fast integration only

## Boundaries

### Always do

- Run `uv run pytest tests/unit/` before any commit
- Validate all scenario YAMLs against Pydantic schema on load
- Log `(model, prompt_hash, temp, seed)` for every episode
- Version-tag all published results with Concord version + model snapshot
- Handle rate limits with exponential backoff (never crash on 429)
- Catch and handle `torch.cuda.OutOfMemoryError` in open-weight adapter
- Pre-register interpretability experiments BEFORE data collection, commit to git

### Ask first

- Adding new dependencies to `pyproject.toml`
- Changing the CLI interface (command names, flags, output format)
- Modifying the scenario schema (breaks all existing YAMLs)
- Changing grader weights or scoring formulas
- Running frontier model evaluations (API cost: ~$5K total)
- Publishing to PyPI
- Pushing datasets to HF Hub

### Never do

- Commit API keys, secrets, or `.env` files
- Vendor AWM or nnsight inside the Concord repo
- Skip test coverage on any grader
- Commit generated outputs (scenarios, episodes, results) to git
- Report results without model snapshot dates and Concord version tags
- Claim reproducibility without documenting provider non-determinism

## Success Criteria (Testable)

1. **Install smoke:** `pip install concord-bench` on fresh Python 3.11+ venv succeeds. `from concord import __version__` works.
2. **Seed scenarios load offline:** `from concord.data import load_seeds` returns ~50 valid `Scenario` objects without network.
3. **HF scenarios load:** `load_scenarios(version="v0.1.0", domain="ecommerce")` returns scenarios from HF Hub.
4. **Single episode:** `concord run --model greedy --scenario <id>` produces valid EpisodeLog with all graders run.
5. **Offer parsing:** Accepts well-formed JSON offers; falls back to regex for malformed offers; raises on unparseable offers.
6. **All action types:** Message, offer, accept, reject, walk_away, escalate all change env state correctly.
7. **Max turns:** Episode terminates at turn 10 with deal=None.
8. **Repeated game:** 5-round sequence loads round N state from round N-1 correctly.
9. **API retry:** 3 retries with exponential backoff on 429; timeout raises TimeoutError.
10. **Open-weight OOM:** Halves batch size, retries; raises OpenWeightOOMError at batch_size=1.
11. **Calibration kappa:** Each LLM judge >= 0.7 Cohen's kappa against author labels on 50-transcript holdout.
12. **State schema version:** Mismatched `schema_version` raises StateSchemaMismatchError.
13. **Cache hit:** Same (model, prompt_hash, temp, seed) returns cached response; no API call.
14. **Budget cap:** Hard stop when daily spend exceeds configured cap.
15. **Reproducibility:** Same seed + model snapshot + temp -> identical transcript (modulo provider non-determinism).

## Open Questions

| # | Question | Status | Owner |
|---|----------|--------|-------|
| Q1 | PyPI package name: `concord-bench` or fallback? | Week 1 decision | Author |
| Q2 | HF dataset handle confirmed? | Week 1 reservation | Author |
| Q3 | Trademark on "Concord" cleared? | Week 1 check | Author |
| Q4 | Research credits secured (OpenAI/Anthropic/Google)? | Week 1 application | Author |
| Q5 | Multi-judge consensus if single-judge kappa < 0.7? | Deferred to implementation | Engineering |
| Q6 | Optional `[scenarios]` extra to bundle 5K scenarios? | Deferred to v0.2 (only if users complain) | Product |
| Q7 | Position paper preregistration on arXiv this week? | Optional, week 1 decision | Author |

## Cross-References

- Strategic context, dream state, v0.2 roadmap: `concord_ceo_plan.md`
- Architecture, module specs, failure modes, performance budget: `concord_eng_plan.md`
- QA test cases, edge cases, critical paths: `test_plan.md`
- Original idea, feedback, step-by-step: `ideation/`
