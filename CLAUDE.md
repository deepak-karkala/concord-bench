# CLAUDE.md

## Project

Concord — canonical environment for evaluating and training agentic LLMs in multi-turn negotiations.

## Commands

```bash
uv sync                                  # Install base deps
uv sync --extra dev                       # Install dev tools (pytest, ruff, mypy)
uv sync --extra synth --extra interp      # Install all optional deps
uv build                                 # Build wheel
uv run pytest                            # Run all tests
uv run pytest tests/unit/                # Unit tests only
uv run pytest tests/integration/         # Integration tests
uv run ruff check src/concord/ tests/    # Lint
uv run mypy src/concord/                 # Type check
uv run concord --help                    # CLI
```

## Architecture

- `src/concord/schemas/` — Pydantic v2 models (Scenario, Offer, EpisodeLog, CulturalProfile)
- `src/concord/synth/` — Synthesis pipeline (AWM enrichment, cultural adapter, repeated game). Imports AWM lazily.
- `src/concord/data/` — Scenario loader (HF Hub + bundled seeds), seed YAMLs
- `src/concord/env/` — NegotiationEnv (turn-based), PettingZoo wrapper, offer parser, SQLite state
- `src/concord/agents/` — Agent protocol, closed-API adapters, open-weight adapter, retry logic
- `src/concord/graders/` — Utility, constraints, truthfulness, privacy, social, calibration
- `src/concord/baselines/` — Scripted opponents (random, greedy, honest, deceptive, time-pressured)
- `src/concord/runners/` — Episode runner, batch runner, LLM cache, budget cap
- `src/concord/analysis/` — Aggregation, bootstrap CI, model cards, plots
- `src/concord/cli.py` — Click-based CLI

## AWM is optional

AWM is a PyPI dep under the `[synth]` extra. Never import it at module load. All synth modules lazily import AWM:

```python
# src/concord/synth/__init__.py
try:
    import awm
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Concord scenario generation requires the [synth] extra. "
        "Install with: pip install concord-bench[synth]"
    )
```

## Conventions

- Python 3.12+, type hints everywhere, Pydantic v2
- `snake_case` for files/functions, `PascalCase` for models/classes
- Google-style docstrings on public API
- `model_validate()` not `parse_obj()`, `model_dump()` not `dict()`
- Async for all LLM calls and batch runners
- Custom exception hierarchy under `ConcordError`
- 100% branch coverage on graders (non-negotiable)
