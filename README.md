# Concord Bench

Concord is a Python benchmark and execution environment for evaluating whether LLM agents can act as faithful representatives for a principal in multi-turn negotiations. It focuses on principal utility, private-information discipline, hard constraints, rational walk-away behavior, and resistance to seller pressure across structured domain-specific deals.

The current release is a shareable research artifact, not a finished publication benchmark. It includes the package, CLI, bundled seed corpus, graders, reporting utilities, and tests. Raw run outputs and full transcript dumps are intentionally excluded from the GitHub repository.

## What Concord Measures

Concord evaluates buyer-side agent behavior in controlled negotiation episodes:

- representing the buyer principal's goals instead of default conversational preferences
- protecting private information such as BATNA-like facts and sensitive context
- honoring hard constraints while still trying to reach useful deals
- walking away when no rational deal exists
- handling pressure, deceptive framing, and unsafe persuasion attempts
- producing structured offers that can be parsed and graded reproducibly

The benchmark is best framed as a fidelity benchmark for principal-aligned representation under pressure.

## What It Does Not Claim Yet

Do not treat this repository as evidence for broad model-family rankings or publication-grade claims. The current state does not yet support strong claims about:

- broad cultural sensitivity
- fully validated semantic hard-constraint grading
- robust frontier-family rankings
- expert-grade domain realism
- finalized transcript release policy
- large-scale generated-corpus findings

Sellers are scripted controlled policies. Human validation and expert anchor work are still incomplete for the strongest semantic claims.

## Install

Requirements:

- Python 3.12+
- `uv`

```bash
git clone https://github.com/deepak-karkala/concord-bench.git
cd concord-bench
uv sync
```

For development tools:

```bash
uv sync --extra dev
```

Optional extras:

```bash
uv sync --extra synth   # scenario synthesis tooling
uv sync --extra interp  # interpretability experiments
```

API-backed runs require provider credentials such as `OPENROUTER_API_KEY`. Keep secrets in your shell environment or a local `.env`; do not commit them.

## Quick Start

Run one deterministic scripted episode:

```bash
uv run concord run \
  --model greedy \
  --seller honest_cooperative \
  --scenario src/concord/data/seed_yamls/ecommerce/seed-ecommerce-002.yaml \
  --seed 42 \
  --output tmp/greedy_ecommerce_002
```

Run a small scripted batch:

```bash
uv run concord run-batch \
  --models always_walk_away,accept_first_valid,constraint_first_cautious,price_only_rational \
  --scenarios src/concord/data/seed_yamls/ecommerce \
  --seeds 42,43,44 \
  --seller honest_cooperative \
  --output tmp/scripted_batch
```

Generate a report for that batch:

```bash
uv run python scripts/smoke_test_report.py \
  --results-dir tmp/scripted_batch \
  --scenarios-dir src/concord/data/seed_yamls/ecommerce \
  --output tmp/scripted_batch/report
```

## Common Commands

| Command | Purpose |
| --- | --- |
| `uv run concord run --help` | Show single-episode CLI options |
| `uv run concord run-batch --help` | Show batch-run CLI options |
| `uv run python scripts/smoke_test_report.py --help` | Show report-generation options |
| `uv run python scripts/reliability_report.py --help` | Show repeated-run reliability options |
| `uv run pytest tests/unit/ -q -m "not requires_api"` | Run offline unit tests |
| `uv run ruff check src/concord/ tests/ scripts/` | Run lint checks |

## Repository Layout

```text
src/concord/
  agents/       Closed-model and open-weight adapters
  baselines/    Scripted buyer and seller policies
  env/          Negotiation runtime and PettingZoo wrapper
  schemas/      Scenario, observation, offer, episode, and validation schemas
  graders/      Utility, truthfulness, privacy, social, constraint, and validation graders
  runners/      Episode and batch execution
  analysis/     Aggregation, preregistration, validation, and reliability helpers
  data/         Seed scenarios, model panels, and corpus utilities
  synth/        Scenario synthesis and audit helpers

scripts/        Reproducibility, audit, reporting, and validation utilities
tests/          Unit, regression, integration tests, and small fixtures
experiments/    Preregistered experiment scaffolding
```

The bundled seed corpus lives in `src/concord/data/seed_yamls/` and is part of the public package.

## Results And Artifacts

The GitHub repository excludes raw run outputs by default:

- `outputs/`
- `dist/`
- `.venv/`
- caches
- local `.env` files
- internal planning, review, and archive documents

Use `tmp/` or another local output directory for new runs. If you publish results, include the Git SHA, command line, model identifiers, seed list, scenario slice, and report artifacts needed for reproduction.

## Development Notes

The project is still pre-release and versioned as `0.1.0.dev0`. Treat metric status and claim boundaries conservatively:

- only metrics explicitly marked headline-safe should support strong claims
- exploratory metrics should stay labeled exploratory
- unsupported metrics should not be presented as findings
- semantic validation and expert anchoring remain open work

Before sharing a new result, run the relevant tests and regenerate reports from the same source artifacts.

## License

Concord is released under the MIT License. See [LICENSE](LICENSE).
