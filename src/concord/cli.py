import asyncio
import json
import sys
from pathlib import Path

import click

from concord import __version__
from concord.data.loader import load_scenarios, load_seeds
from concord.schemas.scenario import Scenario


@click.group()
@click.version_option(__version__, prog_name="concord")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress non-error output")
@click.pass_context
def main(ctx: click.Context, quiet: bool) -> None:
    """Concord — evaluate agentic LLMs in multi-turn negotiations."""
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet


@main.command()
@click.option("--model", required=True, help="Model ID (e.g., greedy, honest, gpt-5.2)")
@click.option("--scenario", required=True, help="Scenario ID or path to seed YAML")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--output", type=click.Path(), help="Output directory for episode log")
@click.pass_context
def run(ctx: click.Context, model: str, scenario: str, seed: int, output: str | None) -> None:
    """Run a single negotiation episode between two scripted or API agents."""
    try:
        from concord.runners.run_episode import run_episode
    except ImportError as e:
        raise click.ClickException(f"Failed to import runner: {e}")

    target = _find_scenario(scenario)
    if target is None:
        raise click.ClickException(f"Scenario not found: {scenario}")

    async def _run():
        return await run_episode(target, buyer_model=model, seller_model="greedy", seed=seed)

    episode = asyncio.run(_run())
    if not ctx.obj.get("quiet"):
        click.echo(f"Episode: {episode.scenario_id}")
        click.echo(f"Turns: {len(episode.turns)}")
        click.echo(f"Deal reached: {episode.deal is not None}")
        click.echo(json.dumps(episode.metadata, indent=2))

    if output:
        out_path = Path(output) / "episode.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(episode.model_dump(), f, indent=2, default=str)


@main.command(name="run-batch")
@click.option("--models", required=True, help="Comma-separated model IDs (e.g., greedy,gpt-5.2)")
@click.option("--scenarios", required=True, help="Domain or 'all' for all seed scenarios")
@click.option("--seeds", default="42", help="Comma-separated seeds")
@click.option("--concurrency", type=int, default=10, help="Max concurrent episodes")
@click.option("--budget-cap", type=float, help="Daily API budget cap in USD")
@click.option("--output", type=click.Path(), default="outputs/batch", help="Output directory")
@click.pass_context
def run_batch(
    ctx: click.Context,
    models: str,
    scenarios: str,
    seeds: str,
    concurrency: int,
    budget_cap: float | None,
    output: str,
) -> None:
    """Run batch evaluation across multiple models and scenarios."""
    try:
        from concord.runners.run_batch import run_batch
    except ImportError as e:
        raise click.ClickException(f"Failed to import batch runner: {e}")

    model_list = [m.strip() for m in models.split(",")]
    seed_list = [int(s.strip()) for s in seeds.split(",")]

    if scenarios == "all":
        scenario_list = load_seeds()
    else:
        scenario_list = load_seeds(domain=scenarios)

    if not scenario_list:
        raise click.ClickException("No scenarios found. Run 'concord generate' first or use seed scenarios.")

    for model in model_list:
        click.echo(f"Running: model={model}, {len(scenario_list)} scenarios, concurrency={concurrency}")
        results = asyncio.run(
            run_batch(
                scenario_list,
                buyer_model=model,
                seller_model="greedy",
                seeds=seed_list,
                concurrency=concurrency,
                budget_cap=budget_cap,
            )
        )
        if not ctx.obj.get("quiet"):
            click.echo(f"  Completed: {len(results)} episodes for {model}")

    if not ctx.obj.get("quiet"):
        click.echo(f"Results saved to {output}")


@main.command()
@click.option("--domain", help="Domain filter (ecommerce, saas_procurement, settlement, ethical_business)")
@click.option("--culture", help="Culture filter (US, JP, IN, BR, MENA)")
@click.option("--count", type=int, default=100, help="Number of scenarios to generate")
@click.option("--output", type=click.Path(), default="outputs/scenarios", help="Output directory")
@click.pass_context
def generate(ctx: click.Context, domain: str | None, culture: str | None, count: int, output: str) -> None:
    """Generate negotiation scenarios from seeds using the synthesis pipeline."""
    try:
        from concord.synth import _check_awm
        _check_awm()
        from concord.synth.enrichment import enrich_awm_scenario
        from concord.synth.cultural_adapter import adapt_for_culture
        from concord.synth.repeated_game import generate_repeated_sequence
        from concord.schemas.culture import Culture
    except ModuleNotFoundError as e:
        raise click.ClickException(
            "Scenario generation requires the [synth] extra. "
            "Install with: pip install concord-bench[synth]"
        )
    except ImportError as e:
        raise click.ClickException(str(e))

    seeds = load_seeds(domain=domain) if domain else load_seeds()
    if not seeds:
        raise click.ClickException("No seed scenarios found. Add seeds to concord/data/seed_yamls/")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    import yaml

    for seed in seeds:
        for cult in (Culture if not culture else [Culture(culture)] if culture in list(Culture) else [seed.culture]):
            adapted = adapt_for_culture(seed, cult) if cult != seed.culture else seed
            path = out_dir / f"{adapted.id}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(adapted.model_dump(mode="json"), f, sort_keys=False, default_flow_style=False)
            generated += 1

        if len(seeds) <= 10:
            repeated = generate_repeated_sequence(seed, num_rounds=5)
            for r_scenario in repeated:
                path = out_dir / f"{r_scenario.id}.yaml"
                with open(path, "w") as f:
                    yaml.safe_dump(r_scenario.model_dump(mode="json"), f, sort_keys=False, default_flow_style=False)
                generated += 1

        if generated >= count:
            break

    if not ctx.obj.get("quiet"):
        click.echo(f"Generated {generated} scenarios in {output}")


@main.command()
@click.option("--episode-log", required=True, type=click.Path(exists=True), help="Path to episode JSON log")
@click.option("--scenario", type=click.Path(exists=True), help="Scenario YAML path (needed for utility and constraint grades)")
@click.option("--output", type=click.Path(), help="Output path for grade report")
@click.pass_context
def grade(ctx: click.Context, episode_log: str, scenario: str | None, output: str | None) -> None:
    """Re-grade a saved episode log and output the score report."""
    import yaml as _yaml

    with open(episode_log) as f:
        data = json.load(f)

    from concord.schemas.episode import EpisodeLog
    from concord.schemas.scenario import Scenario
    from concord.graders.utility import compute_principal_utility
    from concord.graders.constraints import check_hard_constraints
    from concord.graders.privacy import detect_batna_leak
    from concord.graders.social import detect_coercion

    try:
        ep = EpisodeLog.model_validate(data)
    except Exception as e:
        raise click.ClickException(f"Invalid episode log: {e}")

    transcript = [{"content": t.content, "agent": t.agent} for t in ep.turns]
    deal = ep.deal

    private_ctx = None
    if scenario:
        with open(scenario) as f:
            sc = Scenario.model_validate(_yaml.safe_load(f))
        private_ctx = sc.buyer_context

    grades = {
        "utility": compute_principal_utility(deal, private_ctx) if (deal and private_ctx) else None,
        "constraint_violations": check_hard_constraints(deal, private_ctx) if (deal and private_ctx) else [],
        "batna_leak": detect_batna_leak(transcript, private_ctx) if private_ctx else None,
        "coercion": detect_coercion(transcript),
    }

    if not ctx.obj.get("quiet"):
        click.echo(json.dumps(grades, indent=2, default=str))

    if output:
        with open(output, "w") as f:
            json.dump(grades, f, indent=2, default=str)


@main.command()
@click.option("--format", "fmt", type=click.Choice(["hf", "jsonl"]), default="hf", help="Export format")
@click.option("--input", "input_dir", type=click.Path(exists=True), required=True, help="Scenarios directory")
@click.option("--output", type=click.Path(), default="outputs/export", help="Output directory")
@click.pass_context
def export(ctx: click.Context, fmt: str, input_dir: str, output: str) -> None:
    """Export scenarios to Hugging Face dataset format."""
    import yaml

    in_path = Path(input_dir)
    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)

    yaml_files = list(in_path.rglob("*.yaml"))
    if not yaml_files:
        raise click.ClickException(f"No YAML files found in {input_dir}")

    if fmt == "jsonl":
        import json
        with open(out_path / "scenarios.jsonl", "w") as outf:
            for yf in yaml_files:
                with open(yf) as inf:
                    data = yaml.safe_load(inf)
                outf.write(json.dumps(data) + "\n")
    else:
        for yf in yaml_files:
            rel = yf.relative_to(in_path)
            dest = out_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(yf.read_text())

    if not ctx.obj.get("quiet"):
        click.echo(f"Exported {len(yaml_files)} scenarios to {output} ({fmt})")


@main.command()
@click.option("--transcripts", required=True, type=click.Path(exists=True), help="Directory of calibration transcript JSONL files")
@click.option("--judge", default="coercion", help="Judge dimension to calibrate")
@click.option("--output", type=click.Path(), help="Output path for calibration report")
@click.pass_context
def calibrate(ctx: click.Context, transcripts: str, judge: str, output: str | None) -> None:
    """Run calibration scoring on hand-labeled transcripts."""
    import json as _json

    from concord.graders.calibration import compute_cohens_kappa
    from concord.graders.social import detect_coercion

    t_dir = Path(transcripts)
    judge_scores: list[float] = []
    author_labels: list[float] = []

    for tf in sorted(t_dir.glob("*.jsonl")):
        data = _json.loads(tf.read_text())
        if judge == "coercion":
            score = detect_coercion(data["transcript"])
            judge_scores.append(score)
            author_labels.append(data["labels"]["coercion_level"])

    if not judge_scores:
        raise click.ClickException("No transcripts found for calibration")

    kappa = compute_cohens_kappa(judge_scores, author_labels)

    if not ctx.obj.get("quiet"):
        click.echo(f"Judge: {judge}")
        click.echo(f"Cohen's kappa: {kappa:.4f}")
        click.echo(f"Samples: {len(judge_scores)}")

    if output:
        with open(output, "w") as f:
            _json.dump({"judge": judge, "kappa": kappa, "n": len(judge_scores)}, f, indent=2)


def _find_scenario(scenario_id: str) -> Scenario | None:
    """Find a scenario by ID from seed YAMLs."""
    for s in load_seeds():
        if s.id == scenario_id:
            return s

    yaml_path = Path(scenario_id)
    if yaml_path.exists() and yaml_path.suffix in (".yaml", ".yml"):
        import yaml
        with open(yaml_path) as f:
            return Scenario.model_validate(yaml.safe_load(f))

    try:
        return load_scenarios(domain=scenario_id)[:1] or None
    except Exception:
        return None
