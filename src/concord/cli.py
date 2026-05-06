import asyncio
import json
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
@click.option("--scenarios", required=True, help="Path to scenarios dir, domain name, or 'all'")
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

    scenarios_path = Path(scenarios)
    if scenarios_path.exists() and scenarios_path.is_dir():
        scenario_list = load_seeds(seed_dir=scenarios_path)
    elif scenarios == "all":
        scenario_list = load_seeds()
    else:
        scenario_list = load_seeds(domain=scenarios)

    if not scenario_list:
        raise click.ClickException("No scenarios found. Run 'concord generate' first or use seed scenarios.")

    out_path = Path(output)
    out_path.mkdir(parents=True, exist_ok=True)

    for model in model_list:
        if not ctx.obj.get("quiet"):
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
        model_dir = out_path / model.replace("/", "_").replace(":", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        for ep in results:
            ep_path = model_dir / f"{ep.scenario_id}_{ep.metadata.get('seed', 0)}.json"
            with open(ep_path, "w") as f:
                json.dump(ep.model_dump(), f, indent=2, default=str)
        if not ctx.obj.get("quiet"):
            click.echo(f"  Completed: {len(results)} episodes for {model}")

    if not ctx.obj.get("quiet"):
        click.echo(f"Results saved to {output}")


@main.command()
@click.option("--domain", default="all",
    type=click.Choice(["all", "ecommerce", "saas_procurement", "settlement", "ethical_business"]),
    help="Domain filter")
@click.option("--culture", default="all",
    type=click.Choice(["all", "US", "JP", "IN", "BR", "MENA"]),
    help="Culture filter (all = 5 cultures)")
@click.option("--count", type=int, default=6000, help="Target scenario count")
@click.option("--awm-count", type=int, default=800, help="AWM base scenarios before enrichment")
@click.option("--output", type=click.Path(), default="outputs/scenarios", help="Output directory")
@click.option("--model", default="deepseek-v4-pro", help="Model for AWM generation")
@click.option("--enrich-model", default="deepseek-v4-pro", help="Model for narrative enrichment")
@click.option("--repeated-game/--no-repeated-game", default=True, help="Generate repeated-game sequences")
@click.option("--repeated-count", type=int, default=40, help="Seeds to expand into 5-round sequences")
@click.option("--dry-run", is_flag=True, help="Estimate cost without making API calls")
@click.pass_context
def generate(
    ctx: click.Context,
    domain: str,
    culture: str,
    count: int,
    awm_count: int,
    output: str,
    model: str,
    enrich_model: str,
    repeated_game: bool,
    repeated_count: int,
    dry_run: bool,
) -> None:
    """Generate negotiation scenarios from seeds using the synthesis pipeline."""
    import yaml

    if dry_run:
        _print_cost_estimate(awm_count, count, repeated_game, repeated_count, culture, enrich_model)
        return

    try:
        from concord.synth.cultural_adapter import adapt_for_culture
        from concord.synth.repeated_game import generate_repeated_sequence
        from concord.schemas.culture import Culture
    except ImportError as e:
        raise click.ClickException(str(e))

    all_seeds = load_seeds(domain=domain if domain != "all" else None)
    if not all_seeds:
        raise click.ClickException("No seed scenarios found. Add seeds to concord/data/seed_yamls/")

    t3_seeds = [s for s in all_seeds if s.metadata.get("difficulty_tier") == 3]
    t0_t2_seeds = [s for s in all_seeds if s.metadata.get("difficulty_tier", 1) < 3]

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    cultures = ["US", "JP", "IN", "BR", "MENA"] if culture == "all" else [culture]

    if awm_count > 0:
        try:
            from concord.synth import _check_awm
            _check_awm()
            from concord.synth.enrichment import enrich_awm_scenario
            awm_base = _run_awm_generation(awm_count, domain, model, ctx)
            if not ctx.obj.get("quiet"):
                click.echo(f"AWM generated {len(awm_base)} base scenarios")
            enriched = [enrich_awm_scenario(s, s.get("domain", domain if domain != "all" else "ecommerce"), "US")
                        for s in awm_base]
        except ModuleNotFoundError:
            click.echo("Warning: AWM not available — skipping AWM generation. Install [synth] extra.", err=True)
            enriched = []
    else:
        enriched = []

    base_scenarios = enriched + t0_t2_seeds
    if not ctx.obj.get("quiet"):
        click.echo(f"Cultural adaptation: {len(base_scenarios)} base × {len(cultures)} cultures")

    for scenario in base_scenarios:
        for cult in cultures:
            try:
                adapted = adapt_for_culture(scenario, Culture(cult)) if cult != getattr(scenario, "culture", "US") else scenario
            except Exception:
                adapted = scenario
            adapted_id = f"{adapted.id}-{cult}" if cult != "US" else adapted.id
            path = out_dir / f"{adapted_id}.yaml"
            dump_data = adapted.model_dump(mode="json")
            dump_data["id"] = adapted_id
            with open(path, "w") as f:
                yaml.safe_dump(dump_data, f, sort_keys=False, default_flow_style=False)
            generated += 1
            if generated >= count:
                break
        if generated >= count:
            break

    for s in t3_seeds:
        path = out_dir / f"{s.id}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(s.model_dump(mode="json"), f, sort_keys=False, default_flow_style=False)
        generated += 1

    if repeated_game:
        repeat_seeds = [s for s in t0_t2_seeds
                       if s.buyer_context.walk_away_threshold is not None
                       and s.seller_context.walk_away_threshold is not None
                       and s.buyer_context.relationship_history][:repeated_count]
        for s in repeat_seeds:
            for cult in cultures:
                try:
                    base = adapt_for_culture(s, Culture(cult)) if cult != getattr(s, "culture", "US") else s
                except Exception:
                    base = s
                sequences = generate_repeated_sequence(base, num_rounds=5)
                for r_scenario in sequences:
                    r_id = f"{r_scenario.id}-{cult}" if cult != "US" else r_scenario.id
                    path = out_dir / f"{r_id}.yaml"
                    dump_data = r_scenario.model_dump(mode="json")
                    dump_data["id"] = r_id
                    with open(path, "w") as f:
                        yaml.safe_dump(dump_data, f, sort_keys=False, default_flow_style=False)
                    generated += 1

    if not ctx.obj.get("quiet"):
        click.echo(f"Generated {generated} scenarios in {output}")


def _run_awm_generation(target_count: int, domain: str, model: str, ctx: click.Context) -> list[dict]:
    """Run AWM generation and return raw scenario dicts."""
    try:
        import awm
        from concord.synth.awm_prompts import AWM_SYSTEM_PROMPT, AWM_DOMAIN_HINTS
    except ImportError:
        return []

    domains = ["ecommerce", "saas_procurement", "settlement", "ethical_business"] if domain == "all" else [domain]
    per_domain = max(1, target_count // len(domains))
    results = []

    for d in domains:
        hint = AWM_DOMAIN_HINTS.get(d, "")
        combined_prompt = AWM_SYSTEM_PROMPT + "\n\n" + hint
        try:
            scenarios = awm.ScenarioSelfInstruct(
                system_prompt=combined_prompt,
                model=model,
                temperature=1.0,
                num_scenarios=per_domain,
            ).generate()
            results.extend(scenarios)
        except Exception as e:
            if not ctx.obj.get("quiet"):
                click.echo(f"AWM generation failed for {d}: {e}", err=True)

    return results


def _print_cost_estimate(
    awm_count: int,
    count: int,
    repeated_game: bool,
    repeated_count: int,
    culture: str,
    enrich_model: str,
) -> None:
    cultures = 5 if culture == "all" else 1
    base_scenarios = awm_count + 192  # AWM + T0-T2 seeds
    cultural_calls = base_scenarios * (cultures - 1)  # US is free (no LLM call)
    enrich_calls = awm_count
    repeat_calls = repeated_count * cultures * 5 if repeated_game else 0

    dsv4_in = 0.003625 / 1_000_000   # deepseek-v4-pro input
    dsv4_out = 0.87 / 1_000_000      # deepseek-v4-pro output
    awm_api_calls = awm_count / 10    # AWM generates 10 scenarios per API call
    awm_cost = awm_api_calls * (2600 * dsv4_in + 3000 * dsv4_out)
    enrich_cost = enrich_calls * (500 * dsv4_in + 150 * dsv4_out)
    cultural_cost = cultural_calls * (700 * dsv4_in + 500 * dsv4_out)
    repeat_cost = repeat_calls * (300 * dsv4_in + 200 * dsv4_out)
    total = awm_cost + enrich_cost + cultural_cost + repeat_cost

    click.echo("Cost estimate (dry run):")
    click.echo(f"  AWM generation ({awm_count} scenarios):        ${awm_cost:.2f}")
    click.echo(f"  Narrative enrichment ({enrich_calls} calls):   ${enrich_cost:.2f}")
    click.echo(f"  Cultural adaptation ({cultural_calls} calls):  ${cultural_cost:.2f}")
    click.echo(f"  Repeated-game ({repeat_calls} scenarios):      ${repeat_cost:.2f}")
    click.echo("  ─────────────────────────────────────────────")
    click.echo(f"  Total estimated cost:                          ${total:.2f}")
    click.echo(f"  Target scenario count:                         ~{count}")


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
