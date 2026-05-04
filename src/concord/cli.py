import click

from concord import __version__


@click.group()
@click.version_option(__version__, prog_name="concord")
def main() -> None:
    """Concord — evaluate agentic LLMs in multi-turn negotiations."""


@main.command()
@click.option("--model", required=True, help="Model ID to evaluate")
@click.option("--scenario", required=True, help="Scenario ID")
@click.option("--seed", type=int, default=42, help="Random seed")
def run(model: str, scenario: str, seed: int) -> None:
    """Run a single negotiation episode."""
    click.echo(f"Concord v{__version__} — run: model={model} scenario={scenario} seed={seed}")


if __name__ == "__main__":
    main()
