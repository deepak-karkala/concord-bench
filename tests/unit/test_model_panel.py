import json

from click.testing import CliRunner

from concord.cli import main
from concord.data import model_panel as model_panel_module


def _sample_catalog() -> list[dict]:
    return [
        {
            "id": "anthropic/claude-haiku-4.5",
            "canonical_slug": "anthropic/claude-haiku-4.5",
            "name": "Claude Haiku 4.5",
            "created": 1760486400,
            "context_length": 200000,
            "pricing": {"prompt": "0.000001", "completion": "0.000005"},
        },
        {
            "id": "anthropic/claude-sonnet-4.6",
            "canonical_slug": "anthropic/claude-sonnet-4.6",
            "name": "Claude Sonnet 4.6",
            "created": 1771286400,
            "context_length": 1000000,
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            "id": "anthropic/claude-opus-4.7",
            "canonical_slug": "anthropic/claude-opus-4.7",
            "name": "Claude Opus 4.7",
            "created": 1776460800,
            "context_length": 1000000,
            "pricing": {"prompt": "0.000015", "completion": "0.000075"},
        },
        {
            "id": "openai/gpt-5.5",
            "canonical_slug": "openai/gpt-5.5",
            "name": "GPT-5.5",
            "created": 1773782400,
            "context_length": 400000,
            "pricing": {"prompt": "0.00001", "completion": "0.00003"},
        },
        {
            "id": "openai/gpt-mini",
            "canonical_slug": "openai/gpt-mini",
            "name": "GPT Mini",
            "created": 1773782401,
            "context_length": 200000,
            "pricing": {"prompt": "0.000001", "completion": "0.000003"},
        },
        {
            "id": "openai/gpt-nano",
            "canonical_slug": "openai/gpt-nano",
            "name": "GPT Nano",
            "created": 1773782402,
            "context_length": 128000,
            "pricing": {"prompt": "0.0000005", "completion": "0.0000015"},
        },
        {
            "id": "google/gemini-3.1-pro",
            "canonical_slug": "google/gemini-3.1-pro",
            "name": "Gemini 3.1 Pro",
            "created": 1771100000,
            "context_length": 1000000,
            "pricing": {"prompt": "0.000007", "completion": "0.000021"},
        },
        {
            "id": "google/gemini-3.2-flash",
            "canonical_slug": "google/gemini-3.2-flash",
            "name": "Gemini 3.2 Flash",
            "created": 1772100000,
            "context_length": 1000000,
            "pricing": {"prompt": "0.000001", "completion": "0.000004"},
        },
    ]


def test_freeze_phase1_panel_confirms_all_slots():
    manifest = model_panel_module.freeze_model_panel("phase1", _sample_catalog())

    assert manifest["panel"] == "phase1"
    assert manifest["summary"] == {
        "total_slots": 4,
        "confirmed_slots": 4,
        "fallback_required_slots": 0,
        "deferred_slots": 0,
    }
    assert [slot["slot_name"] for slot in manifest["slots"]] == [
        "baseline_greedy",
        "anthropic_fast",
        "anthropic_strong",
        "openai_frontier",
    ]
    assert manifest["slots"][0]["backend"] == "scripted"
    assert manifest["slots"][1]["model_id"] == "openrouter/anthropic/claude-haiku-4.5"
    assert manifest["slots"][2]["model_id"] == "openrouter/anthropic/claude-sonnet-4.6"
    assert manifest["slots"][3]["model_id"] == "openrouter/openai/gpt-5.5"


def test_freeze_phase15_marks_missing_slots_deferred():
    partial_catalog = _sample_catalog()[:3]

    manifest = model_panel_module.freeze_model_panel("phase15", partial_catalog)

    assert manifest["summary"]["total_slots"] == 9
    assert manifest["summary"]["confirmed_slots"] == 4
    assert manifest["summary"]["deferred_slots"] == 5

    deferred = {
        slot["slot_name"]: slot
        for slot in manifest["slots"]
        if slot["status"] == "deferred"
    }
    assert "openai_frontier" in deferred
    assert deferred["openai_frontier"]["model_id"] is None
    assert deferred["openai_frontier"]["backend"] == "openrouter"
    assert deferred["openai_frontier"]["candidate_slugs"] == ["openai/gpt-5.5", "openai/gpt-5"]


def test_cli_freeze_model_panel_writes_manifest(temp_dir, monkeypatch):
    output_path = temp_dir / "phase1_manifest.json"

    monkeypatch.setattr(
        model_panel_module,
        "fetch_openrouter_models",
        lambda api_key: _sample_catalog(),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "freeze-model-panel",
            "--panel",
            "phase1",
            "--output",
            str(output_path),
            "--api-key",
            "test-key",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    manifest = json.loads(output_path.read_text())
    assert manifest["panel"] == "phase1"
    assert manifest["summary"]["confirmed_slots"] == 4
    assert manifest["slots"][1]["model_id"] == "openrouter/anthropic/claude-haiku-4.5"
