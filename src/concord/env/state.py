import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from concord.exceptions import ConcordError
from concord.schemas.scenario import Scenario

SCHEMA_VERSION = 1


class StateError(ConcordError):
    pass


class StateSchemaMismatchError(StateError):
    pass


def save_state(
    db_path: str | Path,
    episode_id: str,
    scenario: Scenario,
    seed: int,
    current_turn: int,
    current_agent: str,
    terminal: bool,
    turns_data: list[dict[str, Any]],
) -> None:
    db = sqlite3.connect(str(db_path))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        """CREATE TABLE IF NOT EXISTS episode_state (
            episode_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            scenario_yaml TEXT NOT NULL,
            seed INTEGER NOT NULL,
            current_turn INTEGER NOT NULL,
            current_agent TEXT NOT NULL,
            terminal INTEGER NOT NULL,
            turns_json TEXT NOT NULL DEFAULT '[]'
        )"""
    )
    scenario_yaml = yaml.safe_dump(scenario.model_dump(), sort_keys=False)
    db.execute(
        """INSERT OR REPLACE INTO episode_state
           (episode_id, schema_version, scenario_yaml, seed, current_turn, current_agent, terminal, turns_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            episode_id,
            SCHEMA_VERSION,
            scenario_yaml,
            seed,
            current_turn,
            current_agent,
            1 if terminal else 0,
            json.dumps(turns_data),
        ),
    )
    db.commit()
    db.close()


def load_state(db_path: str | Path, episode_id: str) -> dict[str, Any]:
    db = sqlite3.connect(str(db_path))
    row = db.execute(
        "SELECT schema_version, scenario_yaml, seed, current_turn, current_agent, terminal, turns_json FROM episode_state WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    db.close()

    if row is None:
        raise StateError(f"Episode state not found: {episode_id}")

    schema_version, scenario_yaml, seed, current_turn, current_agent, terminal, turns_json = row

    if schema_version != SCHEMA_VERSION:
        raise StateSchemaMismatchError(
            f"Schema version mismatch for {episode_id}: stored={schema_version}, current={SCHEMA_VERSION}"
        )

    scenario = Scenario.model_validate(yaml.safe_load(scenario_yaml))
    return {
        "episode_id": episode_id,
        "schema_version": schema_version,
        "scenario": scenario,
        "seed": seed,
        "current_turn": current_turn,
        "current_agent": current_agent,
        "terminal": bool(terminal),
        "turns": json.loads(turns_json),
    }
