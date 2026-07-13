"""Unit tests for the red-team generator, validator and safety guards."""
import json
import tempfile
from pathlib import Path

from src import config
from src.redteam import (dedupe, generate_scenarios, validate_scenario,
                         write_redteam)


def test_generation_is_deterministic():
    a = generate_scenarios(30, seed=7)
    b = generate_scenarios(30, seed=7)
    assert [s["text"] for s in a] == [s["text"] for s in b]


def test_all_generated_scenarios_are_valid_and_tagged_synthetic():
    for s in generate_scenarios(30):
        assert validate_scenario(s) == []
        assert s["provenance"] == "redteam_synthetic"


def test_covers_the_three_fake_categories_plus_controls():
    cats = {s["category"] for s in generate_scenarios(30)}
    assert {"plain_fluent", "authority_fluent", "trope_laden", "real_control"} <= cats


def test_fake_and_real_labels_are_assigned_correctly():
    scen = generate_scenarios(9)
    assert all(s["label"] == "FAKE" for s in scen if s["category"] != "real_control")
    assert all(s["label"] == "REAL" for s in scen if s["category"] == "real_control")


def test_validate_flags_missing_fields_and_bad_provenance():
    assert "missing 'text'" in validate_scenario({"label": "FAKE", "provenance": "redteam_synthetic"})
    bad = validate_scenario({"text": "x", "label": "MAYBE", "style": "s",
                             "category": "c", "domain": "d", "provenance": "external"})
    assert any("label must be" in p for p in bad)
    assert any("provenance must be" in p for p in bad)


def test_dedupe_removes_normalized_duplicates():
    new = [{"text": "The  Mayor  Banned  Bikes"}, {"text": "the mayor banned bikes"}]
    assert len(dedupe(new, set())) == 1


def test_write_refuses_the_citable_benchmark():
    raised = False
    try:
        write_redteam(generate_scenarios(3), path=config.SCENARIOS_FILE)
    except ValueError as exc:
        raised = "citable benchmark" in str(exc)
    assert raised, "write_redteam must refuse the citable benchmark path"


def test_write_appends_and_dedupes():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "redteam.json"
        first = write_redteam(generate_scenarios(9, seed=1), path=path)
        assert first["added"] > 0
        # Re-writing the same generated set adds nothing new (dedup on text).
        second = write_redteam(generate_scenarios(9, seed=1), path=path)
        assert second["added"] == 0
        data = json.loads(path.read_text())
        assert "NOT the" in data["description"]  # documents it is not the citable set
