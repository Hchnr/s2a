"""Check frozen training identities and the 24-case reference roster."""

import copy
import json

import pytest

from scripts.simulation.runtime import prepare_broader_recovery_pilot as run


def test_configuration_rejects_changed_budgets_roster_and_typed_flags():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for key, value in (
        ("max_inner_steps_per_phase", 160801),
        ("max_probe_control_periods", 1),
        ("fresh_cases", 1),
        ("training_authorized", 0),
        ("automatic_retry", True),
    ):
        with pytest.raises(ValueError, match="registered roster"):
            run.validate_config(dict(config, **{key: value}))
    for mutate in (
        lambda data: data["sources"][0].update(sign=True),
        lambda data: data["sources"][0].update(demo="demo_48"),
        lambda data: data["training_preparation"].update(sha256="0" * 64),
        lambda data: data["variant_order"].reverse(),
    ):
        changed = copy.deepcopy(config)
        mutate(changed)
        with pytest.raises(ValueError, match="registered roster"):
            run.validate_config(changed)


@pytest.fixture
def source_identity():
    source = copy.deepcopy(run.SOURCES[0])
    record = dict(task_index=0, task_id=8, demo="demo_40", initial_state_sha256="training-sha")
    spec = dict(index=3, task_id=8, seed=42, periods=170)
    trajectory_ref = dict(path="trajectory", sha256="hash")
    anchor = dict(
        anchor_frame=0,
        source_identity=dict(task_index=0, demo="demo_40", initial_state_sha256="training-sha"),
        trajectory_ref=trajectory_ref,
    )
    trained = dict(
        training_anchors=[anchor],
        exclusion=dict(
            forbidden_demo_keys=[], validation_demo_keys=[], forbidden_initial_state_sha256=[], fresh_sources=[]
        ),
    )
    native = dict(
        record=copy.deepcopy(record),
        complete=True,
        passed=True,
        native_quality_passed=True,
        physics_steps=170,
        arrays=trajectory_ref,
    )
    audit = dict(case=spec, producer_status="closed", checker_status="closed", qualified=True, replayed=True)
    return source, dict(spec=spec), dict(record=record), trained, audit, native


@pytest.mark.parametrize("corruption", ["validation", "forbidden_sha", "fresh_key", "fresh_sha", "duplicate"])
def test_excluded_or_ambiguous_training_identity_is_rejected(source_identity, corruption):
    source, origin, manifest, trained, audit, native = copy.deepcopy(source_identity)
    assert run.validate_source(source, origin, manifest, trained, audit, native) == trained["training_anchors"][0]
    if corruption == "validation":
        trained["exclusion"]["validation_demo_keys"].append([0, "demo_40"])
    elif corruption == "forbidden_sha":
        trained["exclusion"]["forbidden_initial_state_sha256"].append("training-sha")
    elif corruption == "fresh_key":
        trained["exclusion"]["fresh_sources"].append(dict(task_index=0, demo="demo_40", initial_state_sha256="other"))
    elif corruption == "fresh_sha":
        trained["exclusion"]["fresh_sources"].append(
            dict(task_index=1, demo="other", initial_state_sha256="training-sha")
        )
    else:
        trained["training_anchors"].append(copy.deepcopy(trained["training_anchors"][0]))
    with pytest.raises(ValueError, match="outside validation and fresh"):
        run.validate_source(source, origin, manifest, trained, audit, native)


@pytest.mark.parametrize("corruption", ["not_replayed", "wrong_native", "wrong_trajectory", "wrong_initial"])
def test_unqualified_or_mismatched_native_evidence_is_rejected(source_identity, corruption):
    source, origin, manifest, trained, audit, native = copy.deepcopy(source_identity)
    if corruption == "not_replayed":
        audit["replayed"] = False
    elif corruption == "wrong_native":
        native["physics_steps"] -= 1
    elif corruption == "wrong_trajectory":
        native["arrays"] = dict(path="different", sha256="hash")
    else:
        native["record"]["initial_state_sha256"] = "different"
    with pytest.raises(ValueError, match="independently replayed"):
        run.validate_source(source, origin, manifest, trained, audit, native)


def test_real_sources_generate_registered_24_case_identity_without_physics():
    config, identity = run.load_inputs()
    cases = identity["cases"]
    assert len(cases) == 24
    assert [row["variant"] for row in cases] == list(run.VARIANTS) * 6
    assert [row["index"] for row in cases] == list(range(24))
    assert len({row["initial_state_sha256"] for row in cases}) == 6
    assert sum(row["periods"] for row in cases) == config["max_periods_per_phase"] == 6432
    assert all(row["timeline_metadata"]["physical_labels_from_reference"] is False for row in cases)
    assert all(len(row["timeline_sha256"]) == 64 for row in cases)
    assert config["training_authorized"] is False and config["fresh_cases"] == 0
