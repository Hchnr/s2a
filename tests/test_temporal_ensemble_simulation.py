"""020 literal physics fork and explicit pre-environment protocol guards."""

import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

import pytest

from s2a.temporal_ensemble_simulation import temporal_variant

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "src/s2a/stability_position_simulation.py"
NEW = ROOT / "src/s2a/temporal_ensemble_simulation.py"
BASE_SHA = "8e807d0fe296899dbda139e143227c23439b579c8de725a2363608f9bee3b136"

CHANGES = [
    ("def run_position_case(", "def run_temporal_case("),
    (
        "variant = position_variant(config, variant_name, source=source)",
        'variant = temporal_variant(config, variant_name, source=source)\n    if type(sampling_seed) is not int or sampling_seed != 42:\n        raise ValueError("020 requires sampling_seed42")',
    ),
    ('schema_version="s2a.position-execution.v1"', 'schema_version="s2a.temporal-equal-execution.v1"'),
    (
        'implementation_origin="stability_recovery_simulation.py at d8421f9; explicit position/hold adapter version"',
        'implementation_origin="literal frozen stability_position_simulation.run_position_case;020 causal temporal reference intervention"',
    ),
    (
        'raw_checkpoint_reference=bool(source == "wam" and variant["model_rotation_alpha"] == 1)',
        "raw_checkpoint_reference=False",
    ),
    (
        'reference_intervention="raw" if variant["model_rotation_alpha"] == 1 else "model_rotation_attenuation",',
        'reference_intervention="causal_temporal_equal_ensemble",\n        reference_ensemble=asdict(TemporalEqualConfig(**config["reference_ensemble"])),\n        original_single_prediction_raw_source_preserved=True,',
    ),
    (
        "        previous_command = np.zeros(7)",
        '        ensemble = CausalTemporalEqualEnsemble(config=TemporalEqualConfig(**config["reference_ensemble"]))\n        previous_command = np.zeros(7)',
    ),
    (
        "                    after = np.asarray(env.sim.get_state().flatten()).copy()",
        "                    ensemble.push(step, full_effective_targets, full_openings)\n                    after = np.asarray(env.sim.get_state().flatten()).copy()",
    ),
    (
        "                desired = desired_tcp @ h_inv",
        "                desired_tcp, desired_opening, temporal_info = ensemble.sample(step)\n                desired = desired_tcp @ h_inv",
    ),
    (
        "                    step=step,\n                    chunk_index=",
        "                    step=step,\n                    temporal_ensemble=temporal_info,\n                    chunk_index=",
    ),
    (
        '        raw_checkpoint_reference=manifest["raw_checkpoint_reference"],',
        '        raw_checkpoint_reference=manifest["raw_checkpoint_reference"],\n        reference_intervention=manifest["reference_intervention"],',
    ),
]


def source_proof():
    """Return reviewable exact AST/diff evidence without executing physics."""
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
    old, new = BASE.read_text(), NEW.read_text()
    old = old[old.index("def run_position_case") :]
    new = new[new.index("def run_temporal_case") :]
    expected = old
    for before, after in CHANGES:
        assert expected.count(before) == 1, before
        expected = expected.replace(before, after)
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(new))
    return dict(
        frozen_position_sha256=BASE_SHA,
        temporal_physical_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),
        all_other_physical_function_ast_nodes_exact=True,
        environment_initialization_controller_motor_physics_scoring_unchanged=True,
        raw_latest128_and_world_paths_unchanged=True,
        explicit_changes=[
            "fixed020 pre-environment guards",
            "honest reference intervention labels",
            "new per-case ensemble; full decoded world path push during paused generation",
            "same-endpoint ensemble admitted target before original inverse/gripper",
            "per-step provenance and result intervention label",
        ],
        source_unified_diff="".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile="frozen:run_position_case",
                tofile="020:run_temporal_case",
            )
        ),
    )


def test_complete_physical_function_has_only_explicit_reference_and_label_changes():
    assert source_proof()["all_other_physical_function_ast_nodes_exact"]


def test_all_registered020_configs_share_the_same_policy():
    paths = [ROOT / "configs/temporal_equal_qualification_20261021.json"]
    paths += [ROOT / f"configs/wam_temporal_equal_task{i}_20261021.json" for i in range(3)]
    for path in paths:
        config = json.loads(path.read_text())
        source = "oracle" if "qualification" in path.name else "wam"
        assert temporal_variant(config, "temporal_equal", source=source) == config["variants"][0]


@pytest.mark.parametrize("change", ["governor", "alpha", "interval", "label", "seed", "gain"])
def test_unregistered_execution_changes_rejected_before_any_environment(change):
    config = copy.deepcopy(json.loads((ROOT / "configs/temporal_equal_qualification_20261021.json").read_text()))
    if change == "governor":
        config["variants"][0]["governor"] = "recovering"
    elif change == "alpha":
        config["variants"][0]["model_rotation_alpha"] = 0.5
    elif change == "interval":
        config["max_observation_interval_steps"] = 16
    elif change == "label":
        config["model_reference_label"] = "raw"
    elif change == "seed":
        config["sampling_seeds"] = [43]
    else:
        config["inverse_weight"] = 0.6
    with pytest.raises(ValueError):
        temporal_variant(config, "temporal_equal", source="oracle")
