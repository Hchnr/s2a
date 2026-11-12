"""Compare binding-free historical inputs and reject changed registrations."""

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import step107_model_inputs as admission


def test_fresh_process_matches_bound_historical_inputs_without_mutating_globals():
    script = """\
import sys
from scripts.simulation.runtime import step107_model_inputs as admission
from scripts.simulation.runtime import run_causal_replan_step107_model as historical

old = historical.previous
older = old.previous
before = (old.CONFIG, old.PLAN, old.OUTPUT, old.source_refs,
          older.CONFIG, older.PLAN, older.OUTPUT, older.load_inputs, older.source_refs)
actual = admission.load_inputs()
after = (old.CONFIG, old.PLAN, old.OUTPUT, old.source_refs,
         older.CONFIG, older.PLAN, older.OUTPUT, older.load_inputs, older.source_refs)
assert before == after
expected = historical.bind().load_inputs()
assert actual == expected
assert not any(name in sys.modules for name in ('torch', 'mujoco', 'robosuite', 'libero', 'cosmos_framework'))
"""
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        LP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:.",
    )
    subprocess.run([sys.executable, "-c", script], cwd=admission.prefix.REPO, env=env, check=True, timeout=900)


@pytest.mark.parametrize("change", ["seed", "model_sha"])
def test_changed_historical_config_is_rejected_before_capture_or_model_access(monkeypatch, change):
    config = copy.deepcopy(admission.historical.FIXED)
    config[change] = 43 if change == "seed" else "f" * 64
    monkeypatch.setattr(admission.prefix, "read", lambda _ref: config)
    monkeypatch.setattr(admission.prefix, "reference", lambda _path: {})
    monkeypatch.setattr(admission.historical, "validate_capture", lambda _value: pytest.fail("capture was accessed"))
    monkeypatch.setattr(admission.engine, "previous_inputs", lambda: pytest.fail("model inputs were accessed"))
    with pytest.raises(ValueError, match="exact historical image, model, device and budgets"):
        admission.load_inputs()


@pytest.mark.parametrize("change", ["weight_count", "weight_hash", "source_count", "checkpoint"])
def test_changed_model_manifests_or_checkpoint_are_rejected(tmp_path, monkeypatch, change):
    cosmos = tmp_path / "cosmos"
    weights = {str(tmp_path / f"weight_{index:03d}"): f"{index:064x}" for index in range(163)}
    fingerprint = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = {**admission.historical.FIXED, "model_sha256": fingerprint}
    prepared = dict(
        model_config=dict(
            cosmos_source_root=str(cosmos),
            model_python=sys.executable,
            checkpoint=dict(name="8400_masked", iteration="iter_000008400"),
        ),
        weight_files={path: {"sha256": digest} for path, digest in weights.items()},
    )
    bridge = dict(
        config=dict(cosmos_source_root=str(cosmos)),
        source_sha256={str(cosmos / f"source_{index:03d}.py"): f"{index:064x}" for index in range(100)},
    )
    if change == "weight_count":
        prepared["weight_files"].pop(next(iter(weights)))
    elif change == "weight_hash":
        prepared["weight_files"][next(iter(weights))]["sha256"] = "f" * 64
    elif change == "source_count":
        bridge["source_sha256"].pop(next(iter(bridge["source_sha256"])))
    else:
        prepared["model_config"]["checkpoint"]["iteration"] = "other"
    monkeypatch.setattr(admission.historical, "FIXED", config)
    monkeypatch.setattr(admission.prefix, "reference", lambda _path: {"kind": "config"})
    monkeypatch.setattr(admission.prefix, "read", lambda ref: config if ref == {"kind": "config"} else {})
    monkeypatch.setattr(admission.prefix, "checked", lambda _ref: None)
    monkeypatch.setattr(admission.historical, "validate_capture", lambda _value: ({}, {"canvas_png": {}}))
    monkeypatch.setattr(admission.engine, "previous_inputs", lambda: (prepared, bridge, {}, {}, {}))
    with pytest.raises(ValueError):
        admission.load_inputs()
