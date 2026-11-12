"""Verify long-trial binding against real prior receipts and model Python."""

import copy
import json
import os
import subprocess

from scripts.simulation.runtime import adaptive_live_bowl_long_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_long_owner as owner
from scripts.simulation.runtime import registered_model_bundle as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prefix.reference(path)


def test_previous_comparison_reports_matching_and_changed_physical_history():
    inputs = admission.load_inputs()
    branch = prefix.read(inputs["prior_recovery_producer"]["branch_ref"])
    result = owner.compare_previous(inputs, branch)
    assert result["all_previous_periods_compared"] and result["compared_periods"] == 115
    assert result["periods_exact"] and result["capture_schedule_matches"] and result["common_captures_match"]
    assert result["prior_capture_steps"] == result["new_capture_steps"] == [107, 111]
    changed = copy.deepcopy(branch)
    _prior, adaptive = owner.historical_audit.periods(changed, "producer")
    adaptive[0]["task_success"] = not adaptive[0]["task_success"]
    result = owner.compare_previous(inputs, changed)
    assert result["periods_exact"] is False and result["periods_within_tolerance"] is False

    shorter = copy.deepcopy(branch)
    shorter["runtime"]["continuation"]["adaptive"]["segments"] = shorter["runtime"]["continuation"]["adaptive"][
        "segments"
    ][:1]
    shorter["runtime"]["adaptive_captures"] = shorter["runtime"]["adaptive_captures"][:1]
    result = owner.compare_previous(inputs, shorter)
    assert result["compared_periods"] == 111 and result["all_previous_periods_compared"] is False
    assert result["periods_exact"] and result["capture_schedule_matches"] is False


def test_real_model_python_accepts_the_same_long_bundle_without_gpu(tmp_path):
    inputs = admission.load_inputs()
    trial = owner.make_trial()
    refs = trial.source_refs()
    bundle = trial.model_bundle(inputs)
    output = tmp_path / "trial"
    simulation = output / "producer/simulation"
    simulation.mkdir(parents=True)
    bundle["model_registration"]["image_root"] = str(simulation.relative_to(admission.ROOT))
    qa, dry = tmp_path / "qa.json", tmp_path / "dry.json"
    write(qa, {"fixture": "qa"})
    write(dry, {"fixture": "dry"})
    bundle_ref = write(output / "producer/model/registered_inputs.json", bundle)
    write(
        output / "producer/registration.json",
        dict(phase="producer", inputs=bundle["input_receipt"], source_refs=refs, model_bundle_ref=bundle_ref),
    )
    write(
        output / "preflight.json",
        dict(
            complete=True,
            passed=True,
            inputs=bundle["input_receipt"],
            source_refs=refs,
            qa_ref=prefix.reference(qa),
            dry_ref=prefix.reference(dry),
            model_bundle_sha256=registered._digest(bundle),
            gpu_uuid=trial.gpu_uuid,
            source_files_verified=len(inputs["model_source_hashes"]),
            weight_files_verified=len(inputs["weight_hashes"]),
            weight_stats=owner.registered.model_owner.file_stats(inputs["weight_hashes"]),
            model_python_sha256=owner.registered.repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
        ),
    )
    script = """
import sys
from dataclasses import replace
from pathlib import Path
from scripts.simulation.runtime import adaptive_live_bowl_long_owner as owner
from scripts.wam.tracking.preview.adaptive_model_worker import validate_registration

output, qa, dry = map(Path, sys.argv[1:])
trial = replace(owner.make_trial(), output=output, qa=qa, dry=dry)
bundle, _ = trial.model_worker_inputs()
validate_registration(bundle['model_registration'], output)
assert Path(sys.executable).resolve() == Path(bundle['prepared']['model_config']['model_python']).resolve()
assert not any(name in sys.modules for name in ('torch', 'mujoco', 'robosuite', 'libero', 'cosmos_framework'))
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = "src:."
    environment.update({key: "1" for key in prefix.THREAD_VARS})
    subprocess.run(
        [inputs["prepared"]["model_config"]["model_python"], "-c", script, str(output), str(qa), str(dry)],
        cwd=admission.ROOT,
        env=environment,
        check=True,
        timeout=900,
    )
