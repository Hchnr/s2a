"""Bind registered seed-43 preflight to the runtime-created image directory."""

from scripts.simulation.runtime import registered_model_bundle, reset_seed43_trial
from scripts.simulation.runtime import reset_seed43_live_admission as admission
from scripts.wam.tracking.preview import reset_seed43_live_request
from tests.research.native_runtime.test_registered_adaptive_trial import setup_trial


def test_registered_bundle_uses_producer_runtime_not_parent_directory(tmp_path, monkeypatch):
    existing, inputs, _source, _refs, _qa, _dry = setup_trial(tmp_path, monkeypatch)
    trial = reset_seed43_trial.RegisteredResetTrial(**vars(existing))
    bundle = trial.model_bundle(inputs)
    assert bundle["model_registration"]["image_root"] == str(trial.output / "producer/simulation/runtime")
    assert bundle["input_receipt"] == trial.input_receipt(inputs)
    assert registered_model_bundle._digest(bundle) != registered_model_bundle._digest(existing.model_bundle(inputs))


def test_real_registered_bundle_is_accepted_without_physics_or_model():
    trial = reset_seed43_trial.make_trial()
    inputs = trial.load_inputs(verify_assets=False)
    trial.require_config(inputs)
    bundle = trial.model_bundle(inputs)
    registration = bundle["model_registration"]
    accepted = reset_seed43_live_request.validate_registration(registration, trial.output)
    assert accepted.seed == 43 and accepted.phase == "ready"
    assert registration["image_root"] == str(
        (admission.OUTPUT / "producer/simulation/runtime").relative_to(admission.ROOT)
    )
    assert len(bundle["weight_hashes"]) == 163
    assert bundle["input_receipt"]["initial"] == inputs["initial"]


def test_source_manifest_includes_corrected_checker_and_worker():
    sources = reset_seed43_trial.source_refs()
    names = {item["path"] for item in sources}
    assert str(admission.ROOT / "scripts/simulation/runtime/reset_live_two_window_replay_v2.py") in names
    assert str(admission.ROOT / "scripts/wam/tracking/preview/reset_seed43_model_worker.py") in names
    assert all(str(admission.ROOT / path) in names for path in reset_seed43_trial.QA_TARGETS)
