"""028 admission keeps teacher evidence, prior budget and frozen model exact."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def preparer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prepare_wam_scaled_state")


def ancestors():
    refs = {
        f"scaled_{k}_ref": dict(path=f"/{k}.json", sha256=k)
        for k in (
            "prepared",
            "fit",
            "fit_verification",
            "final_peer",
            "owner",
        )
    }
    features = dict(path="/features.json", sha256="features")
    gate = dict(
        passed=True, checks=dict(pooled_improvement=True, beats_constant_bias=True, all_sigmas_nonregressing=True)
    )
    scaled = dict(
        schema_version="s2a.scaled-readout-preparation.v1",
        source_features_ref=features,
        source_sha256={str(i): "h" for i in range(84)},
        snapshot_sha256={str(i): "h" for i in range(84)},
        input_sha256={str(i): "h" for i in range(874)},
    )
    fit = dict(
        schema_version="s2a.scaled-readout-fit.v1",
        complete=True,
        passed=True,
        prepared_ref=refs["scaled_prepared_ref"],
        features_ref=features,
        candidate_fits=1,
        fit_attempts=1,
        producer_fit_groups=1,
        reused_controls=1,
        fitted_rows=2880,
        validation_rows=2880,
        eligible_for_runtime_qualification=True,
        eligible_for_sampling=False,
        teacher_gate=gate,
    )
    checked = dict(
        passed=True,
        phase="fit",
        fit_verified=True,
        prepared=refs["scaled_prepared_ref"],
        independent_candidate_reconstructions=1,
        reused_controls_verified=1,
        teacher_gate=gate,
        eligible_for_runtime_qualification=True,
        eligible_for_sampling=False,
        input_sha256={"/fit.json": "fit"},
    )
    commands = [dict(launch=f"launch{i}", exit=f"exit{i}", release=f"release{i}") for i in range(2)]
    final = dict(
        passed=True,
        prepared=refs["scaled_prepared_ref"],
        fit=refs["scaled_fit_ref"],
        fit_verification=refs["scaled_fit_verification_ref"],
        fit_owner=refs["scaled_owner_ref"],
        source_before_after_unchanged=True,
        frozen_sources=84,
        source_snapshots=84,
        weight_stat_count=163,
        teacher_gate=gate,
        eligible_for_runtime_qualification=True,
        eligible_for_sampling=False,
        reused026_full128_raw_and_bias_prediction_bits_exact=True,
        reused026_training_weights_and_actual_sigmas_exact=True,
        owned_cpu_children=copy.deepcopy(commands),
        actual_budget=dict(
            candidate_fits=1,
            independent_candidate_reconstructions=1,
            **{
                k: 0
                for k in (
                    "new_control_fits",
                    "gpu_jobs",
                    "new_teacher_calls",
                    "wam_denoise_calls",
                    "complete_generations",
                    "state_label_scoring",
                    "physics_steps",
                    "fresh_cases",
                )
            },
        ),
    )
    owner = dict(
        complete=True,
        passed=True,
        prepared_ref=refs["scaled_prepared_ref"],
        producer_result=refs["scaled_fit_ref"],
        eligible_for_runtime_qualification=True,
        eligible_for_sampling=False,
        automatic_retry=False,
        owned_commands=commands,
    )
    return scaled, fit, checked, final, owner, refs


def test_cpu_environment_rejected_before_ancestry_or_model_read(preparer, monkeypatch):
    monkeypatch.chdir(preparer.REPO)
    for cuda, ld in (("6", ""), ("0", ""), ("", "/tmp"), (None, "")):
        if cuda is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
        monkeypatch.setenv("LD_LIBRARY_PATH", ld)
        with pytest.raises(ValueError, match="explicitly empty CUDA/LD"):
            preparer.prepare()


def test_all_teacher_checks_required_and_original_sampling_false_preserved(preparer):
    parts = ancestors()
    preparer.scaled_admission(*parts)
    changed = copy.deepcopy(parts)
    changed[1]["teacher_gate"]["checks"]["all_sigmas_nonregressing"] = False
    with pytest.raises(ValueError, match="every original027 teacher gate"):
        preparer.scaled_admission(*changed)
    changed = copy.deepcopy(parts)
    changed[1]["eligible_for_sampling"] = True
    with pytest.raises(ValueError, match="original sampling False"):
        preparer.scaled_admission(*changed)
    changed = copy.deepcopy(parts)
    changed[2]["input_sha256"]["/fit.json"] = "different fit"
    with pytest.raises(ValueError, match="single reconstruction/teacher identity"):
        preparer.scaled_admission(*changed)


def test_no_repeated_fit_or_unrecorded_prior_sampling_budget_admitted(preparer):
    parts = ancestors()
    parts[1]["candidate_fits"] = 2
    with pytest.raises(ValueError, match="single-fit teacher"):
        preparer.scaled_admission(*parts)
    parts = ancestors()
    parts[3]["actual_budget"]["wam_denoise_calls"] = 35
    with pytest.raises(ValueError, match="CPU-only budget"):
        preparer.scaled_admission(*parts)
    parts = ancestors()
    parts[4]["owned_commands"][1]["exit"] = "unrelated exit"
    with pytest.raises(ValueError, match="owned exit refs"):
        preparer.scaled_admission(*parts)


def model_sources(module, tmp_path):
    model = tmp_path / "adapt.npz"
    arrays = dict(
        weights=np.zeros((1281, 10), np.float64), center=np.zeros(1280, np.float64), scale=np.asarray(6.0, np.float64)
    )
    np.savez(model, **arrays)
    prediction = tmp_path / "predictions.npz"
    np.savez(prediction, adapt=np.zeros((2, 128), np.float32))
    model_ref, predictions_ref = module.reference(model), module.reference(prediction)
    fit = dict(
        models=dict(adapt=model_ref),
        teacher_predictions=predictions_ref,
        learned_numbers=14091,
        ridge=0.1,
        deployment_dtype="torch.bfloat16",
        reused_control_quantized_predictions_exact=True,
    )
    checked = dict(
        input_sha256={r["path"]: r["sha256"] for r in (model_ref, predictions_ref)},
        max_parameter_abs_difference=1e-15,
        max_preprocessing_abs_difference=dict(center=0.0, scale=0.0),
    )
    final = dict(reused026_full128_raw_and_bias_prediction_bits_exact=True)
    return arrays, fit, checked, final


def test_frozen_model_shape_precision_and_audit_bound_without_any_reconstruction(preparer, tmp_path):
    arrays, fit, checked, final = model_sources(preparer, tmp_path)
    result = preparer.model_admission(fit, checked, final, preparer.Evidence())
    assert result["learned_numbers"] == 14091 and result["new_parameter_reconstructions"] == 0
    assert result["adapted_model_ref"] == fit["models"]["adapt"]
    arrays["center"] = arrays["center"].astype(np.float32)
    np.savez(tmp_path / "adapt.npz", **arrays)
    fit["models"]["adapt"] = preparer.reference(tmp_path / "adapt.npz")
    checked["input_sha256"][str(tmp_path / "adapt.npz")] = fit["models"]["adapt"]["sha256"]
    with pytest.raises(ValueError, match="FP64/finite"):
        preparer.model_admission(fit, checked, final, preparer.Evidence())


def test_nonpositive_scale_and_changed_actual_prediction_reference_rejected(preparer, tmp_path):
    arrays, fit, checked, final = model_sources(preparer, tmp_path)
    checked["input_sha256"][fit["teacher_predictions"]["path"]] = "unrelated predictions"
    with pytest.raises(ValueError, match="actual model/predictions"):
        preparer.model_admission(fit, checked, final, preparer.Evidence())
    checked["input_sha256"][fit["teacher_predictions"]["path"]] = fit["teacher_predictions"]["sha256"]
    arrays["scale"] = np.asarray(0.0, np.float64)
    np.savez(tmp_path / "adapt.npz", **arrays)
    fit["models"]["adapt"] = preparer.reference(tmp_path / "adapt.npz")
    checked["input_sha256"][str(tmp_path / "adapt.npz")] = fit["models"]["adapt"]["sha256"]
    with pytest.raises(ValueError, match="global scalar scale"):
        preparer.model_admission(fit, checked, final, preparer.Evidence())


def test_inherited_observation_references_cannot_enter_actual_input_hash_reads(preparer, tmp_path):
    ep, png = tmp_path / "never-opened.npz", tmp_path / "never-opened.png"
    evidence = preparer.Evidence()
    for path in (ep, png):
        with pytest.raises(ValueError, match="must not reopen EP/PNG"):
            preparer.cache_only_hashes({str(path): "not-read"}, {str(ep)}, evidence)
    assert not evidence.hashes and not ep.exists() and not png.exists()


def test_new_source_closure_adds_exactly12_and_refuses_missing_entry(preparer, monkeypatch, tmp_path):
    monkeypatch.setattr(preparer, "REPO", tmp_path)
    monkeypatch.setattr(preparer, "CONFIG", tmp_path / "new-config.json")
    monkeypatch.setattr(preparer, "PLAN", tmp_path / "new-plan.md")
    inherited = {str(tmp_path / f"old{i}.py"): f"frozen{i}" for i in range(84)}
    files = [tmp_path / p for p in preparer.NEW_SOURCES] + [preparer.CONFIG, preparer.PLAN]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source\n")
    source = preparer.source_closure(dict(source_sha256=inherited), preparer.Evidence())
    assert len(source) == 96 and all(source[p] == digest for p, digest in inherited.items())
    # The numbered manifest remains bound when write_once sorts JSON keys.
    snapshots = {f"{i:03d}_{Path(p).name}": h for i, (p, h) in enumerate(source.items())}
    roundtrip = json.loads(json.dumps(dict(source_sha256=source, snapshot_sha256=snapshots), sort_keys=True))
    assert roundtrip["snapshot_sha256"] == snapshots
    files[0].unlink()
    with pytest.raises(FileNotFoundError):
        preparer.source_closure(dict(source_sha256=inherited), preparer.Evidence())
    assert not (tmp_path / "preparation").exists()
