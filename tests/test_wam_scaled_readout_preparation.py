"""027 cache-only ancestry, immutable failure, control and source boundaries."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def preparer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("prepare_wam_scaled_readout")


def test_cpu_environment_guard_runs_before_reading_any_ancestor(preparer, monkeypatch):
    monkeypatch.chdir(preparer.REPO)
    for cuda, ld in (("6", ""), ("0", ""), ("", "/tmp"), (None, "")):
        if cuda is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
        monkeypatch.setenv("LD_LIBRARY_PATH", ld)
        with pytest.raises(ValueError, match="explicitly empty CUDA/LD"):
            preparer.prepare()


def test_cache_only_mapping_refuses_episode_and_png_before_opening_them(preparer, tmp_path):
    episode, png = tmp_path / "unopened.npz", tmp_path / "unopened.png"
    evidence = preparer.Evidence()
    for path, forbidden in ((episode, {str(episode)}), (png, set())):
        with pytest.raises(ValueError, match="must not reopen EP/PNG"):
            preparer.cache_only_hashes({str(path): "not-read"}, forbidden, evidence)
    assert evidence.hashes == {} and not episode.exists() and not png.exists()
    cache = tmp_path / "actual-feature.npz"
    np.savez(cache, hidden=np.ones((2, 4)))
    expected = preparer.sha256(cache)
    preparer.cache_only_hashes({str(cache): expected}, set(), evidence)
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA differs"):
        evidence.recheck()


def ancestors():
    refs = dict(
        flow_prepared_ref=dict(path="/prepared.json", sha256="prepared"),
        flow_fit_ref=dict(path="/fit.json", sha256="fit"),
        flow_fit_verification_ref=dict(path="/verified.json", sha256="verified"),
    )
    source = dict(path="/features.json", sha256="features")
    flow = dict(
        schema_version="s2a.flow-readout-preparation.v1",
        source_sha256={str(i): "h" for i in range(72)},
        snapshot_sha256={str(i): "h" for i in range(72)},
        input_sha256={str(i): "h" for i in range(761)},
        source_features_ref=source,
    )
    prep_peer = dict(passed=True, phase="prepare", preparation_verified=True, prepared=refs["flow_prepared_ref"])
    fit = dict(
        complete=True,
        passed=True,
        prepared_ref=refs["flow_prepared_ref"],
        features_ref=source,
        producer_fit_groups=2,
        fit_attempts=2,
        fitted_rows=2880,
        validation_rows=2880,
        eligible_for_sampling=False,
        teacher_gate=dict(passed=False),
    )
    fit_peer = dict(
        passed=True,
        phase="fit",
        fit_verified=True,
        prepared=refs["flow_prepared_ref"],
        independent_model_reconstructions=2,
        eligible_for_sampling=False,
        teacher_gate=dict(passed=False),
        input_sha256={"/fit.json": "fit"},
    )
    final = dict(
        passed=True,
        prepared=refs["flow_prepared_ref"],
        fit=refs["flow_fit_ref"],
        fit_verification=refs["flow_fit_verification_ref"],
        eligible_for_sampling=False,
        old025_eligible_for_sampling=False,
        teacher_gate=dict(passed=False),
        source_before_after_unchanged=True,
        sampling_and_orchestration_directory_absent=True,
    )
    return flow, prep_peer, fit, fit_peer, final, refs


def test_original_failed_fit_and_independent_ref_identity_cannot_be_promoted(preparer):
    parts = ancestors()
    preparer.flow_admission(*parts)
    changed = copy.deepcopy(parts)
    changed[2]["eligible_for_sampling"] = True
    with pytest.raises(ValueError, match="failed teacher result"):
        preparer.flow_admission(*changed)
    changed = copy.deepcopy(parts)
    changed[3]["input_sha256"]["/fit.json"] = "unrelated-fit"
    with pytest.raises(ValueError, match="independent failed gate"):
        preparer.flow_admission(*changed)
    changed = copy.deepcopy(parts)
    changed[4]["teacher_gate"]["passed"] = True
    with pytest.raises(ValueError, match="formal final peer"):
        preparer.flow_admission(*changed)
    changed = copy.deepcopy(parts)
    changed[0]["input_sha256"].pop("0")
    with pytest.raises(ValueError, match="761-input"):
        preparer.flow_admission(*changed)


def control_files(module, tmp_path):
    flow = dict(
        anchors=[dict(name="train", episode=1, task_index=0), dict(name="validation", episode=2, task_index=0)],
        teacher_calls=[
            dict(
                anchor_name="train" if i < 72 else "validation",
                split="train" if i < 72 else "validation",
                sigma=(0.2, 0.5, 0.8)[i % 3],
            )
            for i in range(144)
        ],
    )
    rows = [
        dict(
            episode=1 if i < 72 else 2,
            anchor_name=c["anchor_name"],
            sigma=c["sigma"],
            split=c["split"],
            task_index=0,
            endpoint=j,
        )
        for i, c in enumerate(flow["teacher_calls"])
        for j in range(40)
    ]
    weights = np.zeros((1281, 10), np.float64)
    weights[-1] = 0.02
    np.savez(tmp_path / "bias.npz", weights=weights)
    predictions = dict(
        raw=np.zeros((5760, 128), np.float32),
        adapt=np.zeros((5760, 128), np.float32),
        bias=np.zeros((5760, 128), np.float32),
        training_weights=np.full(2880, 1 / 2880),
        sigma_actual=np.asarray([np.float32(r["sigma"]) for r in rows], np.float64),
    )
    np.savez(tmp_path / "predictions.npz", **predictions)
    fit = dict(
        models=dict(bias=module.reference(tmp_path / "bias.npz")),
        teacher_predictions=module.reference(tmp_path / "predictions.npz"),
        teacher_rows=rows,
    )
    return flow, fit, weights, predictions


def test_existing_control_keeps_full_predictions_and_real_sigma_without_refitting(preparer, tmp_path):
    flow, fit, weights, predictions = control_files(preparer, tmp_path)
    observed = preparer.verify_control(flow, fit, preparer.Evidence())
    assert observed["control_fit_count"] == 0 and observed["control_rows"] == 5760
    assert observed["control_predictions_recomputed_here"] is False
    assert observed["control_prediction_reproduction_required_during_fit"] is True
    weights[0, 0] = 1e-6
    np.savez(tmp_path / "bias.npz", weights=weights)
    fit["models"]["bias"] = preparer.reference(tmp_path / "bias.npz")
    with pytest.raises(ValueError, match="ten-parameter bias"):
        preparer.verify_control(flow, fit, preparer.Evidence())
    weights[0, 0] = 0
    np.savez(tmp_path / "bias.npz", weights=weights)
    fit["models"]["bias"] = preparer.reference(tmp_path / "bias.npz")
    predictions["sigma_actual"][0] = 0.2  # Incorrect nominal value, rather than exact FP32 expansion.
    np.savez(tmp_path / "predictions.npz", **predictions)
    fit["teacher_predictions"] = preparer.reference(tmp_path / "predictions.npz")
    with pytest.raises(ValueError, match="FP32-expanded sigma"):
        preparer.verify_control(flow, fit, preparer.Evidence())


def test_reused_control_row_split_must_match_original_cached_teacher_clock(preparer, tmp_path):
    flow, fit, _, _ = control_files(preparer, tmp_path)
    fit["teacher_rows"][2880]["split"] = "train"
    with pytest.raises(ValueError, match="row/split/target"):
        preparer.verify_control(flow, fit, preparer.Evidence())


def test_source_closure_exactly_extends72_without_rewriting_any_old_source(preparer, monkeypatch, tmp_path):
    monkeypatch.setattr(preparer, "REPO", tmp_path)
    monkeypatch.setattr(preparer, "CONFIG", tmp_path / "new-config.json")
    monkeypatch.setattr(preparer, "PLAN", tmp_path / "new-plan.md")
    inherited = {str(tmp_path / f"old{i}.py"): f"frozen{i}" for i in range(72)}
    sources = [tmp_path / p for p in preparer.NEW_SOURCES] + [preparer.CONFIG, preparer.PLAN]
    for path in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen source\n")
    closure = preparer.source_closure(dict(source_sha256=inherited), preparer.Evidence())
    assert len(closure) == 84 and all(closure[p] == h for p, h in inherited.items())
    sources[0].unlink()
    with pytest.raises(FileNotFoundError):
        preparer.source_closure(dict(source_sha256=inherited), preparer.Evidence())
    assert not (tmp_path / "preparation").exists()


def test_prior_cpu_exit_and_reap_are_bound_to_the_owned_birth(preparer, monkeypatch, tmp_path):
    entries = []
    for i in range(2):
        values = dict(
            launch=dict(child=dict(pid=10 + i, start_ticks=20 + i)),
            exit=dict(actual_wait=True, automatic_retry=False, returncode=0),
            release=dict(
                pid=10 + i, start_ticks=20 + i, reaped=True, returncode=0, signal_sent=False, original_error=None
            ),
        )
        refs = dict(role=f"cpu{i}")
        for key, value in values.items():
            path = tmp_path / f"{i}_{key}.json"
            path.write_text(json.dumps(value))
            refs[key] = preparer.reference(path)
        entries.append(refs)
    monkeypatch.setattr(preparer, "terminal", lambda item: dict(original_birth_terminal=True, expected=item))
    assert len(preparer.owned_cpu_exits(dict(owned_cpu_children=entries), preparer.Evidence())) == 2
    changed = tmp_path / "1_release.json"
    value = json.loads(changed.read_text())
    value["start_ticks"] += 1
    changed.write_text(json.dumps(value))
    entries[1]["release"] = preparer.reference(changed)
    with pytest.raises(ValueError, match="normally reaped"):
        preparer.owned_cpu_exits(dict(owned_cpu_children=entries), preparer.Evidence())
