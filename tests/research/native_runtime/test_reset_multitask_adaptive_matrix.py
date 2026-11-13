"""Exercise four-GPU scheduling and failure accounting without running a model."""

import json
from types import SimpleNamespace

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_multitask_adaptive_600 as matrix


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return prefix.reference(path)


def setup_matrix(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    output.mkdir()
    monkeypatch.setattr(matrix.admission, "OUTPUT", output)
    config = prefix.read(matrix.admission.config_ref())
    sources = [{"source": "fixed"}]
    monkeypatch.setattr(matrix.admission, "source_refs", lambda: sources)

    def trial(index):
        spec = next(row for row in config["cases"] if row["case_index"] == index)
        return SimpleNamespace(output=output / f"case_{index:02d}", gpu_uuid=spec["gpu_uuid"])

    monkeypatch.setattr(matrix.admission, "make_trial", trial)
    monkeypatch.setattr(matrix.registered, "gpu_observation", lambda _uuid: {"memory_mib": 0, "compute_apps": []})
    write(
        output / "preflight.json",
        dict(
            complete=True,
            passed=True,
            config_ref=matrix.admission.config_ref(),
            case_indices=config["case_indices"],
            source_refs=sources,
        ),
    )
    return output, config, trial


def fake_workers(trial, calls, *, phase, failed=None, interrupted=False):
    def run(specs, *, cwd, timeout_seconds):
        assert cwd == prefix.REPO and timeout_seconds == 11400
        indices = [int(spec["key"]) for spec in specs]
        calls.append(indices)
        rows = []
        for index, spec in zip(indices, specs, strict=True):
            assert spec["argv"][-4:] == ["--case-index", str(index), "--phase", phase]
            spec["log"].write_text("synthetic case owner", encoding="utf-8")
            failed_case = index == failed or interrupted
            write(
                trial(index).output / phase / "result.json",
                dict(complete=not failed_case, passed=not failed_case, gpu_released=True, counts={"control_steps": 8}),
            )
            rows.append(
                dict(
                    key=str(index),
                    returncode=1 if failed_case else 0,
                    actual_wait=True,
                    reaped=True,
                    signal_sent=interrupted,
                    log=str(spec["log"]),
                )
            )
        return dict(workers=rows, launch_error="interrupted" if interrupted else None)

    return run


def test_six_cases_use_two_disjoint_waves_and_keep_failed_denominator(tmp_path, monkeypatch):
    output, _config, trial = setup_matrix(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(matrix, "run_owned_workers", fake_workers(trial, calls, phase="producer", failed=2))
    result = matrix.run_matrix("producer")
    assert calls == [[0, 2, 4, 6], [8, 10]]
    assert result["complete"] and not result["passed"]
    assert [row["case_index"] for row in result["cases"]] == [0, 2, 4, 6, 8, 10]
    assert [row["case_index"] for row in result["cases"] if not row["case_passed"]] == [2]
    assert len(result["waves"]) == 2 and (output / "orchestration/producer/result.json").is_file()


def test_interrupted_first_wave_never_starts_remaining_cases(tmp_path, monkeypatch):
    _output, _config, trial = setup_matrix(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(matrix, "run_owned_workers", fake_workers(trial, calls, phase="producer", interrupted=True))
    result = matrix.run_matrix("producer")
    assert calls == [[0, 2, 4, 6]] and not result["complete"] and not result["passed"]
    assert [row["case_index"] for row in result["cases"] if row.get("skip_reason")] == [8, 10]
    assert all(row["skip_reason"] == "earlier_wave_interrupted" for row in result["cases"][-2:])


def test_checker_skips_failed_or_absent_producers_but_replays_other_cases(tmp_path, monkeypatch):
    _output, config, trial = setup_matrix(tmp_path, monkeypatch)
    for index in config["case_indices"]:
        if index != 2:
            write(
                trial(index).output / "producer/result.json",
                dict(complete=index != 4, passed=index != 4, gpu_released=True),
            )
    calls = []
    monkeypatch.setattr(matrix, "run_owned_workers", fake_workers(trial, calls, phase="checker"))
    result = matrix.run_matrix("checker")
    assert calls == [[0, 6], [8, 10]] and not result["complete"] and not result["passed"]
    skips = {row["case_index"]: row["skip_reason"] for row in result["cases"] if "skip_reason" in row}
    assert skips == {2: "missing_producer_result", 4: "producer_not_closed"}


@pytest.mark.parametrize("change", ["case_indices", "source_refs"])
def test_changed_group_preflight_is_rejected_before_any_case(tmp_path, monkeypatch, change):
    output, _config, _trial = setup_matrix(tmp_path, monkeypatch)
    path = output / "preflight.json"
    value = json.loads(path.read_text())
    value[change] = []
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="source-matched six-case preflight"):
        matrix.run_matrix("producer")
    assert not (output / "orchestration").exists()


def setup_preflight(tmp_path, monkeypatch, *, busy=False, changed_model=False):
    output = tmp_path / "matrix"
    monkeypatch.setattr(matrix.admission, "OUTPUT", output)
    config = prefix.read(matrix.admission.config_ref())
    sources = [{"source": "fixed"}]
    before = {"src/fixed.py": "a" * 64}
    monkeypatch.setattr(matrix.admission, "source_refs", lambda: sources)
    monkeypatch.setattr(matrix, "snapshot", lambda _root: {"sources": before})
    monkeypatch.setattr(
        matrix.registered,
        "gpu_observation",
        lambda _uuid: {"memory_mib": 1 if busy else 0, "compute_apps": []},
    )
    qa, dry = tmp_path / "qa.json", tmp_path / "dry.json"
    write(qa, {"qa": True})
    write(dry, {"dry": True})
    monkeypatch.setattr(matrix.admission, "QA", qa)

    def trial(index):
        spec = next(row for row in config["cases"] if row["case_index"] == index)
        model_sha = "other" if changed_model and index == 10 else "same"
        data = dict(
            index=index,
            model_source_hashes={"model.py": model_sha},
            weight_hashes={"weight": "digest"},
            prepared={"model_config": {"model_python": "/synthetic/python"}},
        )
        return SimpleNamespace(
            output=output / f"case_{index:02d}",
            qa=qa,
            dry=dry,
            gpu_uuid=spec["gpu_uuid"],
            load_inputs=lambda: data,
            require_config=lambda _inputs: None,
            check_cpu_admission=lambda _inputs: before,
            input_receipt=lambda item: {"case": item["index"]},
            model_bundle=lambda item: {"case": item["index"]},
        )

    monkeypatch.setattr(matrix.admission, "make_trial", trial)
    calls = []

    def manifests(_inputs):
        calls.append("weights")
        return dict(source_files_verified=1004, weight_files_verified=163, weight_stats={"fixed": "stats"})

    def probe(_python, _directory):
        calls.append("cv2")
        return dict(actual_wait=True, reaped=True, returncode=0)

    monkeypatch.setattr(matrix.model_owner, "verify_model_manifests", manifests)
    monkeypatch.setattr(matrix.live_cv2_admission, "run_cpu_probe", probe)
    monkeypatch.setattr(matrix.repro, "sha256", lambda _path: "b" * 64)
    return output, calls


def test_preflight_hashes_shared_model_once_and_binds_all_six_cases(tmp_path, monkeypatch):
    output, calls = setup_preflight(tmp_path, monkeypatch)
    ref = matrix.preflight()
    value = prefix.read(ref)
    assert calls == ["weights", "cv2"]
    assert value["complete"] and value["passed"] and len(value["case_preflight_refs"]) == 6
    probe_refs = []
    for index, case_ref in zip(value["case_indices"], value["case_preflight_refs"], strict=True):
        case = prefix.read(case_ref)
        assert case["inputs"] == {"case": index} and case["weight_files_verified"] == 163
        assert case["gpu_before"] == {"memory_mib": 0, "compute_apps": []}
        assert prefix.read(case["cv2_probe_ref"])["reaped"]
        probe_refs.append(case["cv2_probe_ref"])
        assert (output / f"case_{index:02d}/preflight.json").is_file()
    assert all(ref == probe_refs[0] for ref in probe_refs)


@pytest.mark.parametrize("change", ["busy", "model"])
def test_preflight_rejects_busy_gpu_or_mixed_model_before_output(tmp_path, monkeypatch, change):
    output, calls = setup_preflight(tmp_path, monkeypatch, busy=change == "busy", changed_model=change == "model")
    with pytest.raises(ValueError):
        matrix.preflight()
    assert not output.exists() and not calls
