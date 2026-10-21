"""028 synthetic CPU evidence, precision and future-label admission boundaries."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_scaled_state.py"
    spec = importlib.util.spec_from_file_location("independent_scaled_state_checker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_fixture():
    rng = np.random.default_rng(7)
    return dict(
        weights=rng.normal(scale=0.005, size=(1281, 10)),
        center=rng.normal(scale=0.05, size=1280),
        scale=np.asarray(6.4),
    )


def precision_fixture(checker):
    model = model_fixture()
    sigmas = np.asarray([0.2, 0.5, 0.8, 0.12809006869792938, 0.9990000128746033], np.float32).astype(np.float64)
    cases, arrays = [], {}
    for case in checker.precision_cases(model, sigmas):
        name = case["name"]
        for key, value in case.items():
            if key != "name":
                arrays[f"{name}_{key}"] = value.copy()
        output = checker.SCALED.quantized_velocity(
            case["hidden"], case["base"], case["sigma_actual"], case["weights"], case["center"], case["scale"]
        )
        bits = (output.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
        arrays[f"{name}_cpu_bits"], arrays[f"{name}_cuda_bits"] = bits.copy(), bits.copy()
        cases.append(
            dict(
                name=name,
                cpu_cuda_bitwise_equal=True,
                actual_hook_matches_apply=True,
                zero_original_object=True,
                other_domain_original_object=True,
                other118_preserved=True,
                input_tensors_unchanged=True,
            )
        )
    primitive = np.asarray(
        [
            0.0,
            -0.0,
            2**-134 - 2**-160,
            2**-134,
            2**-134 + 2**-160,
            -(2**-134) + 2**-160,
            -(2**-134),
            -(2**-134) - 2**-160,
            1 + 2**-8 + 2**-30,
            -1 - 2**-8 - 2**-30,
        ]
    )
    arrays.update(
        primitive_values=primitive,
        primitive_cpu_bits=checker.FLOW.bf16_bits(primitive),
        primitive_cuda_bits=checker.FLOW.bf16_bits(primitive),
    )
    return arrays, cases, model, sigmas


def test_precision_covers_twenty_cases_actual_parameters_and_all128_output_bits(checker):
    arrays, cases, model, sigmas = precision_fixture(checker)
    checker.precision_arrays(arrays, cases, model, sigmas)
    assert [case["name"] for case in cases[-5:]] == [f"sigma{i}_candidate" for i in range(5)]
    for key, position in (
        ("sigma0_positive_cpu_bits", (0, 0)),
        ("sigma4_candidate_cuda_bits", (1, 127)),
        ("primitive_cuda_bits", 1),
    ):
        original = arrays[key][position].copy()
        arrays[key][position] ^= np.int16(1)
        with pytest.raises(ValueError, match="BF16 bits|primitive RNE"):
            checker.precision_arrays(arrays, cases, model, sigmas)
        arrays[key][position] = original
    arrays["sigma0_candidate_weights"][0, 0] += 1e-12
    with pytest.raises(ValueError, match="candidate weights input bits"):
        checker.precision_arrays(arrays, cases, model, sigmas)


def test_candidate_fixture_has_reproducible_synthetic_hidden_and_no_source_episode_inputs(checker):
    model, sigmas = (
        model_fixture(),
        np.asarray([0.2, 0.5, 0.8, 0.12809006869792938, 0.9990000128746033], np.float32).astype(np.float64),
    )
    cases = list(checker.precision_cases(model, sigmas))
    rng = np.random.default_rng(42)
    for index, case in enumerate(cases[15:]):
        np.testing.assert_array_equal(case["hidden"], rng.integers(-32, 33, (2, 1280)).astype(np.float32) / 128)
        np.testing.assert_array_equal(case["base"], rng.integers(-16, 17, (2, 128)).astype(np.float32) / 16)
        np.testing.assert_array_equal(case["sigma_actual"], np.full((2, 1), sigmas[index]))
        np.testing.assert_array_equal(case["weights"], model["weights"])
        np.testing.assert_array_equal(case["center"], model["center"])
        assert case["scale"] == model["scale"]
    assert cases[0]["sigma_actual"][0, 0] != 0.2
    assert all(case["scale"] == 3 for case in cases[:15])


def test_precision_rejects_changed_center_scale_nominal_sigma_or_omitted_actual_candidate(checker):
    arrays, cases, model, sigmas = precision_fixture(checker)
    for key, value in (
        ("sigma0_positive_center", np.ones(1280)),
        ("sigma0_positive_scale", np.asarray(1.0)),
        ("sigma0_candidate_sigma_actual", np.full((2, 1), 0.2)),
    ):
        original = arrays[key]
        arrays[key] = value
        with pytest.raises(ValueError, match="input bits"):
            checker.precision_arrays(arrays, cases, model, sigmas)
        arrays[key] = original
    with pytest.raises(ValueError, match="fifteen synthetic plus five"):
        checker.precision_arrays(arrays, cases[:-1], model, sigmas)


def test_source96_and_snapshots_survive_sorted_json_and_require_actual_new_entrypoints(checker, tmp_path):
    old = {str(checker.HELPER): checker.HELPER_SHA, **{str(tmp_path / f"old{i}.py"): f"{i:064x}" for i in range(83)}}
    old = json.loads(json.dumps(old, sort_keys=True))
    sources = dict(old)
    sources.update({str(checker.REPO / p): "a" * 64 for p in checker.NEW_SOURCES})
    sources.update({str(checker.CONFIG): checker.CONFIG_SHA, str(checker.PLAN): checker.PLAN_SHA})
    path = tmp_path / "preparation/prepared.json"
    snapshots = {
        str(path.parent / "source_snapshot" / f"{i:03d}_{Path(p).name}"): sha
        for i, (p, sha) in enumerate(sources.items())
    }
    prepared = json.loads(json.dumps(dict(source_sha256=sources, snapshot_sha256=snapshots), sort_keys=True))
    scaled = dict(source_sha256=old)
    checker.source_closure(prepared, scaled, path)
    for entry in (
        "scripts/run_wam_scaled_state.py",
        "scripts/probe_wam_scaled_readout_quantization.py",
        "scripts/sample_wam_scaled_readout.py",
        "scripts/verify_wam_scaled_state.py",
        "scripts/prepare_wam_scaled_state.py",
    ):
        changed = dict(sources)
        changed.pop(str(checker.REPO / entry))
        with pytest.raises(ValueError, match="required96 source"):
            checker.source_closure(dict(prepared, source_sha256=changed), scaled, path)
    changed = dict(snapshots)
    changed[next(iter(changed))] = "f" * 64
    with pytest.raises(ValueError, match="snapshot paths/digests"):
        checker.source_closure(dict(prepared, snapshot_sha256=changed), scaled, path)


def test_raw_episode_zero_sources_and_every_png_suffix_are_forbidden_before_hashing(checker):
    forbidden = checker.SCALED.forbidden_observations(
        dict(
            anchors=[dict(arrays_path="/val.npz", image_path="/current.png")],
            zero_identity_cases=[dict(sample=dict(arrays_path="/zero.npz", image_path="/zero.png"))],
        )
    )

    class NoRead:
        def check(self, *args):
            raise AssertionError("raw EP/PNG must not be opened")

    for path in ("/val.npz", "/zero.npz", "/zero.png", "/unregistered.PNG"):
        with pytest.raises(ValueError, match="reopened inherited"):
            checker.cache_only_hashes({path: "a" * 64}, forbidden, NoRead())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def lifecycle_fixture(checker, tmp_path):
    prepared = dict(output_root=str(tmp_path))
    directory = tmp_path / "orchestration/precision"
    launch = directory / "model/launch.json"
    exit_path, release_path = directory / "model/exit.json", directory / "model/ownership/release.json"
    write_json(
        launch,
        dict(
            selected_environment=dict(CUDA_VISIBLE_DEVICES="6", LD_LIBRARY_PATH=""),
            child=dict(pid=123, start_ticks=456),
        ),
    )
    write_json(exit_path, dict(actual_wait=True, automatic_retry=False, returncode=0))
    write_json(
        release_path,
        dict(
            pid=123,
            start_ticks=456,
            reaped=True,
            returncode=0,
            signal_sent=False,
            original_error=None,
            only_owned_child_signalled=True,
        ),
    )
    observed = dict(
        physical_gpu=6, uuid=checker.GPU_UUID, memory_mib=0, compute_apps=[], seconds_since_actual_owned_wait=0.02
    )
    gpu_path, polls_path = directory / "gpu_after.json", directory / "resource_observations.jsonl"
    write_json(gpu_path, dict(observed, released=True))
    polls_path.write_text(json.dumps(observed) + "\n")
    raw = dict(complete=True, passed=True, candidate_model_ref=dict(path="frozen.npz", sha256="a" * 64))
    write_json(tmp_path / "precision/worker_result.json", raw)
    actual = dict(
        raw,
        exit_ref=checker.reference(exit_path),
        release_ref=checker.reference(release_path),
        gpu_after_ref=checker.reference(gpu_path),
        resource_observations_ref=checker.reference(polls_path),
    )
    return prepared, actual, directory


def test_actual_worker_exit_release_and_resource_evidence_do_not_require_unfinished_owner(checker, tmp_path):
    prepared, actual, directory = lifecycle_fixture(checker, tmp_path)
    assert not (directory / "result.json").exists()
    checker.public_lifecycle(actual, prepared, "precision", checker.OLD.Evidence())
    changed = dict(actual, passed=False)
    with pytest.raises(ValueError, match="changed actual worker"):
        checker.public_lifecycle(changed, prepared, "precision", checker.OLD.Evidence())
    gpu = json.loads(Path(actual["gpu_after_ref"]["path"]).read_text())
    gpu["memory_mib"] = 1
    write_json(Path(actual["gpu_after_ref"]["path"]), gpu)
    changed = dict(actual, gpu_after_ref=checker.reference(actual["gpu_after_ref"]["path"]))
    with pytest.raises(ValueError, match="GPU6 release"):
        checker.public_lifecycle(changed, prepared, "precision", checker.OLD.Evidence())


def sampling_fixture():
    candidate = dict(path="frozen027.npz", sha256="a" * 64)
    fit, fit_verified = dict(path="027fit.json", sha256="b" * 64), dict(path="027verified.json", sha256="c" * 64)
    prepared = dict(
        adapted_model_ref=candidate,
        scaled_fit_ref=fit,
        scaled_fit_verification_ref=fit_verified,
        checkpoint="8400/model",
    )
    labels = [f"zero:{i}" for i in range(3)] + [f"sampling:{i}" for i in range(96)]
    actual = dict(
        schema_version="s2a.scaled-state-sampling.v1",
        complete=True,
        passed=True,
        fit_ref=fit,
        fit_verification_ref=fit_verified,
        adapted_model_ref=candidate,
        all_original_parameters_frozen=True,
        original_parameters_require_grad_false=True,
        head_call_counts=dict(zero=105, raw=1680, adapt=1680),
        physics_steps=0,
        fresh_cases=0,
        new_teacher_calls=0,
        new_candidate_fits=0,
        new_episode_reads=0,
        wam_model_initializations=1,
        denoise_calls=3465,
        zero_identity=[{}] * 3,
        generations=[{}] * 96,
        runtime=dict(checkpoint_path="8400/model", ema_weights=False),
        denoise_ledger=[
            dict(index=i * 35 + j, label=label, completed=True) for i, label in enumerate(labels) for j in range(35)
        ],
    )
    return prepared, actual


def test_sampling_freeze_source_and_complete99_budget_cannot_be_relaxed(checker):
    prepared, actual = sampling_fixture()
    checker.sampling_execution_contract(actual, prepared)
    for key, value in (
        ("original_parameters_require_grad_false", False),
        ("denoise_calls", 3360),
        ("denoise_ledger", actual["denoise_ledger"][105:]),
        ("new_candidate_fits", 1),
        ("adapted_model_ref", dict(path="newfit.npz", sha256="d" * 64)),
    ):
        with pytest.raises(ValueError):
            checker.sampling_execution_contract(dict(actual, **{key: value}), prepared)


def test_precision_or_all99_contract_failure_prevents_any_future_label_read(checker, monkeypatch, tmp_path):
    prepared, actual = sampling_fixture()
    prepared["output_root"] = str(tmp_path)
    planned_ref = dict(path="synthetic-ref", sha256="a" * 64)
    actual.update(prepared_ref=planned_ref, precision_ref=planned_ref, precision_verification_ref=planned_ref)
    monkeypatch.setattr(checker, "reference", lambda path: planned_ref)
    events = []

    class Evidence:
        def json(self, path):
            return actual

    monkeypatch.setattr(checker, "precision_admission", lambda *args: events.append("precision"))
    monkeypatch.setattr(checker, "public_lifecycle", lambda *args: events.append("released"))
    monkeypatch.setattr(checker, "sampling_records", lambda *args: events.append("all99") or [])
    monkeypatch.setattr(checker, "score_states", lambda *args: (events.append("states") or {}, dict(passed=False)))
    result = checker.verify_sampling(prepared, {}, Path("precision"), Path("verified"), Path("sampling"), Evidence())
    assert events == ["precision", "released", "all99", "states"] and result["physics_authorized"] is False
    for boundary in ("precision_admission", "public_lifecycle", "sampling_records"):
        original = getattr(checker, boundary)

        def reject(*args):
            raise ValueError("incomplete prerequisite")

        monkeypatch.setattr(checker, boundary, reject)
        events.clear()
        with pytest.raises(ValueError, match="incomplete prerequisite"):
            checker.verify_sampling(prepared, {}, Path("precision"), Path("verified"), Path("sampling"), Evidence())
        assert "states" not in events
        monkeypatch.setattr(checker, boundary, original)


def test_original027_teacher_receipt_alone_cannot_admit_sampling(checker):
    candidate = dict(path="027model.npz", sha256="a" * 64)
    original = dict(
        complete=True,
        passed=True,
        candidate_model_ref=candidate,
        phase="fit",
        fit_verified=True,
        eligible_for_runtime_qualification=True,
        eligible_for_sampling=False,
    )

    class Evidence:
        def json(self, path):
            return original

    with pytest.raises(ValueError, match="actual precision admission"):
        checker.precision_admission(
            dict(adapted_model_ref=candidate), {}, Path("precision"), Path("027fit"), Evidence()
        )


def test_explicit_phase_arguments_reject_extra_or_missing_evidence(checker):
    fields = ("precision", "precision_verification", "sampling")
    for phase, count in (("prepare", 0), ("precision", 1), ("sampling", 3)):
        values = {key: Path("record.json") if i < count else None for i, key in enumerate(fields)}
        checker.phase_arguments(SimpleNamespace(phase=phase, **values))
        changed = dict(values)
        changed["sampling"] = Path("extra.json") if changed["sampling"] is None else None
        with pytest.raises(ValueError, match="explicit phase"):
            checker.phase_arguments(SimpleNamespace(phase=phase, **changed))


def test_state_gate_keeps_components_episode_seed_limits_and_never_authorizes_physics(checker):
    def group():
        return {
            str(h): {
                method: dict(
                    position_rmse_m=0.01,
                    rotation_rmse_rad=0.1,
                    opening_mae=0.1,
                    joint_score=9.0 if method == "adapt" else 10.0 if method == "raw" else 11.0,
                )
                for method in ("raw", "adapt", "hold", "constant_velocity")
            }
            for h in (8, 16, 40)
        }

    groups = dict(
        pooled=group(), by_episode={str(e): group() for e in range(6)}, by_seed={str(s): group() for s in (42, 43)}
    )
    assert checker.OLD.independent_state_gate(groups)["passed"]
    broken = copy.deepcopy(groups)
    broken["pooled"]["8"]["adapt"]["joint_score"] = np.nextafter(9.0, 10.0)
    assert not checker.OLD.independent_state_gate(broken)["passed"]
    broken = copy.deepcopy(groups)
    broken["by_seed"]["43"]["16"]["adapt"]["joint_score"] = 10.5000001
    assert not checker.OLD.independent_state_gate(broken)["passed"]
