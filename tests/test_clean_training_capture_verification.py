"""Synthetic capture/receipt contracts only; no model, future label or real exit claims."""

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from s2a.artifacts import REPO
from s2a.wam_clean_capture import CleanCapturePoint
from s2a.wam_clean_compact import CompactCaptureSink, CompactRequest


@pytest.mark.parametrize("seed", [42, 11])
@pytest.mark.parametrize("fault", [None, "branch", "future_plan", "subtask", "task"])
def test_caption_matches_actual_seed_renderer_without_adding_missing_plan(checker, monkeypatch, seed, fault):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    producer = importlib.import_module("collect_wam_clean_training")
    # Published renderer gold: seed42 draw=.4583 includes[], seed11=.9012 omits.
    caption = dict(
        task_title="current task",
        command="current task",
        embodiment="panda",
        duration="2s",
        fps=20,
        action_frequency_hz=20,
        resolution={"W": 960, "H": 640},
        aspect_ratio="3:2",
    )
    if seed == 42:
        caption["task_plan"] = []
    if fault == "branch":
        if seed == 42:
            del caption["task_plan"]
        else:
            caption["task_plan"] = []
    elif fault == "future_plan":
        caption["task_plan"] = ["expert future"]
    elif fault == "subtask":
        caption["current_subtask"] = "expert future"
    elif fault == "task":
        caption["command"] = "different task"
    original = copy.deepcopy(caption)
    for check in (producer.validate_training_caption, checker.training_caption):
        if fault is None:
            check(caption, "current task", seed)
        else:
            with pytest.raises(ValueError):
                check(caption, "current task", seed)
    assert caption == original


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("verify_wam_clean_training_capture")


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "tests"))
    return importlib.import_module("test_wam_clean_compact")


def ref(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def online(protocol, step=0, adapted=False):
    original = protocol.point(step, adapted=adapted)
    arrays = dict(original.arrays)
    arrays["condition_vision"] = np.zeros((protocol.VISION[2], 1, 1), np.float32)
    arrays["condition_vision"][0] = 1
    return CleanCapturePoint(original.metadata, arrays)


def fixture_point(tmp_path, checker, protocol, *, adapted=False):
    current = tmp_path / "request"
    writer = CompactCaptureSink(
        current / "compact",
        CompactRequest(
            "native-anchor0",
            101,
            "training",
            "D1" if adapted else "D0",
            "task0/demo2/a0",
            "b" * 64 if adapted else None,
            "a" * 64,
        ),
        expected_vision_shape=protocol.VISION,
        expected_grid=protocol.GRID,
    )
    original = online(protocol, adapted=adapted)
    point_ref = writer(original)
    evidence = checker.Evidence()
    record = evidence.json(point_ref)
    arrays = evidence.arrays(record["arrays_ref"], current / "compact/step00.npz")
    header = evidence.json(record["request_ref"])
    step = dict(
        step_index=0,
        solver_sigma=protocol.GRID[0][0],
        solver_timestep=protocol.GRID[0][1],
        returned_original_velocity_object=not adapted,
    )
    coefficients = np.zeros((1291, 10))
    coefficients[-1] = protocol.GRID[0][0] / 8
    head = (coefficients, np.zeros(1290), 1.0) if adapted else None
    return record, arrays, header, step, head, original


@pytest.mark.parametrize("adapted", [False, True])
def test_independent_saved_action_math_checks_raw_and_fixed_w1(tmp_path, checker, protocol, adapted):
    record, arrays, header, step, head, _ = fixture_point(tmp_path, checker, protocol, adapted=adapted)
    changed = checker.verify_point(
        record, arrays, header, step, head=head, vision_shape=protocol.VISION, time_vision_shape=(2,)
    )
    assert changed == (400 if adapted else 0)


@pytest.mark.parametrize("fault", [None, "default_training", "wrong_eval_seed", "unknown_purpose"])
def test_evaluation_math_uses_explicit_purpose_without_rewriting_witnesses(tmp_path, checker, protocol, fault):
    record, arrays, header, step, _, _ = fixture_point(tmp_path, checker, protocol)
    record["metadata"].update(purpose="evaluation", noise_seed=42)
    header["request"].update(purpose="evaluation", noise_seed=42)
    if fault == "wrong_eval_seed":
        record["metadata"]["noise_seed"] = header["request"]["noise_seed"] = 101
    before = copy.deepcopy((record, header))
    kwargs = dict(vision_shape=protocol.VISION, time_vision_shape=(2,))
    if fault != "default_training":
        kwargs["purpose"] = "invalid" if fault == "unknown_purpose" else "evaluation"
    if fault is None:
        assert checker.verify_point(record, arrays, header, step, **kwargs) == 0
    else:
        with pytest.raises(ValueError, match="purpose"):
            checker.verify_point(record, arrays, header, step, **kwargs)
    assert (record, header) == before


def test_evaluation_trace_preserves_declared_purpose_and_rejects_default_training(checker, protocol):
    trace, sources, noise = trace_fixture(protocol)
    trace.update(purpose="evaluation", seed=42)
    trace["actual_sampler_arguments"]["seed"] = [42]
    entry = dict(name="native-anchor0", sampling_seed=42)
    original = copy.deepcopy(trace)
    with pytest.raises(ValueError, match="purpose"):
        checker.verify_trace(trace, entry, protocol.GRID, sources, noise)
    checker.verify_trace(trace, entry, protocol.GRID, sources, noise, purpose="evaluation")
    assert trace == original


@pytest.mark.parametrize(
    "fault", ["c0_from_bf16", "delta_sign", "wrong_w1", "double_delegate", "future_array", "omitted_witness"]
)
def test_rehashed_capture_cannot_hide_wrong_math_or_future_values(tmp_path, checker, protocol, fault):
    record, arrays, header, step, head, _ = fixture_point(tmp_path, checker, protocol, adapted=True)
    if fault == "c0_from_bf16":
        arrays["c0_active10"] = np.ones((40, 10)) - step["solver_sigma"]
    elif fault == "delta_sign":
        arrays["velocity_delta_active10"] *= -1
    elif fault == "wrong_w1":
        head[0][-1] += 0.1
    elif fault == "future_array":
        arrays["target_clean_residual"] = np.zeros((40, 10))
    elif fault == "omitted_witness":
        record["online_checks"]["vision_velocity_bits_unchanged"] = False
    else:
        step["returned_original_velocity_object"] = True
    record["compact_arrays"] = {k: checker.summary(a) for k, a in arrays.items()}
    for name in checker.SMALL:
        record["source_arrays"][name].update(checker.summary(arrays[name]))
    with pytest.raises(ValueError):
        checker.verify_point(
            record, arrays, header, step, head=head, vision_shape=protocol.VISION, time_vision_shape=(2,)
        )


def test_direct_bf16_avoids_float32_midpoint_double_rounding(checker):
    midpoint = np.float64(1 + 2**-8)
    values = np.array([midpoint, np.nextafter(midpoint, np.inf), -np.nextafter(midpoint, np.inf), 2**-134, 2**-133])
    np.testing.assert_array_equal(
        checker.deploy(values, "torch.bfloat16"), np.array([1, 1 + 2**-7, -(1 + 2**-7), 0, 2**-133], np.float32)
    )


@pytest.mark.parametrize(
    "fault", [None, "future_vision", "active_condition", "inactive_unclamped", "current_latent", "sample_sha"]
)
def test_full_initial_mask_current_prefix_and_actual_first_sample_binding(tmp_path, checker, protocol, fault):
    record, arrays, _, _, _, original = fixture_point(tmp_path, checker, protocol)
    noise = original.arrays["sample_fp32_flat"].copy()
    vision_mask = np.zeros(protocol.VISION, np.float32)
    vision_mask[:, :, :1] = 1
    action_mask = np.ones((40, 128), np.float32)
    action_mask[:, :10] = 0
    initial = dict(
        initial_noise=noise,
        condition_mask=np.concatenate((vision_mask.reshape(-1), action_mask.reshape(-1))),
        first_condition_latent=noise[:8].reshape(protocol.VISION)[:, :, :1].copy(),
    )
    metadata = dict(
        name="native-anchor0",
        generation_index=0,
        sampling_seed=101,
        inference_only_current_image=True,
        caption="current-task",
    )
    entry = dict(name="native-anchor0", index=0, sampling_seed=101)
    first = dict(record, _sample_active=arrays["sample_active10_fp32"])
    if fault == "future_vision":
        initial["condition_mask"][4] = 1
    elif fault == "active_condition":
        initial["condition_mask"][8] = 1
    elif fault == "inactive_unclamped":
        initial["initial_noise"][18] = 1
    elif fault == "current_latent":
        initial["first_condition_latent"][0, 0, 0, 0, 0] += 1
    elif fault == "sample_sha":
        first = copy.deepcopy(first)
        first["source_arrays"]["sample_fp32_flat"]["snapshot_sha256"] = "0" * 64
    if fault is None:
        checker.verify_initial(metadata, initial, entry, "current-task", first, vision_shape=protocol.VISION)
    else:
        with pytest.raises(ValueError):
            checker.verify_initial(metadata, initial, entry, "current-task", first, vision_shape=protocol.VISION)


def trace_fixture(protocol):
    steps = [
        dict(
            step_index=i,
            solver_step_index_before=None if i == 0 else i,
            solver_sigma=sigma,
            solver_timestep=time,
            denoise_calls=1,
            head_calls=1,
            velocity_delegations=1,
            conversion_calls=1,
            conversion_velocity_is_deployed_storage_view=True,
            conversion_sample_is_original_storage_view=True,
            conversion_returned_original_object=True,
            returned_original_velocity_object=True,
        )
        for i, (sigma, time) in enumerate(protocol.GRID)
    ]
    source, noise = {"frozen-source": "a" * 64}, dict(training_seeds=[101, 102], evaluation_seeds=[42, 43])
    trace = dict(
        complete=True,
        name="native-anchor0",
        seed=101,
        purpose="training",
        sampler_calls=1,
        parameters_before={"version": 1},
        parameters_after={"version": 1},
        source_sha256=source,
        noise_roster=noise,
        feature_terms=[],
        solver_sigma_dtype="torch.float32",
        terminal_sigma=0,
        terminal_neural_calls=0,
        steps=steps,
        sigma_values=[x[0] for x in protocol.GRID] + [0.0],
        scheduler_timesteps=[x[1] for x in protocol.GRID],
        actual_sampler_arguments=dict(num_steps=35, shift=5.0, seed=[101]),
    )
    return trace, source, noise


@pytest.mark.parametrize("fault", [None, "short", "conversion", "terminal", "parameters", "source"])
def test_trace_requires_complete_grid_and_single_actual_storage_delegation(checker, protocol, fault):
    trace, source, noise = trace_fixture(protocol)
    if fault == "short":
        trace["steps"].pop()
    elif fault == "conversion":
        trace["steps"][3]["conversion_velocity_is_deployed_storage_view"] = False
    elif fault == "terminal":
        trace["terminal_neural_calls"] = 1
    elif fault == "parameters":
        trace["parameters_after"]["version"] += 1
    elif fault == "source":
        trace["source_sha256"] = {}
    entry = dict(name="native-anchor0", sampling_seed=101)
    if fault is None:
        checker.verify_trace(trace, entry, protocol.GRID, source, noise)
    else:
        with pytest.raises(ValueError):
            checker.verify_trace(trace, entry, protocol.GRID, source, noise)


def test_complete35_compact_chain_is_read_independently_and_reordering_rejected(tmp_path, checker, protocol):
    current = tmp_path / "request"
    writer = CompactCaptureSink(
        current / "compact",
        CompactRequest("native-anchor0", 101, "training", "D0", "task0/demo2/a0", None, "a" * 64),
        expected_vision_shape=protocol.VISION,
        expected_grid=protocol.GRID,
    )
    for i in range(35):
        writer(online(protocol, i))
    evidence = checker.Evidence()
    ledger = evidence.json(writer.finish())
    header = evidence.json(ledger["request_ref"])
    trace, _, _ = trace_fixture(protocol)
    first, count, changed = checker.audit_points(
        evidence, ledger, header, current, trace["steps"], vision_shape=protocol.VISION, time_vision_shape=(2,)
    )
    assert count == 35 and changed == 0 and first["metadata"]["step_index"] == 0
    evidence.finish()
    ledger["points"][0], ledger["points"][1] = ledger["points"][1], ledger["points"][0]
    with pytest.raises(ValueError, match="path/request/order"):
        checker.audit_points(
            evidence, ledger, header, current, trace["steps"], vision_shape=protocol.VISION, time_vision_shape=(2,)
        )


def test_opaque_future_archive_is_rejected_before_any_decoder(tmp_path, checker, monkeypatch):
    future = tmp_path / "trajectory.npz"
    future.write_bytes(b"future archive must remain opaque")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("future NPZ was decoded"))
    with pytest.raises(ValueError, match="escaped"):
        checker.Evidence().arrays(ref(future), tmp_path / "capture/compact/step00.npz")


@pytest.mark.parametrize(
    "fault", [None, "fake_wait", "signal", "wrong_birth", "gpu_memory", "gpu_minor", "wrong_round", "wrong_python"]
)
def test_normal_exit_and_gpu_release_are_required_external_receipts(tmp_path, checker, fault):
    prepared = dict(path="/prepared.json", sha256="a" * 64)
    registration = dict(path="/execution.json", sha256="b" * 64)
    launch = dict(
        child=dict(pid=123, start_ticks=456),
        selected_environment=dict(CUDA_VISIBLE_DEVICES=checker.GPU, LD_LIBRARY_PATH=""),
        command=[
            "model-python",
            "scripts/collect_wam_clean_training.py",
            "--prepared",
            prepared["path"],
            "--prepared-sha256",
            prepared["sha256"],
            "--execution-registration",
            registration["path"],
            "--execution-registration-sha256",
            registration["sha256"],
            "--round",
            "D0",
            "--gpu-uuid",
            checker.GPU,
            "--seed",
            "42",
        ],
    )
    exit_record = dict(returncode=0, actual_wait=True, automatic_retry=False)
    release = dict(
        pid=123,
        start_ticks=456,
        reaped=True,
        returncode=0,
        original_error=None,
        signal_sent=False,
        only_owned_child_signalled=True,
    )
    gpu = dict(released=True, uuid=checker.GPU, memory_mib=0, compute_apps=[], device_minor=6, visible_gpu=2)
    if fault == "fake_wait":
        exit_record["actual_wait"] = False
    elif fault == "signal":
        release["signal_sent"] = True
    elif fault == "wrong_birth":
        release["start_ticks"] += 1
    elif fault == "gpu_memory":
        gpu["memory_mib"] = 1
    elif fault == "gpu_minor":
        gpu["device_minor"] = 0
    elif fault == "wrong_round":
        launch["command"][launch["command"].index("--round") + 1] = "D1"
    elif fault == "wrong_python":
        launch["command"][0] = "other-model-python"
    references = {}
    for name, value in (
        ("worker_launch", launch),
        ("worker_exit", exit_record),
        ("worker_release", release),
        ("gpu_after", gpu),
    ):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        references[name + "_ref"] = ref(path)
    if fault is None:
        checker.lifecycle(
            checker.Evidence(),
            references,
            prepared,
            registration,
            dict(round_name="D0"),
            tmp_path,
            model_python="model-python",
        )
    else:
        with pytest.raises(ValueError):
            checker.lifecycle(
                checker.Evidence(),
                references,
                prepared,
                registration,
                dict(round_name="D0"),
                tmp_path,
                model_python="model-python",
            )


def test_cli_refuses_metadata_only_admission_without_creating_a_pass(tmp_path):
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps(dict(schema_version="s2a.clean-training-preparation.v1")))
    registration = tmp_path / "execution.json"
    registration.write_text(
        json.dumps(
            dict(
                schema_version="s2a.clean-training-execution-registration.v1",
                prepared_ref=ref(prepared),
                passed=False,
                complete=False,
                readiness=dict(ready_for_execution=False),
                automatic_retry=False,
                fresh_performance_authorized=False,
            )
        )
    )
    output = tmp_path / "unwritten.json"
    command = [sys.executable, str(REPO / "scripts/verify_wam_clean_training_capture.py")]
    for name in (
        "prepared",
        "execution-registration",
        "collection",
        "worker-launch",
        "worker-exit",
        "worker-release",
        "gpu-after",
    ):
        path = registration if name == "execution-registration" else prepared
        command += ["--" + name, str(path), "--" + name + "-sha256", ref(path)["sha256"]]
    command += ["--output", str(output), "--seed", "42"]
    result = subprocess.run(
        command,
        cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and "independent final execution registration required" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "producer_schema",
        "not_metadata_qualified",
        "execution_ready",
        "execution_authorized",
        "not_rehashed",
        "wrong_preparation",
        "unfrozen_verifier",
        "changed_verifier",
    ],
)
def test_independent_metadata_gate_requires_its_frozen_verifier(tmp_path, checker, fault):
    verifier = tmp_path / "independent.py"
    verifier.write_text("# synthetic independent verifier\n")
    prepared = dict(path="/registered/prepared.json", sha256="a" * 64)
    metadata = dict(
        schema_version="s2a.clean-training-preparation-independent.v1",
        metadata_qualified=True,
        execution_ready=False,
        execution_authorized=False,
        consumed_before_after_exact=True,
        passed=True,
        complete=True,
        prepared_ref=prepared,
        verifier_ref=ref(verifier),
    )
    registration = dict(source_sha256={str(verifier.resolve()): ref(verifier)["sha256"]})
    if fault == "producer_schema":
        metadata["schema_version"] = "s2a.clean-training-preparation.v1"
    elif fault == "not_metadata_qualified":
        metadata["metadata_qualified"] = False
    elif fault in ("execution_ready", "execution_authorized"):
        metadata[fault] = True
    elif fault == "not_rehashed":
        metadata["consumed_before_after_exact"] = False
    elif fault == "wrong_preparation":
        metadata["prepared_ref"] = dict(prepared, sha256="b" * 64)
    elif fault == "unfrozen_verifier":
        registration["source_sha256"] = {}
    elif fault == "changed_verifier":
        verifier.write_text("# changed after registration\n")
    if fault is None:
        checker.independent_metadata(checker.Evidence(), metadata, prepared, registration)
    else:
        with pytest.raises(ValueError):
            checker.independent_metadata(checker.Evidence(), metadata, prepared, registration)


@pytest.mark.parametrize("fault", [None, "future", "future_action", "wrong_seed", "wrong_image", "wrong_camera"])
def test_current_request_refuses_extra_future_inputs(checker, fault):
    entry = dict(name="train_anchor", sampling_seed=101)
    anchor = dict(task="current task", image_ref=dict(path="/current.png"))
    action = dict(
        action_dim_mask=checker.MASK,
        action_domain=checker.DOMAIN,
        action_fps=20,
        camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
        return_video=False,
        selected_roles=["auxiliary_0", "left_wrist"],
        task_title="current task",
    )
    request = dict(
        action_chunk_size=40,
        domain_name="franka_panda__parallel_gripper",
        extra=dict(action128=action),
        fps=20,
        guidance=1.0,
        image_size=960,
        model_mode="wam",
        name="train_anchor",
        num_frames=41,
        num_steps=35,
        prompt="current task",
        seed=101,
        shift=5.0,
        vision_path="/current.png",
    )
    if fault == "future":
        request["future_states"] = [1]
    elif fault == "future_action":
        action["action"] = [[1]]
    elif fault == "wrong_seed":
        request["seed"] = 42
    elif fault == "wrong_image":
        request["vision_path"] = "/future.png"
    elif fault == "wrong_camera":
        action["selected_roles"] = ["future_camera"]
    if fault is None:
        checker.current_request(request, entry, anchor)
    else:
        with pytest.raises(ValueError):
            checker.current_request(request, entry, anchor)


@pytest.mark.parametrize("fault", [None, "seed", "text", "noise_bit", "current_bit", "array_roster"])
def test_d1_requires_exact_d0_initial_noise_and_current_condition(checker, fault):
    original = dict(sampling_seed=101, caption="current task", text_tokens=[1, 2], inference_only_current_image=True)
    metadata = copy.deepcopy(original)
    arrays = dict(
        initial_sample=np.zeros((1, 8), np.float32),
        condition_mask=np.ones((1, 8), np.float32),
        initial_current_latent=np.zeros((1, 2), np.float32),
    )
    paired = {k: a.copy() for k, a in arrays.items()}
    if fault == "seed":
        metadata["sampling_seed"] += 1
    elif fault == "text":
        metadata["text_tokens"] = [1, 3]
    elif fault == "noise_bit":
        arrays["initial_sample"][0, 0] = -0.0
    elif fault == "current_bit":
        arrays["initial_current_latent"][0, 0] = np.nextafter(np.float32(0), np.float32(1))
    elif fault == "array_roster":
        arrays["extra"] = np.zeros(1)
    if fault is None:
        checker.paired_initial(metadata, arrays, original, paired)
    else:
        with pytest.raises(ValueError):
            checker.paired_initial(metadata, arrays, original, paired)
