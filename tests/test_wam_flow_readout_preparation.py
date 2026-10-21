"""026 preparation: cached provenance, whole-source isolation and release gates."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def preparer():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_wam_flow_readout.py"
    spec = importlib.util.spec_from_file_location("flow_readout_preparation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inherited(module):
    config = dict(
        domain=dict(name=module.DOMAIN, normalization_id=31, projection_id=0, active_slots=10),
        train_episodes=list(range(6)),
        validation_episodes=list(range(6, 12)),
    )
    old = dict(
        domain=dict(config["domain"], active_mask=module.ACTIVE.tolist()),
        episodes=[],
        anchors=[],
        teacher_calls=[],
        validation_generations=[],
        zero_identity_cases=[dict(task_index=i) for i in range(3)],
        input_sha256={},
    )
    png_inputs = {}
    for episode in range(12):
        split = "train" if episode < 6 else "validation"
        names = []
        for index, frame in enumerate((0, 16, 32, 48)):
            name = f"ep{episode}_a{index}"
            names.append(name)
            # Deliberately absent: metadata validation must not open these refs.
            image = f"/nonexistent/source/ep{episode}_a{index}.png"
            arrays = f"/nonexistent/source/ep{episode}.npz"
            request = dict(
                model_mode="wam",
                vision_path=image,
                prompt="task",
                action_chunk_size=40,
                num_frames=41,
                num_steps=35,
                fps=20,
                guidance=1.0,
                shift=5.0,
                seed=42,
                extra=dict(
                    action128=dict(
                        task_title="task",
                        action_domain=module.DOMAIN,
                        action_dim_mask=module.ACTIVE.tolist(),
                    )
                ),
            )
            anchor = dict(
                name=name,
                episode=episode,
                source_episode_id=f"source:{episode}",
                task_index=episode // 4,
                task="task",
                split=split,
                original_split="diagnostic" if episode < 6 else "study_heldout",
                anchor_index=index,
                anchor_frame=frame,
                arrays_path=arrays,
                arrays_sha256=f"array{episode}",
                image_path=image,
                image_sha256=f"image{episode}:{index}",
                request=request,
                valid_future_steps=40,
                state_mask128=module.ACTIVE.tolist(),
                measured_state128=[0.0] * 128,
            )
            old["anchors"].append(anchor)
            old["input_sha256"][arrays] = anchor["arrays_sha256"]
            png_inputs[image] = anchor["image_sha256"]
            for sigma in (0.2, 0.5, 0.8):
                old["teacher_calls"].append(
                    dict(
                        anchor_name=name,
                        call_index=len(old["teacher_calls"]),
                        noise_seed=42,
                        sigma=sigma,
                        split=split,
                    )
                )
            if split == "validation":
                for seed in (42, 43):
                    for variant in ("raw", "adapt"):
                        sample_request = copy.deepcopy(request)
                        sample_request["seed"] = seed
                        old["validation_generations"].append(
                            dict(
                                anchor_name=name,
                                generation_index=len(old["validation_generations"]),
                                name=f"{name}_{variant}_s{seed}",
                                pair_id=f"{name}_s{seed}",
                                request=sample_request,
                                sampling_seed=seed,
                                variant=variant,
                            )
                        )
        old["episodes"].append(
            dict(
                episode=episode,
                source_episode_id=f"source:{episode}",
                split=split,
                task_index=episode // 4,
                anchors=names,
                frames=[0, 16, 32, 48],
            )
        )
    return config, old, png_inputs


def test_cpu_environment_required_before_any_input(preparer, monkeypatch):
    monkeypatch.chdir(preparer.REPO)
    for cuda, ld in (("6", ""), ("0", ""), ("", "/tmp"), (None, "")):
        if cuda is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
        monkeypatch.setenv("LD_LIBRARY_PATH", ld)
        with pytest.raises(ValueError, match="explicitly empty CUDA/LD"):
            preparer.prepare()


def test_metadata_reuse_does_not_open_absent_episode_or_png_references(preparer):
    config, old, admitted = inherited(preparer)
    before = copy.deepcopy(old)
    assert preparer.validate_rosters(config, old, admitted) == dict(
        whole_source_episodes=12,
        anchors=48,
        teacher_calls=144,
        conditional_generations=96,
        zero_cases=3,
    )
    assert old == before


def test_whole_source_alias_leak_and_changed_split_rejected(preparer):
    config, old, admitted = inherited(preparer)
    old["episodes"][6]["source_episode_id"] = old["episodes"][0]["source_episode_id"]
    with pytest.raises(ValueError, match="whole-source EP leaks"):
        preparer.validate_rosters(config, old, admitted)
    config, old, admitted = inherited(preparer)
    old["anchors"][24]["arrays_sha256"] = "different"
    with pytest.raises(ValueError, match="array aliases"):
        preparer.validate_rosters(config, old, admitted)
    config, old, admitted = inherited(preparer)
    old["anchors"][24]["original_split"] = "diagnostic"
    with pytest.raises(ValueError, match="EP identity"):
        preparer.validate_rosters(config, old, admitted)


def test_real_masks_and_uniform_source_clock_are_inherited_without_repair(preparer):
    config, old, admitted = inherited(preparer)
    old["anchors"][0]["state_mask128"][9] = False
    with pytest.raises(ValueError, match="valid mask"):
        preparer.validate_rosters(config, old, admitted)
    config, old, admitted = inherited(preparer)
    old["episodes"][0]["frames"][1] = 17
    old["anchors"][1]["anchor_frame"] = 17
    with pytest.raises(ValueError, match="uniform anchor clock"):
        preparer.validate_rosters(config, old, admitted)
    config, old, admitted = inherited(preparer)
    old["anchors"][0]["valid_future_steps"] = 39
    with pytest.raises(ValueError, match="valid mask"):
        preparer.validate_rosters(config, old, admitted)


def test_domain_namespaces_teacher_order_and_paired_requests_cannot_change(preparer):
    config, old, admitted = inherited(preparer)
    config["domain"]["normalization_id"] = 0
    with pytest.raises(ValueError, match="normalization31"):
        preparer.validate_rosters(config, old, admitted)
    config, old, admitted = inherited(preparer)
    old["teacher_calls"][0]["sigma"] = 0.5
    with pytest.raises(ValueError, match="144 teacher"):
        preparer.validate_rosters(config, old, admitted)
    config, old, admitted = inherited(preparer)
    old["validation_generations"][1]["request"]["seed"] = 43
    with pytest.raises(ValueError, match="96 paired"):
        preparer.validate_rosters(config, old, admitted)


def cached(module):
    shapes = dict(
        hidden=(40, 1280),
        head_base=(40, 128),
        projection_ids=(40,),
        base=(40, 128),
        target=(40, 128),
        xt=(40, 128),
        valid=(40, 128),
        epsilon=(40, 128),
        x0_normalized=(40, 128),
        sigma_action=(40, 1),
        condition_action=(40, 1),
    )
    arrays = {k: np.zeros(v, np.float32) for k, v in shapes.items()}
    arrays["valid"] = np.broadcast_to(module.ACTIVE, (40, 128)).copy()
    arrays["projection_ids"] = np.zeros(40, np.int64)
    arrays["sigma_action"].fill(0.2)
    meta = dict(array_index={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()})
    return meta, arrays


def test_cache_actual_fp32_sigma_valid_mask_and_architecture_verified(preparer):
    metadata, arrays = cached(preparer)
    preparer.verify_cached_arrays(metadata, arrays, dict(sigma=0.2))
    assert float(arrays["sigma_action"][0, 0]) != 0.2
    arrays["valid"][39, 9] = False
    with pytest.raises(ValueError, match="real active10"):
        preparer.verify_cached_arrays(metadata, arrays, dict(sigma=0.2))
    metadata, arrays = cached(preparer)
    arrays["sigma_action"] = arrays["sigma_action"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype.*sigma_action"):
        preparer.verify_cached_arrays(metadata, arrays, dict(sigma=0.2))
    metadata, arrays = cached(preparer)
    arrays["sigma_action"][0] = 0.5
    with pytest.raises(ValueError, match="actual FP32 sigma"):
        preparer.verify_cached_arrays(metadata, arrays, dict(sigma=0.2))


def test_failed025_gate_cannot_be_promoted_by_reuse(preparer):
    ref = dict(path="/source/prepared.json", sha256="prepared")
    feature_ref = dict(path="/source/features.json", sha256="features")
    features = dict(
        complete=True,
        passed=True,
        all_original_parameters_frozen=True,
        denoise_calls=249,
        prepared_ref=ref,
    )
    shared = dict(
        passed=True,
        prepared=ref,
        teacher_calls_verified=144,
        zero_identity_generations_verified=3,
        actual_denoise_calls_verified=249,
        features_verified=True,
        input_sha256={ref["path"]: ref["sha256"], feature_ref["path"]: feature_ref["sha256"]},
    )
    verification = dict(shared, phase="features")
    failed = dict(
        shared,
        phase="fit",
        fit_verified=True,
        independent_model_reconstructions=2,
        eligible_for_sampling=True,
        teacher_gate=dict(passed=True, checks=dict(all_sigmas_nonregressing=False)),
    )
    with pytest.raises(ValueError, match="failed gate must remain False"):
        preparer.feature_admission(
            {},
            features,
            verification,
            failed,
            dict(source_prepared_ref=ref, source_features_ref=feature_ref),
            preparer.Evidence(),
        )


def test_birth_and_actual_physical_gpu_release_cannot_be_assumed(preparer, monkeypatch):
    monkeypatch.setattr(preparer, "birth", lambda *_args, **_kwargs: dict(start_ticks=10, proc_state="S"))
    with pytest.raises(ValueError, match="still live"):
        preparer.terminal(dict(pid=123, start_ticks=10))
    assert preparer.terminal(dict(pid=123, start_ticks=9))["original_birth_terminal"] is True

    def query(command, **_kwargs):
        if "--query-gpu=index,uuid,memory.used" in command:
            return f"6, {preparer.GPU_UUID}, 0\n"
        return ""

    monkeypatch.setattr(preparer.subprocess, "check_output", query)
    assert preparer.current_resource()["memory_mib"] == 0
    monkeypatch.setattr(preparer.subprocess, "check_output", lambda *_args, **_kwargs: "0, bad, 0")
    with pytest.raises(ValueError, match="GPU6 release"):
        preparer.current_resource()


def test_original_false_owner_and_actual_model_reap_preserved_without_launch_owner_key(preparer, monkeypatch, tmp_path):
    root = tmp_path / "old"
    stage = root / "orchestration/features"
    (stage / "model/ownership").mkdir(parents=True)
    (root / "preparation").mkdir()
    (root / "preparation/prepared.json").write_text("{}")

    def write(path, value):
        path.write_text(json.dumps(value))

    model = dict(pid=862631, start_ticks=170222073, ppid=862480)
    owner = dict(pid=862480, start_ticks=170221627)
    launch = dict(child=model, command=["python", "worker.py"], selected_environment=dict(CUDA_VISIBLE_DEVICES="6"))
    write(stage / "model/launch.json", launch)
    write(stage / "result.json", dict(complete=False, passed=False))
    write(stage / "model/exit.json", dict(actual_wait=True, automatic_retry=False, returncode=0))
    release = dict(**model, returncode=0, reaped=True, signal_sent=False, original_error=None)
    write(stage / "model/ownership/release.json", release)
    tool = tmp_path / "tool.json"
    write(
        tool,
        dict(
            tool_handle_terminal=True,
            tool_reported_exit_code=1,
            original_owner_result=preparer.reference(stage / "result.json"),
        ),
    )
    peer = tmp_path / "peer.json"
    write(
        peer,
        dict(
            passed=True,
            current_resource_release_verified=True,
            original_owner_passed=False,
            prepared=preparer.reference(root / "preparation/prepared.json"),
        ),
    )
    observed = tmp_path / "observed.json"
    write(
        observed,
        dict(
            passed=True,
            actual_model=model,
            actual_owner=owner,
            launch=preparer.reference(stage / "model/launch.json"),
        ),
    )
    for name, path in (("OWNER_EXIT", tool), ("RELEASE", peer), ("LAUNCH_PROOF", observed)):
        monkeypatch.setattr(preparer, name, path)
        monkeypatch.setattr(preparer, f"{name}_SHA", preparer.sha256(path))
    monkeypatch.setattr(preparer, "terminal", lambda item: dict(expected=item, original_birth_terminal=True))
    monkeypatch.setattr(preparer, "current_resource", lambda: dict(physical_gpu=6, memory_mib=0, compute_apps=[]))
    old = dict(output_root=str(root), config=dict(output_dir=str(root)))
    admitted = preparer.resource_admission(old, preparer.Evidence())
    assert admitted["original_owner_passed"] is False and admitted["original_owner_tool_exit"] == 1
    assert admitted["original_births"][1]["expected"] == owner
    write(stage / "result.json", dict(complete=True, passed=True))
    with pytest.raises(ValueError, match="owner False"):
        preparer.resource_admission(old, preparer.Evidence())
    write(stage / "result.json", dict(complete=False, passed=False))
    release["reaped"] = False
    write(stage / "model/ownership/release.json", release)
    with pytest.raises(ValueError, match="normal exit/reap"):
        preparer.resource_admission(old, preparer.Evidence())


def test_missing_new_source_fails_closed_and_evidence_detects_mutation(preparer, monkeypatch, tmp_path):
    monkeypatch.setattr(preparer, "LOCAL_SOURCES", ("not-yet-created-model-driver.py",))
    with pytest.raises(FileNotFoundError):
        preparer.source_closure(dict(source_sha256={}), dict(cosmos_repo=str(tmp_path)), preparer.Evidence())
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(passed=True)))
    evidence = preparer.Evidence()
    evidence.json(source)
    source.write_text(json.dumps(dict(passed=False)))
    with pytest.raises(ValueError, match="SHA differs"):
        evidence.recheck()
