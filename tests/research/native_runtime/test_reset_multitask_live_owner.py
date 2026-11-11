"""Exercise GPU binding, child reaping and fixed-denominator failure reporting."""

import json
import sys

import pytest

from s2a.infrastructure.lifecycle.owned_socket_pair import OwnedSocketPairInterrupted
from s2a.infrastructure.publication import publish_result as publish
from scripts.simulation.runtime import reset_multitask_live_admission as admission
from scripts.simulation.runtime import reset_multitask_live_workers as workers
from scripts.simulation.runtime import run_reset_multitask_live_48 as owner


def test_gpu_observation_queries_only_registered_uuid(monkeypatch):
    first, second = admission.CASES[:2]
    responses = iter(
        [
            f"{first['gpu_uuid']}, 0\n{second['gpu_uuid']}, 90\n",
            f"{second['gpu_uuid']}, 123, 90\n",
        ]
    )
    monkeypatch.setattr(owner.subprocess, "check_output", lambda *_args, **_kwargs: next(responses))
    assert owner.gpu_observation(first["gpu_uuid"]) == dict(memory_mib=0, compute_apps=[])
    with pytest.raises(ValueError, match="registered GPU"):
        owner.gpu_observation("0")


def test_allocated_case_ports_are_distinct():
    ports = owner.allocate_ports([0, 4, 8])
    assert set(ports) == {"0", "4", "8"} and len(set(ports.values())) == 3
    assert all(0 < value < 65536 for value in ports.values())


def test_only_explicit_prelaunch_rejection_can_have_no_lifecycle_receipt():
    assert owner.closed_descendants({"worker_launch_attempted": False})
    assert not owner.closed_descendants({"worker_launch_attempted": True})
    assert not owner.closed_descendants({})


def test_interrupted_case_reaps_its_pair_and_preserves_gpu_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "OUTPUT", tmp_path)
    directory = workers.case_directory("producer", 0)
    (directory / "simulation").mkdir(parents=True)
    (directory / "model").mkdir()
    inputs = dict(config=dict(worker_timeout_seconds=60), prepared=dict(model_config=dict(model_python=sys.executable)))
    monkeypatch.setattr(owner, "worker_inputs", lambda *_args: (inputs, dict(master_ports={"0": 34567})))
    monkeypatch.setattr(
        owner.physical, "_cpu_environment", lambda _directory: {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
    )
    monkeypatch.setattr(owner.prior_owner, "_model_environment", lambda _directory: {"LD_LIBRARY_PATH": ""})
    monkeypatch.setattr(owner, "gpu_observation", lambda _uuid: dict(memory_mib=0, compute_apps=[]))

    def interrupt(specs, *, timeout_seconds):
        assert timeout_seconds == 60
        assert specs["simulation"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert specs["model"]["env"]["CUDA_VISIBLE_DEVICES"] == admission.CASES[0]["gpu_uuid"]
        assert specs["model"]["env"]["MASTER_PORT"] == "34567"
        raise OwnedSocketPairInterrupted(
            dict(
                workers=[
                    dict(name="simulation", actual_wait=True, reaped=True),
                    dict(name="model", actual_wait=True, reaped=True),
                ]
            )
        )

    monkeypatch.setattr(owner, "run_owned_socket_pair", interrupt)
    with pytest.raises(OwnedSocketPairInterrupted):
        owner.case_owner(0, "producer")
    result = json.loads((directory / "result.json").read_text())
    assert result["worker_launch_attempted"] and result["interrupted"] and result["gpu_released"]
    assert not result["passed"] and owner.closed_descendants(result)


@pytest.mark.parametrize("failed_index", [None, 4])
def test_owner_keeps_three_case_denominator_and_replays_only_completed(tmp_path, monkeypatch, failed_index):
    monkeypatch.setattr(admission, "OUTPUT", tmp_path)
    monkeypatch.setattr(owner, "OUTPUT", tmp_path)
    inputs = dict(config=admission.FIXED, source_sha256={})
    monkeypatch.setattr(admission, "load_inputs", lambda **_kwargs: inputs)
    monkeypatch.setattr(admission, "input_receipt", lambda _inputs: {"synthetic": True})
    monkeypatch.setattr(admission, "source_refs", lambda: [])
    monkeypatch.setattr(owner, "model_bundle", lambda _inputs, index: {"case_index": index})
    monkeypatch.setattr(owner, "check_qa_and_dry", lambda _inputs: {})
    monkeypatch.setattr(owner, "checked_preflight", lambda _inputs: {})
    monkeypatch.setattr(owner, "snapshot", lambda _root: {"sources": {}})
    monkeypatch.setattr(owner, "gpu_observation", lambda _uuid: dict(memory_mib=0, compute_apps=[]))
    launched = []

    def fake_workers(specs, *, cwd, timeout_seconds):
        assert cwd == owner.ROOT and timeout_seconds == admission.FIXED["worker_timeout_seconds"] + 30
        lifecycle = []
        for spec in specs:
            index = int(spec["key"])
            phase = "checker" if "case-checker" in spec["argv"] else "producer"
            launched.append((phase, index))
            directory = workers.case_directory(phase, index)
            children = publish(directory / "workers.json", dict(workers=[dict(actual_wait=True, reaped=True)]))
            passed = phase == "checker" or index != failed_index
            counts = dict(
                control_steps=16, native_steps=400, models=3, contexts=3, model_requests=int(phase == "producer")
            )
            case = dict(
                complete=passed,
                passed=passed,
                worker_launch_attempted=True,
                workers_ref=children,
                branch_ref={"path": f"branch-{index}", "sha256": "b" * 64},
                model_ref={"path": f"model-{index}", "sha256": "c" * 64},
                counts=counts,
            )
            publish(directory / "result.json", case)
            lifecycle.append(
                dict(key=str(index), actual_wait=True, reaped=True, returncode=0 if passed else 1, signal_sent=False)
            )
        return dict(workers=lifecycle, launch_error=None)

    monkeypatch.setattr(owner, "run_owned_workers", fake_workers)
    if failed_index is None:
        owner.execute("producer")
    else:
        with pytest.raises(ValueError, match="All registered cases"):
            owner.execute("producer")
    result = json.loads((tmp_path / "producer/result.json").read_text())
    assert result["registered_cases"] == [0, 4, 8]
    assert result["completed_cases"] == (3 if failed_index is None else 2)
    assert result["owners_closed"] and result["descendants_reaped"] and result["gpu_released"]
    assert launched == [("producer", 0), ("producer", 4), ("producer", 8)]
    if failed_index is not None:
        with pytest.raises(ValueError, match="All registered cases"):
            owner.execute("checker")
        replay = json.loads((tmp_path / "checker/result.json").read_text())
        assert replay["registered_cases"] == [0, 4, 8] and replay["completed_cases"] == 2
        assert replay["cases"]["4"]["not_executed"] == "producer case unavailable"
        assert launched[-2:] == [("checker", 0), ("checker", 8)]
        assert replay["automatic_retry"] is False


def test_model_bundle_is_bound_to_case_gpu_task_and_endpoint(monkeypatch):
    spec = admission.CASES[0]
    initial = dict(step=8, sim_time_s=0.65, state_sha256="a" * 64)
    receipt = dict(cases=[dict(spec=spec, initial=initial)])
    weights = {f"/weight/{index}": {"sha256": f"{index:064x}"} for index in range(163)}
    bundle = dict(
        input_receipt=receipt,
        config=admission.FIXED,
        prepared=dict(weight_files=weights, model_config=dict(model_python="/mock/python")),
        weight_hashes={path: value["sha256"] for path, value in weights.items()},
        model_registration=dict(
            task=spec["task"],
            image_root=str((workers.case_directory("producer", 0) / "simulation/runtime").relative_to(owner.ROOT)),
            session="reset_multitask_case0_live_s42_20261111",
            initial=dict(initial),
            seed=42,
            model_sha256=admission.FIXED["model_sha256"],
            max_requests=5,
            denoise_steps=35,
        ),
    )
    registration = dict(phase="producer", inputs=receipt, source_refs=["source"], model_bundle_refs={"0": "bundle"})
    preflight = dict(
        complete=True,
        passed=True,
        inputs=receipt,
        source_refs=["source"],
        gpu_before={item["gpu_uuid"]: dict(memory_mib=0, compute_apps=[]) for item in admission.CASES},
        weight_files_verified=163,
        qa_ref=str(owner.QA),
        dry_ref=str(owner.DRY),
        weight_stats={"unchanged": True},
        model_python_sha256="python-sha",
    )
    values = {
        str(owner.OUTPUT / "producer/registration.json"): registration,
        "bundle": bundle,
        str(owner.OUTPUT / "preflight.json"): preflight,
    }
    monkeypatch.setattr(owner.prefix, "reference", lambda path: str(path))
    monkeypatch.setattr(owner.prefix, "read", lambda ref: values[ref])
    monkeypatch.setattr(admission, "source_refs", lambda: ["source"])
    monkeypatch.setattr(owner.prior_owner, "file_stats", lambda _weights: {"unchanged": True})
    monkeypatch.setattr(owner.repro, "sha256", lambda _path: "python-sha")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", spec["gpu_uuid"])
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    for key in owner.prefix.THREAD_VARS:
        monkeypatch.setenv(key, "1")
    assert owner.model_worker_inputs(0)[0] is bundle
    bundle["model_registration"]["task"] = admission.CASES[1]["task"]
    with pytest.raises(ValueError, match="own task, GPU and current endpoint"):
        owner.model_worker_inputs(0)
    bundle["model_registration"]["task"] = spec["task"]
    bundle["model_registration"]["initial"]["state_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="own task, GPU and current endpoint"):
        owner.model_worker_inputs(0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", admission.CASES[1]["gpu_uuid"])
    with pytest.raises(ValueError, match="registered GPU and environment"):
        owner.model_worker_inputs(0)
