"""Current-image WAM collection with preview S2A and archived measured endpoints."""

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.coupled_preview_wam_control import CHECKPOINT, make_case_config
from s2a.coupled_preview_wam_execution import PORT, admit_development
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_policy import case_order

REQUEST_KEYS = {
    "name",
    "model_mode",
    "prompt",
    "vision_path",
    "domain_name",
    "image_size",
    "action_chunk_size",
    "fps",
    "num_frames",
    "seed",
    "num_steps",
    "guidance",
    "shift",
    "extra",
}
ACTION_KEYS = {
    "action_domain",
    "action_dim_mask",
    "action_fps",
    "task_title",
    "selected_roles",
    "camera_layout",
    "return_video",
}


@contextmanager
def bounded_http(log, case, directory):
    """Process-local transport cap; request JSON and frozen controls are untouched."""
    original = requests.Session

    class BoundedSession(original):
        def request(self, method, url, **kwargs):
            require(
                (method.upper(), url)
                in (("GET", f"http://127.0.0.1:{PORT}/info"), ("POST", f"http://127.0.0.1:{PORT}/generate")),
                "unregistered worker HTTP route",
            )
            timeout = kwargs.get("timeout")
            require(type(timeout) in (int, float) and timeout > 0, "explicit positive HTTP timeout required")
            kwargs["timeout"] = min(timeout, 300)
            body = kwargs.get("json")
            if method.upper() == "POST":
                require(
                    isinstance(body, dict)
                    and set(body) == REQUEST_KEYS
                    and set(body["extra"]) == {"action128"}
                    and set(body["extra"]["action128"]) == ACTION_KEYS,
                    "privileged/future worker HTTP input",
                )
                require(
                    body["seed"] == case["sampling_seed"] and Path(body["vision_path"]).resolve().parent == directory,
                    "worker HTTP seed/current-image case binding differs",
                )
            record = dict(
                method=method.upper(),
                url=url,
                timeout_s=kwargs["timeout"],
                request_name=None if body is None else body["name"],
                request_canonical_sha256=None
                if body is None
                else hashlib.sha256(
                    json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
                ).hexdigest(),
            )
            try:
                response = super().request(method, url, **kwargs)
                record.update(completed=True, status_code=response.status_code)
                return response
            except BaseException as error:
                record.update(completed=False, error=dict(type=type(error).__name__, message=str(error)))
                raise
            finally:
                log.write(json.dumps(record) + "\n")
                log.flush()

    requests.Session = BoundedSession
    try:
        yield
    finally:
        requests.Session = original


def execute(
    prepared_ref, registration_ref, state_owner_ref, state_verification_ref, config_ref, *, case_index, dry=False
):
    strict_native_environment()
    require(type(case_index) is int and 0 <= case_index < 12, "fixed development case index required")
    case = case_order()[case_index]
    prepared, registration, _, _, control = admit_development(
        prepared_ref,
        registration_ref,
        state_owner_ref,
        state_verification_ref,
        required_sources=[
            __file__,
            REPO / "scripts/run_coupled_preview_wam_development.py",
            REPO / "src/s2a/native_environment.py",
        ],
    )
    config = read(config_ref)
    require(
        config
        == make_case_config(
            prepared, registration, prepared_ref=prepared_ref, registration_ref=registration_ref, case=case
        ),
        "development case/control config was changed",
    )
    root = (REPO / config["output_dir"]).resolve()
    require(root == Path(prepared["root"]) / "development/cases" / case["case_id"], "case root differs")
    if dry:
        return dict(passed=True, dry=True, case=case, new_model_initializations=0, physics_steps=0, requests=0)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="s2a.preview-wam-development-case.v1",
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        state_owner_ref=state_owner_ref,
        state_verification_ref=state_verification_ref,
        config_ref=config_ref,
        case=case,
        actual_model_gpu_uuid=prepared["config"]["gpu_uuid"],
        new_model_initializations=0,
        expert_file_reads=0,
        future_label_reads=0,
        fresh_cases=0,
        http_timeout_cap_s=300,
        preview_controller=True,
        served_reference_variant=case["variant"],
        raw_checkpoint_performance_evidence=case["variant"] == "raw",
        reference_scope="terminal058 is an adapted model system; raw is the paired same-controller baseline",
    )
    try:
        guard = control["cv2_guard_registration"]
        readiness = restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
        result["cv2_readiness_ref"] = publish(root / "cv2_readiness.json", readiness)
        # Imports after the explicit, probe-bound preload cannot re-mutate LD.
        from collect_wam_native_training import endpoint, render_anchor, save_arrays

        from s2a.coupled_preview_wam_simulation import run_preview_wam_case
        from s2a.native_training import pack_gripper_caches, synchronous_labels

        strict_native_environment()
        fixture = json.loads(Path(config["fixture"]).read_text())
        records = [record for record in fixture["records"] if record["name"] == case["record_name"]]
        require(
            len(records) == 1 and records[0]["task_index"] == case["task_index"],
            "fixed initial fixture identity differs",
        )
        parameters = read(config["response_parameters_ref"])
        template = json.loads(Path(config["policy_template"]).read_text())
        h = np.asarray(json.loads(Path(template["tool_contract"]).read_text())["source_from_tcp"])
        samples, images = [], []

        def observe(env, index):
            sample, receipt = endpoint(env, index)
            samples.append(sample)
            with (root / "endpoints.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt, allow_nan=False) + "\n")
            if index in (0, 16, 56):
                images.append(render_anchor(env, index, sample, receipt, root))

        directory = (
            root / "simulation" / f"{CHECKPOINT}_position_hold" / f"{case['record_name']}_s{case['sampling_seed']}"
        )
        with (root / "http.jsonl").open("x") as log, bounded_http(log, case, directory):
            physical = run_preview_wam_case(
                config,
                root,
                records[0],
                source="wam",
                variant_name="position_hold",
                sampling_seed=case["sampling_seed"],
                checkpoint=CHECKPOINT,
                config_path=config_ref["path"],
                oracle_reference=None,
                endpoint_observer=observe,
                response_parameters=parameters,
            )
        strict_native_environment()
        result["physical_result_ref"] = reference(directory / "result.json")
        result["physical_case_dir"] = str(directory)
        result["http_ref"] = reference(root / "http.jsonl")
        calls = [json.loads(line) for line in (root / "http.jsonl").read_text().splitlines()]
        require(
            len(calls) == 76
            and calls[0]["method"] == "GET"
            and sum(c["method"] == "POST" for c in calls) == 75
            and all(c["completed"] and c["status_code"] == 200 and c["timeout_s"] <= 300 for c in calls)
            and all(c["timeout_s"] == 300 for c in calls if c["method"] == "POST"),
            "exact HTTP75 request/timeout budget failed",
        )
        require(
            physical["complete"] is True
            and physical["error"] is None
            and physical["steps"] == 600
            and physical["chunks"] == 75
            and physical["sampling_seed"] == case["sampling_seed"],
            "full600/75 physics execution failed",
        )
        require(len(samples) == 601, "Complete601 measured endpoints required")
        arrays = {key: np.stack([s[key] for s in samples]) for key in samples[0] if key != "gripper_current_action"}
        arrays.update(pack_gripper_caches([s["gripper_current_action"] for s in samples]))
        arrays.update(synchronous_labels(arrays["source_pose"], arrays["opening"], h, arrays["sim_time"]))
        result["arrays_ref"] = save_arrays(root / "actual_trajectory.npz", arrays)
        result["endpoint_receipts_ref"] = reference(root / "endpoints.jsonl")
        result["images"] = images
        result["label_source"] = "measured physical endpoints; not reference targets"
        result.update(
            complete=True,
            passed=True,
            physics_steps=600,
            actual_requests=75,
            final_environment=strict_native_environment(),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(root / "worker_result.json", result)
    return reference(root / "worker_result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration", "state-owner", "state-verification", "case-config"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--case-index", type=int, choices=range(12), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    refs = [
        dict(path=str(getattr(args, key).resolve()), sha256=getattr(args, key + "_sha256"))
        for key in ("prepared", "execution_registration", "state_owner", "state_verification", "case_config")
    ]
    print(json.dumps(execute(*refs, case_index=args.case_index, dry=args.dry)))


if __name__ == "__main__":
    main()
