"""Isolated CPU wrapper around the byte-frozen position executor; no new control."""

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import requests

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.native_environment import restore_cv2_environment, strict_native_environment
from s2a.terminal_development_control import CHECKPOINT, make_case_config
from s2a.terminal_development_execution import PORT, admit_development
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
    prepared, _, _, _, control = admit_development(
        prepared_ref,
        registration_ref,
        state_owner_ref,
        state_verification_ref,
        required_sources=[
            __file__,
            REPO / "scripts/run_wam_terminal_development.py",
            REPO / "src/s2a/native_environment.py",
        ],
    )
    config = read(config_ref)
    require(config == make_case_config(prepared, case), "development case/control config was changed")
    root = (REPO / config["output_dir"]).resolve()
    require(root == Path(prepared["root"]) / "development/cases" / case["case_id"], "case root differs")
    if dry:
        return dict(passed=True, dry=True, case=case, new_model_initializations=0, physics_steps=0, requests=0)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version="s2a.terminal-development-case.v1",
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        state_owner_ref=state_owner_ref,
        state_verification_ref=state_verification_ref,
        config_ref=config_ref,
        case=case,
        actual_model_gpu_uuid=prepared["config"]["gpu_uuid"],
        legacy_executor_gpu_label_only=4,
        new_model_initializations=0,
        expert_file_reads=0,
        future_label_reads=0,
        fresh_cases=0,
        http_timeout_cap_s=300,
        frozen_executor_modified=False,
        served_reference_variant=case["variant"],
        raw_checkpoint_performance_evidence=case["variant"] == "raw",
        legacy_executor_raw_reference_flag_scope="CPU alpha1 preserves the served path; terminal058 is an adapted model system, not raw checkpoint output",
    )
    try:
        guard = control["cv2_guard_registration"]
        readiness = restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
        result["cv2_readiness_ref"] = publish(root / "cv2_readiness.json", readiness)
        # Imports after the explicit, probe-bound preload cannot re-mutate LD.
        from s2a.stability_position_simulation import execution_inputs, run_position_case

        strict_native_environment()
        fixture = json.loads(Path(config["fixture"]).read_text())
        records = [record for record in fixture["records"] if record["name"] == case["record_name"]]
        require(
            len(records) == 1 and records[0]["task_index"] == case["task_index"],
            "fixed initial fixture identity differs",
        )
        inputs = execution_inputs(config, config_ref["path"])
        result["execution_snapshot_ref"] = publish(root / "execution_snapshot.json", dict(source_sha256=inputs))
        directory = (
            root / "simulation" / f"{CHECKPOINT}_position_hold" / f"{case['record_name']}_s{case['sampling_seed']}"
        )
        with (root / "http.jsonl").open("x") as log, bounded_http(log, case, directory):
            physical = run_position_case(
                config,
                root,
                records[0],
                source="wam",
                variant_name="position_hold",
                sampling_seed=case["sampling_seed"],
                checkpoint=CHECKPOINT,
                config_path=config_ref["path"],
                frozen_inputs=inputs,
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
