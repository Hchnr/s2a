"""Read saved finite differences only; never call a solver or a forward model."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from s2a.saved_trajectory_differences import diagnose_saved_differences

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "plans/v0.0.1/saved_trajectory_differences_20261028.md"
OUTPUT = REPO / "outputs/tracking_v5_saved_differences_20261028"
SOURCES = (
    Path(__file__),
    REPO / "src/s2a/saved_trajectory_differences.py",
    REPO / "tests/test_saved_trajectory_differences.py",
    PLAN,
)
CPU = dict(
    CUDA_VISIBLE_DEVICES="",
    LD_LIBRARY_PATH="",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    LP_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
CASES = {
    "prefix": (
        "prefix_preserving_trajectory_control",
        "89da5e5287ef592edefa6eaea90b801c603b6abfaaf527a56e7dcb8f4258bebb",
        12,
        9,
    ),
    "early": (
        "paired_reference_phase_early",
        "9f751f169790027391cfc9317e71a4687245b9b0688e01d8c0c83967b84f5101",
        13,
        10,
    ),
    "late": (
        "paired_reference_phase_late",
        "8c817b99c99405ce2b5c29e8e0221279b461190b4a929a742ca9500c01d8be1e",
        13,
        10,
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), "All diagnostic inputs and sources stay in the current repository")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n"
    with Path(path).open("x") as stream:
        stream.write(payload)


class Evidence:
    def __init__(self):
        self.refs = {}

    def bind(self, ref):
        require(reference(ref["path"]) == ref, "Evidence SHA mismatch")
        require(self.refs.get(ref["path"], ref) == ref, "Previously observed evidence changed")
        self.refs[ref["path"]] = ref
        return ref

    def read(self, ref):
        self.bind(ref)
        payload = Path(ref["path"]).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == ref["sha256"], "Evidence changed during JSON read")
        return json.loads(payload)

    def finish(self):
        for ref in list(self.refs.values()):
            self.bind(ref)


def checker_snapshot(evidence, name, producer_ref):
    stem, _, horizon, _ = CASES[name]
    root = REPO / f"outputs/tracking_v5_{stem}_check_execution_20261028"
    check_root = REPO / f"outputs/tracking_v5_{stem}_check_20261028"
    owner_ref = reference(root / "registration.json")
    owner_reg = evidence.read(owner_ref)
    launch_ref = reference(root / "capture/launch.json")
    launch = evidence.read(launch_ref)
    reg_ref = reference(check_root / "registration.json")
    reg = evidence.read(reg_ref)
    script = REPO / (
        "scripts/check_tracking_v5_prefix_preserving_trajectory_control.py"
        if name == "prefix"
        else "scripts/check_tracking_v5_paired_reference_phase.py"
    )
    command = [str(REPO / "outputs/libero_env311/bin/python"), str(script)]
    if name != "prefix":
        command += ["--case", name]
    require(
        owner_reg["stage"] == "check"
        and reg["producer_ref"] == producer_ref
        and launch["command"] == command
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and launch["automatic_retry"] is False,
        "Checker identity and exact producer binding",
    )
    evidence.bind(dict(path=str(script), sha256=owner_reg["source_sha256"][str(script)]))
    child = launch["child"]
    snapshot = dict(
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        producer_ref=producer_ref,
        owner_registration_ref=owner_ref,
        checker_registration_ref=reg_ref,
        launch_ref=launch_ref,
        child=child,
        complete=False,
        passed=False,
    )
    if (root / "result.json").exists():
        result_ref = reference(root / "result.json")
        owner = evidence.read(result_ref)
        require(owner["registration_ref"] == owner_ref, "Checker terminal owner registration")
        snapshot.update(status="terminal", full_check_status="failed", owner_result_ref=result_ref)
        if owner["complete"] is owner["passed"] is True:
            require(owner["owned_command"]["launch"] == launch_ref, "Checker terminal launch identity")
            exited = evidence.read(owner["owned_command"]["exit"])
            release = evidence.read(owner["owned_command"]["release"])
            require(
                exited["actual_wait"] is True
                and exited["returncode"] == 0
                and release["reaped"] is True
                and release["returncode"] == 0
                and release["signal_sent"] is False
                and release["original_error"] is None
                and release["pid"] == child["pid"]
                and release["start_ticks"] == child["start_ticks"],
                "Checker actual wait/reap0 without signal",
            )
            verified = evidence.read(owner["result_ref"])
            require(
                verified["registration_ref"] == reg_ref
                and verified["complete"] is verified["passed"] is True
                and verified["trace_count"] == 4000
                and verified["numerical_inner_steps"] == 4000 * horizon * 25
                and verified["mj_forward_queries"] == 4000 * horizon * 26 + 1,
                "Full independent checker budget and result",
            )
            snapshot.update(
                complete=True,
                passed=True,
                full_check_status="passed",
                verified_ref=owner["result_ref"],
                exit=exited,
                release=release,
            )
        return snapshot
    proc = Path("/proc") / str(child["pid"])
    try:
        stat = (proc / "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        require(
            int(fields[19]) == child["start_ticks"] and int(fields[1]) == child["ppid"],
            "Checker live process birth and parent identities",
        )
        if fields[0] == "Z":
            return dict(snapshot, status="awaiting_owner_result", process_state=fields[0])
        live_command = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        require(live_command == command, "Checker live command identity")
    except FileNotFoundError:
        return dict(snapshot, status="awaiting_owner_result")
    return dict(snapshot, status="running", process_state=fields[0])


def run(qa_ref):
    require({key: os.environ.get(key) for key in CPU} == CPU, "Explicit CPU environment and single numerical threads")
    require(not OUTPUT.exists(), "New output directory required; no overwrite or automatic retry")
    evidence = Evidence()
    source_before = [evidence.bind(reference(path)) for path in SOURCES]
    qa = evidence.read(qa_ref)
    require(
        qa["complete"] is qa["passed"] is qa["actual_wait"] is True
        and qa["returncode"] == 0
        and qa["source_refs_before"] == qa["source_refs_after"] == source_before
        and qa["cpu_environment"] == CPU,
        "Completed native QA for this exact diagnostic revision",
    )
    evidence.bind(qa["log_ref"])
    require(
        qa["command"][:5]
        == [
            str(REPO / "outputs/libero_env311/bin/python"),
            "-m",
            "pytest",
            "-q",
            "tests/test_saved_trajectory_differences.py",
        ],
        "Formal native synthetic QA command",
    )
    producers, registrations, producer_refs = {}, {}, {}
    for name, (stem, sha, horizon, _) in CASES.items():
        ref = dict(path=str(REPO / f"outputs/tracking_v5_{stem}_20261028/verified.json"), sha256=sha)
        producer_refs[name] = ref
        data = evidence.read(ref)
        reg = evidence.read(data["registration_ref"])
        require(
            data["complete"] is data["passed"] is True
            and data["control_qualified"] is False
            and reg["endpoint"] == 368
            and reg["horizon"] == horizon
            and reg["seed"] == 42
            and data["answer"]["response_evaluations"] == 4000
            and len(data["answer"]["evaluated_candidates"]) == 4000
            and data["answer"]["finite_difference_step"] == 1e-4
            and data["answer"]["optimization_safety_buffer"] == 1e-8,
            "Exact registered saved experiment and original finite-difference contract",
        )
        for source in reg["source_refs"]:
            evidence.bind(source)
        producers[name], registrations[name] = data, reg
    common = (
        "current_q",
        "current_velocity",
        "previous_command",
        "previous_site",
        "previous_physical_delta",
        "latched_rotation_goal",
        "initial_joint",
        "kp",
        "kd",
        "joint_limits",
        "torque_limits",
        "site_from_tcp",
        "past_inner_error_rad",
        "terminal_normals",
        "fixed_gripper_state",
    )
    for name in ("early", "late"):
        data, reg = producers[name], registrations[name]
        require(reg["prior_prefix_ref"] == producer_refs["prefix"], "Same prior prefix result")
        require(reg["chunks_ref"] == registrations["prefix"]["chunks_ref"], "Same already generated reference chunk")
        require(
            all(np.array_equal(data[key], producers["prefix"][key]) for key in common), "Same causal initial inputs"
        )
        require(np.array_equal(data["source_targets_tcp"][:12], producers["prefix"]["desired_tcp"]), "Original targets")
        contract = data["case_contract"]
        require(contract["original_node_endpoint_indices"][9] == 10, "Same original cursor9 at retimed index10")
        require(contract["actual_consumed_reference_count"] == 0, "Diagnostic has not consumed reference")
    evidence.bind(registrations["prefix"]["chunks_ref"])
    checks_before = {name: checker_snapshot(evidence, name, producer_refs[name]) for name in CASES}
    require(checks_before["prefix"]["passed"] is True, "H12 prefix full independent audit is closed")
    evidence.finish()
    inputs_before = list(evidence.refs.values())
    OUTPUT.mkdir()
    registration = dict(
        producer_refs=producer_refs,
        source_refs=source_before,
        qa_ref=qa_ref,
        input_refs=inputs_before,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        cpu_environment=CPU,
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        seed=42,
        random_sampling=False,
        automatic_retry=False,
        plan_ref=reference(PLAN),
        checker_before=checks_before,
        target_original_cursor=9,
        target_indices={name: values[3] for name, values in CASES.items()},
    )
    write(OUTPUT / "registration.json", registration)
    result = dict(
        complete=False,
        passed=False,
        registration_ref=reference(OUTPUT / "registration.json"),
        control_qualified=False,
        retiming_qualified=False,
        task_qualified=False,
        new_forward_responses=0,
        new_dynamics_queries=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        checker_before=checks_before,
        cases={},
    )
    try:
        for name, data in producers.items():
            result["cases"][name] = diagnose_saved_differences(
                data["answer"], np.asarray(data["previous_command"][:6]), target_index=CASES[name][3]
            )
        result["checker_after"] = {name: checker_snapshot(evidence, name, producer_refs[name]) for name in CASES}
        evidence.finish()
        source_after = [reference(path) for path in SOURCES]
        require(source_after == source_before, "Diagnostic sources unchanged")
        result.update(
            complete=True,
            passed=True,
            source_refs_before=source_before,
            source_refs_after=source_after,
            input_refs_before=inputs_before,
            input_refs_after=list(evidence.refs.values()),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        write(OUTPUT / "result.json", result)
        raise
    write(OUTPUT / "result.json", result)
    print(json.dumps(dict(result_ref=reference(OUTPUT / "result.json"), complete=True, passed=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-ref", type=Path, required=True)
    parser.add_argument("--qa-sha256", required=True)
    args = parser.parse_args()
    run(dict(path=str(args.qa_ref.resolve()), sha256=args.qa_sha256))


if __name__ == "__main__":
    main()
