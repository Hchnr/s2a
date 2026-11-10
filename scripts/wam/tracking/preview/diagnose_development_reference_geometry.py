"""Diagnose all twelve independently reviewed v4 development cases on CPU."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

from s2a.geometry import inverse
from s2a.infrastructure.artifacts import REPO
from s2a.research.wam.tracking.preview.development_reference_geometry import (
    StaticReferenceProbe,
    first_closure,
    migrate_xml,
)
from s2a.research.wam.training.clean.clean_training_runtime import publish, read, reference, require

OUTPUT = REPO / "outputs/multicase_development_reference_geometry_20261110"
BASE = REPO / "outputs/tracking_preview_wam_development_v4_20261027/development/cases"
REVIEW = {
    "path": str(REPO / "outputs/tracking_wam_prefetch_review_execution_20261028/verified.json"),
    "sha256": "c5e74dd75394dd8ec46b328a8cd1f55f20aa3893e7d08f0becb67bd23eb6cccb",
}
SNAPSHOT = {
    "path": str(REPO / "outputs/prefix_trust_asset_snapshot_20261029/manifest.json"),
    "sha256": "c9eaaca963a82f504ed38e67df847fc54c144d030f660be52f6439db0b7b3566",
}
QA = REPO / "outputs/qa/multicase_development_reference_geometry_v2_20261110/result.json"
PLAN = REPO / "plans/v0.0.1/tracking/joint_control/multicase_development_reference_geometry_20261110.md"
LIBRARY = REPO / "src/s2a/research/wam/tracking/preview/development_reference_geometry.py"
TEST = REPO / "tests/research/wam/tracking/preview/test_development_reference_geometry.py"
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LP_NUM_THREADS")
TARGETS = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}


def safe_environment():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("LD_LIBRARY_PATH") == "", "CPU environment")
    require(all(os.environ.get(key) == "1" for key in THREAD_VARS), "Single-thread numerical workers")
    require(Path.cwd() == REPO, "Commands must run from the repository")


def safe_ref(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(REPO), "Evidence must be lexically inside this repository")
    resolved = path.resolve()
    require(resolved.is_relative_to(REPO), "Evidence may not leave this repository")
    return reference(resolved)


def expected_case(index):
    task = index // 4
    seed = (42, 43)[(index % 4) // 2]
    variant = ("raw", "terminal058")[index % 2]
    return f"task{task}_init0_s{seed}_{variant}", (8, 9, 6)[task]


def asset_paths(snapshot):
    require(len(snapshot["assets"]) == 96 and snapshot["file_reference_count"] == 99, "Fixed asset closure")
    paths = {}
    for asset in snapshot["assets"]:
        declared, ref = asset["declared_path"], asset["snapshot_ref"]
        require(declared not in paths and safe_ref(ref["path"]) == ref, "Snapshot asset identity")
        paths[declared] = ref["path"]
    return paths


def verified_input(path, source_map, refs):
    ref = safe_ref(path)
    require(source_map.get(ref["path"]) == ref["sha256"], "Input differs from independent development review")
    refs[ref["path"]] = ref["sha256"]
    return Path(ref["path"])


def process_case(index, review, paths, probe_factory=StaticReferenceProbe):
    expected, task_id = expected_case(index)
    reviewed = review["cases"][index]
    require(reviewed["case"]["case_id"] == expected and reviewed["case"]["case_index"] == index, "Fixed case order")
    require(reviewed["mechanical"]["steps"] == 600 and reviewed["mechanical"]["chunks"] == 75, "Closed case")
    source_map, refs = review["source_and_artifact_sha256"], {}
    case_dir = BASE / expected
    worker_path = verified_input(case_dir / "worker_result.json", source_map, refs)
    worker = json.loads(worker_path.read_bytes())
    require(worker["complete"] is worker["passed"] is True and worker["physics_steps"] == 600, "Complete worker")
    folder = Path(worker["physical_case_dir"])
    require(folder == Path(reviewed["mechanical"]["case"]) and folder.is_relative_to(case_dir), "Fixed physical case")
    manifest_path = verified_input(folder / "manifest.json", source_map, refs)
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest["record"]["task_id"] == task_id, "Task and target mapping")
    target = TARGETS[task_id]
    steps_path = verified_input(folder / "steps.jsonl", source_map, refs)
    xml_path = verified_input(folder / "model.xml", source_map, refs)
    chunks_path = verified_input(folder / "chunks.json", source_map, refs)
    controller_path = verified_input(folder / "controller.json", source_map, refs)
    tool_path = verified_input(Path(manifest["runtime_config"]["tool_contract"]), source_map, refs)
    arrays_path = verified_input(Path(worker["arrays_ref"]["path"]), source_map, refs)
    require(safe_ref(arrays_path) == worker["arrays_ref"], "Worker array identity")

    rows = [json.loads(line) for line in steps_path.read_text().splitlines()]
    chunks = json.loads(chunks_path.read_bytes())
    require(len(rows) == 600 and len(chunks) == 75, "All logged periods and generated chunks")
    require([r["step"] for r in rows] == list(range(600)), "Contiguous physical steps")
    controller = json.loads(controller_path.read_bytes())
    tool = np.asarray(json.loads(tool_path.read_bytes())["source_from_tcp"], dtype=float)
    site_mapping = np.asarray(controller["source_from_site"], dtype=float)
    require(tool.shape == site_mapping.shape == (4, 4), "Fixed tool and controller transforms")
    print(json.dumps({"case": expected, "stage": "model_initialization_requested"}), flush=True)
    model = mujoco.MjModel.from_xml_string(migrate_xml(xml_path.read_bytes(), paths).decode("utf-8"))
    print(json.dumps({"case": expected, "stage": "model_initialized", "models": 1}), flush=True)
    probe = probe_factory(model, target)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
    require(min(joints) >= 0 and all(model.jnt_limited[j] for j in joints), "Seven limited original robot joints")
    qaddrs = np.asarray([model.jnt_qposadr[j] for j in joints])
    limits = np.asarray([model.jnt_range[j] for j in joints])

    with np.load(arrays_path, allow_pickle=False) as arrays:
        states = arrays["sim_states"]
        require(len(states) == 601 and arrays["endpoint_step"].tolist() == list(range(601)), "All 601 endpoints")
        np.testing.assert_array_equal(arrays["sim_time"], states[:, 0])
        q = states[:, 1 + qaddrs]
        logged_q = np.asarray([r["coupled_dynamics"]["robot_joint_position"] for r in rows])
        np.testing.assert_allclose(logged_q, q[:-1], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(np.asarray([r["success"] for r in rows]), arrays["native_success"][1:])
        np.testing.assert_allclose([r["opening_after"] for r in rows], arrays["opening"][1:], atol=1e-12, rtol=0)
        margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
        endpoint_flags = []
        for endpoint, state in enumerate(states):
            probe.restore(state)
            endpoint_flags.append(probe.contacts())
            if (endpoint + 1) % 100 == 0 or endpoint == 600:
                print(
                    json.dumps(
                        {"case": expected, "stage": "endpoints", "completed": endpoint + 1, "queries": probe.queries}
                    ),
                    flush=True,
                )
        queries = []
        for query, chunk in enumerate(chunks):
            step = 8 * query
            require(chunk["step"] == step, "Original eight-step query timing")
            openings = np.asarray(chunk["openings"], dtype=float)
            targets = np.asarray(chunk["effective_targets_tcp"], dtype=float)
            require(targets.shape == (40, 4, 4) and np.all(np.isfinite(targets)), "Finite 40-point TCP reference")
            first = first_closure(openings)
            probe.restore(states[step])
            original = probe.contacts()
            require(original == endpoint_flags[step], "Endpoint and query restore contacts match")
            native = probe.site_transform()
            probe.place(native)
            require(probe.contacts() == original, "Identity relocation preserves target contacts")
            variants = []
            if first is not None:
                desired_site = targets[first] @ inverse(tool) @ site_mapping
                for name, opening in (("predicted", float(openings[first])), ("zero", 0.0)):
                    probe.restore(states[step])
                    probe.set_opening(opening)
                    probe.place(desired_site)
                    variants.append({"name": name, "opening": opening, "pad_contacts": probe.contacts()})
            require(probe.data.time == states[step, 0], "Static query never advances time")
            queries.append({"query": query, "step": step, "first_closure_index": first, "variants": variants})
            if (query + 1) % 10 == 0 or query == 74:
                print(
                    json.dumps(
                        {"case": expected, "stage": "reference", "completed": query + 1, "queries": probe.queries}
                    ),
                    flush=True,
                )

    for path, digest in refs.items():
        require(safe_ref(path)["sha256"] == digest, "Consumed input changed during case")
    errors = np.asarray([r["tcp_error"] for r in rows])
    objects = [r["objects"][target] for r in rows]
    logged_grasp = [bool(o["grasp_contact"]) for o in objects]
    after_both = [all(f) for f in endpoint_flags[1:]]
    outside = margin < 0
    with_close = [q for q in queries if q["first_closure_index"] is not None]
    require(probe.queries == 601 + 2 * 75 + 6 * len(with_close) and probe.queries <= 1201, "Actual query accounting")
    outcome = reviewed["outcome"]
    require(sum(bool(r["success"]) for r in rows) == outcome["successful_steps"], "Reviewed success labels")
    return {
        "case": reviewed["case"],
        "target": target,
        "input_sha256": refs,
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        "rotation_rmse_rad": float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        "task_success_steps": outcome["successful_steps"],
        "stable": outcome["stable"],
        "logged_grasp_steps": sum(logged_grasp),
        "logged_target_max_height_gain_m": float(max(o["height_gain_m"] for o in objects)),
        "opening_target_range": [
            float(min(r["opening_target"] for r in rows)),
            float(max(r["opening_target"] for r in rows)),
        ],
        "opening_actual_range": [
            float(min(r["opening_after"] for r in rows)),
            float(max(r["opening_after"] for r in rows)),
        ],
        "endpoint_min_joint_margin_rad": float(margin.min()),
        "endpoint_outside_any_joint": int(np.sum(np.any(outside, axis=1))),
        "endpoint_outside_by_joint": np.sum(outside, axis=0).tolist(),
        "endpoint_pad_contacts": endpoint_flags,
        "endpoint_any_pad_contact_count": sum(any(f) for f in endpoint_flags),
        "endpoint_both_pad_contact_count": sum(all(f) for f in endpoint_flags),
        "logged_grasp_vs_endpoint_both_mismatches": sum(a != b for a, b in zip(logged_grasp, after_both, strict=True)),
        "queries_with_closure": len(with_close),
        "queries_closure_in_executed_prefix": sum(q["first_closure_index"] < 8 for q in with_close),
        "static_predicted_both_pads": sum(all(q["variants"][0]["pad_contacts"]) for q in with_close),
        "static_zero_both_pads": sum(all(q["variants"][1]["pad_contacts"]) for q in with_close),
        "queries": queries,
        "mj_forward_queries": probe.queries,
        "max_object_rotation_change": getattr(probe, "max_object_rotation_change", None),
        "max_qpos_change_during_placement": getattr(probe, "max_qpos_change_during_placement", None),
        "physics_steps": 0,
        "world_model_calls": 0,
        "fresh_cases": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(12))
    args = parser.parse_args()
    safe_environment()
    review, snapshot = read(REVIEW), read(SNAPSHOT)
    require(review["complete"] is review["passed"] is True and len(review["cases"]) == 12, "Full reviewed matrix")
    if args.case_index is not None:
        registration = read(safe_ref(OUTPUT / "registration.json"))
        require(registration["review_ref"] == REVIEW and registration["snapshot_ref"] == SNAPSHOT, "Owner registration")
        paths = {a["declared_path"]: a["snapshot_ref"]["path"] for a in snapshot["assets"]}
        result = process_case(args.case_index, review, paths)
        print(publish(OUTPUT / f"case_{args.case_index:02d}/result.json", result), flush=True)
        return

    require(not OUTPUT.exists(), "One new output directory; no overwrite or retry")
    qa_ref = safe_ref(QA)
    qa = read(qa_ref)
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "QA closed",
    )
    assets = asset_paths(snapshot)
    sources = [safe_ref(p) for p in (Path(__file__), LIBRARY, TEST, PLAN)]
    qa_sources = read(qa["source_before_ref"])["sources"]
    require(
        all(qa_sources.get(str(Path(ref["path"]).relative_to(REPO))) == ref["sha256"] for ref in sources),
        "QA and production must use the same implementation, plan and tests",
    )
    registration = publish(
        OUTPUT / "registration.json",
        {
            "review_ref": REVIEW,
            "snapshot_ref": SNAPSHOT,
            "qa_ref": qa_ref,
            "source_refs": sources,
            "cases": 12,
            "max_queries_per_case": 1201,
            "max_total_queries": 14412,
            "max_models": 12,
            "physics_steps": 0,
            "world_model_calls": 0,
            "fresh_cases": 0,
            "automatic_retry": False,
        },
    )

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Owner received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    workers = []
    launch_error = None
    try:
        for index in range(12):
            directory = OUTPUT / f"case_{index:02d}"
            directory.mkdir()
            log = (directory / "worker.log").open("x")
            command = [
                sys.executable,
                "-m",
                "scripts.wam.tracking.preview.diagnose_development_reference_geometry",
                "--case-index",
                str(index),
            ]
            try:
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            workers.append({"index": index, "process": process, "log": log, "command": command})
    except BaseException:
        launch_error = traceback.format_exc()
        for worker in workers:
            if worker["process"].poll() is None:
                try:
                    worker["process"].terminate()
                    worker["signaled_during_launch"] = True
                except ProcessLookupError:
                    pass
    deadline = time.monotonic() + 1200
    lifecycle = []
    for worker in workers:
        process = worker["process"]
        signaled = worker.get("signaled_during_launch", False)
        try:
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            signaled = True
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        except BaseException:
            launch_error = traceback.format_exc()
            for other in workers:
                if other["process"].poll() is None:
                    try:
                        other["process"].terminate()
                        other["signaled_during_launch"] = True
                    except ProcessLookupError:
                        pass
            signaled = worker.get("signaled_during_launch", False)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        worker["log"].close()
        lifecycle.append(
            {
                "case_index": worker["index"],
                "pid": process.pid,
                "returncode": code,
                "actual_wait": True,
                "reaped": True,
                "signal_sent": signaled,
                "log_ref": safe_ref(OUTPUT / f"case_{worker['index']:02d}/worker.log"),
            }
        )
    workers_ref = publish(OUTPUT / "workers.json", {"workers": lifecycle, "launch_error": launch_error})
    if (
        launch_error
        or len(lifecycle) != 12
        or not all(w["returncode"] == 0 and not w["signal_sent"] for w in lifecycle)
    ):
        publish(
            OUTPUT / "result.json",
            {
                "complete": False,
                "passed": False,
                "registration_ref": registration,
                "workers_ref": workers_ref,
                "automatic_retry": False,
            },
        )
        raise ValueError("All 12 workers must succeed; partial results and lifecycle are retained")
    require(asset_paths(read(SNAPSHOT)) == assets, "Snapshot assets unchanged")
    require(all(safe_ref(ref["path"]) == ref for ref in sources), "Frozen implementation unchanged")
    results = [read(safe_ref(OUTPUT / f"case_{i:02d}/result.json")) for i in range(12)]
    count = sum(case["mj_forward_queries"] for case in results)
    require(count <= 14412, "Total query budget")
    print(
        publish(
            OUTPUT / "result.json",
            {
                "complete": True,
                "passed": True,
                "registration_ref": registration,
                "workers_ref": workers_ref,
                "case_refs": [safe_ref(OUTPUT / f"case_{i:02d}/result.json") for i in range(12)],
                "mj_forward_queries": count,
                "model_initializations": 12,
                "physics_steps": 0,
                "world_model_calls": 0,
                "fresh_cases": 0,
                "sources_unchanged": True,
                "scope": "Saved-development endpoints and frozen-object static geometry; no motion or task qualification",
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
