"""Recompute paired diagnostic state screens; incomplete inputs cannot select a model.

Only saved JSON/NumPy evidence is read. No service, model or physics is started.
Metrics pool valid endpoints, never padded future frames or per-query RMSEs.
"""

import argparse
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from s2a.checkpoint_audit import extrapolate_history, state_metrics
from s2a.geometry import integrate_local, pose9_to_transform
from s2a.prediction_audit import audit_action_response

LABELS = ("8800_g15", "8800_g10", "15400_g15", "15400_g10")
PREFIXES = (8, 16, 40)
EXECUTION_FILES = (
    "scripts/run_wam_state_matrix.py",
    "src/s2a/checkpoint_audit.py",
    "src/s2a/geometry.py",
    "src/s2a/prediction_audit.py",
    "src/s2a/artifacts.py",
)


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        key = str(path.resolve())
        if key in self.hashes and self.hashes[key] != digest:
            raise ValueError(f"evidence changed during collection: {path}")
        self.hashes[key] = digest
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def check(self, path, expected):
        self.read(path)
        if self.hashes[str(Path(path).resolve())] != expected:
            raise ValueError(f"SHA256 mismatch: {path}")

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.check(path, digest)


def equal_metrics(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("saved metric fields differ from recomputation")
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=0, err_msg=key)


def expected_request(config, sample, seed, name):
    return dict(
        name=name,
        model_mode="wam",
        prompt=sample["task"],
        vision_path=str(Path(sample["image_path"]).resolve()),
        domain_name="franka_panda__parallel_gripper",
        image_size=960,
        action_chunk_size=40,
        fps=config["control_fps"],
        num_frames=41,
        seed=seed,
        num_steps=config["sampling_steps"],
        guidance=config["guidance"],
        shift=config["shift"],
        extra=dict(
            action128=dict(
                action_domain=config["action_domain"],
                action_dim_mask=[True] * 10 + [False] * 118,
                action_fps=config["control_fps"],
                task_title=sample["task"],
                selected_roles=["auxiliary_0", "left_wrist"],
                camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                return_video=False,
            )
        ),
    )


def verify_cell(label, root, evidence):
    config_path = Path("configs") / f"wam_stability_screen_{label}_20261021.json"
    config = evidence.json(config_path)
    output = Path(config["output_dir"])
    if output.resolve() != (root / "screening" / label).resolve():
        raise ValueError("screen output root differs from frozen configuration")
    if (
        config["seed"] != 42
        or config["offline_horizons"] != [40]
        or config["offline_prompt_modes"] != ["task_only"]
        or config["sampling_seeds"] != [42, 43]
        or config["control_fps"] != 20
        or len(config["checkpoints"]) != 1
    ):
        raise ValueError("screen protocol differs")
    checkpoint = config["checkpoints"][0]
    sample_path = output / "samples/manifest.json"
    samples = evidence.json(sample_path)
    source_path = samples["source_manifest"]
    evidence.check(source_path, samples["source_manifest_sha256"])
    original = evidence.json(source_path)
    selected = [s for s in original["cases"] if s["split"] == "diagnostic"]
    if samples["cases"] != selected or samples["config"] != config or len(selected) != 24:
        raise ValueError("requires exactly the unchanged original24 diagnostic samples")
    if len({s["name"] for s in selected}) != 24 or len({s["episode"] for s in selected}) != 6:
        raise ValueError("duplicate sample names or unexpected development episode count")
    source_heldout = {s["episode"] for s in original["cases"] if s["split"] == "study_heldout"}
    if {s["episode"] for s in selected} & source_heldout:
        raise ValueError("development and historical heldout episodes overlap")
    expected = {f"{s['name']}_h40_task_only_s{seed}": (s, seed) for s in selected for seed in (42, 43)}
    if len(expected) != 48:
        raise ValueError("screen does not define48 unique requests")
    directory = output / "state_queries" / checkpoint["name"]
    progress = dict(expected=48, verified=0, complete=False, missing=sorted(expected), in_progress=[])
    if not (directory / "manifest.json").exists():
        return config, selected, {}, progress
    run = evidence.json(directory / "manifest.json")
    expected_checkpoint = str(Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model")
    if run["immutable"] != dict(
        config=config,
        samples_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        checkpoint=expected_checkpoint,
    ):
        raise ValueError("state-query immutable manifest differs")
    info = run["server_info"]
    if info["checkpoint_path"] != expected_checkpoint or info["ema_weights"]:
        raise ValueError("state-query server uses unexpected weights")
    if not info["acceptance_intervention"]["active_slot_mask"]:
        raise ValueError("state-query active-slot intervention missing")
    for path in EXECUTION_FILES:
        evidence.check(path, run["code"]["files_sha256"][path])
    orchestration = evidence.json(output / "orchestration/manifest.json")
    evidence.check(config_path, orchestration["config_sha256"])
    evidence.check(sample_path, orchestration["samples_sha256"])
    if orchestration["config"] != config or orchestration["physical_gpu"] != checkpoint["gpu"]:
        raise ValueError("orchestration protocol or GPU mapping differs")
    model_config_path = Path(checkpoint["run"]) / "config.yaml"
    saved = yaml.safe_load(evidence.read(model_config_path))
    template = evidence.json(config["policy_template"])
    domain = template["action_domain"]
    config = dict(config, action_domain=domain)
    spec = saved["model"]["config"]["action128_spec"]
    transform = spec["transforms"][spec["registry"]["domains"].index(domain)]
    complete_path = directory / "complete.json"
    complete = evidence.json(complete_path) if complete_path.exists() else None
    if complete is not None:
        if complete["cases"] != 48 or set(complete["result_files"]) != set(expected):
            raise ValueError("complete marker does not cover exactly48 frozen cases")
        evidence.check(directory / "manifest.json", complete["manifest_sha256"])
    present = {p.parent.name: p for p in directory.glob("*/result.json")}
    if not set(present) <= set(expected):
        raise ValueError("unexpected state-query result outside frozen48 requests")
    verified = {}
    for name, result_path in sorted(present.items()):
        if complete is not None:
            evidence.check(result_path, complete["result_files"][name])
        try:
            # A worker can still be writing a new result before its complete marker.
            result = json.loads(result_path.read_bytes())
        except json.JSONDecodeError:
            if complete is not None:
                raise
            progress["in_progress"].append(name)
            continue
        evidence.check(result_path, hashlib.sha256(result_path.read_bytes()).hexdigest())
        sample, seed = expected[name]
        case_dir = result_path.parent
        request = evidence.json(case_dir / "request.json")
        if request != expected_request(config, sample, seed, name):
            raise ValueError(f"request differs from the frozen sample/sampler: {label}/{name}")
        evidence.check(case_dir / "response.json", result["response_sha256"])
        response = evidence.json(case_dir / "response.json")
        if response["status"] != "success":
            raise ValueError("saved model response is not successful")
        content = response["outputs"][0]["content"]
        metadata = content["action128"]
        if metadata["checkpoint_path"] != expected_checkpoint or metadata["seed"] != seed:
            raise ValueError("response weight/seed identity differs")
        contract = audit_action_response(content, transform, domain=domain, fps=config["control_fps"])
        equal_metrics(contract, result["contract"])
        if contract["denormalization_max_abs"] > 2e-7:
            raise ValueError("model action denormalization differs")
        action = np.asarray(content["action"], dtype=float)
        if action.shape != (40, 128):
            raise ValueError("model action must have40 physical Action128 endpoints")
        for key, value in dict(
            name=name,
            checkpoint=checkpoint["name"],
            sample=sample["name"],
            task=sample["task"],
            episode=sample["episode"],
            split="diagnostic",
            phase=sample["phase"],
            anchor_frame=sample["anchor_frame"],
            horizon=40,
            sampling_seed=seed,
            prompt_mode="task_only",
        ).items():
            if result[key] != value:
                raise ValueError(f"result identity differs: {key}")
        evidence.check(sample["arrays_path"], sample["arrays_sha256"])
        evidence.check(sample["image_path"], sample["image_sha256"])
        with np.load(sample["arrays_path"]) as arrays:
            states = arrays["states"]
        frame = sample["anchor_frame"]
        count = min(40, sample["valid_future_steps"], len(states) - frame - 1)
        if count < 1 or result["valid_future_steps"] != count:
            raise ValueError("valid future coverage differs")
        poses = pose9_to_transform(states[:, :9])
        truth = poses[frame + 1 : frame + 1 + count]
        opening = states[frame + 1 : frame + 1 + count, 9]
        predictions = {
            "wam": integrate_local(poses[frame], action[:count, :9])[1:],
            "hold": np.repeat(poses[frame : frame + 1], count, axis=0),
            "constant_velocity": extrapolate_history(poses[max(0, frame - 4) : frame + 1], count),
        }
        metrics = {
            method: state_metrics(
                truth,
                predicted,
                opening,
                np.clip(action[:count, 9], 0, 1) if method == "wam" else np.repeat(states[frame, 9], count),
            )
            for method, predicted in predictions.items()
        }
        equal_metrics(metrics["wam"], result["metrics"])
        for method in ("hold", "constant_velocity"):
            equal_metrics(metrics[method], result["baselines"][method])
        verified[name] = dict(sample=sample, sampling_seed=seed, metrics=metrics)
    progress.update(verified=len(verified), missing=sorted(set(expected) - set(verified)))
    orchestration_complete = output / "orchestration/complete.json"
    if complete is not None and orchestration_complete.exists():
        done = evidence.json(orchestration_complete)
        if not done["complete"] or done["queries"] != 48:
            raise ValueError("coordinator completion differs")
        evidence.check(complete_path, done["screen_sha256"])
        progress["complete"] = len(verified) == 48
    return config, selected, verified, progress


def aggregate(cases):
    result = {}
    strata = {"all": cases}
    for case in cases:
        sample = case["sample"]
        strata.setdefault("task:" + sample["task"], [])
        strata.setdefault("phase:" + sample["phase"], [])
    for key in list(strata):
        if key.startswith("task:"):
            strata[key] = [c for c in cases if c["sample"]["task"] == key[5:]]
        elif key.startswith("phase:"):
            strata[key] = [c for c in cases if c["sample"]["phase"] == key[6:]]
    for key, selected in strata.items():
        result[key] = {}
        for prefix in PREFIXES:
            methods = {}
            for method in ("wam", "hold", "constant_velocity"):
                if not selected:
                    methods[method] = None
                    continue
                position = np.concatenate([c["metrics"][method]["position_errors_m"][:prefix] for c in selected])
                rotation = np.concatenate([c["metrics"][method]["rotation_errors_rad"][:prefix] for c in selected])
                opening = np.concatenate([c["metrics"][method]["opening_errors"][:prefix] for c in selected])
                methods[method] = dict(
                    queries=len(selected),
                    episodes=len({c["sample"]["episode"] for c in selected}),
                    distinct_anchors=len({(c["sample"]["episode"], c["sample"]["anchor_frame"]) for c in selected}),
                    valid_endpoints=len(position),
                    full_prefix_queries=sum(c["metrics"][method]["endpoints"] >= prefix for c in selected),
                    position_rmse_m=float(np.sqrt(np.mean(position**2))),
                    position_rmse_mm=float(np.sqrt(np.mean(position**2)) * 1000),
                    rotation_rmse_rad=float(np.sqrt(np.mean(rotation**2))),
                    opening_mae=float(np.mean(np.abs(opening))),
                )
            result[key][str(prefix)] = methods
    return result


def recommend(groups):
    keys = ("position_rmse_m", "rotation_rmse_rad", "opening_mae")
    vectors = {
        label: np.array([groups[label]["metrics"]["all"][str(p)]["wam"][k] for p in PREFIXES for k in keys])
        for label in LABELS
    }
    shortlisted = [
        label
        for label in LABELS
        if not any(
            np.all(other <= vectors[label]) and np.any(other < vectors[label])
            for name, other in vectors.items()
            if name != label
        )
    ]
    return dict(
        scope="Diagnostic state quality only; no task success or fresh heldout claim",
        rule="Pareto dominance across position/SO3/opening errors at each prefix8/16/40; all9 metrics minimized",
        next_validation_shortlist=shortlisted,
        unique_recommendation=shortlisted[0] if len(shortlisted) == 1 else None,
        interpretation=(
            "One configuration dominates the remaining screen; validate it independently in closed-loop tasks."
            if len(shortlisted) == 1
            else "Metrics trade off; retain the shortlist for controlled task validation, without an invented mixed-unit score."
        ),
        best_by_prefix={
            str(p): {k: min(LABELS, key=lambda label: groups[label]["metrics"]["all"][str(p)]["wam"][k]) for k in keys}
            for p in PREFIXES
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/wam_stability_20261021"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("run this CPU-only summarizer with empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    destination = args.output or args.root / "screening_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(destination.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists() and json.loads(destination.read_text()).get("complete"):
            raise FileExistsError("complete summary is immutable; use a new --output for independent verification")
        evidence = Evidence()
        groups, snapshots = {}, {}
        for label in LABELS:
            config, selected, cases, progress = verify_cell(label, args.root, evidence)
            snapshots[label] = (selected, cases)
            groups[label] = dict(checkpoint=config["checkpoints"][0], guidance=config["guidance"], progress=progress)
        first = snapshots[LABELS[0]][0]
        if any(selected != first for selected, _ in snapshots.values()):
            raise ValueError("the four cells do not share identical sample identities and metadata")
        common = sorted(set.intersection(*(set(cases) for _, cases in snapshots.values())))
        complete = all(g["progress"]["complete"] for g in groups.values())
        if complete and len(common) != 48:
            raise ValueError("the four completed cells do not share exactly48 cases")
        for label, (_, cases) in snapshots.items():
            groups[label]["metrics"] = aggregate([cases[name] for name in common])
        evidence.read(__file__)
        evidence.recheck()
        summary = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            complete=complete,
            status="complete" if complete else "partial",
            expected_queries_per_cell=48,
            common_verified_queries=len(common),
            common_case_names=common,
            aggregation="Valid-endpoint weighted; all4 cells use the same verified request intersection, including partial snapshots",
            groups=groups,
            recommendation=recommend(groups) if complete else None,
            evidence_boundaries=[
                "Only24 diagnostic phase samples from6 historical development episodes; not fresh heldout or proven pretraining unseen.",
                "Phase proxies/fallbacks and repeated anchors/seeds are not independent episodes or verified grasp events.",
                "Prefixes8/16/40 use only available future states, excluding padded labels; full-prefix query counts are explicit.",
                "Position/SO3/opening describe demonstration state error, not unique correct trajectories or closed-loop task success.",
                "Partial cells cannot select a winner. Complete recommendations concern model-side state quality only.",
            ],
            input_sha256=evidence.hashes,
        )
        temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
        with temporary.open("x") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
        print(json.dumps(dict(output=str(destination), complete=complete, common_queries=len(common))))


if __name__ == "__main__":
    main()
