"""Verify guidance1.0 stage screens and compare only four complete48-query cells.

Reads saved evidence only. It does not start inference or physical simulation.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from summarize_wam_stability_screen import (
    EXECUTION_FILES,
    Evidence,
    aggregate,
    equal_metrics,
    expected_request,
    verify_cell,
)

from s2a.artifacts import sha256
from s2a.checkpoint_audit import extrapolate_history, state_metrics
from s2a.geometry import integrate_local, pose9_to_transform
from s2a.prediction_audit import audit_action_response

LABELS = ("8400_g10", "8600_g10", "8800_g10", "15400_g10")


def verify_earlier(label, root, evidence):
    config_path = Path(f"configs/wam_stability_screen_{label}_20261021.json")
    config = evidence.json(config_path)
    output = root / "earlier_screening" / label
    if Path(config["output_dir"]).resolve() != output.resolve():
        raise ValueError("earlier-stage output differs from its explicit configuration")
    required = dict(
        seed=42,
        offline_horizons=[40],
        offline_prompt_modes=["task_only"],
        sampling_seeds=[42, 43],
        sampling_steps=35,
        guidance=1.0,
        shift=5.0,
        control_fps=20,
    )
    if any(config[k] != value for k, value in required.items()) or len(config["checkpoints"]) != 1:
        raise ValueError("earlier-stage request protocol differs")
    checkpoint = config["checkpoints"][0]
    stage = int(label.split("_")[0])
    if (
        checkpoint["name"] != f"{stage}_masked"
        or checkpoint["iteration"] != f"iter_{stage:09d}"
        or checkpoint["gpu"] != {8400: 4, 8600: 6}[stage]
        or checkpoint["port"] != {8400: 9144, 8600: 9146}[stage]
    ):
        raise ValueError("checkpoint identity or physical GPU allocation differs")
    sample_path = output / "samples/manifest.json"
    samples = evidence.json(sample_path)
    evidence.check(samples["source_manifest"], samples["source_manifest_sha256"])
    original = evidence.json(samples["source_manifest"])
    selected = [s for s in original["cases"] if s["split"] == "diagnostic"]
    if samples["config"] != config or samples["cases"] != selected or len(selected) != 24:
        raise ValueError("diagnostic24 sample identity or configuration changed")
    expected = {f"{s['name']}_h40_task_only_s{seed}": (s, seed) for s in selected for seed in (42, 43)}
    if len(expected) != 48 or len({s["episode"] for s in selected}) != 6:
        raise ValueError("unexpected request or development-episode count")
    directory = output / "state_queries" / checkpoint["name"]
    complete_path = directory / "complete.json"
    done_path = output / "orchestration/complete.json"
    progress = dict(
        expected=48,
        result_files_seen_unverified=len(list(directory.glob("*/result.json"))),
        verified=0,
        complete=False,
    )
    if not complete_path.exists() or not done_path.exists():
        return config, selected, {}, progress
    complete, done = evidence.json(complete_path), evidence.json(done_path)
    if complete["cases"] != 48 or set(complete["result_files"]) != set(expected):
        raise ValueError("completion does not cover all48 frozen requests")
    if not done["complete"] or done["queries"] != 48:
        raise ValueError("coordinator completion differs")
    evidence.check(complete_path, done["screen_sha256"])
    manifest_path = directory / "manifest.json"
    evidence.check(manifest_path, complete["manifest_sha256"])
    run = evidence.json(manifest_path)
    expected_checkpoint = str(Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model")
    if run["immutable"] != dict(config=config, samples_sha256=sha256(sample_path), checkpoint=expected_checkpoint):
        raise ValueError("state-query configuration/sample/weight manifest differs")
    info = run["server_info"]
    if (
        info["checkpoint_path"] != expected_checkpoint
        or info["ema_weights"]
        or not info["acceptance_intervention"]["active_slot_mask"]
    ):
        raise ValueError("server weight or active-slot intervention differs")
    for path in EXECUTION_FILES:
        evidence.check(path, run["code"]["files_sha256"][path])
    orchestration = evidence.json(output / "orchestration/manifest.json")
    evidence.check(config_path, orchestration["config_sha256"])
    evidence.check(sample_path, orchestration["samples_sha256"])
    evidence.check(
        "outputs/wam_independent_20261020/checkpoint_fingerprints.json",
        orchestration["checkpoint_fingerprints_sha256"],
    )
    if orchestration["config"] != config or orchestration["physical_gpu"] != checkpoint["gpu"]:
        raise ValueError("owned orchestration configuration/GPU differs")
    if {p.parent.name for p in directory.glob("*/result.json")} != set(expected):
        raise ValueError("result files do not cover exactly the frozen48 requests")
    template = evidence.json(config["policy_template"])
    domain = template["action_domain"]
    request_config = dict(config, action_domain=domain)
    spec = yaml.safe_load(evidence.read(Path(checkpoint["run"]) / "config.yaml"))["model"]["config"]["action128_spec"]
    transform = spec["transforms"][spec["registry"]["domains"].index(domain)]
    cases = {}
    for name, (sample, seed) in sorted(expected.items()):
        case = directory / name
        evidence.check(case / "result.json", complete["result_files"][name])
        result = evidence.json(case / "result.json")
        request = evidence.json(case / "request.json")
        if request != expected_request(request_config, sample, seed, name):
            raise ValueError("saved request differs from the frozen observation/sampler")
        evidence.check(case / "response.json", result["response_sha256"])
        response = evidence.json(case / "response.json")
        if response["status"] != "success":
            raise ValueError("saved model response is not successful")
        content = response["outputs"][0]["content"]
        if content["action128"]["checkpoint_path"] != expected_checkpoint or content["action128"]["seed"] != seed:
            raise ValueError("raw response weight or sampling seed differs")
        contract = audit_action_response(content, transform, domain=domain, fps=20)
        equal_metrics(contract, result["contract"])
        if contract["denormalization_max_abs"] > 2e-7:
            raise ValueError("physical/source-unit de-normalization differs")
        action = np.asarray(content["action"])
        if action.shape != (40, 128):
            raise ValueError("wrong H40 Action128 shape")
        identity = dict(
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
        )
        if any(result[k] != value for k, value in identity.items()):
            raise ValueError("result identity differs")
        evidence.check(sample["arrays_path"], sample["arrays_sha256"])
        evidence.check(sample["image_path"], sample["image_sha256"])
        with np.load(sample["arrays_path"]) as data:
            states = data["states"]
        frame = sample["anchor_frame"]
        count = min(40, sample["valid_future_steps"], len(states) - frame - 1)
        if count < 1 or count != result["valid_future_steps"]:
            raise ValueError("valid unpadded future coverage differs")
        poses = pose9_to_transform(states[:, :9])
        truth, opening = poses[frame + 1 : frame + 1 + count], states[frame + 1 : frame + 1 + count, 9]
        predictions = dict(
            wam=integrate_local(poses[frame], action[:count, :9])[1:],
            hold=np.repeat(poses[frame : frame + 1], count, axis=0),
            constant_velocity=extrapolate_history(poses[max(0, frame - 4) : frame + 1], count),
        )
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
        cases[name] = dict(sample=sample, sampling_seed=seed, metrics=metrics)
    progress.update(verified=len(cases), complete=len(cases) == 48)
    return config, selected, cases, progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/wam_stability_20261021"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU-only comparison requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    if not args.output.resolve().is_relative_to(Path("outputs").resolve()):
        raise ValueError("summary must be a new artifact under repository outputs")
    if args.output.exists():
        raise FileExistsError("read-only evidence comparison refuses to overwrite an earlier snapshot")
    evidence = Evidence()
    groups, selected_cases, cells = {}, {}, {}
    for label in LABELS:
        verifier = verify_earlier if label.startswith(("8400_", "8600_")) else verify_cell
        config, selected, cases, progress = verifier(label, args.root, evidence)
        if any(
            config[k] != value for k, value in dict(guidance=1.0, sampling_steps=35, shift=5.0, control_fps=20).items()
        ):
            raise ValueError("comparison requires matched guidance1.0/35steps/shift5/20Hz")
        groups[label] = dict(checkpoint=config["checkpoints"][0], guidance=1.0, progress=progress)
        selected_cases[label], cells[label] = selected, cases
    if any(selected != selected_cases[LABELS[0]] for selected in selected_cases.values()):
        raise ValueError("the four stages do not share identical diagnostic24 observations")
    complete = all(group["progress"]["complete"] for group in groups.values())
    if complete:
        names = set(cells[LABELS[0]])
        if len(names) != 48 or any(set(cases) != names for cases in cells.values()):
            raise ValueError("four completed matrices do not share exactly48 request identities")
        for label in LABELS:
            groups[label]["metrics"] = aggregate([cells[label][name] for name in sorted(names)])
        # Minimize each reported quantity separately; no mixed-unit invented score.
        comparison = {
            str(prefix): {
                key: {label: groups[label]["metrics"]["all"][str(prefix)]["wam"][key] for label in LABELS}
                for key in ("position_rmse_mm", "rotation_rmse_rad", "opening_mae")
            }
            for prefix in (8, 16, 40)
        }
    else:
        comparison = None
    evidence.read(__file__)
    evidence.read("scripts/summarize_wam_stability_screen.py")
    evidence.recheck()
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=complete,
        expected_queries_per_checkpoint=48,
        additional_query_budget=96,
        groups=groups,
        comparison=comparison,
        interpretation="State comparison is withheld until all four complete48-query matrices verify. "
        "Metrics pool only available unpadded future endpoints at prefixes8/16/40. "
        "8400/8600/8800 belong to the old22-source run;15400 is the separate23-source continuation. "
        "These are24 diagnostic phase anchors in6 historical development episodes, not new task trials, "
        "fresh holdout or proof of pretraining-unseen data. State error is not closed-loop task success.",
        input_sha256=evidence.hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), complete=complete, state_comparison_available=complete)))


if __name__ == "__main__":
    main()
