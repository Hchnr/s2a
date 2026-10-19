"""Query frozen checkpoints on the same episode-disjoint state evaluation set."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
import yaml

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.checkpoint_audit import extrapolate_history, state_metrics
from s2a.geometry import integrate_local, pose9_to_transform
from s2a.prediction_audit import audit_action_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("experiment seed differs")
    ck = next(x for x in c["checkpoints"] if x["name"] == args.checkpoint)
    path = out / "samples/manifest.json"
    samples = json.loads(path.read_text())
    template = json.loads(Path(c["policy_template"]).read_text())
    spec = yaml.safe_load((Path(ck.get("spec_run", ck["run"])) / "config.yaml").read_text())["model"]["config"][
        "action128_spec"
    ]
    domain = template["action_domain"]
    transform = spec["transforms"][spec["registry"]["domains"].index(domain)]
    endpoint = f"http://127.0.0.1:{ck['port']}"
    session = requests.Session()
    session.trust_env = False
    response = session.get(endpoint + "/info", timeout=15)
    response.raise_for_status()
    info = response.json()
    expected = str(Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model")
    if info["checkpoint_path"] != expected or info["ema_weights"]:
        raise ValueError("unexpected server weights")
    dest = out / "state_queries" / ck["name"]
    dest.mkdir(parents=True, exist_ok=args.resume)
    immutable = dict(config=c, samples_sha256=sha256(path), checkpoint=expected)
    manifest = dest / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["immutable"] != immutable:
            raise ValueError("cannot resume changed data/configuration")
        for file in ("scripts/run_wam_state_matrix.py", "src/s2a/checkpoint_audit.py", "src/s2a/geometry.py"):
            if sha256(file) != previous["code"]["files_sha256"][file]:
                raise ValueError("cannot resume changed execution code")
    else:
        write_json(manifest, dict(immutable=immutable, code=code_fingerprint(), server_info=info))
    results = []
    for sample in samples["cases"]:
        if (
            sha256(sample["arrays_path"]) != sample["arrays_sha256"]
            or sha256(sample["image_path"]) != sample["image_sha256"]
        ):
            raise ValueError("prepared sample changed")
        with np.load(sample["arrays_path"]) as data:
            states = data["states"]
        poses = pose9_to_transform(states[:, :9])
        frame = sample["anchor_frame"]
        for horizon in c["offline_horizons"]:
            count = min(horizon, sample["valid_future_steps"])
            truth = poses[frame + 1 : frame + 1 + count]
            openings = states[frame + 1 : frame + 1 + count, 9]
            hold = np.repeat(poses[frame : frame + 1], count, axis=0)
            extrapolated = extrapolate_history(poses[max(0, frame - 4) : frame + 1], count)
            baseline = dict(
                hold=state_metrics(truth, hold, openings, np.repeat(states[frame, 9], count)),
                constant_velocity=state_metrics(truth, extrapolated, openings, np.repeat(states[frame, 9], count)),
            )
            for prompt_mode in c["offline_prompt_modes"]:
                prompt = sample["task"] if prompt_mode == "task_only" else sample["published_prompts"][str(horizon)]
                for sampling_seed in c["sampling_seeds"]:
                    name = f"{sample['name']}_h{horizon}_{prompt_mode}_s{sampling_seed}"
                    case_dir = dest / name
                    request = dict(
                        name=name,
                        model_mode="wam",
                        prompt=prompt,
                        vision_path=str(Path(sample["image_path"]).resolve()),
                        domain_name="franka_panda__parallel_gripper",
                        image_size=960,
                        action_chunk_size=horizon,
                        fps=c["control_fps"],
                        num_frames=horizon + 1,
                        seed=sampling_seed,
                        num_steps=c["sampling_steps"],
                        guidance=c["guidance"],
                        shift=c["shift"],
                        extra=dict(
                            action128=dict(
                                action_domain=domain,
                                action_dim_mask=[True] * 10 + [False] * 118,
                                action_fps=c["control_fps"],
                                task_title=sample["task"],
                                selected_roles=["auxiliary_0", "left_wrist"],
                                camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                                return_video=False,
                            )
                        ),
                    )
                    result_path = case_dir / "result.json"
                    if result_path.exists() and args.resume:
                        if json.loads((case_dir / "request.json").read_text()) != request:
                            raise ValueError("resume request differs")
                        result = json.loads(result_path.read_text())
                        if sha256(case_dir / "response.json") != result["response_sha256"]:
                            raise ValueError("response changed")
                        results.append(result)
                        continue
                    case_dir.mkdir(exist_ok=False)
                    write_json(case_dir / "request.json", request)
                    start = time.monotonic()
                    response = session.post(endpoint + "/generate", json=request, timeout=900)
                    response.raise_for_status()
                    raw = response.json()
                    write_json(case_dir / "response.json", raw)
                    if raw["status"] != "success":
                        raise ValueError(f"model response failed: {raw}")
                    content = raw["outputs"][0]["content"]
                    if (
                        content["action128"]["checkpoint_path"] != expected
                        or content["action128"]["seed"] != sampling_seed
                    ):
                        raise ValueError("wrong response identity")
                    contract = audit_action_response(content, transform, domain=domain, fps=c["control_fps"])
                    if contract["denormalization_max_abs"] > 2e-7:
                        raise ValueError("denormalization mismatch")
                    action = np.asarray(content["action"])
                    if len(action) != horizon:
                        raise ValueError("wrong response horizon")
                    predicted = integrate_local(poses[frame], action[:count, :9])[1:]
                    result = dict(
                        name=name,
                        checkpoint=ck["name"],
                        sample=sample["name"],
                        task=sample["task"],
                        episode=sample["episode"],
                        split=sample["split"],
                        phase=sample["phase"],
                        anchor_frame=frame,
                        horizon=horizon,
                        valid_future_steps=count,
                        sampling_seed=sampling_seed,
                        prompt_mode=prompt_mode,
                        elapsed_s=time.monotonic() - start,
                        contract=contract,
                        metrics=state_metrics(truth, predicted, openings, np.clip(action[:count, 9], 0, 1)),
                        baselines=baseline,
                        response_sha256=sha256(case_dir / "response.json"),
                    )
                    write_json(result_path, result)
                    results.append(result)
                    write_json(dest / "progress.json", dict(completed=len(results), last_case=name))
                    print(
                        json.dumps(
                            dict(
                                completed=len(results),
                                name=name,
                                position_rmse_m=result["metrics"]["position_rmse_m"],
                                rotation_rmse_rad=result["metrics"]["rotation_rmse_rad"],
                            )
                        ),
                        flush=True,
                    )
    expected_count = (
        len(samples["cases"]) * len(c["offline_horizons"]) * len(c["offline_prompt_modes"]) * len(c["sampling_seeds"])
    )
    if len(results) != expected_count:
        raise ValueError("incomplete matrix")
    write_json(
        dest / "complete.json",
        dict(
            cases=len(results),
            manifest_sha256=sha256(manifest),
            result_files={r["name"]: sha256(dest / r["name"] / "result.json") for r in results},
        ),
    )


if __name__ == "__main__":
    main()
