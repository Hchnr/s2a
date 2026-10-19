"""Audit state/delta alignment and quantify training-label clipping in state space."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.checkpoint_audit import state_metrics
from s2a.geometry import integrate_local, pose9_to_transform
from s2a.prediction_audit import denormalize_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    out = root / "state_label_audit"
    out.mkdir(exist_ok=False)
    samples_path = root / "samples/manifest.json"
    samples = json.loads(samples_path.read_text())["cases"]
    model_config = Path(c["checkpoints"][-1]["run"]) / "config.yaml"
    spec = yaml.safe_load(model_config.read_text())["model"]["config"]["action128_spec"]
    domain = json.loads(Path(c["policy_template"]).read_text())["action_domain"]
    transform = spec["transforms"][spec["registry"]["domains"].index(domain)]
    mask = np.asarray(transform["normalization_mask"], dtype=bool)
    rows = []
    for sample in samples:
        with np.load(sample["arrays_path"]) as data:
            states, actions = data["states"], data["actions"]
        poses = pose9_to_transform(states[:, :9])
        frame = sample["anchor_frame"]
        for horizon in c["offline_horizons"]:
            count = min(horizon, len(states) - frame - 1)
            raw = actions[frame : frame + count].copy()
            normalized = raw.copy()
            normalized[:, mask] = (
                raw[:, mask] * np.asarray(transform["scale"])[mask] + np.asarray(transform["bias"])[mask]
            )
            unclipped = normalized.copy()
            normalized[:, mask] = np.clip(normalized[:, mask], transform["clip_min"], transform["clip_max"])
            clipped = denormalize_action(normalized, transform)
            truth = poses[frame + 1 : frame + 1 + count]
            opening = states[frame + 1 : frame + 1 + count, 9]
            reconstructed = integrate_local(poses[frame], raw[:, :9])[1:]
            clipped_states = integrate_local(poses[frame], clipped[:, :9])[1:]
            roundtrip = state_metrics(truth, reconstructed, opening, raw[:, 9])
            rows.append(
                dict(
                    sample=sample["name"],
                    episode=sample["episode"],
                    horizon=horizon,
                    valid_endpoints=count,
                    raw_delta_roundtrip=roundtrip,
                    clipped_target_state_error=state_metrics(truth, clipped_states, opening, clipped[:, 9]),
                    clipped_elements=int(np.count_nonzero(unclipped != normalized)),
                )
            )
    maximum_position = max(max(r["raw_delta_roundtrip"]["position_errors_m"]) for r in rows)
    maximum_rotation = max(max(r["raw_delta_roundtrip"]["rotation_errors_rad"]) for r in rows)
    write_json(
        out / "result.json",
        dict(
            code=code_fingerprint(),
            samples_sha256=sha256(samples_path),
            checkpoint_config_sha256=sha256(model_config),
            rows=rows,
            raw_delta_max_position_m=maximum_position,
            raw_delta_max_rotation_rad=maximum_rotation,
            interpretation="Published state labels, not expert controller commands; clipped target is an offline diagnostic, never an online reference.",
        ),
    )
    if maximum_position > 1e-5 or maximum_rotation > 1e-5:
        raise ValueError("raw published delta labels do not reconstruct their future state")
    print(json.dumps(dict(windows=len(rows), max_position_m=maximum_position, max_rotation_rad=maximum_rotation)))


if __name__ == "__main__":
    main()
