"""Audit all saved live-model outputs against their frozen checkpoint contract."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.libero_audit import post_action_reference
from s2a.prediction_audit import audit_action_response, geometry_roundtrip


def main():
    import h5py

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_audit_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-config")
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen config")
    sim = config["simulation"]
    if args.policy_config:
        sim, output = load_config(args.policy_config)
    previous = Path(sim["output_dir"])
    fixture = json.loads(Path(sim["fixture"]).read_text())
    h = np.asarray(json.loads(Path(sim["tool_contract"]).read_text())["source_from_tcp"])
    rows, geometry, inputs, specifications = [], [], {}, {}
    for path in sorted((previous / "policy").glob("*/trial*/response_*.json")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        config_path = Path(manifest["server_info"]["config_file"])
        if str(config_path) not in inputs:
            inputs[str(config_path)] = sha256(config_path)
            specifications[str(config_path)] = yaml.safe_load(config_path.read_text())["model"]["config"][
                "action128_spec"
            ]
        spec = specifications[str(config_path)]
        domain = sim["action_domain"]
        selected = spec["transforms"][spec["registry"]["domains"].index(domain)]
        content = json.loads(path.read_text())["outputs"][0]["content"]
        row = audit_action_response(content, selected, domain=domain, fps=sim["control_fps"])
        row.update(
            path=str(path),
            sha256=sha256(path),
            candidate=path.parent.parent.name,
            trial=int(path.parent.name.removeprefix("trial")),
        )
        if row["denormalization_max_abs"] > 1e-6:
            raise ValueError(f"denormalization contract failed: {path}")
        rows.append(row)
    expected_responses = sum(
        len(json.loads((previous / "policy" / candidate / f"trial{trial}" / "result.json").read_text())["requests"])
        for candidate in ("baseline", "compensated")
        for trial in sim["trials"]
    )
    if len(rows) != expected_responses:
        raise ValueError("saved live-policy output matrix is incomplete")
    inputs[fixture["source_hdf5"]] = sha256(fixture["source_hdf5"])
    inputs[sim["tool_contract"]] = sha256(sim["tool_contract"])
    with h5py.File(fixture["source_hdf5"]) as source:
        for record in fixture["episodes"]:
            demo = source["data"][record["demo"]]
            poses, _ = post_action_reference(
                demo["obs/ee_pos"][:], demo["obs/ee_ori"][:], demo["obs/gripper_states"][:]
            )
            geometry.append(dict(episode=record["episode"], **geometry_roundtrip(poses, h)))
    result = dict(
        code=code_fingerprint(),
        inputs_sha256=inputs,
        responses=len(rows),
        cases=rows,
        geometry=geometry,
        worst_denormalization_error=max(r["denormalization_max_abs"] for r in rows),
        mean_opening_outside_fraction=float(np.mean([r["opening_outside_fraction"] for r in rows])),
        conclusion="Normalization and SE(3) encoding consistency are separate from prediction quality.",
        tuning_performed=False,
        checkpoint_iteration=sim["checkpoint_iteration"],
    )
    write_json(output / "prediction_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "code", "inputs_sha256")}))


if __name__ == "__main__":
    main()
