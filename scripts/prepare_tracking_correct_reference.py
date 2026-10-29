"""Freeze the original eighteen correct-reference profiles for new-adapter evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import validate_transform


def main():
    root = REPO / "outputs/tracking_correct_reference_20261027"
    root.mkdir(exist_ok=False)
    prior_ref = reference(REPO / "configs/coupled_preview_control_registered_20261026.json")
    prior = read(prior_ref)
    summary_ref = reference(REPO / "outputs/coupled_preview_control_20261026/regression_summary.json")
    summary = read(summary_ref)
    require(
        summary["complete"] and summary["performance_qualified"] and summary["candidate_stable"] == 18,
        "Original qualification",
    )
    require(
        len(prior["profiles"]) == 18 and sum(p["physics_steps"] for p in prior["profiles"]) == 4907, "Original schedule"
    )
    config = dict(
        seed=42,
        prior_registration_ref=prior_ref,
        prior_qualified_summary_ref=summary_ref,
        case_indices=list(range(18)),
        physics_steps=4907,
        stable_success_last_steps=40,
        maximum_position_rmse_m=0.01,
        maximum_so3_rmse_rad=0.05,
        control_freq=20,
        online_chunk_size=40,
        max_observation_interval_steps=8,
        world_model_calls=0,
        fresh_cases=0,
        candidate="tracking primary; old first pose caps reported only; other stages unchanged",
        scope="correct-reference qualification inputs; not WAM or fresh task acceptance",
    )
    path = REPO / "configs/tracking_correct_reference_20261027.json"
    require(not path.exists(), "No overwrite")
    path.write_text(json.dumps(config, indent=2) + "\n")
    candidates = read(
        read(reference(REPO / "outputs/tracking_margin_feedback_execution_20261027/result.json"))["registration_ref"]
    )
    sources = dict(candidates["source_sha256"])
    for p in (Path(__file__), path):
        r = reference(p)
        sources[r["path"]] = r["sha256"]
    for p, digest in sources.items():
        checked(dict(path=p, sha256=digest))
    profiles = []
    for i, profile in enumerate(prior["profiles"]):
        checked(profile["trajectory_ref"])
        with np.load(checked(profile["trajectory_ref"]), allow_pickle=False) as data:
            if profile["format"] == "native_endpoints":
                poses, opening = data["source_pose"][1:], data["opening"][1:]
            elif profile["format"] == "world_reference":
                poses, opening = data["poses"], data["openings"]
            else:
                raise ValueError("Unknown original format")
            validate_transform(poses)
            require(
                len(poses) == len(opening) == profile["physics_steps"] and np.isfinite(opening).all(),
                "Original reference budget",
            )
        folder = REPO / f"outputs/coupled_preview_control_20261026/case{i:02d}"
        manifests = list(folder.glob("simulation/*/*/manifest.json"))
        require(len(manifests) == 1, "Unique original execution")
        manifest_ref = reference(manifests[0])
        manifest = read(manifest_ref)
        require(manifest["record"] == profile["record"], "Record identity")
        collection_ref = reference(folder / "collection_result.json")
        collection = read(collection_ref)
        verified_ref = reference(folder / "independent_verified.json")
        verified = read(verified_ref)
        require(verified["complete"] and verified["performance_qualified"], "Original case independently qualified")
        with np.load(checked(collection["arrays_ref"]), allow_pickle=False) as actual:
            require(len(actual["sim_states"]) == profile["physics_steps"] + 1, "Original endpoints")
            initial_sha = hashlib.sha256(actual["sim_states"][0].tobytes()).hexdigest()
        profiles.append(
            dict(
                case_index=i,
                profile=profile,
                manifest_ref=manifest_ref,
                execution_config_ref=reference(folder / "execution_config.json"),
                controller_ref=reference(manifests[0].parent / "controller.json"),
                baseline_rows_ref=reference(manifests[0].parent / "steps.jsonl"),
                baseline_collection_ref=collection_ref,
                baseline_verified_ref=verified_ref,
                initial_state_sha256=initial_sha,
                tool_ref=reference(manifest["runtime_config"]["tool_contract"]),
            )
        )
    print(
        json.dumps(
            publish(
                root / "prepared.json",
                dict(
                    complete=True,
                    passed=True,
                    config_ref=reference(path),
                    profiles=profiles,
                    source_sha256=sources,
                    physics_steps=0,
                    world_model_calls=0,
                    fresh_cases=0,
                    control_qualified=False,
                    scope="all eighteen unchanged references and original qualified cases readable; no new execution",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
