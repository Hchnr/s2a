"""Verify the complete frozen matrix and report scene-aware task effectiveness."""

import argparse
import json
from pathlib import Path

import numpy as np
from verify_wam_acceptance import verify_case

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.geometry import pose_error
from s2a.task_acceptance import scene_bootstrap, task_outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_independent_20261020.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    destination = root / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    fixtures_path = root / "simulation/fixtures.json"
    fixture_data = json.loads(fixtures_path.read_text())
    if fixture_data["config"] != c:
        raise ValueError("frozen fixture configuration changed")
    for path, digest in fixture_data["exclusions"].items():
        if sha256(path) != digest:
            raise ValueError("exclusion source changed")
    records = fixture_data["records"]
    fixtures = {r["name"]: r for r in records}
    variants = {x["name"]: x for x in c["checkpoints"]}
    labels = ["native", "oracle", *variants]
    expected = {(label, r["name"], s) for label in labels for r in records for s in c["sampling_seeds"]}
    paths = list((root / "simulation").glob("*/*/result.json"))
    actual = {
        (p.parent.parent.name, json.loads(p.read_text())["record"], json.loads(p.read_text())["sampling_seed"])
        for p in paths
    }
    if actual != expected or len(paths) != len(expected):
        raise ValueError(f"incomplete or unexpected matrix: expected {len(expected)}, found {len(paths)}")
    cases, integrity = [], []
    for path in sorted(paths):
        integrity.append(verify_case(path, c, fixtures, variants))
        result = json.loads(path.read_text())
        manifest = json.loads(path.with_name("manifest.json").read_text())
        if (
            manifest["config"]["action_chunk_size"] != c["online_chunk_size"]
            or manifest["config"]["execution_horizon"] != c["online_execution_horizon"]
        ):
            raise ValueError("generation/execution horizon differs")
        steps = [json.loads(line) for line in path.with_name("steps.jsonl").read_text().splitlines()]
        outcome = task_outcome(
            [r["success"] for r in steps],
            expected_steps=c["online_max_steps"],
            stable_steps=c["acceptance"]["stable_success_last_steps"],
        )
        row = dict(
            label=path.parent.parent.name,
            record=result["record"],
            seed=result["sampling_seed"],
            outcome=outcome,
            object_progress=result["object_progress"],
        )
        if result["source"] != "native":
            h = np.asarray(json.loads(Path(manifest["config"]["tool_contract"]).read_text())["source_from_tcp"])
            errors = pose_error(
                np.asarray([s["desired_source"] for s in steps]) @ h, np.asarray([s["after_source"] for s in steps]) @ h
            )
            position = np.linalg.norm(errors[:, :3], axis=1)
            rotation = np.linalg.norm(errors[:, 3:], axis=1)
            row["tracking"] = dict(
                position_rmse_m=float(np.sqrt(np.mean(position**2))),
                rotation_rmse_rad=float(np.sqrt(np.mean(rotation**2))),
                within_qualified_pointwise_fraction=float(
                    np.mean(
                        (position <= c["qualification_position_tolerance_m"])
                        & (rotation <= c["qualification_rotation_tolerance_rad"])
                    )
                ),
            )
        if result["source"] == "wam":
            chunks = json.loads(path.with_name("chunks.json").read_text())
            actions = [np.asarray(ch["actions"]) for ch in chunks]
            horizon = c["online_execution_horizon"]
            # Descriptive opening threshold only; no task-dependent command change.
            row["opening_diagnostics"] = dict(
                chunks=len(actions),
                executed_prefix_min=float(min(a[:horizon, 9].min() for a in actions)),
                full_prediction_min=float(min(a[:, 9].min() for a in actions)),
                deferred_closure_chunks=sum(
                    bool(a[:horizon, 9].min() >= 0.35 and a[horizon:, 9].min() < 0.35) for a in actions
                ),
            )
        cases.append(row)
    groups = {}
    for label in labels:
        grid = np.zeros((len(c["tasks"]), c["online_initial_states_per_task"], len(c["sampling_seeds"])), dtype=bool)
        selected = [row for row in cases if row["label"] == label]
        for row in selected:
            record = fixtures[row["record"]]
            grid[record["task_index"], record["initial_index"], c["sampling_seeds"].index(row["seed"])] = row[
                "outcome"
            ]["stable"]
        group = scene_bootstrap(grid, seed=args.seed)
        group.update(
            stable=int(grid.sum()),
            ever=sum(r["outcome"]["ever"] for r in selected),
            terminal=sum(r["outcome"]["terminal"] for r in selected),
            per_task_stable_fraction=grid.mean(axis=(1, 2)).tolist(),
            meets_operational_threshold=bool(
                grid.mean() >= c["acceptance"]["reliable_overall_fraction"]
                and np.all(grid.mean(axis=(1, 2)) >= c["acceptance"]["reliable_each_task_fraction"])
            ),
        )
        groups[label] = group
    report = dict(
        code=code_fingerprint(),
        config=c,
        fixtures_sha256=sha256(fixtures_path),
        cases=cases,
        groups=groups,
        limitations=[
            (
                "Previously evaluated development scenes; not an independent assessment."
                if "fixture_source" in c
                else "Experiment-independent physical initial states; pretraining visibility unknown."
            ),
            "Bootstrap intervals are descriptive and can degenerate for all-zero/all-one samples; they do not certify population reliability.",
            "Tracking errors and deferred closure are diagnostics, not unique causal attribution.",
        ],
    )
    write_json(root / "integrity.json", dict(cases=integrity))
    write_json(destination, report)
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
