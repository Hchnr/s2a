"""Aggregate completed evidence without treating missing cases as failures."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from s2a.artifacts import load_config, sha256, write_json


def summarize(rows, field="metrics"):
    curves = [r[field] if field == "metrics" else r["baselines"][field] for r in rows]
    positions = np.concatenate([r["position_errors_m"] for r in curves])
    rotations = np.concatenate([r["rotation_errors_rad"] for r in curves])
    openings = np.concatenate([r["opening_errors"] for r in curves])
    result = dict(
        cases=len(rows),
        episodes=len({r["episode"] for r in rows}),
        distinct_anchors=len({(r["episode"], r["anchor_frame"]) for r in rows}),
        endpoints=len(positions),
        position_rmse_m=float(np.sqrt(np.mean(positions**2))),
        rotation_rmse_rad=float(np.sqrt(np.mean(rotations**2))),
        opening_mae=float(np.mean(abs(openings))),
    )
    if field == "metrics":
        frames = sum(r["contract"]["frames"] for r in rows)
        result["raw_output"] = dict(
            generated_endpoints=frames,
            opening_outside_fraction=sum(
                r["contract"]["frames"] * r["contract"]["opening_outside_fraction"] for r in rows
            )
            / frames,
            opening_min=min(r["raw_opening"]["minimum"] for r in rows),
            opening_max=max(r["raw_opening"]["maximum"] for r in rows),
            opening_clip_distance_mean=sum(r["raw_opening"]["clip_distance_sum"] for r in rows) / frames,
            opening_clip_distance_max=max(r["raw_opening"]["clip_distance_max"] for r in rows),
            rot6d_column_norm_error_mean=sum(
                r["contract"]["frames"] * r["contract"]["rot6d_column_norm_error_mean"] for r in rows
            )
            / frames,
        )
    result["endpoints_by_time"] = []
    for i in (0, 3, 7, 15, 23, 31, 39):
        valid = [r for r in curves if r["endpoints"] > i]
        if valid:
            result["endpoints_by_time"].append(
                dict(
                    time_s=(i + 1) / 20,
                    count=len(valid),
                    position_rmse_m=float(np.sqrt(np.mean([r["position_errors_m"][i] ** 2 for r in valid]))),
                    rotation_rmse_rad=float(np.sqrt(np.mean([r["rotation_errors_rad"][i] ** 2 for r in valid]))),
                    opening_mae=float(np.mean([abs(r["opening_errors"][i]) for r in valid])),
                )
            )
    return result


def tcp_position_errors(path, steps):
    manifest = json.loads(path.with_name("manifest.json").read_text())
    h = np.asarray(json.loads(Path(manifest["config"]["tool_contract"]).read_text())["source_from_tcp"])
    target = np.asarray([r["desired_source"] for r in steps]) @ h
    actual = np.asarray([r["after_source"] for r in steps]) @ h
    return np.linalg.norm(target[:, :3, 3] - actual[:, :3, 3], axis=1)


def physics_diagnostics(paths, source):
    """Separate reference tracking and task-object progress from task predicates."""
    objects = {0: "akita_black_bowl_1", 1: "wine_bottle_1", 2: "cream_cheese_1"}
    rows = []
    for path in paths:
        result = json.loads(path.read_text())
        if result["error"] or result["steps"] != result["expected_steps"]:
            continue
        task = int(result["record"].split("_")[0].removeprefix("task"))
        steps = [json.loads(line) for line in path.with_name("steps.jsonl").read_text().splitlines()]
        item = dict(
            record=result["record"],
            sampling_seed=result["sampling_seed"],
            task_index=task,
            task_object=objects[task],
            task_object_progress=result["object_progress"][objects[task]],
            success=result["success"],
            terminal_success=result["terminal_success"],
            tracking_applicable=source != "native",
        )
        if source != "native":
            errors = np.asarray([r["error"] for r in steps])
            position, rotation = np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)
            tcp_position = tcp_position_errors(path, steps)
            opening = np.asarray([r["opening_target"] for r in steps])
            item.update(
                position_rmse_m=float(np.sqrt(np.mean(position**2))),
                position_error_point="controller source (grip_site position / right_hand axes)",
                position_rmse_tcp_m=float(np.sqrt(np.mean(tcp_position**2))),
                position_max_tcp_m=float(tcp_position.max()),
                tcp_position_within_10mm_fraction=float(np.mean(tcp_position <= 0.01)),
                rotation_rmse_rad=float(np.sqrt(np.mean(rotation**2))),
                position_max_m=float(position.max()),
                rotation_max_rad=float(rotation.max()),
                # These compare achieved tracking with the frozen tolerance,
                # not a claim that every arbitrary reference lies in the probe domain.
                position_within_10mm_fraction=float(np.mean(position <= 0.01)),
                rotation_within_005rad_fraction=float(np.mean(rotation <= 0.05)),
                opening_target_min=float(opening.min()),
                opening_target_max=float(opening.max()),
                opening_mae=float(np.mean([abs(r["opening_after"] - r["opening_target"]) for r in steps])),
                saturated_steps=sum(r["saturated"] for r in steps),
                rate_limited_steps=sum(r["rate_limited"] for r in steps),
            )
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    samples = json.loads((root / "samples/manifest.json").read_text())["cases"]
    expected_queries = (
        len(samples) * len(c["offline_horizons"]) * len(c["offline_prompt_modes"]) * len(c["sampling_seeds"])
    )
    expected_names = {
        f"{sample['name']}_h{h}_{prompt}_s{seed}"
        for sample in samples
        for h in c["offline_horizons"]
        for prompt in c["offline_prompt_modes"]
        for seed in c["sampling_seeds"]
    }
    names = ["8400", "8600", "8800", "warm6200_masked", "8400_masked", "8600_masked", "8800_masked"]
    matrices, groups, fine_groups, fingerprints = [], defaultdict(list), defaultdict(list), {}
    by_name = {}
    all_complete = True
    for name in names:
        directory = root / "state_queries" / name
        paths = sorted(directory.glob("*/result.json"))
        if not {p.parent.name for p in paths}.issubset(expected_names):
            raise ValueError("state result outside the frozen sample matrix")
        complete = len(paths) == expected_queries and (directory / "complete.json").exists()
        all_complete &= complete
        matrices.append(dict(name=name, expected=expected_queries, completed=len(paths), complete=complete))
        if complete:
            declaration = json.loads((directory / "complete.json").read_text())
            if declaration["cases"] != expected_queries:
                raise ValueError("completion count differs")
            fingerprints[str(directory / "complete.json")] = sha256(directory / "complete.json")
            for path in paths:
                if sha256(path) != declaration["result_files"][path.parent.name]:
                    raise ValueError("result changed after completion")
        for path in paths:
            row = json.loads(path.read_text())
            if row["checkpoint"] != name or row["name"] != path.parent.name:
                raise ValueError("state result identity differs from its matrix")
            response = json.loads(path.with_name("response.json").read_text())["outputs"][0]["content"]
            raw_opening = np.asarray(response["action"])[:, 9]
            clip_distance = abs(raw_opening - np.clip(raw_opening, 0, 1))
            row["raw_opening"] = dict(
                minimum=float(raw_opening.min()),
                maximum=float(raw_opening.max()),
                clip_distance_sum=float(clip_distance.sum()),
                clip_distance_max=float(clip_distance.max()),
            )
            key = (name, row["split"], row["prompt_mode"], row["horizon"])
            groups[key].append(row)
            fine_groups[key + (row["task"], row["phase"])].append(row)
            by_name[(name, row["name"])] = row
    summaries = []
    for key, rows in groups.items():
        summaries.append(
            dict(
                checkpoint=key[0],
                split=key[1],
                prompt_mode=key[2],
                horizon=key[3],
                model=summarize(rows),
                hold=summarize(rows, "hold"),
                constant_velocity=summarize(rows, "constant_velocity"),
            )
        )
    detail = []
    for key, rows in fine_groups.items():
        detail.append(
            dict(
                checkpoint=key[0],
                split=key[1],
                prompt_mode=key[2],
                horizon=key[3],
                task=key[4],
                phase=key[5],
                model=summarize(rows),
                hold=summarize(rows, "hold"),
                constant_velocity=summarize(rows, "constant_velocity"),
            )
        )
    paired = []
    for checkpoint in ("8400", "8600", "8800"):
        pairs = defaultdict(list)
        for (name, case), stock in by_name.items():
            if name == checkpoint and (checkpoint + "_masked", case) in by_name:
                masked = by_name[(checkpoint + "_masked", case)]
                pairs[(stock["split"], stock["prompt_mode"], stock["horizon"])].append((stock, masked))
        for key, values in pairs.items():
            before, after = summarize([v[0] for v in values]), summarize([v[1] for v in values])
            paired.append(
                dict(
                    checkpoint=checkpoint,
                    split=key[0],
                    prompt_mode=key[1],
                    horizon=key[2],
                    stock=before,
                    masked=after,
                    masked_minus_stock={
                        k: after[k] - before[k] for k in ("position_rmse_m", "rotation_rmse_rad", "opening_mae")
                    },
                )
            )
    physics = {}
    simulation_complete = True
    fixtures = json.loads((root / "simulation/fixtures.json").read_text())["records"]
    expected_physics = {(r["name"], seed) for r in fixtures for seed in c["sampling_seeds"]}
    for name in ["native", "oracle", *[x for x in names if x.endswith("_masked")]]:
        paths = sorted((root / "simulation" / name).glob("*/result.json"))
        rows = [json.loads(p.read_text()) for p in paths]
        expected = len(c["tasks"]) * c["online_initial_states_per_task"] * len(c["sampling_seeds"])
        valid = [r for r in rows if r["error"] is None and r["steps"] == r["expected_steps"] == c["online_max_steps"]]
        keys = [(r["record"], r["sampling_seed"]) for r in valid]
        if len(keys) != len(set(keys)) or not set(keys).issubset(expected_physics):
            raise ValueError("duplicate or unexpected physics initial-state/seed pair")
        complete = len(valid) == expected
        simulation_complete &= complete
        physics[name] = dict(
            expected=expected,
            completed=len(valid),
            complete=complete,
            success=sum(r["success"] for r in valid),
            terminal_success=sum(r["terminal_success"] for r in valid),
            cases=rows,
            diagnostics=physics_diagnostics(paths, name),
        )
        if complete:
            fingerprints.update({str(p): sha256(p) for p in paths})
    probes = []
    for path in sorted((root / "simulation").glob("probe_*/*/result.json")):
        result = json.loads(path.read_text())
        if result["steps"] != 180 or result["error"]:
            raise ValueError("incomplete executor probe")
        steps = [json.loads(line) for line in (path.parent / "steps.jsonl").read_text().splitlines()]
        errors = np.asarray([r["error"] for r in steps])
        tcp_errors = tcp_position_errors(path, steps)
        probes.append(
            dict(
                name=path.parent.parent.name,
                result=result,
                position_p95_m=float(np.quantile(np.linalg.norm(errors[:, :3], axis=1), 0.95)),
                position_max_m=float(np.linalg.norm(errors[:, :3], axis=1).max()),
                rotation_p95_rad=float(np.quantile(np.linalg.norm(errors[:, 3:], axis=1), 0.95)),
                rotation_max_rad=float(np.linalg.norm(errors[:, 3:], axis=1).max()),
                position_rmse_tcp_m=float(np.sqrt(np.mean(tcp_errors**2))),
                position_max_tcp_m=float(tcp_errors.max()),
            )
        )
    complete = all_complete and simulation_complete and len(probes) == 6
    if not args.partial and not complete:
        raise ValueError("evidence matrix is incomplete; use --partial for an explicitly provisional snapshot")
    sample_by_name = {sample["name"]: sample for sample in samples}
    labels = json.loads((root / "state_label_audit/result.json").read_text())["rows"]
    label_groups = defaultdict(list)
    for row in labels:
        sample = sample_by_name[row["sample"]]
        label_groups[(sample["split"], row["horizon"])].append(
            dict(
                episode=row["episode"],
                anchor_frame=sample["anchor_frame"],
                baselines={"clipped_label": row["clipped_target_state_error"]},
            )
        )
    report = dict(
        complete=complete,
        state_complete=all_complete,
        checkpoint_matrices=matrices,
        state=summaries,
        state_by_phase_task=detail,
        paired_serving_effects=paired,
        clipped_label_reconstruction=[
            dict(split=key[0], horizon=key[1], metrics=summarize(rows, "clipped_label"))
            for key, rows in label_groups.items()
        ],
        physics=physics,
        probes=probes,
        fingerprints=fingerprints,
        phase_proxy_fallbacks=[dict(sample=r["name"], phase=r["phase"]) for r in samples if not r["event_detected"]],
        interpretation=[
            "Study-heldout is episode-disjoint within this experiment, not proven pretraining-unseen.",
            "Six independent physics initial states, each evaluated under two sampling seeds.",
            "Published captions are privileged offline diagnosis; online uses task-only instructions.",
            "Missing and infrastructure-failed cases are never counted as task failures.",
            "Native-command error fields in early artifacts describe step motion, not reference tracking.",
            "Constant velocity extrapolates observed pose motion; both causal baselines hold the current gripper opening.",
            "Opening state MAE uses the physical [0,1] projection; raw range violations and their magnitude are reported separately over all generated endpoints.",
            "Original physics position errors refer to the controller source point; canonical TCP tracking is recomputed explicitly with the frozen tool transform. SO(3) error magnitude is unchanged by this fixed right transform.",
        ],
    )
    out = root / ("summary_partial.json" if args.partial else "summary.json")
    if out.exists() and not args.partial:
        raise FileExistsError("complete acceptance summaries must not be silently overwritten")
    write_json(out, report)
    print(
        json.dumps(
            dict(
                complete=complete,
                state=matrices,
                physics={k: dict(completed=v["completed"], success=v["success"]) for k, v in physics.items()},
            )
        )
    )


if __name__ == "__main__":
    main()
