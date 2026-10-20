"""Audit verified endpoint logs without running WAM or advancing simulation."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else REPO / path).resolve()


def digest(path):
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"input changed while hashing: {path}")
    return value.hexdigest()


def read_json(path, inputs):
    raw = path.read_bytes()
    signature = hashlib.sha256(raw).hexdigest()
    if str(path) in inputs and inputs[str(path)] != signature:
        raise ValueError(f"input changed: {path}")
    inputs[str(path)] = signature
    return json.loads(raw)


def rotation_angle(desired, measured):
    relative = desired[..., :3, :3] @ np.swapaxes(measured[..., :3, :3], -1, -2)
    return np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return dict(min=float(values.min()), median=float(np.median(values)), max=float(values.max()))


def intervals(mask, dt):
    result = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i if value else i - 1
            result.append(
                dict(
                    first_step=start,
                    last_step=end,
                    first_endpoint_time_s=(start + 1) * dt,
                    last_endpoint_time_s=(end + 1) * dt,
                    steps=end - start + 1,
                )
            )
            start = None
    return result


def outcome(success, stable_steps):
    trailing = 0
    for value in success[::-1]:
        if not value:
            break
        trailing += 1
    return dict(
        ever=bool(success.any()),
        terminal=bool(success[-1]),
        stable=bool(success[-stable_steps:].all()),
        successful_steps=int(success.sum()),
        trailing_success_steps=trailing,
    )


def analyze(evidence, expected, config, inputs, evidence_path, saved_outcome):
    label, record, seed = expected
    source = "oracle" if label == "oracle" else "wam"
    identity = (evidence["record"], evidence["sampling_seed"], evidence["source"], evidence["checkpoint"])
    if identity != (record["name"], seed, source, None if source == "oracle" else label):
        raise ValueError(f"integrity identity differs: {evidence_path}")
    if evidence["steps"] != config["online_max_steps"]:
        raise ValueError(f"integrity budget incomplete: {evidence_path}")
    case_dir = resolve(config["output_dir"]) / "simulation" / label / f"{record['name']}_s{seed}"
    verified = {}
    for raw_path, signature in evidence["files"].items():
        path = resolve(raw_path)
        if path.parent != case_dir or digest(path) != signature:
            raise ValueError(f"integrity path or SHA differs: {path}")
        if str(path) in inputs and inputs[str(path)] != signature:
            raise ValueError(f"conflicting input SHA: {path}")
        inputs[str(path)] = signature
        verified[path.name] = path
    required = {"result.json", "manifest.json", "steps.jsonl", "controller.json", "chunks.json"}
    if not required.issubset(verified):
        raise ValueError(f"integrity lacks required raw evidence: {case_dir}")
    result = read_json(verified["result.json"], inputs)
    manifest = read_json(verified["manifest.json"], inputs)
    rows = [json.loads(line) for line in verified["steps.jsonl"].read_text().splitlines()]
    count = config["online_max_steps"]
    if (
        result["error"] is not None
        or result["steps"] != count
        or result["expected_steps"] != count
        or len(rows) != count
        or [r["step"] for r in rows] != list(range(count))
        or manifest["record"] != record
        or manifest["source"] != source
        or result["record"] != record["name"]
        or result["sampling_seed"] != seed
        or result["source"] != source
        or result["checkpoint"] != (None if source == "oracle" else label)
    ):
        raise ValueError(f"incomplete or inconsistent case: {case_dir}")
    dt = 1 / config["control_fps"]
    np.testing.assert_allclose([r["actual_dt"] for r in rows], dt, atol=1e-10, rtol=0)
    for field in ("control_fps", "inverse_weight", "max_command_rate", "opening_hysteresis"):
        if manifest["config"][field] != config[field]:
            raise ValueError(f"control configuration differs: {case_dir}: {field}")
    if manifest["config"]["seed"] != seed:
        raise ValueError(f"simulation seed differs: {case_dir}")
    if any(type(r["success"]) is not bool for r in rows):
        raise ValueError("success must be the recorded boolean task predicate")
    success = np.asarray([r["success"] for r in rows])
    task_outcome = outcome(success, config["acceptance"]["stable_success_last_steps"])
    if saved_outcome is None or saved_outcome != task_outcome:
        raise ValueError(f"recorded integrity outcome differs or is absent: {evidence_path}")
    if result["success"] != task_outcome["ever"] or result["terminal_success"] != task_outcome["terminal"]:
        raise ValueError(f"result outcome differs: {case_dir}")
    tool_path = resolve(manifest["config"]["tool_contract"])
    tool = read_json(tool_path, inputs)
    if inputs[str(tool_path)] != manifest["tool_sha256"]:
        raise ValueError("tool contract changed")
    h = np.asarray(tool["source_from_tcp"], dtype=float)
    measured = np.asarray([r["measured_source"] for r in rows], dtype=float)
    desired = np.asarray([r["desired_source"] for r in rows], dtype=float)
    actual = np.asarray([r["after_source"] for r in rows], dtype=float)
    command = np.asarray([r["command"] for r in rows], dtype=float)
    if any(a.shape != (count, 4, 4) or not np.isfinite(a).all() for a in (measured, desired, actual)):
        raise ValueError(f"invalid pose log: {case_dir}")
    if command.shape != (count, 7) or not np.isfinite(command).all():
        raise ValueError(f"invalid command log: {case_dir}")
    desired_tcp, actual_tcp = desired @ h, actual @ h
    position_error = np.linalg.norm(desired_tcp[:, :3, 3] - actual_tcp[:, :3, 3], axis=1)
    rotation_error = rotation_angle(desired_tcp, actual_tcp)
    pre_command_rotation = rotation_angle(desired, measured)
    opening = np.asarray([r["opening_target"] for r in rows], dtype=float)
    actual_opening = np.asarray([r["opening_after"] for r in rows], dtype=float)
    if not np.isfinite(opening).all() or not np.isfinite(actual_opening).all():
        raise ValueError(f"invalid opening log: {case_dir}")
    target = TARGETS[record["task_index"]]
    contact = np.asarray([r["objects"][target]["grasp_contact"] for r in rows], dtype=bool)
    mode = manifest["config"].get("gripper_mode", "intent")
    sign_applicable = mode in ("intent", "latch", "native") and not any(
        r.get("gripper_mode") == "position" for r in rows
    )
    g = command[:, 6]
    if sign_applicable and not np.isin(g, (-1, 1)).all():
        raise ValueError(f"nonbinary command in sign-controlled gripper log: {case_dir}")
    times = (np.arange(count) + 1) * dt
    first_contact = int(np.flatnonzero(contact)[0]) if contact.any() else None

    def event(index):
        if index is None:
            return None
        row = rows[index]
        return dict(
            step=index,
            line=index + 1,
            command_start_time_s=index * dt,
            endpoint_time_s=(index + 1) * dt,
            command=float(g[index]) if sign_applicable else None,
            reference_opening=float(opening[index]),
            actual_opening=float(actual_opening[index]),
            reference_change_from_previous=float(opening[index] - opening[index - 1]) if index else None,
            prior_endpoint_grasp_contact=bool(contact[index - 1]) if index else None,
            endpoint_grasp_contact=bool(contact[index]),
            endpoint_success=bool(success[index]),
            tcp_tracking_error_m=float(position_error[index]),
            rotation_tracking_error_rad=float(rotation_error[index]),
            target_object_position=row["objects"][target]["position"],
            target_height_gain_m=row["objects"][target]["height_gain_m"],
        )

    def first(indices):
        return event(int(indices[0])) if len(indices) else None

    sign = dict(applicable=sign_applicable, mode=mode)
    if sign_applicable:
        reverse = np.flatnonzero(g[1:] != g[:-1]) + 1
        opens = np.flatnonzero((g[:-1] > 0) & (g[1:] < 0)) + 1
        closes = np.flatnonzero((g[:-1] < 0) & (g[1:] > 0)) + 1
        near_closed_opens = opens[opening[opens] < 0.35]
        sign.update(
            positive_close_steps=int((g > 0).sum()),
            negative_open_steps=int((g < 0).sum()),
            zero_steps=int((g == 0).sum()),
            direction_reversals_full=int(len(reverse)),
            direction_reversals_first_5s=int((times[reverse] <= 5).sum()),
            close_to_open_transitions=int(len(opens)),
            open_to_close_transitions=int(len(closes)),
            first_close_command=first(np.flatnonzero(g > 0)),
            first_open_command=first(np.flatnonzero(g < 0)),
            near_closed_open_transitions=int(len(near_closed_opens)),
            first_near_closed_open_transition=first(near_closed_opens),
            first_open_transition_after_first_grasp=(
                first(opens[opens >= first_contact]) if first_contact is not None else None
            ),
            contact_associated_open_transitions=[event(int(i)) for i in opens if contact[i - 1] or contact[i]],
        )
    losses = np.flatnonzero(contact[:-1] & ~contact[1:]) + 1
    tracking = {}
    for name, mask in {"first_5s": times <= 5, "full_budget": np.ones(count, dtype=bool)}.items():
        tracking[name] = dict(
            steps=int(mask.sum()),
            canonical_tcp_position_rmse_m=float(np.sqrt(np.mean(position_error[mask] ** 2))),
            rotation_rmse_rad=float(np.sqrt(np.mean(rotation_error[mask] ** 2))),
            saturated_steps=sum(bool(r["saturated"]) for r, keep in zip(rows, mask, strict=True) if keep),
            rate_limited_steps=sum(bool(r["rate_limited"]) for r, keep in zip(rows, mask, strict=True) if keep),
            rotation_inverse_fallback_steps=int(((pre_command_rotation > 0.25) & mask).sum()),
        )
    contact_runs = intervals(contact, dt)
    return dict(
        label=label,
        source=source,
        record=record["name"],
        task_index=record["task_index"],
        sampling_seed=seed,
        result_path=str(verified["result.json"]),
        integrity_evidence=str(evidence_path),
        outcome=task_outcome,
        command=sign,
        opening=dict(
            reference=describe(opening),
            measured=describe(actual_opening),
            mae=float(np.mean(np.abs(actual_opening - opening))),
            first_reference_below_0_35=first(np.flatnonzero(opening < 0.35)),
        ),
        grasp=dict(
            target_object=target,
            contact_steps=int(contact.sum()),
            first_contact=event(first_contact),
            first_loss_after_first_contact=first(losses),
            contact_intervals=contact_runs,
            longest_contact_run_steps=max((r["steps"] for r in contact_runs), default=0),
            loss_endpoints_while_open_command=int((g[losses] < 0).sum()) if sign_applicable else None,
        ),
        tracking=tracking,
    )


def summarize(cases):
    groups = defaultdict(list)
    for case in cases:
        for name in (case["label"], f"{case['label']}/task{case['task_index']}"):
            groups[name].append(case)
        groups["wam_all" if case["source"] == "wam" else "oracle_all"].append(case)
    result = {}
    for name, rows in sorted(groups.items()):
        sign_rows = [r for r in rows if r["command"]["applicable"]]
        result[name] = dict(
            trials=len(rows),
            initial_states=len({r["record"] for r in rows}),
            tasks=sorted({r["task_index"] for r in rows}),
            ever=sum(r["outcome"]["ever"] for r in rows),
            terminal=sum(r["outcome"]["terminal"] for r in rows),
            stable=sum(r["outcome"]["stable"] for r in rows),
            no_target_grasp_contact=sum(r["grasp"]["contact_steps"] == 0 for r in rows),
            binary_sign_command_cases=len(sign_rows),
            direction_reversals_full=describe([r["command"]["direction_reversals_full"] for r in sign_rows]),
            direction_reversals_first_5s=describe([r["command"]["direction_reversals_first_5s"] for r in sign_rows]),
            positive_close_steps=describe([r["command"]["positive_close_steps"] for r in sign_rows]),
            saturated_steps=describe([r["tracking"]["full_budget"]["saturated_steps"] for r in rows]),
            rate_limited_steps=describe([r["tracking"]["full_budget"]["rate_limited_steps"] for r in rows]),
            rotation_inverse_fallback_steps=describe(
                [r["tracking"]["full_budget"]["rotation_inverse_fallback_steps"] for r in rows]
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_independent_20261020.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = resolve(args.output)
    if not output.is_relative_to(REPO / "outputs") or output == REPO / "outputs":
        raise ValueError("output must be a new file inside this repository outputs/")
    if output.exists():
        raise FileExistsError(output)
    inputs = {}
    config_path = resolve(args.config)
    config = read_json(config_path, inputs)
    if args.seed != config["seed"] or args.seed != 42 or "fixture_source" in config:
        raise ValueError("requires seed 42 and the main independent collection")
    root = resolve(config["output_dir"])
    if not root.is_relative_to(REPO / "outputs"):
        raise ValueError("input collection must be inside repository outputs/")
    fixture_path = root / "simulation/fixtures.json"
    fixtures = read_json(fixture_path, inputs)
    if fixtures["config"] != config:
        raise ValueError("frozen fixture configuration changed")
    records = fixtures["records"]
    if len({r["name"] for r in records}) != len(records):
        raise ValueError("duplicate fixture record")
    labels = [c["name"] for c in config["checkpoints"]]
    expected = {
        (label, record["name"], seed): (label, record, seed)
        for label in labels + ["oracle"]
        for record in records
        for seed in config["sampling_seeds"]
    }
    checks = set((root / "incremental_checks").glob("*.json"))
    oracle_path = root / "oracle_integrity.json"
    oracle = read_json(oracle_path, inputs)["cases"] if oracle_path.exists() else []
    evidence = {}
    for item in oracle:
        key = ("oracle", item["record"], item["sampling_seed"])
        if key not in expected or key in evidence:
            raise ValueError("unexpected or duplicated oracle integrity case")
        evidence[key] = (item, oracle_path, item.get("outcome"))
    for key in expected:
        label, name, seed = key
        check_path = root / "incremental_checks" / f"{label}_{name}_s{seed}.json"
        if label == "oracle" or check_path not in checks:
            continue
        check = read_json(check_path, inputs)
        saved_outcome = check.get("outcome", check.get("diagnostics", {}).get("outcome"))
        evidence[key] = (check["integrity"], check_path, saved_outcome)
    coverage = []
    cases = []
    for key, identity in expected.items():
        label, name, seed = key
        result_path = root / "simulation" / label / f"{name}_s{seed}" / "result.json"
        status = "verified_complete_analyzed" if key in evidence else "not_verified_not_analyzed"
        coverage.append(
            dict(label=label, record=name, sampling_seed=seed, status=status, result_present=result_path.exists())
        )
        if key in evidence:
            item, path, saved_outcome = evidence[key]
            cases.append(analyze(item, identity, config, inputs, path, saved_outcome))
    inputs[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    for path, signature in inputs.items():
        if digest(Path(path)) != signature:
            raise ValueError(f"input changed during audit: {path}")
    counts = {
        label: dict(
            expected=sum(r["label"] == label for r in coverage),
            verified_complete_analyzed=sum(r["label"] == label for r in cases),
            not_verified_not_analyzed=sum(
                r["label"] == label and r["status"] != "verified_complete_analyzed" for r in coverage
            ),
        )
        for label in labels + ["oracle"]
    }
    report = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed,
        scope="Read-only diagnostic snapshot of existing verified complete WAM and main-collection oracle logs.",
        config_path=str(config_path),
        inputs_sha256=inputs,
        coverage_counts=counts,
        coverage=coverage,
        definitions={
            "time": "Command acts over [step/fps, (step+1)/fps]; logged state/contact/success belong to the endpoint.",
            "reversals": "Adjacent recorded -1/+1 command changes; initial open is not a transition. Position placeholders are excluded.",
            "first_5s": "Post-hoc descriptive window with endpoint time <=5s; not an acceptance criterion.",
            "near_closed": "Reference opening <0.35 is descriptive; OpeningIntent does not use this absolute threshold.",
            "first_open_after_grasp": "First +1 to -1 transition at or after the first grasp-contact endpoint; temporal association only.",
            "rotation_fallback": "Recomputed desired-versus-current source SO(3) angle >0.25rad, matching OSC inverse fallback condition.",
            "contact_intervals": "Runs of sampled native grasp-contact endpoints; not continuous contact durations or force estimates.",
        },
        cases=cases,
        groups=summarize(cases),
        limitations=[
            "Missing or unverified trials are not failures; result presence alone is not completeness verification.",
            "This does not execute simulation, query WAM, recompute physical task predicates, or change the frozen protocol.",
            "WAM and oracle coverage can differ; compare matched task/initial-state strata, not just pooled summaries.",
            "Repeated seeds are not independent scenes. Partial snapshots do not estimate population success or rank checkpoints.",
            "Command reversals, contact loss, saturation and rate limiting are co-occurrences, not unique causal attribution.",
            "Successful examples remain included; an absolute opening MAE does not by itself measure grasp force or task quality.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), coverage_counts=counts), ensure_ascii=False))


if __name__ == "__main__":
    main()
