"""Plot completed release-hold evidence without a model or physics process."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

from s2a.artifacts import sha256
from s2a.geometry import pose_error
from s2a.task_acceptance import task_outcome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/wam_release_hold_20261020"
MODES = ("baseline", "hold20", "hold40")
COLORS = dict(baseline="#0072B2", hold20="#D55E00", hold40="#009E73")
STYLES = dict(baseline="-", hold20="--", hold40=":")
TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def read_json(path, inputs):
    path = resolve(path)
    raw = path.read_bytes()
    signature = hashlib.sha256(raw).hexdigest()
    if str(path) in inputs and inputs[str(path)] != signature:
        raise ValueError(f"input changed: {path}")
    inputs[str(path)] = signature
    return json.loads(raw)


def verify_hashes(files, inputs):
    for raw_path, expected in files.items():
        path = resolve(raw_path)
        if sha256(path) != expected or (str(path) in inputs and inputs[str(path)] != expected):
            raise ValueError(f"evidence SHA differs: {path}")
        inputs[str(path)] = expected


def load_evidence(seed):
    inputs = {}
    summary = read_json(ROOT / "summary.json", inputs)
    c = summary["config"]
    if seed != 42 or c["seed"] != seed or summary["independent_wam_success_rate"] is not False:
        raise ValueError("requires seed42 and the fixed-reference diagnostic summary")
    if (c["steps"], c["stable_steps"], c["control_fps"], c["gripper_modes"]) != (600, 40, 20, list(MODES)):
        raise ValueError("release-hold protocol differs")
    report = resolve(c["report_path"])
    if report != REPO / "reports/wam_release_hold_20261020.md":
        raise ValueError("unexpected release-hold report path")
    verify_hashes({str(report): summary["report_sha256"]}, inputs)
    verify_hashes(summary["files_sha256"], inputs)
    prepared = read_json(ROOT / "manifest.json", inputs)
    if prepared["config"] != c:
        raise ValueError("prepared and summarized protocols differ")
    verify_hashes({"configs/wam_release_hold_20261020.json": prepared["config_sha256"]}, inputs)
    verify_hashes(prepared["implementation_sha256"], inputs)
    for source in prepared["cases"]:
        verify_hashes(source["inputs_sha256"], inputs)
    expected = {(case["name"], mode) for case in c["cases"] for mode in MODES}
    results = {(case["name"], case["mode"]): case for case in summary["cases"]}
    if len(results) != 12 or len(summary["cases"]) != 12 or set(results) != expected:
        raise ValueError("requires all four registered sources and all three modes")
    cases = {}
    for source in prepared["cases"]:
        name = source["name"]
        target = TARGETS[source["record"]["task_index"]]
        tool = read_json(source["simulation_config"]["tool_contract"], inputs)
        h = np.asarray(tool["source_from_tcp"])
        cases[name] = {}
        for mode in MODES:
            directory = ROOT / "physics" / name / mode
            result = read_json(directory / "result.json", inputs)
            if result != results[name, mode] or result["error"] is not None or result["steps"] != 600:
                raise ValueError(f"raw result differs or is incomplete: {directory}")
            path = directory / "steps.jsonl"
            if str(path) not in inputs:
                raise ValueError(f"step log is not covered by summary SHA: {path}")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != 600 or [r["step"] for r in rows] != list(range(600)):
                raise ValueError(f"incomplete or unordered step log: {path}")
            starts = np.asarray([r["command_start_time_s"] for r in rows])
            endpoints = np.asarray([r["endpoint_time_s"] for r in rows])
            np.testing.assert_allclose(starts, np.arange(600) / 20, atol=1e-12, rtol=0)
            np.testing.assert_allclose(endpoints, (np.arange(600) + 1) / 20, atol=1e-12, rtol=0)
            np.testing.assert_allclose([r["actual_dt"] for r in rows], 0.05, atol=1e-10, rtol=0)
            outcome = task_outcome([r["success"] for r in rows], expected_steps=600, stable_steps=40)
            if outcome != result["outcome"]:
                raise ValueError(f"native task outcome differs: {directory}")
            desired = np.asarray([r["desired_source"] for r in rows]) @ h
            actual = np.asarray([r["after_source"] for r in rows]) @ h
            error = pose_error(desired, actual)
            np.testing.assert_allclose(error, [r["tracking_tcp_error"] for r in rows], atol=1e-12, rtol=0)
            command = np.asarray([r["command"][-1] for r in rows])
            if not np.isin(command, [-1.0, 1.0]).all():
                raise ValueError("unexpected native gripper command")
            active = np.asarray([r["hold_active"] for r in rows], dtype=bool)
            trigger = result["verification"]["intervention_trigger"]
            expected_active = np.zeros(600, dtype=bool)
            if trigger is not None:
                start = trigger["step"]
                expected_active[start : start + result["requested_hold_steps"]] = True
            np.testing.assert_array_equal(active, expected_active)
            cases[name][mode] = dict(
                result=result,
                command_start=starts,
                endpoint=endpoints,
                command=command,
                opening_target=np.asarray([r["opening_target"] for r in rows]),
                opening_after=np.asarray([r["opening_after"] for r in rows]),
                contact=np.asarray([r["objects"][target]["grasp_contact"] for r in rows], dtype=bool),
                height_mm=np.asarray([r["objects"][target]["height_gain_m"] for r in rows]) * 1000,
                success=np.asarray([r["success"] for r in rows], dtype=bool),
                position_mm=np.linalg.norm(error[:, :3], axis=1) * 1000,
                rotation_rad=np.linalg.norm(error[:, 3:], axis=1),
                target=target,
            )
        for mode in MODES[1:]:
            np.testing.assert_array_equal(
                cases[name][mode]["opening_target"], cases[name]["baseline"]["opening_target"]
            )
    verify_hashes(inputs.copy(), {})
    return c, cases, inputs


def decorate_time_axes(axes, modes):
    for axis in axes:
        axis.grid(alpha=0.2)
        for mode in MODES[1:]:
            result = modes[mode]["result"]
            trigger = result["verification"]["intervention_trigger"]
            if trigger is not None:
                start = trigger["command_start_time_s"]
                end = start + result["requested_hold_steps"] / 20
                axis.axvspan(start, end, color=COLORS[mode], alpha=0.055, zorder=0)
                axis.axvline(start, color=COLORS[mode], linestyle="--", linewidth=0.75, alpha=0.65)


def plot_case(name, modes, destination):
    fig, grid = plt.subplots(2, 4, figsize=(22, 9))
    command_axis, opening_axis, contact_axis, height_axis, success_axis, position_axis, rotation_axis, note_axis = (
        grid.flat
    )
    opening_axis.plot(
        modes["baseline"]["endpoint"],
        modes["baseline"]["opening_target"],
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="fixed desired opening",
    )
    notes = ["Saved WAM world reference; no new WAM query.", "Simulation contact diagnostic; not a success rate."]
    for lane, mode in enumerate(MODES):
        data = modes[mode]
        t = data["endpoint"]
        style = dict(color=COLORS[mode], linestyle=STYLES[mode], linewidth=1.5, label=mode)
        command_axis.stairs(data["command"], np.r_[data["command_start"], 30.0], **style)
        opening_axis.plot(t, data["opening_after"], **style)
        contact_axis.scatter(t[data["contact"]], np.full(data["contact"].sum(), lane), color=COLORS[mode], s=9)
        height_axis.plot(t, data["height_mm"], **style)
        success_axis.scatter(
            t[data["success"]], np.full(data["success"].sum(), lane), color=COLORS[mode], s=18, marker="|"
        )
        position_axis.plot(t, data["position_mm"], **style)
        rotation_axis.plot(t, data["rotation_rad"], **style)
        result = data["result"]
        outcome = result["outcome"]
        trigger = result["verification"]["intervention_trigger"]
        interval = (
            f"[{trigger['command_start_time_s']:.2f}, "
            f"{trigger['command_start_time_s'] + result['requested_hold_steps'] / 20:.2f}) s"
            if trigger is not None
            else "none"
        )
        notes += [
            "",
            f"{mode}: forced-close command interval {interval}",
            f"  ever / terminal / stable = {int(outcome['ever'])} / {int(outcome['terminal'])} / {int(outcome['stable'])}",
            f"  target contact endpoints = {int(data['contact'].sum())}",
        ]
    command_axis.set(
        title="Actual gripper command (first 6 s)",
        xlabel="Command interval start time (s)",
        ylabel="Native command",
        xlim=(0, 6),
        yticks=[-1, 1],
        yticklabels=["-1: open", "+1: close"],
        ylim=(-1.15, 1.15),
    )
    opening_axis.set(
        title="Desired and actual opening (first 6 s)",
        xlabel="Measured endpoint time (s)",
        ylabel="Opening [0, 1]",
        xlim=(0, 6),
        ylim=(-0.05, 1.05),
    )
    for axis, title in ((contact_axis, "Target native two-finger contact"), (success_axis, "Native TASK SUCCESS")):
        axis.set(
            title=title,
            xlabel="Predicate endpoint time (s); discrete markers",
            yticks=range(3),
            yticklabels=MODES,
            ylim=(-0.5, 2.5),
            xlim=(0, 30),
        )
        for lane in range(3):
            axis.axhline(lane, color="gray", alpha=0.15, linewidth=0.5)
    height_axis.set(
        title="Target body height gain", xlabel="Endpoint time (s)", ylabel="Height gain (mm)", xlim=(0, 30)
    )
    position_axis.set(
        title="Canonical TCP tracking error", xlabel="Endpoint time (s)", ylabel="Position norm (mm)", xlim=(0, 30)
    )
    rotation_axis.set(
        title="Canonical TCP orientation tracking error",
        xlabel="Endpoint time (s)",
        ylabel="SO(3) norm (rad)",
        xlim=(0, 30),
    )
    notes += [
        "",
        "Contact markers are NOT task success.",
        "Colored shading = commanded hold interval [start, end).",
        "Trigger uses contact known before the command.",
        "Endpoint state is measured 0.05 s after command start.",
        "600 steps / 30 s; final 40 successes define stable.",
        "Study seed 42; original simulation seed 43 preserved.",
        "Arm feedback may change after gripper intervention.",
    ]
    note_axis.axis("off")
    note_axis.text(0, 1, "\n".join(notes), va="top", fontsize=8.5, transform=note_axis.transAxes)
    decorate_time_axes(list(grid.flat)[:-1], modes)
    for axis in (command_axis, opening_axis, height_axis, position_axis, rotation_axis):
        axis.legend(fontsize=8)
    fig.suptitle(
        f"{name}: fixed-reference release-hold diagnostic; target = {modes['baseline']['target']}", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    files = {}
    for suffix in ("png", "pdf"):
        path = destination / f"{name}.{suffix}"
        fig.savefig(path, dpi=160)
        files[str(path.relative_to(REPO))] = sha256(path)
    plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("explicit CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' are required")
    destination = ROOT / "diagnostics"
    if destination.exists():
        raise FileExistsError("release-hold diagnostics already exist")
    c, cases, inputs = load_evidence(args.seed)
    destination.mkdir(exist_ok=False)
    outputs = {}
    for name, modes in cases.items():
        outputs.update(plot_case(name, modes, destination))
    verify_hashes(inputs, {})
    manifest = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed,
        source_simulation_seeds=[43],
        source_summary=str((ROOT / "summary.json").relative_to(REPO)),
        input_sha256=inputs,
        implementation_sha256={str(Path(__file__).resolve().relative_to(REPO)): sha256(__file__)},
        output_sha256=outputs,
        parameters=dict(
            colors=COLORS,
            line_styles=STYLES,
            control_fps=c["control_fps"],
            steps=c["steps"],
            stable_success_tail_steps=c["stable_steps"],
            command_opening_window_s=[0, 6],
            tracking_contact_success_window_s=[0, 30],
            figsize_inches=[22, 9],
            png_dpi=160,
        ),
        interpretation="Four post-hoc fixed-world references; simulation contact diagnosis, not independent WAM success rate. Command intervals and measured endpoint times are distinct; contact is not task success.",
    )
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
