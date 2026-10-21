"""CPU-only diagnostics, gated on the fully verified24-case WAM pilot snapshot."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO, sha256

LABELS = ("8800_g15", "8800_g10", "15400_g15", "15400_g10")
VARIANTS = ("baseline", "governor_only")
RECORDS = ("task0_init0", "task1_init0", "task2_init0")
COLORS = {"baseline": "#0072B2", "governor_only": "#D55E00"}
DISPLAY = {"baseline": "baseline", "governor_only": "original governor_only (v1)"}
AUDITS = REPO / "outputs/wam_stability_20261021/pilot_audits"
SNAPSHOT = AUDITS / "complete24_verified.json"
DESTINATION = AUDITS / "diagnostics"


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(case):
    return f"{case['label']}_{case['variant']}_{case['record']}_s{case['sampling_seed']}"


def merge_hashes(target, incoming):
    for name, digest in incoming.items():
        path = str(Path(name).resolve())
        if path in target and target[path] != digest:
            raise ValueError(f"conflicting input SHA: {path}")
        target[path] = digest


def recheck(expected):
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"input SHA256 mismatch: {path}")


def pinned_json(path, expected):
    path = path.resolve()
    if str(path) not in expected:
        raise ValueError(f"plot input absent from verified snapshot/check files: {path}")
    raw = path.read_bytes()
    import hashlib

    if hashlib.sha256(raw).hexdigest() != expected[str(path)]:
        raise ValueError(f"plot input changed: {path}")
    return json.loads(raw)


def verified_snapshot():
    """No output is created until the actual audit schema and all hashes pass."""
    if not SNAPSHOT.is_file():
        raise FileNotFoundError(f"requires the complete24 verified snapshot: {SNAPSHOT}")
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    snapshot_digest = sha256(SNAPSHOT)
    snapshot = json.loads(SNAPSHOT.read_bytes())
    if not (
        snapshot["seed"] == 42
        and snapshot["expected"] == snapshot["verified"] == len(snapshot["cases"]) == 24
        and snapshot["complete"] is True
        and snapshot["full_pilot_matrix_verified"] is True
        and not snapshot["pending_not_counted_as_task_failure"]
        and not snapshot["execution_errors_excluded_from_success_denominators"]
    ):
        raise ValueError("partial/error snapshots cannot produce formal pilot diagnostics")
    expected_identities = {
        f"{label}_{variant}_{record}_s42" for label in LABELS for variant in VARIANTS for record in RECORDS
    }
    cases = {identity(case): case for case in snapshot["cases"]}
    if set(cases) != expected_identities or len(snapshot["cases"]) != len(cases):
        raise ValueError("verified snapshot is not the exact24 registered pilot identities")
    for label in LABELS:
        for variant in VARIANTS:
            count = snapshot["counts"][f"{label}/{variant}"]
            if not (count["expected"] == count["verified"] == 3 and count["pending_or_execution_error"] == 0):
                raise ValueError(f"incomplete registered cell: {label}/{variant}")
    # Actual audit schema: snapshot.input_sha256 and
    # snapshot.incremental_check_sha256; each incremental check has files_sha256.
    expected = {}
    merge_hashes(expected, snapshot["input_sha256"])
    merge_hashes(expected, snapshot["incremental_check_sha256"])
    recheck(expected)
    if len(snapshot["incremental_check_sha256"]) != 24:
        raise ValueError("requires exactly24 independently hashed case checks")
    audit_script = REPO / "scripts/audit_wam_stability_pilots.py"
    audit_digest = sha256(audit_script)
    checked_identities = set()
    for path in snapshot["incremental_check_sha256"]:
        check = pinned_json(Path(path), expected)
        case_identity = identity(check["verification"])
        if check["verification"] != cases[case_identity] or case_identity in checked_identities:
            raise ValueError(f"snapshot and incremental verification differ: {path}")
        if check["audit_script_sha256"] != audit_digest:
            raise ValueError(f"case check audit source differs: {path}")
        checked_identities.add(case_identity)
        merge_hashes(expected, check["files_sha256"])
    if checked_identities != expected_identities:
        raise ValueError("incremental checks do not cover the registered matrix")
    merge_hashes(expected, {str(SNAPSHOT): snapshot_digest, str(audit_script): audit_digest})
    recheck(expected)
    return snapshot, cases, expected


def case_arrays(label, variant, record, audited, expected):
    config = pinned_json(REPO / "configs" / f"wam_stability_pilot_{label}_20261021.json", expected)
    checkpoint = config["checkpoints"][0]["name"]
    case = REPO / config["output_dir"] / "simulation" / f"{checkpoint}_{variant}" / f"{record}_s42"
    manifest = pinned_json(case / "manifest.json", expected)
    result = pinned_json(case / "result.json", expected)
    chunks = pinned_json(case / "chunks.json", expected)
    raw_path = (case / "steps.jsonl").resolve()
    if str(raw_path) not in expected or sha256(raw_path) != expected[str(raw_path)]:
        raise ValueError(f"steps not pinned by the completed audit: {raw_path}")
    rows = [json.loads(line) for line in raw_path.read_bytes().splitlines()]
    if not (
        len(rows) == result["steps"] == result["expected_steps"] == audited["steps"] == 600
        and result["error"] is None
        and config["control_fps"] == 20
        and result["sampling_seed"] == 42
    ):
        raise ValueError(f"requires600 complete20Hz steps: {case}")
    if [row["step"] for row in rows] != list(range(600)):
        raise ValueError(f"step sequence differs: {case}")
    tool = pinned_json(Path(manifest["runtime_config"]["tool_contract"]), expected)
    h = np.asarray(tool["source_from_tcp"])
    actual = np.asarray([row["after_source"] for row in rows]) @ h
    desired = np.asarray([row["desired_tcp"] for row in rows])
    position = np.linalg.norm(desired[:, :3, 3] - actual[:, :3, 3], axis=1)
    angle = Rotation.from_matrix(desired[:, :3, :3] @ np.swapaxes(actual[:, :3, :3], 1, 2)).magnitude()
    logged = np.asarray([row["tcp_error"] for row in rows])
    np.testing.assert_allclose(position, np.linalg.norm(logged[:, :3], axis=1), atol=1e-10, rtol=0)
    np.testing.assert_allclose(angle, np.linalg.norm(logged[:, 3:], axis=1), atol=1e-10, rtol=0)
    dt = np.asarray([row["actual_dt"] for row in rows])
    np.testing.assert_allclose(dt, 0.05, atol=1e-9, rtol=0)
    endpoint_times = np.cumsum(dt)
    np.testing.assert_allclose(endpoint_times[-1], 30, atol=1e-9, rtol=0)
    before_times = np.r_[0.0, endpoint_times[:-1]]
    nominal_opening, phase = [], []
    for row in rows:
        chunk = chunks[row["chunk_index"]]
        offset = row["step"] - chunk["step"]
        if not 0 <= offset < len(chunk["openings"]):
            raise ValueError(f"nominal generated reference index outside its chunk: {case}")
        nominal_opening.append(chunk["openings"][offset])
        progress = row["reference_progress"]
        phase.append(progress["model_time_s"] if variant == "governor_only" else (offset + 1) / 20)
        if (progress is None) != (variant == "baseline"):
            raise ValueError(f"unexpected governor state: {case}")
    target_name = audited["target"]["name"]
    contact = np.asarray([row["objects"][target_name]["grasp_contact"] for row in rows], dtype=bool)
    success = np.asarray([row["success"] for row in rows], dtype=bool)
    if int(contact.sum()) != audited["target"]["native_bilateral_contact_steps"]:
        raise ValueError(f"contact recount differs: {case}")
    if (
        bool(success.any()) != audited["outcome"]["ever"]
        or bool(success[-1]) != audited["outcome"]["terminal"]
        or bool(success[-40:].all()) != audited["outcome"]["stable"]
    ):
        raise ValueError(f"native success recount differs: {case}")
    arrays = dict(
        time=endpoint_times,
        query_times=before_times[[chunk["step"] for chunk in chunks]],
        position_mm=1000 * position,
        angle=angle,
        generated_opening=np.asarray(nominal_opening),
        admitted_opening=np.asarray([row["opening_target"] for row in rows]),
        actual_opening=np.asarray([row["opening_after"] for row in rows]),
        phase=np.asarray(phase),
        scale=None
        if variant == "baseline"
        else np.asarray([row["reference_progress"]["tracking_scale"] for row in rows]),
        bridge=None if variant == "baseline" else np.asarray([row["reference_progress"]["bridging"] for row in rows]),
        height_mm=np.asarray([row["objects"][target_name]["height_gain_m"] for row in rows]) * 1000,
        contact=contact,
        success=success,
    )
    summary = dict(
        label=label,
        variant=variant,
        record=record,
        task=manifest["record"]["task"],
        target=target_name,
        steps=600,
        physical_duration_s=float(endpoint_times[-1]),
        native_contact_endpoint_count=int(contact.sum()),
        native_success_endpoint_count=int(success.sum()),
        native_outcome=audited["outcome"],
        first_contact_endpoint_time_s=None if not contact.any() else float(endpoint_times[contact][0]),
        first_success_endpoint_time_s=None if not success.any() else float(endpoint_times[success][0]),
        query_pre_step_times_s=arrays["query_times"].tolist(),
        generated_opening_nominal_index="step - active_chunk.step; not the admitted governor phase",
    )
    return arrays, summary


def plot_setting(label, cases, destination):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(8, 3, figsize=(19, 23), sharex=True, constrained_layout=True)
    for column, record in enumerate(RECORDS):
        for variant in VARIANTS:
            data, summary = cases[(label, variant, record)]
            color, name, times = COLORS[variant], DISPLAY[variant], data["time"]
            axes[0, column].plot(times, data["position_mm"], color=color, label=name, linewidth=1.1)
            axes[1, column].plot(times, data["angle"], color=color, label=name, linewidth=1.1)
            opening_axis = axes[2 if variant == "baseline" else 3, column]
            opening_axis.plot(
                times, data["actual_opening"], color=color, linewidth=1.4, label="actual at post-step endpoint"
            )
            opening_axis.plot(
                times,
                data["generated_opening"],
                color=color,
                linewidth=1.1,
                linestyle=":",
                alpha=0.6,
                label="generated at nominal model index",
            )
            opening_axis.plot(
                times,
                data["admitted_opening"],
                color=color,
                linewidth=1.1,
                linestyle="--",
                label="admitted desired for this step",
            )
            opening_axis.set_ylim(-0.05, 1.05)
            opening_axis.legend(fontsize=7, loc="best")
            axes[4, column].plot(
                times,
                data["phase"],
                color=color,
                linewidth=1.1,
                label=f"{name}: {'nominal' if variant == 'baseline' else 'admitted'} model phase",
            )
            axes[4, column].scatter(
                data["query_times"],
                np.full(len(data["query_times"]), 2.15),
                color=color,
                marker="v",
                s=15,
                label=f"{name}: query before step",
            )
            axes[6, column].plot(times, data["height_mm"], color=color, linewidth=1.1, label=f"{name}: height gain")
            axes[6, column].scatter(
                times[data["contact"]],
                data["height_mm"][data["contact"]],
                color=color,
                edgecolors="black",
                linewidths=0.35,
                marker="o",
                s=12,
                label=f"{name}: native bilateral contact endpoints",
            )
            offset = 0 if variant == "baseline" else 2
            axes[7, column].scatter(times, offset + data["success"].astype(float), color=color, marker=".", s=5)
        governed = cases[(label, "governor_only", record)][0]
        scale_axis = axes[4, column].twinx()
        scale_axis.plot(
            governed["time"],
            governed["scale"],
            color=COLORS["governor_only"],
            alpha=0.4,
            linestyle=":",
            linewidth=1,
            label="v1 tracking scale",
        )
        scale_axis.set(ylim=(-0.05, 1.1), ylabel="v1 scale [0,1]")
        scale_axis.tick_params(axis="y", colors=COLORS["governor_only"])
        axes[4, column].set_ylim(-0.05, 2.3)
        handles, names = axes[4, column].get_legend_handles_labels()
        other_handles, other_names = scale_axis.get_legend_handles_labels()
        axes[4, column].legend(handles + other_handles, names + other_names, fontsize=6.5, loc="best")
        axes[5, column].scatter(
            governed["time"], governed["bridge"].astype(float), color=COLORS["governor_only"], marker=".", s=6
        )
        axes[5, column].set(yticks=[0, 1], yticklabels=["path", "bridge"], ylim=(-0.2, 1.2))
        axes[7, column].set(
            yticks=[0, 1, 2, 3],
            yticklabels=["B false", "B true", "G false", "G true"],
            ylim=(-0.3, 3.3),
            xlabel="physical post-step endpoint time (s)",
        )
        title = cases[(label, "baseline", record)][1]["task"]
        axes[0, column].set_title(f"{record}\n{title}", fontsize=10)
        labels = (
            "canonical TCP error (mm)",
            "SO(3) error (rad)",
            "baseline opening [0,1]",
            "original v1 governor opening [0,1]",
            "model phase since query (s)",
            "v1 state at endpoint; baseline N/A",
            "target height gain (mm) / native contact",
            "native task-success predicate at endpoint",
        )
        for row, ylabel in enumerate(labels):
            ax = axes[row, column]
            ax.set(xlim=(0, 30), ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.axvspan(28, 30, color="grey", alpha=0.08)
        for row in (0, 1, 6):
            axes[row, column].legend(fontsize=6.5, loc="best")
    figure.suptitle(
        f"{label}: complete verified24-case pilot; baseline blue / original governor_only orange; seed42\n"
        "Post-step TCP / aperture / native predicates; triangles = pre-step queries. Contact circles and success dots are endpoint samples, not continuous events.\n"
        "Generated opening follows nominal model index; admitted opening is the reference executed. Grey28–30s = last40 control intervals, sampled at their ends. Inference latency excluded; old init0 development scenes.",
        fontsize=10,
    )
    paths = []
    for suffix in ("png", "pdf"):
        path = destination / f"{label}_pilot_diagnostics.{suffix}"
        figure.savefig(path, dpi=150)
        paths.append(path)
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU plotting requires explicit CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=''")
    snapshot, audited_cases, expected = verified_snapshot()
    source_digest = sha256(__file__)
    data, summaries = {}, []
    for label in LABELS:
        for variant in VARIANTS:
            for record in RECORDS:
                key = f"{label}_{variant}_{record}_s42"
                arrays, summary = case_arrays(label, variant, record, audited_cases[key], expected)
                data[(label, variant, record)] = (arrays, summary)
                summaries.append(summary)
    recheck(expected)
    import matplotlib

    matplotlib.use("Agg")
    DESTINATION.mkdir(exist_ok=False)
    output_paths = []
    for label in LABELS:
        output_paths.extend(plot_setting(label, data, DESTINATION))
    diagnostics = DESTINATION / "diagnostics.json"
    write_new(
        diagnostics,
        dict(
            seed=args.seed,
            cases=summaries,
            verified_cases=24,
            physical_steps=14400,
            contact_and_success="native simulator predicates at refreshed post-step endpoints; no recomputation of contact physics",
            limitations="Old init0 development pilots. Diagnostics do not establish unseen-scene acceptance, continuous contact duration or a unique causal failure attribution.",
        ),
    )
    output_paths.append(diagnostics)
    recheck(expected)
    if sha256(__file__) != source_digest:
        raise ValueError("plot source changed during execution")
    manifest = dict(
        seed=args.seed,
        source_sha256={str(Path(__file__).resolve()): source_digest},
        snapshot=str(SNAPSHOT),
        snapshot_sha256=sha256(SNAPSHOT),
        snapshot_utc=snapshot["snapshot_utc"],
        input_sha256=expected,
        output_sha256={str(path): sha256(path) for path in output_paths},
        complete24_required=True,
        all_input_hashes_rechecked_before_and_after=True,
        parameters=dict(
            labels=LABELS,
            records=RECORDS,
            variants=VARIANTS,
            colors=COLORS,
            rows=8,
            columns=3,
            figure_inches=[19, 23],
            dpi=150,
            endpoint_time="cumulative actual_dt; predicates are discrete post-step samples",
            query_time="cumulative actual_dt before the chunk's query step",
            generated_opening="active chunk at nominal step-minus-query index",
            admitted_opening="saved opening_target used for the executed step",
            phase="baseline nominal index; original governor stored model_time_s",
            tracking_scale_and_bridge="original governor only; baseline not applicable",
        ),
        runtime=dict(
            python=platform.python_version(),
            **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib")},
        ),
    )
    write_new(DESTINATION / "manifest.json", manifest)
    print(
        json.dumps(
            dict(
                complete=True,
                figures=8,
                verified_cases=24,
                destination=str(DESTINATION),
                manifest_sha256=sha256(DESTINATION / "manifest.json"),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
