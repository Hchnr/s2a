"""Final-only017 window diagnostics and exportable figures; no model or physics."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, write_new
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO, sha256

TARGETS = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}
TITLES = {8: "Bowl to plate", 9: "Wine bottle to rack", 6: "Cream cheese to bowl"}


def outcome(flags):
    assert len(flags) >= 40 and all(type(flag) is bool for flag in flags)
    trailing = next((i for i, flag in enumerate(reversed(flags)) if not flag), len(flags))
    return dict(
        ever=any(flags),
        terminal=flags[-1],
        stable=all(flags[-40:]),
        successful_steps=sum(flags),
        trailing_success_steps=trailing,
    )


def distribution(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    return dict(
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(np.max(values)),
    )


def summarize(case, evidence):
    directory = Path(case["case"])
    manifest = evidence.json(directory / "manifest.json")
    result = evidence.json(directory / "result.json")
    template = evidence.json(manifest["experiment"]["policy_template"])
    h = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"], dtype=float)
    rows = [json.loads(line) for line in evidence.read(directory / "steps.jsonl").splitlines()]
    assert len(rows) == case["steps"] == 1200 and case["chunks"] == 150
    assert case["raw_checkpoint_evidence"] and not case["component_ablation"]
    actual = np.asarray([r["after_source"] for r in rows]) @ h
    desired = np.asarray([r["desired_tcp"] for r in rows])
    pos = np.linalg.norm(desired[:, :3, 3] - actual[:, :3, 3], axis=1)
    rot = Rotation.from_matrix(desired[:, :3, :3] @ np.transpose(actual[:, :3, :3], (0, 2, 1))).magnitude()
    logged = np.asarray([r["tcp_error"] for r in rows])
    np.testing.assert_allclose(pos, np.linalg.norm(logged[:, :3], axis=1), atol=1e-12, rtol=0)
    np.testing.assert_allclose(rot, np.linalg.norm(logged[:, 3:], axis=1), atol=1e-12, rtol=0)
    opening = np.asarray([r["opening_after"] for r in rows])
    reference_opening = np.asarray([r["opening_target"] for r in rows])
    opening_error = np.abs(reference_opening - opening)
    full = case["tracking"]["admitted"]
    np.testing.assert_allclose(
        [np.sqrt(np.mean(pos**2)), np.sqrt(np.mean(rot**2)), np.mean(opening_error)],
        [full["position_rmse_m"], full["so3_rmse_rad"], result["opening_mae"]],
        atol=1e-12,
        rtol=0,
    )
    flags = [r["success"] for r in rows]
    assert outcome(flags) == case["outcome"]
    assert outcome(flags[:600]) == case["first600_outcome"]
    assert outcome(flags[600:]) == case["last600_outcome"]
    assert case["first600"]["all600_shared_rows_exact"]
    assert case["first600"]["old600_outcome"] == outcome(flags[:600])
    origin = rows[0]["physical_time_before_s"]
    elapsed = np.asarray([r["physical_time_after_s"] - origin for r in rows])
    np.testing.assert_allclose(elapsed, np.arange(1, 1201) / 20, atol=1e-9, rtol=0)
    intervals = []
    for i, flag in enumerate(flags):
        if flag and (i == 0 or not flags[i - 1]):
            intervals.append(dict(start_step=i, end_step_exclusive=i + 1))
        elif flag:
            intervals[-1]["end_step_exclusive"] = i + 1
    for interval in intervals:
        interval["first_endpoint_elapsed_s"] = float(elapsed[interval["start_step"]])
        interval["last_endpoint_elapsed_s"] = float(elapsed[interval["end_step_exclusive"] - 1])
    task_id = manifest["record"]["task_id"]
    target = TARGETS[task_id]
    target_positions = np.asarray([r["objects"][target]["position"] for r in rows])
    contact = np.asarray([r["objects"][target]["grasp_contact"] for r in rows], dtype=bool)
    height = target_positions[:, 2] - target_positions[0, 2]
    geometry = np.linalg.norm(desired[:, :2, 3] - target_positions[:, :2], axis=1)
    windows = {}
    for label, start, stop in (
        ("original_0_to_30s", 0, 600),
        ("additional_30_to_60s", 600, 1200),
        ("all_60s", 0, 1200),
    ):
        index = slice(start, stop)
        windows[label] = dict(
            start_step=start,
            stop_step_exclusive=stop,
            steps=stop - start,
            outcome=outcome(flags[start:stop]),
            admitted_tracking=dict(
                position_rmse_m=float(np.sqrt(np.mean(pos[index] ** 2))),
                so3_rmse_rad=float(np.sqrt(np.mean(rot[index] ** 2))),
                opening_mae=float(np.mean(opening_error[index])),
                position_errors_m=distribution(pos[index]),
                rotation_errors_rad=distribution(rot[index]),
            ),
            saturated_steps=sum(r["saturated"] for r in rows[index]),
            rate_limited_steps=sum(r["rate_limited"] for r in rows[index]),
            target_dual_contact_steps=int(np.sum(contact[index])),
            maximum_target_height_above_first_endpoint_m=float(np.max(height[index])),
            final_target_height_above_first_endpoint_m=float(height[stop - 1]),
            descriptive_reference_to_target_center_xy_m=distribution(geometry[index]),
        )
    value = dict(
        record=case["record"],
        case=case["case"],
        task_id=task_id,
        task=TITLES[task_id],
        windows=windows,
        native_success_intervals=intervals,
        old600_outcome_preserved=True,
        first_success_endpoint_s=float(elapsed[flags.index(True)]) if any(flags) else None,
        longest_native_success_duration_s=max(
            ((x["end_step_exclusive"] - x["start_step"]) / 20 for x in intervals), default=0.0
        ),
        final_40_native_predicates=flags[-40:],
    )
    return value, dict(
        title=TITLES[task_id],
        elapsed=elapsed,
        pos=pos,
        rot=rot,
        opening=opening,
        reference_opening=reference_opening,
        height=height,
        contact=contact,
        flags=flags,
        intervals=intervals,
    )


def plot_all(traces, output):
    # Plotting caches and every export stay inside this new artifact directory.
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(17, 9), sharex=True, layout="constrained")
    for row, data in enumerate(traces):
        t = data["elapsed"]
        for ax in axes[row]:
            ax.axvline(30, color="black", ls="--", lw=1, label="Original 30 s cutoff")
            ax.axvspan(58, 60, color="gray", alpha=0.12, label="Final 40 endpoints")
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 60)
        axes[row, 0].plot(t, data["pos"] * 1000, color="tab:blue", lw=0.9)
        axes[row, 0].set_ylabel(data["title"] + "\nTCP error (mm)")
        axes[row, 1].plot(t, data["rot"], color="tab:orange", lw=0.9)
        axes[row, 1].set_ylabel("SO(3) error (rad)")
        axes[row, 2].plot(t, data["reference_opening"], label="Reference", color="tab:green", lw=1)
        axes[row, 2].plot(t, data["opening"], label="Measured", color="tab:blue", lw=0.9, alpha=0.85)
        axes[row, 2].set_ylabel("Gripper aperture [0, 1]")
        axes[row, 2].set_ylim(-0.04, 1.04)
        axes[row, 3].plot(t, data["height"] * 1000, color="tab:purple", lw=1, label="Target height")
        axes[row, 3].scatter(
            t[data["contact"]], data["height"][data["contact"]] * 1000, s=5, color="black", label="Dual contact"
        )
        for interval in data["intervals"]:
            axes[row, 3].axvspan(
                interval["first_endpoint_elapsed_s"], interval["last_endpoint_elapsed_s"], color="green", alpha=0.18
            )
        axes[row, 3].set_ylabel("Target height above first endpoint (mm)")
        axes[row, 3].text(
            0.02, 0.95, "Green band = native success", transform=axes[row, 3].transAxes, va="top", fontsize=8
        )
    axes[0, 0].legend(loc="upper left", fontsize=7)
    axes[0, 2].legend(loc="best", fontsize=8)
    axes[0, 3].legend(loc="lower right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Physical elapsed time (s); inference wait excluded")
    fig.suptitle("017 raw WAM: all three old scenes, fixed 60 s budget", fontsize=14)
    paths = [output / "long_raw_tracking_and_progress.png", output / "long_raw_tracking_and_progress.pdf"]
    for path in paths:
        fig.savefig(path, dpi=170)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    output = args.output.resolve()
    assert args.audit.resolve().is_relative_to(REPO / "outputs")
    assert output.is_relative_to(REPO / "outputs") and output != REPO / "outputs" and not output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(args.audit, args.audit_sha256)
    audit = evidence.json(args.audit)
    assert audit["schema_version"] == "s2a.long-raw-readonly-wam-audit.v1"
    assert audit["complete"] and audit["all_cases_independently_verified"] and audit["verified_cases"] == 3
    assert audit["ready_binding"]["all12_historical_process_identities_verified"]
    assert sum(x["exact_requests"] for x in audit["final_http_and_directory_budget"]) == 450
    assert all(r["owned_service_released_normally"] for r in audit["resources"])
    for name, digest in audit["input_sha256"].items():
        evidence.check(name, digest)
    cases = sorted(audit["cases"], key=lambda case: case["partition"])
    assert [case["record"] for case in cases] == [f"task{i}_init0" for i in range(3)]
    summaries, traces = zip(*(summarize(case, evidence) for case in cases), strict=True)
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    figures = plot_all(traces, output)
    evidence.recheck()
    result = dict(
        schema_version="s2a.long-raw-final-window-diagnostics.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        wam_trials=3,
        physics_steps=3600,
        h40_queries=450,
        cases=summaries,
        source_audit=dict(path=str(args.audit.resolve()), sha256=args.audit_sha256),
        input_sha256=evidence.hashes,
        output_sha256={str(path): sha256(path) for path in figures},
        new_model_requests=0,
        new_physics_steps=0,
        limitations=[
            "Old development scenes only; no new initial-state performance or real robot.",
            "Contact/height and target-center geometry are retrospective diagnostics, never controller inputs.",
            "Native predicates define success; contact, height and tracking error do not replace them.",
            "The separate 30 s and 60 s outcomes retain the original 600-step verdicts.",
        ],
    )
    path = output / "diagnostics.json"
    write_new(path, result)
    print(json.dumps(dict(path=str(path), sha256=sha256(path), complete=True)))


if __name__ == "__main__":
    main()
