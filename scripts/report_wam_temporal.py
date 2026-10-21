"""Final-only020 state tracking, causal-reference coverage and native outcomes.

Reads authenticated completed logs only. No model, HTTP, environment or physics
is started. Changing a reference is not a raw-checkpoint score or causal proof.
"""

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

AUDITOR = REPO / "scripts/audit_wam_temporal_wam.py"
AUDITOR_SHA = "441ee4855187550d1bc5188f23b0b5e1431cdbbfb3a4832074d3b3df905eca49"
BASELINE = (
    REPO / "outputs/wam_stability_20261021/short_replan_audit/final12_verified_with_unused_checkpoint_exception.json"
)
BASELINE_SHA = "a4560bb70cc21e204dc7bd0e35fc3aa2a7cab7365b394a6a4c1727473b60ad12"
INTERVENTION = "causal_temporal_equal_ensemble"
TARGETS = {8: "akita_black_bowl_1", 9: "wine_bottle_1", 6: "cream_cheese_1"}
TITLES = {8: "Bowl to plate", 9: "Wine bottle to rack", 6: "Cream cheese to bowl"}


def outcome(flags):
    assert len(flags) == 600 and all(type(flag) is bool for flag in flags)
    trailing = 0
    for flag in reversed(flags):
        if not flag:
            break
        trailing += 1
    return dict(
        ever=any(flags),
        terminal=flags[-1],
        stable=all(flags[-40:]),
        successful_steps=sum(flags),
        trailing_success_steps=trailing,
    )


def intervals(flags, elapsed):
    result = []
    for step, flag in enumerate(flags):
        if flag and (step == 0 or not flags[step - 1]):
            result.append(dict(start_step=step, end_step_exclusive=step + 1))
        elif flag:
            result[-1]["end_step_exclusive"] = step + 1
    for interval in result:
        interval["first_endpoint_elapsed_s"] = float(elapsed[interval["start_step"]])
        interval["last_endpoint_elapsed_s"] = float(elapsed[interval["end_step_exclusive"] - 1])
        interval["duration_s"] = (interval["end_step_exclusive"] - interval["start_step"]) / 20
    return result


def distribution(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    return dict(median=float(np.median(values)), p95=float(np.quantile(values, 0.95)), maximum=float(np.max(values)))


def errors(desired, actual):
    pos = np.linalg.norm(desired[:, :3, 3] - actual[:, :3, 3], axis=1)
    rot = Rotation.from_matrix(desired[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)).magnitude()
    assert np.isfinite(pos).all() and np.isfinite(rot).all()
    return pos, rot


def tracking(pos, rot, opening):
    return dict(
        position_rmse_m=float(np.sqrt(np.mean(pos**2))),
        so3_rmse_rad=float(np.sqrt(np.mean(rot**2))),
        opening_mae=float(np.mean(np.abs(opening))),
        opening_rmse=float(np.sqrt(np.mean(opening**2))),
        position_errors_m=distribution(pos),
        rotation_errors_rad=distribution(rot),
    )


def validate_complete(audit):
    """Reject actual partial receipts before reading live case logs or exporting."""
    assert audit["schema_version"] == "s2a.temporal-equal-wam-independent-audit.v1"
    assert audit["complete"] and audit["all_cases_independently_verified"] and audit["normal_process_release"], (
        "final-only reporter requires complete independently verified and released evidence"
    )
    assert not audit["errors"] and audit["raw_checkpoint_evidence"] is False
    assert audit["reference_intervention"] == INTERVENTION
    assert audit["registered_budget"] == dict(
        wam_trials=3, physics_steps=1800, maximum_h40_queries=225, fresh_initial_states=0
    )
    assert (
        audit["verified_cases"],
        audit["verified_physics_steps"],
        audit["verified_queries"],
        audit["verified_http_queries"],
    ) == (3, 1800, 225, 225)
    assert len(audit["cases"]) == len(audit["resources"]) == 3
    assert [c["partition"] for c in audit["cases"]] == [0, 1, 2]
    assert [c["record"] for c in audit["cases"]] == [f"task{i}_init0" for i in range(3)]
    identities = []
    for part, (case, resource) in enumerate(zip(audit["cases"], audit["resources"], strict=True)):
        assert case["status"] == "verified" and resource["partition"] == part
        assert resource["owned_service_released_normally"] and resource["recorded_error"] is None
        assert (resource["physical_gpu"], resource["port"]) == (4 + part, 9844 + part)
        assert resource["final_http_budget"]["successful_http_requests"] == 75
        assert resource["final_http_budget"]["no_extra_post"]
        assert set(resource["processes"]) == {"coordinator", "model", "matrix", "physics"}
        for process in resource["processes"].values():
            assert process["present"] is False and not Path(f"/proc/{process['pid']}").exists()
            identities.append(process["pid"])
        result = case["result"]
        assert (result["steps"], result["chunks"]) == (600, 75)
        assert result["status"] == "verified" and result["raw_checkpoint_evidence"] is False
        assert result["reference_intervention"] == INTERVENTION
        assert result["independent_temporal_ensemble_replay"]["passed"]
        assert result["first_query020_provenance"]["first_query_full40x128_raw_exact"]
    assert len(identities) == len(set(identities)) == 12


def reference_coverage(rows, chunks):
    """Independently read stored world paths; integer endpoints, no new targets."""
    assert [c["step"] for c in chunks] == list(range(0, 600, 8))
    assert all(len(c["raw128"]) == 40 for c in chunks)
    paths = {c["step"]: np.asarray(c["raw_targets_tcp"], dtype=float) for c in chunks}
    openings = {c["step"]: np.asarray(c["openings"], dtype=float) for c in chunks}
    assert all(v.shape == (40, 4, 4) for v in paths.values())
    assert all(v.shape == (40,) for v in openings.values())
    used = {q: set() for q in paths}
    exposure = {q: 0.0 for q in paths}
    counts, weights, identity_steps = [], [], 0
    for step, row in enumerate(rows):
        log = row["temporal_ensemble"]
        eligible = [q for q in paths if q <= step < q + 40]
        pairs = [(q, step - q) for q in eligible]
        contributors = log["contributors"]
        assert [(v["query_step"], v["endpoint_index"]) for v in contributors] == pairs
        count = len(pairs)
        assert 1 <= count <= 5 and log["contributor_count"] == count
        assert log["physical_step"] == step and log["absolute_endpoint"] == step + 1
        assert log["equal_weight"] == 1 / count
        poses, aperture = [], []
        for item, (query, index) in zip(contributors, pairs, strict=True):
            assert item["absolute_endpoint"] == query + index + 1 == step + 1
            assert item["weight"] == 1 / count
            np.testing.assert_allclose(item["target_tcp"], paths[query][index], atol=1e-10, rtol=0)
            np.testing.assert_allclose(item["opening"], openings[query][index], atol=1e-10, rtol=0)
            poses.append(paths[query][index])
            aperture.append(openings[query][index])
            used[query].add(index)
            exposure[query] += 1 / count
        np.testing.assert_allclose(row["raw_desired_tcp"], poses[-1], atol=1e-10, rtol=0)
        np.testing.assert_allclose(row["raw_opening_at_model_time"], aperture[-1], atol=1e-10, rtol=0)
        pose_array = np.asarray(poses)
        aperture_array = np.asarray(aperture)
        identity = (
            np.max(np.abs(pose_array - poses[-1])) <= 1e-12
            and np.max(np.abs(aperture_array - aperture_array[-1])) <= 1e-12
        )
        assert log["identity_fast_path"] == bool(identity)
        if identity:
            target, opening = poses[-1], aperture[-1]
            identity_steps += 1
        else:
            mean = np.mean(pose_array[:, :3, :3], axis=0)
            u, singular, vt = np.linalg.svd(mean)
            assert np.linalg.det(mean) > 0 and min(singular) >= 1e-6
            target = np.eye(4)
            target[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
            target[:3, 3] = np.mean(pose_array[:, :3, 3], axis=0)
            opening = float(np.mean(aperture))
        np.testing.assert_allclose(row["desired_tcp"], target, atol=1e-10, rtol=0)
        np.testing.assert_allclose(row["opening_target"], opening, atol=1e-10, rtol=0)
        counts.append(count)
        weights.append(1 / count)
    per_query = []
    for query in paths:
        assert used[query] == set(range(min(40, 600 - query)))
        per_query.append(
            dict(
                query_step=query,
                generated_endpoints=40,
                recorded_contributor_endpoints=len(used[query]),
                unavailable_after_physical_budget=40 - len(used[query]),
                equal_weight_exposure=exposure[query],
            )
        )
    contributed = sum(len(value) for value in used.values())
    assert contributed == 2920 and abs(sum(exposure.values()) - 600) <= 1e-10
    result = dict(
        generated_endpoints=3000,
        recorded_contributor_endpoints=contributed,
        contributor_endpoint_fraction=contributed / 3000,
        physical_targets=600,
        unavailable_after_physical_budget=3000 - contributed,
        per_query=per_query,
        identity_fast_path_steps=identity_steps,
        contributor_count_histogram={str(n): counts.count(n) for n in range(1, 6)},
        scope="Reference endpoint participation, not physical-step count, accuracy or task success. Old newest-query cursor discard is not ensemble suffix discard.",
    )
    return result, np.asarray(counts), np.asarray(weights)


def summarize(case, evidence, *, ensemble):
    directory = (REPO / case["case"]).resolve()
    manifest = evidence.json(directory / "manifest.json")
    result = evidence.json(directory / "result.json")
    template = evidence.json(manifest["experiment"]["policy_template"])
    h = np.asarray(evidence.json(template["tool_contract"])["source_from_tcp"], dtype=float)
    rows = [json.loads(line) for line in evidence.read(directory / "steps.jsonl").splitlines()]
    assert len(rows) == case["steps"] == 600 and case["chunks"] == 75
    assert [row["step"] for row in rows] == list(range(600))
    assert result["source"] == "wam" and result["sampling_seed"] == 42
    assert result["complete"] and result["error"] is None
    assert manifest["raw_checkpoint_reference"] is (not ensemble)
    assert manifest["reference_intervention"] == (INTERVENTION if ensemble else "raw")
    actual = np.asarray([row["after_source"] for row in rows]) @ h
    admitted = np.asarray([row["desired_tcp"] for row in rows])
    raw = np.asarray([row["raw_desired_tcp"] for row in rows])
    opening = np.asarray([row["opening_after"] for row in rows])
    admitted_open = np.asarray([row["opening_target"] for row in rows])
    raw_open = np.asarray([row["raw_opening_at_model_time"] for row in rows])
    admitted_pos, admitted_rot = errors(admitted, actual)
    raw_pos, raw_rot = errors(raw, actual)
    summaries = dict(
        admitted=tracking(admitted_pos, admitted_rot, admitted_open - opening),
        latest_raw_model_clock=tracking(raw_pos, raw_rot, raw_open - opening),
    )
    for name, audit_name in (("admitted", "admitted"), ("latest_raw_model_clock", "raw_at_model_clock")):
        for metric in ("position_rmse_m", "so3_rmse_rad"):
            np.testing.assert_allclose(
                summaries[name][metric], case["tracking"][audit_name][metric], atol=1e-12, rtol=0
            )
    np.testing.assert_allclose(summaries["admitted"]["opening_mae"], result["opening_mae"], atol=1e-12, rtol=0)
    if ensemble:
        np.testing.assert_allclose(
            summaries["admitted"]["opening_mae"], case["opening_tracking"]["mae"], atol=1e-12, rtol=0
        )
    flags = [row["success"] for row in rows]
    native = outcome(flags)
    assert native == case["outcome"] == result["task_outcome"]
    elapsed = np.asarray([row["physical_time_after_s"] - rows[0]["physical_time_before_s"] for row in rows])
    np.testing.assert_allclose(elapsed, np.arange(1, 601) / 20, atol=1e-9, rtol=0)
    task_id = manifest["record"]["task_id"]
    target = TARGETS[task_id]
    object_positions = np.asarray([row["objects"][target]["position"] for row in rows])
    contact = [row["objects"][target]["grasp_contact"] for row in rows]
    assert all(type(value) is bool for value in contact)
    height = object_positions[:, 2] - object_positions[0, 2]
    reference_pos_change, reference_rot_change = errors(admitted, raw)
    native_intervals = intervals(flags, elapsed)
    value = dict(
        record=case["record"],
        case=str(directory),
        task=manifest["record"]["task"],
        task_id=task_id,
        source="wam",
        sampling_seed=42,
        physical_steps=600,
        h40_queries=75,
        raw_checkpoint_evidence=not ensemble,
        reference_intervention=INTERVENTION if ensemble else "raw",
        tracking=summaries,
        native_outcome=native,
        native_success_intervals=native_intervals,
        first_success_endpoint_s=float(elapsed[flags.index(True)]) if any(flags) else None,
        longest_native_success_duration_s=max((v["duration_s"] for v in native_intervals), default=0.0),
        final_40_native_predicates=flags[-40:],
        retrospective_object_progress=dict(
            target=target,
            dual_contact_steps=sum(contact),
            dual_contact_intervals=intervals(contact, elapsed),
            maximum_height_above_first_endpoint_m=float(np.max(height)),
            final_height_above_first_endpoint_m=float(height[-1]),
            final_dual_contact=contact[-1],
            admitted_reference_to_object_center_xy_m=distribution(
                np.linalg.norm(admitted[:, :2, 3] - object_positions[:, :2], axis=1)
            ),
            latest_raw_reference_to_object_center_xy_m=distribution(
                np.linalg.norm(raw[:, :2, 3] - object_positions[:, :2], axis=1)
            ),
        ),
        constraints=dict(
            saturated_steps=sum(row["saturated"] for row in rows),
            rate_limited_steps=sum(row["rate_limited"] for row in rows),
        ),
        admitted_minus_latest_raw=dict(
            position_m=distribution(reference_pos_change),
            rotation_rad=distribution(reference_rot_change),
            opening_absolute=distribution(np.abs(admitted_open - raw_open)),
        ),
    )
    if ensemble:
        coverage, counts, weights = reference_coverage(rows, evidence.json(directory / "chunks.json"))
        previous = case["ensemble_reference_usage"]
        for name in (
            "generated_endpoints",
            "recorded_contributor_endpoints",
            "physical_targets",
            "unavailable_after_physical_budget",
        ):
            assert coverage[name] == previous[name]
        assert coverage["per_query"] == previous["per_query"]
        value["ensemble_reference_usage"] = coverage
    else:
        counts, weights = None, None
    return value, dict(
        title=TITLES[task_id],
        elapsed=elapsed,
        admitted_pos=admitted_pos,
        admitted_rot=admitted_rot,
        raw_pos=raw_pos,
        raw_rot=raw_rot,
        opening=opening,
        admitted_open=admitted_open,
        raw_open=raw_open,
        height=height,
        contact=np.asarray(contact),
        flags=flags,
        intervals=native_intervals,
        counts=counts,
        weights=weights,
    )


def group(cases):
    assert len(cases) == 3
    return {
        key: dict(
            successes=sum(c["native_outcome"][key] for c in cases),
            trials=3,
            rate=sum(c["native_outcome"][key] for c in cases) / 3,
        )
        for key in ("ever", "terminal", "stable")
    }


def baseline(evidence):
    evidence.check(BASELINE, BASELINE_SHA)
    audit = evidence.json(BASELINE)
    assert audit["complete"] and audit["all12_cases_independently_verified"]
    selected = [c for c in audit["cases"] if c["profile"] == "interval8_alpha1"]
    assert [c["record"] for c in selected] == [f"task{i}_init0" for i in range(3)]
    summaries = []
    for case in selected:
        folder = REPO / case["case"]
        for name in ("steps.jsonl", "manifest.json", "result.json"):
            path = (folder / name).resolve()
            evidence.check(path, audit["source_before_after_hashes"][str(path)])
        summary, _ = summarize(case, evidence, ensemble=False)
        summaries.append(summary)
    return dict(
        source_audit=dict(path=str(BASELINE), sha256=BASELINE_SHA),
        cases=summaries,
        group_outcome=group(summaries),
        original_coordinator_final_gate_success=audit["original_coordinator_final_gate_success"],
        comparison_scope="Different closed-loop strategies on the same old initial states/seed42; first-query equality only. Subsequent observations differ and these are not paired identical trajectories or causal attribution.",
    )


def plot_all(traces, output):
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(21, 9), sharex=True, layout="constrained")
    for row, data in enumerate(traces):
        elapsed = data["elapsed"]
        for ax in axes[row]:
            ax.axvspan(28, 30, color="gray", alpha=0.12)
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 30)
        axes[row, 0].plot(elapsed, data["raw_pos"] * 1000, color="tab:orange", lw=0.8, label="Latest raw")
        axes[row, 0].plot(elapsed, data["admitted_pos"] * 1000, color="tab:blue", lw=0.8, label="Ensemble admitted")
        axes[row, 0].set_ylabel(data["title"] + "\nTCP error (mm)")
        axes[row, 1].plot(elapsed, data["raw_rot"], color="tab:orange", lw=0.8, label="Latest raw")
        axes[row, 1].plot(elapsed, data["admitted_rot"], color="tab:blue", lw=0.8, label="Ensemble admitted")
        axes[row, 1].set_ylabel("SO(3) error (rad)")
        axes[row, 2].plot(elapsed, data["raw_open"], color="tab:orange", lw=0.8, label="Latest raw")
        axes[row, 2].plot(elapsed, data["admitted_open"], color="tab:green", lw=0.8, label="Ensemble admitted")
        axes[row, 2].plot(elapsed, data["opening"], color="tab:blue", lw=0.8, label="Measured")
        axes[row, 2].set_ylabel("Gripper aperture [0, 1]")
        axes[row, 2].set_ylim(-0.04, 1.04)
        axes[row, 3].plot(elapsed, data["height"] * 1000, color="tab:purple", lw=0.8, label="Target height")
        axes[row, 3].scatter(
            elapsed[data["contact"]], data["height"][data["contact"]] * 1000, s=5, color="black", label="Dual contact"
        )
        for interval in data["intervals"]:
            axes[row, 3].axvspan(
                interval["first_endpoint_elapsed_s"], interval["last_endpoint_elapsed_s"], color="green", alpha=0.2
            )
        axes[row, 3].set_ylabel("Height above first endpoint (mm)")
        axes[row, 3].text(
            0.02, 0.96, "Green band = native success", transform=axes[row, 3].transAxes, va="top", fontsize=7
        )
        axes[row, 4].step(elapsed, data["counts"], where="post", color="tab:blue", label="Contributors")
        axes[row, 4].set_ylabel("Contributor count")
        axes[row, 4].set_yticks(range(1, 6))
        axes[row, 4].set_ylim(0.8, 5.2)
        weight_axis = axes[row, 4].twinx()
        weight_axis.step(elapsed, data["weights"], where="post", color="tab:orange", ls="--", label="Weight 1/n")
        weight_axis.set_ylim(0.15, 1.05)
        weight_axis.set_ylabel("Equal weight 1/n", color="tab:orange")
        axes[row, 4].text(
            0.04,
            0.53,
            "2920/3000 reference endpoints\n600 physical targets",
            transform=axes[row, 4].transAxes,
            fontsize=7,
        )
    for col in (0, 2, 3):
        axes[0, col].legend(loc="best", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Physical elapsed time (s); inference wait excluded")
    fig.suptitle(
        "020 causal temporal equal ensemble: three old scenes, fixed 30 s; shaded final40 endpoints", fontsize=13
    )
    paths = [output / "temporal_tracking_and_progress.png", output / "temporal_tracking_and_progress.pdf"]
    for path in paths:
        fig.savefig(path, dpi=170)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cpu_environment()
    assert Path.cwd().resolve() == REPO
    output = args.output.resolve()
    assert args.audit.resolve().is_relative_to(REPO / "outputs")
    assert output.is_relative_to(REPO / "outputs") and output != REPO / "outputs" and not output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(args.audit, args.audit_sha256)
    audit = evidence.json(args.audit)
    validate_complete(audit)
    assert audit["input_sha256"][str(AUDITOR)] == AUDITOR_SHA
    evidence.check(AUDITOR, AUDITOR_SHA)
    for name, digest in audit["input_sha256"].items():
        evidence.check(name, digest)
    summaries, traces = zip(*(summarize(c["result"], evidence, ensemble=True) for c in audit["cases"]), strict=True)
    assert group(summaries) == audit["group_outcome"]
    comparison = baseline(evidence)
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    figures = plot_all(traces, output)
    evidence.recheck()
    result = dict(
        schema_version="s2a.temporal-equal-final-diagnostics.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        wam_trials=3,
        physics_steps=1800,
        h40_queries=225,
        raw_checkpoint_evidence=False,
        reference_intervention=INTERVENTION,
        group_outcome=group(summaries),
        cases=summaries,
        old010_raw8_seed42_comparison=comparison,
        source_audit=dict(path=str(args.audit.resolve()), sha256=args.audit_sha256),
        input_sha256=evidence.hashes,
        output_sha256={str(path): sha256(path) for path in figures},
        new_model_requests=0,
        new_physics_steps=0,
        fresh_initial_states=0,
        limitations=[
            "All3 old-scene trials must be complete and normally released before this report exports; oracle is excluded from the WAM denominator.",
            "References are changed by causal uniform temporal ensemble; this is neither a raw checkpoint score nor pure robot response compensation.",
            "Latest raw and ensemble tracking errors use the same measured state, but the reference itself changes; lower admitted error does not prove better task intent.",
            "010 comparison shares initial states and first-query inputs only; later observations diverge, so it is not same-trajectory paired causal evidence.",
            "2920/3000 describes reference endpoint participation with80 endpoints beyond the fixed budget, not2920 physical steps, accuracy or total newest-query suffix discard.",
            "Contact, height and object-center geometry are retrospective diagnostics; they neither enter control nor replace native success or certify force/grasp/release.",
            "Three old initial states/seed42 are development evidence, not multi-seed/fresh acceptance or real-robot deployment.",
        ],
    )
    path = output / "diagnostics.json"
    write_new(path, result)
    print(json.dumps(dict(path=str(path), sha256=sha256(path), complete=True)))


if __name__ == "__main__":
    main()
