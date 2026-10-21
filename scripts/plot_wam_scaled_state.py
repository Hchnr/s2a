"""Render completed028 state evidence from exactly four final JSON receipts."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HORIZONS = (8, 16, 40)
METHODS = ("raw", "adapt", "hold", "constant_velocity")
LABELS = ("Raw WAM", "Frozen adapted WAM", "Hold", "Constant velocity")
COLORS = ("#4477AA", "#EE6677", "#228833", "#999999")
METRICS = (
    ("position_rmse_m", 1000, "Position RMSE (mm)"),
    ("rotation_rmse_rad", 1, "SO(3) RMSE (rad)"),
    ("opening_mae", 1, "Opening MAE (normalized)"),
)
CHECKS = (
    "short_joint_improvement",
    "strictly_beats_hold_and_constant_velocity",
    "all_components_protected",
    "at_least_five_episodes",
    "no_episode_short_outlier",
    "both_seeds_protected",
)
CAPTION = "028 | Development state prediction; not task success or fresh evaluation"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_ref(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "invalid JSON evidence reference")
    require(isinstance(value["path"], str) and Path(value["path"]).is_absolute(), "reference path must be absolute")
    require(
        isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in value["sha256"]),
        "invalid reference SHA256",
    )
    return value


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid aggregate metric")
    return value


def read_four(args):
    """Do not follow references or rehash the audit's underlying input mapping."""
    values, refs, original = {}, {}, {}
    for role in ("sampling_result", "verification", "owner_result", "teacher_verification"):
        path = getattr(args, role).resolve()
        raw = path.read_bytes()
        values[role] = json.loads(raw)
        refs[role] = dict(path=str(path), sha256=digest(raw))
        original[str(path)] = raw
    require(len(original) == 4, "requires four distinct final JSON files")
    return values, refs, original


def validate(values, refs):
    sampling, audit, owner, teacher = (
        values[k] for k in ("sampling_result", "verification", "owner_result", "teacher_verification")
    )
    for value in values.values():
        require(value.get("passed") is True and "error" not in value, "requires successful complete audit receipts")
    require(
        sampling.get("schema_version") == "s2a.scaled-state-sampling.v1"
        and sampling.get("complete") is True
        and sampling["denoise_calls"] == len(sampling["denoise_ledger"]) == 3465
        and len(sampling["zero_identity"]) == 3
        and len(sampling["generations"]) == 96
        and sampling["head_call_counts"] == dict(zero=105, raw=1680, adapt=1680)
        and sampling["all_original_parameters_frozen"] is True
        and sampling["original_parameters_require_grad_false"] is True,
        "requires complete frozen99-generation/3465-denoise sampling",
    )
    require(
        sampling["physics_steps"]
        == sampling["fresh_cases"]
        == sampling["new_teacher_calls"]
        == sampling["new_candidate_fits"]
        == sampling["new_episode_reads"]
        == 0
        and sampling["wam_model_initializations"] == 1,
        "sampling execution budget differs",
    )
    require(
        audit.get("schema_version") == "s2a.scaled-state-independent-verification.v1"
        and audit.get("phase") == "sampling"
        and audit.get("sampling_verified") is True
        and audit["zero_identity_generations_verified"] == 3
        and audit["generations_verified"] == 96
        and audit["actual_denoise_calls_verified"] == 3465
        and audit["paired_initial_states_verified"] == 48
        and audit["actual_solver_sigma_traces_verified"] == 99
        and audit["validation_episode_states_read"] == 6,
        "requires the complete independent028 state audit",
    )
    require(
        audit["input_sha256"].get(refs["sampling_result"]["path"]) == refs["sampling_result"]["sha256"]
        and audit["input_sha256"].get(refs["teacher_verification"]["path"]) == refs["teacher_verification"]["sha256"]
        and sampling["fit_verification_ref"] == refs["teacher_verification"]
        and audit["prepared"] == sampling["prepared_ref"],
        "state audit/producer/teacher preparation or SHA binding differs",
    )
    require(
        teacher.get("schema_version") == "s2a.scaled-readout-independent-verification.v1"
        and teacher.get("phase") == "fit"
        and teacher.get("fit_verified") is True
        and teacher["teacher_gate"]["passed"] is True
        and teacher["eligible_for_runtime_qualification"] is True
        and teacher["eligible_for_sampling"] is False,
        "requires original027 teacher success with unchanged samplingFalse",
    )
    for key in ("fit_ref", "adapted_model_ref"):
        ref = valid_ref(sampling[key])
        require(
            audit["input_sha256"].get(ref["path"]) == ref["sha256"]
            and teacher["input_sha256"].get(ref["path"]) == ref["sha256"],
            "original027 candidate or fit binding differs",
        )
    require(
        owner.get("schema_version") == "s2a.scaled-state-owner.v1"
        and owner.get("stage") == "sampling"
        and owner.get("complete") is True
        and owner["automatic_retry"] is False
        and owner["producer_result"] == refs["sampling_result"]
        and owner["prepared_ref"] == sampling["prepared_ref"]
        and len(owner["owned_commands"]) == 2,
        "requires normally completed final028 sampling owner receipt",
    )
    for command in owner["owned_commands"]:
        require(set(command) == {"launch", "exit", "release"}, "owned lifecycle receipt roster differs")
        for ref in command.values():
            valid_ref(ref)
    for public, source in (("exit_ref", "exit"), ("release_ref", "release")):
        require(sampling[public] == owner["owned_commands"][0][source], "model lifecycle reference differs")
    for key in ("gpu_after_ref", "resource_observations_ref"):
        valid_ref(sampling[key])
        require(sampling[key] == owner[key], "public resource-release reference differs")
    groups, gate = audit["sampling_state_metrics"], audit["sampling_gate"]
    require(set(groups) == {"pooled", "by_episode", "by_task", "by_seed"}, "state grouping roster differs")
    require(
        len(groups["by_episode"]) == 6 and len(groups["by_task"]) == 3 and set(groups["by_seed"]) == {"42", "43"},
        "six-episode/three-task/two-seed coverage differs",
    )
    grouped = [(groups["pooled"], 48)]
    for name, count in (("by_episode", 8), ("by_task", None), ("by_seed", 24)):
        grouped.extend((group, count) for group in groups[name].values())
    for group, count in grouped:
        require(set(group) == {str(h) for h in HORIZONS}, "only the audited three prefixes may be rendered")
        for h in HORIZONS:
            require(set(group[str(h)]) == set(METHODS), "state method roster differs")
            for metrics in group[str(h)].values():
                endpoints = metrics["valid_endpoints"]
                require(
                    type(endpoints) is int
                    and endpoints > 0
                    and endpoints % h == 0
                    and (count is None or endpoints == count * h),
                    "complete-prefix endpoint denominator differs",
                )
                for key in ("position_rmse_m", "rotation_rmse_rad", "opening_mae", "joint_score"):
                    finite(metrics[key])
    for field in ("by_episode", "by_task", "by_seed"):
        for h in HORIZONS:
            for method in METHODS:
                require(
                    sum(group[str(h)][method]["valid_endpoints"] for group in groups[field].values()) == 48 * h,
                    "grouped endpoint denominators do not match the pooled complete roster",
                )
    require(set(gate["checks"]) == set(CHECKS), "registered six state checks differ")
    expected_components = {f"h{h}_{key}" for h in HORIZONS for key, _, _ in METRICS}
    require(set(gate["component_checks"]) == expected_components, "registered nine component checks differ")
    require(set(gate["episode_passes"]) == set(groups["by_episode"]), "episode gate roster differs")
    for mapping in (gate["checks"], gate["component_checks"], gate["episode_passes"]):
        require(all(type(v) is bool for v in mapping.values()), "gate decisions must be actual booleans")
    require(type(gate["passed"]) is bool and gate["passed"] == all(gate["checks"].values()), "gate summary differs")
    require(
        audit["eligible_for_development_physics_registration"] is gate["passed"]
        and owner["eligible_for_development_physics_registration"] is gate["passed"]
        and audit["physics_authorized"] is owner["physics_authorized"] is False
        and audit["physics_tasks_executed"] == 0
        and audit["raw_checkpoint_task_success_proven"] is False
        and audit["adapted_task_capability_proven"] is False,
        "state eligibility must not be promoted to task/fresh/physics success",
    )
    for group in (
        teacher["teacher_metrics"]["validation"]["pooled"],
        *teacher["teacher_metrics"]["validation"]["by_sigma"].values(),
    ):
        for method in ("raw", "adapt", "bias"):
            finite(group["mse"][method])
    return groups, gate


def ratio(adapted, raw):
    return adapted / raw if raw else None


def ratio_text(value):
    return "undefined (raw=0)" if value is None else f"{value:.6f}"


def figures(groups, gate, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    paths = []
    with PdfPages(output / "state_evidence.pdf") as pdf:
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
        for row, h in enumerate(HORIZONS):
            for col, (key, scale, title) in enumerate(METRICS):
                ax = axes[row, col]
                ax.bar(LABELS, [groups["pooled"][str(h)][m][key] * scale for m in METHODS], color=COLORS)
                decision = "PASS" if gate["component_checks"][f"h{h}_{key}"] else "FAIL"
                ax.set(title=f"H{h} (first {h / 20:g}s), {decision}", ylabel=title)
                ax.tick_params(axis="x", rotation=25, labelsize=8)
                ax.grid(axis="y", alpha=0.2)
        fig.suptitle(CAPTION + "\nCumulative endpoint prefixes; no instantaneous-time curve or confidence interval")
        path = output / "physical_components.png"
        fig.savefig(path, dpi=170)
        pdf.savefig(fig)
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, h in zip(axes, (8, 16), strict=True):
            group = groups["pooled"][str(h)]
            ax.bar(LABELS, [group[m]["joint_score"] for m in METHODS], color=COLORS)
            ax.axhline(0.90 * group["raw"]["joint_score"], color="black", ls="--", label="Pooled raw x 0.90")
            ax.set(title=f"H{h}: first {h / 20:g}s", ylabel="Dimensionless joint squared-error score")
            ax.tick_params(axis="x", rotation=20, labelsize=9)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(CAPTION + "\nJoint = mean[(position/.01m)^2 + (SO(3)/.05rad)^2 + (opening/.1)^2]")
        path = output / "short_joint_controls.png"
        fig.savefig(path, dpi=170)
        pdf.savefig(fig)
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for row, field in enumerate(("by_episode", "by_seed")):
            names = sorted(groups[field], key=int)
            for col, h in enumerate((8, 16)):
                ax = axes[row, col]
                ratios = [
                    ratio(
                        groups[field][n][str(h)]["adapt"]["joint_score"], groups[field][n][str(h)]["raw"]["joint_score"]
                    )
                    for n in names
                ]
                ax.bar(range(len(names)), [float("nan") if v is None else v for v in ratios], color=COLORS[1])
                ax.axhline(1.05, color="black", ls="--", label="EP / seed upper bound 1.05")
                if field == "by_episode":
                    ax.axhline(1.25, color="#AA3377", ls=":", label="Any-EP upper bound 1.25")
                for i, value in enumerate(ratios):
                    if value is None:
                        ax.text(i, 0.02, "raw=0\nundefined", ha="center", fontsize=8)
                ax.set(
                    title=f"{'Episode' if row == 0 else 'Seed'} H{h} (first {h / 20:g}s)",
                    ylabel="Adapt / raw joint score",
                    xticks=range(len(names)),
                    xticklabels=names,
                    ylim=(0, max([1.35, *[v * 1.12 for v in ratios if v is not None]])),
                )
                ax.grid(axis="y", alpha=0.2)
                ax.legend(fontsize=8)
        fig.suptitle(CAPTION + "\nRequired: at least 5/6 EPs and both seeds protected; pooled 0.90 is not an EP gate")
        path = output / "episode_seed_ratios.png"
        fig.savefig(path, dpi=170)
        pdf.savefig(fig)
        plt.close(fig)
        paths.append(path)
    return [*paths, output / "state_evidence.pdf"]


def summary(values, refs, groups, gate):
    audit, teacher = values["verification"], values["teacher_verification"]
    decision = "PASSED" if gate["passed"] else "FAILED"
    lines = [
        "# 028 frozen scaled readout: development state evidence",
        "",
        f"**The registered state gate {decision}. The independent execution/state audit passed.**",
        "",
        "This is development state prediction, not closed-loop task success or fresh evaluation. "
        "Original027 teacher evidence retains sampling=False. New028 state success only permits separately registering "
        "development physics; physics remains unauthorized and no physics/fresh task was executed.",
        "",
        "Complete evidence: 3 zero identities + 96 generations, 3465 denoises, 48 exact initial-state pairs, "
        "99 actual solver-sigma traces, six validation episodes and seeds42/43. "
        "The final owner completed after actual owned wait/reap and resource release; this renderer consumes "
        "the final JSON receipts and does not reopen their referenced lifecycle or label files.",
        "",
        "Generated local deltas are integrated from the current measured source EEF state, bypassing S2A and physics. "
        "Opening is clipped to[0,1]. H8/H16/H40 aggregate the first8/16/40 endpoints through0.4/0.8/2.0s. "
        "They are cumulative prefixes, not single-time errors. Joint score is "
        "mean[(position/.01m)^2 + (SO(3)/.05rad)^2 + (opening/.1)^2].",
        "",
        "## Registered quality checks",
        "",
        "| Check | Passed |",
        "| --- | --- |",
    ]
    lines += [f"| {key} | {gate['checks'][key]} |" for key in CHECKS]
    lines += [
        "",
        "All nine physical component checks require BOTH <=1.05raw and <=raw+absolute_limit "
        "(2mm /0.01rad /0.01opening). At least5/6 EPs must satisfy <=1.05raw at both short horizons; "
        "all EPs must satisfy <=1.25raw. Both seeds must satisfy <=1.05raw at both short horizons. "
        "H8/H16 pooled joint must be <=.90raw and strictly below hold and constant velocity.",
        "",
        "| Component | Passed |",
        "| --- | --- |",
    ]
    lines += [f"| {key} | {value} |" for key, value in gate["component_checks"].items()]
    lines += [
        "",
        "| Episode | H8 adapt/raw joint | H16 adapt/raw joint | Both short horizons <=1.05raw |",
        "| --- | --- | --- | --- |",
    ]
    for name, group in sorted(groups["by_episode"].items(), key=lambda x: int(x[0])):
        ratios = [
            ratio_text(ratio(group[str(h)]["adapt"]["joint_score"], group[str(h)]["raw"]["joint_score"]))
            for h in (8, 16)
        ]
        lines.append(f"| {name} | {ratios[0]} | {ratios[1]} | {gate['episode_passes'][name]} |")
    lines += ["", "## Audited physical-state metrics", ""]
    for label, groupings in (
        ("Pooled", {"all": groups["pooled"]}),
        ("Episode", groups["by_episode"]),
        ("Task (diagnostic; no extra task gate)", groups["by_task"]),
        ("Seed", groups["by_seed"]),
    ):
        lines += [
            f"### {label}",
            "",
            "| Group | Prefix | Method | Position RMSE mm | SO(3) RMSE rad | Opening MAE | Joint score | Endpoints |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, group in groupings.items():
            for h in HORIZONS:
                for method in METHODS:
                    m = group[str(h)][method]
                    lines.append(
                        f"| {name} | H{h} | {method} | {1000 * m['position_rmse_m']:.6f} | "
                        f"{m['rotation_rmse_rad']:.9f} | {m['opening_mae']:.9f} | "
                        f"{m['joint_score']:.9f} | {m['valid_endpoints']} |"
                    )
        lines.append("")
    lines += [
        "## Original027 teacher evidence (different quantity)",
        "",
        "Teacher MSE is normalized active10 velocity error against epsilon-x0 at true-future noised latents, "
        "after deployment BF16 quantization. It is not physical state error. The candidate was fitted once "
        "previously; this report performs no fit or parameter reconstruction.",
        "",
        "| Validation group | Raw MSE | Adapt MSE | Reused time-bias MSE |",
        "| --- | --- | --- | --- |",
    ]
    validation = teacher["teacher_metrics"]["validation"]
    teacher_groups = [("pooled", validation["pooled"]), *[(f"sigma {k}", v) for k, v in validation["by_sigma"].items()]]
    for name, group in teacher_groups:
        m = group["mse"]
        lines.append(f"| {name} | {m['raw']:.12g} | {m['adapt']:.12g} | {m['bias']:.12g} |")
    lines += [
        "",
        f"Development physics registration eligible: `{audit['eligible_for_development_physics_registration']}`. "
        "Physics authorized: `False`. Original raw-checkpoint task success proven: `False`. "
        "Adapted task capability proven: `False`.",
        "",
        "The EPs are reused development validation, not a fresh or certified pretraining-unseen split. "
        "Paired anchors/endpoints/seeds are correlated; no confidence interval or independent-trial claim is made. "
        "Only three prefix aggregates are saved, so no dense endpoint curve is reconstructed.",
        "",
        "## Source JSON receipts",
        "",
        "| Role | Path | SHA256 |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {role} | `{ref['path']}` | `{ref['sha256']}` |" for role, ref in refs.items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("sampling-result", "verification", "owner-result", "teacher-verification", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "JSON report requires repository cwd and emptyCUDA/LD",
    )
    output = args.output_dir.resolve()
    require(output.is_relative_to(REPO / "outputs") and not output.exists(), "refuse existing/non-output destination")
    values, refs, original = read_four(args)
    groups, gate = validate(values, refs)
    source = Path(__file__).resolve()
    source_sha = digest(source.read_bytes())

    def unchanged():
        for filename, raw in original.items():
            require(Path(filename).read_bytes() == raw, "final input JSON changed during rendering")
        require(digest(source.read_bytes()) == source_sha, "renderer source changed")

    unchanged()
    output.mkdir(parents=True)
    paths = figures(groups, gate, output)
    summary_path = output / "summary.md"
    with summary_path.open("x") as stream:
        stream.write(summary(values, refs, groups, gate))
    paths.append(summary_path)
    unchanged()
    manifest = dict(
        schema_version="s2a.scaled-state-json-presentation.v1",
        complete=True,
        seed=args.seed,
        state_gate_passed=gate["passed"],
        input_json_refs=refs,
        source_ref=dict(path=str(source), sha256=source_sha),
        output_sha256={str(path): digest(path.read_bytes()) for path in paths},
        inherited_audit_ref=refs["verification"],
        inherited_input_sha256=values["verification"]["input_sha256"],
        inherited_input_files_reopened=False,
        full_audit_independently_reexecuted=False,
        display_units=dict(position="mm", rotation="rad", opening="normalized", joint="dimensionless squared score"),
        prefixes=[dict(endpoints=h, through_seconds=h / 20, aggregation="cumulative prefix") for h in HORIZONS],
        new_model_calls=0,
        new_fit_groups=0,
        new_episode_reads=0,
        new_png_reads=0,
        new_npz_reads=0,
        physics_steps=0,
        fresh_cases=0,
        task_success_claimed=False,
    )
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output_dir=str(output), state_gate_passed=gate["passed"], input_json_files_read=4)))


if __name__ == "__main__":
    main()
