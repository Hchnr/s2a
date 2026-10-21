"""029 presentation from four final JSONs; no labels, tensors or fitting are read."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGGREGATION = "mean within each EP, then equal mean across EPs; point_max_abs denotes mean point maxima"
CAPTION = "029 | Reused development EPs; no task/fresh result or causal intervention"
COMPONENTS = ("active10", "position", "rotation6d", "opening")
COLORS = ("#4477AA", "#EE6677", "#228833")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_inputs(args):
    values, refs, raw = {}, {}, {}
    for role in ("prepared", "collection_result", "verification", "owner_result"):
        path = getattr(args, role).resolve()
        content = path.read_bytes()
        values[role] = json.loads(content)
        refs[role] = dict(path=str(path), sha256=sha(content))
        raw[str(path)] = content
    require(len(raw) == 4, "four distinct final JSON inputs are required")
    return values, refs, raw


def scalar_metrics(metrics):
    require(isinstance(metrics, dict) and metrics, "missing audited diagnostic metrics")
    for key, value in metrics.items():
        require(type(value) in (int, float) and math.isfinite(value), f"nonfinite diagnostic metric: {key}")
        if not key.endswith((".two_e_delta", ".mse_change")):
            require(value >= 0, f"negative squared/absolute diagnostic metric: {key}")


def validate(values, refs):
    prepared, result, audit, owner = (
        values[k] for k in ("prepared", "collection_result", "verification", "owner_result")
    )
    require(
        prepared["schema_version"] == "s2a.free-trajectory-preparation.v1"
        and len(prepared["source_sha256"]) == len(prepared["snapshot_sha256"]) == 111
        and len(prepared["anchors"]) == 24
        and len(prepared["free_generations"]) == 48
        and len(prepared["teacher_calls"]) == 144,
        "requires final registered111-source preparation and24/48/144 roster",
    )
    for record in (result, audit, owner):
        require(record.get("passed") is True and "error" not in record, "partial or unaudited final JSON rejected")
    require(
        result["schema_version"] == "s2a.free-trajectory-collection.v1"
        and result["complete"] is True
        and result["prepared_ref"] == refs["prepared"]
        and result["denoise_calls"] == 1824
        and len(result["free_generations"]) == 48
        and len(result["teacher_calls"]) == 144
        and result["head_call_counts"] == dict(free=1680, teacher=144)
        and result["all_original_parameters_frozen"] is result["original_parameters_require_grad_false"] is True
        and result["wam_model_initializations"] == 1,
        "requires complete frozen48-free/144-teacher/1824 collection",
    )
    require(
        all(result[k] == 0 for k in ("fit_groups", "train_episode_reads", "physics_steps", "fresh_cases")),
        "collection diagnostic budget differs",
    )
    require(
        result["runtime"]["checkpoint_path"] == prepared["checkpoint"] and result["runtime"]["ema_weights"] is False,
        "collection checkpoint identity differs",
    )
    labels = [f"free:{i}" for i in range(48) for _ in range(35)] + [f"teacher:{i}" for i in range(144)]
    require(
        result["denoise_ledger"] == [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)],
        "complete registered denoise ledger differs",
    )
    require(
        audit["schema_version"] == "s2a.free-trajectory-independent-verification.v1"
        and audit["phase"] == "collection"
        and audit["collection_verified"] is True
        and audit["prepared"] == refs["prepared"]
        and audit["input_sha256"].get(refs["prepared"]["path"]) == refs["prepared"]["sha256"]
        and audit["input_sha256"].get(refs["collection_result"]["path"]) == refs["collection_result"]["sha256"]
        and audit["complete_free_replays"] == 48
        and audit["full128_raw_and_normalized_exact"] is True
        and audit["free_saved_points"] == audit["teacher_saved_points"] == 144
        and audit["actual_denoise_calls"] == 1824,
        "complete independent collection audit or producer SHA binding missing",
    )
    require(
        audit["eligible_for_fit"] is audit["physics_authorized"] is audit["task_success_proven"] is False
        and audit["original028_state_gate"] is audit["original027_eligible_for_sampling"] is False,
        "diagnostic must retain original failures and cannot authorize fitting or physics",
    )
    require(
        owner["schema_version"] == "s2a.free-trajectory-owner.v1"
        and owner["stage"] == "collection"
        and owner["complete"] is True
        and owner["automatic_retry"] is owner["eligible_for_fit"] is owner["physics_authorized"] is False
        and owner["prepared_ref"] == refs["prepared"]
        and owner["producer_result"] == refs["collection_result"]
        and len(owner["owned_commands"]) == 2,
        "normally completed collection owner evidence missing",
    )
    for public, command in (("exit_ref", "exit"), ("release_ref", "release")):
        require(result[public] == owner["owned_commands"][0][command], "model lifecycle reference differs")
    for key in ("gpu_after_ref", "resource_observations_ref"):
        require(result[key] == owner[key], "public resource-release reference differs")
    for key in ("exit_ref", "release_ref", "gpu_after_ref", "resource_observations_ref"):
        evidence = result[key]
        require(
            audit["input_sha256"].get(evidence["path"]) == evidence["sha256"],
            "independent audit omitted the bound public lifecycle evidence",
        )
    exited = audit["actual_owned_model_exit"]
    require(
        type(exited["pid"]) is type(exited["start_ticks"]) is int
        and exited["pid"] > 0
        and exited["start_ticks"] > 0
        and exited["reaped"] is True
        and exited["signal_sent"] is False,
        "independent audit did not verify normal owned model exit",
    )
    model_ref = prepared["adapted_model_ref"]
    require(
        audit["input_sha256"].get(model_ref["path"]) == model_ref["sha256"],
        "fixed027 model identity omitted from independent audit",
    )
    for planned, actual in zip(prepared["free_generations"], result["free_generations"], strict=True):
        require(all(actual[k] == v for k, v in planned.items()) and len(actual["points"]) == 3, "free roster differs")
    for planned, actual in zip(prepared["teacher_calls"], result["teacher_calls"], strict=True):
        require(all(actual[k] == v for k, v in planned.items()), "teacher roster differs")
    require(
        len(prepared["precision_nodes"]) == 5
        and [n["step_index"] for n in prepared["precision_nodes"]] == [0, 19, 29, 33, 34],
        "registered actual sigma-node roster differs",
    )
    rows, diagnostics = audit["diagnostic_rows"], audit["diagnostics"]
    require(len(rows) == 144, "complete144 audited point rows required")
    anchors = {a["name"]: a for a in prepared["anchors"]}
    for row, call in zip(rows, prepared["teacher_calls"], strict=True):
        anchor = anchors[call["anchor_name"]]
        expected = dict(
            call_index=call["call_index"],
            anchor_name=call["anchor_name"],
            episode=anchor["episode"],
            task=anchor["task_index"],
            seed=call["sampling_seed"],
            sigma=call["sigma_actual"],
            step_index=call["step_index"],
        )
        require(all(row[k] == v for k, v in expected.items()), "audited diagnostic row identity differs")
        scalar_metrics(row["metrics"])
    require(
        set(diagnostics) == {"pooled", "by_sigma", "by_seed", "by_episode", "by_task"},
        "diagnostic group roster differs",
    )
    cells = [(diagnostics["pooled"], rows)]
    for field in ("sigma", "seed", "episode", "task"):
        members = sorted({r[field] for r in rows})
        require(set(diagnostics[f"by_{field}"]) == {str(x) for x in members}, "diagnostic subgroup identities differ")
        cells += [(diagnostics[f"by_{field}"][str(x)], [r for r in rows if r[field] == x]) for x in members]
    columns = set(rows[0]["metrics"])
    for group, members in cells:
        require(
            group["points"] == len(members)
            and group["episodes"] == len({r["episode"] for r in members})
            and group["endpoints"] == 40 * len(members)
            and group["aggregation"] == AGGREGATION
            and set(group["metrics"]) == columns,
            "audited EP-equal aggregation or denominators differ",
        )
        scalar_metrics(group["metrics"])
    require(
        len(diagnostics["by_sigma"]) == 3 and set(diagnostics["by_seed"]) == {"42", "43"}, "sigma/seed coverage differs"
    )
    return diagnostics


def figures(prepared, diagnostics, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    nodes = prepared["precision_nodes"][1:4]
    metrics = [diagnostics["by_sigma"][str(n["sigma_actual"])]["metrics"] for n in nodes]
    labels = [f"Step {n['step_index']}\nsigma~{n['sigma_actual']:.5f}" for n in nodes]
    paths = []

    def save(fig, name, pdf):
        path = output / f"{name}.png"
        fig.savefig(path, dpi=170)
        pdf.savefig(fig)
        plt.close(fig)
        paths.append(path)

    with PdfPages(output / "free_trajectory_diagnostics.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for row, branch in enumerate(("teacher", "free")):
            for col, (m, label) in enumerate(zip(metrics, labels, strict=True)):
                ax = axes[row, col]
                ax.bar(
                    ["Raw", "Offline027"],
                    [m[f"velocity.{branch}.active10.{v}_error.mse"] for v in ("raw", "offline027")],
                    color=COLORS[:2],
                )
                ax.set(title=f"{branch}: {label}", ylabel="Velocity MSE (active10)")
                ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            CAPTION
            + "\nTeacher: epsilon-x0 | Free: v_pull=(sample_FP32-x0)/sigma; compare variants only within each branch"
        )
        save(fig, "velocity_targets", pdf)

        fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
        for row, (m, label) in enumerate(zip(metrics, labels, strict=True)):
            for col, component in enumerate(COMPONENTS):
                ax = axes[row, col]
                ax.bar(
                    ["Raw", "Offline027"],
                    [m[f"clean.{component}.{v}_actual_error.mse"] for v in ("raw", "offline027")],
                    color=COLORS[:2],
                )
                ax.set(title=f"{label} | {component}", ylabel="Normalized clean-action MSE")
                ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            CAPTION
            + "\nActual CUDA conversion on saved raw points; these are action-latent errors, not physical state errors"
        )
        save(fig, "free_clean_action", pdf)

        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for row, arithmetic in enumerate(("actual", "ideal")):
            for col, (m, label) in enumerate(zip(metrics, labels, strict=True)):
                ax = axes[row, col]
                keys = ("two_e_delta", "delta_squared", "mse_change")
                ax.bar(
                    ["2 e delta", "delta squared", "MSE change"],
                    [m[f"decomposition.active10.{arithmetic}.{k}"] for k in keys],
                    color=COLORS,
                )
                ax.axhline(0, color="black", lw=0.7)
                ax.set(title=f"{arithmetic}: {label}", ylabel="Mean normalized clean-error contribution")
                ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            CAPTION
            + "\nFree-path counterfactual: MSE change = direction term + squared update; no actual adapted rollout"
        )
        save(fig, "free_direction_magnitude", pdf)

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for col, (key, title) in enumerate(
            (
                ("mean_row_norm", "Mean centered/scaled row norm"),
                ("point_mean_shift_norm", "Mean per-point feature-mean norm"),
            )
        ):
            ax = axes[0, col]
            for offset, branch in enumerate(("teacher", "free")):
                ax.bar(
                    [i + (offset - 0.5) * 0.34 for i in range(3)],
                    [m[f"hidden.{branch}.{key}"] for m in metrics],
                    width=0.32,
                    color=COLORS[offset],
                    label=branch,
                )
            ax.set(title=title, xticks=range(3), xticklabels=labels)
            ax.legend()
        differences = (
            (axes[0, 2], "hidden.paired_scaled_difference.mse", "Paired centered/scaled hidden difference MSE"),
            (axes[1, 0], "packed.vision.paired_difference.mse", "Paired packed video latent difference MSE"),
            (axes[1, 1], "packed.action.paired_difference.mse", "Paired packed action latent difference MSE"),
            (
                axes[1, 2],
                "packed.vision.current_condition_difference.mse",
                "First-condition video latent difference MSE",
            ),
        )
        for ax, key, title in differences:
            ax.bar(labels, [m[key] for m in metrics], color=COLORS[0])
            ax.set(title=title)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=0.2)
            ax.set_ylim(bottom=0)
        fig.suptitle(
            CAPTION + "\nOriginal027 center/scale only; latent/input differences do not identify a unique failure cause"
        )
        save(fig, "hidden_packed_differences", pdf)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        for ax, keys, names, title in (
            (
                axes[0],
                ("quantization.active10.q.mse", "quantization.active10.q_over_sigma.mse"),
                ("q", "q / sigma"),
                "BF16 packed action minus FP32 sample",
            ),
            (
                axes[1],
                ("clean.active10.raw_actual_minus_ideal.mse", "clean.active10.offline027_actual_minus_ideal.mse"),
                ("Raw", "Offline027"),
                "Actual conversion minus ideal FP64 (MSE)",
            ),
            (
                axes[2],
                (
                    "clean.active10.raw_actual_minus_ideal.point_max_abs",
                    "clean.active10.offline027_actual_minus_ideal.point_max_abs",
                ),
                ("Raw", "Offline027"),
                "Mean point maximum absolute difference",
            ),
        ):
            for offset, (key, name) in enumerate(zip(keys, names, strict=True)):
                ax.bar(
                    [i + (offset - 0.5) * 0.34 for i in range(3)],
                    [m[key] for m in metrics],
                    width=0.32,
                    label=name,
                    color=COLORS[offset],
                )
            ax.set(title=title, xticks=range(3), xticklabels=labels)
            ax.grid(axis="y", alpha=0.2)
            ax.set_ylim(bottom=0)
            ax.legend()
        fig.suptitle(
            CAPTION + "\nq/sigma is a quantization term, not evidence that original epsilon-x0 teacher labels are wrong"
        )
        save(fig, "quantization_conversion", pdf)
    return [*paths, output / "free_trajectory_diagnostics.pdf"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("validate", "render"))
    for key in ("prepared", "collection-result", "verification", "owner-result", "output-dir"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "presentation requires repository cwd and emptyCUDA/LD",
    )
    output = args.output_dir.resolve()
    require(output.is_relative_to(REPO / "outputs") and not output.exists(), "refuse existing/non-output destination")
    values, refs, original = read_inputs(args)
    diagnostics = validate(values, refs)
    if args.stage == "validate":
        print(json.dumps(dict(validated=True, final_json_inputs=4, output_created=False)))
        return
    source = Path(__file__).resolve()
    source_sha = sha(source.read_bytes())

    def unchanged():
        for name, raw in original.items():
            require(Path(name).read_bytes() == raw, "final JSON changed during rendering")
        require(sha(source.read_bytes()) == source_sha, "renderer source changed")

    unchanged()
    output.mkdir(parents=True)
    paths = figures(values["prepared"], diagnostics, output)
    summary = dict(
        schema_version="s2a.free-trajectory-json-summary.v1",
        seed=42,
        input_json_refs=refs,
        complete_released_collection_verified=True,
        precision_nodes=values["prepared"]["precision_nodes"],
        sigma_display="Figure labels round actual sigma to five decimals; exact values remain in precision_nodes and by_sigma keys",
        diagnostics=diagnostics,
        original028_state_gate=False,
        original027_eligible_for_sampling=False,
        eligible_for_fit=False,
        physics_authorized=False,
        task_success_proven=False,
        aggregation=AGGREGATION,
        teacher_target="epsilon-x0 on true-future noising",
        free_target="(actual FP32 free sample - selected demonstration x0) / actual sigma",
        clean_error_scope="Only free raw-path points; actual CUDA mixed-precision normalized clean-action errors, not physical state errors",
        offline027_scope="Frozen027 readout applied to saved raw hidden; not an actual adapted trajectory or a causal intervention",
        interpretation=values["verification"]["interpretation"],
        denominator_scope="Six reused development EPs; correlated anchors/seeds/points/endpoints, not144 independent trials",
        absent_statistics="No teacher clean-conversion metric is saved; no such statistic is inferred from velocity MSE",
    )
    summary_path = output / "summary.json"
    with summary_path.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    paths.append(summary_path)
    unchanged()
    manifest = dict(
        schema_version="s2a.free-trajectory-json-presentation.v1",
        complete=True,
        input_json_refs=refs,
        source_ref=dict(path=str(source), sha256=source_sha),
        output_sha256={str(path): sha(path.read_bytes()) for path in paths},
        inherited_audit_ref=refs["verification"],
        inherited_input_files_reopened=False,
        full_independent_audit_reexecuted=False,
        input_json_files_read=4,
        png_count=5,
        pdf_pages=5,
        new_model_calls=0,
        new_fits=0,
        new_parameter_reconstructions=0,
        new_episode_reads=0,
        new_png_input_reads=0,
        new_npz_reads=0,
        new_label_scores=0,
        scheduler_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output_dir=str(output), final_json_inputs=4, diagnostic_points=144, model_calls=0)))


if __name__ == "__main__":
    main()
