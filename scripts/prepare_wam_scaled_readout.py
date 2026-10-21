"""Freeze027's cache-only inputs and the independently verified026 control.

No original episode arrays/PNGs, model, preprocessing statistics or parameter
fit are opened/computed here. Existing cache archives and metadata are audited.
"""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
from prepare_wam_flow_readout import (
    Evidence,
    external_identity,
    reference,
    require,
    terminal,
    validate_rosters,
    weight_stats,
)

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_train_scale_readout_20261021.json"
CONFIG_SHA = "dd0e0124d256a953af48ea2e548cca6603cd9f777d6bd5e4f620548591a2c06b"
PLAN = REPO / "plans/v0.0.1/027_train_scale_readout_20261021.md"
PLAN_SHA = "586ddf2656d39752efd7979e1877af940004e14b058d900c7ef464661cbf9fb1"
NEW_SOURCES = (
    "src/s2a/wam_scaled_readout.py",
    "tests/test_wam_scaled_readout.py",
    "scripts/fit_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_fit.py",
    "scripts/prepare_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_preparation.py",
    "scripts/verify_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_verification.py",
    "scripts/run_wam_scaled_readout.py",
    "tests/test_wam_scaled_readout_runner.py",
)
FLOW_REFS = (
    ("source_flow_prepared", "flow_prepared_ref"),
    ("source_flow_preparation_verification", "flow_preparation_verification_ref"),
    ("source_flow_fit", "flow_fit_ref"),
    ("source_flow_fit_verification", "flow_fit_verification_ref"),
    ("source_flow_final_peer", "flow_final_peer_ref"),
)
INHERITED = (
    "source_prepared_ref",
    "source_features_ref",
    "source_features_verification_ref",
    "source_failed_fit_verification_ref",
    "anchors",
    "episodes",
    "teacher_calls",
    "validation_generations",
    "zero_identity_cases",
    "domain",
    "checkpoint",
    "weight_files",
    "external_cosmos",
    "conservative_source_exclusion",
)


def forbidden_observation_paths(prepared):
    paths = {str(Path(a[k]).resolve()) for a in prepared["anchors"] for k in ("arrays_path", "image_path")}
    for case in prepared["zero_identity_cases"]:
        for key in ("arrays_path", "image_path"):
            paths.add(str(Path(case["sample"][key]).resolve()))
    return paths


def cache_only_hashes(mapping, forbidden, evidence):
    """Never accidentally expand an ancestor map back to source pixels/state."""
    for path, digest in mapping.items():
        source = Path(path).resolve()
        require(str(source) not in forbidden and source.suffix.lower() != ".png", "027 must not reopen EP/PNG refs")
        evidence.check(source, digest)


def flow_admission(flow, prep_peer, fit, fit_peer, final_peer, refs):
    require(
        flow["schema_version"] == "s2a.flow-readout-preparation.v1"
        and len(flow["source_sha256"]) == len(flow["snapshot_sha256"]) == 72
        and len(flow["input_sha256"]) == 761,
        "026 frozen72-source/761-input preparation differs",
    )
    require(
        prep_peer["passed"] is True
        and prep_peer["phase"] == "prepare"
        and prep_peer["preparation_verified"] is True
        and prep_peer["prepared"] == refs["flow_prepared_ref"],
        "026 independent preparation admission differs",
    )
    require(
        fit["complete"] is True
        and fit["passed"] is True
        and fit["prepared_ref"] == refs["flow_prepared_ref"]
        and fit["features_ref"] == flow["source_features_ref"]
        and fit["producer_fit_groups"] == fit["fit_attempts"] == 2
        and fit["fitted_rows"] == fit["validation_rows"] == 2880
        and fit["eligible_for_sampling"] is False
        and fit["teacher_gate"]["passed"] is False,
        "026 actual two-fit failed teacher result must remain False",
    )
    require(
        fit_peer["passed"] is True
        and fit_peer["phase"] == "fit"
        and fit_peer["fit_verified"] is True
        and fit_peer["prepared"] == refs["flow_prepared_ref"]
        and fit_peer["independent_model_reconstructions"] == 2
        and fit_peer["eligible_for_sampling"] is False
        and fit_peer["teacher_gate"]["passed"] is False
        and fit_peer["input_sha256"].get(refs["flow_fit_ref"]["path"]) == refs["flow_fit_ref"]["sha256"],
        "026 independent failed gate/fit identity differs",
    )
    require(
        final_peer["passed"] is True
        and final_peer["prepared"] == refs["flow_prepared_ref"]
        and final_peer["fit"] == refs["flow_fit_ref"]
        and final_peer["fit_verification"] == refs["flow_fit_verification_ref"]
        and final_peer["eligible_for_sampling"] is False
        and final_peer["old025_eligible_for_sampling"] is False
        and final_peer["teacher_gate"]["passed"] is False
        and final_peer["source_before_after_unchanged"] is True
        and final_peer["sampling_and_orchestration_directory_absent"] is True,
        "026 formal final peer/failure provenance differs",
    )


def verify_control(flow, fit, evidence):
    """Audit original control bytes/row identities, without fitting or applying it."""
    model_ref, predictions_ref = fit["models"]["bias"], fit["teacher_predictions"]
    model = evidence.check(model_ref["path"], model_ref["sha256"])
    predictions = evidence.check(predictions_ref["path"], predictions_ref["sha256"])
    with np.load(model, allow_pickle=False) as archive:
        require(archive.files == ["weights"], "026 bias archive roster differs")
        weights = archive["weights"]
        require(
            weights.shape == (1281, 10)
            and weights.dtype == np.float64
            and np.isfinite(weights).all()
            and not np.any(weights[:-1]),
            "026 control is not its original finite ten-parameter bias",
        )
    anchors = {a["name"]: a for a in flow["anchors"]}
    expected_rows = [
        dict(
            episode=anchors[c["anchor_name"]]["episode"],
            anchor_name=c["anchor_name"],
            sigma=c["sigma"],
            split=c["split"],
            task_index=anchors[c["anchor_name"]]["task_index"],
            endpoint=endpoint,
        )
        for c in flow["teacher_calls"]
        for endpoint in range(40)
    ]
    require(fit["teacher_rows"] == expected_rows, "026 bias prediction row/split/target identities differ")
    with np.load(predictions, allow_pickle=False) as archive:
        require(
            set(archive.files) == {"raw", "adapt", "bias", "training_weights", "sigma_actual"},
            "026 actual prediction archive roster differs",
        )
        for name in ("raw", "adapt", "bias"):
            value = archive[name]
            require(
                value.shape == (5760, 128) and value.dtype == np.float32 and np.isfinite(value).all(),
                "026 actual BF16-expanded control predictions differ",
            )
        row_weights, sigma = archive["training_weights"], archive["sigma_actual"]
        require(
            row_weights.shape == (2880,)
            and row_weights.dtype == np.float64
            and np.allclose(row_weights, 1 / 2880, rtol=0, atol=1e-18)
            and sigma.dtype == np.float64
            and np.array_equal(sigma, np.asarray([np.float32(r["sigma"]) for r in expected_rows], np.float64)),
            "026 cached training weights or exact FP32-expanded sigma differs",
        )
    return dict(
        model_ref=model_ref,
        predictions_ref=predictions_ref,
        control_fit_count=0,
        control_rows=5760,
        original_train_rows=2880,
        original_validation_rows=2880,
        control_coefficients_hidden_exact_zero=True,
        control_predictions_recomputed_here=False,
        control_prediction_reproduction_required_during_fit=True,
    )


def owned_cpu_exits(final_peer, evidence):
    entries = final_peer["owned_cpu_children"]
    require(len(entries) == 2, "026 final peer must preserve both actual CPU children")
    records = []
    for entry in entries:
        launch, exit_record, release = (evidence.ref(entry[k]) for k in ("launch", "exit", "release"))
        require(
            exit_record == dict(actual_wait=True, automatic_retry=False, returncode=0),
            "026 prior CPU child has no successful actual wait",
        )
        require(
            all(release[k] == launch["child"][k] for k in ("pid", "start_ticks"))
            and release["returncode"] == 0
            and release["reaped"] is True
            and release["signal_sent"] is False
            and release["original_error"] is None,
            "026 prior CPU child was not normally reaped",
        )
        records.append(
            dict(
                role=entry["role"],
                exit=entry["exit"],
                release=entry["release"],
                current_birth=terminal(launch["child"]),
            )
        )
    return records


def source_closure(flow, evidence):
    sources = dict(flow["source_sha256"])
    for path in [REPO / p for p in NEW_SOURCES] + [CONFIG, PLAN]:
        sources[str(path.resolve())] = sha256(evidence.check(path))
    require(len(sources) == 84, "027 requires inherited72 plus new10 sources and plan/config")
    return sources


def prepare():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "027 preparation requires S2A cwd and explicitly empty CUDA/LD",
    )
    evidence = Evidence()
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    root = REPO / config["output_dir"]
    require(root.is_relative_to(REPO / "outputs") and not root.exists(), "027 refuses existing/non-output root")
    refs, ancestors = {}, {}
    for source, name in FLOW_REFS:
        path = REPO / config[source]
        ancestors[name] = evidence.json(path, config[f"{source}_sha256"])
        refs[name] = reference(path)
    flow, prep_peer, fit, fit_peer, final_peer = (ancestors[name] for _, name in FLOW_REFS)
    flow_admission(flow, prep_peer, fit, fit_peer, final_peer, refs)
    forbidden = forbidden_observation_paths(flow)
    for mapping in (
        flow["source_sha256"],
        flow["snapshot_sha256"],
        flow["input_sha256"],
        prep_peer["input_sha256"],
        fit["input_sha256"],
        fit_peer["input_sha256"],
        final_peer["input_sha256"],
    ):
        cache_only_hashes(mapping, forbidden, evidence)
    weight_stats(flow["weight_files"])
    require(external_identity(config["cosmos_repo"]) == flow["external_cosmos"], "026 external clean8ed source differs")
    old = evidence.ref(flow["source_prepared_ref"])
    old_verified = evidence.ref(flow["source_features_verification_ref"])
    validate_rosters(config, old, old_verified["input_sha256"])
    for key in INHERITED:
        if key in old:
            require(flow[key] == old[key], f"026 changed inherited025 {key}")
    control = verify_control(flow, fit, evidence)
    exits = owned_cpu_exits(final_peer, evidence)
    sources = source_closure(flow, evidence)  # Missing new source rejects before creating any output.
    root.mkdir()
    directory = root / "preparation"
    directory.mkdir()
    try:
        snapshots = directory / "source_snapshot"
        snapshots.mkdir()
        copies = {}
        for i, (path, digest) in enumerate(sources.items()):
            target = snapshots / f"{i:03d}_{Path(path).name}"
            shutil.copyfile(path, target)
            require(sha256(target) == digest, "027 source snapshot differs")
            copies[str(target)] = digest
        evidence.recheck()
        weight_stats(flow["weight_files"])
        require(external_identity(config["cosmos_repo"]) == flow["external_cosmos"], "external source changed")
        result = dict(
            schema_version="s2a.scaled-readout-preparation.v1",
            config=config,
            config_ref=reference(CONFIG),
            registration=reference(PLAN),
            output_root=str(root),
            **refs,
            **{key: copy.deepcopy(flow[key]) for key in INHERITED},
            reused_bias_ref=control["model_ref"],
            reused_bias_predictions_ref=control["predictions_ref"],
            source_admission=dict(
                original_flow_source_count=72,
                original_flow_actual_input_count=761,
                original_weight_stat_count=163,
                original025_eligible_for_sampling=False,
                original026_eligible_for_sampling=False,
                inherited_episode_refs_not_reopened=True,
                inherited_png_refs_not_reopened=True,
                preprocessing_statistics_computed=False,
                actual_control=control,
                prior_owned_cpu_exits=exits,
                historical_initial_physical_sync_certified=False,
                pretraining_visibility="unknown",
            ),
            budget=config["budget"],
            source_sha256=sources,
            snapshot_sha256=copies,
            input_sha256=evidence.hashes,
            new_model_calls=0,
            new_teacher_calls=0,
            new_episode_reads=0,
            new_png_reads=0,
            fit_count=0,
            physics_steps=0,
            fresh_performance_cases=0,
            full_weight_bytes_rehashed=False,
        )
        path = directory / "prepared.json"
        write_once(path, result)
        return path
    except BaseException as error:
        write_once(directory / "error.json", dict(type=type(error).__name__, message=str(error), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(args.seed == 42, "027 requires registered seed42")
    path = prepare()
    print(json.dumps(dict(prepared=reference(path), model_calls=0, fit_count=0, episode_reads=0)))


if __name__ == "__main__":
    main()
