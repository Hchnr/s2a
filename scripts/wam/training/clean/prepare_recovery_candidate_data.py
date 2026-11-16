"""Prepare traceable CPU-only recovery-window candidates without fitting a model."""

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.broader_recovery_h40 import clipping_flags
from s2a.research.wam.training.clean.clean_training_data import (
    AnchorIdentity,
    SourceIdentity,
    derive_noise_seed,
    normalize_active10,
)
from s2a.research.wam.training.recovery_candidate_data import (
    SEED_NAMESPACE,
    candidate_seed,
    clipping_translation_effect,
    transition_multiplicities,
)
from s2a.research.wam.training.recovery_h40_contract import audit_window
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_broader_event_images as roster
from scripts.simulation.runtime import prepare_broader_recovery_pilot as pilot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/recovery_candidate_data_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/recovery_candidate_data_20261116.md"
OUTPUT = ROOT / "outputs/recovery_candidate_data_20261116"
QA = ROOT / "outputs/qa/recovery_candidate_data_final_20261116/result.json"
REFERENCES = {
    "image_audit": (
        "outputs/broader_event_images_20261116/audit.json",
        "44007db728f81d265f4a5ac15999473c096ecadf92adeb6d741de6308af32d17",
    ),
    "image_dry": (
        "outputs/broader_event_images_20261116/dry.json",
        "c89f19651e6bae10f4453565ee2bf57c81b5e9f0a50e79a7971fcd4574fb7087",
    ),
    "original_h40": (
        "outputs/broader_recovery_h40_20261116/result.json",
        "ca4c9d8910be0949ed877072bcfe8f03d598760108d4ff1ee57f035565625ba4",
    ),
    "event_union": (
        "outputs/broader_recovery_events_20261116/union.json",
        "b46077c35855bd7785cbb0444d6768e50cec98b0f88541a226032fce773bee60",
    ),
    "base_preparation": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json",
        "e6fab51219fba6d108bd6b858f2467a9a0e54c5fc22890f8abd7c318742f8e54",
    ),
    "normalization": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/capture/D0/normalization.json",
        "9eb7fb8b9885958d8b947fd20727ae301813df068ec3dcaf1c3d8865180fedd9",
    ),
}


def validate_config(config):
    fixed = dict(
        schema="s2a.recovery-candidate-data.v1",
        output="outputs/recovery_candidate_data_20261116",
        seed=42,
        sampling_roots=[101, 102],
        seed_namespace=SEED_NAMESPACE,
        recovery_cases=24,
        source_groups=6,
        original_windows=72,
        event_windows=36,
        horizon=40,
        base_training_anchors=150,
        base_training_sources=30,
        fixed_normalization_id=31,
        fixed_projection_id=0,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
        **{key: dict(path=path, sha256=sha) for key, (path, sha) in REFERENCES.items()},
    )
    prefix.require(
        pilot._same_typed(config, fixed), "The fixed candidate sources and zero-execution scope are required"
    )


def source_key(source):
    return source["task_index"], source["demo"], source["initial_state_sha256"]


def image_pixels(evidence, ref):
    path = evidence.path(ref)
    with Image.open(path) as image:
        prefix.require(image.mode == "RGB" and image.size == (640, 960), "The actual current RGB canvas is required")
        pixels = np.asarray(image).copy()
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def require_qa():
    qa_ref = prefix.reference(QA)
    qa = prefix.read(qa_ref)
    sources = snapshot(ROOT)["sources"]
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["configured_parallelism"] == 64
        and qa["skipped_count"] == 0
        and not qa["collection"]["collection_errors"]
        and sources == prefix.read(qa["source_before_ref"])["sources"],
        "A source-matched 64-worker CPU regression is required",
    )
    return qa_ref, sources


def candidate_inputs(evidence):
    config_ref = prefix.reference(CONFIG)
    config = evidence.json(config_ref)
    validate_config(config)
    refs = {key: preparation.configured_ref(config[key]) for key in REFERENCES}
    image_audit, dry, h40, union, base, normalization = (evidence.json(refs[key]) for key in REFERENCES)
    prefix.require(
        image_audit["complete"] is image_audit["passed"] is image_audit["sources_unchanged"] is True
        and h40["complete"] is h40["passed"] is union["complete"] is union["passed"] is True
        and dry["complete"] is dry["passed"] is True
        and image_audit["training_authorized"] is h40["training_authorized"] is union["training_authorized"] is False
        and dry["physics_steps"] == dry["gpu_calls"] == dry["world_model_calls"] == 0
        and h40["validation_identity"]["identity_sets_disjoint"] is True
        and h40["validation_identity"]["fresh_future_labels_read"] is False
        and h40["validation_identity"]["validation_future_labels_read"] is False
        and h40["normalization_ref"] == refs["normalization"]
        and len(h40["windows"]) == config["original_windows"]
        and len(image_audit["cases"]) == len(dry["inputs"]["cases"]) == config["recovery_cases"]
        and image_audit["images"] == config["event_windows"]
        and len(base["training_anchors"]) == config["base_training_anchors"]
        and len(base["training_requests"]) == 300,
        "The completed training-only recovery, image and validation evidence must match",
    )
    return config, config_ref, refs, image_audit, dry, h40, union, base, normalization


def build_candidates(evidence, image_audit, dry, h40, union, base, normalization):
    """Reconstruct all labels and identities while retaining every registered window."""
    cases = dry["inputs"]["cases"]
    base_sources = {source_key(row["source_identity"]) for row in base["training_anchors"]}
    prefix.require(len(base_sources) == 30, "The 150 original anchors must retain all 30 training sources")
    old_seeds = {row["sampling_seed"] for row in base["training_requests"]}
    prefix.require(len(old_seeds) == 300, "The original 300 D0 seeds must be distinct")
    fresh = {source_key(row) for row in base["exclusion"]["fresh_sources"]}
    fresh_keys = {(task, demo) for task, demo, _ in fresh}
    fresh_hashes = {digest for _, _, digest in fresh}
    validation_keys = {tuple(row) for row in base["exclusion"]["validation_demo_keys"]}
    prefix.require(len(fresh) == 12 and len(validation_keys) == 6, "All held-out metadata identities are required")
    source_groups = set()
    for index, row in enumerate(cases):
        source = row["source"]
        ident = dict(
            task_index=source["source"]["task_index"],
            demo=source["source"]["demo"],
            initial_state_sha256=source["initial_state_sha256"],
        )
        key = source_key(ident)
        prefix.require(
            row["index"] == source["index"] == h40["cases"][index]["case_index"] == index
            and h40["cases"][index]["source_identity"] == ident
            and h40["cases"][index]["variant"] == source["variant"]
            and key in base_sources
            and key[:2] not in fresh_keys | validation_keys
            and key[2] not in fresh_hashes,
            "Each recovery variant must remain in its original training source, outside held-out metadata",
        )
        source_groups.add(key)
    prefix.require(len(source_groups) == 6, "Exactly six already-known training sources are required")

    windows = []
    for row in h40["windows"]:
        windows.append(
            dict(
                case_index=row["case_index"],
                anchor_step=row["anchor_step"],
                kind="original",
                image_ref=row["image_ref"],
                endpoint=row["current_endpoint"],
                label=row,
            )
        )
    for case in image_audit["cases"]:
        for row in case["images"]:
            windows.append(
                dict(
                    case_index=case["case_index"],
                    anchor_step=row["anchor_step"],
                    kind="event_pre",
                    image_ref=row["image_ref"],
                    endpoint=row["endpoint"],
                    label=row["label_check"],
                )
            )
    windows.sort(key=lambda item: (item["case_index"], item["anchor_step"]))
    expected = [(index, frame) for index in range(24) for frame in sorted((0, 16, 56, *roster.ANCHORS[index]))]
    prefix.require(
        [(row["case_index"], row["anchor_step"]) for row in windows] == expected,
        "The full 108-window roster must match",
    )
    multiplicity = transition_multiplicities([(row["case_index"], row["anchor_step"]) for row in windows])
    prefix.require(
        abs(float(np.sum(1.0 / multiplicity)) - union["totals"]["union"]["unique_case_transition_rows"]) < 1e-9,
        "Window multiplicity must reproduce the actual case/time union",
    )
    baseline_images = []
    for row in base["training_anchors"]:
        baseline_images.append(
            dict(
                source_identity=row["source_identity"],
                anchor_frame=row["anchor_frame"],
                image_ref=row["image_ref"],
                rgb_sha256=image_pixels(evidence, row["image_ref"]),
            )
        )

    records, physical_rows, normalized_rows, flag_rows = [], [], [], []
    candidate_seeds, legacy_seeds, current_case, data, tool = set(), [], None, None, None
    for array_row, row in enumerate(windows):
        index, frame = row["case_index"], row["anchor_step"]
        spec, source = cases[index], cases[index]["source"]
        if index != current_case:
            current_case = index
            data = load_data(evidence, spec["arrays_ref"])
            tool = np.asarray(
                evidence.json(source["original_identity"]["refs"]["tool"])["source_from_tcp"], dtype=np.float64
            )
        identity = h40["cases"][index]["source_identity"]
        prefix.require(
            row["endpoint"]["endpoint_step"] == frame
            and row["endpoint"]["sim_time"] == float(data["sim_time"][frame])
            and row["endpoint"]["sim_state_sha256"]
            == hashlib.sha256(np.ascontiguousarray(data["sim_states"][frame]).tobytes()).hexdigest(),
            "Every image must retain its actual original endpoint",
        )
        label = audit_window(data, frame, tool, normalization)
        label.pop("observed_openings")
        label.pop("actual_gripper_commands")
        prefix.require(
            all(row["label"][key] == value for key, value in label.items()),
            "The reconstructed original H40 label must match its audit",
        )
        physical = np.asarray(data["action128"][frame : frame + 40], dtype=np.float64).copy()
        normalized = normalize_active10(physical, **normalization)["normalized_active10"]
        flags = clipping_flags(physical, normalization)
        rotations = (data["source_pose"][frame : frame + 40] @ tool)[:, :3, :3]
        effect = clipping_translation_effect(physical, rotations, normalization)
        prefix.require(
            np.array_equal(flags[:, :3], effect["clipped_mask"]) and int(flags.sum()) == label["clipped_count"],
            "Clipping flags must match the fixed original normalization",
        )
        seed_identity = dict(
            **identity,
            case_index=index,
            variant=source["variant"],
            anchor_step=frame,
            image_sha256=row["image_ref"]["sha256"],
            trajectory_sha256=spec["arrays_ref"]["sha256"],
        )
        seeds = [candidate_seed(seed_identity, root) for root in (101, 102)]
        for seed in seeds:
            value = seed["sampling_seed"]
            prefix.require(
                value not in candidate_seeds | old_seeds | {42, 43}, "Candidate noise identity collided; no retry"
            )
            candidate_seeds.add(value)
        old_anchor = AnchorIdentity(SourceIdentity(**identity), frame)
        legacy_seeds.extend(derive_noise_seed(old_anchor, root) for root in (101, 102))
        records.append(
            dict(
                array_row=array_row,
                case_index=index,
                source_identity=identity,
                variant=source["variant"],
                anchor_step=frame,
                kind=row["kind"],
                offline_event_selection=row["kind"] == "event_pre",
                image_ref=row["image_ref"],
                rgb_sha256=image_pixels(evidence, row["image_ref"]),
                trajectory_ref=spec["arrays_ref"],
                tool_ref=source["original_identity"]["refs"]["tool"],
                endpoint=row["endpoint"],
                label_sha256=label["normalized_sha256"],
                clipped_count=label["clipped_count"],
                clipped_per_active_slot=label["clipped_per_active_slot"],
                clipping_max_prefix_error_m=effect["max_prefix_error_m"],
                clipping_final_error_m=effect["final_error_m"],
                seed_identity=seed_identity,
                candidate_seeds=seeds,
            )
        )
        physical_rows.append(physical[:, :10])
        normalized_rows.append(normalized)
        flag_rows.append(flags)
    physical = np.stack(physical_rows)
    normalized = np.stack(normalized_rows)
    flags = np.stack(flag_rows)
    prefix.require(
        physical.shape == normalized.shape == flags.shape == (108, 40, 10)
        and int(flags.sum()) == h40["window_clipping_total"] + image_audit["candidate_clipped_count"]
        and len(candidate_seeds) == 216,
        "The full candidate labels, clipped flags and proposed seed identities must close",
    )
    by_pixel = defaultdict(list)
    for row in baseline_images:
        by_pixel[row["rgb_sha256"]].append(
            dict(kind="base", source_identity=row["source_identity"], anchor_step=row["anchor_frame"])
        )
    for row in records:
        by_pixel[row["rgb_sha256"]].append(
            dict(
                kind="recovery",
                case_index=row["case_index"],
                anchor_step=row["anchor_step"],
                array_row=row["array_row"],
                source_identity=row["source_identity"],
            )
        )
    collisions = []
    for digest, group in sorted(by_pixel.items()):
        if len(group) < 2:
            continue
        indices = [row["array_row"] for row in group if row["kind"] == "recovery"]
        maximum = max(
            (
                float(np.max(np.abs(normalized[a].astype(np.float64) - normalized[b].astype(np.float64))))
                for i, a in enumerate(indices)
                for b in indices[i + 1 :]
            ),
            default=None,
        )
        collisions.append(dict(rgb_sha256=digest, members=group, recovery_label_max_abs=maximum))
    metrics = dict(
        total_recovery_windows=len(records),
        original_windows=sum(row["kind"] == "original" for row in records),
        event_windows=sum(row["kind"] == "event_pre" for row in records),
        window_label_rows=int(multiplicity.size),
        unique_case_transition_rows=union["totals"]["union"]["unique_case_transition_rows"],
        repeated_case_transition_rows=int(multiplicity.size - union["totals"]["union"]["unique_case_transition_rows"]),
        clipped_scalars=int(flags.sum()),
        clipped_per_active_slot=flags.sum(axis=(0, 1)).tolist(),
        max_clipping_prefix_error_m=max(row["clipping_max_prefix_error_m"] for row in records),
        max_clipping_final_error_m=max(row["clipping_final_error_m"] for row in records),
        base_and_recovery_images=len(baseline_images) + len(records),
        distinct_rgb_images=len(by_pixel),
        rgb_collision_groups=collisions,
        legacy_seed_requests=len(legacy_seeds),
        legacy_distinct_seeds=len(set(legacy_seeds)),
        legacy_overlap_with_base_seeds=len(set(legacy_seeds) & old_seeds),
        candidate_distinct_seeds=len(candidate_seeds),
        candidate_overlap_with_base_seeds=len(candidate_seeds & old_seeds),
    )
    return records, baseline_images, physical, normalized, flags, multiplicity, metrics


def prepare():
    prefix.environment()
    original.require_no_simulator_import()
    qa_ref, before = require_qa()
    evidence = Evidence()
    for path in (
        PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/wam/training/recovery_candidate_data.py",
        ROOT / "tests/research/wam/training/test_recovery_candidate_data.py",
        ROOT / "tests/research/wam/training/test_recovery_candidate_preparation.py",
    ):
        evidence.path(prefix.reference(path))
    config, config_ref, refs, image_audit, dry, h40, union, base, normalization = candidate_inputs(evidence)
    records, baseline, physical, normalized, flags, multiplicity, metrics = build_candidates(
        evidence, image_audit, dry, h40, union, base, normalization
    )
    evidence.verify()
    prefix.require(snapshot(ROOT)["sources"] == before, "Sources changed during candidate preparation")
    original.require_no_simulator_import()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "labels.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            physical_active10=physical,
            normalized_active10=normalized,
            clipped_mask=flags,
            case_transition_multiplicity=multiplicity,
        )
    labels_ref = prefix.reference(OUTPUT / "labels.npz")
    return publish_result(
        OUTPUT / "manifest.json",
        dict(
            complete=True,
            passed=True,
            schema=config["schema"],
            config_ref=config_ref,
            qa_ref=qa_ref,
            input_refs=list(evidence.refs.values()),
            source_refs=refs,
            labels_ref=labels_ref,
            windows=records,
            base_images=baseline,
            metrics=metrics,
            selection_uses_future_training_labels=True,
            unknown_near_duplicates_excluded=False,
            world_model_calls=0,
            gpu_calls=0,
            physical_steps=0,
            fits=0,
            fresh_cases=0,
            training_authorized=False,
        ),
    )


if __name__ == "__main__":
    print(prepare(), flush=True)
