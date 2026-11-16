"""Independently verify the CPU-only recovery candidate data package."""

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from s2a.infrastructure.publication import publish_result
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence, load_data
from scripts.verification.wam.training.verify_broader_recovery_h40 import scalar_normalization
from scripts.wam.training.clean import prepare_recovery_candidate_data as primary

ROOT = prefix.REPO
OUTPUT = primary.OUTPUT


def scalar_clipping_effect(physical, rotations, normalization):
    """Independently accumulate scalar FP32 clip loss in the measured base frame."""
    transform = normalization["transform"]
    total = np.zeros(3, dtype=np.float64)
    maximum = 0.0
    for step in range(40):
        local = np.zeros(3, dtype=np.float64)
        for slot in range(3):
            scale = np.float32(transform["scale"][slot])
            product = np.float32(np.float32(physical[step, slot]) * scale)
            value = np.float32(product + np.float32(transform["bias"][slot]))
            clipped = np.clip(value, np.float32(transform["clip_min"]), np.float32(transform["clip_max"]))
            local[slot] = (float(clipped) - float(value)) / float(scale)
        total = total + rotations[step] @ local
        maximum = max(maximum, float(np.linalg.norm(total)))
    return maximum, float(np.linalg.norm(total))


def pixel_hash(evidence, ref):
    with Image.open(evidence.path(ref)) as image:
        prefix.require(image.mode == "RGB" and image.size == (640, 960), "Actual RGB canvas required")
        return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def verify_content(evidence, manifest, arrays, inputs):
    config, config_ref, refs, images, dry, h40, union, base, normalization = inputs
    prefix.require(
        manifest["complete"] is manifest["passed"] is True
        and manifest["schema"] == config["schema"]
        and manifest["config_ref"] == config_ref
        and manifest["source_refs"] == refs
        and manifest["training_authorized"] is False
        and manifest["selection_uses_future_training_labels"] is True
        and manifest["unknown_near_duplicates_excluded"] is False
        and manifest["world_model_calls"]
        == manifest["gpu_calls"]
        == manifest["physical_steps"]
        == manifest["fits"]
        == manifest["fresh_cases"]
        == 0
        and set(arrays) == {"physical_active10", "normalized_active10", "clipped_mask", "case_transition_multiplicity"},
        "The no-fit candidate manifest and exact array roster are required",
    )
    physical, normalized, flags, multiplicity = (
        arrays[key]
        for key in ("physical_active10", "normalized_active10", "clipped_mask", "case_transition_multiplicity")
    )
    prefix.require(
        physical.shape == normalized.shape == flags.shape == (108, 40, 10)
        and physical.dtype == np.float64
        and normalized.dtype == np.float32
        and flags.dtype == np.bool_
        and multiplicity.shape == (108, 40)
        and multiplicity.dtype == np.int32,
        "The candidate arrays must have the fixed numeric layouts",
    )
    expected = {}
    for row in h40["windows"]:
        expected[(row["case_index"], row["anchor_step"])] = ("original", row["image_ref"], row["current_endpoint"], row)
    for case in images["cases"]:
        for row in case["images"]:
            key = (case["case_index"], row["anchor_step"])
            prefix.require(key not in expected, "Event and original anchors must not alias")
            expected[key] = ("event_pre", row["image_ref"], row["endpoint"], row["label_check"])
    keys = sorted(expected)
    rows = manifest["windows"]
    prefix.require(
        len(keys) == len(rows) == 108 and [(r["case_index"], r["anchor_step"]) for r in rows] == keys,
        "Every fixed case/anchor must occur once",
    )
    counts = Counter((case, step) for case, start in keys for step in range(start, start + 40))
    prefix.require(
        len(counts) == union["totals"]["union"]["unique_case_transition_rows"],
        "The independently counted transition union differs",
    )
    old_seeds = {r["sampling_seed"] for r in base["training_requests"]}
    candidate_seeds, legacy_seeds, rgb_groups = set(), [], defaultdict(list)
    last_case, data, tool = None, None, None
    maxima, finals = [], []
    for index, row in enumerate(rows):
        case, start = keys[index]
        spec, original_case = dry["inputs"]["cases"][case], h40["cases"][case]
        source = spec["source"]
        kind, image_ref, endpoint, saved_label = expected[(case, start)]
        if last_case != case:
            last_case = case
            data = load_data(evidence, spec["arrays_ref"])
            tool = np.asarray(
                evidence.json(source["original_identity"]["refs"]["tool"])["source_from_tcp"], dtype=np.float64
            )
        prefix.require(
            row["array_row"] == index
            and row["kind"] == kind
            and row["offline_event_selection"] is (kind == "event_pre")
            and row["source_identity"] == original_case["source_identity"]
            and row["variant"] == source["variant"]
            and row["image_ref"] == image_ref
            and row["trajectory_ref"] == spec["arrays_ref"]
            and row["tool_ref"] == source["original_identity"]["refs"]["tool"]
            and row["endpoint"] == endpoint
            and endpoint["sim_time"] == float(data["sim_time"][start])
            and endpoint["sim_state_sha256"]
            == hashlib.sha256(np.ascontiguousarray(data["sim_states"][start]).tobytes()).hexdigest(),
            "Candidate identity, current image and original actual endpoint differ",
        )
        actual = np.asarray(data["action128"][start : start + 40], dtype=np.float64)
        rebuilt, rebuilt_flags = scalar_normalization(actual, normalization)
        prefix.require(
            np.array_equal(physical[index], actual[:, :10])
            and np.array_equal(normalized[index], rebuilt)
            and np.array_equal(flags[index], rebuilt_flags)
            and row["label_sha256"] == saved_label["normalized_sha256"] == hashlib.sha256(rebuilt.tobytes()).hexdigest()
            and row["clipped_count"] == int(rebuilt_flags.sum()) == saved_label["clipped_count"]
            and row["clipped_per_active_slot"] == rebuilt_flags.sum(axis=0).tolist()
            and np.array_equal(multiplicity[index], [counts[(case, step)] for step in range(start, start + 40)]),
            "Measured labels, scalar normalization, clipping or overlap counts differ",
        )
        rotations = (data["source_pose"][start : start + 40] @ tool)[:, :3, :3]
        maximum, final = scalar_clipping_effect(actual, rotations, normalization)
        prefix.require(
            abs(maximum - row["clipping_max_prefix_error_m"]) < 1e-12
            and abs(final - row["clipping_final_error_m"]) < 1e-12,
            "Independent scalar clipping impact differs",
        )
        maxima.append(maximum)
        finals.append(final)
        identity = dict(
            **original_case["source_identity"],
            case_index=case,
            variant=source["variant"],
            anchor_step=start,
            image_sha256=image_ref["sha256"],
            trajectory_sha256=spec["arrays_ref"]["sha256"],
        )
        prefix.require(
            row["seed_identity"] == identity and len(row["candidate_seeds"]) == 2,
            "The complete variant-specific seed identity is required",
        )
        for root, saved in zip((101, 102), row["candidate_seeds"], strict=True):
            canonical = json.dumps(
                dict(namespace="s2a.recovery-candidate.v1", identity=identity, root_seed=root),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            digest = hashlib.sha256(canonical.encode("ascii")).digest()
            seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
            prefix.require(
                saved
                == dict(root_seed=root, sampling_seed=seed, identity_sha256=digest.hex(), canonical_json=canonical)
                and seed not in candidate_seeds | old_seeds | {42, 43},
                "Candidate seed collision or noncanonical identity",
            )
            candidate_seeds.add(seed)
            old = json.dumps(
                dict(task=identity["task_index"], demo=identity["demo"], anchor_frame=start, root_seed=root),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
            legacy_seeds.append(int.from_bytes(hashlib.sha256(old).digest()[:8], "big") & ((1 << 63) - 1))
        rgb = pixel_hash(evidence, image_ref)
        prefix.require(rgb == row["rgb_sha256"], "Recovery current RGB pixels changed")
        rgb_groups[rgb].append(
            dict(
                kind="recovery",
                case_index=case,
                anchor_step=start,
                array_row=index,
                source_identity=row["source_identity"],
            )
        )
    prefix.require(
        len(manifest["base_images"]) == len(base["training_anchors"]) == 150,
        "The original training image denominator differs",
    )
    base_groups = defaultdict(list)
    for saved, actual in zip(manifest["base_images"], base["training_anchors"], strict=True):
        rgb = pixel_hash(evidence, actual["image_ref"])
        prefix.require(
            saved
            == dict(
                source_identity=actual["source_identity"],
                anchor_frame=actual["anchor_frame"],
                image_ref=actual["image_ref"],
                rgb_sha256=rgb,
            ),
            "Original training current image or identity changed",
        )
        base_groups[rgb].append(
            dict(kind="base", source_identity=saved["source_identity"], anchor_step=saved["anchor_frame"])
        )
    collisions = []
    for digest in sorted(set(base_groups) | set(rgb_groups)):
        group = base_groups[digest] + rgb_groups[digest]
        if len(group) < 2:
            continue
        indices = [r["array_row"] for r in group if r["kind"] == "recovery"]
        maximum = max(
            (
                float(np.max(np.abs(normalized[a].astype(np.float64) - normalized[b].astype(np.float64))))
                for j, a in enumerate(indices)
                for b in indices[j + 1 :]
            ),
            default=None,
        )
        collisions.append(dict(rgb_sha256=digest, members=group, recovery_label_max_abs=maximum))
    metrics = manifest["metrics"]
    prefix.require(
        metrics["total_recovery_windows"] == 108
        and metrics["original_windows"] == 72
        and metrics["event_windows"] == 36
        and metrics["window_label_rows"] == 4320
        and metrics["unique_case_transition_rows"] == len(counts)
        and metrics["repeated_case_transition_rows"] == 4320 - len(counts)
        and metrics["clipped_scalars"] == int(flags.sum())
        and metrics["clipped_per_active_slot"] == flags.sum(axis=(0, 1)).tolist()
        and abs(metrics["max_clipping_prefix_error_m"] - max(maxima)) < 1e-12
        and abs(metrics["max_clipping_final_error_m"] - max(finals)) < 1e-12
        and metrics["base_and_recovery_images"] == 258
        and metrics["distinct_rgb_images"] == len(set(base_groups) | set(rgb_groups))
        and metrics["rgb_collision_groups"] == collisions
        and metrics["legacy_seed_requests"] == len(legacy_seeds)
        and metrics["legacy_distinct_seeds"] == len(set(legacy_seeds))
        and metrics["legacy_overlap_with_base_seeds"] == len(set(legacy_seeds) & old_seeds)
        and metrics["candidate_distinct_seeds"] == len(candidate_seeds) == 216
        and metrics["candidate_overlap_with_base_seeds"] == 0,
        "The independent image, clipping, overlap and seed summaries must match",
    )
    return dict(
        windows=len(rows),
        labels=4320,
        unique_transitions=len(counts),
        images=258,
        candidate_seeds=len(candidate_seeds),
        clipped_scalars=int(flags.sum()),
    )


def verify():
    prefix.environment()
    original.require_no_simulator_import()
    qa_ref, before = primary.require_qa()
    evidence = Evidence()
    inputs = primary.candidate_inputs(evidence)
    manifest_ref = prefix.reference(OUTPUT / "manifest.json")
    manifest = evidence.json(manifest_ref)
    prefix.require(manifest["qa_ref"] == qa_ref, "Preparation and verification must share the original QA receipt")
    for ref in manifest["input_refs"]:
        evidence.path(ref)
    with np.load(evidence.path(manifest["labels_ref"]), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    summary = verify_content(evidence, manifest, arrays, inputs)
    evidence.verify()
    prefix.require(snapshot(ROOT)["sources"] == before, "Sources changed during independent verification")
    original.require_no_simulator_import()
    return publish_result(
        OUTPUT / "independent.json",
        dict(
            complete=True,
            passed=True,
            manifest_ref=manifest_ref,
            labels_ref=manifest["labels_ref"],
            qa_ref=qa_ref,
            input_refs=list(evidence.refs.values()),
            summary=summary,
            world_model_calls=0,
            gpu_calls=0,
            physical_steps=0,
            fits=0,
            fresh_cases=0,
            training_authorized=False,
        ),
    )


if __name__ == "__main__":
    try:
        print(verify(), flush=True)
    except Exception as error:
        print(
            publish_result(
                OUTPUT / "verification_failure.json",
                dict(
                    complete=False,
                    passed=False,
                    error=dict(type=type(error).__name__, message=str(error)),
                    automatic_retry=False,
                ),
            ),
            flush=True,
        )
        raise
