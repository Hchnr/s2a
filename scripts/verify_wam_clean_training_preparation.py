"""Independent stdlib-only038 metadata audit; no producer/NumPy/model imports.

Opaque future NPZ files are hashed as bytes, never decoded. Seed derivation,
source identities, sample ordering, provenance and all counters are recomputed
here rather than delegated to preparation or training-data implementations.
Metadata qualification explicitly grants no execution or fresh qualification.
"""

import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("/share/project/eai_pwm/home/hcr/repos/s2a")
RECOVERY = Path("outputs/external_deletion_recovery_20261022")
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
VAL = (
    (1254, 0, "demo_32"),
    (1126, 0, "demo_16"),
    (868, 1, "demo_27"),
    (1139, 1, "demo_4"),
    (1209, 2, "demo_14"),
    (1176, 2, "demo_24"),
)
ORDER = "round_D0_then_D1/task_index/numeric_demo/anchor_frame/root_101_then_102/step_0_to_34/endpoint_0_to_39"
SEED_DESCRIPTION = "UTF8(json.dumps({task:int,demo:canonical_demo_N,anchor_frame:int,root_seed:int},sort_keys=True,separators=(',',':'),ensure_ascii=True)); int.from_bytes(SHA256(payload).digest()[:8],'big') & ((1<<63)-1); reject42/43/collisions_without_retry"
NUMERICS = {
    "features": "hidden1280_then_C0active10",
    "actual_sample": "original_FP32",
    "C0": "sample-sigma*raw_velocity",
    "target": "normalized_native_x0-C0",
    "loss": "direct_clean_no_sigma_weight",
    "center_scale": "train_only_global_vector_L2_RMS",
    "bias_regularized": True,
}
BUDGET = {
    "raw_training_requests": 300,
    "w1_training_requests": 300,
    "state_zero_requests": 3,
    "state_paired_requests": 96,
    "training_and_state_requests": 699,
    "training_and_state_denoise": 24465,
    "development_requests": 900,
    "development_denoise": 31500,
    "total_requests": 1599,
    "total_denoise": 55965,
    "model_initializations": 4,
    "producer_fits": 2,
    "independent_reconstructions": 2,
    "development_physics_steps": 7200,
    "new_teacher_calls": 0,
    "future_rgb_frames": 0,
    "fresh_cases": 0,
}
RESOURCES = {
    "d0_timeout_s": 86400,
    "d1_timeout_s": 86400,
    "state_timeout_s": 43200,
    "development_timeout_s": 172800,
    "fit_timeout_s": 7200,
    "independent_fit_timeout_s": 7200,
    "audit_timeout_s": 1800,
    "http_timeout_s": 300,
    "gpu_release_timeout_s": 120,
    "blas_threads": 8,
    "maximum_fit_block_rows": 1400,
}
REQUEST_KEYS = {
    "action_chunk_size",
    "domain_name",
    "extra",
    "fps",
    "guidance",
    "image_size",
    "model_mode",
    "name",
    "num_frames",
    "num_steps",
    "prompt",
    "seed",
    "shift",
    "vision_path",
}
ACTION_KEYS = {
    "action_dim_mask",
    "action_domain",
    "action_fps",
    "camera_layout",
    "return_video",
    "selected_roles",
    "task_title",
}
ENTRYPOINTS = {
    "collect": "scripts/collect_wam_clean_training.py",
    "fit": "scripts/fit_wam_clean_training.py",
    "independent_fit": "scripts/verify_wam_clean_training_fit.py",
    "state_sampling": "scripts/sample_wam_clean_training_state.py",
    "state_audit": "scripts/verify_wam_clean_training_state.py",
    "development_service": "scripts/serve_wam_clean_training.py",
    "development_owner": "scripts/run_wam_clean_training_development.py",
    "independent_prepare": "scripts/verify_wam_clean_training_preparation.py",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    value = Path(path)
    require(".." not in value.parts, "parent traversal in a declared reference")
    value = value if value.is_absolute() else REPO / value
    if value.is_relative_to(OLD_ROOT):
        relative = value.relative_to(OLD_ROOT)
        candidates = []
        if relative.parts[:2] == ("outputs", "wam_stability_20261021"):
            candidates.append(REPO / RECOVERY / Path(*relative.parts[2:]))
        candidates.append(REPO / relative)
    else:
        candidates = [value]
    for candidate in candidates:
        actual = candidate.resolve()
        require(not actual.is_relative_to(OLD_ROOT), "reference resolves into the deleted historical root")
        if actual.is_file():
            return actual
    raise FileNotFoundError(f"no safe existing metadata artifact: {path}")


class Evidence:
    def __init__(self):
        self.consumed = {}

    def check(self, reference):
        require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "invalid declared reference")
        require(
            isinstance(reference["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]), "invalid SHA"
        )
        path = resolve(reference["path"])
        require(file_sha(path) == reference["sha256"], f"artifact SHA differs: {reference['path']}")
        self.consumed[str(path)] = reference["sha256"]
        return path

    def json(self, reference):
        path = self.check(reference)
        require(path.suffix == ".json", "JSON parser refuses an opaque future/archive file")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "JSON changed during consumption")
        return json.loads(raw)

    def png(self, reference):
        path = self.check(reference)
        with path.open("rb") as stream:
            header = stream.read(24)
        require(
            len(header) == 24
            and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR"
            and struct.unpack(">II", header[16:24]) == (640, 960),
            "current PNG signature/IHDR differs",
        )
        return dict(path=str(path), sha256=reference["sha256"])

    def unchanged(self):
        require(
            all(file_sha(path) == digest for path, digest in self.consumed.items()),
            "consumed artifact changed during audit",
        )


def identity(record):
    task, demo, digest = record["task_index"], record["demo"], record["initial_state_sha256"]
    require(type(task) is int and task in (0, 1, 2), "invalid task identity")
    require(isinstance(demo, str) and re.fullmatch(r"demo_(0|[1-9][0-9]*)", demo), "noncanonical native demo identity")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "missing actual initial SHA")
    return task, demo, digest


def seed_identity(task, demo, frame, root):
    require(
        type(task) is int
        and task in (0, 1, 2)
        and isinstance(demo, str)
        and re.fullmatch(r"demo_(0|[1-9][0-9]*)", demo)
        and type(frame) is int
        and frame >= 0
        and type(root) is int
        and root in (101, 102),
        "invalid anchor/root for noise derivation",
    )
    payload = json.dumps(
        dict(task=task, demo=demo, anchor_frame=frame, root_seed=root),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    seed = int.from_bytes(bytes.fromhex(digest)[:8], "big") & ((1 << 63) - 1)
    require(seed not in (42, 43), "derived training seed collides with evaluation")
    return payload, digest, seed


def budget_and_readiness(p):
    config = p["config"]
    require(
        config["schema_version"] == "s2a.clean-training-config.v1"
        and config["seed"] == 42
        and config["bindings_complete"] is True,
        "unbound config or seed",
    )
    require(
        config["budget"] == p["budget"] == BUDGET and config["resources"] == p["resources"] == RESOURCES,
        "fixed complete038 compute/resource budget differs",
    )
    require(
        config["noise_roots"] == [101, 102]
        and config["evaluation_seeds"] == [42, 43]
        and config["validation_episodes"] == [row[0] for row in VAL]
        and config["ridge"] == 0.1
        and config["feature_terms"] == []
        and config["gpu_uuid"] == GPU
        and config["automatic_retry"] is False,
        "training/loss/noise/model contract differs",
    )
    require(
        config["numerical_contract"] == NUMERICS and p["noise_seed_algorithm"] == SEED_DESCRIPTION,
        "registered numerical/noise derivation contract differs",
    )
    require(config["execution_entrypoints"] == ENTRYPOINTS, "registered eight execution entrypoints differ")
    require(
        config["preparation_only"] is True
        and config["execution_authorized"] is p["execution_authorized"] is p["fresh_performance_authorized"] is False,
        "metadata preparation falsely authorizes execution",
    )
    readiness = p["readiness"]
    require(
        readiness["metadata_contract_complete"] is True
        and readiness["ready_for_execution"] is False
        and readiness["entrypoint_presence_is_not_qualification"] is True,
        "entrypoint presence or metadata was promoted to execution qualification",
    )
    require(
        set(readiness["missing_entrypoints"]) <= set(ENTRYPOINTS.values()) and readiness["reasons"],
        "missing-entrypoint evidence differs",
    )
    for name in (
        "future_label_archive_decodes",
        "new_model_initializations",
        "new_model_calls",
        "new_fits",
        "physics_steps",
        "fresh_performance_reads",
    ):
        require(type(p[name]) is int and p[name] == 0, "preparation consumed a forbidden execution/label budget")
    require(
        p["original037_owner_passed"] is False and p["full_weight_bytes_rehashed"] is False,
        "historical False or weight-verification scope was rewritten",
    )


def verify_sources(p, evidence):
    root = Path(p["root"]).resolve()
    expected_root = (REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023").resolve()
    require(
        root == expected_root and p["config"]["output_dir"] == str(root.relative_to(REPO)), "038 root identity differs"
    )
    require(evidence.json(p["config_ref"]) == p["config"], "prepared config differs from actual file")
    evidence.check(p["plan_ref"])
    require(
        Path(p["plan_ref"]["path"]).resolve() == (REPO / p["config"]["plan_path"]).resolve(), "038 plan binding differs"
    )
    expected_sources = {(REPO / name).resolve() for name in p["config"]["source_paths"]}
    expected_sources.update((resolve(p["config_ref"]["path"]), resolve(p["plan_ref"]["path"])))
    require(
        {Path(name).resolve() for name in p["source_sha256"]} == expected_sources,
        "complete preparation source roster differs",
    )
    expected_snapshots = {}
    for name, digest in p["source_sha256"].items():
        path = evidence.check(dict(path=name, sha256=digest))
        require(path.is_relative_to(REPO), "preparation source escaped S2A")
        target = root / "preparation/source" / path.relative_to(REPO)
        expected_snapshots[str(target)] = digest
        evidence.check(dict(path=str(target), sha256=digest))
    require(p["snapshot_sha256"] == expected_snapshots, "source/snapshot mapping differs")
    require(set(p["input_sha256"]) == set(p["input_read_kinds"]), "input SHA/read-scope roster differs")
    require(
        set(p["input_read_kinds"].values())
        <= {"metadata", "opaque_future_label_sha_only", "current_png", "qualified_model_source"},
        "unregistered input read scope",
    )
    for name, digest in p["input_sha256"].items():
        evidence.check(dict(path=name, sha256=digest))
    return dict(source_files=len(expected_sources), snapshots=len(expected_snapshots), inputs=len(p["input_sha256"]))


def completed(value, message):
    require(value.get("passed") is True and value.get("complete") is True, message)


def owned(command, evidence):
    exit_record, release, launch = (evidence.json(command[key]) for key in ("exit", "release", "launch"))
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False,
        "normal owned wait/reap/release is missing",
    )
    require(
        (release["pid"], release["start_ticks"]) == (launch["child"]["pid"], launch["child"]["start_ticks"]),
        "owned PID/birth identity differs",
    )


def qualification_chain(p, evidence):
    refs = p["prerequisite_refs"]
    require(refs == p["config"]["refs"], "qualification refs differ from frozen config")
    q = {name: evidence.json(ref) for name, ref in refs.items()}
    for name in (
        "settled_collection",
        "settled_audit",
        "settled_owner",
        "settled_peer",
        "bridge_qualification",
        "native_mask_review",
        "native_mask_owner",
        "native_mask_peer",
    ):
        completed(q[name], f"incomplete qualification: {name}")
    peer, audit, owner = (q[name] for name in ("settled_peer", "settled_audit", "settled_owner"))
    require(
        peer["prepared_ref"] == audit["prepared_ref"] == refs["settled_prepared"]
        and peer["verified_ref"] == owner["verified_ref"] == refs["settled_audit"]
        and peer["owner_ref"] == refs["settled_owner"],
        "034 qualification lineage differs",
    )
    quality = peer["new_quality"]
    require(
        quality["total_cases"] == quality["stable_native_cases"] == peer["native_prefix_exact_cases"] == 30
        and quality["stable_native_by_task"] == {"0": 10, "1": 10, "2": 10}
        and quality["eligible_for_model_training_registration"]
        is audit["eligible_for_model_training_registration"]
        is True
        and peer["sources_and_inputs_before_after_exact"] is True,
        "034 all30 qualification failed",
    )
    require(
        peer["preserved_original034_owner_passed"] is False
        and evidence.json(peer["original_failed_owner_ref"])["passed"] is False,
        "original034 False was changed",
    )
    for name in ("worker_lifecycle", "audit_lifecycle"):
        value = peer[name]
        require(
            value["actual_wait"] is True
            and value["returncode"] == 0
            and value["reaped"] is True
            and value["signal_sent"] is False,
            "034 normal lifecycle is incomplete",
        )
    qualified, failed, gpu = (q[name] for name in ("bridge_qualification", "bridge_failed_owner", "bridge_gpu_after"))
    require(
        qualified["bridge_qualified"] is True
        and qualified["prepared_ref"] == refs["bridge_prepared"]
        and qualified["actual_requests"] == 7
        and qualified["actual_denoise_calls"] == 245
        and qualified["parameters_before"] == qualified["parameters_after"],
        "037 complete frozen-model7/245 evidence differs",
    )
    require(
        failed["passed"] is False and failed["gpu_after_ref"] == refs["bridge_gpu_after"],
        "original037 False was changed",
    )
    owned(failed["owned_commands"][0], evidence)
    require(
        gpu["released"] is True and gpu["memory_mib"] == 0 and gpu["compute_apps"] == [] and gpu["uuid"] == GPU,
        "037 GPU release is incomplete",
    )
    registration, review, corrected, final = (
        q[name] for name in ("native_mask_prepared", "native_mask_review", "native_mask_owner", "native_mask_peer")
    )
    require(
        corrected["bridge_qualified"] is review["bridge_qualified"] is final["bridge_qualified"] is True
        and corrected["registration_ref"] == refs["native_mask_prepared"]
        and corrected["review_ref"] == refs["native_mask_review"]
        and corrected["preserved_original_owner_passed"] is False,
        "039 joint qualification status differs",
    )
    for name, target in (
        ("original_failed_owner_ref", "bridge_failed_owner"),
        ("original_prepared_ref", "bridge_prepared"),
        ("qualification_ref", "bridge_qualification"),
    ):
        require(corrected[name] == registration[name] == refs[target], "039 original source binding differs")
    require(
        review["actual_online_points_verified"] == final["counts"]["points"] == 245
        and review["nonzero_changed_active_values"] == final["counts"]["changed_values"] == 102
        and final["counts"]["native_initial_masks"] == 7
        and final["consumed_source_and_artifacts_before_after_exact"] is True,
        "039 independent full-denominator evidence differs",
    )
    for name, target in (
        ("owner039_ref", "native_mask_owner"),
        ("formal_ref", "native_mask_review"),
        ("worker_ref", "bridge_qualification"),
        ("prepared_ref", "bridge_prepared"),
    ):
        require(final[name] == refs[target], "039 peer source identity differs")
    for name in ("native_mask_owner", "native_mask_review"):
        ref = refs[name]
        require(final["consumed_sha256"].get(ref["path"]) == ref["sha256"], "039 peer consumed-source SHA differs")
    require(len(corrected["owned_commands"]) == 1, "039 CPU owner budget differs")
    owned(corrected["owned_commands"][0], evidence)
    bridge = q["bridge_prepared"]
    require(
        p["model_config"] == bridge["config"]
        and p["qualified_model_source_sha256"] == bridge["source_sha256"]
        and p["weight_files"] == bridge["weight_files"]
        and len(p["weight_files"]) == 163,
        "actual qualified model/source163-stat identity differs",
    )
    for name, digest in p["qualified_model_source_sha256"].items():
        evidence.check(dict(path=name, sha256=digest))
    for name, expected in p["weight_files"].items():
        actual = Path(name).stat()
        require(
            actual.st_size == expected["bytes"] and actual.st_mtime_ns == expected["mtime_ns"],
            "checkpoint weight stat changed",
        )
    require(p["zero_identity_cases"] == bridge["requests3"], "three historical zero cases were replaced")
    return q


def request_binding(actual, template, *, name, image, seed, task):
    require(
        set(actual) == set(template) == REQUEST_KEYS
        and set(actual["extra"]) == {"action128"}
        and set(actual["extra"]["action128"]) == ACTION_KEYS,
        "unregistered/future request field",
    )
    expected = dict(template, name=name, vision_path=image["path"], seed=seed)
    require(actual == expected and actual["prompt"] == task, "request differs beyond current image/name/noise")
    action = actual["extra"]["action128"]
    require(
        actual["action_chunk_size"] == 40
        and actual["num_frames"] == 41
        and actual["num_steps"] == 35
        and actual["guidance"] == 1
        and actual["shift"] == 5
        and actual["fps"] == action["action_fps"] == 20
        and actual["model_mode"] == "wam"
        and action["task_title"] == task
        and action["action_dim_mask"] == [True] * 10 + [False] * 118
        and action["return_video"] is False,
        "current-only fixed sampling/mask contract differs",
    )


def training(p, q, evidence):
    native, collection = q["settled_prepared"], q["settled_collection"]
    records = sorted(native["records"], key=lambda r: (r["task_index"], int(r["demo"][5:])))
    require(
        len(records) == 30 and all(sum(r["task_index"] == t for r in records) == 10 for t in range(3)),
        "30-source task balance differs",
    )
    ids = [identity(record) for record in records]
    require(
        len({row[:2] for row in ids}) == len({row[2] for row in ids}) == 30,
        "duplicate training demo or physical initial identity",
    )
    original = native["original_prepared"]
    fresh = evidence.json(original["fresh_fixtures_ref"])
    fresh_ids = [identity(record) for record in fresh["records"]]
    require(
        len(fresh_ids) == len(set(fresh_ids)) == 12
        and len({row[2] for row in fresh_ids}) == 12
        and all(sum(row[0] == t for row in fresh_ids) == 4 for t in range(3)),
        "fresh metadata exclusion is not twelve unique sources",
    )
    excluded = {
        (row["task_index"], demo) for row in original["selection"]["tasks"] for demo in row["excluded_demo_ids"]
    }
    banned_demos = excluded | {(row[1], row[2]) for row in VAL} | {row[:2] for row in fresh_ids}
    banned_sha = set(original["forbidden_initial_state_sha256"]) | {row[2] for row in fresh_ids}
    require(
        all(row[:2] not in banned_demos and row[2] not in banned_sha for row in ids),
        "training leaks a validation/fresh/historical source or physical alias",
    )
    ex = p["exclusion"]
    require(
        ex["fresh_fixture_ref"] == original["fresh_fixtures_ref"]
        and [identity(row) for row in ex["fresh_sources"]] == fresh_ids
        and {tuple(row) for row in ex["validation_demo_keys"]} == {(r[1], r[2]) for r in VAL}
        and {tuple(row) for row in ex["forbidden_demo_keys"]} == excluded
        and ex["forbidden_initial_state_sha256"] == original["forbidden_initial_state_sha256"]
        and ex["validation_initial_sync_certified"] is False,
        "exclusion scope/unknown synchronization was rewritten",
    )
    cases = {}
    require(
        collection["prepared_ref"] == p["prerequisite_refs"]["settled_prepared"] and len(collection["cases"]) == 30,
        "034 collection denominator differs",
    )
    for row in collection["cases"]:
        case = evidence.json(row["case"])
        completed(case, "034 case incomplete")
        key = identity(case["record"])
        require(key not in cases and case["native_quality_passed"] is True, "duplicate/unqualified native case")
        require(
            case["record"] in records and case["prepared_ref"] == p["prerequisite_refs"]["settled_prepared"],
            "native case/source lineage differs",
        )
        evidence.check(case["arrays"])  # Opaque byte identity only, never NPZ parsing.
        cases[key] = (row["case"], case)
    require(len(p["training_anchors"]) == 150 and len(p["training_requests"]) == 300, "150/300 complete roster differs")
    templates = {row["request"]["prompt"]: row["request"] for row in q["bridge_prepared"]["requests3"]}
    seeds, cursor = [], 0
    for record, key in zip(records, ids, strict=True):
        n = record["native_steps"]
        require(type(n) is int and 43 <= n <= 600, "invalid native trajectory length")
        frames = [j * (n - 40) // 3 for j in range(4)] + [n + 40]
        require(record["anchor_frames"] == frames and len(set(frames)) == 5, "five native/settled anchors differ")
        case_ref, case = cases[key]
        require(
            case["native_steps"] == n and case["endpoint_count"] == n + 81 and case["future_rgb_frames"] == 0,
            "native endpoint metadata differs",
        )
        for field, shape, dtype in (
            ("state128", [n + 81, 128], "float64"),
            ("action128", [n + 81, 128], "float64"),
            ("state_mask", [n + 81, 128], "bool"),
            ("action_mask", [n + 81, 128], "bool"),
            ("action_valid", [n + 81], "bool"),
        ):
            require(
                case["array_index"][field] == dict(shape=shape, dtype=dtype),
                "opaque label array layout metadata differs",
            )
        require([row["frame_index"] for row in case["anchors"]] == frames, "case current anchor order differs")
        for frame, current in zip(frames, case["anchors"], strict=True):
            anchor = p["training_anchors"][cursor]
            expected_name = f"task{key[0]}_{key[1]}_a{frame}"
            require(
                identity(anchor["source_identity"]) == key
                and anchor["anchor_frame"] == frame
                and anchor["name"] == expected_name,
                "training anchor identity/order differs",
            )
            image = evidence.png(current["canvas_png"])
            expected_array = dict(path=str(evidence.check(case["arrays"])), sha256=case["arrays"]["sha256"])
            require(
                anchor["case_ref"] == case_ref
                and anchor["declared_trajectory_ref"] == case["arrays"]
                and anchor["trajectory_ref"] == expected_array
                and anchor["image_ref"] == image,
                "anchor current-image/opaque-label source binding differs",
            )
            require(
                current["render_before"] == current["render_after"]
                and current["canvas_shape"] == [960, 640, 3]
                and current["future_rgb_frames"] == 0
                and current["current_endpoint"]["endpoint_step"] == frame
                and current["current_endpoint"]["sim_state_sha256"] == current["render_before"]["sim_state_sha256"]
                and anchor["current_endpoint"] == current["current_endpoint"],
                "current image is not physics-bound",
            )
            require(
                anchor["future_window_steps"] == 40
                and anchor["future_labels_parsed"] is False
                and anchor["task"] == record["task"]
                and anchor["native_steps"] == n
                and anchor["tool_source_from_tcp"] == original["tool_source_from_tcp"],
                "training label/clock/tool metadata differs",
            )
            for root in (101, 102):
                index = cursor * 2 + (root - 101)
                request = p["training_requests"][index]
                payload, digest, seed = seed_identity(key[0], key[1], frame, root)
                require(
                    (request["task_index"], request["demo"], request["initial_state_sha256"], request["anchor_frame"])
                    == (*key, frame)
                    and request["root_seed"] == root
                    and request["index"] == index
                    and request["canonical_json"] == payload
                    and request["identity_sha256"] == digest
                    and request["sampling_seed"] == seed
                    and request["anchor_name"] == expected_name,
                    "independent SHA/noise/row identity differs",
                )
                name = f"train_{expected_name}_r{root}"
                require(request["name"] == name, "training request name/order differs")
                request_binding(
                    request["request"],
                    templates[record["task"]],
                    name=name,
                    image=image,
                    seed=seed,
                    task=record["task"],
                )
                seeds.append(seed)
            cursor += 1
    require(len(seeds) == len(set(seeds)) == 300 and not set(seeds) & {42, 43}, "training noise collision/leakage")
    require(p["noise_roster"] == dict(training_seeds=seeds, evaluation_seeds=[42, 43]), "actual NoiseRoster differs")
    require(
        p["row_order"] == ORDER
        and p["union_rows"] == 840000
        and p["training_weight_per_row"] == dict(W1="1/420000", W2="1/840000")
        and p["training_rounds"]
        == [
            dict(name="D0", policy="raw", request_indices=list(range(300)), rows=420000),
            dict(name="D1", policy="fixed_W1", request_indices=list(range(300)), rows=420000),
        ],
        "two fixed rounds, full rows or global weights differ",
    )
    return dict(
        training_sources=30, training_anchors=cursor, training_noise_seeds=len(seeds), training_requests=len(seeds)
    )


def validation(p, q, evidence):
    reference = q["bridge_prepared"]["config"]["historical_prepared_ref"]
    require(p["historical_validation_ref"] == reference, "historical six-EP source differs")
    historical = evidence.json(reference)
    require(
        len(p["validation_anchors"]) == 24 and len(p["validation_generations"]) == 96,
        "24/96 complete validation roster differs",
    )
    templates = {row["request"]["prompt"]: row["request"] for row in q["bridge_prepared"]["requests3"]}
    cursor = 0
    for episode, task, demo in VAL:
        links = [row for row in historical["conservative_source_exclusion"]["records"] if row["r2_episode"] == episode]
        require(
            len(links) == 1 and (links[0]["task_index"], links[0]["demo"]) == (task, demo),
            "full validation EP/native demo identity differs",
        )
        original = sorted(
            (a for a in historical["anchors"] if a["episode"] == episode), key=lambda a: a["anchor_index"]
        )
        require(
            len(original) == 4 and [a["anchor_index"] for a in original] == [0, 1, 2, 3],
            "validation EP was truncated/replaced",
        )
        for old in original:
            row = p["validation_anchors"][cursor]
            image = evidence.png(dict(path=old["image_path"], sha256=old["image_sha256"]))
            declared = dict(path=old["arrays_path"], sha256=old["arrays_sha256"])
            actual = dict(path=str(evidence.check(declared)), sha256=declared["sha256"])
            require(
                old["split"] == "validation" and old["valid_future_steps"] == 40,
                "invalid validation future-window metadata",
            )
            for key in (
                "name",
                "episode",
                "task_index",
                "task",
                "anchor_index",
                "anchor_frame",
                "measured_state128",
                "state_mask128",
                "source_episode_id",
            ):
                require(row[key] == old[key], "validation source/current measurement metadata changed")
            require(
                row["task_index"] == task
                and row["source_demo"] == demo
                and row["image_ref"] == image
                and row["declared_arrays_ref"] == declared
                and row["arrays_ref"] == actual
                and row["initial_sync_certified"] is row["future_labels_parsed"] is False,
                "validation/future reference or unknown synchronization was rewritten",
            )
            request_binding(
                row["request_template"],
                templates[row["task"]],
                name=row["name"],
                image=image,
                seed=42,
                task=row["task"],
            )
            for seed in (42, 43):
                for variant in ("raw", "w2"):
                    index = cursor * 4 + (seed - 42) * 2 + (variant == "w2")
                    generation = p["validation_generations"][index]
                    name = f"{row['name']}_{variant}_s{seed}"
                    require(
                        generation["index"] == index
                        and generation["name"] == name
                        and generation["anchor_name"] == row["name"]
                        and generation["variant"] == variant
                        and generation["sampling_seed"] == seed,
                        "validation pair/order/seed changed",
                    )
                    request_binding(
                        generation["request"],
                        templates[row["task"]],
                        name=name,
                        image=image,
                        seed=seed,
                        task=row["task"],
                    )
            cursor += 1
    return dict(validation_episodes=6, validation_anchors=cursor, validation_generations=96)


def audit(prepared_ref):
    evidence = Evidence()
    p = evidence.json(prepared_ref)
    require(p["schema_version"] == "s2a.clean-training-preparation.v1", "wrong038 preparation schema")
    budget_and_readiness(p)
    counts = verify_sources(p, evidence)
    qualified = qualification_chain(p, evidence)
    counts.update(training(p, qualified, evidence))
    counts.update(validation(p, qualified, evidence))
    admitted = dict(p["input_sha256"], **p["source_sha256"], **p["snapshot_sha256"])
    admitted[prepared_ref["path"]] = prepared_ref["sha256"]
    require(
        all(admitted.get(path) == digest for path, digest in evidence.consumed.items()),
        "preparation omitted a consumed metadata/input SHA",
    )
    evidence.unchanged()
    return dict(
        counts=counts,
        metadata_qualified=True,
        execution_ready=False,
        execution_authorized=False,
        missing_entrypoints=p["readiness"]["missing_entrypoints"],
        current_entrypoint_presence={name: (REPO / path).is_file() for name, path in ENTRYPOINTS.items()},
        entrypoint_presence_is_not_execution_qualification=True,
        consumed_sha256=evidence.consumed,
        consumed_before_after_exact=True,
        future_label_archives_decoded=0,
        full_weight_content_rehash=False,
        weight_stat_files=163,
    )


def verify(args):
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "independent038 audit requires seed42 and empty CUDA/LD",
    )
    output = Path(args.output).absolute()
    require(
        output.resolve() == output and output.is_relative_to((REPO / "outputs").resolve()),
        "audit output escaped S2A outputs",
    )
    require(not output.exists(), "refuses existing audit output")
    prepared_ref = dict(path=str(Path(args.prepared).absolute()), sha256=args.prepared_sha256)
    source_ref = dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__))
    receipt = dict(
        schema_version="s2a.clean-training-preparation-independent.v1",
        passed=False,
        complete=False,
        prepared_ref=prepared_ref,
        verifier_ref=source_ref,
        metadata_qualified=False,
        execution_ready=False,
        execution_authorized=False,
        new_model_calls=0,
        new_fit_calls=0,
        new_physics_steps=0,
        new_fresh_cases=0,
        future_label_archives_decoded=0,
    )
    try:
        receipt.update(audit(prepared_ref))
        require(file_sha(__file__) == source_ref["sha256"], "independent checker source changed")
        receipt.update(passed=True, complete=True)
    except Exception as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
