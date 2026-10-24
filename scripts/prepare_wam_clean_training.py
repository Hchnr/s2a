"""Metadata-only preparation for038; never decode future trajectory archives.

This binds completed034/037/039 evidence, current images and opaque label-file
SHA references. It enumerates the fixed roster using clean_training_data only.
The resulting preparation is deliberately not executable: independent runtime,
fit/QR, state and development entrypoint admission must be supplied separately.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import struct
from dataclasses import asdict
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_data import ROW_ORDER, SEED_ALGORITHM, SourceIdentity, build_training_roster, noise_roster
from s2a.relocated_artifacts import RelocatedArtifacts

SCHEMA = "s2a.clean-training-preparation.v1"
CONFIG_SCHEMA = "s2a.clean-training-config.v1"
DEFAULT_CONFIG = REPO / "configs/wam_clean_training_20261023.json"
VAL_EPISODES = (1254, 1126, 868, 1139, 1209, 1176)
VAL_DEMOS = ((0, "demo_16"), (0, "demo_32"), (1, "demo_4"), (1, "demo_27"), (2, "demo_14"), (2, "demo_24"))
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
BUDGET = dict(
    raw_training_requests=300,
    w1_training_requests=300,
    state_zero_requests=3,
    state_paired_requests=96,
    training_and_state_requests=699,
    training_and_state_denoise=24465,
    development_requests=900,
    development_denoise=31500,
    total_requests=1599,
    total_denoise=55965,
    model_initializations=4,
    producer_fits=2,
    independent_reconstructions=2,
    development_physics_steps=7200,
    new_teacher_calls=0,
    future_rgb_frames=0,
    fresh_cases=0,
)
RESOURCES = dict(
    d0_timeout_s=86400,
    d1_timeout_s=86400,
    state_timeout_s=43200,
    development_timeout_s=172800,
    fit_timeout_s=7200,
    independent_fit_timeout_s=7200,
    audit_timeout_s=1800,
    http_timeout_s=300,
    gpu_release_timeout_s=120,
    blas_threads=8,
    maximum_fit_block_rows=1400,
)
REF_NAMES = frozenset(
    (
        "settled_prepared",
        "settled_collection",
        "settled_audit",
        "settled_owner",
        "settled_peer",
        "bridge_prepared",
        "bridge_qualification",
        "bridge_failed_owner",
        "bridge_gpu_after",
        "native_mask_prepared",
        "native_mask_review",
        "native_mask_owner",
        "native_mask_peer",
    )
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


class Evidence:
    """Byte checks and JSON only; no NPZ/NPY decoding API is exposed."""

    def __init__(self):
        self.resolver = RelocatedArtifacts(REPO, "outputs/external_deletion_recovery_20261022")
        self.inputs, self.read_kinds = {}, {}

    def check(self, ref, *, kind="metadata"):
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "requires an exact path/SHA reference")
        require(isinstance(ref["sha256"], str) and len(ref["sha256"]) == 64, "reference SHA is pending or malformed")
        path = self.resolver.resolve_path(ref["path"])
        require(sha(path) == ref["sha256"], f"input SHA changed: {ref['path']}")
        self.inputs[str(path)] = ref["sha256"]
        self.read_kinds[str(path)] = kind
        return dict(path=str(path), sha256=ref["sha256"])

    def json(self, ref):
        actual = self.check(ref)
        require(Path(actual["path"]).suffix == ".json", "metadata JSON reader refuses binary/future arrays")
        data = Path(actual["path"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == actual["sha256"], "metadata changed during read")
        return json.loads(data)

    def png(self, ref):
        actual = self.check(ref, kind="current_png")
        with Path(actual["path"]).open("rb") as stream:
            header = stream.read(24)
        require(
            len(header) == 24
            and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR"
            and struct.unpack(">II", header[16:24]) == (640, 960),
            "current canvas must retain the registered640x960 PNG header",
        )
        return actual

    def finish(self):
        require(
            all(sha(path) == expected for path, expected in self.inputs.items()), "input changed during preparation"
        )


def validate_config(config):
    require(config["schema_version"] == CONFIG_SCHEMA and config["seed"] == 42, "038 config/seed differs")
    require(
        config["bindings_complete"] is True and set(config["refs"]) == REF_NAMES, "qualification bindings are pending"
    )
    require(
        config["budget"] == BUDGET and config["resources"] == RESOURCES, "fixed038 budget/resource contract differs"
    )
    require(config["noise_roots"] == [101, 102] and config["evaluation_seeds"] == [42, 43], "noise contract differs")
    require(config["ridge"] == 0.1 and config["feature_terms"] == [], "no ridge/time-feature search is registered")
    require(config["validation_episodes"] == list(VAL_EPISODES), "six complete validation EP identities differ")
    require(config["gpu_uuid"] == GPU and config["automatic_retry"] is False, "GPU/retry contract differs")
    require(
        config["preparation_only"] is True and config["execution_authorized"] is False,
        "metadata preparation cannot authorize execution",
    )
    for ref in config["refs"].values():
        require(
            isinstance(ref, dict)
            and set(ref) == {"path", "sha256"}
            and isinstance(ref["sha256"], str)
            and len(ref["sha256"]) == 64,
            "qualification SHA binding is pending",
        )
    root = (REPO / config["output_dir"]).resolve()
    require(
        config["output_dir"] == "outputs/wam_stability_20261021/clean_residual_training_20261023"
        and root.is_relative_to((REPO / "outputs").resolve()),
        "preparation output escaped the registered new038 root",
    )
    return root


def _complete(value, message):
    require(value.get("passed") is True and value.get("complete") is True, message)


def _lifecycle(value, message):
    require(value.get("actual_wait") is True and value.get("returncode") == 0, message)
    require(value.get("reaped", True) is True and value.get("signal_sent", False) is False, message)


def _owned_command(evidence, command):
    exit_record, release = evidence.json(command["exit"]), evidence.json(command["release"])
    _lifecycle(exit_record, "actual owned wait0 is missing")
    require(
        release.get("reaped") is True and release.get("returncode") == 0 and release.get("signal_sent") is False,
        "actual owned normal release is missing",
    )
    launch = evidence.json(command["launch"])
    require(
        launch["child"]["pid"] == release["pid"] and launch["child"]["start_ticks"] == release["start_ticks"],
        "owned PID/birth binding differs",
    )


def prerequisites(config, evidence):
    refs = config["refs"]
    values = {key: evidence.json(ref) for key, ref in refs.items()}
    settled, audit, owner, peer = (
        values[key] for key in ("settled_prepared", "settled_audit", "settled_owner", "settled_peer")
    )
    for key in ("settled_collection", "settled_audit", "settled_owner", "settled_peer"):
        _complete(values[key], f"034 prerequisite incomplete: {key}")
    require(
        audit["prepared_ref"] == peer["prepared_ref"] == refs["settled_prepared"]
        and owner["verified_ref"] == peer["verified_ref"] == refs["settled_audit"]
        and peer["owner_ref"] == refs["settled_owner"],
        "034 independent lineage differs",
    )
    quality = peer["new_quality"]
    require(
        quality["total_cases"] == quality["stable_native_cases"] == 30
        and quality["stable_native_by_task"] == {"0": 10, "1": 10, "2": 10}
        and quality["eligible_for_model_training_registration"]
        is audit["eligible_for_model_training_registration"]
        is True,
        "034 full30 data qualification failed",
    )
    require(
        peer["native_prefix_exact_cases"] == 30
        and peer["sources_and_inputs_before_after_exact"] is True
        and peer["preserved_original034_owner_passed"] is False,
        "034 failure/prefix evidence was rewritten",
    )
    _lifecycle(peer["worker_lifecycle"], "034 worker lifecycle incomplete")
    _lifecycle(peer["audit_lifecycle"], "034 audit lifecycle incomplete")
    require(evidence.json(peer["original_failed_owner_ref"])["passed"] is False, "original034 False must be preserved")
    bridge, qualified, old = (values[key] for key in ("bridge_prepared", "bridge_qualification", "bridge_failed_owner"))
    _complete(qualified, "037 model producer is incomplete")
    require(
        qualified["bridge_qualified"] is True
        and qualified["prepared_ref"] == refs["bridge_prepared"]
        and qualified["actual_requests"] == 7
        and qualified["actual_denoise_calls"] == 245,
        "037 saved7/245 bridge evidence differs",
    )
    require(old["passed"] is False and len(old["owned_commands"]) >= 1, "original037 owner False must remain visible")
    _owned_command(evidence, old["owned_commands"][0])
    gpu = values["bridge_gpu_after"]
    require(
        old["gpu_after_ref"] == refs["bridge_gpu_after"]
        and gpu["released"] is True
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["uuid"] == GPU,
        "037 GPU normal release evidence is incomplete",
    )
    registration, review, corrected, final = (
        values[key] for key in ("native_mask_prepared", "native_mask_review", "native_mask_owner", "native_mask_peer")
    )
    for key in ("native_mask_review", "native_mask_owner", "native_mask_peer"):
        _complete(values[key], f"039 joint qualification incomplete: {key}")
    require(
        corrected["bridge_qualified"] is review["bridge_qualified"] is True
        and corrected["registration_ref"] == refs["native_mask_prepared"]
        and corrected["review_ref"] == refs["native_mask_review"]
        and corrected["original_failed_owner_ref"]
        == registration["original_failed_owner_ref"]
        == refs["bridge_failed_owner"]
        and corrected["original_prepared_ref"] == registration["original_prepared_ref"] == refs["bridge_prepared"]
        and corrected["qualification_ref"] == registration["qualification_ref"] == refs["bridge_qualification"]
        and corrected["preserved_original_owner_passed"] is False,
        "039 joint qualification lineage differs",
    )
    require(
        review["actual_online_points_verified"] == 245 and review["nonzero_changed_active_values"] > 0,
        "039 full math/nonzero denominator differs",
    )
    require(len(corrected["owned_commands"]) == 1, "039 fixed single CPU audit command differs")
    _owned_command(evidence, corrected["owned_commands"][0])
    require(
        final["owner039_ref"] == refs["native_mask_owner"]
        and final["formal_ref"] == refs["native_mask_review"]
        and final["worker_ref"] == refs["bridge_qualification"]
        and final["prepared_ref"] == refs["bridge_prepared"]
        and final["consumed_source_and_artifacts_before_after_exact"] is True,
        "039 final peer is not bound to this exact completed owner",
    )
    for key in ("native_mask_owner", "native_mask_review"):
        ref = refs[key]
        require(final["consumed_sha256"].get(ref["path"]) == ref["sha256"], "039 peer consumed-source binding missing")
    require(values["settled_collection"]["prepared_ref"] == refs["settled_prepared"], "034 collection lineage differs")
    return values


def _identity(record):
    return SourceIdentity(record["task_index"], record["demo"], record["initial_state_sha256"])


def _request(template, *, name, image, seed, task):
    require(
        template["prompt"] == task and set(template["extra"]) == {"action128"}, "qualified task-only template differs"
    )
    require(
        set(template)
        == {
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
        },
        "unregistered/future model input field",
    )
    require(
        set(template["extra"]["action128"])
        == {
            "action_dim_mask",
            "action_domain",
            "action_fps",
            "camera_layout",
            "return_video",
            "selected_roles",
            "task_title",
        },
        "privileged model action condition",
    )
    value = copy.deepcopy(template)
    value.update(name=name, vision_path=image["path"], seed=seed)
    return value


def training_metadata(config, evidence, lineage):
    prepared, collection, bridge = (
        lineage[key] for key in ("settled_prepared", "settled_collection", "bridge_prepared")
    )
    original = prepared["original_prepared"]
    fresh = evidence.json(original["fresh_fixtures_ref"])
    fresh_sources = tuple(_identity(record) for record in fresh["records"])
    forbidden = tuple(
        (item["task_index"], demo) for item in original["selection"]["tasks"] for demo in item["excluded_demo_ids"]
    )
    anchors = build_training_roster(
        prepared["records"],
        validation_sources=(),
        fresh_sources=fresh_sources,
        validation_demo_keys=VAL_DEMOS,
        forbidden_demo_keys=forbidden,
        forbidden_initial_state_sha256=original["forbidden_initial_state_sha256"],
    )
    require(len(collection["cases"]) == 30, "034 case metadata denominator differs")
    cases = {}
    for case_ref in collection["cases"]:
        case = evidence.json(case_ref["case"])
        _complete(case, "034 saved case is incomplete")
        require(
            case["native_quality_passed"] is True and case["prepared_ref"] == config["refs"]["settled_prepared"],
            "034 case qualification differs",
        )
        record = case["record"]
        identity = (record["task_index"], record["demo"])
        require(identity not in cases, "duplicate034 case identity")
        require(
            record in prepared["records"]
            and case["native_steps"] == record["native_steps"]
            and case["endpoint_count"] == record["native_steps"] + 81
            and case["future_rgb_frames"] == 0,
            "034 saved endpoint/record contract differs",
        )
        array_ref = evidence.check(case["arrays"], kind="opaque_future_label_sha_only")
        count = case["endpoint_count"]
        for key, shape, dtype in (
            ("state128", [count, 128], "float64"),
            ("action128", [count, 128], "float64"),
            ("state_mask", [count, 128], "bool"),
            ("action_mask", [count, 128], "bool"),
            ("action_valid", [count], "bool"),
        ):
            require(
                case["array_index"][key] == dict(shape=shape, dtype=dtype), "034 metadata-only label layout differs"
            )
        require(
            [a["frame_index"] for a in case["anchors"]] == record["anchor_frames"], "034 current anchor roster differs"
        )
        cases[identity] = (case, case_ref["case"], array_ref)
    templates = {entry["request"]["prompt"]: entry["request"] for entry in bridge["requests3"]}
    require(len(templates) == 3, "requires three qualified task request templates")
    rows, by_anchor = [], {}
    for anchor in anchors:
        case, case_ref, array_ref = cases[(anchor.source.task_index, anchor.source.demo)]
        current = next(item for item in case["anchors"] if item["frame_index"] == anchor.anchor_frame)
        require(
            current["future_rgb_frames"] == 0
            and current["canvas_shape"] == [960, 640, 3]
            and current["render_before"] == current["render_after"],
            "current canvas/physics binding differs",
        )
        require(
            current["current_endpoint"]["endpoint_step"] == anchor.anchor_frame
            and current["render_before"]["sim_state_sha256"] == current["current_endpoint"]["sim_state_sha256"],
            "current image is not bound to its measured endpoint",
        )
        image = evidence.png(current["canvas_png"])
        name = f"task{anchor.source.task_index}_{anchor.source.demo}_a{anchor.anchor_frame}"
        value = dict(
            name=name,
            source_identity=asdict(anchor.source),
            anchor_frame=anchor.anchor_frame,
            task=case["record"]["task"],
            native_steps=case["native_steps"],
            case_ref=case_ref,
            declared_trajectory_ref=case["arrays"],
            trajectory_ref=array_ref,
            image_ref=image,
            current_endpoint=current["current_endpoint"],
            tool_source_from_tcp=original["tool_source_from_tcp"],
            future_window_steps=40,
            future_labels_parsed=False,
        )
        rows.append(value)
        by_anchor[anchor] = value
    requests = []
    roster = noise_roster(anchors)
    for index, noise in enumerate(roster):
        anchor = by_anchor[noise.anchor]
        name = f"train_{anchor['name']}_r{noise.root_seed}"
        requests.append(
            dict(
                index=index,
                name=name,
                anchor_name=anchor["name"],
                **noise.as_dict(),
                request=_request(
                    templates[anchor["task"]],
                    name=name,
                    image=anchor["image_ref"],
                    seed=noise.sampling_seed,
                    task=anchor["task"],
                ),
            )
        )
    require(
        len(rows) == 150 and len(requests) == len({r["sampling_seed"] for r in requests}) == 300,
        "training roster denominator differs",
    )
    return (
        rows,
        requests,
        dict(
            fresh_fixture_ref=original["fresh_fixtures_ref"],
            fresh_sources=[asdict(value) for value in fresh_sources],
            validation_demo_keys=VAL_DEMOS,
            validation_initial_sync_certified=False,
            forbidden_demo_keys=forbidden,
            forbidden_initial_state_sha256=original["forbidden_initial_state_sha256"],
        ),
    )


def validation_metadata(lineage, evidence):
    bridge = lineage["bridge_prepared"]
    historical_ref = bridge["config"]["historical_prepared_ref"]
    historical = evidence.json(historical_ref)
    source_links = historical["conservative_source_exclusion"]["records"]
    templates = {entry["request"]["prompt"]: entry["request"] for entry in bridge["requests3"]}
    anchors = []
    for episode in VAL_EPISODES:
        selected = sorted(
            (a for a in historical["anchors"] if a["episode"] == episode), key=lambda a: a["anchor_index"]
        )
        require(
            len(selected) == 4 and [a["anchor_index"] for a in selected] == [0, 1, 2, 3],
            "validation four-anchor roster differs",
        )
        links = [item for item in source_links if item["r2_episode"] == episode]
        require(
            len(links) == 1 and (links[0]["task_index"], links[0]["demo"]) in VAL_DEMOS,
            "validation native-source identity differs",
        )
        for item in selected:
            require(
                item["split"] == "validation"
                and item["valid_future_steps"] == 40
                and item["task_index"] == links[0]["task_index"],
                "validation future-window metadata differs",
            )
            array_declared = dict(path=item["arrays_path"], sha256=item["arrays_sha256"])
            arrays = evidence.check(array_declared, kind="opaque_future_label_sha_only")
            image = evidence.png(dict(path=item["image_path"], sha256=item["image_sha256"]))
            anchors.append(
                dict(
                    name=item["name"],
                    episode=episode,
                    task_index=item["task_index"],
                    task=item["task"],
                    anchor_index=item["anchor_index"],
                    anchor_frame=item["anchor_frame"],
                    image_ref=image,
                    arrays_ref=arrays,
                    declared_arrays_ref=array_declared,
                    measured_state128=item["measured_state128"],
                    state_mask128=item["state_mask128"],
                    source_episode_id=item["source_episode_id"],
                    source_demo=links[0]["demo"],
                    initial_sync_certified=False,
                    future_labels_parsed=False,
                    request_template=_request(
                        templates[item["task"]], name=item["name"], image=image, seed=42, task=item["task"]
                    ),
                )
            )
    require(
        len(anchors) == 24 and len({a["name"] for a in anchors}) == 24, "complete six-EP validation identity differs"
    )
    generations = []
    for anchor in anchors:
        for seed in (42, 43):
            for variant in ("raw", "w2"):
                name = f"{anchor['name']}_{variant}_s{seed}"
                request = dict(anchor["request_template"], name=name, seed=seed)
                generations.append(
                    dict(
                        index=len(generations),
                        name=name,
                        anchor_name=anchor["name"],
                        variant=variant,
                        sampling_seed=seed,
                        request=request,
                    )
                )
    return anchors, generations, historical_ref


def prepare(config_path=DEFAULT_CONFIG):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "preparation requires S2A cwd and empty CUDA/LD",
    )
    config_path = Path(config_path).resolve()
    require(config_path.is_relative_to(REPO), "config must remain in S2A")
    config = json.loads(config_path.read_text())
    root = validate_config(config)
    require(not root.exists(), "refuses existing038 output root or retry")
    evidence = Evidence()
    config_ref = reference(config_path)
    evidence.check(config_ref)
    lineage = prerequisites(config, evidence)
    anchors, requests, exclusion = training_metadata(config, evidence, lineage)
    validation, generations, historical_ref = validation_metadata(lineage, evidence)
    bridge = lineage["bridge_prepared"]
    for path, expected in bridge["source_sha256"].items():
        evidence.check(dict(path=path, sha256=expected), kind="qualified_model_source")
    for path, stat in bridge["weight_files"].items():
        actual = Path(path).stat()
        require(
            actual.st_size == stat["bytes"] and actual.st_mtime_ns == stat["mtime_ns"], "frozen163 weight stat changed"
        )
    plan = (REPO / config["plan_path"]).resolve()
    require(plan.is_relative_to(REPO / "plans"), "038 plan must remain in S2A plans")
    sources = {}
    for relative in config["source_paths"] + [str(config_path.relative_to(REPO)), str(plan.relative_to(REPO))]:
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO) and path.is_file(), "declared preparation source missing or outside S2A")
        sources[str(path)] = sha(path)
    missing = [path for path in config["execution_entrypoints"].values() if not (REPO / path).is_file()]
    prepared = dict(
        schema_version=SCHEMA,
        root=str(root),
        config_ref=config_ref,
        config=config,
        plan_ref=reference(plan),
        prerequisite_refs=config["refs"],
        training_anchors=anchors,
        training_requests=requests,
        noise_roster=dict(training_seeds=[r["sampling_seed"] for r in requests], evaluation_seeds=[42, 43]),
        training_rounds=[
            dict(name="D0", policy="raw", request_indices=list(range(300)), rows=420000),
            dict(name="D1", policy="fixed_W1", request_indices=list(range(300)), rows=420000),
        ],
        row_order=ROW_ORDER,
        noise_seed_algorithm=SEED_ALGORITHM,
        union_rows=840000,
        training_weight_per_row=dict(W1="1/420000", W2="1/840000"),
        exclusion=exclusion,
        validation_anchors=validation,
        validation_generations=generations,
        historical_validation_ref=historical_ref,
        zero_identity_cases=bridge["requests3"],
        model_config=bridge["config"],
        qualified_model_source_sha256=bridge["source_sha256"],
        weight_files=bridge["weight_files"],
        full_weight_bytes_rehashed=False,
        new_model_initializations=0,
        future_label_archive_decodes=0,
        new_model_calls=0,
        new_fits=0,
        physics_steps=0,
        fresh_performance_reads=0,
        budget=BUDGET,
        resources=RESOURCES,
        source_sha256=sources,
        input_sha256=evidence.inputs,
        input_read_kinds=evidence.read_kinds,
        readiness=dict(
            metadata_contract_complete=True,
            ready_for_execution=False,
            missing_entrypoints=missing,
            reasons=[
                "metadata preparation only; independent runtime/fit/QR/state/development admission is not implemented"
            ],
            entrypoint_presence_is_not_qualification=True,
        ),
        execution_authorized=False,
        fresh_performance_authorized=False,
        original037_owner_passed=False,
    )
    evidence.finish()
    snapshots = root / "preparation/source"
    snapshots.mkdir(parents=True, exist_ok=False)
    saved = {}
    for path, expected in sources.items():
        target = snapshots / Path(path).relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha(target) == expected and sha(path) == expected, "source changed during snapshot")
        saved[str(target)] = expected
    prepared["snapshot_sha256"] = saved
    evidence.finish()
    destination = root / "preparation/prepared.json"
    with destination.open("x") as stream:
        json.dump(prepared, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return reference(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config)))


if __name__ == "__main__":
    main()
