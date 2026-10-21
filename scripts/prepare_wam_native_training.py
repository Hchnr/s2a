"""030 source/shape preparation. No simulator, WAM, HDF observations or success labels."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256
from s2a.geometry import validate_transform
from s2a.native_training_sources import (
    initial_state_hash,
    numeric_demo_order,
    select_training_sources,
    training_anchor_frames,
    validate_selected_shapes,
)
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_synchronous_native_training_20261021.json"
CONFIG_SHA = "966c33348ce0a78f09e310f9d9efcf6d22201502babc8834cc8a2c4aa756115a"
PLAN = REPO / "plans/v0.0.1/030_synchronous_native_training_20261021.md"
PLAN_SHA = "fb4363797be72702da54710c6a910338d8a8769bca12d0de133b7633032db155"
NEW_SOURCES = (
    "src/s2a/native_training_sources.py",
    "scripts/prepare_wam_native_training.py",
    "tests/test_native_training_sources.py",
    "src/s2a/native_training.py",
    "scripts/collect_wam_native_training.py",
    "tests/test_native_training.py",
    "scripts/run_wam_native_training.py",
    "tests/test_native_training_runner.py",
    "scripts/verify_wam_native_training.py",
    "tests/test_native_training_verification.py",
)
RUNTIME_SOURCE_PATHS = (
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/robots/single_arm.py",
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/binding_utils.py",
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/transform_utils.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def source_stat(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def checked_json(path, expected_sha, inputs):
    path = Path(path).resolve()
    require(sha256(path) == expected_sha, f"source JSON SHA changed: {path}")
    inputs[str(path)] = expected_sha
    return json.loads(path.read_text())


def assert_config(config):
    require(
        config["seed"] == 42
        and config["training_demos_per_task"] == 10
        and config["exclude_historical_demos"] is True
        and config["exclude_validation_demos_by_task"]
        == [["demo_16", "demo_32"], ["demo_4", "demo_27"], ["demo_14", "demo_24"]]
        and (config["minimum_native_steps"], config["maximum_native_steps"]) == (43, 600)
        and config["terminal_hold_steps"] == config["horizon"] == 40
        and config["native_uniform_anchors"] == 4
        and config["terminal_hold_anchor"] is True
        and config["control_fps"] == 20
        and config["render_resolution"] == 256
        and config["automatic_retry"] is False
        and config["model_training_authorized"] is False
        and config["fresh_performance_authorized"] is False,
        "030 registered source/clock/authorization contract changed",
    )
    require(
        config["budget"]
        == dict(
            environment_initializations=30,
            maximum_physics_steps=19200,
            current_canvas_images=150,
            future_rgb_frames=0,
            wam_initializations=0,
            denoise_calls=0,
            fits=0,
            fresh_cases=0,
        ),
        "030 fixed budget changed",
    )


def prior_admission(config, inputs):
    audit_ref = dict(
        path=str((REPO / config["prior_diagnostic_verification"]).resolve()),
        sha256=config["prior_diagnostic_verification_sha256"],
    )
    final_ref = dict(
        path=str((REPO / config["prior_diagnostic_final_peer"]).resolve()),
        sha256=config["prior_diagnostic_final_peer_sha256"],
    )
    audit = checked_json(**dict(path=audit_ref["path"], expected_sha=audit_ref["sha256"], inputs=inputs))
    final = checked_json(final_ref["path"], final_ref["sha256"], inputs)
    require(
        audit["passed"] is True
        and audit["complete_free_replays"] == 48
        and audit["actual_denoise_calls"] == 1824
        and audit["full128_raw_and_normalized_exact"] is True
        and audit["original028_state_gate"] is False
        and audit["eligible_for_fit"] is False
        and audit["physics_authorized"] is False
        and final["complete"] is True
        and final["passed"] is True
        and final["verification"] == audit_ref
        and final["prepared"] == audit["prepared"]
        and final["source_before_after_unchanged"] is True
        and final["frozen_sources_verified"] == final["source_snapshots_verified"] == 111
        and final["task_success_proven"] is False,
        "030 requires the complete unchanged029 diagnostic, retaining original False",
    )
    prior = checked_json(audit["prepared"]["path"], audit["prepared"]["sha256"], inputs)
    require(len(prior["source_sha256"]) == len(prior["snapshot_sha256"]) == 111, "029 source111 closure differs")
    # Do not expand prior input_sha256: it includes inherited observation refs.
    for path, digest in prior["source_sha256"].items():
        require(sha256(path) == digest, f"original111 source changed: {path}")
    for path, digest in prior["snapshot_sha256"].items():
        require(sha256(path) == digest, f"original111 source snapshot changed: {path}")
    return dict(
        prior_diagnostic_verification_ref=audit_ref,
        prior_diagnostic_final_peer_ref=final_ref,
        prior_prepared_ref=audit["prepared"],
    ), prior


def source_closure(prior):
    listed = subprocess.check_output(["git", "ls-files", "src", "scripts", "tests"], cwd=REPO, text=True).splitlines()
    paths = {str((REPO / p).resolve()) for p in listed if p.endswith(".py")}
    paths.update(str((REPO / p).resolve()) for p in NEW_SOURCES)
    paths.update((str(CONFIG), str(PLAN)))
    sources = dict(prior["source_sha256"])
    for path in sorted(paths):
        require(Path(path).is_file(), f"030 source closure incomplete: {path}")
        current = sha256(path)
        require(path not in sources or sources[path] == current, f"original111 source changed: {path}")
        sources[path] = current
    return dict(sorted(sources.items()))


def metadata_sources(path, excluded_ids, *, opener=None):
    """Never access excluded demo datasets; inspect candidate initial/XML/shape only."""
    if opener is None:
        import h5py

        opener = h5py.File
    rows = []
    with opener(path, "r") as handle:
        group = handle["data"]
        for name in numeric_demo_order(group.keys()):
            if name in excluded_ids:
                continue
            demo = group[name]
            initial = np.asarray(demo["states"][0], dtype=np.float64)
            xml = demo.attrs["model_file"]
            xml = xml.decode() if isinstance(xml, bytes) else xml
            require(isinstance(xml, str) and bool(xml.strip()), "candidate model XML is missing")
            rows.append(
                dict(
                    demo=name,
                    initial_state=initial.tolist(),
                    initial_state_sha256=initial_state_hash(initial),
                    model_xml=xml,
                    model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                    actions_shape=[int(v) for v in demo["actions"].shape],
                )
            )
    return rows


def reservation_admission(config, inputs):
    fresh_ref = dict(path=str((REPO / config["fresh_config"]).resolve()), sha256=config["fresh_config_sha256"])
    fixture_ref = dict(path=str((REPO / config["fresh_fixtures"]).resolve()), sha256=config["fresh_fixtures_sha256"])
    fresh = checked_json(fresh_ref["path"], fresh_ref["sha256"], inputs)
    fixture = checked_json(fixture_ref["path"], fixture_ref["sha256"], inputs)
    require(
        fixture["config"] == fresh and len(fixture["records"]) == 12 and config["tasks"] == fresh["tasks"],
        "original fresh reservation differs",
    )
    forbidden, task_sources, excluded, metadata = set(), [], [], []
    for task_index in range(3):
        records = [record for record in fixture["records"] if record["task_index"] == task_index]
        require(
            len(records) == 4 and len({record["demo"] for record in records}) == 4,
            "fresh per-task four identities differ",
        )
        source = str(Path(fresh["simulation_sources"][task_index]).resolve())
        for record in records:
            digest = initial_state_hash(record["initial_state"])
            require(
                digest == record["initial_state_sha256"] and digest not in forbidden, "fresh initial identity differs"
            )
            require(
                str(Path(record["source_hdf5"]).resolve()) == source and record["task"] == config["tasks"][task_index],
                "fresh task/source identity differs",
            )
            forbidden.add(digest)
        require(
            len({record["source_hdf5_sha256"] for record in records})
            == len({record["task_id"] for record in records})
            == 1,
            "fresh task source metadata inconsistent",
        )
        task_sources.append(dict(path=source, sha256=records[0]["source_hdf5_sha256"], task_id=records[0]["task_id"]))
        excluded.append(
            set(fresh["excluded_demos_by_task"][task_index])
            | set(config["exclude_validation_demos_by_task"][task_index])
            | {r["demo"] for r in records}
        )
    frozen_exclusions = {str(Path(path).resolve()): digest for path, digest in fixture["exclusions"].items()}
    for path in fresh["previous_simulation_fixtures"]:
        path = str(Path(path).resolve())
        require(path in frozen_exclusions, "historical fixture omitted from frozen reservation exclusions")
        ref = dict(path=path, sha256=frozen_exclusions[path])
        historical = checked_json(ref["path"], ref["sha256"], inputs)
        for record in historical.get("records", historical.get("episodes", [])):
            digest = initial_state_hash(record["initial_state"])
            require(digest == record["initial_state_sha256"], "historical initial identity differs")
            forbidden.add(digest)
        metadata.append(ref)
    return (
        dict(fresh_config_ref=fresh_ref, fresh_fixtures_ref=fixture_ref, historical_fixture_refs=metadata),
        task_sources,
        excluded,
        forbidden,
    )


def export_actions(records, inputs, *, opener=None):
    if opener is None:
        import h5py

        opener = h5py.File
    for record in records:
        with opener(record["source_hdf5"], "r") as handle:
            actions = np.asarray(handle["data"][record["demo"]]["actions"][:])
        require(
            actions.shape == (record["native_steps"], 7)
            and np.issubdtype(actions.dtype, np.floating)
            and np.isfinite(actions).all(),
            "fixed selected native actions invalid; no replacement authorized",
        )
        path = Path(record["expert_path"])
        path.parent.mkdir(parents=True, exist_ok=False)
        with path.open("xb") as stream:
            np.save(stream, actions, allow_pickle=False)
        record["expert_sha256"] = sha256(path)
        record["expert_dtype"] = str(actions.dtype)
        inputs[str(path)] = record["expert_sha256"]


def prepare(config_path=CONFIG, *, seed=42):
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "030 preparation requires S2A cwd and empty CUDA/LD",
    )
    require(Path(config_path).resolve() == CONFIG and seed == 42, "030 requires the registered config and seed42")
    inputs = {}
    config = checked_json(CONFIG, CONFIG_SHA, inputs)
    require(sha256(PLAN) == PLAN_SHA, "030 registered plan changed")
    assert_config(config)
    root = (REPO / config["output_dir"]).resolve()
    require(root.is_relative_to(REPO / "outputs") and not root.exists(), "030 refuses existing or non-output root")
    prior_refs, prior = prior_admission(config, inputs)
    sources = source_closure(prior)
    reserve_refs, task_sources, excluded, forbidden = reservation_admission(config, inputs)
    policy_ref = reference(REPO / config["policy_template"])
    policy = checked_json(policy_ref["path"], policy_ref["sha256"], inputs)
    tool_ref = reference(policy["tool_contract"])
    tool = checked_json(tool_ref["path"], tool_ref["sha256"], inputs)
    h = validate_transform(np.asarray(tool["source_from_tcp"], dtype=np.float64))
    require(h.shape == (4, 4), "030 requires one canonical TCP transform")
    runtime_refs = [reference(REPO / path) for path in RUNTIME_SOURCE_PATHS]
    for ref in runtime_refs:
        inputs[ref["path"]] = ref["sha256"]
    stats, candidates = {}, []
    for task_index, source in enumerate(task_sources):
        stats[source["path"]] = source_stat(source["path"])
        require(sha256(source["path"]) == source["sha256"], "native source content differs from frozen reservation")
        inputs[source["path"]] = source["sha256"]
        candidates.append(metadata_sources(source["path"], excluded[task_index]))
    selected, selection = select_training_sources(candidates, excluded, forbidden)
    root.mkdir(parents=True, exist_ok=False)
    (root / "preparation").mkdir()
    try:
        validate_selected_shapes(selected)  # All30 fixed before any selected action contents are opened.
    except ValueError as error:
        write_once(
            root / "preparation" / "selection_rejected.json",
            dict(
                schema_version="s2a.native-training-selection-rejection.v1",
                metadata_admission=False,
                config_ref=reference(CONFIG),
                registration_ref=reference(PLAN),
                selection=selection,
                selected_actions_shapes=[[item["actions_shape"] for item in task] for task in selected],
                source_sha256=sources,
                input_sha256=inputs,
                source_stats=stats,
                error=str(error),
                selected_action_contents_read=0,
                automatic_retry=False,
                model_training_authorized=False,
            ),
        )
        raise
    records = []
    for task_index, task in enumerate(selected):
        source = task_sources[task_index]
        selection["tasks"][task_index]["source_hdf5"] = source["path"]
        for item in task:
            name = f"task{task_index}_native_{item['demo']}"
            records.append(
                dict(
                    **item,
                    name=name,
                    task_index=task_index,
                    task=config["tasks"][task_index],
                    task_id=source["task_id"],
                    source_hdf5=source["path"],
                    source_hdf5_sha256=source["sha256"],
                    source_stat=stats[source["path"]],
                    native_steps=item["actions_shape"][0],
                    anchor_frames=training_anchor_frames(item["actions_shape"][0]),
                    expert_path=str(root / "sources" / name / "expert.npy"),
                )
            )
    export_actions(records, inputs)
    snapshots = {}
    directory = root / "preparation" / "source_snapshot"
    directory.mkdir()
    for index, (path, digest) in enumerate(sorted(sources.items())):
        snapshot = directory / f"{index:04d}_{Path(path).name}"
        shutil.copyfile(path, snapshot)
        require(sha256(snapshot) == digest, "030 source changed while snapshotting")
        snapshots[str(snapshot)] = digest
    for path, digest in inputs.items():
        require(sha256(path) == digest, f"030 input changed during preparation: {path}")
    require(
        all(source_stat(path) == value for path, value in stats.items()),
        "native source stat changed during preparation",
    )
    prepared = dict(
        schema_version="s2a.native-training-preparation.v1",
        seed=42,
        config=config,
        config_ref=reference(CONFIG),
        registration_ref=reference(PLAN),
        output_root=str(root),
        **prior_refs,
        **reserve_refs,
        policy_template_ref=policy_ref,
        policy_config=policy,
        tool_contract_ref=tool_ref,
        tool_source_from_tcp=h.tolist(),
        runtime_source_refs=runtime_refs,
        source_sha256=sources,
        snapshot_sha256=snapshots,
        inherited_source_sha256=prior["source_sha256"],
        source_stats=stats,
        input_sha256=inputs,
        selection=selection,
        forbidden_initial_state_sha256=sorted(forbidden),
        records=records,
        budget=config["budget"],
        metadata_admission=True,
        native_success_verified=False,
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        actual_new_physics_steps=0,
        actual_wam_calls=0,
    )
    path = root / "preparation" / "prepared.json"
    write_once(path, prepared)
    return path


def read_prepared(path, expected_sha):
    """Owner/worker source guard; no HDF API or inherited observation expansion."""
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "030 source guard requires S2A cwd and empty CUDA/LD",
    )
    inputs = {}
    path = Path(path).resolve()
    prepared = checked_json(path, expected_sha, inputs)
    require(
        prepared["schema_version"] == "s2a.native-training-preparation.v1" and prepared["seed"] == 42,
        "030 prepared schema/seed differs",
    )
    config = checked_json(CONFIG, CONFIG_SHA, inputs)
    assert_config(config)
    root = (REPO / config["output_dir"]).resolve()
    require(
        prepared["config"] == config
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["output_root"] == str(root)
        and path == root / "preparation" / "prepared.json"
        and root.is_relative_to(REPO / "outputs"),
        "030 prepared config/output mapping differs",
    )
    require(
        prepared["registration_ref"] == dict(path=str(PLAN), sha256=PLAN_SHA) and sha256(PLAN) == PLAN_SHA,
        "030 plan identity differs",
    )
    prior_refs, prior = prior_admission(config, inputs)
    reserve_refs, task_sources, excluded, forbidden = reservation_admission(config, inputs)
    require(
        all(prepared[key] == value for key, value in {**prior_refs, **reserve_refs}.items()),
        "030 prior/fresh source refs differ",
    )
    require(
        prepared["inherited_source_sha256"] == prior["source_sha256"]
        and prepared["source_sha256"] == source_closure(prior),
        "030 current full source closure differs",
    )
    expected_snapshots = {
        str(root / "preparation" / "source_snapshot" / f"{index:04d}_{Path(source).name}"): digest
        for index, (source, digest) in enumerate(sorted(prepared["source_sha256"].items()))
    }
    require(prepared["snapshot_sha256"] == expected_snapshots, "030 sorted source snapshot binding differs")
    policy_ref = reference(REPO / config["policy_template"])
    policy = checked_json(policy_ref["path"], policy_ref["sha256"], inputs)
    tool_ref = reference(policy["tool_contract"])
    tool = checked_json(tool_ref["path"], tool_ref["sha256"], inputs)
    runtime_refs = [reference(REPO / source) for source in RUNTIME_SOURCE_PATHS]
    require(prepared["runtime_source_refs"] == runtime_refs, "030 actual simulator transform source changed")
    for ref in runtime_refs:
        inputs[ref["path"]] = ref["sha256"]
    require(
        prepared["policy_template_ref"] == policy_ref
        and prepared["policy_config"] == policy
        and prepared["tool_contract_ref"] == tool_ref,
        "030 policy/tool source identity differs",
    )
    require(prepared["forbidden_initial_state_sha256"] == sorted(forbidden), "030 initial blacklist differs")
    expected_inputs = dict(inputs)
    expected_inputs.pop(str(path))  # The prepared file cannot include its own SHA.
    for source in task_sources:
        expected_inputs[source["path"]] = source["sha256"]
    for record in prepared["records"]:
        expert = (root / "sources" / record["name"] / "expert.npy").resolve()
        require(
            str(expert) == record["expert_path"] and expert.is_relative_to(root / "sources"),
            "030 expert path escaped source root",
        )
        expected_inputs[str(expert)] = record["expert_sha256"]
    require(prepared["input_sha256"] == expected_inputs, "030 actual-input roster changed or expanded")
    for field in ("source_sha256", "snapshot_sha256", "input_sha256"):
        for source, digest in prepared[field].items():
            require(sha256(source) == digest, f"030 changed {field}: {source}")
    require(
        set(prepared["source_stats"]) == {source["path"] for source in task_sources}
        and all(source_stat(source) == value for source, value in prepared["source_stats"].items()),
        "030 source stats differ",
    )
    require(
        len(prepared["records"]) == 30
        and prepared["budget"] == config["budget"]
        and prepared["metadata_admission"] is True
        and prepared["native_success_verified"] is False
        and prepared["eligible_for_model_training_registration"] is False
        and prepared["model_training_authorized"] is False
        and prepared["fresh_performance_authorized"] is False,
        "030 preparation does not authorize training/physics success",
    )
    require(
        [record["task_index"] for record in prepared["records"]] == [task for task in range(3) for _ in range(10)],
        "030 task/record order differs",
    )
    by_task = [[record for record in prepared["records"] if record["task_index"] == task] for task in range(3)]
    require(all(len(task) == 10 for task in by_task), "030 requires all30 fixed sources")
    validate_selected_shapes(by_task)
    selected, selection = select_training_sources(
        [task["candidates"] for task in prepared["selection"]["tasks"]], excluded, forbidden
    )
    for task_index, source in enumerate(task_sources):
        selection["tasks"][task_index]["source_hdf5"] = source["path"]
        for expected, actual in zip(selected[task_index], by_task[task_index], strict=True):
            require(
                all(actual[key] == value for key, value in expected.items()), "030 fixed selection metadata differs"
            )
    require(selection == prepared["selection"], "030 single-RNG source selection changed")
    seen = set(forbidden)
    for task_index, task in enumerate(by_task):
        for record in task:
            source = task_sources[task_index]
            require(
                record["name"] == f"task{task_index}_native_{record['demo']}"
                and record["task"] == config["tasks"][task_index]
                and record["task_id"] == source["task_id"]
                and record["source_hdf5"] == source["path"]
                and record["source_hdf5_sha256"] == source["sha256"]
                and record["source_stat"] == prepared["source_stats"][source["path"]]
                and hashlib.sha256(record["model_xml"].encode()).hexdigest() == record["model_xml_sha256"],
                "030 record task/source/XML binding differs",
            )
            require(
                record["demo"] not in excluded[task_index]
                and record["initial_state_sha256"] == initial_state_hash(record["initial_state"])
                and record["initial_state_sha256"] not in seen,
                "030 reserved/historical/selected initial alias",
            )
            seen.add(record["initial_state_sha256"])
            require(
                record["anchor_frames"] == training_anchor_frames(record["native_steps"])
                and record["actions_shape"] == [record["native_steps"], 7],
                "030 selected anchor/shape differs",
            )
            expert = Path(record["expert_path"]).resolve()
            require(
                expert == root / "sources" / record["name"] / "expert.npy"
                and prepared["input_sha256"].get(str(expert)) == record["expert_sha256"],
                "030 expert source binding differs",
            )
    require(
        np.array_equal(
            validate_transform(np.asarray(prepared["tool_source_from_tcp"])), np.asarray(tool["source_from_tcp"])
        ),
        "030 TCP matrix changed",
    )
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    path = prepare(args.config, seed=args.seed)
    print(
        json.dumps(
            dict(prepared=str(path), sha256=sha256(path), metadata_admission=True, native_success_verified=False)
        )
    )


if __name__ == "__main__":
    main()
