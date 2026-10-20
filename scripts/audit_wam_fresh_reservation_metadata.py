"""Independently authenticate a12-initial-state reservation without running physics."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/wam_stability_fresh_reservation_20261021.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype=float).tobytes()).hexdigest()


def stat_tuple(path):
    value = Path(path).stat()
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("metadata audit requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("metadata audit output must be a new repository artifact")
    inputs = {}

    def read(path):
        filename = str(Path(path).resolve())
        digest = sha(filename)
        if filename in inputs:
            assert inputs[filename] == digest, filename
        inputs[filename] = digest
        return json.loads(Path(filename).read_text())

    config = read(CONFIG)
    frozen = subprocess.check_output(["git", "show", f"ac2663f:{CONFIG}"])
    assert sha(CONFIG) == hashlib.sha256(frozen).hexdigest()
    assert config["seed"] == 42 and config["online_initial_states_per_task"] == 4
    assert config["candidate_status"].startswith("Not selected")
    assert not config["full_native_rotation_identity_complete"]
    assert not config["historical_initial_physical_synchronization_certified"]
    fixture_path = Path(config["output_dir"]) / "simulation/fixtures.json"
    fixtures = read(fixture_path)
    assert fixtures["config"] == config and len(fixtures["records"]) == 12
    assert fixtures["code"]["files_sha256"][CONFIG] == sha(CONFIG)
    assert set(fixtures["exclusions"]) == set(config["exclusion_sources"])
    for path, expected in fixtures["exclusions"].items():
        assert sha(path) == expected, path
        inputs[str(Path(path).resolve())] = expected
    for path, expected in config["offline_source_evidence_sha256"].items():
        assert sha(path) == expected
        inputs[str(Path(path).resolve())] = expected
    assert len(config["previous_simulation_fixtures"]) == 5
    previous, task_ids, previous_summaries = set(), {}, []
    for path in config["previous_simulation_fixtures"]:
        old = read(path)
        values = []
        for record in old["records"]:
            digest = array_sha(record["initial_state"])
            assert digest == record["initial_state_sha256"]
            previous.add(digest)
            values.append(digest)
            language, task_id = record["task"], record["task_id"]
            assert language not in task_ids or task_ids[language] == task_id
            task_ids[language] = task_id
        previous_summaries.append(
            dict(path=path, sha256=sha(path), records=len(values), unique_initials=len(set(values)))
        )
    selected_hashes = [r["initial_state_sha256"] for r in fixtures["records"]]
    assert len(set(selected_hashes)) == 12 and not (set(selected_hashes) & previous)
    by_task = {i: [r for r in fixtures["records"] if r["task_index"] == i] for i in range(3)}
    assert len(config["tasks"]) == 3 and all(len(records) == 4 for records in by_task.values())
    for item in config["offline_source_exclusion_records"]:
        assert not item["initial_sync_certified"]
        assert item["demo"] in config["excluded_demos_by_task"][item["task_index"]]
    assert len(config["offline_source_exclusion_records"]) == 12
    generator = np.random.default_rng(42)
    seen = previous.copy()
    audited, selections, source_signatures = [], [], []
    for task_index, source in enumerate(config["simulation_sources"]):
        before = stat_tuple(source)
        source_sha = sha(source)
        inputs[str(Path(source).resolve())] = source_sha
        with h5py.File(source, "r") as handle:
            excluded = set(config["excluded_demos_by_task"][task_index])
            if task_index == 0:
                excluded.update(("demo_27", "demo_7", "demo_9", "demo_15", "demo_0"))
            all_demos = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
            assert len(all_demos) == 50
            allowed = [name for name in all_demos if name not in excluded]
            allowed = [name for name in allowed if array_sha(handle["data"][name]["states"][0]) not in seen]
            selection = list(map(str, generator.choice(allowed, size=4, replace=False)))
            records = sorted(by_task[task_index], key=lambda item: item["initial_index"])
            assert selection == [r["demo"] for r in records], "seed42 metadata selection did not reproduce"
            selections.append(
                dict(
                    task_index=task_index,
                    eligible_demos=allowed,
                    reproduced_seed42_demos=selection,
                    excluded_demos=sorted(excluded),
                )
            )
            for index, record in enumerate(records):
                demo = handle["data"][record["demo"]]
                assert record["name"] == f"task{task_index}_init{index}"
                assert record["task"] == config["tasks"][task_index]
                assert record["task_id"] == task_ids[record["task"]]
                assert record["episode"] == record["demo"] and record["demo"] not in excluded
                assert Path(record["source_hdf5"]).resolve() == Path(source).resolve()
                assert record["source_hdf5_sha256"] == source_sha
                initial = np.asarray(demo["states"][0], dtype=float)
                np.testing.assert_array_equal(initial, record["initial_state"])
                digest = array_sha(initial)
                assert digest == record["initial_state_sha256"] and digest not in seen
                seen.add(digest)
                xml = demo.attrs["model_file"]
                assert record["model_xml"] == xml
                xml_sha = hashlib.sha256(xml.encode()).hexdigest()
                expert = Path(record["expert_path"])
                assert expert.resolve().is_relative_to((Path(config["output_dir"]) / "simulation").resolve())
                assert sha(expert) == record["expert_sha256"]
                inputs[str(expert.resolve())] = record["expert_sha256"]
                np.testing.assert_array_equal(np.load(expert), demo["actions"][:])
                audited.append(
                    dict(
                        name=record["name"],
                        task_index=task_index,
                        task_id=record["task_id"],
                        demo=record["demo"],
                        source_hdf5=str(Path(source).resolve()),
                        source_hdf5_sha256=source_sha,
                        initial_state_sha256=digest,
                        source_and_fixture_model_xml_sha256=xml_sha,
                        expert_path=str(expert),
                        expert_sha256=record["expert_sha256"],
                        expert_actions_equal_hdf=True,
                    )
                )
        assert stat_tuple(source) == before
        source_signatures.append(
            dict(path=str(Path(source).resolve()), sha256=source_sha, bytes_mtime_ctime=list(before))
        )
    # Inputs are immutable snapshots; sources were fully hashed once and stat
    # checked after the read, rather than hashing gigabytes a second time.
    source_paths = {item["path"] for item in source_signatures}
    for path, digest in inputs.items():
        if path not in source_paths:
            assert sha(path) == digest, path
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    value = dict(
        schema_version="s2a.fresh-reservation-metadata-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        config_path=CONFIG,
        config_sha256=sha(CONFIG),
        config_frozen_commit="ac2663f",
        fixture_path=str(fixture_path),
        fixture_sha256=sha(fixture_path),
        reserved_initial_states=12,
        unique_initial_state_hashes=12,
        per_task_counts=[4, 4, 4],
        previous_real_fixture_files=previous_summaries,
        previous_unique_initial_hashes=len(previous),
        previous_initial_hash_overlap=0,
        exclusion_demo_overlap=0,
        seed42_selection_reproduced=True,
        selections=selections,
        records=audited,
        source_signatures=source_signatures,
        input_sha256=inputs,
        new_model_calls=0,
        new_physics_steps=0,
        candidate_performance_read=False,
        fixture_selection_reexecuted=False,
        historical_r1_rotation_identity_certified=False,
        historical_image_pose_initial_physical_synchronization_certified=False,
        pretraining_unseen_certified=False,
        scope="Metadata-only independent audit of excluded prior physical initial-state SHA and conservative "
        "source-demo exclusions (including two suspected partial-content sources). Authentication of the new fixture "
        "against HDF first states/XML/actions does not certify historical R1 rotation, image/pose synchronization, "
        "pretraining unseen data or task success. No reserved trial was run.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), sha256=sha(output), complete=True, reservations=12, prior_overlap=0)))


if __name__ == "__main__":
    main()
