"""Content-match historical offline episodes before reserving fresh initial states.

Export uses the S2A Arrow environment; match uses the existing h5py simulation
venv. Neither stage imports a simulator or reads candidate evaluation outcomes.
Only all-command identities and the matched historical demonstrations' initial
states are read. Optional draft updates require all12 unique content matches.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(path):
    p = Path(path).resolve()
    before = p.stat()
    sha = digest(p)
    after = p.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    return dict(path=str(p), sha256=sha, bytes=after.st_size, mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(payload)


def unchanged(item, *, full_hash):
    p = Path(item["path"])
    stat = p.stat()
    assert (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (item["bytes"], item["mtime_ns"], item["ctime_ns"]), (
        str(p)
    )
    if full_hash:
        assert digest(p) == item["sha256"], str(p)


def export(args):
    import pyarrow.parquet as pq

    output = Path(args.output).resolve()
    assert output.is_relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    provenance, selection, draft = read(args.provenance), read(args.selection), read(args.draft)
    r1 = Path(provenance["r1_root"]).resolve()
    inputs = [
        signature(p)
        for p in (args.provenance, args.selection, args.draft, __file__, "scripts/audit_wam_independent_sources.py")
    ]
    meta_path = r1 / "meta/episodes/chunk-000/file-000.parquet"
    inputs.append(signature(meta_path))
    metadata = {
        r["episode_index"]: r
        for r in pq.read_table(
            meta_path, columns=["episode_index", "data/chunk_index", "data/file_index", "tasks"]
        ).to_pylist()
    }
    episodes = []
    for item in selection["selection"]:
        identity = json.loads(item["metadata"]["source_episode_id"])["source_id"]
        source_root, episode = identity.rsplit(":", 1)
        assert Path(source_root).resolve() == r1
        episode = int(episode)
        m = metadata[episode]
        assert item["task"] in m["tasks"] and draft["tasks"][item["task_index"]] == item["task"]
        episodes.append(
            dict(
                r2_episode=item["metadata"]["episode_index"],
                r1_episode=episode,
                source_episode_id=identity,
                task=item["task"],
                task_index=item["task_index"],
                split=item["split"],
                metadata=m,
            )
        )
    assert len(episodes) == len({r["r1_episode"] for r in episodes}) == 12
    arrays = {}
    partitions = sorted({(r["metadata"]["data/chunk_index"], r["metadata"]["data/file_index"]) for r in episodes})
    for chunk, file in partitions:
        path = r1 / f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"
        ids = [
            r["r1_episode"]
            for r in episodes
            if (r["metadata"]["data/chunk_index"], r["metadata"]["data/file_index"]) == (chunk, file)
        ]
        inputs.append(signature(path))
        rows = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "action.native_control"],
            filters=[("episode_index", "in", ids)],
        ).to_pylist()
        for episode in ids:
            selected = sorted((r for r in rows if r["episode_index"] == episode), key=lambda r: r["frame_index"])
            assert [r["frame_index"] for r in selected] == list(range(len(selected)))
            values = np.asarray([r["action.native_control"] for r in selected], dtype=float)
            assert values.ndim == 2 and values.shape[1] == 10 and np.isfinite(values).all()
            arrays[f"r1_{episode}"] = values
    archive = output / "historical_native_controls.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    for item in inputs:
        unchanged(item, full_hash=False)
    write_new(
        output / "export.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            draft_path=str(Path(args.draft).resolve()),
            inputs=inputs,
            episodes=episodes,
            archive=signature(archive),
            projection_tolerance=provenance["max_native_projection_error"],
            projection_source="scripts/audit_wam_independent_sources.py:76",
            new_model_calls=0,
            new_physics_steps=0,
        ),
    )
    print(output / "export.json")


def match(args):
    import h5py

    output = Path(args.output).resolve()
    exported = read(output / "export.json")
    for item in exported["inputs"]:
        unchanged(item, full_hash=False)
    unchanged(exported["archive"], full_hash=True)
    draft = read(args.draft)
    assert Path(args.draft).resolve() == Path(exported["draft_path"])
    with np.load(exported["archive"]["path"]) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    matches = {r["r1_episode"]: [] for r in exported["episodes"]}
    source_inputs = [signature(p) for p in draft["simulation_sources"]]
    catalogs = []
    for task_index, path in enumerate(draft["simulation_sources"]):
        with h5py.File(path, "r") as source:
            demos = sorted(source["data"], key=lambda name: int(name.split("_")[-1]))
            assert len(demos) == 50
            catalogs.append(demos)
            for demo in demos:
                actions = np.asarray(source["data"][demo]["actions"][:])
                assert actions.ndim == 2 and actions.shape[1] == 7
                rot = Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
                projected = np.column_stack((actions[:, :3], rot[:, :, 0], rot[:, :, 1], (1 - actions[:, 6]) / 2))
                for episode in exported["episodes"]:
                    expected = arrays[f"r1_{episode['r1_episode']}"]
                    if expected.shape != projected.shape:
                        continue
                    error = float(np.max(np.abs(expected - projected)))
                    if error < exported["projection_tolerance"]:
                        initial = np.asarray(source["data"][demo]["states"][0], dtype=float)
                        matches[episode["r1_episode"]].append(
                            dict(
                                task_index=task_index,
                                hdf5=str(Path(path).resolve()),
                                demo=demo,
                                command_rows=len(actions),
                                full_command_projection_max_abs=error,
                                initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                                model_xml_sha256=hashlib.sha256(
                                    source["data"][demo].attrs["model_file"].encode()
                                ).hexdigest(),
                            )
                        )
    records = [dict(episode, matches=matches[episode["r1_episode"]]) for episode in exported["episodes"]]
    unique = all(len(r["matches"]) == 1 and r["matches"][0]["task_index"] == r["task_index"] for r in records)
    union = [set(names) for names in draft["excluded_demos_by_task"]]
    if unique:
        for record in records:
            union[record["task_index"]].add(record["matches"][0]["demo"])
    union = [sorted(names, key=lambda name: int(name.split("_")[-1])) for names in union]
    for item in source_inputs + exported["inputs"]:
        unchanged(item, full_hash=False)
    report = output / "matches.json"
    write_new(
        report,
        dict(
            schema_version="s2a.offline-demo-exclusion.v1",
            created_utc=datetime.now(timezone.utc).isoformat(),
            unique12_content_matches=unique,
            searched_sources=source_inputs,
            searched_demos_per_source=[50, 50, 50],
            exported_controls=exported["archive"],
            export_json_sha256=digest(output / "export.json"),
            projection_tolerance=exported["projection_tolerance"],
            strict_less_than_tolerance=True,
            projection_source="scripts/audit_wam_independent_sources.py:76",
            records=records,
            original_excluded_demos=draft["excluded_demos_by_task"],
            proposed_excluded_demos=union,
            available_demo_names_before_initial_hash_filter=[
                sorted(set(all_demos) - set(excluded), key=lambda name: int(name.split("_")[-1]))
                for all_demos, excluded in zip(catalogs, union)
            ],
            input_files_sha256={item["path"]: item["sha256"] for item in exported["inputs"] + source_inputs},
            source_stat_unchanged_after_read=True,
            large_file_hash_policy="One full SHA per source, stat(size,mtime,ctime) unchanged after read; no repeated bulk hashing.",
            new_model_calls=0,
            new_physics_steps=0,
            candidate_evaluation_read=False,
            scope="Historical offline12 exclusion only; unique whole-command content match, no guessed metadata demo number. No fresh fixture selection/physics/model or pretraining-unseen claim.",
        ),
    )
    if args.update_draft:
        if not unique:
            raise ValueError("missing/ambiguous content matches: draft remains unchanged")
        original = next(item for item in exported["inputs"] if item["path"] == str(Path(args.draft).resolve()))
        unchanged(original, full_hash=True)
        draft["excluded_demos_by_task"] = union
        additions = [
            args.selection,
            args.provenance,
            str(Path(__file__).relative_to(ROOT)),
            str(report.relative_to(ROOT)),
        ]
        draft["exclusion_sources"] = list(dict.fromkeys(draft["exclusion_sources"] + additions))
        Path(args.draft).write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n")
        write_new(
            output / "draft_update.json",
            dict(
                draft_path=str(Path(args.draft).resolve()),
                before_sha256=original["sha256"],
                after_sha256=digest(args.draft),
                match_report_sha256=digest(report),
                changed_fields=["excluded_demos_by_task", "exclusion_sources"],
                reservation_prepare_not_executed=True,
            ),
        )
    print(
        json.dumps(
            dict(
                report=str(report),
                sha256=digest(report),
                unique12_content_matches=unique,
                exclusions=[len(names) for names in union],
                draft_updated=args.update_draft and unique,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("export", "match"), required=True)
    parser.add_argument("--draft", default="configs/wam_stability_fresh_reservation_20261021.json")
    parser.add_argument("--provenance", default="configs/libero_provenance_20261019.json")
    parser.add_argument("--selection", default="outputs/wam_checkpoint_acceptance_20261019v2/samples/selection.json")
    parser.add_argument("--output", default="outputs/wam_stability_20261021/fresh_offline_episode_exclusion")
    parser.add_argument("--update-draft", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("requires empty CPU CUDA/LD environment")
    if Path.cwd().resolve() != ROOT:
        raise ValueError("run from S2A repository")
    if args.stage == "export":
        export(args)
    else:
        match(args)


if __name__ == "__main__":
    main()
