"""Check strict XYZ/gripper sequence identity for two unresolved historical episodes.

This is partial command-content evidence only. It does not infer rotation formats,
initial-state identity, physical synchronization or full native-control identity.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_stat(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("partial provenance audit requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("output must be a new repository outputs artifact")
    source_root = ROOT / "outputs/wam_stability_20261021/fresh_offline_episode_exclusion"
    export_path, match_path = source_root / "export.json", source_root / "matches.json"
    exported, prior = json.loads(export_path.read_text()), json.loads(match_path.read_text())
    archive = Path(exported["archive"]["path"])
    assert sha(archive) == exported["archive"]["sha256"]
    input_sha = {str(p): sha(p) for p in (Path(__file__), export_path, match_path, archive)}
    sources = prior["searched_sources"]
    tolerance = prior["projection_tolerance"]
    assert tolerance == 1e-6
    for source in sources:
        assert file_stat(source["path"]) == (source["bytes"], source["mtime_ns"], source["ctime_ns"])
    with np.load(archive) as saved:
        expected = {episode: np.asarray(saved[f"r1_{episode}"], dtype=float) for episode in (384, 720)}
    metadata = {r["r1_episode"]: r for r in exported["episodes"]}
    records = {
        episode: dict(
            r1_episode=episode,
            r2_episode=metadata[episode]["r2_episode"],
            task=metadata[episode]["task"],
            task_index=metadata[episode]["task_index"],
            r1_rows=len(values),
            exact_four_column_matches=[],
        )
        for episode, values in expected.items()
    }
    searched = 0
    windows = {episode: 0 for episode in expected}
    for task_index, source in enumerate(sources):
        with h5py.File(source["path"], "r") as handle:
            demos = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
            assert len(demos) == 50
            for demo in demos:
                searched += 1
                native = np.asarray(handle["data"][demo]["actions"][:], dtype=float)
                assert native.ndim == 2 and native.shape[1] == 7 and np.isfinite(native).all()
                four = np.column_stack((native[:, :3], (1 - native[:, 6]) / 2))
                # Keep exactly the established full-control auditor projection;
                # no Euler/quaternion/transpose/scale alternatives are searched.
                rotations = Rotation.from_rotvec(native[:, 3:6]).as_matrix()
                columns = np.column_stack((rotations[:, :, 0], rotations[:, :, 1]))
                for episode, values in expected.items():
                    r1_four = values[:, [0, 1, 2, 9]]
                    n, m = len(values), len(native)
                    direction = "whole_r1_in_hdf" if n < m else "whole_hdf_in_r1" if m < n else "whole_both"
                    for offset in range(abs(m - n) + 1):
                        r1_start, hdf_start = (0, offset) if n <= m else (offset, 0)
                        count = min(m, n)
                        windows[episode] += 1
                        difference = np.abs(r1_four[r1_start : r1_start + count] - four[hdf_start : hdf_start + count])
                        if float(difference.max()) >= tolerance:
                            continue
                        rotational = np.abs(
                            values[r1_start : r1_start + count, 3:9] - columns[hdf_start : hdf_start + count]
                        )
                        four_error, rotation_error = float(difference.max()), float(rotational.max())
                        records[episode]["exact_four_column_matches"].append(
                            dict(
                                task_index=task_index,
                                source_hdf5=source["path"],
                                source_hdf5_sha256=source["sha256"],
                                demo=demo,
                                hdf_rows=m,
                                direction=direction,
                                r1_slice_half_open=[r1_start, r1_start + count],
                                hdf_slice_half_open=[hdf_start, hdf_start + count],
                                compared_rows=count,
                                xyz_gripper_max_abs=four_error,
                                xyz_gripper_per_column_max_abs=difference.max(axis=0).tolist(),
                                rotation_columns_max_abs=rotation_error,
                                rotation_columns_rmse=float(np.sqrt(np.mean(rotational**2))),
                                rotation_columns_per_column_max_abs=rotational.max(axis=0).tolist(),
                                rotation_rows_over_tolerance=int(np.count_nonzero(rotational.max(axis=1) >= tolerance)),
                                full10_projection_max_abs=max(four_error, rotation_error),
                                full10_projection_matches_at_same_slice=max(four_error, rotation_error) < tolerance,
                            )
                        )
    assert searched == 150
    for episode, record in records.items():
        matches = record["exact_four_column_matches"]
        record.update(
            searched_windows=windows[episode],
            matching_windows=len(matches),
            matching_demos=len({(m["source_hdf5"], m["demo"]) for m in matches}),
            other_matching_windows=max(0, len(matches) - 1),
            unique_four_column_content_match=len(matches) == 1 and matches[0]["task_index"] == record["task_index"],
            full_native_control_identity_established=False,
            initial_state_identity_established=False,
            image_pose_physical_synchronization_established=False,
        )
        record["status"] = "unique_partial_content_match" if record["unique_four_column_content_match"] else "unknown"
    for source in sources:
        assert file_stat(source["path"]) == (source["bytes"], source["mtime_ns"], source["ctime_ns"])
    for path, digest in input_sha.items():
        assert sha(path) == digest
    result = dict(
        schema_version="s2a.partial-native-command-identity.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        tolerance=tolerance,
        strict_less_than_tolerance=True,
        projection="R1.native_control[0,1,2,9] versus HDF actions[0,1,2] and (1-actions[6])/2",
        rotation_diagnostic="Unchanged prior auditor projection: scipy Rotation.from_rotvec(HDF actions[3:6]) "
        "first/second matrix columns versus R1.native_control[3:9]. No alternative rotation format is searched.",
        sequence_rule="Entire shorter sequence against every contiguous fixed-offset slice of the longer; "
        "equal-length sequences counted once. No interpolation, arbitrary skips, prefix removal or temporal scaling.",
        source_catalogs=3,
        searched_demos=searched,
        searched_sources_bound_by_prior_full_sha_and_current_stat=sources,
        source_hash_policy="Reuse prior complete HDF byte SHA; current size/mtime/ctime checked before and after read.",
        input_sha256=input_sha,
        records=list(records.values()),
        new_model_calls=0,
        new_physics_steps=0,
        draft_read=False,
        draft_updated=False,
        candidate_evaluation_read=False,
        scope="Partial command-content evidence for conservative source exclusion only. Full native-control, "
        "initial-state and physical image/pose synchronization identities remain unestablished; no nearest-match identity claim.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), sha256=sha(output), records=list(records.values()))))


if __name__ == "__main__":
    main()
