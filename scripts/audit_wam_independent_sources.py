"""Join frozen physics demos to R1 commands, R2 catalog and prior offline samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_independent_20261020.json")
    parser.add_argument("--provenance-config", default="configs/libero_provenance_20261019.json")
    parser.add_argument(
        "--prior-offline-selection", default="outputs/wam_checkpoint_acceptance_20261019v2/samples/selection.json"
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    output = root / "source_audit.json"
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads(Path(args.provenance_config).read_text())
    r1, r2 = Path(provenance["r1_root"]), Path(c["r2_root"])
    if r2 != Path(provenance["r2_root"]):
        raise ValueError("source contract differs")
    meta1 = r1 / "meta/episodes/chunk-000/file-000.parquet"
    meta2 = r2 / "meta/episodes/chunk-000/file-000.parquet"
    catalog = Path(c["training_catalog"])
    catalog_rows = pq.read_table(catalog).to_pylist()
    if len(catalog_rows) != 1:
        raise ValueError("expected the audited single LIBERO catalog entry")
    data = json.loads(catalog_rows[0]["data"])
    if Path(data["root"]).resolve() != r2.resolve():
        raise ValueError("training catalog root differs")
    fingerprints = {
        str(p): sha256(p)
        for p in (
            meta1,
            meta2,
            catalog,
            Path(args.provenance_config),
            Path(args.prior_offline_selection),
            root / "simulation/fixtures.json",
        )
    }
    prior = {
        x["metadata"]["episode_index"] for x in json.loads(Path(args.prior_offline_selection).read_text())["selection"]
    }
    current = pq.read_table(meta2).to_pylist()
    metadata = [row for row in pq.read_table(meta1).to_pylist() if any(t in row["tasks"] for t in c["tasks"])]
    groups = {}
    for chunk, index in sorted({(r["data/chunk_index"], r["data/file_index"]) for r in metadata}):
        path = r1 / f"data/chunk-{chunk:03d}/file-{index:03d}.parquet"
        ids = [r["episode_index"] for r in metadata if (r["data/chunk_index"], r["data/file_index"]) == (chunk, index)]
        rows = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "action.native_control"],
            filters=[("episode_index", "in", ids)],
        ).to_pylist()
        fingerprints[str(path)] = sha256(path)
        for row in sorted(rows, key=lambda x: (x["episode_index"], x["frame_index"])):
            groups.setdefault(row["episode_index"], []).append(row["action.native_control"])
    cases = []
    for record in json.loads((root / "simulation/fixtures.json").read_text())["records"]:
        if sha256(record["expert_path"]) != record["expert_sha256"]:
            raise ValueError("expert commands changed")
        actions = np.load(record["expert_path"])
        rot = Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
        projected = np.column_stack((actions[:, :3], rot[:, :, 0], rot[:, :, 1], (1 - actions[:, 6]) / 2))
        matches = []
        for index, values in groups.items():
            values = np.asarray(values)
            if values.shape == projected.shape:
                error = float(np.max(abs(values - projected)))
                if error < provenance["max_native_projection_error"]:
                    matches.append(dict(r1_episode=index, max_abs=error))
        if len(matches) > 1:
            raise ValueError("ambiguous full command-content identity")
        related = []
        if matches:
            identity = str(r1) + ":" + str(matches[0]["r1_episode"])
            for row in current:
                if json.loads(row["source_episode_id"]).get("source_id") == identity:
                    related.append(
                        dict(
                            r2_episode=row["episode_index"],
                            catalog_eligible=all(row.get(k) == v for k, v in data["episode_filter"].items()),
                            prior_offline_sample=row["episode_index"] in prior,
                        )
                    )
        cases.append(
            dict(
                record=record["name"],
                task=record["task"],
                demo=record["demo"],
                r1_matches=matches,
                r2=related,
                pretraining_consumption="unknown; catalog eligibility is not a training consumption ledger",
            )
        )
    write_json(
        output,
        dict(
            code=code_fingerprint(),
            files_sha256=fingerprints,
            cases=cases,
            interpretation="Physical-study holdout only; do not claim pretraining-unseen data. Unmatched sources are unresolved, not unseen.",
        ),
    )
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
