"""Distinguish a catalog path match from historical publication byte identity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_followup_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen config")
    matched_path = out / "alignment/matched_targets.json"
    matched = json.loads(matched_path.read_text())
    root = Path(matched["catalog_data"]["root"])
    report_path = Path(c["training_catalog"]).parent / "REPORT.json"
    contract_path = root / "meta/eef_state_contract.json"
    norm_path = root / "meta/action_normalization.json"
    catalog_report = json.loads(report_path.read_text())
    contract = json.loads(contract_path.read_text())
    inputs = {str(p): sha256(p) for p in (matched_path, report_path, contract_path, norm_path)}
    metadata = []
    for relative, expected in catalog_report["source_metadata_sha256"].items():
        path = root / relative
        current = sha256(path)
        inputs[str(path)] = current
        metadata.append(
            dict(path=str(path), catalog_sha256=expected, current_sha256=current, equal=expected == current)
        )
    policy = json.loads(Path(c["policy_config"]).read_text())
    config_path = Path(policy["checkpoint_run"]) / "config.yaml"
    inputs[str(config_path)] = sha256(config_path)
    frozen = yaml.safe_load(config_path.read_text())["model"]["config"]["action128_spec"]
    frozen = frozen["transforms"][frozen["registry"]["domains"].index(policy["action_domain"])]
    current = json.loads(norm_path.read_text())["domains"][policy["action_domain"]]["transform"]
    normalization = {}
    for key in ("scale", "bias", "normalization_mask", "clip_min", "clip_max"):
        a, b = np.asarray(frozen[key]), np.asarray(current[key])
        normalization[key] = dict(
            equal=bool(np.array_equal(a, b)),
            max_abs=float(np.max(abs(a.astype(float) - b.astype(float)))),
            checkpoint_active_first10=a.tolist()[:10] if a.ndim else a.tolist(),
            current_active_first10=b.tolist()[:10] if b.ndim else b.tolist(),
        )
    meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
    meta = pq.read_table(meta_path, columns=["episode_index", "data_path"]).to_pylist()
    cases = []
    for record in matched["cases"]:
        relative = next(r["data_path"] for r in meta if r["episode_index"] == record["r2_episode"])
        path = root / relative
        inputs[str(path)] = inputs.get(str(path)) or sha256(path)
        columns = ["action", "action_dim_mask", "action_control_mask", "action_valid"]
        rows = pq.read_table(
            path,
            columns=columns + ["eef_state_revision.original_" + k for k in columns],
            filters=[("episode_index", "=", record["r2_episode"])],
        ).to_pylist()
        comparisons = {}
        for key in columns:
            a = np.asarray([r[key] for r in rows])
            b = np.asarray([r["eef_state_revision.original_" + key] for r in rows])
            comparisons[key] = dict(
                equal=bool(np.array_equal(a, b)),
                max_abs=float(np.max(abs(a.astype(float) - b.astype(float)))),
            )
        receipt_path = root / "_receipts" / Path(relative).with_suffix(".json")
        receipt = json.loads(receipt_path.read_text())
        inputs[str(receipt_path)] = sha256(receipt_path)
        if receipt["output_sha256"] != inputs[str(path)]:
            raise ValueError("publication receipt does not match current Parquet")
        cases.append(
            dict(
                trial=record["trial"],
                r2_episode=record["r2_episode"],
                rows=len(rows),
                comparisons=comparisons,
                receipt_verified=True,
            )
        )
    result = dict(
        code=code_fingerprint(),
        inputs_sha256=inputs,
        metadata=metadata,
        catalog_info_matches_pre_state_revision=(
            contract["source_metadata_sha256"][str(root / "meta/info.json")]
            == matched["source_hashes"]["catalog_info_sha256"]
        ),
        checkpoint_vs_current_normalization=normalization,
        cases=cases,
        historical_training_input_bytes_verified=False,
        training_consumption_verified=False,
        notes=[
            "Current path identity does not prove equality to historical training bytes.",
            "Original-action equality and publication receipts constrain the documented state revision, not all earlier history.",
            "Unscaled rotation/opening diagnostics do not depend on translation normalization changes.",
        ],
    )
    write_json(out / "alignment/revision.json", result)
    print(
        json.dumps(
            dict(
                metadata_unchanged=sum(r["equal"] for r in metadata),
                metadata_total=len(metadata),
                normalization=normalization,
                cases=cases,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
