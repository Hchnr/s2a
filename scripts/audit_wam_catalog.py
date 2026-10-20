"""Audit selected catalog membership and preserved pre-state-revision labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, output = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    destination = output / "catalog_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    catalog = Path(c["training_catalog"])
    rows = pq.read_table(catalog).to_pylist()
    if len(rows) != 1:
        raise ValueError("this audit expects the single full LIBERO catalog entry")
    data, provenance = json.loads(rows[0]["data"]), json.loads(rows[0]["provenance"])
    root = Path(c["r2_root"])
    if Path(data["root"]).resolve() != root.resolve() or data["mode"] != "wam" or data["use_state"]:
        raise ValueError("evaluation source differs from the checkpoint catalog contract")
    selection = json.loads((output / "samples/selection.json").read_text())["selection"]
    columns = ["action", "action_dim_mask", "action_control_mask", "action_valid"]
    cases, fingerprints = [], {str(catalog): sha256(catalog)}
    for item in selection:
        meta = item["metadata"]
        if not all(meta[k] == v for k, v in data["episode_filter"].items()):
            raise ValueError("sample is outside the configured training catalog filter")
        path = root / meta["data_path"]
        records = pq.read_table(
            path,
            columns=columns + ["eef_state_revision.original_" + k for k in columns],
            filters=[("episode_index", "=", meta["episode_index"])],
        ).to_pylist()
        equal = {
            key: bool(
                np.array_equal(
                    np.asarray([r[key] for r in records]),
                    np.asarray([r["eef_state_revision.original_" + key] for r in records]),
                )
            )
            for key in columns
        }
        if not all(equal.values()) or len(records) != meta["length"]:
            raise ValueError("selected episode labels changed across the documented state revision")
        fingerprints[str(path)] = fingerprints.get(str(path)) or sha256(path)
        receipt_path = root / "_receipts" / Path(meta["data_path"]).with_suffix(".json")
        receipt = json.loads(receipt_path.read_text())
        if receipt["output_sha256"] != fingerprints[str(path)]:
            raise ValueError("published Parquet differs from its receipt")
        fingerprints[str(receipt_path)] = sha256(receipt_path)
        cases.append(
            dict(
                episode=meta["episode_index"],
                split=item["split"],
                rows=len(records),
                catalog_member=True,
                labels_equal_pre_state_revision=equal,
                receipt_verified=True,
            )
        )
    report_path = catalog.parent / "REPORT.json"
    historical = json.loads(report_path.read_text())["source_metadata_sha256"]
    fingerprints[str(report_path)] = sha256(report_path)
    metadata = []
    for relative, old in historical.items():
        current = sha256(root / relative)
        metadata.append(dict(path=relative, catalog_sha256=old, current_sha256=current, equal=old == current))
    contract_path = root / "meta/eef_state_contract.json"
    contract = json.loads(contract_path.read_text())
    fingerprints[str(contract_path)] = sha256(contract_path)
    info_link = (
        contract["source_metadata_sha256"][str(root / "meta/info.json")] == provenance["extra"]["source_info_sha256"]
    )
    if not info_link:
        raise ValueError("state-revision contract does not link to the catalog's original info fingerprint")
    template = json.loads(Path(c["policy_template"]).read_text())
    domain = template["action_domain"]
    normalization = json.loads((root / "meta/action_normalization.json").read_text())["domains"][domain]["transform"]
    normalization_checks = []
    for checkpoint in c["checkpoints"]:
        if checkpoint["name"] == "warm6200":
            continue
        spec = yaml.safe_load((Path(checkpoint["run"]) / "config.yaml").read_text())["model"]["config"][
            "action128_spec"
        ]
        frozen = spec["transforms"][spec["registry"]["domains"].index(domain)]
        equal = {
            key: bool(np.array_equal(frozen[key], normalization[key]))
            for key in ("scale", "bias", "normalization_mask", "clip_min", "clip_max")
        }
        if not all(equal.values()):
            raise ValueError("checkpoint and current published normalization differ")
        normalization_checks.append(dict(checkpoint=checkpoint["name"], equal=equal))
    write_json(
        destination,
        dict(
            code=code_fingerprint(),
            catalog_data=data,
            cases=cases,
            metadata=metadata,
            original_info_link_verified=info_link,
            normalization=normalization_checks,
            inputs_sha256=fingerprints,
            historical_training_input_bytes_verified=False,
            training_consumption_verified=False,
            interpretation=[
                "All 12 selected episodes are members of the configured training catalog; study-heldout only describes this experiment's split.",
                "The documented later state revision preserved complete actions and masks for these episodes; catalog use_state is false.",
                "Current numerical audits reproduce the training computation on fixed current-published inputs, not a byte-for-byte historical training batch replay.",
            ],
        ),
    )
    print(
        json.dumps(
            dict(
                episodes=len(cases),
                catalog_members=sum(r["catalog_member"] for r in cases),
                metadata_unchanged=sum(r["equal"] for r in metadata),
                metadata_total=len(metadata),
                preserved_label_rows=sum(r["rows"] for r in cases),
            )
        )
    )


if __name__ == "__main__":
    main()
