"""Detached CPU owner for one056 aggregate producer and independent full QR."""

import argparse
import json
import os
from pathlib import Path

from run_wam_clean_bridge import check_owned_command
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_aggregation import ROWS, admit_aggregation
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.durable_stage_supervisor import accept_detached_launch, cpu_environment, launch_detached

ROOT = REPO / "outputs/clean_on_policy_aggregation_20261024/aggregate_fit"


def admit(regref):
    spec = read(regref)
    require(
        Path(spec["root"]) == ROOT and Path(regref["path"]) == ROOT / "registration.json",
        "056 fixed aggregate output root",
    )
    admit_aggregation(spec)
    require(
        {r["path"]: r["sha256"] for r in spec["source_refs"]} == spec["source_sha256"],
        "056 aggregate source reference list differs",
    )
    for mapping in (spec["source_sha256"], spec["snapshot_sha256"]):
        require(bool(mapping), "056 empty aggregate closure")
        for path, digest in mapping.items():
            checked(dict(path=path, sha256=digest))
    for field, name in (
        ("entrypoint_ref", "fit_wam_clean_aggregation.py"),
        ("auditor_ref", "verify_wam_clean_aggregation.py"),
        ("owner_ref", "run_wam_clean_aggregation.py"),
    ):
        require(
            spec[field] == reference(REPO / "scripts" / name)
            and spec["source_sha256"].get(spec[field]["path"]) == spec[field]["sha256"],
            "056 aggregate execution role differs",
        )
    require(
        spec["cpu_python"] == str(REPO / ".venv/bin/python") and spec["fresh_authorized"] is False,
        "056 CPU/fresh policy differs",
    )
    return spec


def worker_command(spec, regref, *, independent=False):
    return [
        spec["cpu_python"],
        "scripts/verify_wam_clean_aggregation.py" if independent else "scripts/fit_wam_clean_aggregation.py",
        "--registration",
        regref["path"],
    ]


def execute(regref):
    spec = admit(regref)
    dry = read(reference(ROOT / "successful_cli_dry.json"))
    require(
        dry["passed"] is True
        and dry["registration_ref"] == regref
        and dry["command"] == worker_command(spec, regref) + ["--dry"],
        "056 exact actual producer dry required",
    )
    folder = ROOT / "orchestration"
    folder.mkdir(exist_ok=False)
    result = dict(
        schema_version="s2a.clean-aggregation-owner.v1",
        complete=False,
        passed=False,
        registration_ref=regref,
        owned_commands=[],
        automatic_retry=False,
        fresh_authorized=False,
        validation_authorized=False,
    )
    try:
        producer = owned_command(worker_command(spec, regref), cpu_environment(), folder / "producer", timeout=43200)
        result["owned_commands"].append(producer)
        check_owned_command(producer)
        pref = reference(ROOT / "result.json")
        produced = read(pref)
        require(
            produced["complete"] is produced["passed"] is True
            and produced["producer_fits"] == 1
            and produced["fit_metadata"]["rows"] == ROWS
            and produced["registration_ref"] == regref,
            "056 full aggregate producer required",
        )
        audit_spec = dict(
            schema_version="s2a.clean-aggregation-audit-registration.v1",
            producer_ref=pref,
            entrypoint_ref=spec["auditor_ref"],
            source_refs=spec["source_refs"],
            independent_reconstructions=1,
            rtol=1e-9,
            atol=1e-11,
            automatic_retry=False,
        )
        aref = publish(ROOT / "audit_registration.json", audit_spec)
        auditor = owned_command(
            worker_command(spec, aref, independent=True), cpu_environment(), folder / "independent", timeout=43200
        )
        result["owned_commands"].append(auditor)
        check_owned_command(auditor)
        vref = reference(ROOT / "independent_verified.json")
        verified = read(vref)
        require(
            verified["complete"] is verified["passed"] is True
            and verified["producer_ref"] == pref
            and verified["params_ref"] == produced["params_ref"]
            and verified["independent_reconstructions"] == 1
            and verified["qr_metadata"]["rows"] == ROWS,
            "056 independent full aggregate QR required",
        )
        admit(regref)
        result.update(
            complete=True,
            passed=True,
            producer_ref=pref,
            verification_ref=vref,
            params_ref=produced["params_ref"],
            candidate_numerically_qualified=True,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result["available_failure_receipts"] = {
            str(path.relative_to(folder)): reference(path)
            for path in folder.rglob("*.json")
            if path.name in ("launch.json", "exit.json", "release.json")
        }
        publish(folder / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "056 aggregate owner requires empty CUDA/LD",
    )
    regref = reference(args.registration)
    spec = admit(regref)
    command = [spec["cpu_python"], "scripts/run_wam_clean_aggregation.py", "--registration", regref["path"]]
    if args.launch:
        print(
            json.dumps(launch_detached(command, directory=ROOT / "supervisor", source_sha256=spec["source_sha256"])),
            flush=True,
        )
        return
    accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=spec["source_sha256"])
    execute(regref)


if __name__ == "__main__":
    main()
