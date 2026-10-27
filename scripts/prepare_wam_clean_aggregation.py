"""Register one056 aggregate fit only after normally completed/audited D2."""

import json
import subprocess
from pathlib import Path

from prepare_wam_synchronous_validation import source_closure

from s2a.artifacts import REPO
from s2a.clean_training_aggregation import RIDGE, ROWS, admit_aggregation
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment

ROOT = REPO / "outputs/clean_on_policy_aggregation_20261024"


def prepare():
    output = ROOT / "aggregate_fit"
    require(not output.exists(), "056 aggregate namespace already exists; no automatic retry")
    previous = read(reference(REPO / "outputs/clean_regularization_probe_20261024/registration.json"))
    spec = dict(
        schema_version="s2a.clean-aggregation-fit-registration.v1",
        rows=ROWS,
        ridge=RIDGE,
        producer_fits=1,
        automatic_retry=False,
        d2_owner_ref=reference(ROOT / "orchestration/result.json"),
        previous_fit_ref=previous["previous_fit_ref"],
        previous_verification_ref=previous["previous_verification_ref"],
    )
    # This must precede output creation, snapshots and any attempt/dry subprocess.
    admit_aggregation(spec)
    entries = [
        "scripts/prepare_wam_clean_aggregation.py",
        "scripts/run_wam_clean_aggregation.py",
        "scripts/fit_wam_clean_aggregation.py",
        "scripts/verify_wam_clean_aggregation.py",
        "src/s2a/clean_training_aggregation.py",
        "tests/test_clean_training_aggregation.py",
        "configs/wam_clean_on_policy_aggregation_20261024.json",
        "plans/v0.0.1/056_clean_on_policy_aggregation_20261024.md",
    ]
    sources = source_closure(entries)
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    output.mkdir()
    (output / "source").mkdir()
    snapshots = {}
    for index, (path, digest) in enumerate(sorted(sources.items())):
        target = output / "source" / f"{index:04d}_{Path(path).name}"
        with target.open("xb") as stream:
            stream.write(checked(dict(path=path, sha256=digest)).read_bytes())
        checked(dict(path=str(target), sha256=digest))
        snapshots[str(target)] = digest
    spec.update(
        root=str(output),
        entrypoint_ref=reference(REPO / "scripts/fit_wam_clean_aggregation.py"),
        auditor_ref=reference(REPO / "scripts/verify_wam_clean_aggregation.py"),
        owner_ref=reference(REPO / "scripts/run_wam_clean_aggregation.py"),
        source_refs=[dict(path=k, sha256=v) for k, v in sources.items()],
        source_sha256=sources,
        snapshot_sha256=snapshots,
        cpu_python=str(REPO / ".venv/bin/python"),
        fresh_authorized=False,
    )
    regref = publish(output / "registration.json", spec)
    command = [spec["cpu_python"], "scripts/fit_wam_clean_aggregation.py", "--registration", regref["path"], "--dry"]
    child = subprocess.run(
        command, cwd=REPO, env=cpu_environment(), text=True, capture_output=True, timeout=600, check=False
    )
    record = dict(
        passed=child.returncode == 0,
        command=command,
        registration_ref=regref,
        returncode=child.returncode,
        stdout=child.stdout,
        stderr=child.stderr,
    )
    if record["passed"]:
        dry = json.loads(child.stdout)
        require(
            dry["passed"] is dry["complete"] is True
            and dry["producer_fits"] == dry["future_labels_decoded"] == 0
            and dry["rows"] == ROWS,
            "056 actual fit dry response differs",
        )
    publish(output / ("successful_cli_dry.json" if record["passed"] else "failed_cli_dry.json"), record)
    require(record["passed"], "056 actual fit CLI dry failed")
    return regref


if __name__ == "__main__":
    print(json.dumps(prepare()), flush=True)
