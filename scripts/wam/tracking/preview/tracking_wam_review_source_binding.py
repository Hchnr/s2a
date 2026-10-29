"""Bind emitted runtime manifests to frozen sources and actual launch arguments."""

from pathlib import Path


def verify_source_binding(ev, physical, prepared, worker, config_ref, command):
    launch = ev.json(command["launch"])
    argv = launch["command"]
    expected = [
        ("--case-config", config_ref),
        ("--prepared", worker["prepared_ref"]),
        ("--execution-registration", worker["execution_registration_ref"]),
    ]
    for flag, ref in expected:
        if argv[argv.index(flag) + 1] != ref["path"] or argv[argv.index(flag + "-sha256") + 1] != ref["sha256"]:
            raise ValueError("Actual pre-execution command differs from bound artifact")
        ev.path(ref)
    path = Path(worker["physical_case_dir"])
    manifest = physical.json(path / "manifest.json")
    signatures = manifest["execution_sha256"]
    if signatures.get(config_ref["path"]) != config_ref["sha256"]:
        raise ValueError("Runtime manifest does not bind actual dispatched config")
    matched = 0
    for file, digest in signatures.items():
        physical.check(file, digest)
        if file in prepared["source_sha256"]:
            if prepared["source_sha256"][file] != digest:
                raise ValueError("Runtime source differs from pre-execution frozen source")
            matched += 1
    if not matched:
        raise ValueError("No frozen runtime source binding")
    if "execution_snapshot_ref" in worker:
        snapshot = ev.json(worker["execution_snapshot_ref"])
        if snapshot["source_sha256"].get(config_ref["path"]) != config_ref["sha256"] or any(
            signatures.get(k) != v for k, v in snapshot["source_sha256"].items()
        ):
            raise ValueError("Explicit pre-execution snapshot differs")
    return dict(
        complete=True,
        passed=True,
        actual_launch_ref=command["launch"],
        config_ref=config_ref,
        matched_frozen_runtime_sources=matched,
        separate_producer_snapshot_present="execution_snapshot_ref" in worker,
        scope="Actual pre-execution argv path/SHA plus prepared immutable sources and emitted runtime manifest; does not invent a missing snapshot artifact",
    )
