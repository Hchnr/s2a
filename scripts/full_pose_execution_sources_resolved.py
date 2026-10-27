"""246 resolve repository-relative historical source declarations before merging."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import sha


def merge(target, path, signature, aliases=None):
    declared = str(path)
    source = Path(path)
    canonical = source.resolve() if source.is_absolute() else (REPO / source).resolve()
    if not source.is_absolute() and not canonical.is_relative_to(REPO):
        raise ValueError(f"Repository-relative source escapes repository: {declared}")
    name = str(canonical)
    if name in target and target[name] != signature:
        raise ValueError(f"Conflicting source SHA for {name}: {declared}")
    target[name] = signature
    if aliases is not None:
        ref = dict(path=name, sha256=signature)
        if declared in aliases and aliases[declared] != ref:
            raise ValueError(f"Conflicting source declaration: {declared}")
        aliases[declared] = ref


def prepare_sources(registration, config, tests):
    runtime, provenance, aliases = {}, {}, {}

    def record(target, path, signature):
        merge(target, path, signature, aliases)

    paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
    paths += [
        config,
        REPO / "plans/v0.0.1/246_relative_provenance_admission_repair_20261026.md",
        REPO / "plans/v0.0.1/245_sqp_control_binding_and_provenance_20261026.md",
        REPO / "plans/v0.0.1/244_full_pose_stationarity_and_source_audit_20261026.md",
        REPO / "tests/fixtures/full_pose_case04_step71.json",
        *[REPO / name for name in tests],
    ]
    for path in paths:
        record(runtime, path.resolve(), sha(path))
    for key, ref in registration.items():
        if not key.endswith("_ref") or not isinstance(ref, dict) or "sha256" not in ref:
            continue
        assert sha(ref["path"]) == ref["sha256"], key
        record(runtime, ref["path"], ref["sha256"])
        artifact = json.loads(Path(ref["path"]).read_text())
        for source, signature in artifact.get("source_sha256", {}).items():
            record(provenance, source, signature)
    prior_root = REPO / "outputs/current_coupled_control_20261026"
    prior_path = prior_root / "closure_verified.json"
    prior = json.loads(prior_path.read_text())
    assert prior["complete"] and prior["verified_cases"] == 18 and prior["owner_normal_wait_verified"]
    record(runtime, prior_path, sha(prior_path))
    for source, signature in prior["source_sha256"].items():
        record(provenance, source, signature)
    for index, profile in enumerate(registration["profiles"]):
        for key in ("trajectory_ref", "case_ref"):
            ref = profile[key]
            assert sha(ref["path"]) == ref["sha256"]
            record(runtime, ref["path"], ref["sha256"])
        case = next((prior_root / f"case{index:02d}").glob("simulation/*/*"))
        manifest_path = case / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["record"]["initial_state_sha256"] == profile["record"]["initial_state_sha256"]
        record(runtime, manifest_path, sha(manifest_path))
        # Actual native simulator code/assets remain live-bound, not relegated to history.
        for source, signature in manifest["simulator_source_sha256"].items():
            record(runtime, source, signature)
    for path, signature in provenance.items():
        assert runtime.get(path, signature) == signature
        if path not in runtime:
            assert sha(path) == signature, path
    # Recheck each live source after discovery, including native code/assets.
    for path, signature in runtime.items():
        assert sha(path) == signature, path
    return runtime, provenance, aliases
