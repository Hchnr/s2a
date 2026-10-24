"""Own the motor hold diagnostic in the user-selected relocated workspace."""

import argparse
import json
import os
import shutil
from pathlib import Path

from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.native_environment import strict_native_environment
from s2a.relocated_artifacts import RelocatedArtifacts
from s2a.source_archive import verify_archived_sources

CONFIG = REPO / "configs/native_motor_hold_20261022.json"
RECOVERY = REPO / "outputs/external_deletion_recovery_20261022"
LOCATOR = RelocatedArtifacts(REPO, RECOVERY)
SOURCES = (
    "plans/v0.0.1/032_native_motor_hold_diagnostic_20261022.md",
    "plans/v0.0.1/033_workspace_relocation_20261022.md",
    "configs/native_motor_hold_20261022.json",
    "scripts/run_native_motor_hold_diagnostic.py",
    "scripts/collect_native_motor_hold_diagnostic.py",
    "scripts/verify_native_motor_hold_diagnostic.py",
    "tests/test_native_motor_hold_collection.py",
    "tests/test_native_motor_hold_verification.py",
    "src/s2a/source_archive.py",
    "tests/test_source_archive.py",
    "src/s2a/relocated_artifacts.py",
    "tests/test_relocated_artifacts.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def resolve_path(path):
    return LOCATOR.resolve_path(path)


def reference(path):
    return LOCATOR.reference(path)


def read(ref):
    return LOCATOR.read(ref)


def publish(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    return reference(path)


def pose_lineage(ref):
    prior = read(ref)
    mapping, snapshots, relocation = {}, {}, []
    old_cosmos = Path("/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm")
    for obj in (prior, prior["static_prepared"]):
        for path, digest in obj["snapshot_sha256"].items():
            snapshots[str(resolve_path(path))] = digest
        for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
            for path, digest in obj[key].items():
                actual = Path(path) if Path(path).is_relative_to(old_cosmos) else resolve_path(path)
                target = str(actual)
                require(target not in mapping or mapping[target] == digest, "conflicting lineage SHA")
                mapping[target] = digest
                if target != path:
                    relocation.append(dict(declared=path, actual=target, sha256=digest))
    evidence = verify_archived_sources(mapping, snapshots, missing_prefix=old_cosmos)
    evidence["relocated_references"] = relocation
    return prior, evidence


def read_prepared(ref):
    strict_native_environment()
    prepared = read(ref)
    require(json.loads(CONFIG.read_text()) == prepared["config"], "motor configuration changed")
    prior, evidence = pose_lineage(prepared["config"]["pose_prepared_ref"])
    require(prior == prepared["pose_prepared"] and evidence == prepared["source_availability"], "lineage changed")
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        for path, digest in prepared[key].items():
            require(sha256(resolve_path(path)) == digest, "frozen source/input changed")
    return prepared


def prepare():
    strict_native_environment()
    config = json.loads(CONFIG.read_text())
    require(
        config["seed"] == 42
        and config["methods"] == ["motor_hold", "pose_motor_hold"]
        and config["physics_steps"] == 1648
        and config["environment_initializations"] == 8
        and config["current_canvases"] == 40
        and config["grip_action_calls"] == 41200
        and not any(config[k] for k in ("training_authorized", "fresh_authorized", "automatic_retry")),
        "fixed motor diagnostic contract changed",
    )
    prior, evidence = pose_lineage(config["pose_prepared_ref"])
    require([r["name"] for r in prior["records"]] == config["cases"], "four-case roster changed")
    for key in ("pose_collection_ref", "pose_verified_ref"):
        document = read(config[key])
        require(document["passed"] and document["complete"], "prior complete cache/audit required")
    records = [dict(r, expert_path=str(resolve_path(r["expert_path"]))) for r in prior["records"]]
    require(2 * sum(r["native_steps"] + 40 for r in records) == 1648, "physics budget changed")
    inputs = {v["path"]: v["sha256"] for k, v in config.items() if k.endswith("_ref")}
    cv2_refs = read(config["cv2_runtime_refs_ref"])
    for ref in cv2_refs:
        inputs[ref["path"]] = ref["sha256"]
    runtime = REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite"
    for name in ("models/grippers/panda_gripper.py", "robots/manipulator.py", "robots/single_arm.py"):
        path = runtime / name
        inputs[str(path)] = sha256(path)
    sources = {str(REPO / p): sha256(REPO / p) for p in SOURCES}
    root = REPO / config["output_dir"]
    require(root == REPO / "outputs/wam_stability_20261021/native_motor_hold_diagnostic", "root changed")
    (root / "preparation/snapshots").mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for index, (path, digest) in enumerate(sources.items()):
        target = root / "preparation/snapshots" / f"{index:02d}_{Path(path).name}"
        shutil.copyfile(path, target)
        require(sha256(target) == digest, "source snapshot differs")
        snapshots[str(target)] = digest
    ref = publish(
        root / "preparation/prepared.json",
        dict(
            root=str(root),
            config=config,
            pose_prepared=prior,
            records=records,
            historical_records=prior["records"],
            source_availability=evidence,
            source_sha256=sources,
            snapshot_sha256=snapshots,
            input_sha256=inputs,
            cv2_probe_ref=config["cv2_probe_ref"],
            cv2_runtime_source_refs=cv2_refs,
        ),
    )
    read_prepared(ref)
    print(json.dumps(ref))


def execute(ref, dry_run):
    prepared = read_prepared(ref)
    root = Path(prepared["root"])
    common = ["--prepared", ref["path"], "--prepared-sha256", ref["sha256"], "--seed", "42"]
    commands = [
        [str(REPO / "outputs/libero_env311/bin/python"), "scripts/collect_native_motor_hold_diagnostic.py", *common],
        [str(REPO / ".venv/bin/python"), "scripts/verify_native_motor_hold_diagnostic.py", *common],
    ]
    if dry_run:
        print(json.dumps(dict(commands=commands, cuda_visible_devices="", new_model_calls=0)))
        return
    require(not any((root / p).exists() for p in ("collection", "review", "orchestration")), "refuses retry")
    folder = root / "orchestration"
    folder.mkdir()
    result = dict(passed=False, complete=False, prepared_ref=ref, owned_commands=[], training_authorized=False)
    try:
        for phase, command in zip(("worker", "independent"), commands, strict=True):
            read_prepared(ref)
            result["owned_commands"].append(
                owned_command(
                    command,
                    dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
                    folder / phase,
                    timeout=1800,
                )
            )
        audit = read(reference(root / "review/verified.json"))
        require(audit["passed"] and audit["complete"], "independent comparison failed")
        read_prepared(ref)
        result.update(passed=True, complete=True, review_ref=reference(root / "review/verified.json"))
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(folder / "result.json", result)
    print(json.dumps(reference(folder / "result.json")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        execute(dict(path=str(Path(args.prepared).resolve()), sha256=args.prepared_sha256), args.dry_run)


if __name__ == "__main__":
    main()
