"""Own a single, frozen thirty-case settled native collection and audit."""

import argparse
import json
import os
import shutil
from pathlib import Path

from run_native_motor_hold_diagnostic import read, reference, require, resolve_path
from run_native_motor_hold_diagnostic import read_prepared as read_motor_prepared
from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.native_environment import strict_native_environment
from s2a.settled_native import settled_anchor_indices

CONFIG = REPO / "configs/settled_native_training_20261022.json"
SOURCES = (
    "plans/v0.0.1/034_settled_native_training_20261022.md",
    "configs/settled_native_training_20261022.json",
    "scripts/run_settled_native_training.py",
    "scripts/collect_settled_native_training.py",
    "scripts/verify_settled_native_training.py",
    "src/s2a/settled_native.py",
    "tests/test_settled_native.py",
    "tests/test_settled_native_training_verification.py",
)


def publish(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return reference(path)


def records_from(original):
    return [
        dict(
            record,
            expert_path=str(resolve_path(record["expert_path"])),
            anchor_frames=settled_anchor_indices(record["native_steps"]),
        )
        for record in original["records"]
    ]


def lineage(config):
    motor = read_motor_prepared(config["motor_prepared_ref"])
    static = motor["pose_prepared"]["static_prepared"]
    for new, old in (
        ("native_prepared_ref", "native_prepared_ref"),
        ("native_collection_ref", "native_collection_ref"),
        ("native_verified_ref", "native_verification_ref"),
    ):
        require(config[new] == static[old], "original native lineage changed")
    require(read(config["native_prepared_ref"]) == static["native_prepared"], "original preparation differs")
    audit = read(config["motor_verified_ref"])
    require(audit["passed"] and audit["complete"], "complete motor audit required")
    collection = read(config["native_collection_ref"])
    require(collection["passed"] and collection["complete"], "complete original collection required")
    verified = read(config["native_verified_ref"])
    require(
        verified["passed"] and verified["collection_verified"] and verified["evidence_integrity_passed"],
        "complete original independent audit required",
    )
    return motor, static["native_prepared"]


def read_prepared(ref):
    strict_native_environment()
    prepared = read(ref)
    config = json.loads(CONFIG.read_text())
    require(config == prepared["config"], "settled configuration changed")
    motor, original = lineage(config)
    require(motor == prepared["motor_prepared"], "motor lineage changed")
    require(original == prepared["original_prepared"], "native preparation changed")
    require(original["records"] == prepared["historical_records"], "original roster changed")
    require(records_from(original) == prepared["records"], "operational roster changed")
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        for path, digest in prepared[key].items():
            require(sha256(resolve_path(path)) == digest, "frozen source/input changed")
    return prepared


def prepare():
    strict_native_environment()
    config = json.loads(CONFIG.read_text())
    require(
        config["seed"] == 42
        and config["environment_initializations"] == 30
        and config["physics_steps"] == 6180
        and config["current_canvases"] == 150
        and config["settling_steps"] == config["terminal_verification_steps"] == 40
        and not any(
            config[k] for k in ("model_training_authorized", "fresh_performance_authorized", "automatic_retry")
        ),
        "fixed thirty-case contract changed",
    )
    motor, original = lineage(config)
    records = records_from(original)
    require(len(records) == len({r["name"] for r in records}) == 30, "thirty unique original cases required")
    require([sum(r["task_index"] == t for r in records) for t in range(3)] == [10, 10, 10], "task roster")
    require(sum(r["native_steps"] + 80 for r in records) == 6180, "physics budget differs")
    sources = {str(REPO / path): sha256(REPO / path) for path in SOURCES}
    inputs = {v["path"]: v["sha256"] for k, v in config.items() if k.endswith("_ref")}
    cv2_refs = read(config["cv2_runtime_refs_ref"])
    inputs.update({ref["path"]: ref["sha256"] for ref in cv2_refs})
    for relative in (".venv/bin/python", "outputs/libero_env311/bin/python"):
        path = (REPO / relative).resolve(strict=True)
        inputs[str(path)] = sha256(path)
    root = REPO / config["output_dir"]
    require(root == REPO / "outputs/wam_stability_20261021/settled_native_training", "output root changed")
    (root / "preparation/snapshots").mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for index, (path, digest) in enumerate(sources.items()):
        target = root / "preparation/snapshots" / f"{index:02d}_{Path(path).name}"
        shutil.copyfile(path, target)
        require(sha256(target) == digest, "snapshot differs")
        snapshots[str(target)] = digest
    ref = publish(
        root / "preparation/prepared.json",
        dict(
            root=str(root),
            config=config,
            motor_prepared=motor,
            original_prepared=original,
            records=records,
            historical_records=original["records"],
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
        [str(REPO / "outputs/libero_env311/bin/python"), "scripts/collect_settled_native_training.py", *common],
        [str(REPO / ".venv/bin/python"), "scripts/verify_settled_native_training.py", *common],
    ]
    if dry_run:
        print(json.dumps(dict(commands=commands, cuda_visible_devices="", new_model_calls=0)))
        return
    require(not any((root / name).exists() for name in ("collection", "review", "orchestration")), "refuses retry")
    folder = root / "orchestration"
    folder.mkdir()
    result = dict(
        passed=False,
        complete=False,
        prepared_ref=ref,
        owned_commands=[],
        model_training_authorized=False,
        fresh_performance_authorized=False,
    )
    try:
        for phase, command in zip(("worker", "independent"), commands, strict=True):
            read_prepared(ref)
            result["owned_commands"].append(
                owned_command(
                    command,
                    dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
                    folder / phase,
                    timeout=3600,
                )
            )
        audit_ref = reference(root / "review/verified.json")
        audit = read(audit_ref)
        require(audit["passed"] and audit["complete"], "independent data protocol audit failed")
        read_prepared(ref)
        result.update(
            passed=True,
            complete=True,
            review_ref=audit_ref,
            eligible_for_model_training_registration=audit["eligible_for_model_training_registration"],
        )
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
