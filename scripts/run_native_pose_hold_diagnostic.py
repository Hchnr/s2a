"""Freeze and own the finite native terminal pose-hold comparison."""

import argparse
import json
import os
import shutil
from pathlib import Path

from run_wam_readout import owned_command

from s2a.artifacts import REPO, sha256
from s2a.native_environment import strict_native_environment

CONFIG = REPO / "configs/native_terminal_pose_hold_20261022.json"
SOURCES = (
    "plans/v0.0.1/031_native_terminal_pose_hold_diagnostic_20261022.md",
    "configs/native_terminal_pose_hold_20261022.json",
    "scripts/run_native_pose_hold_diagnostic.py",
    "scripts/collect_native_pose_hold_diagnostic.py",
    "scripts/verify_native_pose_hold_diagnostic.py",
    "tests/test_native_pose_hold_diagnostic.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def read(ref):
    require(reference(ref["path"]) == ref, "input SHA changed")
    return json.loads(Path(ref["path"]).read_text())


def publish(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    return reference(path)


def prepare():
    strict_native_environment()
    config = json.loads(CONFIG.read_text())
    require(
        config["seed"] == 42
        and config["methods"] == ["baseline", "latched_pose"]
        and config["hold_steps"] == 40
        and config["physics_steps"] == 1648
        and config["environment_initializations"] == 8
        and config["current_canvases"] == 40
        and not any(config[k] for k in ("training_authorized", "fresh_authorized", "automatic_retry")),
        "fixed diagnostic contract changed",
    )
    prior = read(config["static_prepared_ref"])
    for key in ("static_collection_ref", "static_verified_ref", "static_owner_ref"):
        document = read(config[key])
        require(document["complete"] and document["passed"], "static diagnostic incomplete")
    for key in ("source_sha256", "snapshot_sha256"):
        for path, digest in prior[key].items():
            require(sha256(path) == digest, "static source/snapshot changed")
    require([r["name"] for r in prior["cases"]] == config["cases"], "fixed four-case roster changed")
    records = [item["record"] for item in prior["cases"]]
    require(2 * sum(r["native_steps"] + 40 for r in records) == 1648, "physics budget changed")
    inputs = {ref["path"]: ref["sha256"] for k, ref in config.items() if k.endswith("_ref")}
    for record in records:
        require(sha256(record["expert_path"]) == record["expert_sha256"], "native commands changed")
        inputs[record["expert_path"]] = record["expert_sha256"]
    osc = REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/controllers/osc.py"
    inputs[str(osc)] = sha256(osc)
    sources = {str(REPO / name): sha256(REPO / name) for name in SOURCES}
    root = REPO / config["output_dir"]
    require(root == REPO / "outputs/wam_stability_20261021/native_terminal_pose_hold_diagnostic", "root changed")
    (root / "preparation/snapshots").mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for index, (path, digest) in enumerate(sources.items()):
        target = root / "preparation/snapshots" / f"{index:02d}_{Path(path).name}"
        shutil.copyfile(path, target)
        require(sha256(target) == digest, "new source snapshot differs")
        snapshots[str(target)] = digest
    prepared = dict(
        config=config,
        root=str(root),
        static_prepared=prior,
        records=records,
        input_sha256=inputs,
        source_sha256=sources,
        snapshot_sha256=snapshots,
    )
    ref = publish(root / "preparation/prepared.json", prepared)
    read_prepared(ref)
    print(json.dumps(ref))


def read_prepared(ref):
    strict_native_environment()
    prepared = read(ref)
    for obj in (prepared, prepared["static_prepared"]):
        for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
            for path, digest in obj[key].items():
                require(sha256(path) == digest, "frozen input/source changed")
    require(json.loads(CONFIG.read_text()) == prepared["config"], "prepared configuration changed")
    return prepared


def execute(ref, dry_run):
    prepared = read_prepared(ref)
    root = Path(prepared["root"])
    common = ["--prepared", ref["path"], "--prepared-sha256", ref["sha256"], "--seed", "42"]
    commands = [
        [str(REPO / "outputs/libero_env311/bin/python"), "scripts/collect_native_pose_hold_diagnostic.py", *common],
        [str(REPO / ".venv/bin/python"), "scripts/verify_native_pose_hold_diagnostic.py", *common],
    ]
    if dry_run:
        print(json.dumps(dict(commands=commands, gpu="", new_model_calls=0)))
        return
    require(not any((root / p).exists() for p in ("collection", "review", "orchestration")), "refuses retry")
    folder = root / "orchestration"
    folder.mkdir()
    result = dict(passed=False, complete=False, prepared_ref=ref, owned_commands=[], training_authorized=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    try:
        for phase, command in zip(("worker", "independent"), commands, strict=True):
            read_prepared(ref)
            result["owned_commands"].append(owned_command(command, env, folder / phase, timeout=1800))
        audit = json.loads((root / "review/verified.json").read_text())
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
