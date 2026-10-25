"""Freeze six053 native inputs and code before any new simulator initialization."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from s2a.artifacts import REPO, sha256
from s2a.synchronous_validation import ROSTER, checked, publish, reference, validate_records


def source_closure(entries):
    found, pending = set(), [REPO / p for p in entries]
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            if isinstance(node, ast.Import):
                names += [n.name for n in node.names]
            for name in names:
                for folder in [REPO / "src", REPO / "scripts"]:
                    candidate = folder / (name.replace(".", "/") + ".py")
                    if candidate.is_file() and candidate.resolve() not in found:
                        pending.append(candidate)
    return {str(p): sha256(p) for p in sorted(found)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    config_path = REPO / "configs/wam_synchronous_validation_20261024.json"
    config = json.loads(config_path.read_text())
    if (
        config["roster"] != [list(r) for r in ROSTER]
        or config["seed"] != 42
        or config["hold_steps"] != 80
        or config["horizon"] != 40
        or config["control_fps"] != 20
        or config["automatic_retry"] is not False
    ):
        raise ValueError("Fixed053 config changed")
    root = REPO / "outputs/synchronous_validation_20261024"
    if root.exists():
        raise FileExistsError(root)
    preflight = json.loads(args.preflight.read_text())
    if preflight["passed"] is not True or preflight["training_fresh_identity_disjoint"] is not True:
        raise ValueError("All-six source identity preflight required")
    old = json.loads(checked(preflight["settled_prepared_ref"]).read_text())
    training = json.loads(checked(preflight["prepared_ref"]).read_text())
    forbidden = {a["source_identity"]["initial_state_sha256"] for a in training["training_anchors"]}
    forbidden |= {a["initial_state_sha256"] for a in training["exclusion"]["fresh_sources"]}
    tool_path = REPO / "outputs/clean_training_development_dependency_repair_20261024/inputs/eef_tool_contract.json"
    if sha256(tool_path) != old["original_prepared"]["tool_contract_ref"]["sha256"]:
        raise ValueError("Relocated tool contract differs")
    policy = dict(old["original_prepared"]["policy_config"], tool_contract=str(tool_path), output_dir=str(root))
    examples = {r["task_index"]: r for r in old["records"]}
    root.mkdir()
    (root / "inputs").mkdir()
    records = []
    for index, (task, demo, count) in enumerate(ROSTER):
        source = preflight["sources"][task]
        path = checked(source)
        with h5py.File(path, "r") as archive:
            group = archive["data"][demo]
            state = np.asarray(group["states"][0], np.float64)
            commands = np.asarray(group["actions"], np.float64)
            xml = group.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()
        name = f"task{task}_validation_{demo}"
        expert = root / "inputs" / (name + ".npy")
        with expert.open("xb") as stream:
            np.save(stream, commands, allow_pickle=False)
        identity = hashlib.sha256(state.tobytes()).hexdigest()
        if commands.shape != (count, 7) or identity != preflight["records"][index]["initial_state_sha256"]:
            raise ValueError("Fixed native metadata changed")
        records.append(
            {
                "name": name,
                "task_index": task,
                "task_id": examples[task]["task_id"],
                "task": examples[task]["task"],
                "demo": demo,
                "native_steps": count,
                "initial_state": state.tolist(),
                "initial_state_sha256": identity,
                "model_xml": xml,
                "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                "source_hdf5_ref": source,
                "expert_path": str(expert),
                "expert_sha256": sha256(expert),
                "anchor_frames": [j * (count - 40) // 3 for j in range(4)],
            }
        )
    validate_records(records, forbidden)
    entries = [
        "scripts/prepare_wam_synchronous_validation.py",
        "scripts/collect_wam_synchronous_validation.py",
        "scripts/verify_wam_synchronous_validation.py",
        "tests/test_synchronous_validation.py",
        "plans/v0.0.1/053_synchronous_validation_20261024.md",
        "src/s2a/__init__.py",
        "configs/wam_synchronous_validation_20261024.json",
    ]
    sources = source_closure(entries)
    # Pin installed robot/control code and the external read-only LIBERO implementation.
    runtime_root = REPO / "outputs/libero_env311/lib/python3.11/site-packages"
    for folder in [runtime_root / "robosuite", runtime_root / "mujoco", Path(policy["libero_root"]) / "libero"]:
        for file in sorted(folder.rglob("*.py")):
            sources[str(file.resolve())] = sha256(file)
    inputs = {str(Path(r["path"]).resolve()): r["sha256"] for r in preflight["sources"]}
    for path in [args.preflight, tool_path, Path(policy["libero_config"]), Path(policy["osmesa_library"])]:
        inputs[str(path.resolve())] = sha256(path)
    for ref in [
        preflight["prepared_ref"],
        preflight["settled_prepared_ref"],
        old["cv2_probe_ref"],
        *old["cv2_runtime_source_refs"],
    ]:
        inputs[str(checked(ref).resolve())] = ref["sha256"]
    snapshots = {}
    snapshot_dir = root / "source"
    snapshot_dir.mkdir()
    for i, (name, digest) in enumerate(sorted(sources.items())):
        target = snapshot_dir / f"{i:04d}_{Path(name).name}"
        with target.open("xb") as stream:
            stream.write(Path(name).read_bytes())
        if sha256(target) != digest:
            raise ValueError("Source changed during snapshot")
        snapshots[str(target)] = digest
    value = {
        "schema_version": "s2a.synchronous-validation-preparation.v1",
        "root": str(root),
        "preflight_ref": reference(args.preflight),
        "config_ref": reference(config_path),
        "records": records,
        "forbidden_initial_hashes": sorted(forbidden),
        "policy_config": policy,
        "tool_contract_ref": reference(tool_path),
        "tool_source_from_tcp": old["original_prepared"]["tool_source_from_tcp"],
        "cv2_probe_ref": old["cv2_probe_ref"],
        "cv2_runtime_source_refs": old["cv2_runtime_source_refs"],
        "budget": {
            "environments": 6,
            "physics_steps": 1212,
            "current_canvas_images": 24,
            "future_rgb_frames": 0,
            "model_calls": 0,
            "fresh_cases": 0,
        },
        "source_sha256": sources,
        "snapshot_sha256": snapshots,
        "input_sha256": inputs,
        "execution_authorized": False,
        "fresh_authorized": False,
    }
    print(json.dumps(publish(root / "prepared.json", value)), flush=True)


if __name__ == "__main__":
    main()
