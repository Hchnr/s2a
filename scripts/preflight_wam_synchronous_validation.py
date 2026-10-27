"""Check fixed six validation demo identities without reading RGB or rolling physics."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from analyze_wam_release_horizon import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root or not args.output.resolve().is_relative_to(root / "outputs"):
        raise ValueError("Requires repository cwd and outputs destination")
    if any(os.environ.get(k) != "" for k in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]):
        raise ValueError("Explicit empty CPU CUDA/LD required")
    prepared_path = (
        root
        / "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json"
    )
    prepared = json.loads(prepared_path.read_text())
    settled_path = root / "outputs/wam_stability_20261021/settled_native_training/preparation/prepared.json"
    settled = json.loads(settled_path.read_text())
    refs = settled["original_prepared"]["inherited_hdf5_refs_stat_only"]
    assets = root / "outputs/wam_checkpoint_acceptance_20261019v2/assets"
    files = [
        Path(refs[0]["path"]),
        assets / "put_the_wine_bottle_on_the_rack_demo.hdf5",
        assets / "put_the_cream_cheese_in_the_bowl_demo.hdf5",
    ]
    expected = [refs[0]["sha256"], refs[2]["sha256"], refs[1]["sha256"]]
    roster = [["demo_16", "demo_32"], ["demo_4", "demo_27"], ["demo_14", "demo_24"]]
    forbidden = {a["source_identity"]["initial_state_sha256"] for a in prepared["training_anchors"]}
    forbidden |= {a["initial_state_sha256"] for a in prepared["exclusion"]["fresh_sources"]}
    rows, sources = [], []
    for task, (path, sha, demos) in enumerate(zip(files, expected, roster, strict=True)):
        ref = reference(path)
        if ref["sha256"] != sha:
            raise ValueError("Native source SHA changed")
        sources.append(ref)
        with h5py.File(path, "r") as archive:
            for demo in demos:
                group = archive["data"][demo]
                state = np.asarray(group["states"][0], dtype=np.float64)
                identity = hashlib.sha256(state.tobytes()).hexdigest()
                if identity in forbidden:
                    raise ValueError("Validation initial state aliases training, fresh or another validation case")
                forbidden.add(identity)
                count = group["actions"].shape[0]
                if group["actions"].shape != (count, 7) or not 43 <= count <= 600:
                    raise ValueError("Selected validation action shape changed; replacement forbidden")
                rows.append(
                    {
                        "task_index": task,
                        "demo": demo,
                        "initial_state_sha256": identity,
                        "initial_state_length": len(state),
                        "native_steps": count,
                        "uniform_complete_anchor_frames": [j * (count - 40) // 3 for j in range(4)],
                        "model_xml_present": "model_file" in group.attrs,
                    }
                )
    if not all(row["model_xml_present"] for row in rows):
        raise ValueError("Missing validation model XML")
    for ref in sources:
        if reference(Path(ref["path"])) != ref:
            raise ValueError("Native source changed during metadata inspection")
    result = {
        "complete": True,
        "passed": True,
        "scope": "Native source metadata and initial identity only; no RGB, rollout, future label or fresh performance read",
        "script_ref": reference(Path(__file__)),
        "helper_ref": reference(Path(__file__).with_name("analyze_wam_release_horizon.py")),
        "prepared_ref": reference(prepared_path),
        "settled_prepared_ref": reference(settled_path),
        "sources": sources,
        "records": rows,
        "training_fresh_identity_disjoint": True,
        "physics_steps": 0,
        "model_calls": 0,
        "fresh_performance_reads": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
