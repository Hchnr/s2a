"""Independent saved endpoint/label witness; not full control qualification."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_wam_release_horizon import reference


def check_labels(data, tool):
    source = data["source_pose"]
    count = len(source)
    tcp = source @ tool
    np.testing.assert_allclose(data["tcp_pose"], tcp, atol=1e-12, rtol=0)
    delta = np.repeat(np.eye(4)[None], count - 1, axis=0)
    rotation = tcp[:-1, :3, :3].transpose(0, 2, 1)
    delta[:, :3, :3] = rotation @ tcp[1:, :3, :3]
    delta[:, :3, 3] = (rotation @ (tcp[1:, :3, 3] - tcp[:-1, :3, 3])[..., None])[..., 0]
    np.testing.assert_allclose(data["delta_pose"], delta, atol=1e-12, rtol=0)
    state = np.zeros((count, 128))
    state[:, :3] = tcp[:, :3, 3]
    state[:, 3:6] = tcp[:, :3, 0]
    state[:, 6:9] = tcp[:, :3, 1]
    state[:, 9] = data["opening"]
    action = np.zeros_like(state)
    action[:-1, :3] = delta[:, :3, 3]
    action[:-1, 3:6] = delta[:, :3, 0]
    action[:-1, 6:9] = delta[:, :3, 1]
    action[:-1, 9] = data["opening"][1:]
    np.testing.assert_allclose(data["state128"], state, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data["action128"], action, atol=1e-12, rtol=0)
    mask = np.zeros_like(state, dtype=bool)
    mask[:, :10] = True
    np.testing.assert_array_equal(data["state_mask"], mask)
    mask[-1] = False
    np.testing.assert_array_equal(data["action_mask"], mask)
    valid = np.arange(count) < count - 1
    np.testing.assert_array_equal(data["action_valid"], valid)
    np.testing.assert_allclose(np.diff(data["sim_time"]), 0.05, atol=1e-9, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, required=True, choices=range(9))
    args = parser.parse_args()
    root = Path(
        "outputs/training_state_recovery_qualification_20261025"
        if args.case_index == 0
        else "outputs/training_state_recovery_continuation_20261025"
    )
    case = root / f"case{args.case_index:02d}"
    config_path = case / "execution_config.json"
    config = json.loads(config_path.read_text())
    template_path = Path(config["policy_template"])
    template = json.loads(template_path.read_text())
    tool_path = Path(template["tool_contract"])
    tool = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"])
    paths = [case / "actual_trajectory.npz", case / "endpoints.jsonl", config_path, template_path, tool_path]
    before = [reference(p) for p in paths]
    receipts = [json.loads(line) for line in paths[1].read_text().splitlines()]
    with np.load(paths[0], allow_pickle=False) as data:
        if len(receipts) != config["online_max_steps"] + 1:
            raise ValueError("Incomplete endpoints")
        check_labels(data, tool)
        for index, receipt in enumerate(receipts):
            actual_sha = hashlib.sha256(data["sim_states"][index].tobytes()).hexdigest()
            if actual_sha != receipt["sim_state_sha256"] or receipt["endpoint_step"] != index:
                raise ValueError("Actual physical endpoint binding changed")
    if before != [reference(p) for p in paths]:
        raise RuntimeError("Inputs changed")
    result = dict(
        passed=True,
        scope="Endpoint label/time/hash witness only; full mechanical/image/exit qualification not established",
        full_qualification=False,
        input_refs=before,
        script_ref=reference(Path(__file__)),
        endpoint_count=len(receipts),
        new_model_calls=0,
        new_physics_steps=0,
        fresh_cases=0,
    )
    with (case / "labels_independently_verified.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(passed=True, endpoints=len(receipts), full_qualification=False)))


if __name__ == "__main__":
    main()
