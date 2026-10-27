"""103 exact existing30-trajectory roster and read-only preparation."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def ref(path):
    return dict(path=str(path.resolve()), sha256=sha256(path))


def main():
    e = Evidence()
    e.read(__file__)
    path = Path("outputs/native_arm_response_data_20261025/reconstruction_result.json")
    data = e.json(path)
    closure_path = path.parent / "closure_verified.json"
    closed = e.json(closure_path)
    sensor_path = Path("outputs/current_wrench_sensor_20261025/closure_verified.json")
    sensor = e.json(sensor_path)
    assert closed["complete"] and closed["data_qualified"] and closed["owner_normal_wait_verified"]
    assert sensor["complete"] and sensor["current_wrench_contract_verified"] and sensor["owner_normal_wait_verified"]
    profiles = []
    for entry in data["trajectories"]:
        meta = Path(entry["source"]["path"]).parent / "result.json"
        e.check(meta, closed["source_sha256"][str(meta.resolve())])
        original = e.json(meta)
        assert original["complete"] and original["passed"] and original["native_quality_passed"]
        e.check(entry["source"]["path"], entry["source"]["sha256"])
        e.check(entry["cache_ref"]["path"], entry["cache_ref"]["sha256"])
        prepared = original["prepared_ref"]
        e.check(prepared["path"], prepared["sha256"])
        with np.load(entry["source"]["path"], allow_pickle=False) as arrays:
            steps = len(arrays["sent_commands"])
        with np.load(entry["cache_ref"]["path"], allow_pickle=False) as cache:
            assert len(cache["step"]) == steps
        profiles.append(
            dict(
                index=entry["index"],
                split=entry["split"],
                task=entry["task"],
                record=original["record"],
                case_ref=ref(meta),
                trajectory_ref=entry["source"],
                cache_ref=entry["cache_ref"],
                physics_steps=steps,
            )
        )
    assert len(profiles) == 30 and [p["index"] for p in profiles] == list(range(30))
    assert sum(p["split"] == "fit" for p in profiles) == 24
    assert sum(p["split"] == "holdout" for p in profiles) == 6
    assert sum(p["physics_steps"] for p in profiles) == 6180
    e.recheck()
    write_once(
        "configs/native_wrench_data_20261025.json",
        dict(
            seed=42,
            profiles=profiles,
            physics_steps=6180,
            endpoints=6210,
            native_data_ref=ref(path),
            native_closure_ref=ref(closure_path),
            sensor_closure_ref=ref(sensor_path),
            source_sha256=e.hashes,
            model_calls=0,
            new_fits=0,
            fresh_cases=0,
        ),
    )
    print("103 registered30/6180periods/6210endpoints, physics0")


if __name__ == "__main__":
    main()
