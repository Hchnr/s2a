"""Bind saved arm qpos to XML order and report proximity to fixed joint limits."""

import json
import xml.etree.ElementTree as ET

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require


def main():
    paired_ref = reference(REPO / "outputs/preview_chunk_diagnostic_verified_20261027/verified.json")
    paired = read(paired_ref)
    require(paired["passed"] is True, "Paired saved-state audit required")
    prepared = read(paired["prepared_ref"])
    cases = []
    for i, case in enumerate(prepared["cases"]):
        manifest = read(case["manifest_ref"])
        xml = ET.fromstring(manifest["record"]["model_xml"])
        joints = [n for n in xml.find("worldbody").iter() if n.tag in ("joint", "freejoint")][:7]
        require([n.get("name") for n in joints] == [f"robot0_joint{k}" for k in range(1, 8)], "Arm qpos order")
        require(all(n.tag == "joint" and n.get("type", "hinge") == "hinge" for n in joints), "Scalar hinges required")
        limits = np.asarray([[float(x) for x in n.get("range").split()] for n in joints])
        rows_ref = reference(REPO / f"outputs/preview_chunk_branches_20261027/case{i:02d}_h40/steps.json")
        qpos = np.asarray([r["sim_state"][1:8] for r in read(rows_ref)])
        margin = np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos)
        cases.append(
            dict(
                case=case["case"],
                manifest_ref=case["manifest_ref"],
                rows_ref=rows_ref,
                joint_limits_rad=limits.tolist(),
                minimum_margin_rad_by_joint=margin.min(axis=0).tolist(),
                within_or_beyond_1mrad_limit_steps_by_joint=(margin <= 0.001).sum(axis=0).tolist(),
                outside_xml_range_steps_by_joint=(margin < 0).sum(axis=0).tolist(),
            )
        )
    print(
        json.dumps(
            publish(
                REPO / "outputs/preview_chunk_joint_limits_20261027/verified.json",
                dict(
                    complete=True,
                    passed=True,
                    paired_ref=paired_ref,
                    source_ref=reference(__file__),
                    cases=cases,
                    physics_steps=0,
                    model_calls=0,
                    fresh_cases=0,
                    scope="saved qpos versus XML limits; does not prove active joint-limit force or reference unreachability",
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
