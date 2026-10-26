"""131 sensor subtree vs known current contact bodies; no dynamics execution."""

import xml.etree.ElementTree as ET
from pathlib import Path

from audit_wam_position import Evidence
from collect_wam_native_training import write_once


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/131_wrench_contact_topology_20261025.md")
    source = e.json("outputs/anchor_contact_diagnosis_20261025/result.json")
    assert source["complete"] and source["contact_forces_measured"] is False
    closed = e.json("outputs/paired_arm_response_20261025/closure_verified.json")
    assert closed["complete"] and closed["data_qualified"]
    groups = []
    for g in source["groups"]:
        folder = Path("outputs/paired_arm_response_20261025") / f"case{7 * g['anchor_group']:02d}"
        path = folder / "collection_result.json"
        e.check(path, closed["source_sha256"][str(path.resolve())])
        record = e.json(path)
        xml = folder / "model.xml"
        e.check(xml, record["source_sha256"][str(xml.resolve())])
        e.check(xml, source["source_sha256"][str(xml.resolve())])
        root = ET.fromstring(xml.read_bytes())
        parents = {child: parent for parent in root.iter() for child in parent}
        sensor = root.find("sensor")
        nodes = [sensor.find(f"{kind}[@name='gripper0_{kind}_ee']") for kind in ("force", "torque")]
        assert all(n is not None for n in nodes)
        assert nodes[0].get("site") == nodes[1].get("site")
        site = next(n for n in root.iter("site") if n.get("name") == nodes[0].get("site"))
        body = parents[site]
        assert body.tag == "body"
        subtree = {b.get("name") for b in body.iter("body")}
        contacts = []
        for contact in g["contacts"]:
            if not contact["robot_contact"]:
                continue
            robot = [name for name in contact["bodies"] if name.startswith(("robot0_", "gripper0_"))]
            assert robot
            contacts.append(
                dict(
                    bodies=contact["bodies"],
                    robot_bodies=robot,
                    all_robot_bodies_in_sensor_subtree=all(name in subtree for name in robot),
                )
            )
        groups.append(
            dict(
                group=g["anchor_group"],
                sensor_body=body.get("name"),
                parent_body=parents[body].get("name"),
                contacts=contacts,
            )
        )
    e.recheck()
    output = Path("outputs/wrench_contact_topology_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            groups=groups,
            source_sha256=e.hashes,
            diagnostic_only=True,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print(groups)


if __name__ == "__main__":
    main()
