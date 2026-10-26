"""143 earliest unused left-table anchor per eligible original fit profile."""

from pathlib import Path

from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/143_left_table_training_selection_20261025.md")
    path = Path("outputs/native_training_table_coverage_20261025/result.json")
    source = e.json(path)
    assert source["complete"] and source["training_only"]
    for p, signature in source["source_sha256"].items():
        e.check(p, signature)
    registries = [
        e.json(f"configs/{name}_20261025.json")
        for name in ("training_response_coverage", "wrench_coverage_interventions", "condition_span_interventions")
    ]
    original = registries[0]
    used = {}
    for config in registries:
        for i, profile in enumerate(config["profiles"]):
            used.setdefault(profile["native_data_index"], set()).add(config["trials"][7 * i]["anchor_cycle"])
    selected = []
    for row in source["trajectories"]:
        index = row["index"]
        profiles = [p for p in original["profiles"] if p["native_data_index"] == index]
        if not profiles:
            continue
        assert index not in (0, 10, 20) and len(profiles) == 1
        candidates = [
            k
            for k in row["table_contact_cycles"]["left"]
            if k > 0 and k not in row["protocol_override_cycles"] and k not in used[index]
        ]
        if candidates:
            selected.append(
                dict(
                    native_index=index,
                    profile=profiles[0],
                    anchor_cycle=min(candidates),
                    candidate_cycles=candidates,
                    excluded_used_cycles=sorted(used[index]),
                    branches=7,
                    physics_steps=7 * (min(candidates) + 1),
                )
            )
    assert len(selected) == 3
    e.recheck()
    output = Path("outputs/left_table_training_selection_20261025")
    output.mkdir(exist_ok=False)
    write_once(
        output / "result.json",
        dict(
            complete=True,
            selection_only=True,
            geometry_source_ref=dict(path=str(path.resolve()), sha256=sha256(path)),
            selected=selected,
            planned_trials=21,
            planned_physics_steps=sum(r["physics_steps"] for r in selected),
            source_sha256=e.hashes,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print([(r["native_index"], r["anchor_cycle"]) for r in selected], sum(r["physics_steps"] for r in selected))


if __name__ == "__main__":
    main()
