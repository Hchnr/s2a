"""Bind proposal-record inputs to the completed physical audit's exact input hashes."""

from pathlib import Path

from audit_wam_position import Evidence

from s2a.durable_stage_supervisor import publish


def main():
    e = Evidence()
    e.read(__file__)
    proposal = e.json("outputs/coupled_pose_preview_records_bounded_20261026/verified.json")
    assert proposal["complete"] and len(proposal["cases"]) == 18
    assert sum(c["periods"] for c in proposal["cases"]) == 4907
    root = Path("outputs/current_coupled_control_20261026")
    bindings = []
    for case_index in range(18):
        folder = root / f"case{case_index:02d}"
        physical_path = folder / "independent_verified.json"
        physical = e.json(physical_path)
        assert physical["complete"] and physical["mechanical_labels_images_passed"]
        assert proposal["source_sha256"][str(physical_path.resolve())] == e.hashes[str(physical_path.resolve())]
        case = next(folder.glob("simulation/*/*"))
        manifest = e.json(case / "manifest.json")
        files = [case / name for name in ("manifest.json", "controller.json", "chunks.json", "steps.jsonl")]
        files.append(Path(manifest["runtime_config"]["tool_contract"]))
        for path in files:
            canonical = str(path.resolve())
            expected = physical["input_sha256"][canonical]
            assert proposal["source_sha256"][canonical] == expected
            e.check(path, expected)
            bindings.append(dict(case=case_index, path=canonical, sha256=expected))
    e.recheck()
    publish(
        "outputs/coupled_pose_preview_records_bounded_20261026/input_binding_verified.json",
        dict(
            complete=True,
            bindings=bindings,
            source_sha256=e.hashes,
            scope="exact record inputs match prior independent physical audit; no new control result",
        ),
    )
    print("18 cases, 4907 proposals, 90 physical-input bindings verified", flush=True)


if __name__ == "__main__":
    main()
