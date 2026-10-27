import json

import numpy as np
import pytest
from collect_wam_native_training import write_once


def test_actual_controller_array_metadata_round_trips_before_file_publication(tmp_path):
    path = tmp_path / "metadata.json"
    write_once(
        path,
        dict(controller_contract=dict(input_min=np.full(6, -1.0), kp=np.arange(6, dtype=float)), steps=np.int64(102)),
    )
    loaded = json.loads(path.read_text())
    assert loaded["controller_contract"]["input_min"] == [-1.0] * 6 and loaded["steps"] == 102
    with pytest.raises(FileExistsError):
        write_once(path, dict(steps=0))
    assert json.loads(path.read_text()) == loaded


def test_nonserializable_or_nonfinite_metadata_never_creates_partial_file(tmp_path):
    for name, value in [("bad", object()), ("nan", np.array([np.nan]))]:
        path = tmp_path / f"{name}.json"
        with pytest.raises((TypeError, ValueError)):
            write_once(path, dict(payload=value))
        assert not path.exists()


def test_entire_collector_only_changes_namespace_and_final_metadata_publication():
    import ast
    from pathlib import Path

    old = Path("scripts/collect_native_gripper_response_v2.py").read_text()
    new = Path("scripts/collect_native_gripper_response_v3.py").read_text()
    new = new.replace(
        "native_gripper_response_interventions_20261025_v3", "native_gripper_response_interventions_20261025_v2"
    )
    new = new.replace("endpoint, save_arrays, write_once", "endpoint, save_arrays")
    assert new.count('write_once(folder / "collection_result.json", output)') == 1
    new = new.replace(
        '        write_once(folder / "collection_result.json", output)',
        '        with (folder / "collection_result.json").open("x") as stream:\n            json.dump(output, stream, indent=2)\n            stream.write("\\n")',
    )
    assert ast.dump(ast.parse(new), include_attributes=False) == ast.dump(ast.parse(old), include_attributes=False)
