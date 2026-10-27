"""055 real entry dispatch and strict suffix/pair/identity boundaries on CPU."""

import copy

import pytest

from s2a.artifacts import REPO
from s2a.synchronous_state_continuation import dispatch_matrix, prefix_raw_records, validate_parameter_scope


def entries():
    return [
        {"index": i, "variant": v}
        for i, v in enumerate(["zero"] * 3 + [v for _ in range(48) for v in ["raw", "probe052"]])
    ]


def test_actual_worker_expression_handles_full_matrix_and_suffix():
    from verify_wam_synchronous_state_continuation_metadata import exercise_worker_dispatch

    worker = REPO / "scripts/sample_wam_synchronous_state_continuation.py"
    assert exercise_worker_dispatch(worker, entries()) == {"requests": 99, "probe": 48, "raw": 48, "zero": 3}
    assert exercise_worker_dispatch(worker, entries()[4:]) == {"requests": 95, "probe": 48, "raw": 47, "zero": 0}


def test_frozen_failed_worker_expression_reproduces_original_boundary():
    from verify_wam_synchronous_state_continuation_metadata import exercise_worker_dispatch

    with pytest.raises(KeyError, match="probe052"):
        exercise_worker_dispatch(REPO / "scripts/sample_wam_synchronous_state.py", entries())


def test_completed_prefix_is_not_dispatched_again():
    zero, head = object(), object()
    suffix = entries()[4:]
    chosen = dispatch_matrix(suffix, zero=zero, probe=head)
    assert len(chosen) == 95 and suffix[0]["index"] == 4 and suffix[-1]["index"] == 98
    assert all(value is not zero for value in chosen)


def test_both_complete_body_proofs():
    from verify_wam_synchronous_state_continuation_metadata import body_proofs

    assert all(v["passed"] for v in body_proofs().values())


def test_prefix_raw_pair_keeps_original_actual_index_and_reference(monkeypatch):
    import s2a.synchronous_state_continuation as module

    raw = {"index": 3, "variant": "raw", "complete": True, "pair_id": "anchor_s42", "initial_ref": "original-initial"}
    failed = {"generations": [{}, {}, {}, raw]}
    monkeypatch.setattr(module, "prefix_evidence", lambda p: ({}, failed))
    pairs = prefix_raw_records({})
    assert pairs == {"anchor_s42": raw} and pairs["anchor_s42"]["initial_ref"] == "original-initial"
    failed["generations"][3]["index"] = 4
    with pytest.raises(ValueError):
        prefix_raw_records({})


def test_process_local_ids_can_differ_across_reinitialization():
    current = {"count": 17, "metadata_sha256": "new-address-and-versions"}
    prefix = {"count": 17, "metadata_sha256": "old-address-and-versions"}
    validate_parameter_scope(current, copy.deepcopy(current), prefix)
    with pytest.raises(ValueError):
        validate_parameter_scope(current, {"count": 17, "metadata_sha256": "changed"}, prefix)
    with pytest.raises(ValueError):
        validate_parameter_scope(current, current, {"count": 18, "metadata_sha256": "old"})


def test_first_cross_process_pair_checks_every_fp32_initial_array(tmp_path):
    import json

    import numpy as np
    from collect_wam_clean_training import same_initial

    from s2a.clean_training_runtime import reference

    refs = []
    for name in ["raw", "probe"]:
        folder = tmp_path / name
        folder.mkdir()
        array = folder / "initial.npz"
        np.savez(
            array,
            initial_noise=np.ones((2, 3), np.float32),
            condition_mask=np.zeros((2, 1), np.float32),
            first_condition_latent=np.ones((1, 2), np.float32),
        )
        meta = {
            "sampling_seed": 42,
            "caption": "same current task",
            "text_tokens": [1, 2],
            "inference_only_current_image": True,
            "arrays": reference(array),
        }
        p = folder / "initial.json"
        p.write_text(json.dumps(meta))
        refs.append(reference(p))
    same_initial(*refs)
    array = tmp_path / "probe/initial.npz"
    np.savez(
        array,
        initial_noise=np.zeros((2, 3), np.float32),
        condition_mask=np.zeros((2, 1), np.float32),
        first_condition_latent=np.ones((1, 2), np.float32),
    )
    meta = json.loads((tmp_path / "probe/initial.json").read_text())
    meta["arrays"] = reference(array)
    (tmp_path / "probe/initial.json").write_text(json.dumps(meta))
    refs[1] = reference(tmp_path / "probe/initial.json")
    with pytest.raises(ValueError):
        same_initial(*refs)


@pytest.mark.parametrize(
    "script",
    [
        "sample_wam_synchronous_state_continuation.py",
        "run_wam_synchronous_state_continuation.py",
        "verify_wam_synchronous_state_continuation.py",
    ],
)
def test_actual_entry_imports_and_cli_parse_before_any_model(script):
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--help"],
        cwd=REPO,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            LD_LIBRARY_PATH="",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(REPO / "src") + ":" + str(REPO / "scripts"),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
