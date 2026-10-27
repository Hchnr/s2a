"""Metadata must detect sampler/math/gate drift and cannot prepare an absent head."""

from pathlib import Path

import pytest

from s2a.artifacts import REPO


@pytest.mark.parametrize("fault", ["sampler", "numeric", "threshold"])
def test_complete_body_proof_rejects_unregistered_changes(tmp_path, monkeypatch, fault):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import verify_wam_aggregate_state_metadata as metadata

    assert metadata.body_proofs()["passed"] is True
    paths = [
        "scripts/sample_wam_synchronous_state.py",
        "scripts/sample_wam_aggregate_state.py",
        "scripts/verify_wam_synchronous_state.py",
        "scripts/verify_wam_aggregate_state.py",
        "src/s2a/synchronous_state_metrics.py",
        "src/s2a/aggregate_state_metrics.py",
    ]
    target, before, after = {
        "sampler": (
            "scripts/sample_wam_aggregate_state.py",
            'counter.scope(entry["name"], 35)',
            'counter.scope(entry["name"], 34)',
        ),
        "numeric": ("scripts/verify_wam_aggregate_state.py", "len(pairs) == 48", "len(pairs) == 47"),
        "threshold": ("src/s2a/aggregate_state_metrics.py", "0.90 *", "0.95 *"),
    }[fault]
    for relative in paths:
        source = (REPO / relative).read_text()
        if relative == target:
            assert before in source
            source = source.replace(before, after)
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setattr(metadata, "REPO", tmp_path)
    with pytest.raises(ValueError):
        metadata.body_proofs()


def test_actual_preparer_cannot_create_validation_before_aggregate_head(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import numpy as np
    import prepare_wam_aggregate_state as prepare

    root = tmp_path / "aggregate_validation"
    monkeypatch.setattr(prepare, "ROOT", root)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("preparation decoded future arrays"))
    with pytest.raises(FileNotFoundError) as error:
        prepare.prepare()
    assert Path(error.value.filename) == root.parent / "aggregate_fit/orchestration/result.json"
    assert not root.exists()
