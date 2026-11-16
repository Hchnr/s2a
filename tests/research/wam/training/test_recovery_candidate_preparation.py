"""Build the real candidate windows while enforcing source and no-fit boundaries."""

import copy
import json

import numpy as np
import pytest

from scripts.wam.training.clean import prepare_recovery_candidate_data as run


def test_config_rejects_changed_sources_or_training_scope():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (dict(training_authorized=True), dict(fits=1), dict(fresh_cases=1), dict(horizon=39)):
        with pytest.raises(ValueError, match="fixed candidate"):
            run.validate_config(dict(config, **change))
    changed = copy.deepcopy(config)
    changed["image_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixed candidate"):
        run.validate_config(changed)


def test_real_candidates_bind_images_labels_variants_and_overlaps():
    evidence = run.Evidence()
    _config, _config_ref, _refs, images, dry, h40, union, base, norm = run.candidate_inputs(evidence)
    records, baseline, physical, normalized, flags, multiplicity, metrics = run.build_candidates(
        evidence, images, dry, h40, union, base, norm
    )
    assert len(records) == 108 and len(baseline) == 150
    assert physical.shape == normalized.shape == flags.shape == (108, 40, 10)
    assert multiplicity.shape == (108, 40)
    assert np.issubdtype(physical.dtype, np.float64)
    assert normalized.dtype == np.float32 and flags.dtype == np.bool_
    assert metrics["unique_case_transition_rows"] == 3612
    assert metrics["repeated_case_transition_rows"] == 708
    assert metrics["clipped_scalars"] == 276
    assert metrics["candidate_distinct_seeds"] == 216
    assert metrics["candidate_overlap_with_base_seeds"] == 0
    assert metrics["legacy_distinct_seeds"] == 54
    assert metrics["legacy_overlap_with_base_seeds"] == 16
    assert sum(item["kind"] == "event_pre" for item in records) == 36
    assert len({(r["case_index"], r["anchor_step"]) for r in records}) == 108
    assert len({tuple(run.source_key(r["source_identity"])) for r in records}) == 6
    assert len({r["rgb_sha256"] for r in records}) == 86


def test_recovery_source_mismatch_stops_before_label_loading(monkeypatch):
    evidence = run.Evidence()
    _config, _config_ref, _refs, images, dry, h40, union, base, norm = run.candidate_inputs(evidence)
    h40 = copy.deepcopy(h40)
    h40["cases"][1]["variant"] = "wrong_variant"
    monkeypatch.setattr(run, "load_data", lambda *_args: pytest.fail("future labels opened before identity admission"))
    with pytest.raises(ValueError, match="original training source"):
        run.build_candidates(evidence, images, dry, h40, union, base, norm)


def test_candidate_preparation_requires_source_matched_reaped_64_worker_qa(monkeypatch):
    qa = dict(
        complete=True,
        passed=True,
        all_tests_passed=True,
        source_unchanged=True,
        actual_wait=True,
        reaped=True,
        configured_parallelism=64,
        skipped_count=0,
        collection=dict(collection_errors=[]),
        source_before_ref="before",
    )
    source = {"script.py": "digest"}
    monkeypatch.setattr(run.prefix, "reference", lambda _path: "qa")
    monkeypatch.setattr(run.prefix, "read", lambda ref: qa if ref == "qa" else dict(sources=source))
    monkeypatch.setattr(run, "snapshot", lambda _root: dict(sources=source))
    assert run.require_qa() == ("qa", source)
    qa["reaped"] = False
    with pytest.raises(ValueError, match="source-matched"):
        run.require_qa()
