"""Only the disclosed alpha1 JSON metadata pair is an allowed exception."""

import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def supplement(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "temporal_metadata_test", root / "scripts/audit_wam_temporal_qualification_alpha_metadata.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunks():
    old = [
        dict(step=i * 8, model_rotation_alpha=1, generation_s=0.1, raw128=[[0.0, 0.5]], anchor_tcp=[1.0, 0.0])
        for i in range(75)
    ]
    new = copy.deepcopy(old)
    for item in new:
        item["model_rotation_alpha"] = 1.0
        item["generation_s"] = 0.2
    return new, old


def test_exact_registered_metadata_pair_disclosed_without_rewriting(supplement):
    new, old = chunks()
    saved = copy.deepcopy((new, old))
    differences = supplement.chunk_type_supplement(new, old)
    assert len(differences) == 75 and all(
        item["current_json_type"] == "float" and item["previous_json_type"] == "int" for item in differences
    )
    assert (new, old) == saved
    assert type(new[0]["model_rotation_alpha"]) is float and type(old[0]["model_rotation_alpha"]) is int


@pytest.mark.parametrize("value", [True, 1, 0.9999999999999999, 1.0000000000000002, 0.0])
def test_alpha_bool_other_type_or_other_value_rejected(supplement, value):
    new, old = chunks()
    new[0]["model_rotation_alpha"] = value
    with pytest.raises(AssertionError):
        supplement.chunk_type_supplement(new, old)


@pytest.mark.parametrize("value", [-0.0, 0, 1e-16])
def test_any_other_signed_zero_type_or_tiny_numeric_change_rejected(supplement, value):
    new, old = chunks()
    new[74]["raw128"][0][0] = value
    with pytest.raises(AssertionError):
        supplement.chunk_type_supplement(new, old)


def test_other_key_or_donor_count_changes_rejected(supplement):
    new, old = chunks()
    new[1]["extra"] = "not registered"
    with pytest.raises(AssertionError):
        supplement.chunk_type_supplement(new, old)
    new, old = chunks()
    with pytest.raises(AssertionError):
        supplement.chunk_type_supplement(new[:-1], old)
