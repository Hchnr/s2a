"""The original failed task stays False; its exact case audit is mandatory."""

import copy
import importlib
import json

import pytest

from s2a.artifacts import REPO


@pytest.fixture
def actual(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    module = importlib.import_module("verify_settled_native_cached_prefix")
    config = json.loads((REPO / "configs/settled_native_training_20261022.json").read_text())
    return module, module.read(config["native_collection_ref"])


def test_actual_preserved_failed_task_and_exact_independent_ten_case_audit(actual):
    module, collection = actual
    failed = module.read(collection["cached_task_ref"])
    before = copy.deepcopy(failed)
    cases = module.historical_cases(collection)
    assert len(cases) == 30
    assert failed == before and failed["passed"] is failed["complete"] is False


@pytest.mark.parametrize(
    "change", ["prefix_false", "wrong_case", "wrong_prepared", "fake_task_success", "missing_case"]
)
def test_unaudited_or_relabelled_cache_rejected(actual, monkeypatch, change):
    module, collection = actual
    original_read = module.read

    def read(ref):
        value = copy.deepcopy(original_read(ref))
        if ref == collection["prefix_verification_ref"]:
            if change == "prefix_false":
                value["prefix_verified"] = False
            elif change == "wrong_case":
                value["cached_case_refs"][0]["sha256"] = "0" * 64
            elif change == "wrong_prepared":
                value["prepared"]["sha256"] = "0" * 64
        if ref == collection["cached_task_ref"]:
            if change == "fake_task_success":
                value["passed"] = value["complete"] = True
            elif change == "missing_case":
                value["cases"].pop()
        return value

    monkeypatch.setattr(module, "read", read)
    with pytest.raises(ValueError):
        module.historical_cases(collection)
