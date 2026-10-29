"""Metadata repair regressions only; no dynamics, experiment, or source rewrite."""

import importlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def repair(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("check_tracking_v5_saved_direction_backtracking_v2")
    # Restore the frozen module after each synthetic test, including nested calls.
    monkeypatch.setattr(module.frozen, "compare_contract", module.compare_contract)
    return module


def archived_endpoint():
    """Exact prefix entry0 metadata from producer SHA0957f819...e26028e."""
    return dict(
        source_trace_index=657,
        fields_checked=["commands", "tcp_errors", "safety_min", "prefix_length", "joint_feasible", "feasible"],
        commands_exact_bytes=True,
        tcp_errors_max_abs_difference=0.0,
        safety_min_abs_difference=0.0,
        strict_labels_equal=True,
        old_full_rollout_available=False,
    )


def test_real_endpoint_metadata_reaches_strings_without_numeric_coercion(repair):
    endpoint = archived_endpoint()
    compared = []

    def numeric(label, actual, saved):
        assert np.asarray(actual).dtype.kind not in "US"
        compared.append(label)
        assert actual == saved

    repair.compare_contract(
        {"entries": [{"endpoint_match": endpoint}]},
        {"entries": [{"endpoint_match": deepcopy(endpoint)}]},
        compare=numeric,
    )
    assert compared == [
        "contract.entries[0].endpoint_match.tcp_errors_max_abs_difference",
        "contract.entries[0].endpoint_match.safety_min_abs_difference",
    ]


@pytest.mark.parametrize("change", ["value", "order", "leaf_type", "container", "length", "nested"])
def test_endpoint_string_sequence_tampering_is_rejected(repair, change):
    actual, saved = archived_endpoint(), archived_endpoint()
    fields = saved["fields_checked"]
    if change == "value":
        fields[0] = "command"
    elif change == "order":
        fields[0], fields[1] = fields[1], fields[0]
    elif change == "leaf_type":
        fields[0] = b"commands"
    elif change == "container":
        saved["fields_checked"] = tuple(fields)
    elif change == "length":
        fields.pop()
    else:
        saved["fields_checked"] = [fields]
    with pytest.raises(ValueError):
        repair.compare_contract(actual, saved)


def test_string_arrays_preserve_shape_kind_and_order_without_width_requirement(repair):
    values = np.array([["alpha", "beta"]], dtype="U5")
    repair.compare_contract(values, values.astype("U20"))
    for invalid in (values.ravel(), values[:, ::-1], values.astype("S5")):
        with pytest.raises(ValueError):
            repair.compare_contract(values, invalid)
    repair.compare_contract([b"alpha", b"beta"], [b"alpha", b"beta"])
    with pytest.raises(ValueError):
        repair.compare_contract(["1", "beta"], [1, "beta"])
    with pytest.raises(ValueError):
        repair.compare_contract(["alpha", "beta"], [np.str_("alpha"), "beta"])


def test_original_numeric_tolerance_and_strict_labels_are_unchanged(repair):
    repair.compare_contract({"numeric": [0.0]}, {"numeric": [9e-11]})
    with pytest.raises(ValueError):
        repair.compare_contract({"numeric": [0.0]}, {"numeric": [1.1e-10]})
    for actual, saved in ((True, 1), (9, 9.0), ([True], [1]), ([9], [9.0]), (False, True)):
        with pytest.raises(ValueError):
            repair.compare_contract({"strict": actual}, {"strict": saved})
    with pytest.raises(ValueError):
        repair.compare_contract([0.0], [float("nan")])


def test_custom_numeric_comparator_still_receives_unmodified_values(repair):
    actual, saved = np.array([1.0]), np.array([1.0 + 1e-9])
    calls = []

    def comparator(label, left, right):
        calls.append((label, left, right))

    repair.compare_contract({"array": actual}, {"array": saved}, compare=comparator)
    assert len(calls) == 1 and calls[0][0] == "contract.array"
    assert calls[0][1] is actual and calls[0][2] is saved


def test_main_redirects_only_metadata_admission_and_output(monkeypatch, repair):
    frozen = repair.frozen
    query_engine, proof, builder = frozen.case_engine, frozen.native_math_proof, frozen.rebuild_requests
    monkeypatch.setattr(frozen, "load_prerequisites", frozen.load_prerequisites)
    monkeypatch.setattr(frozen, "CHECK_ROOT", frozen.CHECK_ROOT)
    calls = []
    monkeypatch.setattr(frozen, "main", lambda dry: calls.append(dry))
    repair.main(dry=True)
    assert calls == [True]
    assert frozen.compare_contract is repair.compare_contract
    assert frozen.load_prerequisites is repair.load_prerequisites
    assert frozen.CHECK_ROOT == repair.CHECK_ROOT
    assert frozen.case_engine is query_engine
    assert frozen.native_math_proof is proof
    assert frozen.rebuild_requests is builder
    assert all(proof()["functions"].values())


def test_repair_refs_merge_without_changing_original_inputs(monkeypatch, repair):
    import sys
    from types import SimpleNamespace

    original_ref = dict(path="original", sha256="a")
    extra_ref = dict(path="failed-owner", sha256="b")
    values = ({"diagnostic": 1}, {"producer": 2}, {"registration": 3}, {"original": original_ref})
    monkeypatch.setattr(repair, "ORIGINAL_LOAD_PREREQUISITES", lambda: values)
    monkeypatch.setitem(
        sys.modules,
        "s2a.saved_direction_backtracking_check_repair",
        SimpleNamespace(repair_prerequisites=lambda: {"failed-owner": extra_ref}),
    )
    bound, finished = [], []

    def remember(ref, refs):
        bound.append(ref)
        refs[ref["path"]] = ref

    monkeypatch.setattr(repair.frozen, "bind_reference", remember)
    monkeypatch.setattr(repair.frozen, "finish", lambda refs: finished.append(deepcopy(refs)))
    answer = repair.load_prerequisites()
    assert all(answer[index] is values[index] for index in range(4))
    assert bound == [extra_ref]
    assert finished == [{"original": original_ref, "failed-owner": extra_ref}]
