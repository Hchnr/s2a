"""Actual file fingerprints for the repair's original-proof/source-overlay boundary."""

import copy
from types import SimpleNamespace

import pytest

from s2a.development_runtime_bridge import admit_runtime, extend_sources
from s2a.durable_stage_supervisor import reference


@pytest.fixture
def bridge(tmp_path):
    paths = {name: tmp_path / (name + ".py") for name in ("original", "generated", "common")}
    for name, path in paths.items():
        path.write_text(f"# {name}\n")
    refs = {name: reference(path) for name, path in paths.items()}
    original = {refs[name]["path"]: refs[name]["sha256"] for name in ("original", "common")}
    q = dict(
        prepared_ref=dict(path="original-prepared", sha256="a" * 64),
        execution_registration_ref=dict(path="original-registration", sha256="b" * 64),
        generated_to_original={str(paths["generated"]): refs["original"]},
        source_sha256={ref["path"]: ref["sha256"] for ref in refs.values()},
    )
    prepared, registration = dict(original_preparation=True), dict(original_registration=True)
    calls = []

    def admit(prepared_ref, registration_ref, *, required_sources):
        calls.append((prepared_ref, registration_ref, required_sources))
        return prepared, registration, original

    runtime = SimpleNamespace(admit_execution=admit)
    return paths, q, runtime, calls, prepared, registration, original


def test_real_original_proof_receives_original_paths_and_json_stays_identical(bridge):
    paths, q, runtime, calls, prepared, registration, original = bridge
    before = copy.deepcopy((q, prepared, registration, original))
    p, r, sources = admit_runtime(
        runtime,
        q["prepared_ref"],
        q["execution_registration_ref"],
        required_sources=[paths["generated"], paths["common"]],
        qualification=q,
    )
    assert p is prepared and r is registration
    assert calls == [(q["prepared_ref"], q["execution_registration_ref"], [paths["original"], paths["common"]])]
    assert sources == q["source_sha256"]
    assert (q, prepared, registration, original) == before


@pytest.mark.parametrize(
    "fault",
    (
        "modified_generated",
        "modified_original",
        "missing_actual",
        "omitted_basis",
        "different_prepared",
        "different_registration",
        "unfrozen_basis",
    ),
)
def test_source_or_identity_substitution_rejected_before_delegation(bridge, fault):
    paths, q, runtime, calls, _, _, original = bridge
    prepared_ref = q["prepared_ref"]
    registration_ref = q["execution_registration_ref"]
    if fault == "modified_generated":
        paths["generated"].write_text("changed")
    elif fault == "modified_original":
        paths["original"].write_text("changed")
    elif fault == "missing_actual":
        del q["source_sha256"][str(paths["generated"])]
    elif fault == "omitted_basis":
        del q["source_sha256"][str(paths["original"])]
    elif fault == "different_prepared":
        prepared_ref = {}
    elif fault == "different_registration":
        registration_ref = {}
    else:
        del original[str(paths["original"])]
    with pytest.raises(ValueError):
        admit_runtime(runtime, prepared_ref, registration_ref, required_sources=[paths["generated"]], qualification=q)
    assert len(calls) == int(fault == "unfrozen_basis")


def test_overlay_cannot_replace_or_skip_any_original_file(bridge):
    paths, q, _, _, _, _, original = bridge
    changed = dict(q["source_sha256"])
    changed[str(paths["original"])] = "0" * 64
    with pytest.raises(ValueError, match="changed an original"):
        extend_sources(original, changed)
    paths["common"].unlink()
    with pytest.raises(ValueError):
        extend_sources(original, q["source_sha256"])
