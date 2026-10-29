"""Synthetic-only regression for bounded in-root asset normalization."""

from pathlib import Path

import pytest
from test_native_runtime_bootstrap import case as shared_case

from s2a import native_runtime_bootstrap as frozen
from s2a import native_runtime_bootstrap_v2 as repaired


@pytest.fixture
def case(tmp_path, monkeypatch):
    return shared_case.__wrapped__(tmp_path, monkeypatch)


def model_with_parent_path(case):
    scenes = case.assets / "scenes"
    textures = case.assets / "textures"
    scenes.mkdir()
    textures.mkdir()
    texture = textures / "light-gray-floor-tile.png"
    texture.write_bytes(b"synthetic texture")
    raw = str(scenes / ".." / "textures" / texture.name)
    return f'<mujoco><asset><texture file="{raw}"/></asset></mujoco>', raw, texture


def test_proof_and_frozen_guards_remain_unchanged(case):
    before = dict(vars(frozen))
    proof = repaired.compile_repair_proof()
    assert proof["run_inverse_ast_equal"] and proof["private_asset_code_equal"]
    assert proof["global_path_guards_unchanged"] and not proof["model_xml_rewritten"]
    assert vars(frozen) == before
    with pytest.raises(ValueError, match="traversing"):
        frozen._safe_path(case.assets / ".." / "assets", (frozen.REPO,))


def test_legal_constructor_parent_paths_preserve_original_compiler_xml(case):
    xml, declared, canonical = model_with_parent_path(case)
    original_class = case.runtime.env_class
    compiled = []
    compile_original = case.runtime.sim_class.from_xml_string

    class Environment(original_class):
        def build(self, text, deterministic):
            super().build(xml if not deterministic else text, deterministic)

    def compile_record(cls, text):
        compiled.append(text)
        return compile_original(text)

    case.runtime.env_class = Environment
    case.runtime.sim_class.from_xml_string = classmethod(compile_record)
    result = repaired.run_native_bootstrap(**case.kwargs)
    assert result["passed"] and result["cleanup_outcome"] == "closed"
    assert result["cleanup_close_called"]
    assert compiled[:2] == [xml, xml]
    for model in result["model_records"][:2]:
        assert Path(model["xml_ref"]["path"]).read_text() == xml
        asset = model["assets"][0]
        assert asset["declared_path"] == declared
        assert asset["canonical_path"] == str(canonical)
        assert asset["reference"] == frozen._reference(canonical)
        assert asset["normalization_changed"]
    assert result["counters"]["model_construct"] == dict(attempted=3, completed=3)
    assert result["counters"]["forward"]["completed"] == 16


@pytest.mark.parametrize("suffix", ["../assets/mesh.stl", "scenes/../../assets/mesh.stl"])
def test_leaving_root_then_returning_is_rejected(case, suffix):
    (case.assets / "scenes").mkdir()
    with pytest.raises(ValueError, match="leaves approved root"):
        repaired.normalize_xml_asset_path(str(case.assets) + "/" + suffix, (case.assets,))


def test_symlink_discarded_by_parent_component_is_still_rejected(case):
    (case.assets / "scenes").symlink_to(case.assets, target_is_directory=True)
    path = str(case.assets / "scenes") + "/../mesh.stl"
    with pytest.raises(ValueError, match="symlink"):
        repaired.normalize_xml_asset_path(path, (case.assets,))


@pytest.mark.parametrize("component", ["missing", "mesh.stl"])
def test_discarded_component_must_exist_and_be_directory(case, component):
    path = str(case.assets / component) + "/../mesh.stl"
    with pytest.raises(ValueError, match="missing|not a directory"):
        repaired.normalize_xml_asset_path(path, (case.assets,))


def test_canonical_identity_does_not_merge_same_basename(case):
    other = case.assets / "other"
    other.mkdir()
    second = other / case.asset.name
    second.write_bytes(b"different asset")
    xml = f'<mujoco><asset><mesh file="{case.asset}"/><mesh file="{second}"/></asset></mujoco>'
    records = repaired._private_asset_records(xml, (case.assets,), None)
    assert len({row["canonical_path"] for row in records}) == 2
    assert len({row["reference"]["sha256"] for row in records}) == 2


def test_canonical_asset_is_bound_to_expected_digest(case):
    xml, _, canonical = model_with_parent_path(case)
    ref = frozen._reference(canonical)
    records = repaired._private_asset_records(xml, (case.assets,), {str(canonical): ref["sha256"]})
    assert records[0]["reference"] == ref
    with pytest.raises(ValueError, match="unbound or changed"):
        repaired._private_asset_records(xml, (case.assets,), {str(canonical): "0" * 64})


def test_partial_failure_before_sim_creation_is_not_reported_as_successful_close(case):
    runtime = case.runtime

    class Inner:
        def __init__(self):
            self.viewer = None
            runtime.sim_class.from_xml_string('<mujoco><asset><mesh file="/outside/mesh.stl"/></asset></mujoco>')

        def close(self):
            case.actions.close_calls += 1
            raise AssertionError("close must not be called before any resources exist")

    class Outer:
        def __init__(self, **kwargs):
            self.env = Inner()

        def close(self):
            raise AssertionError("outer .env is not assigned yet")

    runtime.env_base_class, runtime.env_class = Inner, Outer
    before = dict(vars(frozen))
    with pytest.raises(repaired.BootstrapFailure, match="approved root") as caught:
        repaired.run_native_bootstrap(**case.kwargs)
    result = caught.value.receipt
    assert result["failure_stage"] == "constructor"
    assert result["counters"]["model_construct"] == dict(attempted=1, completed=0)
    assert result["cleanup_target"] == "inner_partial"
    assert result["cleanup_outcome"] == "no_resources_created"
    assert result["cleanup_close_called"] is result["cleanup_complete"] is False
    assert "cleanup_error" not in result and not result["passed"]
    assert case.actions.close_calls == 0
    assert vars(frozen) == before


def test_partial_sim_is_closed_once_and_original_failure_remains(case):
    runtime = case.runtime

    class Inner:
        def __init__(self):
            self.sim = runtime.sim_class.from_xml_string(case.xml)
            raise RuntimeError("after actual synthetic model creation")

        def close(self):
            case.actions.close_calls += 1

    class Outer:
        def __init__(self, **kwargs):
            self.env = Inner()

        def close(self):
            raise AssertionError("partial outer must not close")

    runtime.env_base_class, runtime.env_class = Inner, Outer
    with pytest.raises(repaired.BootstrapFailure, match="after actual synthetic") as caught:
        repaired.run_native_bootstrap(**case.kwargs)
    result = caught.value.receipt
    assert result["cleanup_complete"] and result["cleanup_close_called"]
    assert result["cleanup_outcome"] == "closed" and case.actions.close_calls == 1
    assert not result["passed"]


def test_output_and_record_guards_are_not_relaxed(case):
    case.kwargs["output"] = case.kwargs["output"] / ".." / "other_output"
    with pytest.raises(repaired.BootstrapFailure, match="traversing"):
        repaired.run_native_bootstrap(**case.kwargs)
    assert case.actions.close_calls == 0


@pytest.mark.parametrize("failure", ["mj_step", "mj_step2", "control_step", "constant_frame"])
def test_original_control_and_render_qualification_still_apply(case, failure):
    case.actions.failure = failure
    with pytest.raises(repaired.BootstrapFailure) as caught:
        repaired.run_native_bootstrap(**case.kwargs)
    assert not caught.value.receipt["bootstrap_verified"]
    if failure != "constant_frame":
        assert caught.value.receipt["counters"][failure] == dict(attempted=1, completed=0)
        assert case.actions.native_steps == 0
