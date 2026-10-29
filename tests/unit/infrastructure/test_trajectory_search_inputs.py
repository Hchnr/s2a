"""Real Git identities and synthetic saved inputs, without model or solver calls."""

import ast
import hashlib
import importlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from s2a.infrastructure import current_layout
from s2a.infrastructure.repository_paths import FORBIDDEN_HISTORICAL_ROOT
from s2a.infrastructure.snapshots import xml_asset_snapshot as snapshot
from s2a.research.trajectory_search import inputs


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return reference(path)


def git(root, *args):
    process = subprocess.Popen(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0"),
    )
    stdout, stderr = process.communicate(timeout=30)
    assert process.wait() == 0, stderr.decode()
    return stdout


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "repository"
    (root / "src/s2a").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "s2a"\n')
    source = root / "src/s2a/subject.py"
    source.write_text("raise RuntimeError('historical source must not execute')\n")
    config = dict(seed=42, max_stages=60, max_responses=122, safety_buffer=1e-8, accept_ratio=0.1)
    old_config = root / "configs/state_vector_objective_scale_20261029.json"
    old_config_ref = write_json(old_config, config)
    plan = root / "plans/v0.0.1/state_vector_dual_recovery_20261029.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("Original immutable plan.\n")
    source_ref, plan_ref = reference(source), reference(plan)
    repair_sources = []
    for relative in inputs.ORIGINAL_REPAIR_SOURCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# immutable original repair source; never imported\n")
        repair_sources.append(path)
    git(root, "init", "--quiet")
    git(root, "add", "pyproject.toml", "src", "configs", "plans", "scripts", "tests")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "base")
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    entries = []
    for path in (source, old_config, plan, *repair_sources):
        relative = path.relative_to(root).as_posix()
        mode, _, oid = git(root, "ls-tree", "HEAD", "--", relative).decode().split("\t")[0].split()
        entries.append(
            dict(
                old_path=relative,
                sha256=reference(path)["sha256"],
                size_bytes=path.stat().st_size,
                git_blob_oid=oid,
                git_mode=mode,
                baseline_commit=commit,
            )
        )
    catalog_ref = write_json(
        root / "docs/migrations/catalog.json",
        dict(
            schema_version=1,
            baseline_commit=commit,
            file_count=len(entries),
            total_size_bytes=sum(row["size_bytes"] for row in entries),
            entries=entries,
        ),
    )
    moved = root / "src/s2a/research/subject.py"
    moved.parent.mkdir()
    source.rename(moved)
    moved.write_text("# current implementation has migrated imports\n")
    new_config = root / "configs/tracking/state_vector_dual_recovery.json"
    new_config.parent.mkdir(parents=True)
    new_config.write_bytes(old_config.read_bytes())
    manifest_ref = write_json(
        root / "docs/migrations/move.json",
        dict(
            catalog_ref=catalog_ref,
            moves=[
                dict(
                    old_path=source.relative_to(root).as_posix(),
                    original_sha256=source_ref["sha256"],
                    new_path=moved.relative_to(root).as_posix(),
                    current_sha256=reference(moved)["sha256"],
                )
            ],
        ),
    )
    data_ref = write_json(root / "outputs/input/data.json", {"original": 368})
    bundle = dict(
        base_response_index=3557,
        trial_response_index=3639,
        probe_response_indices=list(range(3558, 3630)),
        phase=dict(stage_index=53, radius=0.005),
        base={"original": 3557},
        probes=[{} for _ in range(72)],
        source={},
        source_registration={},
        control={},
        nominal_deltas=[0.0001] * 72,
        input_refs=[source_ref, data_ref],
        closure_ref=data_ref,
        producer_ref=data_ref,
        source_bundle_ref=data_ref,
    )
    bundle_ref = write_json(root / "outputs/input/bundle.json", bundle)
    prior = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        costs={"variant": {"response_calls_per_route": 5, "stage_count": 10}},
        variant=dict(witness_indices=[], selected_response_index=None, termination_reason="radius_below_minimum"),
        source_refs_unchanged=[source_ref],
    )
    diagnostic = dict(
        complete=True,
        passed=True,
        sources_unchanged=True,
        new_activity=dict(qp_solves=0, fits=0),
        original_complete_witness_count=0,
        original_failures_retained=True,
        original_ledger_audit=dict(passed=True),
        records=[
            dict(
                original_solver_status=0,
                original_solver_success=True,
                original_qualified=i < 3,
            )
            for i in range(10)
        ],
        source_refs_unchanged=[source_ref],
        fitted_model="must never initialize the new search",
    )
    prior_ref = write_json(root / "outputs/input/prior.json", prior)
    diagnostic_ref = write_json(root / "outputs/input/diagnostic.json", diagnostic)
    return dict(
        root=root,
        commit=commit,
        catalog_ref=catalog_ref,
        manifest_ref=manifest_ref,
        source_ref=source_ref,
        moved=moved,
        plan_ref=plan_ref,
        old_config_ref=old_config_ref,
        new_config=new_config,
        config=config,
        data_ref=data_ref,
        bundle=bundle,
        bundle_ref=bundle_ref,
        prior=prior,
        prior_ref=prior_ref,
        diagnostic=diagnostic,
        diagnostic_ref=diagnostic_ref,
    )


def context(fixture):
    return inputs.SearchInputContext(
        root=fixture["root"],
        baseline_commit=fixture["commit"],
        catalog_ref=fixture["catalog_ref"],
        migration_refs=(fixture["manifest_ref"],),
        cache_directory=fixture["root"] / "outputs/cache",
    )


def load(fixture, ctx=None):
    return inputs.load_dual_recovery_inputs(
        context(fixture) if ctx is None else ctx,
        config_path=fixture["new_config"],
        current_source_paths=(fixture["moved"],),
        support_paths=(),
        bundle_ref=fixture["bundle_ref"],
        prior_analysis_ref=fixture["prior_ref"],
        diagnostic_ref=fixture["diagnostic_ref"],
        historical_config_ref=fixture["old_config_ref"],
    )


def test_original_stencil_and_current_sources_have_distinct_identity_tables(fixture):
    bundle, config, ctx = load(fixture)
    assert bundle == fixture["bundle"] and config == fixture["config"]
    assert "fitted_model" not in bundle and bundle["base_response_index"] == 3557
    registration = ctx.registration()
    sources = registration["historical_source_identities"]
    original = next(row for row in sources if row["declared_ref"] == fixture["source_ref"])
    assert original["resolution_kind"] == "git_cache" and original["identity_scope"] == "byte_identity_only"
    assert Path(original["storage_ref"]["path"]).read_text().startswith("raise RuntimeError")
    assert str(fixture["root"] / "src/s2a/subject.py") not in ctx.runtime_refs
    assert ctx.current_source_refs[str(fixture["moved"])] == reference(fixture["moved"])
    assert registration["historical_source_execution_admitted"] is False
    assert registration["previous_failed_checker_qualification_changed"] is False
    assert all(
        row["actual_wait"] and row["reaped"] and row["returncode"] == 0
        for row in registration["identity_git_operations"]
    )


def test_same_path_old_and_current_source_digests_do_not_conflict(fixture):
    old = fixture["root"] / "src/s2a/subject.py"
    old.write_text("# current source may occupy the original declared path\n")
    ctx = context(fixture)
    ctx.bind_current_source(old)
    ctx.remember_historical(fixture["source_ref"])
    assert ctx.runtime_refs[str(old)]["sha256"] != fixture["source_ref"]["sha256"]
    assert ctx.registration()["historical_source_identities"][0]["storage_ref"]["path"] != str(old)


def test_root_metadata_has_explicit_supplement_identity_separate_from_new_live_bytes(fixture):
    root = fixture["root"]
    path = root / "pyproject.toml"
    original = reference(path)
    mode, _, oid = git(root, "ls-tree", "HEAD", "--", "pyproject.toml").decode().split("\t")[0].split()
    row = dict(
        old_path="pyproject.toml",
        sha256=original["sha256"],
        size_bytes=path.stat().st_size,
        git_blob_oid=oid,
        git_mode=mode,
        baseline_commit=fixture["commit"],
    )
    supplement = write_json(
        root / "docs/migrations/root_files.json",
        dict(
            schema_version=1,
            baseline_commit=fixture["commit"],
            file_count=1,
            total_size_bytes=row["size_bytes"],
            entries=[row],
        ),
    )
    path.write_text(path.read_text() + '\n[tool.pytest.ini_options]\npythonpath=["src", "."]\n')
    ctx = inputs.SearchInputContext(
        root=root,
        baseline_commit=fixture["commit"],
        catalog_ref=fixture["catalog_ref"],
        additional_catalog_refs=(supplement,),
        migration_refs=(fixture["manifest_ref"],),
        cache_directory=root / "outputs/cache",
    )
    ctx.bind_current_source(path)
    ctx.remember_historical(original)
    receipt = ctx.registration()["historical_source_identities"][0]
    assert receipt["catalog_ref"] == supplement and receipt["resolution_kind"] == "git_cache"
    assert ctx.current_source_refs[str(path)]["sha256"] != original["sha256"]


@pytest.mark.parametrize("defect", ["artifact", "current_source", "manifest", "historical_cache"])
def test_consumed_bytes_and_migration_metadata_are_rechecked(fixture, defect):
    _, _, ctx = load(fixture)
    if defect == "artifact":
        changed = Path(fixture["data_ref"]["path"])
    elif defect == "current_source":
        changed = fixture["moved"]
    elif defect == "manifest":
        changed = Path(fixture["manifest_ref"]["path"])
    else:
        changed = Path(ctx.registration()["historical_source_identities"][0]["storage_ref"]["path"])
    changed.write_text("changed\n")
    with pytest.raises(ValueError):
        ctx.finish()


@pytest.mark.parametrize("defect", ["sha", "unknown_source", "untracked_root", "old_output_root", "traversal"])
def test_unknown_or_forbidden_declarations_never_become_current_fallback(fixture, defect):
    ctx = context(fixture)
    ref = deepcopy(fixture["source_ref"])
    if defect == "sha":
        ref["sha256"] = "0" * 64
    elif defect == "unknown_source":
        ref["path"] = str(fixture["root"] / "src/s2a/unregistered.py")
    elif defect == "untracked_root":
        ref["path"] = str(fixture["root"] / "long_joint_inverse.py")
    elif defect == "old_output_root":
        ref["path"] = str(FORBIDDEN_HISTORICAL_ROOT / "outputs/never_read.json")
    else:
        ref["path"] = str(fixture["root"]) + "/src/../unregistered.py"
    with pytest.raises(ValueError):
        ctx.remember_historical(ref)


def test_legacy_root_source_is_resolved_as_declared_bytes_without_visiting_old_root(fixture):
    ctx = context(fixture)
    ref = dict(fixture["source_ref"], path=str(FORBIDDEN_HISTORICAL_ROOT / "src/s2a/subject.py"))
    ctx.remember_historical(ref)
    receipt = ctx.registration()["historical_source_identities"][0]
    assert receipt["declared_ref"] == ref and receipt["resolution_kind"] == "git_cache"


@pytest.mark.parametrize(
    "key,value",
    [
        ("base_response_index", 1),
        ("trial_response_index", 1),
        ("probe_response_indices", []),
        ("phase", {"stage_index": 54, "radius": 0.005}),
    ],
)
def test_original_stencil_indices_and_phase_are_fixed(fixture, key, value):
    fixture["bundle"][key] = value
    fixture["bundle_ref"] = write_json(Path(fixture["bundle_ref"]["path"]), fixture["bundle"])
    with pytest.raises(ValueError, match="original 3557"):
        load(fixture)


@pytest.mark.parametrize(
    "which,key,value",
    [
        ("prior", "passed", False),
        ("prior", "sources_unchanged", False),
        ("diagnostic", "passed", False),
        ("diagnostic", "original_failures_retained", False),
        ("diagnostic", "original_complete_witness_count", 1),
        ("diagnostic", "new_activity", {"fits": False}),
    ],
)
def test_closed_baseline_and_zero_activity_failures_are_preserved(fixture, which, key, value):
    fixture[which][key] = value
    fixture[which + "_ref"] = write_json(Path(fixture[which + "_ref"]["path"]), fixture[which])
    with pytest.raises(ValueError):
        load(fixture)


@pytest.mark.parametrize("defect", ["stage", "responses", "gate", "ratio", "format"])
def test_current_configuration_must_match_original_values_and_bytes(fixture, defect):
    config = deepcopy(fixture["config"])
    if defect == "stage":
        config["max_stages"] = 61
    elif defect == "responses":
        config["max_responses"] = 123
    elif defect == "gate":
        config["safety_buffer"] = 1e-9
    elif defect == "ratio":
        config["accept_ratio"] = 0.01
    if defect == "format":
        fixture["new_config"].write_text(json.dumps(config, separators=(",", ":")))
    else:
        write_json(fixture["new_config"], config)
    with pytest.raises(ValueError, match="All numeric gates"):
        load(fixture)


def test_historical_registration_keeps_declared_paths_after_migration(fixture):
    _, config, ctx = load(fixture)
    registration = dict(config_ref=fixture["old_config_ref"], plan_ref=fixture["plan_ref"], config=config)
    inputs.validate_historical_registration(
        ctx, registration, config=config, old_config_ref=fixture["old_config_ref"], old_plan_ref=fixture["plan_ref"]
    )
    registration["config_ref"] = reference(fixture["new_config"])
    with pytest.raises(ValueError, match="Original declared"):
        inputs.validate_historical_registration(
            ctx, registration, config=config, old_config_ref=fixture["old_config_ref"], old_plan_ref=fixture["plan_ref"]
        )


def test_runtime_helpers_may_only_extend_strict_physical_inputs(fixture):
    _, _, ctx = load(fixture)
    extra = write_json(fixture["root"] / "outputs/native.json", {"observed": 368})
    ctx.runtime_refs[extra["path"]] = extra
    ctx.finish()
    Path(extra["path"]).write_text("changed")
    with pytest.raises(ValueError):
        ctx.finish()


@pytest.fixture
def external_asset(fixture, monkeypatch):
    asset_root = fixture["root"].parent / "external/libero/assets"
    asset_root.mkdir(parents=True)
    monkeypatch.setattr(inputs, "LIBERO_ASSET_ROOT", asset_root)
    asset = asset_root / "meshes/subject.msh"
    asset.parent.mkdir()
    asset.write_bytes(b"fixed synthetic mesh bytes\n")
    ref = reference(asset)
    fixture["bundle"]["input_refs"].append(ref)
    fixture["bundle_ref"] = write_json(Path(fixture["bundle_ref"]["path"]), fixture["bundle"])
    return asset_root, asset, ref


def test_external_fixed_asset_keeps_exact_bundle_receipt_separate_from_sources(fixture, external_asset):
    _, asset, ref = external_asset
    _, _, ctx = load(fixture)
    registration = ctx.registration()
    assert registration["external_artifact_identities"] == [dict(declared_ref=ref, bundle_ref=fixture["bundle_ref"])]
    assert str(asset) not in ctx.runtime_refs and str(asset) not in ctx.current_source_refs
    assert ref not in registration["historical_artifact_refs"]
    assert registration["external_artifact_execution_admitted"] is False
    assert ctx.historical_bytes(ref) == asset.read_bytes()
    with pytest.raises(ValueError, match="inside repository"):
        ctx.bind_current_source(asset)
    # Inserting a read-only mesh into the runtime source map cannot escalate it.
    ctx.runtime_refs[str(asset)] = ref
    with pytest.raises(ValueError, match="inside repository"):
        ctx.finish()


@pytest.mark.parametrize("defect", ["unregistered", "changed_sha", "unknown_asset"])
def test_external_asset_requires_exact_registered_path_and_sha(fixture, external_asset, defect):
    asset_root, _, ref = external_asset
    ctx = context(fixture)
    if defect != "unregistered":
        ctx.register_external_bundle_artifacts(fixture["bundle_ref"])
    if defect == "changed_sha":
        ref = dict(ref, sha256="0" * 64)
    elif defect == "unknown_asset":
        other = asset_root / "meshes/unlisted.msh"
        other.write_bytes(b"unlisted asset\n")
        ref = reference(other)
    with pytest.raises(ValueError, match="exact SHA-bound bundle"):
        ctx.remember_historical(ref)


@pytest.mark.parametrize("when", ["first_read", "finish"])
def test_external_asset_byte_drift_is_rejected_on_consumption_and_finish(fixture, external_asset, when):
    _, asset, ref = external_asset
    if when == "finish":
        _, _, ctx = load(fixture)
    else:
        ctx = context(fixture)
        ctx.register_external_bundle_artifacts(fixture["bundle_ref"])
    asset.write_bytes(b"changed mesh bytes\n")
    with pytest.raises(ValueError, match="External artifact SHA256 mismatch"):
        ctx.finish() if when == "finish" else ctx.remember_historical(ref)


@pytest.mark.parametrize("defect", ["source", "outside_assets", "forbidden_root", "symlink"])
def test_external_bundle_cannot_admit_code_other_roots_or_links(fixture, external_asset, monkeypatch, defect):
    asset_root, asset, ref = external_asset
    if defect == "source":
        path = asset_root / "execute.py"
        path.write_text("raise RuntimeError('must not execute or admit as source')\n")
        ref = reference(path)
    elif defect == "outside_assets":
        path = asset_root.parent / "outside.msh"
        path.write_bytes(asset.read_bytes())
        ref = reference(path)
    elif defect == "forbidden_root":
        ref = dict(ref, path=str(FORBIDDEN_HISTORICAL_ROOT / "outputs/forbidden.msh"))
    else:
        path = asset_root / "linked.msh"
        path.symlink_to(asset)
        ref = dict(ref, path=str(path))
    fixture["bundle"]["input_refs"][-1] = ref
    fixture["bundle_ref"] = write_json(Path(fixture["bundle_ref"]["path"]), fixture["bundle"])
    digest = inputs._stream_digest

    def guarded_digest(path):
        assert not Path(path).is_relative_to(FORBIDDEN_HISTORICAL_ROOT)
        assert str(path) != ref["path"]  # Reject this invalid declaration before reading its bytes.
        return digest(path)

    monkeypatch.setattr(inputs, "_stream_digest", guarded_digest)
    with pytest.raises(ValueError):
        load(fixture)


def test_external_allowlist_cannot_be_loaded_from_a_bundle_with_changed_bytes(fixture, external_asset):
    ctx = context(fixture)
    Path(fixture["bundle_ref"]["path"]).write_text('{"input_refs": []}\n')
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        ctx.register_external_bundle_artifacts(fixture["bundle_ref"])


@pytest.mark.parametrize("approved", [False, True])
def test_default_context_requires_the_shared_approved_current_manifest(fixture, monkeypatch, approved):
    monkeypatch.setattr(inputs, "BASELINE_COMMIT", fixture["commit"])
    monkeypatch.setattr(current_layout, "CURRENT_MIGRATION_REF", fixture["manifest_ref"] if approved else None)
    kwargs = dict(root=fixture["root"], catalog_ref=fixture["catalog_ref"], additional_catalog_refs=())
    if approved:
        ctx = inputs.make_input_context(**kwargs)
        assert ctx.migration_refs == (fixture["manifest_ref"],)
        assert inputs.MIGRATION_PATH == current_layout.CURRENT_MIGRATION_PATH
        ctx.finish()
    else:
        with pytest.raises(ValueError, match="explicitly approved and pinned"):
            inputs.make_input_context(**kwargs)
        # Explicit fixture references remain available without approving a default.
        ctx = inputs.make_input_context(**kwargs, migration_refs=(fixture["manifest_ref"],))
        assert ctx.migration_refs == (fixture["manifest_ref"],)


@pytest.fixture
def model_proof(fixture, monkeypatch):
    root = fixture["root"]
    safe = root / "outputs/synthetic_source_assets"
    safe.mkdir(parents=True)
    monkeypatch.setattr(snapshot, "REPO", root)
    monkeypatch.setattr(snapshot, "SAFE_ROOT", safe)
    assets = []
    for index in range(96):
        path = safe / f"mesh_{index:03d}.msh"
        path.write_bytes(b"synthetic nonphysical mesh " + str(index).encode())
        assets.append(path)
    xml = root / inputs.ORIGINAL_XML_PATH
    xml.parent.mkdir(parents=True)
    declarations = assets + assets[:3]
    xml.write_text(
        "<mujoco><asset>"
        + "".join(f'<mesh name="m{i}" file="{path}"/>' for i, path in enumerate(declarations))
        + "</asset></mujoco>\n"
    )
    history = root / inputs.HISTORICAL_ASSET_MANIFEST_PATH
    history_ref = write_json(
        history, dict(assets={str(path): {"sha256": reference(path)["sha256"]} for path in assets})
    )
    snapshot_ref = snapshot.prepare_asset_snapshot(reference(xml), history_ref, root / inputs.SNAPSHOT_ROOT)
    manifest = snapshot.verify_asset_snapshot(snapshot_ref)
    compiled = dict(
        complete=True,
        passed=True,
        manifest_ref=deepcopy(snapshot_ref),
        runtime=dict(mujoco="2.3.7", numpy="1.26.4", python="3.11.synthetic"),
        environment=deepcopy(inputs.CPU_NATIVE_ENVIRONMENT),
        model_dimensions=dict(nq=9, nv=9),
        timestep=0.002,
        response_evaluations=0,
        query_count=0,
        numerical_inner_steps=0,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        source_refs=[reference(root / path) for path in inputs.ORIGINAL_REPAIR_SOURCE_PATHS],
    )
    compile_ref = write_json(root / inputs.SNAPSHOT_ROOT / "compile.json", compiled)
    bundle = dict(
        input_refs=[snapshot_ref, compile_ref, reference(xml), history_ref],
    )
    return dict(
        bundle=bundle,
        original_registration=dict(xml_ref=reference(xml), original_control="unchanged"),
        snapshot_ref=snapshot_ref,
        compile_ref=compile_ref,
        manifest=manifest,
        compiled=compiled,
    )


def prove_model(fixture, model_proof, ctx=None):
    return inputs.model_registration(
        model_proof["original_registration"],
        context(fixture) if ctx is None else ctx,
        model_proof["bundle"],
        current_source_paths=(fixture["moved"],),
    )


def repin_compilation(model_proof):
    old = model_proof["compile_ref"]
    updated = write_json(Path(old["path"]), model_proof["compiled"])
    model_proof["bundle"]["input_refs"] = [
        updated if ref == old else ref for ref in model_proof["bundle"]["input_refs"]
    ]
    model_proof["compile_ref"] = updated


def test_model_asset_proof_remains_exact_while_migrated_identity_is_separate(fixture, model_proof):
    # A current repair source may differ while its original compile byte
    # identity remains available through Git; it is not executed from cache.
    changed = fixture["root"] / inputs.ORIGINAL_REPAIR_SOURCE_PATHS[0]
    changed.write_text("# migrated current imports differ\n")
    ctx = context(fixture)
    registration, proof = prove_model(fixture, model_proof, ctx)
    assert registration == dict(
        model_proof["original_registration"], xml_ref=model_proof["manifest"]["migrated_xml_ref"]
    )
    assert proof == dict(
        snapshot_ref=model_proof["snapshot_ref"],
        compile_ref=model_proof["compile_ref"],
        original_xml_ref=model_proof["original_registration"]["xml_ref"],
        historical_manifest_ref=model_proof["manifest"]["historical_manifest_ref"],
        migrated_xml_ref=model_proof["manifest"]["migrated_xml_ref"],
        asset_count=96,
        complete_model_preserved=True,
        model_body_or_geom_pruning=False,
    )
    receipt = ctx.registration()["model_snapshot_identities"][0]
    assert receipt["historical_compile_source_refs"] == model_proof["compiled"]["source_refs"]
    assert receipt["historical_compile_identity_scope"] == "byte_identity_only"
    assert receipt["current_source_refs"] == [reference(fixture["moved"])]
    assert receipt["historical_compilation_qualification_migrated"] is receipt["migrated_execution_qualified"] is False
    assert receipt["new_model_compilations"] == receipt["new_forward_queries"] == 0
    assert receipt["previous_failed_checker_qualification_changed"] is False


@pytest.mark.parametrize("target", ["source_asset", "copied_asset", "model_xml"])
def test_model_proof_rejects_missing_asset_bytes(fixture, model_proof, target):
    manifest = model_proof["manifest"]
    ref = (
        manifest["assets"][0]["safe_source_ref"]
        if target == "source_asset"
        else manifest["assets"][0]["snapshot_ref"]
        if target == "copied_asset"
        else manifest["migrated_xml_ref"]
    )
    Path(ref["path"]).unlink()
    with pytest.raises((ValueError, OSError)):
        prove_model(fixture, model_proof)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_source",
        "source_sha",
        "manifest_ref",
        "runtime",
        "environment",
        "dimensions",
        "timestep",
        "query",
        "passed",
    ],
)
def test_model_proof_preserves_original_compilation_gates(fixture, model_proof, defect):
    compiled = model_proof["compiled"]
    if defect == "missing_source":
        compiled["source_refs"].pop()
    elif defect == "source_sha":
        compiled["source_refs"][0]["sha256"] = "0" * 64
    elif defect == "manifest_ref":
        compiled["manifest_ref"]["sha256"] = "0" * 64
    elif defect == "runtime":
        compiled["runtime"]["mujoco"] = "unexpected"
    elif defect == "environment":
        compiled["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif defect == "dimensions":
        compiled["model_dimensions"]["nq"] = 0
    elif defect == "timestep":
        compiled["timestep"] = 0.004
    elif defect == "query":
        compiled["query_count"] = 1
    else:
        compiled["passed"] = False
    repin_compilation(model_proof)
    with pytest.raises(ValueError, match="compilation|Compilation"):
        prove_model(fixture, model_proof)


@pytest.mark.parametrize(
    "key,value", [("file_reference_count", 98), ("unique_asset_count", 95), ("byte_exact_inverse", False)]
)
def test_model_proof_keeps_all_99_references_96_assets_and_inverse_bytes(fixture, model_proof, monkeypatch, key, value):
    # The real pure asset verifier is exercised by the other asset tests; this
    # mutation isolates the additional fixed 99/96 admission assertions.
    verified = deepcopy(model_proof["manifest"])
    verified[key] = value
    monkeypatch.setattr(snapshot, "verify_asset_snapshot", lambda ref: verified)
    with pytest.raises(ValueError, match="99-reference model"):
        prove_model(fixture, model_proof)


@pytest.mark.parametrize("target", ["copied_asset", "compile", "current_source"])
def test_model_proof_finish_rechecks_all_consumed_refs(fixture, model_proof, target):
    ctx = context(fixture)
    prove_model(fixture, model_proof, ctx)
    path = (
        Path(model_proof["manifest"]["assets"][0]["snapshot_ref"]["path"])
        if target == "copied_asset"
        else Path(model_proof["compile_ref"]["path"])
        if target == "compile"
        else fixture["moved"]
    )
    path.write_bytes(b"changed after proof\n")
    with pytest.raises(ValueError):
        ctx.finish()


@pytest.mark.parametrize("kind", ["probe", "check"])
def test_actual_entry_checks_current_source_identity_before_inputs_or_numerics(monkeypatch, kind):
    producer = importlib.import_module(
        "scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery"
    )
    entry = importlib.import_module(f"scripts.tracking.trajectory_search.{kind}_tracking_v5_state_vector_dual_recovery")
    calls = []

    class RejectBeforeUse:
        def __init__(self, **kwargs):
            calls.append("construct current identity guard")

        def verify_present(self):
            calls.append("verify before use")
            raise ValueError("synthetic current source drift")

    def no_inputs():
        raise AssertionError("Inputs or numerical work reached before identity admission")

    monkeypatch.setattr(entry, "CurrentScriptImports", RejectBeforeUse)
    monkeypatch.setattr(producer, "load_inputs", no_inputs)
    with pytest.raises(ValueError, match="synthetic current source drift"):
        entry.main(dry=True)
    assert calls == ["construct current identity guard", "verify before use"]


def test_input_module_never_imports_producer_model_solver_or_old_loaders():
    source = Path(inputs.__file__).read_text()
    tree = ast.parse(source)
    names = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith(("scripts", "scipy", "mujoco", "torch")) for name in names)
    assert "load_prior_inputs" not in source
    assert "1e-10" not in source  # The original fresh decision gate remains in its checker.
