"""035 owner boundaries use synthetic files and driver replies, never a model."""

import copy
import importlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from s2a.artifacts import REPO, sha256
from s2a.relocated_artifacts import RelocatedArtifacts


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("run_wam_clean_bridge")


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return dict(path=str(path), sha256=sha256(path))


@pytest.fixture
def prepared(runner, tmp_path, monkeypatch):
    config = json.loads((REPO / "configs/wam_clean_bridge_20261022.json").read_text())
    root = tmp_path / "outputs/bridge"
    config.update(output_dir="outputs/bridge", cosmos_source_root=str(root / "source/cosmos"))
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "CONFIG", tmp_path / "config.json")
    monkeypatch.setattr(runner, "LOCATOR", RelocatedArtifacts(tmp_path, tmp_path / "recovery"))
    historical = dict(anchors=[], free_generations=[None] * 33, weight_files={})
    for index, name in zip(
        (0, 16, 32), (config["nonzero_request_name"], "task1_raw_s42", "task2_raw_s42"), strict=True
    ):
        image = tmp_path / f"image_{index}.png"
        image.write_bytes(f"synthetic image {index}".encode())
        response_ref = dump(tmp_path / f"response_{index}.json", dict(synthetic=index))
        anchor = dict(name=f"anchor{index}", image_path=str(image), image_sha256=sha256(image), anchor_frame=0)
        historical["anchors"].append(anchor)
        historical["free_generations"][index] = dict(
            variant="raw",
            sampling_seed=42,
            anchor_name=anchor["name"],
            name=name,
            request=dict(
                vision_path=str(image),
                seed=42,
                model_mode="wam",
                guidance=1.0,
                shift=5.0,
                num_steps=35,
                action_chunk_size=40,
            ),
            cache=dict(response=response_ref),
        )
    for name in ("config.yaml", "model/.metadata", ".save_complete", *(f"model/{i}.distcp" for i in range(160))):
        path = tmp_path / "weights" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        stat = path.stat()
        historical["weight_files"][str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(path))
    config["historical_prepared_ref"] = dump(tmp_path / "historical.json", historical)
    dump(runner.CONFIG, config)
    samplers = root / "source/cosmos/cosmos_framework/model/generator/diffusion/samplers"
    samplers.mkdir(parents=True)
    sampler, converter = samplers / "unipc.py", samplers / "fm_solvers_unipc.py"
    sampler.write_text("# sampler\n")
    converter.write_text("# converter\n")
    snapshot = root / "source/s2a/example.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("# preserved source\n")
    requests = runner.historical_requests(config, historical)
    sources = {str(p): sha256(p) for p in (Path(runner.__file__), sampler, converter, runner.CONFIG)}
    inputs = {config["historical_prepared_ref"]["path"]: config["historical_prepared_ref"]["sha256"]}
    for entry in requests:
        for key in ("image_ref", "expected_response_ref"):
            inputs[entry[key]["path"]] = entry[key]["sha256"]
    small = {
        path: value["sha256"]
        for path, value in historical["weight_files"].items()
        if Path(path).name in ("config.yaml", ".metadata", ".save_complete")
    }
    inputs.update(small)
    data = dict(
        schema_version=runner.SCHEMA,
        root=str(root),
        config=config,
        source_sha256=sources,
        snapshot_sha256={str(snapshot): sha256(snapshot)},
        input_sha256=inputs,
        weight_files=historical["weight_files"],
        checkpoint_small_metadata_sha256=small,
        requests3=requests,
        sampler_source_ref=runner.reference(sampler),
        scheduler_conversion_source_ref=runner.reference(converter),
        weight_integrity_contract="historical163_stat_and_three_small_metadata_sha_no_new_shard_hash",
    )
    reference = dump(root / "preparation/prepared.json", data)
    return data, reference


def test_configuration_rejects_forbidden_gpu_fitting_fresh_noise_and_budget(runner, prepared):
    data, _ = prepared
    good = data["config"]
    assert runner.validate_config(good) == Path(data["root"])
    for uuid in ("0", "2", "6", "GPU-forbidden", None):
        bad = copy.deepcopy(good)
        bad["gpu_uuid"] = uuid
        with pytest.raises(ValueError, match="allowed UUID"):
            runner.validate_config(bad)
    for field in ("model_training_authorized", "fresh_performance_authorized", "automatic_retry"):
        bad = copy.deepcopy(good)
        bad[field] = True
        with pytest.raises(ValueError, match="does not authorize"):
            runner.validate_config(bad)
    for field in ("requests", "denoise_calls", "fit_groups", "physics_steps", "fresh_cases"):
        bad = copy.deepcopy(good)
        bad["budget"][field] += 1
        with pytest.raises(ValueError, match="budget"):
            runner.validate_config(bad)
    bad = copy.deepcopy(good)
    bad["noise_roster"]["training_seeds"] = [42]
    with pytest.raises(ValueError, match="noise rosters"):
        runner.validate_config(bad)


def test_preparation_binds_sources_inputs_original_requests_and_weight_stat(runner, prepared):
    data, reference = prepared
    assert runner.read_prepared(reference) == data
    changed = copy.deepcopy(data)
    changed["requests3"][0]["request"]["future_state"] = [1, 2, 3]
    altered = dump(Path(data["root"]) / "altered.json", changed)
    with pytest.raises(ValueError, match="request/image roster"):
        runner.read_prepared(altered)
    source = Path(next(iter(data["snapshot_sha256"])))
    source.write_text("# changed after freezing\n")
    with pytest.raises(ValueError, match="frozen source/input changed"):
        runner.read_prepared(reference)


def bind_config_and_plan(runner, data):
    root = Path(data["root"])
    config_ref = dump(runner.REPO / "configs/continuation.json", data["config"])
    plan = runner.REPO / "plans/continuation.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Fixed continuation: seven requests, no fitting\n")
    data.update(config_ref=config_ref, plan_ref=runner.reference(plan))
    for name in ("config_ref", "plan_ref"):
        bound = data[name]
        source = Path(bound["path"])
        snapshot = root / "source/s2a" / source.relative_to(runner.REPO)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        data["source_sha256"][str(source)] = bound["sha256"]
        data["snapshot_sha256"][str(snapshot)] = bound["sha256"]
    return dump(root / "preparation/bound.json", data)


def test_bound_preparation_uses_its_config_without_replacing_legacy_default(runner, prepared):
    data, _ = prepared
    reference = bind_config_and_plan(runner, data)
    # A separate run remains tied to its own bytes, even when the legacy
    # default config now describes an unrelated output. Neither is rewritten.
    legacy = json.loads(runner.CONFIG.read_text())
    legacy["output_dir"] = "outputs/unrelated"
    dump(runner.CONFIG, legacy)
    del data["source_sha256"][str(runner.CONFIG)]
    reference = dump(Path(data["root"]) / "preparation/bound.json", data)
    assert runner.read_prepared(reference) == data
    assert json.loads(runner.CONFIG.read_text()) == legacy


@pytest.mark.parametrize("changed", ("config_bytes", "plan_bytes", "config_source", "plan_snapshot", "partial"))
def test_bound_preparation_rejects_replaced_bytes_or_missing_source_binding(runner, prepared, changed):
    data, _ = prepared
    reference = bind_config_and_plan(runner, data)
    if changed in ("config_bytes", "plan_bytes"):
        name = "config_ref" if changed == "config_bytes" else "plan_ref"
        source = Path(data[name]["path"])
        # Whitespace leaves config JSON semantics identical; SHA identity still
        # rejects replacing a registered source with equivalent content.
        source.write_bytes(source.read_bytes() + b"\n")
    else:
        if changed == "config_source":
            del data["source_sha256"][data["config_ref"]["path"]]
        elif changed == "plan_snapshot":
            snapshot = Path(data["root"]) / "source/s2a" / Path(data["plan_ref"]["path"]).relative_to(runner.REPO)
            del data["snapshot_sha256"][str(snapshot)]
        else:
            del data["plan_ref"]
        reference = dump(Path(data["root"]) / "preparation/bound.json", data)
    with pytest.raises(ValueError, match="SHA256|plan changed|source or snapshot|bound together"):
        runner.read_prepared(reference)


@pytest.mark.parametrize("explicit", (False, True))
def test_prepare_freezes_selected_config_plan_and_never_overwrites_a_run(runner, prepared, monkeypatch, explicit):
    data, _ = prepared
    legacy_bytes = runner.CONFIG.read_bytes()
    root = runner.REPO / "outputs/new_bridge"
    config = copy.deepcopy(data["config"])
    config.update(
        output_dir="outputs/new_bridge", cosmos_source_root=str(root / "source/cosmos"), model_python=str(runner.CONFIG)
    )
    config_path = runner.REPO / "configs/new_bridge.json"
    dump(config_path, config)
    plan_path = runner.REPO / "plans/new_bridge.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# New independent qualification\n")
    owner = runner.REPO / "scripts/run_wam_clean_bridge.py"
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text("# synthetic frozen owner\n")
    (runner.REPO / "pyproject.toml").write_text("# synthetic project\n")
    monkeypatch.setattr(runner, "__file__", str(owner))
    monkeypatch.setattr(runner, "cpu_owner", lambda: None)
    if not explicit:
        monkeypatch.setattr(runner, "CONFIG", config_path)
        monkeypatch.setattr(runner, "PLAN", plan_path)

    def archive(config, destination):
        paths = destination / "source/cosmos/cosmos_framework/model/generator/diffusion/samplers"
        paths.mkdir(parents=True)
        files = [paths / name for name in ("unipc.py", "fm_solvers_unipc.py")]
        for path in files:
            path.write_text("# synthetic frozen sampler\n")
        return {str(path): sha256(path) for path in files}, dict(synthetic=True)

    monkeypatch.setattr(runner, "freeze_cosmos_source", archive)
    kwargs = dict(config_path=config_path, plan_path=plan_path) if explicit else {}
    reference = runner.prepare(**kwargs)
    actual = runner.read_prepared(reference)
    assert actual["config_ref"] == runner.reference(config_path)
    assert actual["plan_ref"] == runner.reference(plan_path)
    assert actual["config"] == config and actual["config"]["budget"] == runner.BUDGET
    for path in (config_path, plan_path):
        snapshot = root / "source/s2a" / path.relative_to(runner.REPO)
        assert snapshot.read_bytes() == path.read_bytes()
    if explicit:
        assert runner.CONFIG.read_bytes() == legacy_bytes
    manifest_bytes = Path(reference["path"]).read_bytes()
    with pytest.raises(ValueError, match="refuses existing"):
        runner.prepare(**kwargs)
    assert Path(reference["path"]).read_bytes() == manifest_bytes


def test_weight_stat_change_and_missing_actual_owner_rejected(runner, prepared):
    data, reference = prepared
    changed = copy.deepcopy(data)
    del changed["source_sha256"][str(Path(runner.__file__))]
    altered = dump(Path(data["root"]) / "without_owner.json", changed)
    with pytest.raises(ValueError, match="actual owner"):
        runner.read_prepared(altered)
    shard = next(Path(p) for p in data["weight_files"] if p.endswith(".distcp"))
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="weight stat"):
        runner.read_prepared(reference)


def test_dry_run_uses_frozen_source_uuid_venv_and_has_no_process_or_gpu_observation(runner, prepared, monkeypatch):
    data, reference = prepared
    monkeypatch.setattr(runner, "cpu_owner", lambda: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not start children or inspect GPUs")

    monkeypatch.setattr(runner, "owned_command", forbidden)
    monkeypatch.setattr(runner, "resource_observation", forbidden)
    actual = runner.execute(reference, dry_run=True)
    assert actual["new_model_calls"] == 0 and actual["worker"][0] == data["config"]["model_python"]
    assert actual["model_environment"]["CUDA_VISIBLE_DEVICES"] == data["config"]["gpu_uuid"]
    assert actual["model_environment"]["PYTHONPATH"].startswith(data["config"]["cosmos_source_root"] + ":")
    assert "fit" not in actual["worker"] and not (Path(data["root"]) / "orchestration").exists()
    assert runner.environment(data, gpu=False)["CUDA_VISIBLE_DEVICES"] == ""


def test_prior_qualification_output_never_restarts(runner, prepared, monkeypatch):
    data, reference = prepared
    (Path(data["root"]) / "qualification").mkdir()
    monkeypatch.setattr(runner, "cpu_owner", lambda: None)
    monkeypatch.setattr(runner, "resource_observation", lambda *args: pytest.fail("prior stage must reject before GPU"))
    with pytest.raises(ValueError, match="refuses prior stage"):
        runner.execute(reference)


def test_busy_uuid_records_failed_owner_without_spawning_or_killing(runner, prepared, monkeypatch):
    data, reference = prepared
    monkeypatch.setattr(runner, "cpu_owner", lambda: None)
    monkeypatch.setattr(
        runner,
        "resource_observation",
        lambda uuid: dict(uuid=uuid, memory_mib=100, compute_apps=["other"], utilization_gpu=20),
    )
    monkeypatch.setattr(runner, "owned_command", lambda *args, **kwargs: pytest.fail("occupied GPU must not spawn"))
    with pytest.raises(ValueError, match="occupied"):
        runner.execute(reference)
    owner = json.loads((Path(data["root"]) / "orchestration/result.json").read_text())
    assert owner["passed"] is owner["complete"] is False and owner["owned_commands"] == []
    assert owner["automatic_retry"] is False and owner["fresh_performance_authorized"] is False


@pytest.mark.parametrize(
    "uuid", tuple(("GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f", "GPU-7a163b30-e2cd-f747-6075-3265058ad0b8"))
)
def test_driver_minor_visible_index_pci_and_uuid_must_all_match(runner, tmp_path, monkeypatch, uuid):
    visible, minor = runner.GPU_IDENTITIES[uuid]
    info = tmp_path / "0000:3b:00.0/information"
    info.parent.mkdir()
    info.write_text(f"GPU UUID: {uuid}\nDevice Minor: {minor}\nBus Location: 0000:3b:00.0\n")

    def query(command, **kwargs):
        assert f"--id={uuid}" in command
        if any("query-compute-apps" in arg for arg in command):
            return ""
        return f"{visible}, {uuid}, 00000000:3B:00.0, 0, 0\n"

    monkeypatch.setattr(runner.subprocess, "check_output", query)
    observed = runner.resource_observation(uuid, driver_root=tmp_path)
    assert observed["device_minor"] == minor and observed["visible_gpu"] == visible and observed["memory_mib"] == 0
    info.write_text(f"GPU UUID: {uuid}\nDevice Minor: 0\nBus Location: 0000:3b:00.0\n")
    with pytest.raises(ValueError, match="device minor"):
        runner.resource_observation(uuid, driver_root=tmp_path)
    with pytest.raises(ValueError, match="allowed UUID"):
        runner.resource_observation("GPU-forbidden", driver_root=tmp_path)


def test_release_timeout_contract_and_normal_release_receipt(runner, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="at most60"):
        runner.wait_release(tmp_path, next(iter(runner.GPU_IDENTITIES)), timeout=61)
    monkeypatch.setattr(runner, "LOCATOR", RelocatedArtifacts(tmp_path, tmp_path / "recovery"))
    monkeypatch.setattr(runner, "resource_observation", lambda uuid: dict(uuid=uuid, memory_mib=0, compute_apps=[]))
    result = runner.read(runner.wait_release(tmp_path, next(iter(runner.GPU_IDENTITIES))))
    assert result["released"] is True and result["seconds_since_actual_owned_wait"] >= 0


def test_git_archive_is_local_complete_source_and_rejects_links(runner, tmp_path, monkeypatch):
    # This writes only a tiny synthetic archive under pytest outputs; no actual
    # 035 prepare, git checkout, external source write, or model is performed.
    config = dict(source_repository="synthetic")
    monkeypatch.setattr(
        runner, "external_identity", lambda _: dict(head=runner.SOURCE_COMMIT, tracked_source_clean=True)
    )
    monkeypatch.setattr(runner, "LOCATOR", RelocatedArtifacts(tmp_path, tmp_path / "recovery"))

    def archive(command, check):
        assert check is True and "archive" in command
        target = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output=")))
        with tarfile.open(target, "w") as stream:
            info = tarfile.TarInfo("cosmos_framework/example.py")
            content = b"# frozen\n"
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))

    monkeypatch.setattr(runner.subprocess, "run", archive)
    root = tmp_path / "normal"
    root.mkdir()
    sources, evidence = runner.freeze_cosmos_source(config, root)
    assert len(sources) == 1 and evidence["archive_ref"]["path"] == str(root / "source/cosmos.tar")
    assert (root / "source/cosmos/cosmos_framework/example.py").read_bytes() == b"# frozen\n"

    def link_archive(command, check):
        target = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output=")))
        with tarfile.open(target, "w") as stream:
            info = tarfile.TarInfo("cosmos_framework/unsafe.py")
            info.type, info.linkname = tarfile.SYMTYPE, "/outside"
            stream.addfile(info)

    monkeypatch.setattr(runner.subprocess, "run", link_archive)
    other = tmp_path / "unsafe"
    other.mkdir()
    with pytest.raises(ValueError, match="link or unsafe"):
        runner.freeze_cosmos_source(config, other)
