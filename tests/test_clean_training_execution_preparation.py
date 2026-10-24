"""CPU-only040 preparation contracts; every receipt/data file here is synthetic."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from s2a import clean_training_seed_plan as SEEDS
from s2a.artifacts import REPO


@pytest.fixture
def prepare(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    module = importlib.import_module("prepare_wam_clean_training_execution")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    for name, value in module.THREADS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("execution preparation decoded future arrays"))
    monkeypatch.setattr(module, "resource_observation", lambda *a, **k: pytest.fail("unregistered real GPU query"))
    return module


def write(path, value, module):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return module.reference(path)


@pytest.mark.parametrize("fault", [None, "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "cwd"])
def test_preparation_requires_the_registered_cpu_environment(prepare, monkeypatch, tmp_path, fault):
    if fault in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(fault, "unregistered")
    elif fault == "OMP_NUM_THREADS":
        monkeypatch.setenv(fault, "4")
    elif fault == "cwd":
        monkeypatch.chdir(tmp_path)
    if fault is None:
        prepare.cpu_environment()
    else:
        with pytest.raises(ValueError):
            prepare.cpu_environment()


@pytest.fixture
def repository(tmp_path, monkeypatch, prepare):
    root = tmp_path / "synthetic_repository"
    root.mkdir()
    monkeypatch.setattr(prepare, "REPO", root)
    for name in ("src/s2a/algorithm.py", *set(prepare.STAGES.values())):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic source, never executable model/physics.\n")
    config = json.loads((REPO / "configs/wam_clean_training_execution_20261023.json").read_text())
    for name in config["required_test_paths"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic test inventory entry.\n")
    for name in (
        "configs/wam_clean_training_execution_20261023.json",
        prepare.PLAN,
        "configs/wam_clean_training_20261023.json",
        "plans/v0.0.1/038_clean_residual_training_20261023.md",
        "pyproject.toml",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic source inventory entry.\n")
    return root, config


def test_source_inventory_includes_every_python_file_and_registered_nonpython_inputs(repository, prepare):
    root, _ = repository
    ignored = root / "outputs/old_capture/source/ignored.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("# Generated evidence is outside the source inventory.\n")
    (root / "src/ignored.txt").write_text("not executable source")
    paths = prepare.source_paths()
    assert paths == sorted(set(paths))
    assert set(root.glob("tests/*.py")) <= set(paths)
    assert {root / "pyproject.toml", root / prepare.PLAN, root / "src/s2a/algorithm.py"} <= set(paths)
    assert ignored not in paths and root / "src/ignored.txt" not in paths
    added = root / "scripts/added_after_qa.py"
    added.write_text("# Newly executable source must force a new inventory and QA SHA.\n")
    assert set(prepare.source_paths()) == set(paths) | {added}


@pytest.mark.parametrize("fault", ["missing_plan", "outside_symlink", "outputs_symlink"])
def test_source_inventory_refuses_missing_or_escaped_executable_sources(repository, tmp_path, prepare, fault):
    root, _ = repository
    if fault == "missing_plan":
        (root / prepare.PLAN).unlink()
    else:
        destination = tmp_path / "outside.py" if fault == "outside_symlink" else root / "outputs/source.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("# May not be frozen as a repository source.\n")
        (root / "scripts/escaped.py").symlink_to(destination)
    with pytest.raises(ValueError, match="source is missing or escaped"):
        prepare.source_paths()


def test_configuration_keeps_the_exact_gpu_resource_and19_module_contract(repository, prepare):
    _, config = repository
    prepare.check_config(config)
    assert config["disk_required_bytes"] == 200 * 1024**3
    assert config["ram_required_bytes"] == 128 * 1024**3
    assert len(config["required_test_paths"]) == 19


@pytest.mark.parametrize(
    "fault",
    [
        "seed",
        "schema",
        "plan",
        "root",
        "disk",
        "ram",
        "gpu",
        "master",
        "development",
        "retry",
        "fresh",
        "missing_test",
        "duplicate_test",
    ],
)
def test_configuration_refuses_relaxed_limits_or_changed_identity(repository, prepare, fault):
    root, config = repository
    if fault == "seed":
        config["seed"] = 43
    elif fault == "schema":
        config["schema_version"] = "other"
    elif fault == "plan":
        config["plan_path"] = "plans/unregistered.md"
    elif fault == "root":
        config["output_dir"] += "_retry"
    elif fault in ("disk", "ram"):
        config[fault + "_required_bytes"] -= 1
    elif fault == "gpu":
        config["gpu_uuid"] = "GPU-forbidden"
    elif fault == "master":
        config["master_port"] += 1
    elif fault == "development":
        config["development_port"] -= 1
    elif fault == "retry":
        config["automatic_retry"] = True
    elif fault == "fresh":
        config["fresh_performance_authorized"] = True
    elif fault == "missing_test":
        (root / config["required_test_paths"][-1]).unlink()
    else:
        config["required_test_paths"][-1] = config["required_test_paths"][0]
    with pytest.raises(ValueError):
        prepare.check_config(config)


@pytest.mark.parametrize(
    "host,maximum,current,expected",
    [(200, None, 99, 200), (200, 300, 100, 200), (200, 240, 100, 140), (100, 300, 100, 100), (200, 100, 100, 0)],
)
def test_effective_memory_uses_the_smaller_host_or_cgroup_headroom(prepare, host, maximum, current, expected):
    assert prepare.effective_memory(host, maximum, current) == expected


@pytest.mark.parametrize(
    "values", [(-1, None, 0), (1, 2, 3), (1, None, -1), (True, None, 0), (1, 2.0, 0), (1, None, 0.0)]
)
def test_invalid_memory_observations_cannot_admit_execution(prepare, values):
    with pytest.raises(ValueError, match="invalid live memory"):
        prepare.effective_memory(*values)


@pytest.mark.parametrize(
    "fault", [None, "disk", "cgroup_headroom", "cpu", "threads", "gpu_memory", "gpu_apps", "gpu_busy"]
)
def test_resource_gate_uses_effective_ram_and_all_fixed_resources(tmp_path, prepare, monkeypatch, fault):
    mem = dict(host=256 * 1024**3, maximum=512 * 1024**3, current=128 * 1024**3)
    free, cpus = prepare.DISK_REQUIRED, 8
    gpu = dict(
        uuid=prepare.GPU,
        memory_mib=0,
        compute_apps=[],
        utilization_gpu=0,
        pci_bus_id="00000000:C1:00.0",
        device_minor=6,
        visible_gpu=2,
    )
    if fault == "disk":
        free -= 1
    elif fault == "cgroup_headroom":
        mem["current"] = mem["maximum"] - prepare.RAM_REQUIRED + 1
    elif fault == "cpu":
        cpus = 7
    elif fault == "threads":
        monkeypatch.setenv("MKL_NUM_THREADS", "16")
    elif fault == "gpu_memory":
        gpu["memory_mib"] = 1
    elif fault == "gpu_apps":
        gpu["compute_apps"] = [dict(pid=123)]
    elif fault == "gpu_busy":
        gpu["utilization_gpu"] = 1
    original_read = Path.read_text
    texts = {
        "/proc/meminfo": f"MemAvailable: {mem['host'] // 1024} kB\n",
        "/sys/fs/cgroup/memory.max": str(mem["maximum"]),
        "/sys/fs/cgroup/memory.current": str(mem["current"]),
    }

    def read(path, *args, **kwargs):
        return texts[str(path)] if str(path) in texts else original_read(path, *args, **kwargs)

    bound = []

    class Socket:
        def __enter__(self):
            return self

        def bind(self, address):
            bound.append(address)

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda root: SimpleNamespace(free=free))
    monkeypatch.setattr(prepare.os, "sched_getaffinity", lambda pid: set(range(cpus)))
    monkeypatch.setattr(prepare.socket, "socket", Socket)
    monkeypatch.setattr(prepare, "resource_observation", lambda uuid: copy.deepcopy(gpu))
    if fault is None:
        value = prepare.observe_resources(tmp_path)
        assert value["complete"] is value["passed"] is True
        assert value["MemAvailable"] == mem["host"]
        assert value["disk_free"] == prepare.DISK_REQUIRED
        assert bound == [("127.0.0.1", 31338), ("127.0.0.1", 31339)]
    else:
        with pytest.raises(ValueError, match="complete budget"):
            prepare.observe_resources(tmp_path)


@pytest.fixture
def normalization(tmp_path, prepare):
    domain = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
    transform = dict(
        normalization_mask=[True] * 3 + [False] * 125, scale=[1.25] * 128, bias=[0.1] * 128, clip_min=-1.0, clip_max=1.0
    )
    spec = dict(
        registry=dict(domains=["unrelated"] * 67, domain_projection_ids=[7] * 67),
        transforms=[copy.deepcopy(transform) for _ in range(67)],
    )
    spec["registry"]["domains"][31] = domain
    spec["registry"]["domain_projection_ids"][31] = 0
    spec["transforms"][0]["scale"][0] = 1000
    return tmp_path / "model_config.yaml", spec, domain


@pytest.mark.parametrize(
    "fault", [None, "dictionary", "short", "domain", "projection", "interpolation", "mask", "extra_field"]
)
def test_normalization_routes_the_real_transform_list31_and_refuses_wrong_contract(prepare, normalization, fault):
    path, spec, domain = normalization
    if fault == "dictionary":
        spec["transforms"] = {domain: spec["transforms"][31]}
    elif fault == "short":
        spec["transforms"] = spec["transforms"][:31]
    elif fault == "domain":
        spec["registry"]["domains"][31] = "wrong-domain"
    elif fault == "projection":
        spec["registry"]["domain_projection_ids"][31] = 1
    elif fault == "interpolation":
        spec["registry"]["domains"][0] = "${unresolved}"
    elif fault == "mask":
        spec["transforms"][31]["normalization_mask"][9] = True
    elif fault == "extra_field":
        spec["transforms"][31]["future_state"] = [1]
    path.write_text(yaml.safe_dump(dict(model=dict(config=dict(action128_spec=spec)))))
    if fault is None:
        actual = prepare.normalization(prepare.reference(path), prepare.Evidence())
        assert actual["normalization_id"] == 31 and actual["projection_id"] == 0
        assert actual["transform"] == spec["transforms"][31]
        assert actual["transform"]["scale"][0] == 1.25
    else:
        with pytest.raises(ValueError):
            prepare.normalization(prepare.reference(path), prepare.Evidence())


def test_sources_are_copied_exclusively_and_existing_snapshots_never_overwritten(repository, prepare):
    root, _ = repository
    source = root / "src/s2a/algorithm.py"
    values = {str(source): prepare.sha(source)}
    output = root / "outputs/execution"
    snapshots, refs = prepare.freeze_sources(output, values)
    target = output / "source/src/s2a/algorithm.py"
    assert snapshots == {str(target): values[str(source)]}
    assert refs[str(source)] == prepare.reference(target)
    assert target.read_bytes() == source.read_bytes()
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        prepare.freeze_sources(output, values)
    assert target.read_bytes() == original


def test_changed_source_is_rejected_before_snapshot_creation(repository, prepare):
    root, _ = repository
    source = root / "src/s2a/algorithm.py"
    values = {str(source): prepare.sha(source)}
    source.write_text("# Changed after final QA.\n")
    output = root / "outputs/execution"
    with pytest.raises(ValueError, match="before snapshot"):
        prepare.freeze_sources(output, values)
    assert not output.exists()


def test_opaque_future_can_only_be_hashed_and_drift_is_detected(tmp_path, prepare):
    path = tmp_path / "opaque_future.npz"
    path.write_bytes(b"Intentionally invalid archive: only byte hashing is permitted.")
    evidence = prepare.Evidence()
    ref = prepare.reference(path)
    assert evidence.check(ref) == path
    with pytest.raises(ValueError, match="binary future"):
        evidence.json(ref)
    path.write_bytes(b"changed while preparing")
    with pytest.raises(ValueError, match="input changed"):
        evidence.finish()


def test_existing_execution_attempt_is_refused_before_any_retry_or_new_receipts(repository, prepare, monkeypatch):
    root, config = repository
    monkeypatch.chdir(root)
    original_ref = write(
        root / "outputs/original_prepared.json",
        dict(
            schema_version="s2a.clean-training-preparation.v1",
            execution_authorized=False,
            readiness=dict(ready_for_execution=False),
        ),
        prepare,
    )
    config["prepared_ref"] = original_ref
    config_ref = write(root / "configs/execution.json", config, prepare)
    output = root / config["output_dir"]
    output.mkdir(parents=True)
    sentinel = output / "original_failed_attempt.json"
    sentinel.write_text('{"passed":false}')
    before = sentinel.read_bytes()
    with pytest.raises(ValueError, match="existing040 proposal/attempt or retry"):
        prepare.prepare(config_ref, dict(path="/must_not_read_qa.json", sha256="f" * 64))
    assert list(output.iterdir()) == [sentinel] and sentinel.read_bytes() == before


@pytest.fixture
def proof(repository, prepare, monkeypatch):
    root, _ = repository
    monkeypatch.chdir(root)
    output = root / "outputs/wam_stability_20261021/clean_residual_training_20261023/execution"
    output.mkdir(parents=True)
    verifier = root / "scripts/synthetic_independent_verifier.py"
    verifier.write_text("# Fixture only; no actual qualification or model execution.\n")
    source = root / "src/s2a/algorithm.py"
    target = output / "source/src/s2a/algorithm.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    prepared_ref = write(root / "outputs/prepared.json", dict(synthetic=True), prepare)
    proposal = dict(
        schema_version="s2a.clean-training-execution-proposal.v1",
        complete=True,
        passed=True,
        readiness=dict(ready_for_execution=False, reason="independent audit still required"),
        execution_authorized=False,
        automatic_retry=False,
        fresh_performance_authorized=False,
        prepared_ref=prepared_ref,
        budget=dict(synthetic_fixed_budget=99),
        source_sha256={str(p): prepare.sha(p) for p in (source, verifier)},
        snapshot_sha256={str(target): prepare.sha(target)},
        input_sha256={prepared_ref["path"]: prepared_ref["sha256"]},
    )
    proposal_ref = write(output / "proposal.json", proposal, prepare)
    verification = dict(
        schema_version="s2a.clean-training-execution-verification.v1",
        complete=True,
        passed=True,
        eligible_for_execution=True,
        prepared_ref=prepared_ref,
        proposal_ref=proposal_ref,
        verifier_ref=prepare.reference(verifier),
    )
    verification_ref = write(output / "verified.json", verification, prepare)
    return proposal_ref, verification_ref, proposal, verification


def test_admission_only_seals_the_exact_independent_proposal_and_never_overwrites(proof, prepare):
    proposal_ref, verification_ref, proposal, _ = proof
    original_bytes = {ref["path"]: Path(ref["path"]).read_bytes() for ref in (proposal_ref, verification_ref)}
    admitted_ref = prepare.admit(proposal_ref, verification_ref)
    admitted = prepare.read(admitted_ref)
    expected = copy.deepcopy(proposal)
    expected.update(
        schema_version="s2a.clean-training-execution-registration.v1",
        execution_authorized=True,
        proposal_ref=proposal_ref,
        execution_verification_ref=verification_ref,
    )
    expected["readiness"]["ready_for_execution"] = True
    assert admitted == expected
    before = Path(admitted_ref["path"]).read_bytes()
    with pytest.raises(FileExistsError):
        prepare.admit(proposal_ref, verification_ref)
    assert Path(admitted_ref["path"]).read_bytes() == before
    assert all(Path(path).read_bytes() == value for path, value in original_bytes.items())


@pytest.mark.parametrize(
    "fault", ["failed_audit", "wrong_proposal", "unfrozen_verifier", "source", "snapshot", "input", "wrong_proof_path"]
)
def test_admission_refuses_stale_or_unrelated_proof_without_publishing_registration(proof, prepare, fault):
    proposal_ref, verification_ref, proposal, verification = proof
    root = Path(proposal_ref["path"]).parent
    if fault == "failed_audit":
        verification["passed"] = False
    elif fault == "wrong_proposal":
        verification["proposal_ref"] = dict(proposal_ref, sha256="f" * 64)
    elif fault == "unfrozen_verifier":
        verification["verifier_ref"] = dict(path="/unfrozen.py", sha256="e" * 64)
    elif fault in ("source", "snapshot", "input"):
        key = {"source": "source_sha256", "snapshot": "snapshot_sha256", "input": "input_sha256"}[fault]
        Path(next(iter(proposal[key]))).write_text("changed after independent proof")
    if fault in ("failed_audit", "wrong_proposal", "unfrozen_verifier"):
        verification_ref = write(Path(verification_ref["path"]), verification, prepare)
    elif fault == "wrong_proof_path":
        verification_ref = write(root / "unregistered_verification.json", verification, prepare)
    with pytest.raises(ValueError):
        prepare.admit(proposal_ref, verification_ref)
    assert not (root / "admitted.json").exists()


@pytest.fixture
def seed_case(prepare):
    anchors = [dict(name=f"任务_anchor{i:03d}", untouched=dict(index=i)) for i in range(150)]
    requests = []
    for anchor in anchors:
        for root in (101, 102):
            index = len(requests)
            old = (1 << 62) + index * 17
            requests.append(
                dict(
                    index=index,
                    name=f"train_{index:03d}",
                    anchor_name=anchor["name"],
                    root_seed=root,
                    sampling_seed=old,
                    request=dict(seed=old, nested=[dict(current_only=True)]),
                )
            )
    prepared = dict(
        training_anchors=anchors,
        training_requests=requests,
        noise_roster=dict(training_seeds=[r["sampling_seed"] for r in requests], evaluation_seeds=[42, 43]),
        noise_seed_algorithm="original immutable63-bit identity algorithm",
        source_sha256={"/original/source.py": "c" * 64},
        input_sha256={"/opaque/future.npz": "d" * 64},
        prerequisite_refs=dict(original_ref=dict(path="/original/prepared.json", sha256="e" * 64)),
        budget=dict(original_rows=840000),
        execution_authorized=False,
    )
    return dict(path="/synthetic/prepared.json", sha256="a" * 64), prepared


def test_all300_seed_plan_identities_follow_the_exact_ascii_sha256_algorithm(seed_case):
    prepared_ref, prepared = seed_case
    before = json.dumps(prepared, sort_keys=True)
    plan = SEEDS.build_seed_plan(prepared_ref, prepared)
    assert plan["schema_version"] == "s2a.clean-training-seed-plan.v1"
    assert plan["complete"] is plan["passed"] is True
    assert plan["algorithm"] == SEEDS.ALGORITHM and plan["prepared_ref"] == prepared_ref
    assert plan["actual_seed_bits"] == 31 and plan["original_seed_bits"] == 63
    assert plan["model_calls"] == plan["fits"] == plan["physics_steps"] == 0
    assert plan["evaluation_seeds"] == [42, 43] and len(plan["entries"]) == 300
    for old, actual in zip(prepared["training_requests"], plan["entries"], strict=True):
        payload = json.dumps(
            ["s2a.clean-training-request-seed31.v1", old["anchor_name"], old["root_seed"], old["sampling_seed"]],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        assert b"\\u" in payload
        digest = hashlib.sha256(payload).digest()
        assert actual == dict(
            index=old["index"],
            name=old["name"],
            anchor_name=old["anchor_name"],
            root_seed=old["root_seed"],
            original_sampling_seed=old["sampling_seed"],
            sampling_seed=int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
            payload_sha256=digest.hex(),
        )
    seeds = [e["sampling_seed"] for e in plan["entries"]]
    assert len(set(seeds)) == 300 and all(0 <= s < 2**31 and s not in (42, 43) for s in seeds)
    assert any(e["sampling_seed"] != (e["original_sampling_seed"] & 0x7FFFFFFF) for e in plan["entries"])
    assert SEEDS.build_seed_plan(prepared_ref, prepared) == plan
    assert json.dumps(prepared, sort_keys=True) == before


def test_seed_overlay_changes_only_the_registered_fields_in_a_deep_copy(seed_case):
    prepared_ref, prepared = seed_case
    plan = SEEDS.build_seed_plan(prepared_ref, prepared)
    plan_ref = dict(path="/registered/seed_plan.json", sha256="b" * 64)
    before = json.dumps([prepared_ref, prepared, plan, plan_ref], sort_keys=True)
    effective = SEEDS.apply_seed_plan(prepared, prepared_ref, plan, plan_ref)
    expected = copy.deepcopy(prepared)
    for entry, mapping in zip(expected["training_requests"], plan["entries"], strict=True):
        entry["sampling_seed"] = entry["request"]["seed"] = mapping["sampling_seed"]
    expected["noise_roster"]["training_seeds"] = [e["sampling_seed"] for e in plan["entries"]]
    expected["noise_seed_algorithm"] = SEEDS.ALGORITHM
    expected["training_seed_plan_ref"] = copy.deepcopy(plan_ref)
    assert effective == expected
    assert json.dumps([prepared_ref, prepared, plan, plan_ref], sort_keys=True) == before
    effective["training_requests"][0]["request"]["nested"][0]["current_only"] = False
    effective["source_sha256"]["/original/source.py"] = "changed returned copy"
    assert prepared["training_requests"][0]["request"]["nested"][0]["current_only"] is True
    assert prepared["source_sha256"]["/original/source.py"] == "c" * 64
    with pytest.raises(ValueError, match="original unmodified"):
        SEEDS.build_seed_plan(prepared_ref, effective)


@pytest.mark.parametrize(
    "fault",
    [
        "requests",
        "anchors",
        "order",
        "root",
        "index",
        "request_seed",
        "old_eval",
        "old_range",
        "old_duplicate",
        "noise_order",
        "eval_roster",
        "typed_eval",
        "anchor",
        "name",
    ],
)
def test_seed_plan_refuses_changed_or_incomplete_original_roster(seed_case, fault):
    prepared_ref, prepared = seed_case
    entry = prepared["training_requests"][0]
    if fault == "requests":
        prepared["training_requests"].pop()
    elif fault == "anchors":
        prepared["training_anchors"].pop()
    elif fault == "order":
        prepared["training_requests"][0], prepared["training_requests"][1] = prepared["training_requests"][1], entry
    elif fault == "root":
        entry["root_seed"] = 103
    elif fault == "index":
        entry["index"] = 0.0
    elif fault == "request_seed":
        entry["request"]["seed"] += 1
    elif fault in ("old_eval", "old_range", "old_duplicate"):
        seed = {"old_eval": 42, "old_range": 2**63, "old_duplicate": prepared["training_requests"][1]["sampling_seed"]}[
            fault
        ]
        entry["sampling_seed"] = entry["request"]["seed"] = prepared["noise_roster"]["training_seeds"][0] = seed
    elif fault == "noise_order":
        prepared["noise_roster"]["training_seeds"].reverse()
    elif fault == "eval_roster":
        prepared["noise_roster"]["evaluation_seeds"] = [42, 44]
    elif fault == "typed_eval":
        prepared["noise_roster"]["evaluation_seeds"] = [42.0, 43.0]
    elif fault == "anchor":
        entry["anchor_name"] = "unregistered anchor"
    elif fault == "name":
        entry["name"] = prepared["training_requests"][1]["name"]
    with pytest.raises(ValueError):
        SEEDS.build_seed_plan(prepared_ref, prepared)


@pytest.mark.parametrize(
    "fault",
    [
        "actual_seed",
        "original_seed",
        "payload",
        "algorithm",
        "prepared_ref",
        "evaluation",
        "complete",
        "missing_entry",
        "extra_field",
        "bits_type",
    ],
)
def test_seed_overlay_refuses_any_tampered_plan_even_if_integer_values_compare_equal(seed_case, fault):
    prepared_ref, prepared = seed_case
    plan = SEEDS.build_seed_plan(prepared_ref, prepared)
    if fault == "actual_seed":
        plan["entries"][0]["sampling_seed"] += 1
    elif fault == "original_seed":
        plan["entries"][0]["original_sampling_seed"] += 1
    elif fault == "payload":
        plan["entries"][0]["payload_sha256"] = "f" * 64
    elif fault == "algorithm":
        plan["algorithm"] = "truncate original63 to31"
    elif fault == "prepared_ref":
        plan["prepared_ref"]["sha256"] = "e" * 64
    elif fault == "evaluation":
        plan["evaluation_seeds"] = [42, 44]
    elif fault == "complete":
        plan["complete"] = 1
    elif fault == "missing_entry":
        plan["entries"].pop()
    elif fault == "extra_field":
        plan["automatic_retry"] = True
    elif fault == "bits_type":
        plan["actual_seed_bits"] = 31.0
    with pytest.raises(ValueError, match="exact original300"):
        SEEDS.apply_seed_plan(prepared, prepared_ref, plan, dict(path="/registered/plan.json", sha256="b" * 64))


@pytest.mark.parametrize("actual_seed,expected_calls", [(42, 1), (43, 1), (7, 2)])
def test_seed_collision_or_evaluation_overlap_stops_without_retry(seed_case, monkeypatch, actual_seed, expected_calls):
    prepared_ref, prepared = seed_case
    calls = []
    digest = actual_seed.to_bytes(4, "big") + b"x" * 28

    def forced(payload):
        calls.append(payload)
        return SimpleNamespace(digest=lambda: digest)

    monkeypatch.setattr(SEEDS.hashlib, "sha256", forced)
    with pytest.raises(ValueError, match="collision/evaluation overlap; no retry"):
        SEEDS.build_seed_plan(prepared_ref, prepared)
    assert len(calls) == expected_calls


def test_actual_b55_preparation_builds_a_complete_seed_plan_without_future_decoding_or_rewrite(prepare):
    path = REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023/preparation/prepared.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "b55c214ffec328ee94f2074007c15dfe62486512aeb660b251fdb21f38e970f7"
    prepared = json.loads(raw)
    original = copy.deepcopy(prepared)
    plan = SEEDS.build_seed_plan(dict(path=str(path), sha256=digest), prepared)
    assert len(plan["entries"]) == 300 and len({e["sampling_seed"] for e in plan["entries"]}) == 300
    assert all(0 <= e["sampling_seed"] < 2**31 and e["sampling_seed"] not in (42, 43) for e in plan["entries"])
    assert [e["original_sampling_seed"] for e in plan["entries"]] == original["noise_roster"]["training_seeds"]
    assert prepared == original and path.read_bytes() == raw
