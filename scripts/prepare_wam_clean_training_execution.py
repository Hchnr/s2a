"""Prepare040's immutable proposal and seal only an independently audited one.

No model, simulator, future-array decoder or fitting implementation is imported.
The separate verifier audits the proposal before an execution envelope exists.
"""

import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import socket
from pathlib import Path

from run_wam_clean_bridge import resource_observation

from s2a.artifacts import REPO
from s2a.clean_execution_proof import validate_envelope
from s2a.clean_training_runtime import GPU, checked, publish, read, reference, require, sha
from s2a.clean_training_seed_plan import build_seed_plan
from s2a.relocated_artifacts import RelocatedArtifacts

CONFIG = REPO / "configs/wam_clean_training_execution_20261023.json"
PLAN = "plans/v0.0.1/040_clean_training_execution_20261023.md"
DISK_REQUIRED, RAM_REQUIRED = 200 * 1024**3, 128 * 1024**3
THREADS = dict(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
STAGES = dict(
    D0="scripts/collect_wam_clean_training.py",
    W1="scripts/fit_wam_clean_training.py",
    D1="scripts/collect_wam_clean_training.py",
    W2="scripts/fit_wam_clean_training.py",
    STATE="scripts/sample_wam_clean_training_state.py",
    DEVELOPMENT="scripts/run_wam_clean_training_development.py",
)


def cpu_environment():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("LD_LIBRARY_PATH") == "",
        "execution preparation needs S2A cwd and empty CUDA/LD",
    )
    require(
        all(os.environ.get(k) == v for k, v in THREADS.items()),
        "preparation must register the actual eight-thread environment",
    )


class Evidence:
    def __init__(self):
        self.locator = RelocatedArtifacts(REPO, "outputs/external_deletion_recovery_20261022")
        self.inputs = {}

    def check(self, ref):
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "exact path/SHA input required")
        path = self.locator.resolve_path(ref["path"])
        require(sha(path) == ref["sha256"], "execution preparation input changed")
        self.inputs[str(path)] = ref["sha256"]
        return path

    def json(self, ref):
        path = self.check(ref)
        require(path.suffix == ".json", "metadata reader refuses binary future archives")
        value = path.read_bytes()
        require(hashlib.sha256(value).hexdigest() == ref["sha256"], "metadata changed during read")
        return json.loads(value)

    def mapping(self, values):
        for path, digest in values.items():
            self.check(dict(path=path, sha256=digest))

    def finish(self):
        require(
            all(sha(path) == expected for path, expected in self.inputs.items()),
            "input changed during execution preparation",
        )


def source_paths():
    paths = {p.resolve() for name in ("src", "scripts", "tests") for p in (REPO / name).rglob("*.py")}
    paths.update(
        REPO / p
        for p in (
            "configs/wam_clean_training_execution_20261023.json",
            PLAN,
            "configs/wam_clean_training_20261023.json",
            "plans/v0.0.1/038_clean_residual_training_20261023.md",
            "pyproject.toml",
        )
    )
    require(
        all(p.is_file() and p.is_relative_to(REPO) and not p.is_relative_to(REPO / "outputs") for p in paths),
        "final source is missing or escaped repository source directories",
    )
    return sorted(paths)


def check_config(config):
    require(
        config["schema_version"] == "s2a.clean-training-execution-config.v1" and config["seed"] == 42,
        "040 config/seed differs",
    )
    require(
        config["plan_path"] == PLAN
        and config["output_dir"] == "outputs/wam_stability_20261021/clean_residual_training_20261023/execution",
        "040 plan/output root differs",
    )
    require(
        config["disk_required_bytes"] == DISK_REQUIRED
        and config["ram_required_bytes"] == RAM_REQUIRED
        and config["gpu_uuid"] == GPU
        and config["master_port"] == 31338
        and config["development_port"] == 31339
        and config["automatic_retry"] is config["fresh_performance_authorized"] is False,
        "fixed resource/GPU/port/retry/fresh contract differs",
    )
    require(
        len(config["required_test_paths"]) == len(set(config["required_test_paths"])) == 19
        and all((REPO / p).is_file() and p.startswith("tests/test_") for p in config["required_test_paths"]),
        "complete related19-test-module contract required",
    )


def effective_memory(host_available, maximum, current):
    require(
        type(host_available) is int
        and host_available >= 0
        and type(current) is int
        and current >= 0
        and (maximum is None or type(maximum) is int and maximum >= current),
        "invalid live memory observation",
    )
    return host_available if maximum is None else min(host_available, maximum - current)


def observe_resources(root):
    fields = dict(
        (line.split(":", 1)[0], line.split(":", 1)[1].strip())
        for line in Path("/proc/meminfo").read_text().splitlines()
    )
    host = int(fields["MemAvailable"].split()[0]) * 1024
    maximum_text = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    maximum = None if maximum_text == "max" else int(maximum_text)
    current = int(Path("/sys/fs/cgroup/memory.current").read_text())
    ports = {}
    for port in (31338, 31339):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
            ports[str(port)] = True
    gpu = resource_observation(GPU)
    value = dict(
        schema_version="s2a.clean-training-execution-resources.v1",
        complete=False,
        passed=False,
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        disk_path=str(root),
        disk_free=shutil.disk_usage(root).free,
        required_disk_bytes=DISK_REQUIRED,
        host_MemAvailable=host,
        cgroup_memory_max=maximum,
        cgroup_memory_current=current,
        MemAvailable=effective_memory(host, maximum, current),
        required_ram_bytes=RAM_REQUIRED,
        threads={key: os.environ.get(key) for key in THREADS},
        CPU=len(os.sched_getaffinity(0)),
        GPU={
            key: gpu[key]
            for key in (
                "uuid",
                "memory_mib",
                "compute_apps",
                "utilization_gpu",
                "pci_bus_id",
                "device_minor",
            )
        },
        ports=ports,
        gpu_driver_identity=gpu,
    )
    require(
        value["disk_free"] >= DISK_REQUIRED
        and value["MemAvailable"] >= RAM_REQUIRED
        and value["CPU"] >= 8
        and value["threads"] == THREADS
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["utilization_gpu"] == 0,
        "actual storage/RAM/CPU/GPU resources do not admit the fixed complete budget",
    )
    value.update(complete=True, passed=True)
    return value


def normalization(config_ref, evidence):
    import yaml

    config = yaml.safe_load(evidence.check(config_ref).read_text())
    spec = config["model"]["config"]["action128_spec"]
    require("${" not in json.dumps(spec), "Action128 spec has unresolved interpolation")
    from collect_wam_clean_training import normalization_spec

    actual = normalization_spec(spec)
    require(
        actual["transform"]["normalization_mask"] == [True] * 3 + [False] * 125,
        "published normalization31 mask differs",
    )
    return dict(schema_version="s2a.clean-training-normalization.v1", config_ref=config_ref, **actual)


def freeze_sources(root, values):
    snapshots, refs = {}, {}
    for filename, expected in values.items():
        source = Path(filename)
        require(
            source.is_relative_to(REPO) and not source.is_relative_to(REPO / "outputs") and sha(source) == expected,
            "source changed before snapshot",
        )
        target = root / "source" / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        require(sha(target) == sha(source) == expected, "source changed during snapshot")
        snapshots[str(target)] = expected
        refs[filename] = reference(target)
    return snapshots, refs


def prepare(config_ref, qa_ref):
    cpu_environment()
    evidence = Evidence()
    config = evidence.json(config_ref)
    check_config(config)
    prepared = evidence.json(config["prepared_ref"])
    require(
        prepared["schema_version"] == "s2a.clean-training-preparation.v1"
        and prepared["execution_authorized"] is False
        and prepared["readiness"]["ready_for_execution"] is False,
        "original metadata-only preparation was rewritten",
    )
    root = REPO / config["output_dir"]
    require(not root.exists(), "refuses existing040 proposal/attempt or retry")
    metadata = evidence.json(config["metadata_verification_ref"])
    require(
        metadata.get("schema_version") == "s2a.clean-training-preparation-independent.v1"
        and metadata.get("prepared_ref") == config["prepared_ref"]
        and metadata.get("metadata_qualified") is True
        and metadata.get("complete") is metadata.get("passed") is True
        and metadata.get("execution_ready") is metadata.get("execution_authorized") is False,
        "independent metadata qualification missing",
    )
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        evidence.mapping(prepared[key])
    evidence.mapping(metadata["consumed_sha256"])
    bridge = evidence.json(prepared["prerequisite_refs"]["bridge_prepared"])
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        evidence.mapping(bridge[key])
    evidence.mapping(bridge["checkpoint_small_metadata_sha256"])
    for path, expected in prepared["weight_files"].items():
        stat = Path(path).stat()
        require(
            stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], "163 weight stat changed"
        )
    control = evidence.json(config["development_control_ref"])
    require(
        control["complete"] is control["passed"] is control["control_contract_reuse_eligible"] is True
        and control["execution_ready"] is False
        and control["new_physics_control_qualification"] is False,
        "conditional qualified control reuse missing",
    )
    guard = control["cv2_guard_registration"]
    for ref in [guard["helper_ref"], guard["exact_probe_ref"], *guard["runtime_source_refs"]]:
        evidence.check(ref)
    evidence.mapping(control["source_sha256"])
    for item in control["evidence_refs"].values():
        evidence.check(item)
    facts = control["actual_facts"]
    closure = evidence.json(facts["source_closure_ref"])

    # The preflight's leaf records retain declared historical paths and actual
    # safe copies. Hash every actual leaf rather than inventing migrated SHAs.
    def consume_leaves(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                evidence.check(value)
            elif isinstance(value.get("actual"), str) and isinstance(value.get("sha256"), str):
                evidence.check(dict(path=value["actual"], sha256=value["sha256"]))
            for item in value.values():
                consume_leaves(item)
        elif isinstance(value, list):
            for item in value:
                consume_leaves(item)

    consume_leaves(control)
    consume_leaves(closure)
    for key, value in closure.items():
        if (
            key.endswith("sha256")
            and isinstance(value, dict)
            and all(isinstance(v, str) and len(v) == 64 for v in value.values())
        ):
            evidence.mapping(value)
    qualification = evidence.json(prepared["prerequisite_refs"]["bridge_qualification"])
    trace_ref = qualification["generations"][0]["trace_ref"]
    evidence.json(trace_ref)
    paths = source_paths()
    sources = {str(path): sha(path) for path in paths}
    require(all(sources.get(p) == s for p, s in prepared["source_sha256"].items()), "metadata14 source changed")
    qa = evidence.json(qa_ref)
    require(
        qa.get("complete") is qa.get("passed") is True
        and qa.get("source_sha256") == sources
        and qa.get("test_paths") == config["required_test_paths"]
        and qa.get("tests_passed", 0) > 0,
        "actual final-source related QA is incomplete or stale",
    )
    for command in qa["commands"]:
        require(command["exit_code"] == 0, "final-source QA command failed")
        evidence.check(command["log_ref"])
    config_yaml = next(
        dict(path=p, sha256=s)
        for p, s in bridge["checkpoint_small_metadata_sha256"].items()
        if Path(p).name == "config.yaml"
    )
    normal = normalization(config_yaml, evidence)
    seed_plan = build_seed_plan(config["prepared_ref"], prepared)
    evidence.finish()
    root.mkdir(parents=True, exist_ok=False)
    stage_result = dict(
        schema_version="s2a.clean-training-execution-preparation-attempt.v1",
        complete=False,
        passed=False,
        config_ref=config_ref,
        qa_ref=qa_ref,
        new_model_calls=0,
        new_fits=0,
        physics_steps=0,
        fresh_cases=0,
    )
    try:
        resource_ref = publish(root / "resources.json", observe_resources(root))
        normal_ref = publish(root / "normalization.json", normal)
        seed_ref = publish(root / "seed_plan.json", seed_plan)
        evidence.check(resource_ref)
        evidence.check(normal_ref)
        evidence.check(seed_ref)
        snapshots, snapshot_refs = freeze_sources(root, sources)
        from serve_wam_clean_training import case_order

        proposal = dict(
            schema_version="s2a.clean-training-execution-proposal.v1",
            complete=True,
            passed=True,
            readiness=dict(ready_for_execution=False, reason="independent040 execution verification is required"),
            execution_authorized=False,
            automatic_retry=False,
            fresh_performance_authorized=False,
            root=prepared["root"],
            config_ref=config_ref,
            prepared_ref=config["prepared_ref"],
            metadata_verification_ref=config["metadata_verification_ref"],
            development_control_ref=config["development_control_ref"],
            qa_ref=qa_ref,
            plan_ref=reference(REPO / PLAN),
            source_sha256=sources,
            snapshot_sha256=snapshots,
            snapshot_refs=snapshot_refs,
            input_sha256=evidence.inputs,
            weight_files=prepared["weight_files"],
            resource_observation_ref=resource_ref,
            normalization_ref=normal_ref,
            training_seed_plan_ref=seed_ref,
            sampler_trace_ref=trace_ref,
            resources=prepared["resources"],
            budget=prepared["budget"],
            master_port=31338,
            stage_entrypoints=STAGES,
            entrypoint_refs={k: reference(REPO / v) for k, v in STAGES.items()},
            development=dict(port=31339, case_order=case_order()),
            new_model_calls=0,
            new_fits=0,
            physics_steps=0,
            fresh_cases=0,
            future_archive_decodes=0,
        )
        evidence.finish()
        require(
            all(sha(path) == expected for path, expected in sources.items()), "source changed while preparing proposal"
        )
        result = publish(root / "proposal.json", proposal)
        stage_result.update(complete=True, passed=True, proposal_ref=result, execution_authorized=False)
    except BaseException as error:
        stage_result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        publish(root / "preparation_result.json", stage_result)
    return result


def admit(proposal_ref, verification_ref):
    cpu_environment()
    proposal, verified = read(proposal_ref), read(verification_ref)
    root = Path(proposal_ref["path"]).parent
    require(
        root == REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023/execution"
        and Path(verification_ref["path"]) == root / "verified.json",
        "fixed execution proof paths required",
    )
    admitted = copy.deepcopy(proposal)
    admitted.update(
        schema_version="s2a.clean-training-execution-registration.v1",
        execution_authorized=True,
        proposal_ref=proposal_ref,
        execution_verification_ref=verification_ref,
    )
    admitted["readiness"]["ready_for_execution"] = True
    verifier = validate_envelope(admitted, proposal, verified, prepared_ref=proposal["prepared_ref"])
    checked(verifier)
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        for path, expected in proposal[key].items():
            checked(dict(path=path, sha256=expected))
    return publish(root / "admitted.json", admitted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "admit"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    for name in ("qa", "proposal", "verification"):
        parser.add_argument(f"--{name}", type=Path)
        parser.add_argument(f"--{name}-sha256")
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()

    def ref(name):
        path, digest = getattr(args, name), getattr(args, name + "_sha256")
        require(path is not None and digest is not None, "stage requires exact path/SHA arguments")
        return dict(path=str(path.resolve()), sha256=digest)

    result = (
        prepare(reference(args.config), ref("qa"))
        if args.stage == "prepare"
        else admit(ref("proposal"), ref("verification"))
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
