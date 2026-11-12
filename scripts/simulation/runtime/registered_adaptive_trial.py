"""Explicitly configured preflight and model admission for adaptive trials."""

import csv
import io
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_model_bundle as registered
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import live_cv2_admission
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner


def gpu_observation(uuid):
    """Observe the explicitly registered device without starting a GPU job."""
    prefix.require(type(uuid) is str and uuid.startswith("GPU-"), "An explicit GPU UUID is required")
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=20
    )
    devices = list(csv.reader(io.StringIO(query), skipinitialspace=True))
    selected = [row for row in devices if len(row) == 2 and row[0].strip() == uuid]
    prefix.require(len(selected) == 1, "The registered GPU must be uniquely visible")
    memory = int(selected[0][1].strip())
    prefix.require(memory >= 0, "GPU memory usage must be nonnegative")
    query_apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        timeout=20,
    )
    apps = [row for row in csv.reader(io.StringIO(query_apps), skipinitialspace=True) if row and row[0].strip() == uuid]
    return {"memory_mib": memory, "compute_apps": [", ".join(field.strip() for field in row) for row in apps]}


@dataclass(frozen=True)
class RegisteredAdaptiveTrial:
    """Keep all trial-specific paths, limits and providers explicit and immutable."""

    output: Path
    qa: Path
    dry: Path
    module: str
    gpu_uuid: str
    required_qa_targets: tuple[str, ...]
    load_inputs: Callable
    input_receipt: Callable
    source_refs: Callable
    model_registration: Callable
    compare_previous: Callable | None = None

    def __post_init__(self):
        outputs = (prefix.REPO / "outputs").resolve()
        paths = (self.output, self.qa, self.dry)
        prefix.require(
            all(
                isinstance(path, Path) and path.is_absolute() and path.resolve().is_relative_to(outputs)
                for path in paths
            )
            and len({path.resolve() for path in paths}) == len(paths)
            and type(self.module) is str
            and self.module.startswith("scripts.simulation.runtime.")
            and type(self.gpu_uuid) is str
            and self.gpu_uuid.startswith("GPU-")
            and type(self.required_qa_targets) is tuple
            and bool(self.required_qa_targets)
            and len(set(self.required_qa_targets)) == len(self.required_qa_targets)
            and all(
                type(path) is str
                and (prefix.REPO / path).resolve().is_relative_to(prefix.REPO / "tests")
                and (prefix.REPO / path).is_file()
                for path in self.required_qa_targets
            )
            and all(
                callable(item)
                for item in (self.load_inputs, self.input_receipt, self.source_refs, self.model_registration)
            )
            and (self.compare_previous is None or callable(self.compare_previous)),
            "A local output, explicit GPU, test targets and immutable trial providers are required",
        )

    def require_config(self, inputs):
        config = inputs["config"]
        prefix.require(
            config["gpu_uuid"] == self.gpu_uuid
            and (prefix.REPO / config["output"]).resolve() == self.output.resolve()
            and (prefix.REPO / config["qa"]).resolve() == self.qa.resolve()
            and (prefix.REPO / config["dry"]).resolve() == self.dry.resolve(),
            "The registered trial paths and GPU must match the immutable context",
        )

    def model_bundle(self, inputs):
        return dict(
            input_receipt=self.input_receipt(inputs),
            prepared=inputs["prepared"],
            bridge=inputs["bridge"],
            template=inputs["template"],
            weight_hashes=inputs["weight_hashes"],
            model_registration=self.model_registration(inputs, self.output / "producer/simulation"),
        )

    def dry_run(self):
        prefix.environment()
        inputs = self.load_inputs()
        self.require_config(inputs)
        result = dict(
            complete=True,
            passed=True,
            inputs=self.input_receipt(inputs),
            source_refs=self.source_refs(),
            model_bundle_sha256=registered._digest(self.model_bundle(inputs)),
            native_models=0,
            physics_steps=0,
            model_initializations=0,
            model_requests=0,
            denoise_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        )
        return publish(self.dry, result)

    def check_cpu_admission(self, inputs):
        qa, dry = prefix.read(prefix.reference(self.qa)), prefix.read(prefix.reference(self.dry))
        before = snapshot(prefix.REPO)["sources"]
        prefix.require(
            qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
            and qa["actual_wait"] is qa["reaped"] is True
            and qa["signal_sent"] is False
            and qa["configured_parallelism"] == qa["execution_parallelism"] == 64
            and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
            and qa["aggregate_node_partition_exact"] is True
            and qa["call_count"] > 0
            and not qa["collection"]["collection_errors"]
            and set(self.required_qa_targets).issubset(set(qa["targets"]))
            and prefix.read(qa["source_before_ref"])["sources"] == before
            and dry["complete"] is dry["passed"] is True
            and dry["automatic_retry"] is False
            and dry["inputs"] == self.input_receipt(inputs)
            and dry["source_refs"] == self.source_refs()
            and dry["model_bundle_sha256"] == registered._digest(self.model_bundle(inputs))
            and dry["native_models"]
            == dry["physics_steps"]
            == dry["model_initializations"]
            == dry["model_requests"]
            == dry["denoise_calls"]
            == dry["fresh_cases"]
            == 0,
            "Matching 64-worker CPU regression and zero-physics dry run are required",
        )
        return before

    def preflight(self):
        prefix.environment()
        inputs = self.load_inputs()
        self.require_config(inputs)
        before, sources = self.check_cpu_admission(inputs), self.source_refs()
        self.output.mkdir(parents=True, exist_ok=False)
        started = dict(complete=False, config=inputs["refs"], pid=os.getpid())
        publish(self.output / "preflight_started.json", started)
        try:
            gpu = gpu_observation(self.gpu_uuid)
            prefix.require(gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "The registered GPU must be idle")
            manifests = model_owner.verify_model_manifests(inputs)
            probe_dir = self.output / "cv2_probe"
            probe_dir.mkdir()
            probe = live_cv2_admission.run_cpu_probe(inputs["prepared"]["model_config"]["model_python"], probe_dir)
            probe_ref = publish(probe_dir / "result.json", probe)
            prefix.require(
                snapshot(prefix.REPO)["sources"] == before and self.source_refs() == sources,
                "Preflight sources changed",
            )
            result = dict(
                complete=True,
                passed=True,
                inputs=self.input_receipt(inputs),
                source_refs=sources,
                qa_ref=prefix.reference(self.qa),
                dry_ref=prefix.reference(self.dry),
                model_bundle_sha256=registered._digest(self.model_bundle(inputs)),
                gpu_uuid=self.gpu_uuid,
                gpu_before=gpu,
                cv2_probe_ref=probe_ref,
                model_python_sha256=repro.sha256(inputs["prepared"]["model_config"]["model_python"]),
                model_initializations=0,
                model_requests=0,
                physics_steps=0,
                automatic_retry=False,
                **manifests,
            )
            return publish(self.output / "preflight.json", result)
        except BaseException as error:
            publish(
                self.output / "preflight_failed.json",
                {**started, "error": {"type": type(error).__name__, "message": str(error)}},
            )
            raise

    def check_preflight(self, inputs):
        self.require_config(inputs)
        value = prefix.read(prefix.reference(self.output / "preflight.json"))
        prefix.require(
            value["complete"] is value["passed"] is True
            and value["inputs"] == self.input_receipt(inputs)
            and value["source_refs"] == self.source_refs()
            and value["qa_ref"] == prefix.reference(self.qa)
            and value["dry_ref"] == prefix.reference(self.dry)
            and value["model_bundle_sha256"] == registered._digest(self.model_bundle(inputs))
            and value["gpu_uuid"] == self.gpu_uuid
            and value["source_files_verified"] == len(inputs["model_source_hashes"])
            and value["weight_files_verified"] == len(inputs["weight_hashes"]) == 163
            and value["model_python_sha256"] == repro.sha256(inputs["prepared"]["model_config"]["model_python"])
            and value["weight_stats"] == model_owner.file_stats(inputs["weight_hashes"]),
            "The source-matched registered preflight is required",
        )
        probe = prefix.read(value["cv2_probe_ref"])
        prefix.require(probe["actual_wait"] is probe["reaped"] is True and probe["returncode"] == 0, "CPU OpenCV probe")
        return value

    def model_worker_inputs(self):
        """Revalidate the producer bundle in the fresh model interpreter."""
        return registered.load_registered_model_bundle(
            output=self.output,
            qa_path=self.qa,
            dry_path=self.dry,
            gpu_uuid=self.gpu_uuid,
            source_refs=self.source_refs,
        )
