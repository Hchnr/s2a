"""Execute one reset-origin adaptive case or replay it in an isolated worker."""

import os
import socket
import sys

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.native_environment import restore_cv2_environment
from scripts.simulation.runtime import adaptive_live_bowl_worker as historical
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_adaptive_audit as audit
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation


def run_worker(index, phase):
    """Use source-matched inputs and publish complete or bounded partial work."""
    prefix.require(phase in ("producer", "checker"), "One registered physical phase is required")
    trial = admission.make_trial(index)
    directory = trial.output / phase / "simulation"

    def execute():
        channel = None
        try:
            inputs = admission.load_inputs(index, verify_assets=False)
            registration = prefix.read(prefix.reference(trial.output / phase / "registration.json"))
            trial.check_preflight(inputs)
            prefix.require(
                registration["phase"] == phase
                and registration["inputs"] == trial.input_receipt(inputs)
                and registration["source_refs"] == trial.source_refs(),
                "The physical worker requires the registered case and source identities",
            )
            producer = responses = None
            if phase == "checker":
                isolation.checker_isolation()
                previous = prefix.read(prefix.reference(trial.output / "producer/result.json"))
                prefix.require(
                    previous["complete"]
                    is previous["passed"]
                    is previous["sources_unchanged"]
                    is previous["gpu_released"]
                    is True
                    and registration["producer_branch_ref"] == previous["branch_ref"]
                    and registration["producer_model_ref"] == previous["model_ref"],
                    "The checker requires the closed matching producer and model",
                )
                producer, model = prefix.read(previous["branch_ref"]), prefix.read(previous["model_ref"])
                audit.verify_model_exchange(producer, model, trial.model_bundle(inputs))
                responses = historical.load_recorded_responses(producer, model)
            else:
                bundle_ref = registration["model_bundle_ref"]
                prefix.require(
                    bundle_ref == prefix.reference(trial.output / "producer/model/registered_inputs.json")
                    and prefix.read(bundle_ref) == trial.model_bundle(inputs),
                    "The producer requires its exact registered model bundle",
                )
                channel = FramedJSONChannel(
                    socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])),
                    max_bytes=2 * 1024 * 1024,
                    timeout_seconds=1750,
                )
            cv2 = inputs["base"]["cv2_admission"]
            cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
            original.require_no_simulator_import()
            from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime
            from scripts.simulation.runtime import reset_adaptive_stream as stream

            case = inputs["case"]
            runtime = ResetBoundedRuntime(
                expected=case["expected"],
                expected_receipt=case["expected_receipt"],
                chunk=case["chunk"],
                controller=case["controller"],
                tool=case["tool"],
            )
            runtime_output = directory / "runtime"
            stream_registration = trial.model_registration(inputs, directory)

            def continuation(env, terminal, observers, receipt, *, progress=None):
                common = dict(
                    case=case,
                    registration=stream_registration,
                    runtime_output=runtime_output,
                    progress=progress,
                )
                if phase == "producer":
                    return stream.run_producer(
                        env,
                        terminal,
                        observers,
                        receipt,
                        baseline=inputs["baseline_periods"],
                        channel=channel,
                        **common,
                    )
                return stream.run_checker(
                    env, terminal, observers, receipt, producer=producer, responses=responses, **common
                )

            actual = runtime.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                inputs["base"]["asset_refs"],
                registered_seed=inputs["config"]["seed"],
                continuation=continuation,
                max_control_steps=inputs["config"]["max_control_steps_per_phase"],
            )
            forbidden = tuple(isolation.FORBIDDEN) + (
                "s2a.research.joint_control.preview.causal_reference_replan",
                "s2a.research.wam.tracking.runtime.adaptive_stream_reference",
            )
            unloaded = all(name not in sys.modules for name in forbidden)
            prefix.require(phase != "checker" or unloaded, "The checker cannot load candidate policies")
            return dict(
                complete=True,
                passed=True,
                case=case["case"],
                runtime=actual,
                cv2_admission=cv2_receipt,
                candidate_modules_unloaded=unloaded,
                fresh_cases=0,
            )
        finally:
            if channel is not None:
                channel.close()

    return managed.run_managed_physical_worker(phase=phase, result_path=directory / "result.json", execute=execute)
