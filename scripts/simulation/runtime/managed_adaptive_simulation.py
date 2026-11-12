"""Run a registered adaptive simulation under the managed receipt boundary."""

import os
import socket
import sys
from pathlib import Path

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.native_environment import restore_cv2_environment
from scripts.simulation.runtime import adaptive_live_bowl_worker as historical
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation


def run_adaptive_simulation_worker(phase, *, output, load_inputs, input_receipt, source_refs, model_registration):
    """Bind one producer/checker to registered inputs, sources and partial receipts.

    The caller supplies the registered output and pure admission functions. The
    parent owns process launch, model communication, budgets and wait/reap.
    This worker does not patch historical globals; admission callbacks must
    preserve the registered identities and saved experiment outputs.
    """
    output = Path(output).resolve()
    prefix.require(
        phase in ("producer", "checker")
        and output.is_relative_to(prefix.REPO / "outputs")
        and all(callable(fn) for fn in (load_inputs, input_receipt, source_refs, model_registration)),
        "A local registered phase and admission functions are required",
    )
    directory = output / phase / "simulation"
    prefix.require(directory.is_dir(), "The phase-local simulation directory must exist")

    def execute():
        channel = None
        try:
            inputs = load_inputs()
            registration = prefix.read(prefix.reference(output / phase / "registration.json"))
            prefix.require(
                registration["phase"] == phase
                and registration["inputs"] == input_receipt(inputs)
                and registration["source_refs"] == source_refs(),
                "The simulation worker requires the exact registered inputs and sources",
            )
            producer = responses = None
            if phase == "checker":
                isolation.checker_isolation()
                producer = prefix.read(registration["producer_branch_ref"])
                model_result = prefix.read(registration["producer_model_ref"])
                prefix.require(
                    producer["complete"]
                    is producer["passed"]
                    is model_result["complete"]
                    is model_result["passed"]
                    is True,
                    "The completed producer and model workers are required",
                )
                responses = historical.load_recorded_responses(producer, model_result)
            else:
                channel = FramedJSONChannel(
                    socket.socket(fileno=int(os.environ["S2A_CHANNEL_FD"])),
                    max_bytes=2 * 1024 * 1024,
                    timeout_seconds=1750,
                )
            source = inputs["prior"]["base"]["source"]
            base = source["previous"]
            old_runtime = base["base"]["base"]["previous"]["base"]
            cv2 = old_runtime["cv2_admission"]
            cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
            original.require_no_simulator_import()
            from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime
            from scripts.simulation.runtime import adaptive_live_bowl as continuation

            case = base["base"]["case"]
            runtime = ResetBoundedRuntime(
                expected=case["expected"],
                expected_receipt=case["expected_receipt"],
                chunk=case["chunk"],
                controller=case["controller"],
                tool=case["tool"],
            )
            runtime_output = directory / "runtime"
            stream_registration = model_registration(inputs, directory)
            runner, options = historical.prefix_options(inputs, phase)

            def run_continuation(env, terminal, observers, receipt, *, progress=None):
                common = dict(
                    prefix_runner=runner,
                    expected_replan_capture=inputs["expected_capture"],
                    registration=stream_registration,
                    runtime_output=runtime_output,
                    case=case,
                    progress=progress,
                    **options,
                )
                if phase == "producer":
                    return continuation.run_producer(
                        env,
                        terminal,
                        observers,
                        receipt,
                        prior_producer=inputs["prior"]["base"]["producer"],
                        expected_response=inputs["expected_response"],
                        channel=channel,
                        **common,
                    )
                return continuation.run_checker(
                    env, terminal, observers, receipt, producer=producer, responses=responses, **common
                )

            actual = runtime.run_feedback(
                case["config"],
                case["record"],
                case["record_xml"],
                runtime_output,
                old_runtime["asset_refs"],
                registered_seed=inputs["config"]["seed"],
                continuation=run_continuation,
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
