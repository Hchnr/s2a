"""Replay a recorded adaptive trajectory with exact same-state endpoint reuse."""

import sys
from types import SimpleNamespace

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.research.native_runtime.endpoint_cache import SameStateEndpointCache
from scripts.simulation.runtime import adaptive_live_bowl_worker as historical
from scripts.simulation.runtime import adaptive_long_cached_checker_admission as admission
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation


def cached_observers(observers, receipt, *, start_step):
    """Reuse only exact same-state observations from the adaptive boundary onward."""
    prefix.require(type(start_step) is int and start_step >= 0, "A nonnegative cache boundary is required")
    cache = SameStateEndpointCache(observers.endpoint)
    receipt["adaptive_cached_endpoint_events"] = cache.events

    def endpoint(env, step):
        return observers.endpoint(env, step) if step < start_step else cache(env, step)

    return SimpleNamespace(
        endpoint=endpoint,
        current=observers.current,
        render_anchor=observers.render_anchor,
        render_cameras=observers.render_cameras,
    )


def candidate_modules_unloaded():
    forbidden = tuple(isolation.FORBIDDEN) + (
        "s2a.research.joint_control.preview.causal_reference_replan",
        "s2a.research.wam.tracking.runtime.adaptive_stream_reference",
    )
    return all(name not in sys.modules for name in forbidden)


def run_worker(*, source_refs):
    """Run one registered CPU checker and preserve interrupted physical receipts."""
    prefix.require(callable(source_refs), "A source provider is required")
    directory = admission.OUTPUT / "checker/simulation"
    prefix.require(directory.is_dir(), "The checker-local simulation directory must exist")

    def execute():
        inputs = admission.load_inputs()
        registration = prefix.read(prefix.reference(admission.OUTPUT / "registration.json"))
        prefix.require(
            registration["inputs"] == admission.input_receipt(inputs) and registration["source_refs"] == source_refs(),
            "The cached checker requires the exact registered inputs and sources",
        )
        isolation.checker_isolation()
        producer = prefix.read(inputs["producer"]["branch_ref"])
        model = prefix.read(inputs["producer"]["model_ref"])
        responses = historical.load_recorded_responses(producer, model)
        long_inputs = inputs["long_inputs"]
        source = long_inputs["prior"]["base"]["source"]
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
        runner, options = historical.prefix_options(long_inputs, "checker")
        stream_registration = admission.replay_registration(inputs, directory)

        def run_continuation(env, terminal, observers, receipt, *, progress=None):
            wrapped = cached_observers(observers, receipt, start_step=long_inputs["config"]["prefix_steps"])
            return continuation.run_checker(
                env,
                terminal,
                wrapped,
                receipt,
                prefix_runner=runner,
                producer=producer,
                expected_replan_capture=long_inputs["expected_capture"],
                responses=responses,
                registration=stream_registration,
                runtime_output=directory / "runtime",
                case=case,
                progress=progress,
                **options,
            )

        actual = runtime.run_feedback(
            case["config"],
            case["record"],
            case["record_xml"],
            directory / "runtime",
            old_runtime["asset_refs"],
            registered_seed=inputs["config"]["seed"],
            continuation=run_continuation,
            max_control_steps=inputs["config"]["max_control_steps"],
        )
        unloaded = candidate_modules_unloaded()
        prefix.require(unloaded, "The checker cannot load candidate policies")
        return dict(
            complete=True,
            passed=True,
            case=case["case"],
            runtime=actual,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=unloaded,
            fresh_cases=0,
        )

    return managed.run_managed_physical_worker(phase="checker", result_path=directory / "result.json", execute=execute)
