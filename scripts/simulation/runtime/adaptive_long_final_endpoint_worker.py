"""Replay saved actions with caching confined to the final endpoint."""

from types import SimpleNamespace

from s2a.infrastructure.native_environment import restore_cv2_environment
from scripts.simulation.runtime import adaptive_live_bowl_worker as historical
from scripts.simulation.runtime import adaptive_long_cached_checker_worker as cached
from scripts.simulation.runtime import adaptive_long_final_endpoint_admission as admission
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation


def final_endpoint_observers(observers, receipt, *, final_step):
    """Keep earlier reads unchanged and reject observations beyond the stop."""
    prefix.require(type(final_step) is int and final_step > 0, "A positive final endpoint is required")
    wrapped = cached.cached_observers(observers, receipt, start_step=final_step)

    def endpoint(env, step):
        prefix.require(type(step) is int and step <= final_step, "The checker cannot observe past its final endpoint")
        return wrapped.endpoint(env, step)

    return SimpleNamespace(
        endpoint=endpoint,
        current=wrapped.current,
        render_anchor=wrapped.render_anchor,
        render_cameras=wrapped.render_cameras,
    )


def run_worker(*, source_refs):
    """Run the separately registered CPU checker and preserve partial receipts."""
    prefix.require(callable(source_refs), "A source provider is required")
    directory = admission.OUTPUT / "checker/simulation"
    prefix.require(directory.is_dir(), "The checker-local simulation directory must exist")

    def execute():
        inputs = admission.load_inputs()
        registration = prefix.read(prefix.reference(admission.OUTPUT / "registration.json"))
        prefix.require(
            registration["inputs"] == admission.input_receipt(inputs) and registration["source_refs"] == source_refs(),
            "The final-endpoint checker requires the exact registered inputs and sources",
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
            wrapped = final_endpoint_observers(observers, receipt, final_step=inputs["config"]["cache_step"])
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
        unloaded = cached.candidate_modules_unloaded()
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
