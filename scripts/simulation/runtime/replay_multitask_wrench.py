"""Replay the six recorded development cases with causal wrench measurements."""

import argparse
import copy
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from s2a.infrastructure.native_environment import restore_cv2_environment
from s2a.infrastructure.publication import publish_result
from s2a.research.gripper.current_wrench import read_current_wrench
from scripts.simulation.runtime import adaptive_live_bowl_worker as historical
from scripts.simulation.runtime import inspect_recorded_wrench as measured
from scripts.simulation.runtime import managed_physical_worker as managed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_adaptive_audit as audit
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission
from scripts.simulation.runtime import run_reset_start_feedback as original
from scripts.simulation.runtime import run_segmented_reset_equivalence as isolation

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/wrench_multitask_replay_20261113"
SOURCE = {
    "path": str(ROOT / "outputs/reset_multitask_adaptive_complete_audit_20261113/result.json"),
    "sha256": "7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe",
}
INDICES = (0, 2, 4, 6, 8, 10)
EXPECTED_STEPS = (133, 25, 156, 45, 462, 204)
KNOWN_SOURCE_CHANGES = {
    "scripts/simulation/runtime/reset_adaptive_audit.py": (
        "9521c1b9d362449db4a49290e0acc390b752f2d9",
        "79c5cd76768e02e0ed080a398e8518b2659be457fe19409b7e59073dad473eee",
        "e6d6a07d27141efc0364cc0b81f2ee04e1bad9f30ef256bbd0fdd8d88acbdfb6",
    ),
    "tests/research/native_runtime/test_reset_multitask_adaptive_worker.py": (
        "124d070159a28dfe20a7d6986e292410e29aa854",
        "210c3d2754f8f97a563d6a6ceb7d99343cade0db8859e4871a1dcc3829fb465e",
        "e4a3a71e335c25f3bfb1779b250953ef0065caa92d1062f82dbb17e0c510ec1c",
    ),
    "tests/research/native_runtime/test_reset_adaptive_audit.py": (
        "3459885467d258c99f48a8cf0428a3fdd45087b6",
        "9701279988dbc288a1347d26bb383a4b13bbcc575022df6a372d76147f368387",
        "5f26852f30f4c6fdb53bc10ef75e76b0343d833e27d863b1f8804d50f543c604",
    ),
}


def verify_original_sources(refs):
    """Preserve original blobs while pinning the explicitly repaired current files."""
    replacements = []
    for ref in refs:
        path = Path(ref["path"])
        relative = str(path.relative_to(ROOT))
        current = prefix.reference(path)
        if current["sha256"] == ref["sha256"]:
            continue
        prefix.require(relative in KNOWN_SOURCE_CHANGES, f"Unregistered source change: {relative}")
        revision, original_sha, current_sha = KNOWN_SOURCE_CHANGES[relative]
        blob = subprocess.run(
            ["git", "show", f"{revision}:{relative}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        prefix.require(
            ref["sha256"] == original_sha
            and hashlib.sha256(blob).hexdigest() == original_sha
            and current["sha256"] == current_sha,
            f"Historical blob and repaired source must match their pinned identities: {relative}",
        )
        replacements.append(dict(original=ref, historical_commit=revision, current=current))
    prefix.require(len(replacements) == len(KNOWN_SOURCE_CHANGES), "All three documented source changes are required")
    return replacements


class CurrentWrenchTrace:
    """Collect one causal observation per completed endpoint, including reset."""

    def __init__(self, last_step, *, reader=read_current_wrench, contacts=measured.gripper_contacts):
        self.last_step, self.reader, self.contacts = last_step, reader, contacts
        self.rows = {}

    def capture(self, env, step, endpoint):
        if type(step) is not int or not 0 <= step <= self.last_step or endpoint["endpoint_step"] != step:
            raise ValueError("Only registered completed endpoints may be recorded")
        wrench = self.reader(env)
        if wrench["time"] != endpoint["sim_time"] or wrench["readonly_verified"] is not True:
            raise ValueError("A read-only wrench must match the current endpoint time")
        row = dict(
            endpoint_step=step,
            sim_time=endpoint["sim_time"],
            sim_state_sha256=endpoint["sim_state_sha256"],
            wrench=wrench,
            offline_gripper_contacts=self.contacts(env),
        )
        if step in self.rows:
            old = self.rows[step]
            if old["sim_time"] != row["sim_time"] or old["sim_state_sha256"] != row["sim_state_sha256"]:
                raise ValueError("Repeated endpoint identity changed")
        else:
            self.rows[step] = row

    def wrap(self, env, terminal, observers):
        self.capture(env, terminal["endpoint_receipt"]["endpoint_step"], terminal["endpoint_receipt"])

        def endpoint(actual_env, step):
            sample, receipt = observers.endpoint(actual_env, step)
            self.capture(actual_env, step, receipt)
            return sample, receipt

        return SimpleNamespace(
            endpoint=endpoint,
            current=observers.current,
            render_anchor=observers.render_anchor,
            render_cameras=observers.render_cameras,
        )


def admit(index):
    """Bind source-audited records without changing any prior failure receipt."""
    prefix.environment()
    prefix.require(index in INDICES, "One of the six registered cases is required")
    review = prefix.read(SOURCE)
    prefix.require(
        review["complete"] is True and [row["case_index"] for row in review["cases"]] == list(INDICES),
        "The completed six-case development review is required",
    )
    row = next(row for row in review["cases"] if row["case_index"] == index)
    prefix.require(row["counts"]["control_steps"] == EXPECTED_STEPS[INDICES.index(index)], "Recorded budget changed")
    owner = prefix.read(row["original_owner_ref"])
    registration = prefix.read(owner["registration_ref"])
    producer, model, bundle = (
        prefix.read(ref) for ref in (row["producer_ref"], row["model_ref"], registration["model_bundle_ref"])
    )
    inputs = admission.load_inputs(index, verify_assets=False)
    trial = admission.make_trial(index)
    prefix.require(
        registration["inputs"] == admission.input_receipt(inputs)
        and bundle == trial.model_bundle(inputs)
        and producer["complete"] is producer["passed"] is model["complete"] is model["passed"] is True
        and producer["case"]["case_index"] == index,
        "The original recorded reset, producer, model and bundle must agree",
    )
    counts = audit.summarize_branch(producer, "producer", inputs["config"])
    exchange = audit.verify_model_exchange(producer, model, bundle)
    prefix.require(
        counts["control_steps"] == row["counts"]["control_steps"] and exchange == row["model_exchange"],
        "The original physical and model counts must match the completed review",
    )
    source_changes = verify_original_sources(registration["source_refs"])
    prefix.require(
        review["fixed_verifier_ref"]["sha256"]
        == KNOWN_SOURCE_CHANGES["scripts/simulation/runtime/reset_adaptive_audit.py"][2],
        "The development review must bind the identical corrected verifier",
    )
    prefix.checked(review["fixed_verifier_ref"])
    return inputs, producer, model, bundle, row, registration, source_changes


def replay(index):
    inputs, producer, model, bundle, row, original_registration, source_changes = admit(index)
    responses = historical.load_recorded_responses(producer, model)
    isolation.checker_isolation()
    original.require_no_simulator_import()
    directory = OUTPUT / f"case_{index:02d}" / "simulation"
    directory.mkdir(parents=True, exist_ok=False)
    trace = CurrentWrenchTrace(row["counts"]["control_steps"])
    trace_result = dict(complete=False, passed=False, source_review=SOURCE, case_index=index, new_model_requests=0)

    def execute():
        cv2 = inputs["base"]["cv2_admission"]
        cv2_receipt = restore_cv2_environment(cv2["cv2_probe_ref"], cv2["cv2_runtime_source_refs"])
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
        registration = copy.deepcopy(bundle["model_registration"])
        registration["image_root"] = str(directory.relative_to(ROOT))

        def continuation(env, terminal, observers, receipt, *, progress=None):
            wrapped = trace.wrap(env, terminal, observers)
            return stream.run_checker(
                env,
                terminal,
                wrapped,
                receipt,
                case=case,
                producer=producer,
                responses=responses,
                registration=registration,
                runtime_output=runtime_output,
                progress=progress,
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
        prefix.require(unloaded, "The independent replay must not load candidate policies")
        return dict(
            complete=True,
            passed=True,
            case=case["case"],
            runtime=actual,
            cv2_admission=cv2_receipt,
            candidate_modules_unloaded=unloaded,
            fresh_cases=0,
        )

    try:
        managed.run_managed_physical_worker(phase="checker", result_path=directory / "result.json", execute=execute)
        checker_ref = prefix.reference(directory / "result.json")
        checker = prefix.read(checker_ref)
        counts = audit.summarize_branch(checker, "checker", inputs["config"])
        comparison = audit.compare_branches(producer, checker)
        n = row["counts"]["control_steps"]
        prefix.require(
            counts["control_steps"] == n and sorted(trace.rows) == list(range(n + 1)), "All endpoints required"
        )
        first, adaptive = audit.periods(producer)
        for period in first + adaptive:
            endpoint = period["endpoint_receipt"]
            prefix.require(
                trace.rows[endpoint["endpoint_step"]]["sim_state_sha256"] == endpoint["sim_state_sha256"],
                "Every sensor endpoint must match the original physical state",
            )
        prefix.require(
            verify_original_sources(original_registration["source_refs"]) == source_changes,
            "Original blobs and repaired sources must remain unchanged during replay",
        )
        for key in ("original_owner_ref", "producer_ref", "model_ref", "checker_ref"):
            prefix.checked(row[key])
        prefix.checked(SOURCE)
        trace_result.update(
            complete=True,
            passed=True,
            original_owner_failed=row["original_owner_failed"],
            source_changes=source_changes,
            producer_ref=row["producer_ref"],
            model_ref=row["model_ref"],
            checker_ref=checker_ref,
            script_ref=prefix.reference(Path(__file__).resolve()),
            checker_counts=counts,
            independent_max_absolute_difference=comparison,
            stop_reason_independently_verified=False,
        )
    except BaseException as error:
        trace_result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        trace_result["rows"] = list(trace.rows.values())
        print(publish_result(directory.parent / "wrench_trace.json", trace_result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_index", type=int, choices=INDICES)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        _, _, _, _, row, _, _ = admit(args.case_index)
        print({"case_index": args.case_index, "control_steps": row["counts"]["control_steps"]})
    else:
        replay(args.case_index)


if __name__ == "__main__":
    main()
