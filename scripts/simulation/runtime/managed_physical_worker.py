"""Publish bounded physical worker results on completion or interruption."""

from pathlib import Path

from s2a.infrastructure.lifecycle.interruptible_worker import WorkerTermination, interruptible_sigterm
from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.development_prefix_runtime_bounded import PrefixReplayFailure
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def run_managed_physical_worker(*, phase, result_path, execute):
    """Run one admitted callback and preserve its final or partial runtime receipt.

    The caller owns input registration, resource budgets and child reaping.
    SIGKILL or a blocked native call may still prevent publication; the owner
    must treat an absent receipt as unknown rather than zero physical work.
    """
    result_path = Path(result_path).resolve()
    prefix.require(
        phase in ("producer", "checker")
        and callable(execute)
        and result_path.is_relative_to(prefix.REPO / "outputs")
        and result_path.name == "result.json"
        and result_path.parent.is_dir()
        and not result_path.exists(),
        "A fresh local worker receipt and registered physical phase are required",
    )
    result = dict(
        complete=False,
        passed=False,
        phase=phase,
        control_qualified=False,
        automatic_retry=False,
        termination_requested=False,
    )
    with interruptible_sigterm():
        try:
            prefix.environment()
            payload = execute()
            prefix.require(
                type(payload) is dict
                and payload.get("complete") is payload.get("passed") is True
                and type(payload.get("runtime")) is dict
                and payload["runtime"].get("complete") is payload["runtime"].get("passed") is True
                and not {"phase", "automatic_retry", "termination_requested", "control_qualified", "error"}
                & payload.keys(),
                "A completed physical runtime receipt is required",
            )
            result.update(payload)
        except BaseException as error:
            if isinstance(error, PrefixReplayFailure):
                result["runtime"] = error.receipt
            result["termination_requested"] = isinstance(error, WorkerTermination) or isinstance(
                error.__cause__, WorkerTermination
            )
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            print(publish_result(result_path, result), flush=True)
    return result
