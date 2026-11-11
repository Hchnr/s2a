"""Apply the corrected scorer to the preserved cached replay, without rerunning it."""

import json

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.native_runtime.cached_replay_comparison import compare_cached_replay
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_endpoint_cache_replay as runner
from scripts.verification.reference_ports import audit_endpoint_cache_partial as partial

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/causal_stream_endpoint_cache_fixed_audit_20261111/result.json"
PARTIAL_AUDIT = {
    "path": str(ROOT / "outputs/causal_stream_endpoint_cache_partial_audit_20261111/result.json"),
    "sha256": "de30a91e74bdfeb7ef00b766c73245e68fda66fd730d897bc8eba0600cce9c51",
}


def load_audit():
    prefix.environment()
    inputs = runner.load_inputs()
    prior = prefix.read(PARTIAL_AUDIT)
    branch = prefix.read(partial.BRANCH)
    comparison = compare_cached_replay(
        branch["runtime"], inputs["base"]["stream"], branch["endpoint_cache_events"], branch["capture"]
    )
    prefix.require(
        prior["complete"] is prior["passed"] is True
        and prior["original_owner_passed"] is False
        and comparison["completed_actions"] == prior["replayed_actions"] == 40
        and comparison["native_steps"] == prior["native_steps"] == 3000
        and comparison["max_difference"] == prior["replay_max_difference"]
        and comparison["reference_final_state_sha256"] == prior["historical_final_state_sha256"]
        and comparison["replay_final_state_sha256"] == prior["replay_final_state_sha256"]
        and comparison["final_state_bytes_equal"] is False,
        "The corrected scorer must agree with the independent partial audit",
    )
    return {
        "complete": True,
        "passed": True,
        "original_owner_passed": False,
        "control_qualified": False,
        "comparison": comparison,
        "capture_ref": prior["capture_ref"],
        "task_success_tail_periods": prior["task_success_tail_periods"],
        "prior_audit_ref": PARTIAL_AUDIT,
        "original_branch_ref": partial.BRANCH,
        "new_model_requests": 0,
        "new_physics_steps": 0,
    }


def main():
    print(json.dumps(publish(OUTPUT, load_audit())), flush=True)


if __name__ == "__main__":
    main()
