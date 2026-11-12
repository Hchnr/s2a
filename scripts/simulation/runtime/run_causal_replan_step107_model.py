"""Reuse the single-image WAM owner for the verified step-107 observation."""

from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_bowl_step88_model as previous

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/causal_replan_step107_model_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/causal_replan_step107_model_20261112.md"
OUTPUT = ROOT / "outputs/causal_replan_step107_model_20261112"
QA = ROOT / "outputs/qa/causal_replan_step107_model_20261112/result.json"
DRY = ROOT / "outputs/causal_replan_step107_model_dry_20261112/result.json"
CAPTURE = {
    "path": str(ROOT / "outputs/causal_replan_step107_capture_20261112/result.json"),
    "sha256": "44d1b205e85815cab0b215f1cdfa6d3cba73671f1d1c6135b318dabe04759656",
}
IMAGE_SHA = "f7cc70d689471bbc96865f825dc44bc9e1c2607bf63bfa56b6114780ec2129a3"
STATE_SHA = "04b8eea4cd6e611302b68ff036594129808fec0bb88cab97f4e55106ee4c148d"
SESSION = "causal_replan_step107_offline_s42_20261112"
MODULE = "scripts.simulation.runtime.run_causal_replan_step107_model"
PREVIOUS_SHA = "0ee66b3754f780bd5615f57bcf1e1d92c5ebb8cfdab1c6b23251b19b1e0b805a"
FIXED = dict(
    schema="s2a.causal-replan-step107-model.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    case_id="task0_init0_s42_raw",
    task=previous.previous.TASK,
    seed=42,
    capture_step=107,
    image_sha256=IMAGE_SHA,
    state_sha256=STATE_SHA,
    gpu_uuid=previous.previous.GPU,
    model_sha256=previous.previous.MODEL_SHA,
    max_model_initializations=1,
    max_model_requests=1,
    denoise_steps=35,
    physics_steps=0,
    fresh_cases=0,
    automatic_retry=False,
    worker_timeout_seconds=2400,
)
_PREVIOUS_SOURCES = previous.source_refs


def validate_capture(result):
    capture, snapshot = result["terminal_capture"]["capture"], result["terminal_capture"]["snapshot"]
    endpoint = capture["current_endpoint"]
    binding = {"sim_time": snapshot["sim_time_s"], "sim_state_sha256": STATE_SHA}
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["control_qualified"] is False
        and result["counts"]["control_steps"] == 107
        and result["counts"]["native_steps"] == 2675
        and result["counts"]["stop"] == dict(reason="requested_observation", at_step=107, next_action_dispatched=False)
        and all(value == 0 for value in result["independent_max_absolute_difference"].values())
        and snapshot["step"] == capture["frame_index"] == endpoint["endpoint_step"] == 107
        and snapshot["state_sha256"] == endpoint["sim_state_sha256"] == STATE_SHA
        and endpoint["sim_time"] == snapshot["sim_time_s"]
        and capture["render_before"] == capture["render_after"] == binding
        and capture["future_rgb_frames"] == 0
        and capture["canvas_png"]["sha256"] == IMAGE_SHA
        and capture["camera_roles"] == ["auxiliary_0", "left_wrist"]
        and capture["canvas_shape"] == [960, 640, 3],
        "The verified step-107 current image and state are required",
    )
    return dict(snapshot), capture


def source_refs():
    paths = (
        Path(__file__).resolve(),
        ROOT / "tests/research/wam/test_causal_replan_step107_model.py",
        ROOT / "scripts/simulation/runtime/run_causal_replan_step107_capture.py",
        ROOT / "src/s2a/research/native_runtime/recorded_reference_prefix.py",
    )
    return _PREVIOUS_SOURCES() + [prefix.reference(path) for path in paths]


def bind():
    prefix.require(
        previous.previous.repro.sha256(previous.__file__) == PREVIOUS_SHA, "The original model adapter changed"
    )
    previous.CONFIG, previous.PLAN, previous.OUTPUT, previous.QA, previous.DRY = CONFIG, PLAN, OUTPUT, QA, DRY
    previous.CAPTURE, previous.IMAGE_SHA, previous.STATE_SHA, previous.SESSION, previous.MODULE, previous.FIXED = (
        CAPTURE,
        IMAGE_SHA,
        STATE_SHA,
        SESSION,
        MODULE,
        FIXED,
    )
    previous.validate_capture, previous.source_refs = validate_capture, source_refs
    return previous.bind()


if __name__ == "__main__":
    bind().main()
