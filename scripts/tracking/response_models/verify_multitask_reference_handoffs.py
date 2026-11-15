"""Independently reconstruct common-time reference handoff measurements."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.infrastructure.publication import publish_result
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_multitask_adaptive_admission as admission

ROOT = prefix.REPO
AUDIT = {
    "path": str(ROOT / "outputs/reset_multitask_adaptive_complete_audit_20261113/result.json"),
    "sha256": "7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe",
}
RESULT = {
    "path": str(ROOT / "outputs/multitask_reference_handoffs_20261113/result.json"),
    "sha256": "af7339e6ac8db6417813288247833be9074beb1dadfba24577830df1aeb061c6",
}
OUTPUT = ROOT / "outputs/multitask_reference_handoffs_20261113/independent_verification.json"


def integrate_scalar(anchor, actions, endpoint):
    """Independently integrate exactly endpoint+1 column-rot6d local deltas."""
    value = np.asarray(anchor, dtype=float).copy()
    for row in actions[: endpoint + 1]:
        x = np.asarray(row[3:6], dtype=float)
        y = np.asarray(row[6:9], dtype=float)
        x = x / np.linalg.norm(x)
        y = y - np.dot(x, y) * x
        y = y / np.linalg.norm(y)
        delta = np.eye(4)
        delta[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
        delta[:3, 3] = row[:3]
        value = value @ delta
    return value


def main():
    prefix.environment()
    audit, result = prefix.read(AUDIT), prefix.read(RESULT)
    prefix.require(result["complete"] is result["passed"] is True, "Completed primary analysis required")
    differences = dict(position=0.0, rotation=0.0, opening=0.0)
    total, refs = 0, [AUDIT, RESULT]
    for case, reported in zip(audit["cases"], result["cases"], strict=True):
        index = case["case_index"]
        prefix.require(index == reported["case_index"], "Identical case order required")
        producer, model = (prefix.read(case[key]) for key in ("producer_ref", "model_ref"))
        refs.extend((case["producer_ref"], case["model_ref"]))
        tool = np.asarray(admission.load_inputs(index, verify_assets=False)["case"]["tool"])
        segments = producer["runtime"]["continuation"]["adaptive"]["segments"]
        captures, responses = producer["runtime"]["adaptive_captures"], model["responses"]
        prefix.require(
            len(segments) == len(captures) == len(responses) == reported["requests"], "All model windows required"
        )
        windows = []
        for segment, capture, response in zip(segments, captures, responses, strict=True):
            raw = prefix.read(response["actual_response_ref"])
            refs.append(response["actual_response_ref"])
            prefix.require(raw["status"] == "success" and len(raw["outputs"]) == 1, "Successful model output required")
            actions = np.asarray(raw["outputs"][0]["content"]["action"], dtype=float)
            prefix.require(actions.shape == (40, 128) and np.isfinite(actions).all(), "Finite H40 actions required")
            windows.append(
                dict(
                    step=segment["start_step"],
                    count=len(segment["continuation"]["periods"]),
                    anchor=np.asarray(capture["current_source_pose"]) @ tool,
                    actions=actions,
                )
            )
        for old, new, saved in zip(windows[:-1], windows[1:], reported["handoffs"], strict=True):
            n = old["count"]
            prefix.require(
                1 <= n <= 8 and new["step"] == old["step"] + n == saved["new_start_step"],
                "The two references must address the same next endpoint",
            )
            left, right = (
                integrate_scalar(old["anchor"], old["actions"], n),
                integrate_scalar(new["anchor"], new["actions"], 0),
            )
            position = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
            rotation = float(Rotation.from_matrix(left[:3, :3].T @ right[:3, :3]).magnitude())
            opening = float(abs(np.clip(old["actions"][n, 9], 0, 1) - np.clip(new["actions"][0, 9], 0, 1)))
            for key, actual, expected in (
                ("position", position, saved["position_difference_m"]),
                ("rotation", rotation, saved["rotation_difference_rad"]),
                ("opening", opening, saved["absolute_opening_difference"]),
            ):
                differences[key] = max(differences[key], abs(actual - expected))
                prefix.require(abs(actual - expected) < 1e-10, f"Independent {key} mismatch")
            total += 1
    prefix.require(total == 207, "The entire original handoff denominator is required")
    for ref in refs:
        prefix.checked(ref)
    print(
        publish_result(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                handoffs_verified=total,
                maximum_absolute_difference=differences,
                source_refs=refs + [prefix.reference(Path(__file__).resolve())],
                new_model_requests=0,
                new_physics_steps=0,
                fresh_cases=0,
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
