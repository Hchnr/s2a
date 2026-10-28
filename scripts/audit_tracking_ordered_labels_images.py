"""Check saved endpoint labels, masks, image bindings, and replay camera pixels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.geometry import validate_transform
from s2a.native_training import state_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-index", type=int, required=True, choices=range(18))
    args = p.parse_args()
    root = REPO / f"outputs/tracking_ordered_labels_images_20261027/case{args.case_index:02d}"
    require(not root.exists(), "No overwrite")
    run = REPO / f"outputs/tracking_correct_reference_ordered_20261027/case{args.case_index:02d}_candidate"
    result_ref = reference(run / "result.json")
    result = read(result_ref)
    require(result["complete"] is result["passed"] is True, "Completed run")
    prepared = read(reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json"))
    tool = np.asarray(read(prepared["profiles"][args.case_index]["tool_ref"])["source_from_tcp"])
    replay_root = REPO / f"outputs/tracking_correct_reference_ordered_check_20261027/case{args.case_index:02d}"
    replay_ref = reference(replay_root / "verified.json") if (replay_root / "verified.json").exists() else None
    if replay_ref:
        require(
            read(replay_ref)["complete"] is read(replay_ref)["passed"] is True,
            "Complete native replay for pixel comparison",
        )
    from s2a.libero_simulation import canvas

    with np.load(checked(result["arrays_ref"]), allow_pickle=False) as a:
        n = result["physics_steps"]
        require(len(a["source_pose"]) == n + 1, "Complete endpoints")
        np.testing.assert_array_equal(a["endpoint_step"], np.arange(n + 1))
        np.testing.assert_allclose(np.diff(a["sim_time"]), 0.05, atol=1e-9, rtol=0)
        tcp = validate_transform(a["source_pose"] @ tool)
        np.testing.assert_allclose(a["tcp_pose"], tcp, atol=1e-12, rtol=0)
        state = np.zeros((n + 1, 128))
        state[:, :3] = tcp[:, :3, 3]
        state[:, 3:6] = tcp[:, :3, 0]
        state[:, 6:9] = tcp[:, :3, 1]
        state[:, 9] = a["opening"]
        delta = np.linalg.inv(tcp[:-1]) @ tcp[1:]
        action = np.zeros_like(state)
        action[:-1, :3] = delta[:, :3, 3]
        action[:-1, 3:6] = delta[:, :3, 0]
        action[:-1, 6:9] = delta[:, :3, 1]
        action[:-1, 9] = a["opening"][1:]
        np.testing.assert_allclose(a["state128"], state, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a["action128"], action, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a["delta_pose"], delta, atol=1e-12, rtol=0)
        state_mask = np.zeros_like(state, dtype=bool)
        state_mask[:, :10] = True
        action_mask = state_mask.copy()
        action_mask[-1] = False
        valid = np.arange(n + 1) < n
        for key, expected in [("state_mask", state_mask), ("action_mask", action_mask), ("action_valid", valid)]:
            require(a[key].dtype == np.bool_, "Boolean masks")
            np.testing.assert_array_equal(a[key], expected)
        np.testing.assert_allclose(
            a["source_pose"][:, :3, 3], a["actual_source_site_pose"][:, :3, 3], atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            a["source_pose"][:, :3, :3], a["actual_eef_body_pose"][:, :3, :3], atol=1e-10, rtol=0
        )
        require([im["frame_index"] for im in result["images"]] == [0, 16, 56], "Fixed camera endpoints")
        for image in result["images"]:
            index = image["frame_index"]
            expected = dict(sim_time=float(a["sim_time"][index]), sim_state_sha256=state_sha256(a["sim_states"][index]))
            require(image["render_before"] == image["render_after"] == expected, "Current physical endpoint binding")
            require(
                image["current_endpoint"]["sim_state_sha256"] == expected["sim_state_sha256"]
                and image["future_rgb_frames"] == 0,
                "Current-only image receipt",
            )
            np.testing.assert_allclose(image["current_source_pose"], a["source_pose"][index], atol=1e-12, rtol=0)
            np.testing.assert_allclose(image["current_opening"], a["opening"][index], atol=1e-12, rtol=0)
            require(
                image["camera_roles"] == ["auxiliary_0", "left_wrist"] and image["live_flip_axes"] == [0, 1],
                "Camera convention",
            )
            with np.load(checked(image["raw_cameras"]), allow_pickle=False) as cameras:
                frames = []
                for j, key in enumerate(("agentview_image", "robot0_eye_in_hand_image")):
                    require(cameras[key].dtype == np.uint8 and cameras[key].shape == (256, 256, 3), "Raw RGB")
                    frames.append(cameras[key][::-1, ::-1].copy())
                    np.testing.assert_array_equal(cameras[f"live_{image['camera_roles'][j]}"], frames[-1])
                    if replay_ref:
                        with np.load(
                            replay_root / f"anchor{index:04d}/current_cameras.npz", allow_pickle=False
                        ) as replay:
                            np.testing.assert_array_equal(cameras[key], replay[key])
                pixels = canvas(frames)
            with Image.open(checked(image["canvas_png"])) as png:
                np.testing.assert_array_equal(pixels, np.asarray(png.convert("RGB")))
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == image["canvas_rgb_sha256"], "Canvas RGB hash")
    ref = publish(
        root / "verified.json",
        dict(
            complete=True,
            passed=True,
            result_ref=result_ref,
            source_ref=reference(Path(__file__)),
            replay_ref=replay_ref,
            replay_pixels_compared=bool(replay_ref),
            endpoint_count=n + 1,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            scope="Saved labels/masks and image binding; raw pixel replay comparison only when replay_pixels_compared is true.",
        ),
    )
    print(json.dumps(ref), flush=True)


if __name__ == "__main__":
    main()
