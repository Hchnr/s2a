"""Capture preregistered current images during an exact saved-command replay."""

from pathlib import Path

from s2a.research.native_runtime.development_prefix_replay import ReplayObservers


def event_image_replay(replay_saved_prefix, anchors):
    """Wrap the existing replay without selecting actions or reading future labels."""
    anchors = tuple(anchors)
    if (
        not anchors
        or any(type(step) is not int or step < 0 for step in anchors)
        or tuple(sorted(set(anchors))) != anchors
    ):
        raise ValueError("Ordered, distinct actual endpoint anchors are required")

    def replay(env, commands, saved, receipts, output, observers, receipt, **kwargs):
        if anchors[-1] > len(commands) or len(receipts) != len(commands) + 1:
            raise ValueError("Every event anchor must be inside the actual replay prefix")
        directory = Path(output) / "event_images"
        directory.mkdir()
        images = []
        receipt["event_capture_partial"] = images
        receipt["unsaved_current_camera_renders"] = 0

        def endpoint(actual_env, step):
            sample, endpoint_receipt = observers.endpoint(actual_env, step)
            if step in anchors:
                if endpoint_receipt != receipts[step]:
                    raise ValueError("The current event endpoint must match the original actual receipt")
                image = observers.render_anchor(actual_env, step, sample, endpoint_receipt, directory)
                if (
                    image["frame_index"] != step
                    or image["current_endpoint"] != endpoint_receipt
                    or image["render_before"] != image["render_after"]
                ):
                    raise ValueError("Each event image must bind the unchanged current endpoint")
                images.append(image)
            return sample, endpoint_receipt

        def render_cameras(actual_env, resolution):
            receipt["unsaved_current_camera_renders"] += 1
            return observers.render_cameras(actual_env, resolution)

        wrapped = ReplayObservers(endpoint, observers.current, lambda *_args: None, render_cameras)
        value = replay_saved_prefix(env, commands, saved, receipts, output, wrapped, receipt, **kwargs)
        if [item["frame_index"] for item in images] != list(anchors):
            raise ValueError("The full registered event-image roster must be captured")
        receipt["event_images"] = images
        receipt.pop("event_capture_partial", None)
        return value

    return replay
