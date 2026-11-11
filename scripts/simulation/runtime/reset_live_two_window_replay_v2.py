"""Replay with the same minimal renderer endpoint schema as the producer."""

from types import SimpleNamespace

from scripts.simulation.runtime import reset_live_two_window_replay as previous


def minimal_renderer_endpoint(endpoint, step):
    """Preserve the current endpoint's actual identity while omitting metadata."""
    if type(endpoint) is not dict or type(step) is not int or endpoint.get("endpoint_step") != step:
        raise ValueError("One actual current endpoint and matching step are required")
    keys = ("endpoint_step", "sim_time", "sim_state_sha256")
    if any(key not in endpoint for key in keys):
        raise ValueError("Current endpoint identity is incomplete")
    return {key: endpoint[key] for key in keys}


def run_checker(env, terminal, observers, receipt, **kwargs):
    """Keep the complete checker receipt and normalize only the renderer argument."""

    def render(actual_env, step, sample, endpoint, output):
        identity = minimal_renderer_endpoint(endpoint, step)
        return observers.render_anchor(actual_env, step, sample, identity, output)

    wrapped = SimpleNamespace(
        endpoint=observers.endpoint,
        current=observers.current,
        render_anchor=render,
        render_cameras=observers.render_cameras,
    )
    return previous.run_checker(env, terminal, wrapped, receipt, **kwargs)
