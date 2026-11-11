"""Restore recorded canvas arithmetic without changing the historical runtime."""

from s2a.research.native_runtime.development_prefix_replay import ReplayObservers
from s2a.research.native_runtime.development_reset_runtime import DevelopmentResetRuntime


def render_with_two_torch_threads(render, receipt, *args, torch_module=None, **kwargs):
    """Use two threads only for this render/canvas call, then restore one."""
    if torch_module is None:
        import torch as torch_module
    before = torch_module.get_num_threads()
    if before != 1:
        raise ValueError("The reset worker must begin with one Torch thread")
    scope = receipt["canvas_thread_scope"] = {"before": before, "requested": 2, "inside": None, "after": None}
    try:
        torch_module.set_num_threads(2)
        scope["inside"] = torch_module.get_num_threads()
        if scope["inside"] != 2:
            raise ValueError("The registered two-thread canvas scope was not applied")
        return render(*args, **kwargs)
    finally:
        torch_module.set_num_threads(1)
        scope["after"] = torch_module.get_num_threads()
        if scope["after"] != 1:
            raise RuntimeError("The Torch thread count could not be restored")


class TwoThreadCanvasResetRuntime(DevelopmentResetRuntime):
    """Wrap only render_anchor in the isolated zero-history runtime namespace."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        observe = self._run.__globals__["replay_saved_prefix"]

        def isolated_observe(env, commands, saved, receipts, output, observers, receipt, **callback_kwargs):
            def render(*args, **render_kwargs):
                return render_with_two_torch_threads(observers.render_anchor, receipt, *args, **render_kwargs)

            isolated = ReplayObservers(observers.endpoint, observers.current, render, observers.render_cameras)
            return observe(env, commands, saved, receipts, output, isolated, receipt, **callback_kwargs)

        self._run.__globals__["replay_saved_prefix"] = isolated_observe
