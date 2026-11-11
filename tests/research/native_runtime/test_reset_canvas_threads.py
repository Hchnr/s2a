"""Image preprocessing restores the worker's thread count even on failure."""

import pytest

from s2a.research.native_runtime.reset_canvas_threads import render_with_two_torch_threads


class FakeTorch:
    def __init__(self, threads=1):
        self.threads, self.calls = threads, []

    def get_num_threads(self):
        return self.threads

    def set_num_threads(self, threads):
        self.calls.append(threads)
        self.threads = threads


def test_render_scope_uses_two_threads_then_restores_one():
    torch, receipt = FakeTorch(), {}

    def render(value):
        assert torch.get_num_threads() == 2
        return value

    assert render_with_two_torch_threads(render, receipt, "image", torch_module=torch) == "image"
    assert torch.calls == [2, 1] and torch.threads == 1
    assert receipt["canvas_thread_scope"] == {"before": 1, "requested": 2, "inside": 2, "after": 1}


def test_failed_render_still_restores_and_non_single_thread_start_is_rejected():
    torch, receipt = FakeTorch(), {}

    def fail():
        raise RuntimeError("camera failure")

    with pytest.raises(RuntimeError, match="camera failure"):
        render_with_two_torch_threads(fail, receipt, torch_module=torch)
    assert torch.threads == 1 and receipt["canvas_thread_scope"]["after"] == 1
    torch = FakeTorch(4)
    with pytest.raises(ValueError, match="begin with one"):
        render_with_two_torch_threads(lambda: None, {}, torch_module=torch)
    assert torch.calls == []
