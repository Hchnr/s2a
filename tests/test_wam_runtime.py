"""Ensure the inference mask intervention preserves paired active noise."""

from types import SimpleNamespace

import pytest

from s2a.wam_runtime import install_active_slot_mask


def test_inactive_slots_cannot_enter_forward_or_velocity():
    torch = pytest.importorskip("torch")
    vision = torch.arange(4.0)
    actions = torch.arange(256.0).reshape(2, 128)
    flat = torch.cat([vision, actions.flatten()])
    clean = SimpleNamespace(x0_tokens_vision=[vision], x0_tokens_action=[actions], num_vision_items_per_sample=None)
    model = SimpleNamespace()
    model._prepare_inference_data = lambda: (None, clean, None, None, [flat.clone()], None, None, True)

    def denoise(*, data_batch_packed):
        token = data_batch_packed.action.tokens[0]
        assert torch.count_nonzero(token[:, 10:]) == 0
        return {"preds_action": [token + 1]}

    model.denoise = denoise
    state_before = torch.get_rng_state().clone()
    evidence = install_active_slot_mask(model, [True] * 10 + [False] * 118)
    noised = model._prepare_inference_data()[4][0]
    torch.testing.assert_close(noised[:4], vision)
    torch.testing.assert_close(noised[4:].reshape(2, 128)[:, :10], actions[:, :10])
    assert torch.count_nonzero(noised[4:].reshape(2, 128)[:, 10:]) == 0
    packed = SimpleNamespace(action=SimpleNamespace(tokens=[actions.clone()]))
    output = model.denoise(data_batch_packed=packed)["preds_action"][0]
    torch.testing.assert_close(output[:, :10], actions[:, :10] + 1)
    assert torch.count_nonzero(output[:, 10:]) == 0
    torch.testing.assert_close(torch.get_rng_state(), state_before)
    assert evidence[0]["inactive_rms_before"] > 0
