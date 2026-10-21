"""026 generation delegation and complete zero-output admission."""

import copy
import importlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sampling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_wam_flow_readout_sampling")


def test_generation_request_delegates_domain_name_and_actual_seed_once(sampling):
    calls = []

    class Bridge:
        @contextmanager
        def request(self, context, **kwargs):
            calls.append((context, kwargs))
            yield

    proxy = sampling.RequestBridge(Bridge())
    with pytest.raises(ValueError, match="unarmed"):
        proxy.request("domain")
    proxy.entry = dict(name="actual-name", sampling_seed=43)
    with proxy.request("full-domain"):
        pass
    assert calls == [("full-domain", dict(name="actual-name", seed=43))]


def test_zero_requires_all40x128_raw_and_normalized_values_exact(sampling):
    zeros = np.zeros((40, 128)).tolist()
    response = dict(outputs=[dict(content=dict(action=zeros, action128=dict(normalized_action=zeros)))])
    sampling.check_zero(response, copy.deepcopy(response))
    changed = copy.deepcopy(response)
    changed["outputs"][0]["content"]["action128"]["normalized_action"][-1][-1] = 1e-30
    with pytest.raises(ValueError, match="complete raw/normalized"):
        sampling.check_zero(changed, response)
    short = copy.deepcopy(response)
    short["outputs"][0]["content"]["action"] = zeros[:8]
    with pytest.raises(ValueError, match="complete raw/normalized"):
        sampling.check_zero(short, response)
