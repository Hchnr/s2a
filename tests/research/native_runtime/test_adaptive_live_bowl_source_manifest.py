"""Check model provenance in a fresh CPU process and after legacy binding."""

import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_source_manifest as manifest
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def test_fresh_process_and_bound_legacy_adapter_produce_the_same_model_sources():
    script = """\
import sys
from scripts.simulation.runtime import adaptive_live_bowl_source_manifest as manifest
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_replan_step107_model as model

fresh = manifest.causal_model_source_refs()
assert fresh[:2] == [prefix.reference(model.CONFIG), prefix.reference(model.PLAN)]
model.bind()
bound = manifest.causal_model_source_refs()
assert fresh == bound == model.source_refs()
assert not any(name in sys.modules for name in ('torch', 'mujoco', 'robosuite', 'libero', 'cosmos_framework'))
"""
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        LP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:.",
    )
    subprocess.run([sys.executable, "-c", script], cwd=prefix.REPO, env=env, check=True, timeout=600)


def test_rejects_unexpected_inherited_source_layout(monkeypatch):
    monkeypatch.setattr(manifest.model, "source_refs", lambda: [{"path": "unexpected", "sha256": "0" * 64}])
    with pytest.raises(ValueError, match="must begin with its active config and plan"):
        manifest.causal_model_source_refs()
