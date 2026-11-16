"""Cross-check actual candidate data with scalar and source reconstruction."""

import copy

import numpy as np
import pytest

from scripts.verification.wam.training import verify_recovery_candidate_data as verify
from scripts.wam.training.clean import prepare_recovery_candidate_data as primary


def actual_content():
    evidence = primary.Evidence()
    inputs = primary.candidate_inputs(evidence)
    config, config_ref, refs, images, dry, h40, union, base, norm = inputs
    rows, baseline, physical, normalized, flags, multiplicity, metrics = primary.build_candidates(
        evidence, images, dry, h40, union, base, norm
    )
    manifest = dict(
        complete=True,
        passed=True,
        schema=config["schema"],
        config_ref=config_ref,
        source_refs=refs,
        windows=rows,
        base_images=baseline,
        metrics=metrics,
        selection_uses_future_training_labels=True,
        unknown_near_duplicates_excluded=False,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
    )
    arrays = dict(
        physical_active10=physical,
        normalized_active10=normalized,
        clipped_mask=flags,
        case_transition_multiplicity=multiplicity,
    )
    return evidence, manifest, arrays, inputs


def test_real_candidate_package_passes_independent_scalar_reconstruction():
    evidence, manifest, arrays, inputs = actual_content()
    summary = verify.verify_content(evidence, manifest, arrays, inputs)
    assert summary == dict(
        windows=108, labels=4320, unique_transitions=3612, images=258, candidate_seeds=216, clipped_scalars=276
    )


def test_tampered_label_and_seed_are_rejected():
    evidence, manifest, arrays, inputs = actual_content()
    altered = {key: value.copy() for key, value in arrays.items()}
    altered["normalized_active10"][0, 0, 0] += np.float32(0.1)
    with pytest.raises(ValueError, match="Measured labels"):
        verify.verify_content(evidence, manifest, altered, inputs)
    changed = copy.deepcopy(manifest)
    changed["windows"][0]["candidate_seeds"][0]["sampling_seed"] += 1
    with pytest.raises(ValueError, match="seed collision"):
        verify.verify_content(evidence, changed, arrays, inputs)
