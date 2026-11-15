"""The original cohort roster and budgets cannot silently change."""

import copy
import json

import pytest

from scripts.simulation.runtime.prepare_reset_correct_reference_cohort import CONFIG, validate_config


@pytest.mark.parametrize("change", ["length", "task", "reorder", "budget", "retry"])
def test_fixed_correct_reference_cohort_rejects_changed_roster_or_budget(change):
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    altered = copy.deepcopy(config)
    if change == "length":
        altered["cases"][15]["periods"] = 599
    elif change == "task":
        altered["cases"][0]["task_id"] = 6
    elif change == "reorder":
        altered["cases"][0], altered["cases"][1] = altered["cases"][1], altered["cases"][0]
    elif change == "budget":
        altered["max_new_periods_per_phase"] -= 1
    else:
        altered["automatic_retry"] = True
    with pytest.raises(ValueError):
        validate_config(altered)
