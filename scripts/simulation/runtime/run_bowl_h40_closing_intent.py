"""Own a bounded amplified-closing probe and independently replay commands."""

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np

from s2a.references.finite_horizon_window import remaining_h40
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as admission
from scripts.simulation.runtime import run_bowl_h40_opening_lead as lead
from scripts.simulation.runtime import run_reset_bowl_fixed_h40 as owner

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/gripper/bowl_h40_closing_intent_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/gripper/bowl_h40_closing_intent_probe_20261112.md"
OUTPUT = ROOT / "outputs/bowl_h40_closing_intent_20261112"
QA = ROOT / "outputs/qa/bowl_h40_closing_intent_20261112/result.json"
DRY = ROOT / "outputs/bowl_h40_closing_intent_dry_20261112/result.json"
MODULE = "scripts.simulation.runtime.run_bowl_h40_closing_intent"
LEAD_AUDIT = {
    "path": str(ROOT / "outputs/bowl_h40_opening_lead_audit_20261112/result.json"),
    "sha256": "005cb8a5e91c8625f2bb7befee71cd9978d0d1fb1276b75629769ec4d67b8577",
}
STATIC_AUDIT = {
    "path": str(ROOT / "outputs/bowl_static_close_feasibility_recovery_audit_20261112/result.json"),
    "sha256": "ca563b70af4bd0514dfc7dacd997acd95eb7aca9581e34e45d4a9bdf9c20ddd3",
}
FIXED = copy.deepcopy(lead.FIXED)
FIXED.update(
    schema="s2a.bowl-h40-closing-intent.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    opening_lead_steps=8,
    opening_gain=3.0,
    before_probe_periods=468,
    before_probe_inner_steps=11700,
    opening_lead_audit_sha256=LEAD_AUDIT["sha256"],
    static_close_audit_sha256=STATIC_AUDIT["sha256"],
)
_BASE_LOAD, _BASE_SOURCES = lead._OLD_LOAD, lead._OLD_SOURCES
_PINNED = (
    *lead._PINNED,
    (
        "scripts/simulation/runtime/run_bowl_h40_opening_lead.py",
        "1800bdca8bc8d7d7297d7000190f2ce2d290d6a4a06bc7d981a92bb4a24b0d75",
    ),
    ("src/s2a/references/opening_lead.py", "45574c8727a65ccddb18d2ea6c4780ed65feac8f3b54b790201932327f8cf216"),
)


def load_inputs(*, verify_assets=True):
    inputs = _BASE_LOAD(verify_assets=verify_assets)
    baseline, contacts, led, static = (
        prefix.read(ref) for ref in (lead.BASELINE, lead.CONTACTS, LEAD_AUDIT, STATIC_AUDIT)
    )
    config = inputs["config"]
    prefix.require(
        all(
            x["complete"] is x["passed"] is True and x["control_qualified"] is False
            for x in (baseline, contacts, led, static)
        )
        and baseline["new_producer_periods"] == baseline["new_checker_periods"] == 48
        and led["new_producer_periods"] == led["new_checker_periods"] == 48
        and led["cumulative_periods"] == config["before_probe_periods"]
        and led["cumulative_inner_steps"] == config["before_probe_inner_steps"]
        and static["summaries"]["fixed"]["witness_count"] > 0
        and static["summaries"]["lead"]["witness_count"] > 0
        and contacts["audit_forwards"] == static["audit_forwards"] == 0,
        "The same audited baselines, static witnesses and prior costs are required",
    )
    refs = dict(
        fixed_h40_audit=lead.BASELINE,
        saved_contacts_audit=lead.CONTACTS,
        opening_lead_audit=LEAD_AUDIT,
        static_close_audit=STATIC_AUDIT,
        fixed_h40_producer=baseline["producer_branch_ref"],
        fixed_h40_checker=baseline["checker_branch_ref"],
        lead_producer=led["producer_branch_ref"],
        lead_checker=led["checker_branch_ref"],
    )
    for key, ref in refs.items():
        prefix.checked(ref, inputs["source_sha256"])
        inputs["refs"][key] = ref
    return inputs


def source_refs():
    files = (
        lead.CONFIG,
        lead.PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/references/opening_intent.py",
        ROOT / "tests/unit/references/test_opening_intent.py",
        ROOT / "tests/research/native_runtime/test_bowl_h40_closing_intent.py",
        ROOT / "reports/tracking/gripper/bowl_h40_opening_lead_20261112.md",
        ROOT / "reports/tracking/gripper/bowl_static_close_feasibility_recovery_20261112.md",
    )
    return _BASE_SOURCES() + [prefix.reference(path) for path in files]


def checker_window(poses, openings, *, consumed):
    """Rebuild the transformation independently without importing the candidate helper."""
    window_poses, _, metadata = remaining_h40(poses, openings, consumed=consumed)
    source = np.asarray(openings, dtype=float)
    indices = np.minimum(consumed + np.arange(40, dtype=int) + 8, 39)
    transformed = np.clip(3 * source[indices] - 2, 0, 1)
    generated = max(0, 32 - consumed)
    metadata.update(
        opening_lead_steps=8,
        opening_gain=3.0,
        opening_generated_endpoints=generated,
        opening_terminal_hold_endpoints=40 - generated,
    )
    return window_poses, transformed, metadata


def bind(phase=None):
    prefix.require(
        all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha for path, sha in _PINNED),
        "Earlier fixed-reference and lead sources must remain unchanged",
    )
    admission.CONFIG, admission.PLAN, admission.OUTPUT, admission.QA, admission.DRY = CONFIG, PLAN, OUTPUT, QA, DRY
    admission.FIXED, admission.load_inputs, admission.source_refs = FIXED, load_inputs, source_refs
    owner.OUTPUT, owner.QA, owner.DRY, owner.MODULE = OUTPUT, QA, DRY, MODULE
    if phase == "producer":
        from s2a.references.opening_intent import amplified_opening_h40
        from scripts.simulation.runtime import reset_bowl_fixed_h40 as producer

        producer.remaining_h40 = lambda poses, openings, *, consumed: amplified_opening_h40(
            poses, openings, consumed=consumed, lead_steps=8, gain=3.0
        )
    elif phase == "checker":
        from scripts.simulation.runtime import reset_bowl_fixed_h40_replay as checker

        checker.remaining_h40 = checker_window
    else:
        prefix.require(phase is None, "Only registered producer or checker phases are allowed")
    return owner


def main():
    phase = sys.argv[sys.argv.index("--worker") + 1] if "--worker" in sys.argv else None
    bind(phase).main()


if __name__ == "__main__":
    main()
