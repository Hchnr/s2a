"""Own the registered opening-lead probe and independent command replay."""

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np

from s2a.references.finite_horizon_window import remaining_h40
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import reset_bowl_fixed_h40_admission as admission
from scripts.simulation.runtime import run_reset_bowl_fixed_h40 as owner

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/gripper/bowl_h40_opening_lead_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/gripper/bowl_h40_opening_lead_20261112.md"
OUTPUT = ROOT / "outputs/bowl_h40_opening_lead_20261112"
QA = ROOT / "outputs/qa/bowl_h40_opening_lead_20261112/result.json"
DRY = ROOT / "outputs/bowl_h40_opening_lead_dry_20261112/result.json"
MODULE = "scripts.simulation.runtime.run_bowl_h40_opening_lead"
BASELINE = {
    "path": str(ROOT / "outputs/reset_bowl_fixed_h40_audit_20261112/result.json"),
    "sha256": "df5e3d2a7ff15fb432f22aad566e4b35c5ea092e1568b3552c05f52484442e8b",
}
CONTACTS = {
    "path": str(ROOT / "outputs/reset_bowl_saved_contacts_audit_20261112/result.json"),
    "sha256": "678b9e3d3a2e56cc89808912e359c6cb6b3f94d356e46ff2e2adf42c9df33b50",
}
FIXED = copy.deepcopy(admission.FIXED)
FIXED.update(
    schema="s2a.bowl-h40-opening-lead.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    opening_lead_steps=4,
    before_probe_periods=372,
    before_probe_inner_steps=9300,
    fixed_h40_audit_sha256=BASELINE["sha256"],
    saved_contacts_audit_sha256=CONTACTS["sha256"],
)
_OLD_CONFIG, _OLD_PLAN = admission.CONFIG, admission.PLAN
_OLD_LOAD, _OLD_SOURCES = admission.load_inputs, admission.source_refs
_PINNED = (
    (
        "scripts/simulation/runtime/run_reset_bowl_fixed_h40.py",
        "310314f3c7ac886c963599b04c6adcabb0ff02fdc10532c5a35029bb5ed9a39a",
    ),
    (
        "scripts/simulation/runtime/reset_bowl_fixed_h40.py",
        "2fda351920c929ca223b4c8d54d02c060dc400612a4f3c2f94b2f8a57875420d",
    ),
    (
        "scripts/simulation/runtime/reset_bowl_fixed_h40_replay.py",
        "cdddd85bf7a8604387abc8841779edc0ce589c51417f8bcad44a1760640bfdbb",
    ),
    (
        "scripts/simulation/runtime/reset_bowl_fixed_h40_admission.py",
        "fdaa99ddea261b376b223cf2e13691e120a07d9e0d396e103f515e50073dccd6",
    ),
    ("src/s2a/references/finite_horizon_window.py", "8610664be72a0c7668e2d92f0ea89d3e4e7e7d8434cd26060755cab07bdf8b0b"),
)


def load_inputs(*, verify_assets=True):
    inputs = _OLD_LOAD(verify_assets=verify_assets)
    baseline, contacts = prefix.read(BASELINE), prefix.read(CONTACTS)
    config = inputs["config"]
    prefix.require(
        baseline["complete"] is baseline["passed"] is contacts["complete"] is contacts["passed"] is True
        and baseline["control_qualified"] is contacts["control_qualified"] is False
        and baseline["new_producer_periods"] == baseline["new_checker_periods"] == 48
        and baseline["cumulative_periods"] == config["before_probe_periods"]
        and baseline["cumulative_inner_steps"] == config["before_probe_inner_steps"]
        and baseline["audit_physics_steps"] == baseline["audit_model_requests"] == 0
        and contacts["audit_physics_steps"] == contacts["audit_forwards"] == 0,
        "The fixed-H40 and saved-contact audits with exact previous costs are required",
    )
    refs = dict(
        fixed_h40_audit=BASELINE,
        saved_contacts_audit=CONTACTS,
        fixed_h40_producer=baseline["producer_branch_ref"],
        fixed_h40_checker=baseline["checker_branch_ref"],
    )
    for key, ref in refs.items():
        prefix.checked(ref, inputs["source_sha256"])
        inputs["refs"][key] = ref
    return inputs


def source_refs():
    files = (
        _OLD_CONFIG,
        _OLD_PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/references/opening_lead.py",
        ROOT / "tests/unit/references/test_opening_lead.py",
        ROOT / "tests/research/native_runtime/test_bowl_h40_opening_lead.py",
        ROOT / "reports/tracking/references/reset_bowl_fixed_h40_probe_20261112.md",
        ROOT / "reports/tracking/references/reset_bowl_saved_contacts_20261112.md",
    )
    return _OLD_SOURCES() + [prefix.reference(path) for path in files]


def checker_window(poses, openings, *, consumed):
    """Independently reconstruct the specified index shift without the candidate helper."""
    window_poses, _, metadata = remaining_h40(poses, openings, consumed=consumed)
    source = np.asarray(openings, dtype=float)
    indices = np.minimum(consumed + np.arange(40, dtype=int) + 4, 39)
    generated = max(0, 36 - consumed)
    metadata.update(
        opening_lead_steps=4, opening_generated_endpoints=generated, opening_terminal_hold_endpoints=40 - generated
    )
    return window_poses, source[indices].copy(), metadata


def bind(phase=None):
    prefix.require(
        all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha for path, sha in _PINNED),
        "The earlier fixed-reference implementation must remain unchanged",
    )
    admission.CONFIG, admission.PLAN, admission.OUTPUT, admission.QA, admission.DRY = CONFIG, PLAN, OUTPUT, QA, DRY
    admission.FIXED, admission.load_inputs, admission.source_refs = FIXED, load_inputs, source_refs
    owner.OUTPUT, owner.QA, owner.DRY, owner.MODULE = OUTPUT, QA, DRY, MODULE
    if phase == "producer":
        from s2a.references.opening_lead import opening_lead_h40
        from scripts.simulation.runtime import reset_bowl_fixed_h40 as producer

        producer.remaining_h40 = lambda poses, openings, *, consumed: opening_lead_h40(
            poses, openings, consumed=consumed, lead_steps=4
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
