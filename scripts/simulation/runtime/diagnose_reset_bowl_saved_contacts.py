"""Inspect recorded bowl contact pairs without dispatching actions or advancing physics."""

import copy
import hashlib
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from s2a.research.native_runtime.saved_contacts import summarize_contacts, target_contacts
from s2a.research.native_runtime.saved_geometry_v2 import reconstruct_endpoint as reconstruct_geometry
from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry_v2 as recovery
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_saved_contacts_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_saved_contacts_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_saved_contacts_20261112"
QA = ROOT / "outputs/qa/reset_bowl_saved_contacts_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_saved_contacts_dry_20261112/result.json"
MODULE = "scripts.simulation.runtime.diagnose_reset_bowl_saved_contacts"
PRIOR = {
    "path": str(ROOT / "outputs/reset_bowl_saved_geometry_recovery_20261112/result.json"),
    "sha256": "7690a98a146d49b446167390c4eddac412c79470b0413428d55f5c4e97cd08f3",
}
FIXED = copy.deepcopy(recovery.FIXED)
FIXED.update(
    schema="s2a.reset-bowl-saved-contacts.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    prior_recovery_sha256=PRIOR["sha256"],
    prior_recovery_models=1,
    prior_recovery_forwards=97,
    left_pad="gripper0_finger1_pad_collision",
    right_pad="gripper0_finger2_pad_collision",
    target_contact_geoms=[f"akita_black_bowl_1_g{i}" for i in range(1, 41)],
)
_BASE_LOAD, _BASE_SOURCES, _BASE_RUN = recovery.load_inputs, recovery.source_refs, recovery.run_geometry


def check_xml(xml, config):
    """Validate the native group-0 object geometry before constructing a model."""
    root = ElementTree.fromstring(xml)
    bodies = root.findall(f".//body[@name='{config['body_name']}']")
    prefix.require(len(bodies) == 1, "The saved XML must have exactly one registered target body")
    body = bodies[0]
    target = [g.get("name") for g in body.iter("geom") if g.get("group", "0") == "0"]
    prefix.require(
        target == config["target_contact_geoms"]
        and len(body.findall(".//body")) == 0
        and len(body.findall("./geom[@name='akita_black_bowl_1_g0'][@group='1']")) == 1,
        "The saved XML target collision and visual groups must match the registered geometry",
    )
    for pad in (config["left_pad"], config["right_pad"]):
        geoms = root.findall(f".//geom[@name='{pad}']")
        prefix.require(len(geoms) == 1 and geoms[0].get("group", "0") == "0", "The registered pad must exist")


def load_inputs(*, verify_assets=True):
    inputs = _BASE_LOAD(verify_assets=verify_assets)
    previous = prefix.read(PRIOR)
    prefix.require(
        previous["complete"] is previous["passed"] is previous["sources_unchanged"] is True
        and previous["automatic_retry"] is False,
        "The earlier successful geometry recovery must be preserved",
    )
    worker, lifecycle = prefix.read(previous["worker_ref"]), prefix.read(previous["workers_ref"])
    counts = worker["counters"]
    prefix.require(
        worker["complete"] is worker["passed"] is True
        and counts["models"] == counts["data"] == 1
        and counts["forwards_attempted"] == counts["forwards_completed"] == 97
        and counts["physics_steps"] == counts["contexts"] == counts["model_requests"] == 0
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False,
        "The earlier 97 forwards and closed worker must be accounted for",
    )
    check_xml(inputs["case"]["record_xml"], inputs["config"])
    for key, ref in (
        ("prior_recovery", PRIOR),
        ("prior_recovery_worker", previous["worker_ref"]),
        ("prior_recovery_workers", previous["workers_ref"]),
    ):
        prefix.checked(ref, inputs["source_sha256"])
        inputs["refs"][key] = ref
    inputs["prior_geometry"] = worker["geometry"]
    return inputs


def source_refs():
    extra = (
        recovery.CONFIG,
        recovery.PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/native_runtime/saved_contacts.py",
        ROOT / "tests/research/native_runtime/test_saved_contacts.py",
        ROOT / "tests/research/native_runtime/test_diagnose_reset_bowl_saved_contacts.py",
        ROOT / "reports/tracking/references/reset_bowl_saved_geometry_recovery_20261112.md",
    )
    return _BASE_SOURCES() + [prefix.reference(path) for path in extra]


def reconstruct_endpoint(model, data, state, *, site_id, body_id, site_from_tcp, forward):
    import mujoco

    point = reconstruct_geometry(
        model, data, state, site_id=site_id, body_id=body_id, site_from_tcp=site_from_tcp, forward=forward
    )
    point["contacts"] = target_contacts(
        model,
        data,
        mujoco=mujoco,
        body_id=body_id,
        target_names=FIXED["target_contact_geoms"],
        left_pad=FIXED["left_pad"],
        right_pad=FIXED["right_pad"],
    )
    return point


def run_geometry(inputs, counters, *, mujoco):
    geometry = _BASE_RUN(inputs, counters, mujoco=mujoco)
    prior = inputs["prior_geometry"]
    pairs = [(geometry["reset"], prior["reset"])] + [
        (point, old)
        for key in ("fixed", "reobserved")
        for point, old in zip(geometry["points"][key], prior["points"][key], strict=True)
    ]
    prefix.require(len(pairs) == 97, "All earlier reconstructed endpoints must be checked")
    for current, old in pairs:
        for name in ("tcp_world_m", "body_origin_world_m"):
            np.testing.assert_allclose(current[name], old[name], atol=1e-12, rtol=0)
        if "state_sha256" in current:
            prefix.require(
                all(current[key] == old[key] for key in ("state_sha256", "target_grasp", "task_success")),
                "Endpoint identities and original task labels must remain unchanged",
            )
    geometry["contact_summaries"] = {name: summarize_contacts(rows) for name, rows in geometry["points"].items()}
    geometry["prior_geometry_checks"] = len(pairs)
    return geometry


def bind_contacts():
    pinned = (
        (Path(recovery.__file__), "e19d6724fbea99d4819fa380aece6ecbfd867a4842f19659ce721e3f8765fd5b"),
        (
            ROOT / "src/s2a/research/native_runtime/saved_geometry_v2.py",
            "1dbd196e8e46218bb78305dc9eae4c7565473583fed28397097b475fda1edc59",
        ),
    )
    prefix.require(
        all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in pinned),
        "The prior recovery sources must remain unchanged",
    )
    engine = recovery.bind_recovery()
    engine.CONFIG, engine.PLAN, engine.OUTPUT, engine.QA, engine.DRY = CONFIG, PLAN, OUTPUT, QA, DRY
    engine.MODULE, engine.FIXED = MODULE, FIXED
    engine.reconstruct_endpoint = reconstruct_endpoint
    engine.load_inputs, engine.source_refs, engine.run_geometry = load_inputs, source_refs, run_geometry
    return engine


def main():
    bind_contacts().main()


if __name__ == "__main__":
    main()
