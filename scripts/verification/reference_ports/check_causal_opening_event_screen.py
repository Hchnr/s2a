"""Independently check recorded release-event admission without a simulator."""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

REPO = prefix.REPO
SCREEN = {
    "path": str(REPO / "outputs/causal_opening_event_screen_20261111/result.json"),
    "sha256": "d511e62fafa8817797637859c4035a4986926c4ce3be4dd837a6eb8b5fc297be",
}
OUTPUT = REPO / "outputs/causal_opening_event_screen_check_20261111"
DRY = REPO / "outputs/causal_opening_event_screen_check_dry_20261111/result.json"
QA = REPO / "outputs/qa/causal_opening_event_screen_check_20261111/result.json"
FORBIDDEN = (
    "scripts.simulation.runtime.screen_causal_opening_events",
    "s2a.research.joint_control.preview.opening_event_admission",
)


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            Path(__file__),
            Path(prefix.__file__),
            REPO / "configs/tracking/joint_control/causal_opening_event_feedback_20261111.json",
            REPO / "plans/v0.0.1/tracking/joint_control/causal_opening_event_feedback_20261111.md",
            REPO / "tests/research/native_runtime/test_causal_opening_screen_check.py",
        )
    ]


def select_chunk(path, target):
    """Decode no future chunks beyond the queried current chunk."""
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    position = text.index("[") + 1
    for index in range(target + 1):
        while text[position].isspace():
            position += 1
        if index:
            if text[position] != ",":
                raise ValueError("Invalid original chunk separator")
            position += 1
            while text[position].isspace():
                position += 1
        item, position = decoder.raw_decode(text, position)
    return item


def independent_limits(xml):
    root = ElementTree.fromstring(xml)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Missing original worldbody")
    joints = [node for node in world.iter() if node.tag in ("joint", "freejoint")]
    names = [f"robot0_joint{i}" for i in range(1, 8)]
    if [node.get("name") for node in joints[:7]] != names:
        raise ValueError("Unexpected original joint state ordering")
    bounds = []
    for index, name in enumerate(names):
        if sum(node.get("name") == name for node in joints) != 1:
            raise ValueError("Duplicate or missing joint identity")
        node = joints[index]
        if node.tag != "joint" or node.get("type", "hinge") not in ("hinge", "slide"):
            raise ValueError("Non-scalar original robot joint")
        try:
            values = [float(value) for value in node.attrib["range"].split()]
        except (KeyError, ValueError) as error:
            raise ValueError("Missing original joint range") from error
        if len(values) != 2 or not np.isfinite(values).all() or values[0] >= values[1]:
            raise ValueError("Invalid original joint range")
        bounds.append(values)
    return names, bounds


def check_case(data):
    spec, refs, recorded = data["case"], data["input_refs"], data["admission"]
    n, source = spec["prefix_steps"], spec["source_step"]
    manifest = prefix.read(refs["manifest"])
    with prefix.checked(refs["steps"]).open() as stream:
        row = json.loads(next(itertools.islice(stream, n, n + 1)))
    q = np.asarray(row["qualified_forward_input"]["x_joint_position"], dtype=float).reshape(-1)
    prefix.require(row["step"] == n and q.shape == (7,) and np.isfinite(q).all(), "Current pre-action state")
    names, limits = independent_limits(manifest["record"]["model_xml"])
    margins = [min(float(value) - lo, hi - float(value)) for value, (lo, hi) in zip(q, limits, strict=True)]
    np.testing.assert_array_equal(q, recorded["joint_positions_rad"])
    np.testing.assert_array_equal(limits, recorded["joint_limits_rad"])
    np.testing.assert_allclose(margins, recorded["joint_margins_rad"], atol=1e-15, rtol=0)
    prefix.require(
        recorded["joint_names"] == names
        and recorded["buffer_rad"] == 0.05
        and recorded["recorded_endpoint_admitted"] == (min(margins) >= 0.05)
        and recorded["minimum_margin_rad"] == min(margins),
        "Original 0.05-rad admission and per-joint margins",
    )
    with np.load(prefix.checked(refs["arrays"]), allow_pickle=False) as archive:
        state = np.asarray(archive["sim_states"][n], dtype=np.float64)
        saved_opening = float(archive["opening"][n])
    np.testing.assert_array_equal(state[1:8], q)
    prefix.require(
        hashlib.sha256(state.tobytes()).hexdigest() == data["current_state_sha256"]
        and abs(saved_opening - row["opening_before"]) <= 1e-12
        and abs(saved_opening - data["current_opening"]) <= 1e-12,
        "Independent original endpoint and aperture binding",
    )
    shadow = prefix.read(refs["shadow"])["data"]
    checked = prefix.read(refs["checker"])["data"]
    record = shadow["records"][n // 8]
    old, latest = (select_chunk(prefix.checked(refs["chunks"]), step // 8) for step in (source, n))
    indices = list(range(n - source, n - source + 8))
    prefix.require(
        shadow["case"]["case_id"] == checked["case"]["case_id"] == spec["case_id"]
        and shadow["first_intervention_step"] == checked["first_intervention_step"] == n
        and record["source"] == "remembered"
        and record["source_step"] == old["step"] == source
        and latest["step"] == n
        and record["source_indices"] == indices
        and record["absolute_endpoints"] == list(range(n + 1, n + 9))
        and record["event"]["direction"] == "opening"
        and record["event"]["absolute_endpoint"] == spec["event_endpoint"]
        and record["baseline_observation_label"] == "first_intervention_on_baseline",
        "First audited opening and causal absolute reference endpoints",
    )
    np.testing.assert_array_equal(record["poses"], np.asarray(old["effective_targets_tcp"])[indices])
    np.testing.assert_array_equal(record["openings"], np.asarray(old["openings"])[indices])
    prefix.require(
        old["openings"][spec["event_endpoint"] - source - 1] > 0.8
        and saved_opening < 0.35
        and not data["prior_inner_samples_checked"]
        and not data["physical_feedback_executed"],
        "Original opening event and static-screen evidence scope",
    )
    for ref in refs.values():
        prefix.checked(ref)
    return dict(
        case=spec,
        minimum_margin_rad=min(margins),
        recorded_endpoint_admitted=min(margins) >= 0.05,
        prior_inner_samples_checked=False,
        physical_feedback_executed=False,
    )


def load_screen():
    screen = prefix.read(SCREEN)
    prefix.require(
        screen["complete"] is screen["passed"] is screen["sources_unchanged"] is True
        and len(screen["case_refs"]) == 4
        and screen["model_initializations"]
        == screen["physics_steps"]
        == screen["world_model_calls"]
        == screen["fresh_cases"]
        == 0,
        "Completed four-case static screen without model or physics",
    )
    cases = [prefix.read(ref)["data"] for ref in screen["case_refs"]]
    prefix.require([item["case"]["case_index"] for item in cases] == [3, 7, 10, 11], "Original full case order")
    return screen, cases


def worker(index):
    prefix.environment()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    screen, cases = load_screen()
    prefix.require(
        screen["case_refs"] == registration["case_refs"] and source_refs() == registration["source_refs"],
        "Frozen independent inputs",
    )
    data = next(item for item in cases if item["case"]["case_index"] == index)
    print(
        publish(OUTPUT / f"case_{index:02d}" / "result.json", dict(complete=True, passed=True, data=check_case(data)))
    )


def execute():
    prefix.environment()
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Independent checker QA closed",
    )
    before = snapshot(REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "Checker QA source identity")
    screen, cases = load_screen()
    sources = source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["source_refs"] == sources
        and dry["case_refs"] == screen["case_refs"],
        "Checker dry identity",
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    registration = publish(OUTPUT / "registration.json", {"case_refs": screen["case_refs"], "source_refs": sources})
    specs = [
        dict(
            key=f"case_{item['case']['case_index']:02d}",
            argv=[
                sys.executable,
                "-m",
                "scripts.verification.reference_ports.check_causal_opening_event_screen",
                "--worker",
                str(item["case"]["case_index"]),
            ],
            log=OUTPUT / f"case_{item['case']['case_index']:02d}" / "worker.log",
        )
        for item in cases
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=1800)
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(OUTPUT / "workers.json", lifecycle)
    result = dict(
        complete=False, passed=False, registration_ref=registration, workers_ref=workers, automatic_retry=False
    )
    try:
        prefix.require(
            not lifecycle["launch_error"]
            and len(lifecycle["workers"]) == 4
            and all(
                item["returncode"] == 0 and item["actual_wait"] and item["reaped"] and not item["signal_sent"]
                for item in lifecycle["workers"]
            ),
            "All independent workers exited and were reaped",
        )
        case_refs = [
            prefix.reference(OUTPUT / f"case_{item['case']['case_index']:02d}" / "result.json") for item in cases
        ]
        checked = [prefix.read(ref)["data"] for ref in case_refs]
        prefix.require(
            [item["case"]["case_index"] for item in checked if item["recorded_endpoint_admitted"]]
            == screen["admitted_cases"]
            and [item["case"]["case_index"] for item in checked if not item["recorded_endpoint_admitted"]]
            == screen["rejected_cases"]
            and source_refs() == sources
            and snapshot(REPO)["sources"] == before,
            "Independent case decisions and sources agree",
        )
        result.update(
            complete=True,
            passed=True,
            sources_unchanged=True,
            case_refs=case_refs,
            admitted_cases=screen["admitted_cases"],
            rejected_cases=screen["rejected_cases"],
            model_initializations=0,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
        )
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=int, choices=[3, 7, 10, 11])
    args = parser.parse_args()
    prefix.environment()
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "Checker cannot import screen implementation")
    if args.dry_run:
        screen, cases = load_screen()
        for case in cases:
            check_case(case)
        print(
            publish(
                DRY,
                dict(
                    complete=True,
                    passed=True,
                    case_refs=screen["case_refs"],
                    source_refs=source_refs(),
                    model_initializations=0,
                    physics_steps=0,
                ),
            )
        )
    elif args.run:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
