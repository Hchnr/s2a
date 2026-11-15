"""Collect current wrench while independently replaying two recorded action streams."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.gripper.current_wrench import read_current_wrench
from scripts.simulation.runtime import causal_stream_live_48_replay as references
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as admission
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_early_causal_innovation_physical as early
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as previous

ROOT = prefix.REPO
OUTPUT = ROOT / "outputs/wrench_endpoint_replay_20261113"
BRANCHES = {
    "historical": (
        "outputs/early_causal_innovation_physical_20261111",
        "bd5a83ab94e8727bef15249b05fcf33d960df664a4f456959123f0408a670372",
        123,
    ),
    "anticipatory": (
        "outputs/anticipatory_causal_physical_20261113",
        "c04090aa12b6812053311e7c5796f0e9cd360d91b9eace2a99434d7b689fa16d",
        122,
    ),
}


def gripper_contacts(env):
    """Record endpoint geometry only; never supply it to action selection."""
    model, data = env.sim.model, env.sim.data
    rows = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        names = [model.geom_id2name(int(contact.geom1)), model.geom_id2name(int(contact.geom2))]
        if any(name and name.startswith("gripper0_") for name in names):
            rows.append({"geoms": names, "distance_m": float(contact.dist)})
    return rows


class EndpointTrace:
    """Observe refreshed completed endpoints without advancing the simulator."""

    def __init__(self, last_step, *, wrench_reader=read_current_wrench, contact_reader=gripper_contacts):
        self.last_step = last_step
        self.wrench_reader = wrench_reader
        self.contact_reader = contact_reader
        self.rows = {}
        self.repeated = 0

    def capture(self, env, step, endpoint):
        if type(step) is not int or not 80 <= step <= self.last_step or endpoint["endpoint_step"] != step:
            raise ValueError("The endpoint must belong to the completed recorded continuation")
        wrench = self.wrench_reader(env)
        if wrench["time"] != endpoint["sim_time"] or wrench["readonly_verified"] is not True:
            raise ValueError("A current, read-only, timestamp-bound sensor sample is required")
        data = env.sim.data
        row = {
            "endpoint_step": step,
            "sim_time": endpoint["sim_time"],
            "sim_state_sha256": endpoint["sim_state_sha256"],
            "wrench": wrench,
            "ctrl": np.asarray(data.ctrl).copy(),
            "actuator_force": np.asarray(data.actuator_force).copy(),
            "qfrc_actuator": np.asarray(data.qfrc_actuator).copy(),
            "offline_gripper_contacts": self.contact_reader(env),
        }
        if step in self.rows:
            old = self.rows[step]
            if old["sim_state_sha256"] != row["sim_state_sha256"] or old["sim_time"] != row["sim_time"]:
                raise ValueError("Repeated endpoint changed state or time")
            self.repeated += 1
        else:
            self.rows[step] = row

    def wrap(self, env, terminal, observers, continuation, receipt, *, progress=None):
        start = terminal["endpoint_receipt"]["endpoint_step"]
        self.capture(env, start, terminal["endpoint_receipt"])

        def observed_endpoint(current_env, step):
            sample, endpoint = observers.endpoint(current_env, step)
            self.capture(current_env, step, endpoint)
            return sample, endpoint

        wrapped = SimpleNamespace(
            endpoint=observed_endpoint,
            current=observers.current,
            render_anchor=observers.render_anchor,
            render_cameras=observers.render_cameras,
        )
        return continuation(env, terminal, wrapped, receipt, progress=progress)


def replay(case):
    prefix.environment()
    original, digest, last_step = BRANCHES[case]
    original = ROOT / original
    branch_ref = {"path": str(original / "producer/simulation/result.json"), "sha256": digest}
    branch = prefix.read(branch_ref)
    prefix.require(branch["complete"] is branch["passed"] is True, "Closed producer required")
    periods = admission.periods(branch)
    prefix.require([row["step"] for row in periods] == list(range(80, last_step)), "Recorded periods must be complete")
    inputs = admission.load_inputs()
    trace = EndpointTrace(last_step)
    case_output = OUTPUT / case
    case_output.mkdir(parents=True, exist_ok=False)
    (case_output / "checker/simulation").mkdir(parents=True)
    result = {
        "complete": False,
        "passed": False,
        "case": case,
        "producer_branch_ref": branch_ref,
        "source_refs": [prefix.reference(Path(__file__).resolve())],
        "new_world_model_requests": 0,
        "new_actions_selected": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
    }

    def traced_prefix(old, directory, continuation):
        def instrument(env, terminal, observers, receipt, *, progress=None):
            return trace.wrap(env, terminal, observers, continuation, receipt, progress=progress)

        return early.bounded_prefix(old, directory, instrument)

    def decode(values, producer, *, image_root, max_windows):
        return references.decode_actual_references(
            values, producer, image_root=original / "producer/simulation/runtime", max_windows=max_windows
        )

    worker = previous.clone(
        cached.checker_worker,
        {"_run_cached_prefix": traced_prefix, "references": SimpleNamespace(decode_actual_references=decode)},
    )
    try:
        worker(output=case_output, loader=lambda _phase: (inputs, {"producer_branch_ref": branch_ref}))
        replay_ref = prefix.reference(case_output / "checker/simulation/result.json")
        replay_result = prefix.read(replay_ref)
        prefix.require(replay_result["complete"] is replay_result["passed"] is True, "Replay must complete")
        prefix.require(sorted(trace.rows) == list(range(80, last_step + 1)), "Every completed endpoint is required")
        for period in periods:
            endpoint = period["endpoint_receipt"]
            row = trace.rows[endpoint["endpoint_step"]]
            prefix.require(row["sim_state_sha256"] == endpoint["sim_state_sha256"], "Endpoint identity changed")
        prefix.checked(branch_ref)
        result.update(complete=True, passed=True, replay_ref=replay_ref)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result.update(rows=list(trace.rows.values()), repeated_endpoint_observations=trace.repeated)
        print(publish_result(case_output / "wrench_trace.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(BRANCHES))
    replay(parser.parse_args().case)


if __name__ == "__main__":
    main()
