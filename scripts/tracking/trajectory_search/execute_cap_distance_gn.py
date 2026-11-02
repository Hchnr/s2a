"""Own one finite CPU cap-distance process, with QA/source binding and reaping."""

import argparse
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

from s2a.infrastructure.publication import publish_result, result_json
from s2a.research.trajectory_search.restored_anchor_sqp_evidence import read, reference, require
from scripts.maintenance.run_cpu_tests import CPU_ENV, snapshot
from scripts.tracking.trajectory_search.check_cap_distance_gn import CONFIG, OUTPUT, REPO

OWNER = REPO / "outputs/cap_distance_gn_owner_20261102"


def signal_owned_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        # The child can exit between a wait timeout and signalling; still wait.
        pass


def main(args):
    require(
        Path.cwd() == REPO and all(os.environ.get(k) == v for k, v in CPU_ENV.items()), "Strict CPU owner environment"
    )
    OWNER.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        passed=False,
        launched=False,
        actual_wait=False,
        reaped=False,
        signal_sent=False,
        automatic_retry=False,
        timed_out=False,
        returncode=None,
    )
    process = None
    before = None
    try:
        require(not OUTPUT.exists(), "No existing cap experiment; no retry")
        qa_ref = reference(Path(args.qa_result))
        require(qa_ref["sha256"] == args.qa_sha, "Pinned QA result")
        qa = read(qa_ref)
        require(
            qa["passed"] and qa["all_tests_passed"] and qa["source_unchanged"] and qa["skipped_count"] == 0,
            "Complete passing CPU QA with no skips",
        )
        qa_sources = read(qa["source_after_ref"])
        before = snapshot(REPO)
        for key in ["sources", "current_map_ref", "driver_ref"]:
            require(before[key] == qa_sources[key], "Current tree must match QA: " + key)
        result["qa_ref"] = qa_ref
        result["source_before_ref"] = publish_result(OWNER / "source_before.json", before)
        spec = read(reference(CONFIG))
        require(spec["owner_timeout_seconds"] == 1800, "Registered owner wall time")
        environment = os.environ.copy()
        environment.update(CPU_ENV)
        command = [sys.executable, "-m", "scripts.tracking.trajectory_search.check_cap_distance_gn"]
        result.update(command=command, cwd=str(REPO), cpu_environment=CPU_ENV, owner_timeout_seconds=1800)
        with (OWNER / "process.log").open("x") as log:
            process = subprocess.Popen(
                command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            result.update(launched=True, pid=process.pid)
            print(result_json(dict(pid=process.pid, log=str(OWNER / "process.log"))), flush=True)
            try:
                result["returncode"] = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                result.update(timed_out=True, signal_sent=True)
                signal_owned_group(process.pid, signal.SIGTERM)
                try:
                    result["returncode"] = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    signal_owned_group(process.pid, signal.SIGKILL)
                    result["returncode"] = process.wait()
            result.update(actual_wait=True, reaped=process.returncode is not None)
        require(result["returncode"] == 0 and not result["timed_out"], "Child must finish normally within budget")
        child_ref = reference(OUTPUT / "result.json")
        child = read(child_ref)
        result["child_result_ref"] = child_ref
        require(
            child["complete"] and child["passed"] and child["sources_unchanged"], "Full independent experiment passed"
        )
        result.update(complete=True, passed=True)
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if process is not None and process.returncode is None:
            result["signal_sent"] = True
            signal_owned_group(process.pid, signal.SIGTERM)
            try:
                result["returncode"] = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_owned_group(process.pid, signal.SIGKILL)
                result["returncode"] = process.wait()
            result.update(actual_wait=True, reaped=True, passed=False)
        if before is not None:
            try:
                after = snapshot(REPO)
                result["source_after_ref"] = publish_result(OWNER / "source_after.json", after)
                result["sources_unchanged"] = all(
                    before[key] == after[key] for key in ["sources", "current_map_ref", "driver_ref"]
                )
                result["passed"] = result["passed"] and result["sources_unchanged"]
            except BaseException as error:
                result.update(source_error=repr(error), passed=False)
        receipt = publish_result(OWNER / "result.json", result)
        print(result_json(dict(result_ref=receipt, complete=result["complete"], passed=result["passed"])), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-result", required=True)
    parser.add_argument("--qa-sha", required=True)
    raise SystemExit(main(parser.parse_args()))
