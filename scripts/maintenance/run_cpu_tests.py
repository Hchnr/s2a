"""Run repository CPU tests with independent subprocess shards and honest results.

Uses pytest already present in the chosen interpreter, without xdist. Collection
errors remain failures even when every available test item passes. The current
layout approval binds sources independently for each invocation.
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

AREAS = ("src", "scripts", "tests", "configs", "plans", "reports", "docs")
ROOT_FILES = (".gitignore", "AGENTS.md", "README.md", "pyproject.toml")
CPU_ENV = dict(
    CUDA_VISIBLE_DEVICES="",
    LD_LIBRARY_PATH="",
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONPATH="src:.",
    PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    LP_NUM_THREADS="1",
    PY_COLORS="0",
)
HOOK = """import json,pathlib,sys,pytest
output=pathlib.Path(sys.argv[1])
class Capture:
    def __init__(self):self.errors=[];self.reports=[]
    def pytest_collectreport(self,report):
        if report.failed:self.errors.append(dict(nodeid=report.nodeid,longrepr=str(report.longrepr)))
    def pytest_collection_finish(self,session):
        with (output/'nodeids.json').open('x') as stream:json.dump([item.nodeid for item in session.items],stream,indent=2)
    def pytest_runtest_logreport(self,report):
        self.reports.append(dict(nodeid=report.nodeid,when=report.when,outcome=report.outcome,wasxfail=getattr(report,'wasxfail',None),longrepr=str(report.longrepr) if report.failed else None))
capture=Capture()
status=pytest.main(sys.argv[2:],plugins=[capture])
with (output/'details.json').open('x') as stream:json.dump(dict(returncode=int(status),collection_errors=capture.errors,test_reports=capture.reports),stream,indent=2)
raise SystemExit(int(status))
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def publish(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return reference(path)


def approved_sources(root):
    command = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *AREAS]
    paths = {path for path in subprocess.check_output(command, cwd=root).decode().split("\0") if path}
    paths.update(ROOT_FILES)
    return sorted(path for path in paths if (root / path).is_file())


def node_input_path(root, relative):
    from s2a.infrastructure.repository_paths import repo_path

    path = repo_path(relative, root=root)
    require(
        path.is_relative_to(root / "outputs") and path.is_file(),
        "Node-list inputs must be canonical repository outputs files",
    )
    return path


def snapshot(root):
    # Repository infrastructure has only standard-library dependencies. Keeping
    # this import inside the run permits --help without importing S2A or pytest.
    from s2a.infrastructure.current_layout import CurrentLayout

    layout = CurrentLayout(root=root)
    manifest_ref = layout.migration_ref
    manifest_path = root / manifest_ref["path"]
    require(reference(manifest_path)["sha256"] == manifest_ref["sha256"], "Approved current-map SHA")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("complete_current_layout") is True, "Complete current source map required")
    for row in manifest["moves"]:
        layout.source_path(row["old_path"])
    layout.finish()
    return dict(
        current_map_ref=manifest_ref,
        sources={path: reference(root / path)["sha256"] for path in approved_sources(root)},
        driver_ref=reference(Path(__file__)),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
    )


def run_child(command, directory, root, deadline):
    row = dict(
        command=command, actual_wait=False, reaped=False, returncode=None, signal_sent=False, automatic_retry=False
    )
    started = time.monotonic()
    remaining = deadline - started
    require(remaining > 0, "Global CPU-test budget exhausted before launch")
    row["remaining_budget_seconds_at_launch"] = remaining
    child = None
    with (directory / "pytest.log").open("xb") as stream:
        try:
            child = subprocess.Popen(
                command,
                cwd=root,
                env=dict(os.environ, **CPU_ENV),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            row["pid"] = child.pid
            child.wait(timeout=remaining)
        except BaseException as error:
            row["execution_error"] = repr(error)
        finally:
            if child is not None:
                if child.poll() is None:
                    row["signal_sent"] = True
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                else:
                    child.wait()
                row.update(actual_wait=True, reaped=True, returncode=child.returncode)
    row.update(elapsed_seconds=time.monotonic() - started, log_ref=reference(directory / "pytest.log"))
    return row


def pytest_command(directory, targets, extra, *, collect=False):
    return [
        sys.executable,
        "-c",
        HOOK,
        str(directory),
        "-q",
        *(["--collect-only", "--continue-on-collection-errors"] if collect else []),
        *extra,
        *targets,
        "-p",
        "no:cacheprovider",
        "--basetemp=" + str(directory / "pytest_tmp"),
    ]


def parsed_child(command, directory, root, deadline, expected=None):
    answer = dict(
        command=command,
        actual_wait=False,
        reaped=False,
        returncode=None,
        signal_sent=False,
        automatic_retry=False,
        nodeids=[],
        collection_errors=[],
        test_reports=[],
        passed=False,
    )
    try:
        answer.update(run_child(command, directory, root, deadline))
        answer["nodeids"] = json.loads((directory / "nodeids.json").read_text())
        details = json.loads((directory / "details.json").read_text())
        answer.update(
            collection_errors=details["collection_errors"],
            test_reports=details["test_reports"],
            nodeids_ref=reference(directory / "nodeids.json"),
            details_ref=reference(directory / "details.json"),
        )
        answer["passed"] = (
            answer["returncode"] == 0
            and answer["actual_wait"] is answer["reaped"] is True
            and not answer["collection_errors"]
            and not answer["signal_sent"]
            and not answer.get("execution_error")
        )
        if expected is not None:
            calls = [row for row in answer["test_reports"] if row["when"] == "call"]
            terminals = []
            for node in expected:
                call = [row for row in calls if row["nodeid"] == node]
                terminals.extend(
                    call
                    or [
                        row
                        for row in answer["test_reports"]
                        if row["nodeid"] == node and row["when"] == "setup" and row["outcome"] == "skipped"
                    ]
                )
            answer["shard_node_order_exact"] = answer["nodeids"] == expected
            answer["terminal_reports"] = terminals
            answer["terminal_outcomes"] = dict(collections.Counter(row["outcome"] for row in terminals))
            answer["call_count"] = len(calls)
            answer["skipped_count"] = sum(row["outcome"] == "skipped" for row in terminals)
            answer["all_tests_passed"] = len(terminals) == len(expected) and all(
                row["outcome"] == "passed" for row in terminals
            )
            answer["passed"] = (
                answer["returncode"] == 0
                and answer["shard_node_order_exact"]
                and not answer["collection_errors"]
                and not answer["signal_sent"]
                and not answer.get("execution_error")
                and collections.Counter(row["nodeid"] for row in terminals) == collections.Counter(expected)
                and not any(row["outcome"] == "failed" for row in answer["test_reports"])
            )
    except BaseException as error:
        answer["error"] = repr(error)
    answer["result_ref"] = publish(directory / "result.json", answer)
    return answer


def main(args):
    root = Path.cwd()
    require(
        (root / "pyproject.toml").is_file() and (root / "src/s2a").is_dir(), "Run from the explicit S2A repository root"
    )
    require(
        type(args.jobs) is int and args.jobs > 0 and args.timeout_seconds > 0, "Positive worker count and global budget"
    )
    from s2a.infrastructure.repository_paths import repo_path

    output = repo_path(args.output, root=root)
    require(
        not Path(args.output).is_absolute()
        and ".." not in Path(args.output).parts
        and output.is_relative_to(root / "outputs/qa"),
        "Exclusive output must be below repository outputs/qa",
    )
    output.mkdir(parents=True, exist_ok=False)
    expected = None
    selected_ref = None
    selected = None
    if args.nodeids:
        require(not args.targets, "Select either direct targets or a pinned node list")
        path = node_input_path(root, args.nodeids)
        require(
            args.nodeids_sha is not None and reference(path)["sha256"] == args.nodeids_sha,
            "Explicit selected-node SHA required",
        )
        selected = json.loads(path.read_text())
        require(
            isinstance(selected, list)
            and selected
            and all(isinstance(node, str) for node in selected)
            and len(selected) == len(set(selected)),
            "Nonempty unique selected node IDs",
        )
        selected_ref = dict(path=str(path), sha256=args.nodeids_sha)
    if args.expected_nodeids:
        path = node_input_path(root, args.expected_nodeids)
        require(
            args.expected_nodeids_sha is not None and reference(path)["sha256"] == args.expected_nodeids_sha,
            "Explicit mapped node-list SHA required",
        )
        expected = json.loads(path.read_text())
        require(
            isinstance(expected, list)
            and all(isinstance(node, str) for node in expected)
            and len(expected) == len(set(expected)),
            "Unique complete expected node IDs",
        )
    if selected is not None:
        require(expected is None or expected == selected, "Selected and expected node lists must agree")
        expected = selected
    targets = selected or args.targets
    if not targets and expected is not None:
        old_files = list(dict.fromkeys(node.split("::", 1)[0] for node in expected))
        all_files = {
            path
            for path in approved_sources(root)
            if path.startswith("tests/") and Path(path).name.startswith("test_") and path.endswith(".py")
        }
        targets = [*old_files, *sorted(all_files - set(old_files))]
    targets = targets or ["tests"]
    for target in targets:
        path = Path(target.split("::", 1)[0])
        require(
            not path.is_absolute() and ".." not in path.parts and path.parts[0] == "tests",
            "Only explicit repository tests paths/nodes",
        )
    result = dict(
        complete=False,
        passed=False,
        configured_parallelism=args.jobs,
        execution_parallelism=0,
        automatic_retry=False,
        environment=CPU_ENV,
        source_unchanged=False,
        actual_wait=False,
        reaped=False,
        signal_sent=False,
        global_budget_seconds=args.timeout_seconds,
        wallclock_global_test_order_claimed=False,
        workers=[],
        targets=targets,
        expected_nodeids_ref=None
        if args.expected_nodeids is None
        else dict(path=str(root / args.expected_nodeids), sha256=args.expected_nodeids_sha),
        selected_nodeids_ref=selected_ref,
    )
    before = None
    try:
        before = snapshot(root)
        before["expected_nodeids_ref"] = result["expected_nodeids_ref"]
        before["selected_nodeids_ref"] = selected_ref
        result["source_before_ref"] = publish(output / "source_before.json", before)
        deadline = time.monotonic() + args.timeout_seconds
        directory = output / "collection"
        directory.mkdir()
        collection = parsed_child(
            pytest_command(directory, targets, args.pytest_arg, collect=True), directory, root, deadline
        )
        result["collection"] = collection
        nodes = collection["nodeids"]
        result["mapped_node_order_exact"] = None if expected is None else nodes == expected
        result["collection_node_count"] = len(nodes)
        require(len(nodes) == len(set(nodes)), "Collection cannot contain duplicate nodes")
        if expected is not None:
            require(nodes == expected, "Mapped collection node IDs/order changed")
        jobs = min(args.jobs, len(nodes)) if not args.collect_only else 0
        result["execution_parallelism"] = jobs
        shards = [nodes[index::jobs] for index in range(jobs)]
        result["shard_plan_ref"] = publish(
            output / "shard_plan.json", dict(configured_parallelism=args.jobs, active_workers=jobs, shards=shards)
        )

        def worker(index):
            directory = output / f"worker_{index:02}"
            directory.mkdir()
            return parsed_child(
                pytest_command(directory, shards[index], args.pytest_arg),
                directory,
                root,
                deadline,
                expected=shards[index],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(worker, index): index for index in range(jobs)}
            workers = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    row = future.result()
                except BaseException as error:
                    row = dict(
                        error=repr(error),
                        actual_wait=False,
                        reaped=False,
                        returncode=None,
                        signal_sent=False,
                        nodeids=[],
                        collection_errors=[],
                        test_reports=[],
                        passed=False,
                    )
                workers.append((futures[future], row))
        workers.sort()
        result["workers"] = [row for _, row in workers]
        observed = [node for row in result["workers"] for node in row["nodeids"]]
        reports = [report for row in result["workers"] for report in row["test_reports"]]
        calls = [report for report in reports if report["when"] == "call"]
        terminals = [report for row in result["workers"] for report in row.get("terminal_reports", [])]
        all_commands = [collection, *result["workers"]]
        result.update(
            complete=True,
            actual_wait=all(row["actual_wait"] for row in all_commands),
            reaped=all(row["reaped"] for row in all_commands),
            signal_sent=any(row["signal_sent"] for row in all_commands),
            call_count=len(calls),
            call_outcomes=dict(collections.Counter(row["outcome"] for row in calls)),
            terminal_outcomes=dict(collections.Counter(row["outcome"] for row in terminals)),
            skipped_count=sum(row["outcome"] == "skipped" for row in terminals),
            xfail_count=sum(row.get("wasxfail") is not None and row["outcome"] == "skipped" for row in terminals),
            xpass_count=sum(row.get("wasxfail") is not None and row["outcome"] == "passed" for row in terminals),
            all_tests_passed=not args.collect_only
            and len(terminals) == len(nodes)
            and all(row["outcome"] == "passed" for row in terminals)
            and not collection["collection_errors"],
            aggregate_node_partition_exact=args.collect_only
            or collections.Counter(observed) == collections.Counter(nodes),
        )
        result["passed"] = (
            collection["returncode"] == 0
            and not collection["collection_errors"]
            and result["actual_wait"] is result["reaped"] is True
            and not result["signal_sent"]
            and result["aggregate_node_partition_exact"]
            and all(row["passed"] for row in result["workers"])
        )
    except BaseException as error:
        result["error"] = repr(error)
        traceback.print_exc()
    finally:
        try:
            after = snapshot(root)
            after["expected_nodeids_ref"] = (
                None
                if result["expected_nodeids_ref"] is None
                else reference(node_input_path(root, args.expected_nodeids))
            )
            after["selected_nodeids_ref"] = (
                None if selected_ref is None else reference(node_input_path(root, args.nodeids))
            )
            result["source_after_ref"] = publish(output / "source_after.json", after)
            result["source_unchanged"] = before == after
        except BaseException as error:
            result["source_after_error"] = repr(error)
        result["passed"] = result["passed"] and result["source_unchanged"]
        result["returncode"] = 0 if result["passed"] else 1
        receipt = publish(output / "result.json", result)
        print(
            json.dumps(
                dict(
                    result_ref=receipt,
                    passed=result["passed"],
                    returncode=result["returncode"],
                    call_count=result.get("call_count"),
                    skipped_count=result.get("skipped_count"),
                    all_tests_passed=result.get("all_tests_passed", False),
                    collection_errors=len(result.get("collection", {}).get("collection_errors", [])),
                )
            ),
            flush=True,
        )
    return result["returncode"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--expected-nodeids")
    parser.add_argument("--expected-nodeids-sha")
    parser.add_argument("--nodeids")
    parser.add_argument("--nodeids-sha")
    parser.add_argument("--pytest-arg", action="append", default=[])
    raise SystemExit(main(parser.parse_args()))
