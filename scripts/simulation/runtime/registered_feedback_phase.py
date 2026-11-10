"""Run one source-pinned producer or replay phase for bounded feedback studies."""

import sys

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def require_closed_workers(lifecycle, expected_keys):
    """Require the complete registered worker set and real wait/reap receipts."""
    workers = lifecycle.get("workers", [])
    prefix.require(
        not lifecycle.get("launch_error")
        and [item.get("key") for item in workers] == list(expected_keys)
        and all(
            item.get("returncode") == 0
            and item.get("actual_wait") is True
            and item.get("reaped") is True
            and item.get("signal_sent") is False
            for item in workers
        ),
        "Every registered feedback worker must exit and be reaped without a signal",
    )


def execute_phase(
    phase,
    *,
    root,
    qa_path,
    dry_path,
    branches,
    worker_module,
    branch_key,
    load_inputs,
    input_receipt,
    source_refs,
    summarize_phase,
    config_key,
):
    """Freeze inputs, own workers, and publish complete or partial evidence."""
    prefix.environment()
    prefix.require(phase in ("producer", "checker"), "One known feedback phase required")
    prefix.require(
        root.resolve().is_relative_to((prefix.REPO / "outputs").resolve()),
        "Feedback outputs must stay in this repository",
    )
    prefix.require(
        worker_module.startswith("scripts.simulation.runtime.") and branches and len(set(branches)) == len(branches),
        "Unique registered branches and runtime module required",
    )
    qa, dry = prefix.read(prefix.reference(qa_path)), prefix.read(prefix.reference(dry_path))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Frozen CPU QA",
    )
    before = snapshot(prefix.REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry identity",
    )
    producer_refs = None
    if phase == "checker":
        producer = prefix.read(prefix.reference(root / "producer/result.json"))
        prefix.require(
            producer["complete"] is producer["passed"] is producer["sources_unchanged"] is True
            and len(producer["branch_refs"]) == len(branches),
            "All registered producer branches closed",
        )
        producer_refs = {
            branch_key(*branch): ref for branch, ref in zip(branches, producer["branch_refs"], strict=True)
        }
    phase_dir = root / phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    registration = publish(
        phase_dir / "registration.json",
        dict(phase=phase, inputs=input_receipt(inputs), source_refs=sources, producer_branch_refs=producer_refs),
    )
    keys = [branch_key(*branch) for branch in branches]
    specs = [
        dict(
            key=key,
            argv=[sys.executable, "-m", worker_module, "--worker", phase, str(index), mode],
            log=phase_dir / key / "worker.log",
        )
        for key, (index, mode) in zip(keys, branches, strict=True)
    ]
    lifecycle = run_owned_workers(specs, cwd=prefix.REPO, timeout_seconds=inputs[config_key]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(phase_dir / "workers.json", lifecycle)
    available = [
        prefix.reference(phase_dir / key / "result.json") for key in keys if (phase_dir / key / "result.json").is_file()
    ]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        phase=phase,
        registration_ref=registration,
        workers_ref=workers,
        available_branch_refs=available,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        require_closed_workers(lifecycle, keys)
        branch_refs = [prefix.reference(phase_dir / key / "result.json") for key in keys]
        branch_results = [prefix.read(ref) for ref in branch_refs]
        counts = summarize_phase(branch_results, phase, inputs[config_key])
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(prefix.REPO)["sources"] == before, "Feedback sources unchanged"
        )
        result.update(complete=True, passed=True, sources_unchanged=True, branch_refs=branch_refs, counts=counts)
    finally:
        print(publish(phase_dir / "result.json", result), flush=True)
