"""Own one source-pinned, bounded set of static model-audit workers."""

import sys

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_feedback_phase as lifecycle_check
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix


def execute_static_phase(
    *, root, qa_path, dry_path, indices, worker_module, load_inputs, input_receipt, source_refs, summarize
):
    """Verify registration, wait/reap every worker, and preserve partial results."""
    prefix.environment()
    prefix.require(
        root.resolve().is_relative_to((prefix.REPO / "outputs").resolve())
        and worker_module.startswith("scripts.tracking.")
        and indices
        and len(set(indices)) == len(indices),
        "A unique static worker set within repository outputs is required",
    )
    qa_ref, dry_ref = prefix.reference(qa_path), prefix.reference(dry_path)
    qa, dry = prefix.read(qa_ref), prefix.read(dry_ref)
    prefix.require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "Closed source-pinned CPU QA",
    )
    before = snapshot(prefix.REPO)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "QA source identity")
    inputs, sources = load_inputs(), source_refs()
    prefix.require(
        dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources,
        "Zero-model dry input and source identity",
    )
    root.mkdir(parents=True, exist_ok=False)
    registration = publish(
        root / "registration.json",
        dict(inputs=input_receipt(inputs), source_refs=sources, qa_ref=qa_ref, dry_ref=dry_ref),
    )
    keys = [f"case_{index:02d}" for index in indices]
    specs = [
        dict(
            key=key,
            argv=[sys.executable, "-m", worker_module, "--worker", str(index)],
            log=root / key / "worker.log",
        )
        for key, index in zip(keys, indices, strict=True)
    ]
    lifecycle = run_owned_workers(specs, cwd=prefix.REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers = publish(root / "workers.json", lifecycle)
    available = [prefix.reference(root / key / "result.json") for key in keys if (root / key / "result.json").is_file()]
    result = dict(
        complete=False,
        passed=False,
        control_qualified=False,
        registration_ref=registration,
        workers_ref=workers,
        available_case_refs=available,
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        lifecycle_check.require_closed_workers(lifecycle, keys)
        case_refs = [prefix.reference(root / key / "result.json") for key in keys]
        cases = [prefix.read(ref) for ref in case_refs]
        prefix.require(
            all(item["complete"] is item["passed"] is True for item in cases)
            and [item["case"]["case_index"] for item in cases] == list(indices),
            "All registered static cases must close in order",
        )
        counts = summarize(cases, inputs["config"])
        for path, digest in inputs["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        prefix.require(
            source_refs() == sources and snapshot(prefix.REPO)["sources"] == before,
            "Static inputs and sources unchanged",
        )
        result.update(complete=True, passed=True, sources_unchanged=True, case_refs=case_refs, counts=counts)
    finally:
        print(publish(root / "result.json", result), flush=True)
