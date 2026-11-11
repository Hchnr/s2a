"""Supervise a new live stream while preserving frozen historical owners."""

import argparse
from pathlib import Path
from types import FunctionType, SimpleNamespace

from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_cached as simulation
from scripts.wam.tracking.preview import causal_stream_live_cached_admission as admission
from scripts.wam.tracking.preview import live_stream_model_48
from scripts.wam.tracking.preview import run_causal_stream_live_48 as frozen

ROOT, OUTPUT, QA, DRY = admission.ROOT, admission.OUTPUT, admission.QA, admission.DRY
MODULE = "scripts.wam.tracking.preview.run_causal_stream_live_cached"
_OWNER_FUNCTIONS = (
    "dry_run",
    "check_cpu_admission",
    "preflight",
    "checked_preflight",
    "worker_inputs",
    "model_worker_inputs",
    "model_bundle",
    "check_model_branch",
    "execute",
)
_FROZEN = {name: getattr(frozen, name) for name in _OWNER_FUNCTIONS}
_FROZEN_SUMMARY = frozen.summarize_branch


def source_refs():
    names = (
        admission.CONFIG,
        admission.PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/wam/tracking/preview/causal_stream_live_cached_admission.py",
        ROOT / "scripts/simulation/runtime/run_causal_stream_live_cached.py",
        ROOT / "src/s2a/research/native_runtime/endpoint_cache.py",
        ROOT / "src/s2a/research/native_runtime/cached_replay_comparison.py",
        ROOT / "tests/research/wam/test_causal_stream_live_cached_admission.py",
        ROOT / "tests/research/wam/test_causal_stream_live_cached_owner.py",
        ROOT / "tests/research/native_runtime/test_causal_stream_live_cached_physical.py",
        ROOT / "tests/research/native_runtime/test_endpoint_cache_stream.py",
    )
    return frozen.source_refs() + [prefix.reference(path) for path in names]


def check_cache_events(branch):
    """Bind every observed and reused endpoint to the actual completed periods."""
    runtime = branch["runtime"]
    periods = [period for row in runtime["continuation"]["segments"] for period in row["continuation"]["periods"]]
    events = runtime["endpoint_cache_events"]
    observed, reused, latest = [], [], None
    for event in events:
        if event["kind"] == "observed":
            observed.append(event)
            latest = event
        else:
            prefix.require(
                event["kind"] == "reused"
                and latest is not None
                and (event["step"], event["time"], event["state_sha256"])
                == (latest["step"], latest["time"], latest["state_sha256"]),
                "Every reused endpoint must match the latest exact-state observation",
            )
            reused.append(event["step"])
    prefix.require(len(observed) == len(periods), "One actual cached observation is required per completed action")
    for event, period in zip(observed, periods, strict=True):
        endpoint = period["endpoint_receipt"]
        prefix.require(
            endpoint["endpoint_step"] == event["step"] == period["step"] + 1
            and endpoint["sim_time"] == event["time"]
            and endpoint["sim_state_sha256"] == event["state_sha256"] == state_sha256(period["actual_state"]),
            "Cached samples must bind actual endpoint state and time",
        )
    return {"observed": len(observed), "reused": reused}


def summarize_branch(branch, *, phase, producer=None):
    counts = _FROZEN_SUMMARY(branch, phase=phase, producer=producer)
    cache = check_cache_events(branch)
    source_complete = (
        branch["runtime"]["continuation"]["complete"]
        if phase == "producer"
        else producer["runtime"]["continuation"]["complete"]
    )
    if source_complete:
        prefix.require(
            cache["reused"] == [88, 96, 104, 112, 120, 128], "All six completed boundaries must reuse endpoints"
        )
    return counts


def build_context(*, output=OUTPUT, qa=QA, dry=DRY, sources=None):
    """Give frozen owner functions isolated dependencies without mutating them.

    The prior owner and its source hashes remain unchanged. Every cloned
    function keeps the exact frozen code while resolving its shared calls in a
    private namespace with the newly registered paths and cached workers.
    """
    prefix.require(
        all(getattr(frozen, name) is function for name, function in _FROZEN.items())
        and frozen.summarize_branch is _FROZEN_SUMMARY,
        "The historical owner functions must remain unchanged",
    )
    namespace = dict(frozen.__dict__)
    namespace.update(
        OUTPUT=Path(output),
        QA=Path(qa),
        DRY=Path(dry),
        MODULE=MODULE,
        admission=admission,
        simulation=simulation,
        source_refs=source_refs if sources is None else sources,
        summarize_branch=summarize_branch,
    )
    for name, original in _FROZEN.items():
        function = FunctionType(
            original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
        )
        function.__kwdefaults__ = dict(original.__kwdefaults__ or {})
        namespace[name] = function
    return SimpleNamespace(**{name: namespace[name] for name in _OWNER_FUNCTIONS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--producer", action="store_true")
    group.add_argument("--checker", action="store_true")
    group.add_argument("--worker", choices=("simulation", "model", "checker"))
    args = parser.parse_args()
    context = build_context()
    if args.dry_run:
        context.dry_run()
    elif args.preflight:
        context.preflight()
    elif args.producer:
        context.execute("producer")
    elif args.checker:
        context.execute("checker")
    elif args.worker == "simulation":
        simulation.simulation_worker(output=OUTPUT, loader=context.worker_inputs)
    elif args.worker == "checker":
        simulation.checker_worker(output=OUTPUT, loader=context.worker_inputs)
    else:
        live_stream_model_48.model_worker(output=OUTPUT, loader=context.model_worker_inputs)


if __name__ == "__main__":
    main()
