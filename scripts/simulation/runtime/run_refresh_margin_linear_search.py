"""Run one registered linearized search on the previously replayed state."""

import argparse
from functools import partial
from pathlib import Path
from types import FunctionType

from s2a.research.trajectory_search.refresh_margin_linear_search import MAX_RESPONSES, solve_linearized_refresh
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_refresh_margin_targeted_search as original

ROOT = prefix.REPO
PLAN = ROOT / "plans/v0.0.1/tracking/references/refresh_margin_linear_search_20261116.md"
OUTPUT = ROOT / "outputs/refresh_margin_linear_search_20261116"
DRY = ROOT / "outputs/refresh_margin_linear_search_dry_20261116.json"
QA = ROOT / "outputs/qa/refresh_margin_linear_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.run_refresh_margin_linear_search"
PRIOR = {
    "path": str(ROOT / "outputs/refresh_margin_targeted_search_20261116/result.json"),
    "sha256": "c6e8965205d0ffdfc0ce9465119a202f7e76ed39138ccf306d9dfc41b3c109ee",
}


def source_refs():
    return original.source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            PLAN,
            ROOT / "src/s2a/research/trajectory_search/refresh_margin_linear_search.py",
            ROOT / "tests/research/trajectory_search/test_refresh_margin_linear_search.py",
            Path(PRIOR["path"]),
        )
    ]


def load_sample():
    sample, identity = original.load_sample()
    previous = prefix.read(PRIOR)
    prefix.require(
        previous["complete"] is previous["passed"] is previous["sources_unchanged"] is True
        and previous["certified_model_witness"] is False
        and previous["counters"]["responses_attempted"] == previous["counters"]["responses_completed"] == 320,
        "The earlier separately budgeted search must be closed and retain its negative result",
    )
    registration = prefix.read(previous["registration_ref"])
    prefix.require(
        registration["identity"] == identity and registration["source_refs"] == original.source_refs(),
        "The earlier search must bind the same causal sample and its unchanged source",
    )
    worker = prefix.read(previous["worker_ref"])
    prefix.require(
        worker["analysis"]["solver"]["eligible_candidates"] == 0
        and worker["analysis"]["solver"]["optimizer"]["message"] == "The extra response budget is exhausted",
        "The earlier response-budget stop must not be rewritten",
    )
    return sample, dict(previous_identity=identity, prior_ref=PRIOR, prior_worker_ref=previous["worker_ref"])


def isolated(function, **overrides):
    """Reuse the audited static owner without modifying its original globals."""
    namespace = function.__globals__.copy()
    namespace.update(overrides)
    copied = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return copied


def analyze(sample, counters):
    counters.update(lp_attempted=0, lp_completed=0)

    def account(event):
        prefix.require(event in ("attempted", "completed"), "Only explicit LP accounting events are permitted")
        counters[f"lp_{event}"] += 1
        prefix.require(counters["lp_attempted"] <= 1 and counters["lp_completed"] <= counters["lp_attempted"], "One LP")

    solver = partial(solve_linearized_refresh, on_lp=account)
    result = isolated(original.analyze, solve_targeted_refresh=solver)(sample, counters)
    prefix.require(
        counters["lp_attempted"] == counters["lp_completed"] == 1
        and counters["responses_attempted"] == counters["responses_completed"] <= MAX_RESPONSES,
        "One LP and at most two actual preview responses are allowed",
    )
    return result


def common():
    return dict(OUTPUT=OUTPUT, DRY=DRY, QA=QA, MODULE=MODULE, load_sample=load_sample, source_refs=source_refs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.dry:
        isolated(original.dry, **common())()
    elif args.execute:
        isolated(original.execute, **common())()
    else:
        isolated(original.worker, analyze=analyze, **common())()


if __name__ == "__main__":
    main()
