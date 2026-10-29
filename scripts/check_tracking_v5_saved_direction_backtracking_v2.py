"""Repair string-sequence metadata comparison without changing frozen numerics.

The first checker stopped after reconstructing one H12 response because its
numeric comparator received endpoint_match.fields_checked, a list of strings.
This entry point retains that failed archive and delegates all reconstruction,
constraints, scores, tolerances, and output accounting to the frozen checker.
Only string-sequence comparison, the fresh output path, and repair provenance
are installed in its module namespace. No response is reused from the failure.
"""

import argparse

import check_tracking_v5_saved_direction_backtracking as frozen
import numpy as np

ORIGINAL_COMPARE_CONTRACT = frozen.compare_contract
ORIGINAL_LOAD_PREREQUISITES = frozen.load_prerequisites
CHECK_ROOT = frozen.REPO / "outputs/tracking_v5_saved_direction_backtracking_check_v2_20261028"


def compare_contract(actual, saved, *, compare=None, name="contract"):
    """Compare homogeneous string sequences exactly; retain every numeric rule."""
    if isinstance(actual, (list, tuple, np.ndarray)):
        array = np.asarray(actual)
        if array.dtype.kind in "US":
            frozen.require(type(saved) is type(actual), f"Exact {name} string container type")
            archived = np.asarray(saved)
            frozen.require(
                array.shape == archived.shape and array.dtype.kind == archived.dtype.kind,
                f"Exact {name} string shape and kind",
            )

            def exact_strings(left, right):
                if isinstance(left, (list, tuple)):
                    frozen.require(
                        type(right) is type(left) and len(right) == len(left),
                        f"Exact {name} nested string sequence",
                    )
                    for item, archived_item in zip(left, right, strict=True):
                        exact_strings(item, archived_item)
                else:
                    frozen.require(
                        isinstance(left, (str, bytes, np.str_, np.bytes_))
                        and type(right) is type(left)
                        and right == left,
                        f"Exact {name} string value and type",
                    )

            if isinstance(actual, np.ndarray):
                frozen.require(np.array_equal(array, archived), f"Exact {name} string values and order")
            else:
                exact_strings(actual, saved)
            return
    return ORIGINAL_COMPARE_CONTRACT(actual, saved, compare=compare, name=name)


def load_prerequisites():
    """Extend unchanged source admission with the closed failed-check evidence."""
    from s2a.saved_direction_backtracking_check_repair import repair_prerequisites

    diagnostic, producers, registrations, refs = ORIGINAL_LOAD_PREREQUISITES()
    repair_refs = repair_prerequisites()
    frozen.require(isinstance(repair_refs, dict), "Repair evidence is a path-to-reference mapping")
    for path, ref in repair_refs.items():
        frozen.require(path == ref["path"], "Repair reference key matches its path")
        frozen.bind_reference(ref, refs)
    frozen.finish(refs)
    return diagnostic, producers, registrations, refs


def install():
    """Redirect recursive metadata comparison and admission in the frozen main."""
    frozen.compare_contract = compare_contract
    frozen.load_prerequisites = load_prerequisites
    frozen.CHECK_ROOT = CHECK_ROOT


def main(dry=False):
    install()
    return frozen.main(dry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Verify closed failure and source evidence; create no output and issue no responses or queries",
    )
    main(parser.parse_args().dry)
