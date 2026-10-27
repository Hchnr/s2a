"""056 one fixed D0/D1/D2 aggregate CPU fit, only after full D2 qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from analyze_wam_clean_capacity import checked
from analyze_wam_release_horizon import reference

from s2a.artifacts import REPO
from s2a.clean_training_aggregation import ROWS, admit_aggregation
from s2a.clean_training_data import normalize_active10
from s2a.wam_clean_streaming import CleanFitBlock, CleanFitLayout, fit_clean_residual_streaming


def execute(spec, output):
    consumed = {}

    def read_json(ref):
        return json.loads(checked(ref, consumed).read_text())

    previous, prepared, collection_refs, collections = admit_aggregation(spec, read_json=read_json)
    for ref in spec["source_refs"]:
        checked(ref, consumed)
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    layout = CleanFitLayout(tuple(previous["fit_metadata"]["feature_columns"]), "active10")
    labels, passes = {}, []

    def replay():
        digests = [hashlib.sha256() for _ in range(3)]
        prefix = [hashlib.sha256(), hashlib.sha256()]
        rows = requests = 0
        for round_index, collection in enumerate(collections):
            normal = read_json(collection["normalization_ref"])
            for request in collection["requests"]:
                anchor = anchors[request["anchor_name"]]
                name = anchor["name"]
                if name not in labels:
                    with np.load(checked(anchor["trajectory_ref"], consumed), allow_pickle=False) as z:
                        start = anchor["anchor_frame"]
                        physical = z["action128"][start : start + 40].copy()
                    labels[name] = normalize_active10(physical, **normal)["normalized_active10"].astype(np.float64)
                ledger = read_json(request["compact_ledger_ref"])
                if not ledger["complete"] or len(ledger["points"]) != 35:
                    raise ValueError("056 incomplete denoise nodes")
                xx, yy = [], []
                for step, ref in enumerate(ledger["points"]):
                    point = read_json(ref)
                    if point["metadata"]["step_index"] != step or point["metadata"]["feature_terms"]:
                        raise ValueError("056 feature/time contract differs")
                    with np.load(checked(point["arrays_ref"], consumed), allow_pickle=False) as z:
                        c0 = z["c0_active10"]
                        xx.append(np.column_stack((z["hidden"].astype(np.float64), c0)))
                        yy.append(labels[name] - c0)
                features, target = np.vstack(xx), np.vstack(yy)
                weights = np.full(len(features), 1 / ROWS)
                arrays = [np.asarray(v, dtype="<f8", order="C").tobytes() for v in (features, target, weights)]
                for digest, payload in zip(digests, arrays, strict=True):
                    digest.update(payload)
                if round_index < 2:
                    for digest, payload in zip(prefix, arrays[:2], strict=True):
                        digest.update(payload)
                requests += 1
                rows += len(features)
                yield CleanFitBlock(features, target, weights, layout)
                if requests % 50 == 0:
                    print(json.dumps(dict(pass_index=len(passes) + 1, requests=requests, rows=rows)), flush=True)
        expected = previous["fit_metadata"]["replay_sha256"]
        if (
            rows != ROWS
            or requests != 900
            or [d.hexdigest() for d in prefix] != [expected[k] for k in ("features", "target_clean_residual")]
        ):
            raise ValueError("056 complete stream or old D0/D1 prefix differs")
        passes.append([d.hexdigest() for d in digests])

    fitted = fit_clean_residual_streaming(replay, layout=layout, ridge=spec["ridge"])
    if fitted.rows != ROWS or len(passes) != 2 or passes[0] != passes[1] or len(labels) != 150:
        raise ValueError("056 full aggregate fit denominator differs")
    for path, sha in consumed.items():
        if reference(Path(path))["sha256"] != sha:
            raise ValueError("056 consumed input changed")
    with (output / "parameters.json").open("x") as stream:
        json.dump(fitted.readout.to_parameters(), stream, indent=2)
        stream.write("\n")
    result = dict(
        schema_version="s2a.clean-aggregation-fit.v1",
        complete=True,
        passed=True,
        producer_fits=1,
        independent_numerical_qualification=False,
        registration_ref=reference(output / "registration.json"),
        params_ref=reference(output / "parameters.json"),
        collection_refs=collection_refs,
        previous_fit_ref=spec["previous_fit_ref"],
        training_prepared_ref=previous["prepared_ref"],
        d2_owner_ref=spec["d2_owner_ref"],
        fit_metadata=fitted.fit_metadata(),
        input_sha256=consumed,
        inputs_unchanged_after=True,
        future_training_windows_decoded=len(labels),
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        deployment_authorized=False,
    )
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO or not args.registration.resolve().is_relative_to(REPO / "outputs"):
        raise ValueError("056 requires repository cwd and an outputs registration")
    if any(os.environ.get(k) != "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")) or any(
        os.environ.get(k) != "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("056 explicit CPU/thread8 environment required")
    spec = json.loads(args.registration.read_text())
    if spec["entrypoint_ref"] != reference(Path(__file__)):
        raise ValueError("056 producer source differs")
    output = args.registration.resolve().parent
    # Admission precedes even the fit-attempt marker; a live D2 is not a fit attempt.
    admit_aggregation(spec)
    if args.dry:
        print(
            json.dumps(
                dict(
                    passed=True,
                    complete=True,
                    registration_ref=reference(args.registration),
                    rows=ROWS,
                    producer_fits=0,
                    future_labels_decoded=0,
                )
            ),
            flush=True,
        )
        return
    with (output / "attempt.json").open("x") as stream:
        json.dump(
            dict(started=True, producer_fit_attempt=1, pid=os.getpid(), registration_ref=reference(args.registration)),
            stream,
        )
    try:
        execute(spec, output)
    except BaseException as error:
        with (output / "failure.json").open("x") as stream:
            json.dump(dict(passed=False, producer_fit_attempt=1, error=repr(error)), stream)
        raise


if __name__ == "__main__":
    main()
