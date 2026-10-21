"""025 CPU-only fixed two-fit readout stage; real fitting is an explicit CLI action."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256
from s2a.wam_readout import ACTIVE_MASK, HIDDEN_SIZE, LIBERO_DOMAIN, DomainContext, ResidualReadout

CONFIG = REPO / "configs/wam_frozen_readout_20261021.json"
CONFIG_SHA = "6534245a114d89167f6d8cd283e55de8685b4a02abca79706436309477860605"
PLAN = REPO / "plans/v0.0.1/025_frozen_wam_readout_20261021.md"
PLAN_SHA = "46b766356ef29c593c895a23130a0415cc83c892afcfc6bb66e07e5bab540dc6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Evidence:
    def __init__(self):
        self.hashes = {}

    def check(self, path, digest=None):
        path = Path(path).resolve()
        value = sha256(path)
        require(digest is None or value == digest, f"source SHA changed: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == value, f"source changed during fit: {path}")
        self.hashes[str(path)] = value
        return path

    def json(self, path, digest=None):
        return json.loads(self.check(path, digest).read_text())

    def json_ref(self, ref):
        return self.json(ref["path"], ref["sha256"])

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.check(path, digest)


def validate_anchors(prepared):
    """The fit split is determined only by whole-source identities, never results."""
    config = prepared["config"]
    train, validation = config["train_episodes"], config["validation_episodes"]
    require(
        len(set(train)) == len(set(validation)) == 6 and not set(train) & set(validation),
        "six disjoint whole episodes per split required",
    )
    expected_episodes = {ep: split for split, values in (("train", train), ("validation", validation)) for ep in values}
    anchors, source_splits = {}, {}
    per_episode = {ep: [] for ep in expected_episodes}
    for anchor in prepared["anchors"]:
        episode = anchor["episode"]
        require(
            episode in expected_episodes and anchor["split"] == expected_episodes[episode],
            "anchor crossed its registered whole-episode split",
        )
        require(anchor["name"] not in anchors, "duplicate anchor identity")
        source = anchor["source_episode_id"]
        require(
            source not in source_splits or source_splits[source] == episode,
            "source episode aliases leak between registered episodes",
        )
        source_splits[source] = episode
        anchors[anchor["name"]] = anchor
        per_episode[episode].append(anchor)
    require(len(anchors) == 48 and len(source_splits) == 12, "expected48 anchors from12 complete source episodes")
    for episode, group in per_episode.items():
        require(
            len(group) == 4 and {a["anchor_index"] for a in group} == set(range(4)),
            "four registered anchors per episode required",
        )
        require(
            len({a["anchor_frame"] for a in group}) == 4 and len({a["arrays_path"] for a in group}) == 1,
            "anchor/source identity differs within episode",
        )
    episode_roster = prepared["episodes"]
    require(
        len(episode_roster) == 12 and {e["episode"] for e in episode_roster} == set(expected_episodes),
        "whole-episode roster differs",
    )
    for item in episode_roster:
        group = per_episode[item["episode"]]
        require(
            item["split"] == expected_episodes[item["episode"]]
            and all(a["source_episode_id"] == item["source_episode_id"] for a in group),
            "whole-source roster disagrees with anchor split",
        )
    return anchors


def validate_feature_array(arrays, meta, torch):
    expected = dict(hidden=(40, HIDDEN_SIZE), base=(40, 128), target=(40, 128), xt=(40, 128), valid=(40, 128))
    for name, shape in expected.items():
        require(name in arrays and arrays[name].shape == shape, f"feature {name} shape differs")
        require(arrays[name].dtype == (np.bool_ if name == "valid" else np.float32), f"feature {name} dtype differs")
        require(np.isfinite(arrays[name]).all(), f"feature {name} is nonfinite")
    require(
        np.array_equal(arrays["valid"], np.broadcast_to(ACTIVE_MASK, (40, 128))),
        "teacher endpoint active10 mask differs",
    )
    require(
        not np.any(arrays["target"][:, 10:]) and not np.any(arrays["xt"][:, 10:]),
        "inactive teacher target/input is nonzero",
    )
    for name in ("hidden", "base"):
        require(meta["original_dtypes"][name] == "torch.bfloat16", f"actual {name} was not captured from BF16")
        require(
            np.array_equal(torch.from_numpy(arrays[name]).to(torch.bfloat16).float().numpy(), arrays[name]),
            f"{name} export is not an exact BF16 expansion",
        )


def load_rows(prepared, features, evidence, torch):
    anchors = validate_anchors(prepared)
    expected = prepared["teacher_calls"]
    calls = features["teacher_calls"]
    require(
        len(calls) == len(expected) == 144 and features["denoise_calls"] == 249,
        "phase1 must contain144 teacher calls plus105 identity denoises",
    )
    require(len(features["zero_identity"]) == 3, "all three exact zero-identity admissions are required")
    # Bind every actual identity artifact; the independent stage checker verifies
    # the original complete40×128 cache comparisons inside those artifacts.
    for ref in features["zero_identity"]:
        evidence.json_ref(ref)
    columns = {k: [] for k in ("hidden", "base", "target", "xt", "valid")}
    rows = []
    seen = set()
    for index, (actual, planned) in enumerate(zip(calls, expected, strict=True)):
        fields = ("call_index", "anchor_name", "split", "sigma")
        require(
            all(actual[k] == planned[k] for k in fields) and actual["call_index"] == index,
            "actual teacher call changed its registered order/split/sigma",
        )
        anchor = anchors[actual["anchor_name"]]
        require(
            actual["split"] == anchor["split"] and actual["sigma"] in (0.2, 0.5, 0.8),
            "teacher call identity differs from source anchor",
        )
        key = (actual["anchor_name"], actual["sigma"])
        require(key not in seen, "duplicate teacher source/sigma")
        seen.add(key)
        meta = evidence.json_ref(actual["bundle"])
        require(
            all(meta[k] == actual[k] for k in ("anchor_name", "split", "sigma")),
            "teacher bundle metadata differs from its admitted call",
        )
        path = evidence.check(meta["arrays"]["path"], meta["arrays"]["sha256"])
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in columns}
        validate_feature_array(arrays, meta, torch)
        for key, value in arrays.items():
            columns[key].append(value)
        rows.extend(
            dict(
                episode=anchor["episode"],
                anchor_name=anchor["name"],
                sigma=actual["sigma"],
                split=anchor["split"],
                task_index=anchor["task_index"],
                endpoint=i,
            )
            for i in range(40)
        )
    require(len(seen) == 144 and len(rows) == 5760, "incomplete fixed teacher endpoint grid")
    require(
        all(sum(r["split"] == s for r in rows) == 2880 for s in ("train", "validation")),
        "complete train/validation teacher rows required",
    )
    return {k: np.concatenate(v) for k, v in columns.items()}, rows


def hierarchical_weights(rows):
    """Episode→anchor→sigma→endpoint equal weights; no validation row is fitted."""
    require(rows and all(r["split"] == "train" for r in rows), "fit weights must contain training rows only")
    episodes = {r["episode"] for r in rows}
    require(len(episodes) == 6, "fit requires six complete training episodes")
    result = np.empty(len(rows), dtype=np.float64)
    for episode in episodes:
        indices = [i for i, r in enumerate(rows) if r["episode"] == episode]
        anchors = {rows[i]["anchor_name"] for i in indices}
        require(len(anchors) == 4, "fit requires four anchors per episode")
        for anchor in anchors:
            members = [i for i in indices if rows[i]["anchor_name"] == anchor]
            sigmas = {rows[i]["sigma"] for i in members}
            require(sigmas == {0.2, 0.5, 0.8}, "fit requires three fixed sigmas")
            for sigma in sigmas:
                selected = [i for i in members if rows[i]["sigma"] == sigma]
                require(
                    len(selected) == 40 and {rows[i]["endpoint"] for i in selected} == set(range(40)),
                    "fit requires40 distinct valid endpoints",
                )
                result[selected] = 1 / (len(episodes) * len(anchors) * len(sigmas) * len(selected))
    require(np.isclose(result.sum(), 1, rtol=0, atol=1e-15), "fit weight total differs")
    return result


def teacher_metrics(predictions, target, rows):
    errors = {
        name: np.mean((value[:, :10].astype(np.float64) - target[:, :10].astype(np.float64)) ** 2, axis=1)
        for name, value in predictions.items()
    }

    def group(indices):
        require(len(indices) > 0, "empty teacher metric group")
        return dict(
            rows=len(indices),
            active_elements=len(indices) * 10,
            mse={name: float(value[indices].mean()) for name, value in errors.items()},
            per_slot_mse={
                name: ((value[indices, :10].astype(np.float64) - target[indices, :10].astype(np.float64)) ** 2)
                .mean(axis=0)
                .tolist()
                for name, value in predictions.items()
            },
        )

    result = {}
    for split in ("train", "validation"):
        indices = [i for i, r in enumerate(rows) if r["split"] == split]
        result[split] = dict(
            pooled=group(indices),
            by_sigma={str(s): group([i for i in indices if rows[i]["sigma"] == s]) for s in (0.2, 0.5, 0.8)},
            by_episode={
                str(e): group([i for i in indices if rows[i]["episode"] == e])
                for e in sorted({rows[i]["episode"] for i in indices})
            },
            by_task={
                str(t): group([i for i in indices if rows[i]["task_index"] == t])
                for t in sorted({rows[i]["task_index"] for i in indices})
            },
        )
    return result


def teacher_gate(metrics):
    validation = metrics["validation"]
    mse = validation["pooled"]["mse"]
    checks = dict(
        pooled_improvement=mse["adapt"] <= 0.95 * mse["raw"],
        beats_constant_bias=mse["adapt"] < mse["bias"],
        all_sigmas_nonregressing=all(
            g["mse"]["adapt"] <= 1.05 * g["mse"]["raw"] for g in validation["by_sigma"].values()
        ),
    )
    return dict(passed=all(checks.values()), checks=checks)


def execute(args):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "025 fit requires explicitly empty CUDA/LD",
    )
    require(args.seed == 42, "025 fixed fit seed is42")
    evidence = Evidence()
    prepared = evidence.json(args.prepared, args.prepared_sha256)
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    require(
        prepared["config"] == config
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["registration"] == reference(PLAN),
        "registered025 preparation/config differs",
    )
    for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
        for path, digest in prepared[key].items():
            evidence.check(path, digest)
    require(str(Path(__file__).resolve()) in prepared["source_sha256"], "preparation omitted actual fit entrypoint")
    for path, item in prepared["weight_files"].items():
        stat = Path(path).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]), "frozen8400 weight stat differs")
    features = evidence.json(args.features, args.features_sha256)
    require(
        features["complete"] is True
        and features["passed"] is True
        and features["prepared_ref"] == reference(args.prepared),
        "feature phase is incomplete or bound to another preparation",
    )
    root = Path(prepared["output_root"])
    require(
        root == (REPO / config["output_dir"]).resolve() and root.is_relative_to(REPO / "outputs"),
        "fit output root differs",
    )
    output = root / "fit"
    output.mkdir(exist_ok=False)
    result = dict(
        schema_version="s2a.frozen-readout-fit.v1",
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        features_ref=reference(args.features),
        models={},
        teacher_metrics=None,
        eligible_for_sampling=False,
        producer_fit_groups=0,
        fit_attempts=0,
        new_model_calls=0,
        physics_steps=0,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        ld_library_path=os.environ["LD_LIBRARY_PATH"],
    )
    try:
        import torch

        arrays, rows = load_rows(prepared, features, evidence, torch)
        selected = np.asarray([r["split"] == "train" for r in rows])
        training_rows = [r for r in rows if r["split"] == "train"]
        weights = hierarchical_weights(training_rows)
        hidden = arrays["hidden"][selected]
        residual = arrays["target"][selected, :10].astype(np.float64) - arrays["base"][selected, :10].astype(np.float64)
        models = {}
        for name, fit in (("adapt", ResidualReadout.fit), ("bias", ResidualReadout.fit_bias)):
            result["fit_attempts"] += 1
            model = fit(hidden, residual, weights)
            result["producer_fit_groups"] += 1
            path = output / f"{name}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, weights=model.weights)
            result["models"][name] = reference(path)
            models[name] = model
        context = DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)
        base = torch.from_numpy(arrays["base"]).to(torch.bfloat16)
        hidden = torch.from_numpy(arrays["hidden"]).to(torch.bfloat16)
        ids = torch.zeros(len(base), dtype=torch.long)
        predictions = dict(raw=base.float().numpy())
        with torch.inference_mode():
            for name, model in models.items():
                predictions[name] = model.apply_torch(hidden, base, context, ids).float().numpy()
                require(
                    np.array_equal(predictions[name][:, 10:], arrays["base"][:, 10:]),
                    "readout changed inactive118 head slots",
                )
        predictions_path = output / "teacher_predictions.npz"
        with predictions_path.open("xb") as stream:
            np.savez_compressed(stream, **predictions, training_weights=weights)
        result["teacher_predictions"] = reference(predictions_path)
        result["teacher_rows"] = rows
        result["teacher_metrics"] = teacher_metrics(predictions, arrays["target"], rows)
        gate = teacher_gate(result["teacher_metrics"])
        result.update(
            complete=True,
            passed=True,
            eligible_for_sampling=gate["passed"],
            teacher_gate=gate,
            torch_version=torch.__version__,
            deployment_dtype="torch.bfloat16",
            ridge=0.1,
            fitted_rows=int(selected.sum()),
            validation_rows=int((~selected).sum()),
            training_episodes=config["train_episodes"],
            validation_episodes=config["validation_episodes"],
            fit_input_contract="active10(target-base), phi=[hidden/sqrt1280,1], weights=episode/anchor/sigma/endpoint",
        )
        evidence.recheck()
        for path, item in prepared["weight_files"].items():
            stat = Path(path).stat()
            require(
                (stat.st_size, stat.st_mtime_ns) == (item["bytes"], item["mtime_ns"]),
                "frozen8400 weight stat changed during CPU fit",
            )
    except BaseException as error:
        result.update(complete=False, passed=False, eligible_for_sampling=False)
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result["input_sha256"] = evidence.hashes
        write_new(output / "fit_result.json", result)
    return output / "fit_result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    path = execute(args)
    print(json.dumps(dict(result=reference(path))))


if __name__ == "__main__":
    main()
