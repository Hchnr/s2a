"""148 independent saved-output scoring and full qualified-source audit."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from scipy.spatial.transform import Rotation

from s2a.artifacts import sha256
from s2a.owned_receipt_contract import normal_receipts


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/148_left_table_forward_qualification_20261025.md")
    train = Path("outputs/left_table_forward_equations_20261025")
    fit = Path("outputs/left_table_fit_20261025")
    validation = Path("outputs/left_table_validation_20261025")
    prepared = e.json(train / "result.json")
    assert prepared["complete"] and prepared["rows"] == 5072 and len(prepared["contrast_sources"]) == 207
    assert len({r["equation_group"] for r in prepared["contrast_sources"]}) == 207
    for result in (prepared,):
        for p, h in result["source_sha256"].items():
            e.check(p, h)
    for root, modes in ((fit, ("fit", "verify")), (validation, ("validation",))):
        for p, h in e.json(root / "source_sha256.json").items():
            e.check(p, h)
        execution = e.json(root / "execution_result.json")
        assert execution["complete"]
        for mode in modes:
            folder = root / mode
            launch = normal_receipts(
                e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable"
            )
            child = launch["child"]
            p = Path("/proc") / str(child["pid"]) / "stat"
            assert not p.exists() or int(p.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
    numeric = e.json(fit / "independent_verified.json")
    assert numeric["complete"] and numeric["numerically_verified"] and numeric["rank"] == 1128
    assert numeric["max_coefficient_difference"] <= 1e-8 and numeric["maximum_regularized_gradient"] <= 1e-10
    ref = numeric["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    parameters = e.json(ref["path"])
    original = e.json(prepared["parameter_ref"]["path"])
    e.check(prepared["parameter_ref"]["path"], prepared["parameter_ref"]["sha256"])
    for key in (
        "state_mean",
        "state_scale",
        "translation_mean",
        "translation_scale",
        "frequencies",
        "joint_frequencies",
    ):
        np.testing.assert_array_equal(parameters[key], original[key])
    config = e.json("configs/left_table_forward_20261025.json")
    ref = config["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    source = e.json(ref["path"])
    scores = e.json(validation / "validation_result.json")
    assert scores["complete"] and scores["offline_forward_eligible"]
    assert len(scores["holdout"]) == len(scores["local_response"]) == 6
    for row in scores["holdout"]:
        entry = next(r for r in source["trajectories"] if r["index"] == row["index"])
        assert entry["split"] == "holdout"
        for ref in (entry["cache_ref"], row["predictions_ref"]):
            e.check(ref["path"], ref["sha256"])
        with np.load(entry["cache_ref"]["path"], allow_pickle=False) as a:
            target = np.c_[a["y_motion"], a["y_rotation_increment"], 0.05 * a["y_twist"]]
        assert len(target) == row["rows"]
        with np.load(row["predictions_ref"]["path"], allow_pickle=False) as a:
            for name in ("fitted", "nominal"):
                predicted = a[name]
                rot = (Rotation.from_rotvec(predicted[:, 3:6]) * Rotation.from_rotvec(target[:, 3:6]).inv()).magnitude()
                measured = dict(rotation=float(np.sqrt(np.mean(rot**2))))
                for key, sl, scale in (
                    ("position", slice(0, 3), 1),
                    ("linear_velocity", slice(6, 9), 20),
                    ("angular_velocity", slice(9, 12), 20),
                ):
                    error = scale * (predicted[:, sl] - target[:, sl])
                    measured[key] = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
                for key, value in measured.items():
                    np.testing.assert_allclose(value, row["metrics"][name][key], rtol=0, atol=1e-12)
        assert all(row["gates"].values())
        for key in row["gates"]:
            assert row["metrics"]["fitted"][key] <= row["metrics"]["nominal"][key]
    assert [r["anchor_group"] for r in scores["local_response"]] == list(range(6))
    for row in scores["local_response"]:
        for channel, m in row["metrics"].items():
            truth = np.asarray(m["measured"])
            f = float(np.linalg.norm(np.asarray(m["predicted"]) - truth))
            n = float(np.linalg.norm(np.asarray(m["nominal"]) - truth))
            np.testing.assert_allclose(
                [f, n], [m["fitted_frobenius_error"], m["nominal_frobenius_error"]], rtol=0, atol=1e-12
            )
            assert row["gates"][channel] and f <= n
    e.recheck()
    out = Path("outputs/left_table_forward_qualification_20261025")
    write_once(
        out / "qualification.json",
        dict(
            complete=True,
            offline_forward_eligible=True,
            holdout_gates=24,
            local_response_gates=24,
            stage_actual_wait_verified=True,
            historical_root_owner_wait_available=False,
            parameter_ref=numeric["parameter_ref"],
            validation_ref=dict(
                path=str((validation / "validation_result.json").resolve()),
                sha256=sha256(validation / "validation_result.json"),
            ),
            source_sha256=e.hashes,
            control_qualified=False,
            new_fits=0,
            physics_steps=0,
            model_calls=0,
            fresh_cases=0,
        ),
    )
    print(
        "148 all48 physical gates independently rescored;three stage receipts and sources verified;control not qualified"
    )


if __name__ == "__main__":
    main()
