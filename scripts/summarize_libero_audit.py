"""Validate physical traces and summarize frozen October 19 control experiments."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import load_config, sha256, write_json
from s2a.geometry import pose_error


def validate_matrix(config_path):
    config, output = load_config(config_path)
    matrix = json.loads((output / "matrix.json").read_text())
    expected = {(c["name"], t) for c in config["cases"] for t in config["trials"]}
    assert not matrix["failures"] and len(matrix["results"]) == len(expected)
    checked, hashes, signatures, traces = set(), {}, set(), {}
    for result in matrix["results"]:
        case, trial = result["case"], result["trial"]
        key = (case["name"], trial)
        assert key in expected and key not in checked and result["error"] is None
        checked.add(key)
        dest = output / case["name"] / f"trial{trial}"
        manifest = json.loads((dest / "manifest.json").read_text())
        assert manifest["config"] == config and manifest["case"] == case and not manifest["code"]["dirty"]
        signatures.add(
            tuple(
                manifest["code"]["files_sha256"][k]
                for k in (
                    "src/s2a/libero_audit.py",
                    "src/s2a/libero_simulation.py",
                    "src/s2a/gripper.py",
                    "src/s2a/osc.py",
                )
            )
        )
        rows = [json.loads(line) for line in (dest / "steps.jsonl").read_text().splitlines()]
        assert len(rows) == result["steps"] and rows
        errors = []
        previous = np.zeros(6)
        for i, row in enumerate(rows):
            assert row["step"] == i and abs(row["actual_dt"] - 0.05) < 1e-8
            error = pose_error(np.array(row["desired_source"]), np.array(row["after_source"]))
            np.testing.assert_allclose(error, row["error"], atol=1e-12)
            command = np.asarray(row["executed"])
            assert np.isfinite(command).all() and np.max(np.abs(command)) <= 1 + 1e-9
            if case["arm"] == "response":
                assert np.max(abs(command[:6] - previous)) <= config["max_command_rate"] * row["actual_dt"] + 1e-8
            previous = command[:6]
            if i:
                np.testing.assert_allclose(row["measured_source"], rows[i - 1]["after_source"], atol=1e-12)
            errors.append(error)
        errors = np.asarray(errors)
        np.testing.assert_allclose(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))), result["position_rmse_m"])
        assert bool(rows[-1]["success"]) == result["success"]
        traces[key] = rows
        for path in dest.iterdir():
            if path.is_file():
                hashes[str(path)] = sha256(path)
    assert checked == expected and len(signatures) == 1
    groups = []
    for case in config["cases"]:
        rows = [r for r in matrix["results"] if r["case"]["name"] == case["name"]]
        groups.append(
            dict(
                name=case["name"],
                episodes=len(rows),
                successes=sum(r["success"] for r in rows),
                position_rmse_mm=float(np.mean([r["position_rmse_m"] for r in rows]) * 1000),
                rotation_rmse_rad=float(np.mean([r["rotation_rmse_rad"] for r in rows])),
                opening_rmse=float(np.mean([r["opening_rmse"] for r in rows])),
                failed_trials=[r["trial"] for r in rows if not r["success"]],
            )
        )
    paired = []
    for grip in ("position", "native", "intent", "latch"):
        for trial in config["trials"]:
            names = ("affine_" + grip, "response_" + grip)
            if not all((name, trial) in traces for name in names):
                continue
            a, b = [traces[(name, trial)] for name in names]
            n = min(len(a), len(b))
            np.testing.assert_allclose(
                [r["desired_source"] for r in a[:n]], [r["desired_source"] for r in b[:n]], atol=1e-12
            )
            metrics = []
            for rows in (a[:n], b[:n]):
                e = np.array([r["error"] for r in rows])
                metrics.append(
                    dict(
                        position_rmse_mm=float(np.sqrt(np.mean(np.sum(e[:, :3] ** 2, axis=1))) * 1000),
                        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(e[:, 3:] ** 2, axis=1)))),
                    )
                )
            paired.append(dict(gripper=grip, trial=trial, common_steps=n, affine=metrics[0], response=metrics[1]))
    return dict(config=config_path, cases=len(checked), groups=groups, matched_prefix=paired), hashes


def validate_policy(config_path, allow_incomplete):
    config, output = load_config(config_path)
    results, hashes, traces = [], {}, {}
    expected = 2 * len(config["trials"])
    for candidate in ("baseline", "compensated"):
        for trial in config["trials"]:
            dest = output / "policy" / candidate / f"trial{trial}"
            if not (dest / "result.json").is_file():
                if allow_incomplete:
                    continue
                raise FileNotFoundError(dest / "result.json")
            result = json.loads((dest / "result.json").read_text())
            assert result["error"] is None and (result["success"] or result["steps"] == config["max_steps"])
            rows = [json.loads(s) for s in (dest / "trace_steps.jsonl").read_text().splitlines()]
            assert len(rows) == result["steps"]
            previous = np.zeros(6)
            for row in rows:
                np.testing.assert_allclose(
                    pose_error(np.array(row["desired_source"]), np.array(row["after_source"])),
                    row["tracking_error_source"],
                    atol=1e-12,
                )
                assert abs(row["actual_dt"] - 0.05) < 1e-8 and np.max(abs(np.array(row["executed"]))) <= 1 + 1e-9
                if candidate == "compensated":
                    assert (
                        np.max(abs(np.array(row["executed"][:6]) - previous))
                        <= config["max_command_rate"] * 0.05 + 1e-8
                    )
                previous = np.array(row["executed"][:6])
            traces[(candidate, trial)] = rows
            results.append(result)
            for path in dest.iterdir():
                if path.is_file():
                    hashes[str(path)] = sha256(path)
    paired = []
    for trial in config["trials"]:
        if not all((name, trial) in traces for name in ("baseline", "compensated")):
            continue
        a, b = [output / "policy" / name / f"trial{trial}" for name in ("baseline", "compensated")]
        ma, mb = [json.loads((d / "manifest.json").read_text()) for d in (a, b)]
        for k in (
            "src/s2a/libero_simulation.py",
            "src/s2a/osc.py",
            "src/s2a/gripper.py",
            "configs/libero_policy_20261019.json",
        ):
            assert ma["code"]["files_sha256"][k] == mb["code"]["files_sha256"][k]
        assert sha256(a / "input_0000.png") == sha256(b / "input_0000.png")
        aa, bb = [
            np.asarray(json.loads((d / "response_0000.json").read_text())["outputs"][0]["content"]["action"])
            for d in (a, b)
        ]
        np.testing.assert_array_equal(aa, bb)
        paired.append(dict(trial=trial, first_prediction_max_abs=float(np.max(abs(aa - bb)))))
    groups = []
    for name in ("baseline", "compensated"):
        rows = [r for r in results if r["candidate"] == name]
        if rows:
            groups.append(
                dict(
                    name=name,
                    episodes=len(rows),
                    successes=sum(r["success"] for r in rows),
                    position_rmse_mm=float(np.mean([r["position_rmse_m"] for r in rows]) * 1000),
                    rotation_rmse_rad=float(np.mean([r["rotation_rmse_rad"] for r in rows])),
                )
            )
    return dict(
        cases=len(results), expected_cases=expected, complete=len(results) == expected, groups=groups, paired=paired
    ), hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    if args.seed != 42:
        raise ValueError("frozen seed is 42")
    regression, h1 = validate_matrix("configs/libero_audit_20261019.json")
    holdout, h2 = validate_matrix("configs/libero_holdout_20261019.json")
    policy, h3 = validate_policy("configs/libero_policy_20261019.json", args.allow_incomplete)
    report = dict(
        seed=42,
        regression=regression,
        holdout=holdout,
        policy=policy,
        integrity_passed=True,
        hardware_executed=False,
        limitations=[
            "One task; episode holdout only for adapter, not model",
            "Oracle reference is privileged; converter inputs are causal",
            "Synchronous model inference pauses physics; no deployment latency claim",
            "Latch assumes near-full-open release; not a unique force inverse",
        ],
    )
    out = Path("outputs/libero_audit_20261019")
    write_json(out / "summary.json", report)
    write_json(out / "artifact_hashes.json", {**h1, **h2, **h3})
    write_json(Path("reports/libero_control_audit_20261019.json"), report)
    lines = [
        "# LIBERO 控制合同与夹爪诊断 — 2026-10-19",
        "",
        "完整性检查通过与任务效果分别报告。所有结果为独立MuJoCo仿真，没有真机执行。",
        "",
    ]
    for title, group in [("已见5条回归", regression), ("额外10条episode", holdout), ("8800实时图像闭环", policy)]:
        lines += [
            "## " + title,
            "",
            "| 条件 | 成功 | 平均位置RMSE mm | 平均旋转RMSE rad |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in group["groups"]:
            lines.append(
                f"| {row['name']} | {row['successes']}/{row['episodes']} | {row['position_rmse_mm']:.3f} | {row['rotation_rmse_rad']:.6f} |"
            )
        lines.append("")
    lines += [
        "oracle使用obs[t]作为actions[t]执行后的目标，保持原world frame；native正控与native-gripper诊断使用专家命令，其余候选不读取专家命令。",
        "",
        "RMSE为各episode完整执行区间的均值，提前成功会改变长度；JSON另列相同参考、相同步数前缀的arm对照。",
        "",
        "五个已见episode属于诊断回归；额外十条在数值评测前冻结，属于同任务adapter episode对照，不能声称世界模型未见任务泛化。",
        "",
        "模型两分支在首chunk后按各自真实图像重新预测，后续参考允许分叉；仅首chunk可视为相同模型目标。",
        "",
        "原生初始化缓存、夹爪控制语义与模型输出质量的证据分别见实施记录和结论。完整trace与指纹位于outputs/。",
        "",
    ]
    if not policy["complete"]:
        lines += ["**实施中：实时模型矩阵尚未完成，不计为全部通过。**", ""]
    Path("reports/libero_control_audit_20261019.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in report.items() if k in ("integrity_passed", "hardware_executed")}))


if __name__ == "__main__":
    main()
