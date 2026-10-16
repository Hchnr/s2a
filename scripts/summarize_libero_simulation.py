"""Validate actual MuJoCo traces and report paired simulation evidence."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO, load_config, sha256, write_json
from s2a.geometry import inverse, pose_error, rot6d_to_matrix, transform, validate_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_simulation.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen config")
    fixture = json.loads(Path(config["fixture"]).read_text())
    h = np.asarray(json.loads(Path(config["tool_contract"]).read_text())["source_from_tcp"])
    groups, cases, hashes, inputs = defaultdict(list), [], {}, {}
    expected = {"probe": 2, "native": 5, "oracle": 10, "prediction_replay": 10, "policy": 10}
    for stage, count in expected.items():
        paths = sorted((output / stage).glob("*/trial*/result.json"))
        if not args.allow_incomplete and len(paths) != count:
            raise ValueError(f"Expected {count} completed {stage} cases, got {len(paths)}")
        for path in paths:
            result = json.loads(path.read_text())
            if result["error"]:
                raise ValueError(f"Execution error in {path}: {result['error']}")
            dest = path.parent
            manifest = json.loads((dest / "manifest.json").read_text())
            if result["initial_state_sha256"] != fixture["episodes"][result["trial"]]["initial_state_sha256"]:
                raise ValueError("initial state mismatch")
            for p, digest in manifest["inputs_sha256"].items():
                if p not in inputs:
                    if sha256(p) != digest:
                        raise ValueError(f"Input file changed: {p}")
                    inputs[p] = digest
                elif inputs[p] != digest:
                    raise ValueError("Different input fingerprints across trials")
            for key in (
                "inverse_weight",
                "max_command_rate",
                "seed",
                "control_fps",
                "execution_horizon",
                "sampling_steps",
            ):
                if manifest["config"][key] != config[key]:
                    raise ValueError(f"Protocol differs for {key}: {path}")
            rows = [json.loads(line) for line in (dest / "trace_steps.jsonl").read_text().splitlines()]
            if len(rows) != result["steps"] or not rows:
                raise ValueError("trace/result length mismatch")
            command = np.asarray([r["executed"][:6] for r in rows])
            contract = json.loads((dest / "controller.json").read_text())
            if np.any(command < np.asarray(contract["input_min"]) - 1e-6) or np.any(
                command > np.asarray(contract["input_max"]) + 1e-6
            ):
                raise ValueError("sent command exceeds simulator interface bounds")
            if result["candidate"] == "compensated":
                changes = np.diff(np.vstack([np.zeros(6), command]), axis=0)
                if np.max(np.abs(changes)) > config["max_command_rate"] / config["control_fps"] + 1e-9:
                    raise ValueError("sent command exceeds configured rate")
            errors = []
            for i, row in enumerate(rows):
                if row["step"] != i or not np.isclose(row["actual_dt"], 1 / config["control_fps"], atol=1e-9, rtol=0):
                    raise ValueError("invalid step index or physical control interval")
                for key in ("measured_source", "desired_source", "desired_tcp", "after_source"):
                    validate_transform(np.asarray(row[key]))
                for key in ("tracking_error_source", "tracking_error_tcp", "actual_motion", "executed"):
                    if not np.isfinite(row[key]).all():
                        raise ValueError("non-finite simulation trace")
                np.testing.assert_allclose(
                    row["tracking_error_source"],
                    pose_error(np.asarray(row["desired_source"]), np.asarray(row["after_source"])),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    row["tracking_error_tcp"],
                    pose_error(np.asarray(row["desired_tcp"]), np.asarray(row["after_source"]) @ h),
                    atol=1e-12,
                )
                if row.get("tracking_valid", True):
                    errors.append(row["tracking_error_tcp"])
            errors = np.asarray(errors)
            for key, part in (("position_rmse_m", slice(0, 3)), ("rotation_rmse_rad", slice(3, 6))):
                value = np.sqrt(np.mean(np.sum(errors[:, part] ** 2, axis=1)))
                np.testing.assert_allclose(result[key], value, atol=1e-12)
            with np.load(dest / "trace.npz", allow_pickle=False) as trace:
                if not all(np.isfinite(trace[k]).all() for k in trace.files):
                    raise ValueError("non-finite trace archive")
            summary = {k: v for k, v in result.items() if k != "requests"}
            valid_rows = [r for r in rows if r.get("tracking_valid", True)]
            summary["aperture_rmse"] = float(
                np.sqrt(np.mean([(r["opening_target"] - r["opening_after"]) ** 2 for r in valid_rows]))
            )
            summary["unscored_terminal_rows"] = len(rows) - len(valid_rows)
            summary.update(
                request_count=len(result["requests"]),
                code_commit=manifest["code"]["commit"],
                code_dirty=manifest["code"]["dirty"],
                core_sources_sha256={
                    k: v
                    for k, v in manifest["code"]["files_sha256"].items()
                    if k in ("src/s2a/osc.py", "src/s2a/libero_simulation.py", "configs/libero_simulation.json")
                },
            )
            if stage == "prediction_replay" and result["candidate"] == "baseline":
                old_path = (
                    Path(config["historical_results"]).parent / "closed_loop" / f"trial{result['trial']}" / "trace.json"
                )
                old = json.loads(old_path.read_text())
                old_command = np.asarray([r["executed"] for r in old["conversions"]])[:, :6]
                position = np.asarray([r["after_source"] for r in rows])[:, :3, 3]
                old_position = np.asarray([r["robot0_eef_pos"] for r in old["observations"][1:]])
                observation = old["observations"][0]
                target = (
                    transform(
                        observation["robot0_eef_pos"], Rotation.from_quat(observation["robot0_eef_quat"]).as_matrix()
                    )
                    @ h
                )
                composition_error = 0.0
                for row in old["conversions"][:8]:
                    action = np.asarray(row["predicted"])
                    target = target @ transform(action[:3], rot6d_to_matrix(action[3:9]))
                    composition_error = max(
                        composition_error, float(np.max(abs(target @ inverse(h) - row["target_pose"])))
                    )
                if composition_error > 1e-10:
                    raise ValueError("TCP target integration differs from historical adapter")
                np.testing.assert_allclose(command[:8], old_command[:8], atol=1e-10, rtol=0)
                np.testing.assert_allclose(position[:8], old_position[:8], atol=1e-10, rtol=0)
                summary["historical_reproduction"] = dict(
                    first8_command_max_abs=float(np.max(abs(command[:8] - old_command[:8]))),
                    first8_target_composition_max_abs=composition_error,
                    first8_position_max_norm_m=float(np.max(np.linalg.norm(position[:8] - old_position[:8], axis=1))),
                    full_position_max_norm_m=float(np.max(np.linalg.norm(position - old_position, axis=1))),
                    note="Initial action/state parity verified; long contact rollouts may diverge under roundoff.",
                )
            if stage == "policy":
                expected_checkpoint = (
                    Path(config["checkpoint_run"]) / "checkpoints" / config["checkpoint_iteration"] / "model"
                )
                if Path(manifest["server_info"]["checkpoint_path"]) != expected_checkpoint:
                    raise ValueError("different checkpoint in policy comparison")
            cases.append(summary)
            groups[(stage, result["candidate"])].append(summary)
            for artifact in dest.iterdir():
                if artifact.is_file():
                    hashes[str(artifact.relative_to(output))] = sha256(artifact)
    cohorts = []
    for (stage, candidate), values in groups.items():
        cohort = dict(
            stage=stage, candidate=candidate, completed=len(values), successes=sum(v["success"] for v in values)
        )
        for key in ("position_rmse_m", "rotation_rmse_rad", "aperture_rmse", "saturation_fraction"):
            cohort["mean_" + key] = float(np.mean([v[key] for v in values]))
        cohorts.append(cohort)
    comparisons = []
    for stage in expected:
        for trial in config["trials"]:
            values = {v["candidate"]: v for v in cases if v["stage"] == stage and v["trial"] == trial}
            if set(values) != {"baseline", "compensated"}:
                continue
            dest = output / stage
            traces = {
                c: [
                    json.loads(line)
                    for line in (dest / c / f"trial{trial}" / "trace_steps.jsonl").read_text().splitlines()
                ]
                for c in values
            }
            n = min(map(len, traces.values()))
            paired = dict(stage=stage, trial=trial, common_steps=n)
            for c, rows in traces.items():
                errors = np.asarray([r["tracking_error_tcp"] for r in rows[:n]])
                paired[c] = dict(
                    position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                    rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                )
            if stage in ("oracle", "prediction_replay"):
                for key in ("desired_tcp", "opening_target"):
                    np.testing.assert_allclose(
                        [r[key] for r in traces["baseline"][:n]],
                        [r[key] for r in traces["compensated"][:n]],
                        atol=1e-12,
                    )
            if stage == "policy":
                manifests = [json.loads((dest / c / f"trial{trial}" / "manifest.json").read_text()) for c in values]
                if manifests[0]["server_info"] != manifests[1]["server_info"]:
                    raise ValueError("paired policy trials used different serving replicas")
                results = [json.loads((dest / c / f"trial{trial}" / "result.json").read_text()) for c in values]
                paired["same_initial_canvas"] = (
                    results[0]["requests"][0]["observation_sha256"] == results[1]["requests"][0]["observation_sha256"]
                )
                if not paired["same_initial_canvas"]:
                    raise ValueError("paired initial camera images differ")
                for key in ("src/s2a/osc.py", "src/s2a/libero_simulation.py", "configs/libero_simulation.json"):
                    if manifests[0]["code"]["files_sha256"][key] != manifests[1]["code"]["files_sha256"][key]:
                        raise ValueError(f"Paired core implementation/config changed: {key}")
                first_predictions = [
                    np.asarray(
                        json.loads((dest / c / f"trial{trial}" / "response_0000.json").read_text())["outputs"][0][
                            "content"
                        ]["action"]
                    )
                    for c in values
                ]
                paired["initial_prediction_max_abs"] = float(np.max(abs(first_predictions[0] - first_predictions[1])))
                paired["same_core_sources"] = True
                horizon = config["execution_horizon"]
                np.testing.assert_allclose(
                    [r["desired_tcp"] for r in traces["baseline"][:horizon]],
                    [r["desired_tcp"] for r in traces["compensated"][:horizon]],
                    atol=1e-12,
                )
                paired["same_initial_chunk_targets"] = True
                paired["initial_chunk_tracking"] = {}
                for c, rows in traces.items():
                    errors = np.asarray([r["tracking_error_tcp"] for r in rows[:horizon]])
                    paired["initial_chunk_tracking"][c] = dict(
                        position_rmse_m=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                        rotation_rmse_rad=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                    )
            comparisons.append(paired)
    complete = len(cases) == sum(expected.values())
    if complete:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_dir = output / "figures"
        figure_dir.mkdir(exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        stages = ["probe", "oracle", "prediction_replay", "policy"]
        for j, (metric, scale, label) in enumerate(
            (
                ("position_rmse_m", 1000, "TCP position RMSE (mm)"),
                ("rotation_rmse_rad", 180 / np.pi, "Rotation RMSE (deg)"),
            )
        ):
            for offset, candidate, color in ((-0.18, "baseline", "#607d8b"), (0.18, "compensated", "#1976d2")):
                values = [
                    next(c["mean_" + metric] for c in cohorts if c["stage"] == stage and c["candidate"] == candidate)
                    * scale
                    for stage in stages
                ]
                axes[j].bar(np.arange(4) + offset, values, width=0.36, label=candidate, color=color)
            axes[j].set_xticks(
                range(4), ["Small motion", "Expert states", "Saved predictions", "Model closed loop"], rotation=18
            )
            axes[j].set_ylabel(label)
            axes[j].set_yscale("log")
            axes[j].grid(axis="y", alpha=0.2)
        axes[0].legend()
        figure.suptitle("Independent MuJoCo execution: tracking and task success are separate")
        for extension in ("png", "pdf"):
            artifact = figure_dir / f"control_tracking.{extension}"
            figure.savefig(artifact, dpi=180)
            hashes[str(artifact.relative_to(output))] = sha256(artifact)
        plt.close(figure)
    report = dict(
        complete=complete,
        checkpoint=str(Path(config["checkpoint_run"]) / "checkpoints" / config["checkpoint_iteration"] / "model"),
        config=config,
        cohorts=cohorts,
        cases=cases,
        common_prefix_comparisons=comparisons,
        source_data_sha256=inputs,
        hardware_executed=False,
        physics="MuJoCo 2.3.7 / robosuite 1.4.1 / original LIBERO controller",
        evidence={
            "probe": "independent small-motion simulation; relative motion requests identical, current pose differs by branch",
            "oracle": "expert observed delta integrated from actual initial TCP; same fixed desired trajectory; not model prediction",
            "prediction_replay": "historical iter5600 saved model targets, same fixed trajectory; no fresh image-conditioned model calls",
            "policy": "frozen iter8800 checkpoint, fresh branch images, same initial states and sampling; subsequent predictions may diverge",
        },
    )
    fingerprints = output / "checkpoint_fingerprints.json"
    if fingerprints.is_file():
        checkpoint = json.loads(fingerprints.read_text())
        for path, entry in checkpoint["checkpoint_files"].items():
            if Path(path).stat().st_size != entry["bytes"]:
                raise ValueError("checkpoint file size changed after full SHA256 capture")
        report["checkpoint_fingerprint"] = dict(
            sha256=sha256(fingerprints),
            checkpoint_files=len(checkpoint["checkpoint_files"]),
            checkpoint_bytes=checkpoint["total_checkpoint_bytes"],
            inference_source_files=len(checkpoint["cosmos_sources_sha256"]),
            note="All checkpoint bytes SHA256 captured; current file sizes checked, full per-file ledger in outputs.",
        )
        hashes[fingerprints.name] = sha256(fingerprints)
    elif complete:
        raise ValueError("Full checkpoint SHA256 ledger is required for a final report")
    for filename, key in (
        ("initial_prediction_diagnostics.json", "initial_prediction_diagnostics"),
        ("engineering_checks.json", "engineering_checks"),
        ("serving_environment.json", "serving_environment"),
    ):
        path = output / filename
        if path.is_file():
            report[key] = json.loads(path.read_text())
            hashes[filename] = sha256(path)
    write_json(output / "summary.json", report)
    lines = [
        "# LIBERO S2A 独立仿真对照",
        "",
        f"完成状态：{len(cases)}/{sum(expected.values())}；{'全部执行完成，完整性检查通过。' if complete else '进行中，未执行项不计为通过。'}",
        "",
        f"Checkpoint：`{config['checkpoint_iteration']}`，同一warm-start run；历史5600权重已清理，旧0/5不能直接作为新权重的baseline。",
        "",
        "任务：put the bowl on the plate；5个冻结初始状态，seed42+trial，20Hz，chunk16/execute8，35采样步，guidance1.5、shift5、无warmup/EMA，模型闭环最多1000步。",
        "",
        "| 输入 / 证据 | 转换 | 完成 | 任务成功 | 平均位置RMSE (m) | 平均旋转RMSE (rad) | 平均夹爪opening RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort in cohorts:
        success = "不适用" if cohort["stage"] == "probe" else f"{cohort['successes']}/{cohort['completed']}"
        lines.append(
            f"| {cohort['stage']} | {cohort['candidate']} | {cohort['completed']} | {success} | "
            f"{cohort['mean_position_rmse_m']:.6f} | {cohort['mean_rotation_rmse_rad']:.6f} | {cohort['mean_aperture_rmse']:.6f} |"
        )
    lines += [
        "",
        "baseline沿用TCP局部delta积分、TCP→source转换、OSC affine inverse及原夹爪position目标。新增补偿使用固定内环PD响应的二阶解析近似、当前实测速度、0.5逆权重及最终每轴5/s命令变化率限制；旋转误差超过0.25rad时关闭旋转逆近似。原OSC增益、物理模型和夹爪控制路径不变。",
        "",
        "probe和oracle输入来自独立运动请求或专家状态。prediction_replay使用旧5600预测且不重新查询图像。policy才是8800权重的实时图像闭环；分支执行不同后，后续预测允许不同。所有结果均为仿真，没有真机执行。",
        "",
        "表中是各episode完整执行期的RMSE均值；成功提前停止可改变样本长度。JSON另存两候选相同步数前缀指标，oracle/固定预测逐点核对相同目标。policy误差衡量各自模型目标的跟踪，不能把下降单独解释为模型预测更正确。",
        "",
        "5个历史episode属于训练可见数据，一个任务的回归试验不代表泛化。解析内环模型未包含接触、力矩限幅和完整非线性；状态跟踪改善与任务成功分别判断。原生expert正控如果未成功也如实计入，不排除样本。",
        "",
        "两候选的夹爪均沿用历史评测的aperture→原position actuator转换，本轮补偿仅作用于机械臂；native正控执行原始gripper action，不覆盖为aperture目标。补偿读取当前仿真control site的线速度/角速度，反馈没有额外估计噪声。历史baseline最初8步的发送命令/位置已与原记录逐项比对；长接触轨迹不要求浮点位级复现。",
        "",
        "policy采用与历史相同的同步仿真：模型推理期间物理时钟暂停，未模拟推理延迟，不代表实时部署性能。初始chunk的专家标签仅由独立离线诊断脚本读取，未进入正在运行的策略或用于重新调参。",
        "",
        "完整请求、物理前后状态、目标、原始/发送命令、控制器参数、视频与源码/数据指纹见 `outputs/libero_simulation/`。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    for name in ("summary.json", "report.md"):
        hashes[name] = sha256(output / name)
    write_json(output / "artifact_hashes.json", hashes)
    write_json(REPO / "reports/libero_simulation_20261016.json", report)
    (REPO / "reports/libero_simulation_20261016.md").write_text("\n".join(lines))
    write_json(
        REPO / "reports/libero_simulation_validation_20261016.json",
        dict(
            complete=complete,
            executed_cases=len(cases),
            expected_cases=sum(expected.values()),
            checked_output_files=len(hashes),
            checked_source_files=len(inputs),
            finite_traces=True,
            physical_clock=True,
            sent_bounds=True,
            compensated_sent_rate=True,
            paired_fixed_references=True,
            performance_accepted=None,
            note="Integrity checks are separate from task success and tracking improvement.",
        ),
    )
    print(json.dumps(dict(complete=complete, cases=len(cases), checked_files=len(hashes))))


if __name__ == "__main__":
    main()
