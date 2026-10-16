"""Standalone study report and publication-style diagnostic plots."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import sha256, write_json


def load_result(out, name):
    return json.loads((out / name).read_text())


def render_report(out):
    out = Path(out)
    audit = load_result(out, "data_audit.json")
    geo = load_result(out, "geometry_metrics.json")
    calibration = load_result(out, "calibration_metrics.json")
    closed = load_result(out, "closed_loop_model_metrics.json")
    params = load_result(out, "adapter_params.json")
    manifest = load_result(out, "sample_manifest.json")
    lines = [
        "# DROID episode 0 — 2026-10-16",
        "",
        "证据范围：只读日志、几何/参考接口代数、单 episode 辨识模型内 rollout。没有独立物理仿真或真机验证。",
        "",
        f"代码 `{manifest['code']['commit']}`；dirty=`{manifest['code']['dirty']}`；seed=42。",
        f"sample SHA256 `{manifest['sample_sha256']}`。完整文件/代码指纹与配置见 sample_manifest.json。",
        "",
        "## 数据与几何",
        "",
        f"选定 {audit['selected_transitions']} 个转移，保留 {audit['stored_observations_with_history']} 个观测（含历史与 endpoint）。",
        f"物理时间 {audit['physical_interval_s']} s；媒体时间 {audit['media_interval_s']} s。",
        "",
        "| 检查 | 位置最大误差 m | 旋转最大误差 rad |",
        "| --- | ---: | ---: |",
    ]
    for key in ["single_step", "integrated", "tcp_to_r1_flange", "tcp_flange_roundtrip"]:
        value = geo[key]
        lines.append(f"| {key} | {value['position_max_m']:.6g} | {value['rotation_max_rad']:.6g} |")
    difference = geo["target_vs_historical_commanded_pose"]
    mismatch = geo["r1_command_decode_vs_recorded_target"]
    lines += [
        "",
        f"observed 下一状态相对原 commanded target 的 RMSE：{difference['position_rmse_m']:.6g} m / {difference['rotation_rmse_rad']:.6g} rad。",
        f"公开参考接口解码历史命令的 target RMSE：{mismatch['position_rmse_m']:.6g} m / {mismatch['rotation_rmse_rad']:.6g} rad；历史接口匹配未确认。",
        "",
        "## 留出响应与命令诊断",
        "",
        f"模型 `{params['model']['structure']}`，history={params['model']['history']}，lag={params['model']['lag']}，ridge={params['model']['ridge']}；fit samples={params['model']['diagnostics']['fit_samples']}。",
        f"计划命令时域 {params['dt_command_planned_s']:.9f} s（只取 fit median）；反馈时域使用已发生的物理时间差。",
        "",
        "| 区间 | 一步 NRMSE | 上一运动 NRMSE | 多步 NRMSE | 多步位置 RMSE m | 多步姿态 RMSE rad |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, p in calibration["forward"].items():
        replay = p["logged_command_rollout"]
        lines.append(
            f"| {name} | {p['one_step']['balanced_nrmse']:.6f} | {p['previous_velocity']['balanced_nrmse']:.6f} | {replay['velocity']['balanced_nrmse']:.6f} | {replay['pose']['position_rmse_m']:.6f} | {replay['pose']['rotation_rmse_rad']:.6f} |"
        )
    lines += [
        "",
        "每个独立区间清空命令/运动历史与延迟队列，剔除 8 个 warm-up 转移；validation/test 计分分别 32/52 转移。",
        "归一化尺度来自 fit。lag 用真实物理毫秒数另存 JSON。接触标签缺失，不虚构接触/自由运动分段。",
        "",
        "| 命令诊断 NRMSE | hold/zero | previous command | A | linear calibrated |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in calibration["command_diagnostics"].items():
        row = [
            values[k]["balanced_nrmse"] for k in ["hold_zero", "previous_command", "analytic_A", "linear_calibrated"]
        ]
        lines.append("| " + name + " | " + " | ".join(f"{x:.6f}" for x in row) + " |")
    lines += [
        "",
        "命令诊断共有支持剔除 history/lag 边界，fit/validation/test 为 172/32/52；不以 command RMSE 选择 PID。",
        "线性逆是正则无约束解后范数投影；模型 history 中的运动量只使用已发生的观测。",
        "",
        "## 模型内控制对照",
        "",
        "增益只在 validation 选择；FF/P/PI/PD/PID 具有相同前馈与限制，所有场景/候选/划分分别初始化。",
        "若 validation 全部候选含无效场景，仅报告网格第一组固定诊断，不将其视为选定/整定成功（selection_status 另存）。",
        "初偏、gain=.7、额外一帧延迟、恒扰动、噪声、静止目标均预先固定。参考是 oracle；每一步状态由候选命令推进。",
        "",
    ]
    if closed["executed"]:
        lines += [
            "| 前馈 | 方法 | test 场景平均 tracking score | 场景全部有效 | 全扰动验收 |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for ff, methods in closed["results"].items():
            for method, values in methods.items():
                valid = all(v["valid"] for v in values["test"].values())
                mean = (
                    float(np.mean([v["balanced_tracking_error"] for v in values["test"].values()])) if valid else None
                )
                passed = closed["acceptance"].get(ff, {}).get(method, {}).get("all_perturbations_passed")
                lines.append(
                    f"| {ff} | {method} | {mean:.6f} | {valid} | {passed if passed is not None else 'baseline'} |"
                    if valid
                    else f"| {ff} | {method} | 无效场景见 JSON | {valid} | False |"
                )
        lines += [
            "",
            "tracking score 按 .01m/.05rad 尺度平衡（越小越好），不等同任务成功率。",
            "完整逐场景位置/姿态 RMSE、终点、静止稳态/超调/调节时间、命令变化率、饱和比例/时长与验收细项保存在 closed_loop_model_metrics.json。",
            "任何全扰动验收失败的方法均不能宣称达到方案的扩展标准。模型自洽也不构成真机稳定性证据。",
            "",
            "| 未执行项 | 缺失条件 |",
            "| --- | --- |",
        ]
        for item, reason in closed["unexecuted"].items():
            lines.append(f"| {item} | {reason} |")
    else:
        lines += [f"未执行：{closed['reason']}"]
    lines += [
        "",
        "## 产物",
        "",
        "sample.npz、sample_manifest.json、data_audit.json、analytic_commands.npz、geometry_metrics.json、adapter_params.json、pid_config.json、calibration_metrics.json、forward_replay.npz、closed_loop_model_metrics.json、closed_loop_traces.npz。",
        "源数据保持只读，完整产物在 outputs/droid_episode0/，不进入 Git。精简结论及摘要在 reports/ 与 plans/v0.0.1/。",
        "",
    ]
    for filename in [
        "forward_replay.png",
        "tracking_comparison.png",
        "analytic_initial_offset.png",
        "linear_initial_offset.png",
    ]:
        if (out / "figures" / filename).exists():
            lines += [f"![{filename}](figures/{filename})", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    summary = {
        "data_audit": audit,
        "geometry": geo,
        "selected_model": {k: params["model"][k] for k in ["structure", "history", "lag", "ridge", "diagnostics"]},
        "forward": calibration["forward"],
        "commands": calibration["command_diagnostics"],
        "code_commit": manifest["code"]["commit"],
        "code_dirty": manifest["code"]["dirty"],
        "sample_sha256": manifest["sample_sha256"],
        "source_fingerprints": manifest["files"],
    }
    if closed["executed"]:
        summary.update(
            {
                "closed_loop_test": {
                    ff: {method: values["test"] for method, values in methods.items()}
                    for ff, methods in closed["results"].items()
                },
                "acceptance": closed["acceptance"],
                "unexecuted": closed["unexecuted"],
            }
        )
    write_json(out / "summary.json", summary)
    artifacts = {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.name != "artifact_hashes.json"}
    artifacts.update({"figures/" + p.name: sha256(p) for p in (out / "figures").glob("*") if p.is_file()})
    write_json(out / "artifact_hashes.json", artifacts)
    return summary


def plot_results(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from s2a.geometry import euler_pose_to_transform, pose_error

    out = Path(out)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    sample = np.load(out / "sample.npz")
    replay = np.load(out / "forward_replay.npz")
    closed = load_result(out, "closed_loop_model_metrics.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, name in enumerate(["validation", "test"]):
        f = replay[name + "_frame_index"]
        ids = np.searchsorted(sample["frame_index"], f)
        actual = euler_pose_to_transform(sample["r1_observation_state_cartesian_position"][ids])
        error = pose_error(replay[name + "_predicted_pose"], actual)
        times = sample["source_physical_timestamp"][ids]
        axes[0, column].plot(times, np.linalg.norm(error[:, :3], axis=1))
        axes[1, column].plot(times, np.linalg.norm(error[:, 3:], axis=1))
        for row in range(2):
            axes[row, column].axvline(times[8], color="gray", ls="--", label="score starts")
            axes[row, column].set(xlabel="physical time (s)", title=f"{name}: local model logged-command rollout")
            axes[row, column].grid(alpha=0.2)
        axes[0, column].set_ylabel("position error (m)")
        axes[1, column].set_ylabel("SO(3) error (rad)")
    fig.savefig(figures / "forward_replay.png", dpi=160)
    plt.close(fig)
    if not closed["executed"]:
        return
    ff_kinds = list(closed["results"])
    fig, axes = plt.subplots(2, len(ff_kinds), figsize=(14, 8), constrained_layout=True, squeeze=False)
    for col, ff in enumerate(ff_kinds):
        methods = closed["results"][ff]
        scenarios = list(methods["FF"]["test"])
        x = np.arange(len(scenarios))
        for j, (method, values) in enumerate(methods.items()):
            for row, metric in enumerate(["position_rmse_m", "rotation_rmse_rad"]):
                values_plot = [values["test"][s].get(metric, np.nan) for s in scenarios]
                axes[row, col].bar(x + (j - 2) * 0.15, values_plot, width=0.14, label=method)
                axes[row, col].set_xticks(x, scenarios, rotation=25, ha="right")
                axes[row, col].set_title(f"{ff} feedforward: LOCAL MODEL test")
                axes[row, col].set_ylabel(metric)
                axes[row, col].grid(axis="y", alpha=0.2)
        axes[0, col].legend(ncols=3)
    fig.savefig(figures / "tracking_comparison.png", dpi=160)
    plt.close(fig)
    traces = np.load(out / "closed_loop_traces.npz")
    pid = load_result(out, "pid_config.json")
    for ff in ff_kinds:
        selected = pid["selected"][ff]
        usable = [m for m in selected if selected[m]["validation_objective"] is not None]
        method = min(usable, key=lambda m: selected[m]["validation_objective"]) if usable else "FF"
        prefix = f"{ff}__{method}__initial_offset__"
        if prefix + "error" not in traces:
            continue
        t = traces[prefix + "times"]
        state, reference = traces[prefix + "state"], traces[prefix + "reference"]
        origin = reference[0]
        actual = pose_error(state, origin)
        target = pose_error(reference, origin)
        fig, axes = plt.subplots(6, 4, figsize=(18, 16), constrained_layout=True)
        for axis in range(6):
            axes[axis, 0].plot(t, actual[:, axis], label="actual model")
            axes[axis, 0].plot(t, target[:, axis], ls="--", label="oracle target")
            for column, key in [(1, "error"), (2, "integral"), (3, "command")]:
                axes[axis, column].plot(t, traces[prefix + key][:, axis])
            for column, title in enumerate(
                ["pose relative to first reference", "tracking error", "PID integral", "dimensionless command"]
            ):
                axes[axis, column].set(xlabel="physical time (s)", title=f"{title}: axis {axis}")
                axes[axis, column].grid(alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(f"LOCAL MODEL test, {ff}/{method} selected on validation, initial offset", fontsize=14)
        fig.savefig(figures / f"{ff}_initial_offset.png", dpi=130)
        plt.close(fig)
