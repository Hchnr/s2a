"""Independently verify follow-up source pairs, model contracts and continuous physics."""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.async_reference import TimedReference
from s2a.geometry import inverse, pose_error
from s2a.prediction_audit import audit_action_response
from s2a.trajectory_audit import trajectory_diagnostics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_followup_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, out = load_config(args.config)
    require(args.seed == c["seed"], "seed differs from frozen config")
    policy = json.loads(Path(c["policy_config"]).read_text())
    dt, horizon = 1 / policy["control_fps"], policy["execution_horizon"]
    fixture = json.loads(Path(policy["fixture"]).read_text())
    h = np.asarray(json.loads(Path(policy["tool_contract"]).read_text())["source_from_tcp"])
    bridge = json.loads(Path(c["published_bridge"]).read_text())["trajectories"]
    matched = json.loads((out / "alignment/matched_targets.json").read_text())
    revision = json.loads((out / "alignment/revision.json").read_text())
    camera = json.loads((out / "camera_parity/summary.json").read_text())
    require(camera["passed"] and len(camera["cases"]) == 30, "camera parity incomplete")
    require(all(r["max_abs"] == 0 for r in camera["cases"]), "camera parity changed")
    sources, commits, artifacts = {}, {}, {}

    def check_inputs(manifest, key="inputs_sha256"):
        for name, digest in manifest[key].items():
            if name not in sources:
                sources[name] = sha256(name)
            require(sources[name] == digest, f"input changed: {name}")

    def check_code(manifest, paths):
        code = manifest["code"]
        for name in paths:
            key = code["commit"], name
            if key not in commits:
                data = subprocess.check_output(["git", "show", f"{key[0]}:{name}"])
                commits[key] = hashlib.sha256(data).hexdigest()
            require(commits[key] == code["files_sha256"][name], f"executed source differs from recorded commit: {key}")

    check_inputs(matched)
    check_inputs(revision)
    check_inputs(camera, "input_sha256")
    check_code(matched, ["scripts/audit_libero_alignment.py"])
    check_code(revision, ["scripts/audit_libero_release_revision.py"])
    check_code(
        camera, ["scripts/verify_libero_camera_capture.py", "src/s2a/cameras.py", "src/s2a/libero_simulation.py"]
    )
    require(
        all(v["equal"] for v in revision["checkpoint_vs_current_normalization"].values()),
        "normalization revision differs",
    )
    require(
        all(v["equal"] for row in revision["cases"] for v in row["comparisons"].values()), "original actions changed"
    )
    config_path = Path(policy["checkpoint_run"]) / "config.yaml"
    spec = yaml.safe_load(config_path.read_text())["model"]["config"]["action128_spec"]
    spec = spec["transforms"][spec["registry"]["domains"].index(policy["action_domain"])]
    model_contracts = []

    def check_response(path):
        response = json.loads(path.read_text())
        require(response["status"] == "success", f"model response failed: {path}")
        contract = audit_action_response(
            response["outputs"][0]["content"], spec, domain=policy["action_domain"], fps=policy["control_fps"]
        )
        require(contract["denormalization_max_abs"] < 1e-6, f"normalization failed: {path}")
        model_contracts.append(contract)
        return np.asarray(response["outputs"][0]["content"]["action"])

    training = json.loads((out / "training_image_queries/summary.json").read_text())
    require(
        len(training["cases"]) == len(c["trials"]) * len(c["training_horizons"]),
        "training-image query matrix incomplete",
    )
    check_inputs(training)
    for case in training["cases"]:
        dest = out / "training_image_queries" / f"trial{case['trial']}_h{case['horizon']}"
        manifest = json.loads((dest / "manifest.json").read_text())
        check_code(
            manifest,
            [
                "scripts/probe_libero_training_images.py",
                "src/s2a/libero_simulation.py",
                "src/s2a/trajectory_audit.py",
                "src/s2a/geometry.py",
            ],
        )
        predicted = check_response(dest / "response_0000.json")
        record = next(r for r in matched["cases"] if r["trial"] == case["trial"])
        rows = record["rows"][c["training_frame_index"] : c["training_frame_index"] + case["horizon"]]
        recomputed = trajectory_diagnostics(
            predicted, np.asarray([r["action"] for r in rows]), np.asarray([r["action_dim_mask"] for r in rows])
        )
        require(recomputed.keys() == case["diagnostics"].keys(), "prediction diagnostic fields changed")
        for key, value in recomputed.items():
            np.testing.assert_allclose(value, case["diagnostics"][key], atol=1e-12, rtol=1e-12)
        require(case["canvas_training_parity_max_abs"] == 0, "training canvas parity failed")
    results, live_pairing = [], []
    for stage in ("synthetic", "live"):
        matrix = json.loads((out / f"matrix_{stage}.json").read_text())
        expected = 30 if stage == "synthetic" else 5
        require(matrix["complete"] and len(matrix["cases"]) == expected, f"{stage} matrix incomplete")
        identities = set()
        for item in matrix["cases"]:
            require(item["exit_code"] == 0, "simulation process failed")
            dest = Path(item["directory"])
            manifest = json.loads((dest / "manifest.json").read_text())
            check_inputs(manifest)
            core = [
                "src/s2a/libero_async.py",
                "src/s2a/async_reference.py",
                "src/s2a/libero_simulation.py",
                "src/s2a/geometry.py",
                "src/s2a/gripper.py",
                "src/s2a/osc.py",
                "scripts/run_libero_async.py",
            ]
            if stage == "live":
                core.append("src/s2a/cameras.py")
            check_code(manifest, core)
            result = json.loads((dest / "result.json").read_text())
            key = result["arm"], result["trial"], result["delay_s"]
            require(key not in identities and result["error"] is None, "duplicate or failed case")
            identities.add(key)
            rows = [json.loads(line) for line in (dest / "steps.jsonl").read_text().splitlines()]
            events = json.loads((dest / "events.json").read_text())
            reference_length = len(bridge[fixture["episodes"][result["trial"]]["demo"]]["source_poses"])
            budget = (
                c["async_steps"]
                if stage == "live"
                else max(c["synthetic_min_steps"], reference_length + policy["oracle_extra_hold_steps"])
            )
            require(len(rows) == result["steps"] == result["expected_steps"] == budget, "physics budget incomplete")
            require(len(events) == result["requests"], "unaccounted inference requests")
            event_map = {e["sequence"]: e for e in events}
            require(len(event_map) == len(events), "duplicate response sequence")
            for event in events:
                if "admitted_at" in event:
                    age = event["admitted_at"] - event["observed_at"]
                    require(abs(age - event["age_s"]) < 1e-8 and age >= 0, "invalid inference age")
                    if event["verdict"] == "accepted":
                        require(age < c["async_max_age_s"], "expired reference accepted")
                    if event["verdict"] == "expired":
                        require(age >= c["async_max_age_s"], "fresh reference incorrectly expired")
            live_references, inflight_physics_steps = {}, []
            if stage == "live":
                require(len(list(dest.glob("response_*.json"))) == len(events), "model responses missing")
                for event in events:
                    number = event["sequence"]
                    predicted = check_response(dest / f"response_{number:04d}.json")
                    request = json.loads((dest / f"request_{number:04d}.json").read_text())
                    require(
                        request["model_mode"] == "wam" and request["seed"] == c["seed"], "WAM request contract differs"
                    )
                    pixels = np.asarray(Image.open(dest / f"input_{number:04d}.png").convert("RGB"))
                    require(
                        hashlib.sha256(pixels.tobytes()).hexdigest() == event["observation_sha256"],
                        "camera hash changed",
                    )
                    anchor = np.asarray(rows[event["capture_step"]]["measured_source"]) @ h
                    live_references[number] = TimedReference.from_actions(
                        number, event["observed_at"], dt, anchor, predicted[:, :10], horizon=horizon
                    )
                    if "admitted_at" in event:
                        physical_steps = sum(
                            r["clock_s"] >= event["observed_at"] and r["clock_s"] < event["admitted_at"] for r in rows
                        )
                        require(physical_steps > 0, "physics paused throughout inference")
                        inflight_physics_steps.append(physical_steps)
                previous = Path(policy["output_dir"]) / "policy/baseline" / f"trial{result['trial']}"
                old_image = np.asarray(Image.open(previous / "input_0000.png").convert("RGB"))
                image = np.asarray(Image.open(dest / "input_0000.png").convert("RGB"))
                old_action = np.asarray(
                    json.loads((previous / "response_0000.json").read_text())["outputs"][0]["content"]["action"]
                )
                action = np.asarray(
                    json.loads((dest / "response_0000.json").read_text())["outputs"][0]["content"]["action"]
                )
                live_pairing.append(
                    dict(
                        trial=result["trial"],
                        initial_canvas_max_abs=int(np.max(abs(image.astype(int) - old_image.astype(int)))),
                        first_prediction_max_abs=float(np.max(abs(action - old_action))),
                    )
                )
            previous_command = np.zeros(6)
            errors, tracking_errors = [], []
            for index, row in enumerate(rows):
                require(row["step"] == index and abs(row["actual_dt"] - dt) < 1e-8, "invalid physics time")
                require(abs(row["simulation_elapsed_s"] - index * dt) < 1e-7, "physics clock discontinuity")
                if index:
                    np.testing.assert_allclose(
                        row["measured_source"], rows[index - 1]["after_source"], atol=1e-12, rtol=0
                    )
                    require(row["clock_s"] >= rows[index - 1]["clock_s"], "decision clock reversed")
                command = np.asarray(row["command"])
                require(np.isfinite(command).all() and np.max(abs(command)) <= 1 + 1e-9, "command bounds failed")
                require(
                    np.max(abs(command[:6] - previous_command)) <= policy["max_command_rate"] * dt + 1e-9,
                    "command rate failed",
                )
                previous_command = command[:6]
                error = pose_error(np.asarray(row["desired_source"]), np.asarray(row["after_source"]))
                np.testing.assert_allclose(row["after_source"], row["physical_after_source"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(error, row["error"], atol=1e-12, rtol=0)
                errors.append(error)
                if row["target_status"] == "tracking":
                    event = event_map[row["reference_sequence"]]
                    require(
                        event["verdict"] == "accepted" and event["admitted_at"] <= row["clock_s"] + 1e-9,
                        "target used before arrival",
                    )
                    require(0 <= row["clock_s"] - event["observed_at"] < c["async_max_age_s"], "stale target executed")
                    expected_index = int(np.floor((row["clock_s"] - event["observed_at"] + 1e-9) / dt))
                    require(row["reference_index"] == expected_index < horizon, "elapsed reference endpoints replayed")
                    if stage == "synthetic":
                        reference = bridge[fixture["episodes"][result["trial"]]["demo"]]
                        endpoint = min(event["capture_step"] + expected_index, len(reference["source_poses"]) - 1)
                        desired = reference["source_poses"][endpoint]
                    else:
                        desired = live_references[event["sequence"]].poses[expected_index] @ inverse(h)
                    np.testing.assert_allclose(row["desired_source"], desired, atol=1e-12, rtol=0)
                    tracking_errors.append(error)
                else:
                    require(command[6] == 0, "hold issued a new gripper intent")
                    np.testing.assert_array_equal(row["gripper_before"], row["gripper_after"])
                    held = rows[index - 1]["desired_source"] if index else row["measured_source"]
                    np.testing.assert_allclose(row["desired_source"], held, atol=1e-12, rtol=0)
            errors = np.asarray(errors)
            np.testing.assert_allclose(
                np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))), result["target_position_rmse_m"]
            )
            require(
                any(r["success"] for r in rows) == result["success"]
                and rows[-1]["success"] == result["terminal_success"],
                "task result differs from trace",
            )
            require(len(tracking_errors) == result["tracking_steps"], "tracking count mismatch")
            result["tracking_position_rmse_m"] = (
                float(np.sqrt(np.mean(np.sum(np.asarray(tracking_errors)[:, :3] ** 2, axis=1))))
                if tracking_errors
                else None
            )
            result["directory"] = str(dest)
            result["inflight_physics_steps_min"] = min(inflight_physics_steps) if inflight_physics_steps else None
            results.append(result)
        expected_keys = (
            {
                (arm, trial, delay)
                for arm in c["synthetic_arms"]
                for trial in c["trials"]
                for delay in c["synthetic_delays_s"]
            }
            if stage == "synthetic"
            else {("affine", trial, 0) for trial in c["trials"]}
        )
        require(identities == expected_keys, "frozen case identities changed")
    groups = defaultdict(list)
    for row in results:
        groups[(row["stage"], row["arm"], row["delay_s"])].append(row)
    grouped = []
    for (stage, arm, delay), rows in sorted(groups.items()):
        tracked = [r["tracking_position_rmse_m"] for r in rows if r["tracking_position_rmse_m"] is not None]
        grouped.append(
            dict(
                stage=stage,
                arm=arm,
                delay_s=delay,
                episodes=len(rows),
                successes=sum(r["success"] for r in rows),
                terminal_successes=sum(r["terminal_success"] for r in rows),
                tracking_steps=sum(r["tracking_steps"] for r in rows),
                accepted=sum(r["accepted"] for r in rows),
                expired=sum(r["expired"] for r in rows),
                tracking_position_rmse_mm=float(np.mean(tracked) * 1000) if tracked else None,
                mean_physical_wall_ratio=float(
                    np.mean([r["physical_elapsed_s"] / r["physical_stop_wall_s"] for r in rows])
                ),
                max_deadline_lateness_s=max(r["max_deadline_lateness_s"] for r in rows),
            )
        )
    training_groups = []
    for horizon in c["training_horizons"]:
        rows = [r for r in training["cases"] if r["horizon"] == horizon]
        training_groups.append(
            dict(
                horizon=horizon,
                episodes=len(rows),
                request_elapsed_mean_s=float(np.mean([r["request_elapsed_s"] for r in rows])),
                diagnostics={
                    key: float(np.mean([r["diagnostics"][key] for r in rows])) for key in rows[0]["diagnostics"]
                },
            )
        )
    archive_index = {}
    for name in ("startup_wrong_interpreter", "timing_every_step", "cached_feedback", "camera_cached_comparison"):
        root = out / name
        if root.exists():
            files = {
                str(path): sha256(path)
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix in {".json", ".jsonl", ".png", ".log"}
                and not {"matplotlib", "numba_cache", "libero_config"}.intersection(path.parts)
            }
            archived_results = [json.loads(path.read_text()) for path in root.rglob("result.json")]
            archive_index[name] = dict(
                completed_physics_cases=len(archived_results),
                completed_physics_steps=sum(r["steps"] for r in archived_results),
                primary_files_sha256=files,
                request_path_remap=(
                    {str(out / "async/live_affine"): str(root / "live_affine")}
                    if name == "timing_every_step"
                    else {str(out / "async"): str(root / "async")}
                ),
                note="Submitted request paths remain historical; resolve within this archive using the prefix map.",
            )
    write_json(out / "archive_index.json", archive_index)
    for pattern in (
        "alignment/*",
        "training_image_queries/**/*.json",
        "training_image_queries/**/*.png",
        "camera_parity/summary.json",
        "async/*/trial*/*.json",
        "async/*/trial*/*.jsonl",
        "async/*/trial*/*.png",
        "matrix_*.json",
        "figures/*",
        "checkpoint_fingerprints.json",
        "wheel_smoke.json",
        "owned_service_cleanup.json",
        "archive_index.json",
    ):
        for path in out.glob(pattern):
            if path.is_file():
                artifacts[str(path)] = sha256(path)
    summary = dict(
        seed=c["seed"],
        complete=True,
        integrity_passed=True,
        verification_code=code_fingerprint(),
        formal_physics_cases=len(results),
        formal_physics_steps=sum(r["steps"] for r in results),
        groups=grouped,
        training_image_queries=training_groups,
        source_revision=revision,
        live_initial_pairing=live_pairing,
        model_contract_responses=len(model_contracts),
        denormalization_max_abs=max(r["denormalization_max_abs"] for r in model_contracts),
        checked_source_files=len(sources),
        checked_executed_revision_files=len(commits),
        primary_artifacts_sha256=artifacts,
        cases=results,
        hardware_executed=False,
        historical_training_bytes_verified=False,
        task_success_qualification=False,
        archived_physics={
            name: {k: value[k] for k in ("completed_physics_cases", "completed_physics_steps")}
            for name, value in archive_index.items()
        },
    )
    write_json(out / "summary.json", summary)
    report = Path("reports/libero_followup_20261019.json")
    write_json(report, summary)
    lines = [
        "# LIBERO 数据对齐、WAM目标与异步执行验证 — 2026-10-19",
        "",
        "本报告承接481项历史冻结实验；新增35项正式物理仿真。完整性通过与任务成功分开判断。",
        "",
        "## Catalog与当前发布",
        "",
        "配置路径匹配，但历史catalog的9项metadata只有4项SHA保持相同。当前eef_state_revision合同保留了catalog的原info指纹；5条完整action、dim/control mask及valid与保留original列逐项一致，receipt通过，checkpoint与当前归一化的scale/bias/mask/clip完全相同。历史训练输入字节及消费记录未恢复。",
        "",
        "当前发布两路视频按PTS解码，S2A canvas与训练纯resize函数逐像素一致；按需相机与强制刷新当前observable的30个比较逐像素一致，缓存图差异另存。NVIDIA/R1原始物理隐藏状态仍未恢复，不把发布frame0与HDF5仿真初态视为同一时刻。",
        "",
        "## 当前发布图像上的WAM目标（离线诊断）",
        "",
        "每种时域5条，frame0当前发布图像与同源R2标签配对。不是任务成功率，也不证明历史训练消费。",
        "",
        "| 预测步数 | 预测旋转均值rad/步 | 标签旋转均值rad/步 | 旋转RMSE rad | raw opening越界 | 积分末端位置误差mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in training_groups:
        d = row["diagnostics"]
        lines.append(
            f"| {row['horizon']} | {d['predicted_rotation_mean_rad']:.6f} | {d['reference_rotation_mean_rad']:.6f} | {d['rotation_rmse_rad']:.6f} | {d['opening_outside_fraction']:.3%} | {1000 * d['integrated_final_position_error_m']:.3f} |"
        )
    lines += [
        "",
        "40步对应catalog的2秒窗口，未消除目标偏差；不据此把40步替换为在线默认。",
        "",
        "## 持续物理执行",
        "",
        "synthetic为已生成发布oracle参考的到达延迟，保持原时间轴并跳过过去目标；它与上一轮的actuator-input delay不同。live为真实WAM HTTP，0.4秒有效期不放宽。所有正式case完整执行预算，不因首次success提前停止。",
        "",
        "| 来源 | arm | 到达延迟s | 首次成功 | 终态成功 | 有效目标tick | 接受/过期参考 | 跟踪RMSE mm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grouped:
        rmse = (
            f"{row['tracking_position_rmse_mm']:.3f}" if row["tracking_position_rmse_mm"] is not None else "无有效目标"
        )
        lines.append(
            f"| {row['stage']} | {row['arm']} | {row['delay_s']} | {row['successes']}/{row['episodes']} | {row['terminal_successes']}/{row['episodes']} | {row['tracking_steps']} | {row['accepted']}/{row['expired']} | {rmse} |"
        )
    lines += [
        "",
        "无有效目标时，固定末端目标和夹爪内部命令保持；其小位置误差不能记为WAM跟踪成功。推理期间真实物理连续推进，逐步检查0.05秒物理时钟、状态连续性、到达前禁用、观测年龄、原world锚点、实际发送值限幅与变化率。",
        "",
        "按需采集减少CPU每步渲染开销，墙钟期限超时仍完整保留；不宣称硬实时。首轮逐步渲染的完整1000步诊断耗费225.293秒墙钟/50秒物理时间，独立归档；端点刷新前的35项也完整归档，均未混入正式35项。35次错误解释器启动均在物理执行前失败，完整归档。",
        "",
        "## 判定",
        "",
        "快速、有效的oracle预测可经本地控制执行；真实WAM响应是否满足有效期、目标是否正确仍是独立条件。当前结果不支持默认部署补偿或真机。学习逆、增益调度、世界模型重训、未见任务及真机仍未执行。",
        "",
        f"独立核验：{len(results)}项正式物理仿真、{summary['formal_physics_steps']}步、{len(model_contracts)}个模型response、{len(artifacts)}个主要产物指纹通过。软件测试与打包结果见实施记录。",
        "",
    ]
    report.with_suffix(".md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "complete",
                    "formal_physics_cases",
                    "formal_physics_steps",
                    "groups",
                    "live_initial_pairing",
                    "model_contract_responses",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
