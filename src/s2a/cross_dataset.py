"""Frozen-cohort geometry checks across Panda datasets, without hardware IO.

The same SE(3) decoder is reused; native frames and gripper semantics come
from each cohort's own contract. Log geometry is not an inverse-dynamics test.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.data import require
from s2a.geometry import (
    error_summary,
    euler_pose_to_transform,
    finite,
    integrate_local,
    inverse,
    pose9_to_transform,
    pose_error,
    transform,
    validate_transform,
)

NATIVE_COLUMNS = {
    "robomind_v1": ["source.puppet_end_effector", "source.master_joint_position"],
    "interndata_m1": [
        "states.effector.position",
        "states.effector.orientation",
        "actions.effector.position",
        "actions.effector.orientation",
        "actions.gripper.position",
    ],
    "interndata_a1": ["states.ee_to_armbase_pose", "actions.ee_to_armbase_pose", "actions.gripper.openness"],
}
R2_COLUMNS = [
    "episode_index",
    "frame_index",
    "timestamp",
    "action",
    "action_valid",
    "action_dim_mask",
    "action_control_mask",
    "observation.state",
    "observation.state_mask",
]


def quaternion_pose(value):
    """Native A1 xyz+wxyz, independently implemented from the documented order."""
    x = finite(value)
    require(x.ndim == 2 and x.shape[1] == 7, "expected Nx7 xyz+wxyz pose")
    require((np.linalg.norm(x[:, 3:], axis=1) > 1e-8).all(), "degenerate native quaternion")
    return transform(x[:, :3], Rotation.from_quat(x[:, [4, 5, 6, 3]]).as_matrix())


def native_to_tcp(adapter, contract):
    if adapter == "robomind_v1":
        return validate_transform(contract["link8_from_canonical_tcp"])
    if adapter == "interndata_m1":
        # Documented R1 effector -> link8 convention, not fitted from test poses.
        offset = transform(np.zeros(3), Rotation.from_euler("z", 45, degrees=True).as_matrix())
        return offset @ validate_transform(contract["link8_from_canonical_tcp"])
    if adapter == "interndata_a1":
        return validate_transform(contract["source_from_tcp"])
    raise ValueError(f"unsupported native convention: {adapter}")


def native_poses(adapter, arrays):
    if adapter == "robomind_v1":
        return euler_pose_to_transform(arrays["source.puppet_end_effector"]), None
    if adapter == "interndata_m1":
        observed = euler_pose_to_transform(
            np.c_[arrays["states.effector.position"], arrays["states.effector.orientation"]]
        )
        commanded = euler_pose_to_transform(
            np.c_[arrays["actions.effector.position"], arrays["actions.effector.orientation"]]
        )
        return observed, commanded
    if adapter == "interndata_a1":
        return quaternion_pose(arrays["states.ee_to_armbase_pose"]), quaternion_pose(
            arrays["actions.ee_to_armbase_pose"]
        )
    raise ValueError(f"unsupported native convention: {adapter}")


def align_native(native, binding, frames):
    """Join by recorded episode-local source row support, never media time."""
    lo, hi = binding["row_range"]
    frames = np.asarray(frames)
    ids = np.asarray(native["frame_index"])
    require(len(np.unique(ids)) == len(ids), "duplicate native frame IDs")
    require(len(np.unique(frames)) == len(frames), "duplicate published frame IDs")
    require(np.array_equal(frames, np.arange(hi - lo)), "published frame support differs from source row interval")
    lookup = {int(frame): i for i, frame in enumerate(ids)}
    required = frames + lo
    require(all(int(frame) in lookup for frame in required), "native row support missing")
    indices = [lookup[int(frame)] for frame in required]
    return {key: value[indices] for key, value in native.items()}


def geometry_checks(rows, observed_native, h_native_tcp):
    frames = rows["frame_index"]
    state_valid = rows["observation.state_mask"][:, :9].all(axis=1)
    require(state_valid.any(), "no valid observed poses")
    tcp = np.broadcast_to(np.eye(4), (len(frames), 4, 4)).copy()
    tcp[state_valid] = pose9_to_transform(rows["observation.state"][state_valid, :9])
    valid = rows["action_valid"].astype(bool) & rows["action_dim_mask"][:, :9].all(axis=1) & state_valid
    valid[-1] = False
    valid[:-1] &= state_valid[1:] & (np.diff(frames) == 1)
    indices = np.flatnonzero(valid)
    require(len(indices) > 0, "no valid adjacent pose transitions")
    predicted_tcp = tcp[indices] @ pose9_to_transform(rows["action"][indices, :9])
    # Never integrate through an invalid action or missing observation.
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    integrated_error = []
    for run in runs:
        integration = integrate_local(tcp[run[0]], rows["action"][run, :9])
        support = np.r_[run, run[-1] + 1]
        integrated_error.append(pose_error(integration, tcp[support]))
    errors = np.concatenate(integrated_error)
    cumulative = {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        "position_max_m": float(np.linalg.norm(errors[:, :3], axis=1).max()),
        "rotation_rmse_rad": float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        "rotation_max_rad": float(np.linalg.norm(errors[:, 3:], axis=1).max()),
    }
    h_inv = inverse(h_native_tcp)
    recovered = tcp @ h_inv
    candidate = predicted_tcp @ h_inv
    metrics = {
        "observations": len(frames),
        "valid_observed_poses": int(state_valid.sum()),
        "valid_transitions": len(indices),
        "excluded_transition_rows": len(frames) - len(indices),
        "continuous_segments": len(runs),
        "single_step": error_summary(predicted_tcp, tcp[indices + 1]),
        "integrated": cumulative,
        "native_observation": error_summary(recovered[state_valid], observed_native[state_valid]),
        "candidate_vs_next_native_observation": error_summary(candidate, observed_native[indices + 1]),
        "hold_current_pose": error_summary(observed_native[indices], observed_native[indices + 1]),
    }
    arrays = {
        "tcp_pose": tcp,
        "native_observed_pose": observed_native,
        "recovered_native_pose": recovered,
        "transition_index": indices,
        "candidate_native_pose": candidate,
        "integrated_error": errors,
        "single_step_error": pose_error(predicted_tcp, tcp[indices + 1]),
    }
    return metrics, arrays


class Inputs:
    def __init__(self):
        self.fingerprints = {}
        self.documents = {}

    def add(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.fingerprints:
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            require(
                (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "input changed while hashing",
            )
            self.fingerprints[key] = {"sha256": digest, "size_bytes": after.st_size}
        return path

    def document(self, path):
        path = self.add(path)
        if str(path) not in self.documents:
            self.documents[str(path)] = json.loads(path.read_text())
        return self.documents[str(path)]


def read_arrays(path, columns, episode):
    table = pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode)])
    require(len(table) > 0, f"episode {episode} missing from {path}")
    return {key: np.asarray(table[key].to_pylist()) for key in columns}


def evaluate_episode(cohort, selection, inputs, out, tolerances):
    root, adapter = Path(cohort["root"]), cohort["native_adapter"]
    ep = selection["episode_index"]
    metadata = inputs.add(root / selection["metadata_path"])
    columns = [key for key in selection if key != "metadata_path"]
    records = pq.read_table(metadata, columns=columns, filters=[("episode_index", "=", ep)]).to_pylist()
    require(len(records) == 1, "episode metadata not unique")
    actual = records[0]
    require(
        actual == {key: value for key, value in selection.items() if key != "metadata_path"}, "frozen metadata changed"
    )
    require(
        actual["training_eligible"] and actual["embodiment_name"] == "franka_panda__parallel_gripper", "cohort mismatch"
    )
    reference = json.loads(actual["state_reference"])
    binding = reference["native_binding"]
    contract = inputs.document(root / cohort["contract"])
    require(reference["contract"] == cohort["contract"] == actual["eef_tool_contract"], "tool contract mismatch")
    receipt_path = root / "_receipts" / Path(actual["data_path"]).with_suffix(".json")
    receipt = inputs.document(receipt_path)
    matches = [e for e in receipt["episodes"] if e["episode_index"] == ep]
    require(len(matches) == 1 and matches[0]["pose_source"] == reference, "receipt/source binding mismatch")
    inputs.add(root / "meta/eef_state_contract.json")
    path = inputs.add(root / actual["data_path"])
    names = pq.ParquetFile(path).schema_arrow.names
    extras = [key for key in ["source_physical_timestamp", "transition_dt", "physical_time_valid"] if key in names]
    rows = read_arrays(path, R2_COLUMNS + extras, ep)
    order = np.argsort(rows["frame_index"])
    rows = {key: value[order] for key, value in rows.items()}
    require(len(rows["frame_index"]) == actual["length"], "published row count mismatch")
    native_path = inputs.add(binding["path"])
    native = read_arrays(native_path, ["frame_index"] + NATIVE_COLUMNS[adapter], binding["episode_index"])
    native = align_native(native, binding, rows["frame_index"])
    observed, commanded = native_poses(adapter, native)
    h = native_to_tcp(adapter, contract)
    metrics, arrays = geometry_checks(rows, observed, h)
    indices = arrays["transition_index"]
    metrics["saved_action_pose_comparison"] = (
        error_summary(arrays["candidate_native_pose"], commanded[indices]) if commanded is not None else None
    )
    metrics["command_semantics"] = (
        "saved Cartesian action fields; actual controller API unverified"
        if commanded is not None
        else "only master joint targets; no verified Cartesian command label"
    )
    gripper = json.loads(actual["eef_gripper_source"])
    gripper_valid = rows["action_dim_mask"][:, 9].astype(bool)
    metrics["gripper"] = {
        "source_semantics": gripper,
        "valid_action_rows": int(gripper_valid.sum()),
        "valid_state_rows": int(rows["observation.state_mask"][:, 9].sum()),
        "controller_encoding_executed": False,
    }
    if adapter == "interndata_a1" and gripper_valid.any():
        metrics["gripper"]["native_command_max_abs_error"] = float(
            np.max(np.abs(rows["action"][gripper_valid, 9] - native["actions.gripper.openness"][gripper_valid]))
        )
    metrics["physical_clock_verified"] = False
    metrics["time_scope"] = "media/source schedule only; no physical velocity, dynamics fit or PID transfer"
    checks = [
        metrics[key]
        for key in ["single_step", "integrated", "native_observation", "candidate_vs_next_native_observation"]
    ]
    metrics["geometry_passed"] = all(
        m["position_max_m"] <= tolerances["position_m"] and m["rotation_max_rad"] <= tolerances["rotation_rad"]
        for m in checks
    )
    arrays.update({key.replace(".", "_"): value for key, value in rows.items()})
    arrays["H_native_tcp"] = h
    if commanded is not None:
        arrays["native_saved_action_pose"] = commanded
    np.savez_compressed(out / f"episode_{ep}.npz", **arrays)
    return {
        "episode_index": ep,
        "source_episode_id": actual["source_episode_id"],
        "evidence_level": cohort["evidence_level"],
        **metrics,
    }


def run_cross_dataset(config, out):
    inputs = Inputs()
    inputs.add(config["publisher_source"])
    inputs.add(config["native_quaternion_source"])
    fingerprint = code_fingerprint()
    results = []
    for cohort in config["cohorts"]:
        target = out / cohort["id"]
        target.mkdir(exist_ok=True)
        episodes = []
        for selection in cohort["episodes"]:
            try:
                metrics = evaluate_episode(cohort, selection, inputs, target, config["geometry_tolerances"])
                metrics["executed"] = True
            except (ValueError, KeyError, FileNotFoundError) as error:
                metrics = {
                    "episode_index": selection["episode_index"],
                    "executed": False,
                    "geometry_passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            episodes.append(metrics)
            print(
                f"{cohort['id']}/{selection['episode_index']}: executed={metrics['executed']} geometry={metrics['geometry_passed']}",
                flush=True,
            )
        completed = [entry for entry in episodes if entry["executed"]]
        summary = {
            "id": cohort["id"],
            "dataset": cohort["dataset"],
            "evidence_level": cohort["evidence_level"],
            "selected_episodes": len(episodes),
            "executed_episodes": len(completed),
            "geometry_passed_episodes": sum(e["geometry_passed"] for e in episodes),
            "valid_transitions": sum(e["valid_transitions"] for e in completed),
        }
        if completed:
            for metric in ["single_step", "integrated", "native_observation", "candidate_vs_next_native_observation"]:
                summary[metric] = {
                    key: max(e[metric][key] for e in completed) for key in ["position_max_m", "rotation_max_rad"]
                }
        results.append({**summary, "episodes": episodes})
    result = {
        "code": fingerprint,
        "seed": config["seed"],
        "config": config,
        "input_fingerprints": inputs.fingerprints,
        "cohorts": results,
        "evidence_level": "cross_dataset_log_geometry_and_saved_action_fields_only",
        "hardware_execution": False,
        "cross_dataset_controller_transfer": False,
        "limitations": [
            "Same Panda release label does not identify serial, exact gripper, controller or historical TCP calibration",
            "Native observations and published states share provenance; this is a representation check, not independent dynamics",
            "No verified physical dt/controller configuration: no transferred DROID gains, velocity scaling or response-model rollout",
            "Simulation log is reported separately and is not independent execution of candidate commands",
        ],
    }
    write_json(out / "cross_dataset_metrics.json", result)
    render_cross_report(result, out)
    write_json(
        out / "artifact_hashes.json",
        {
            str(p.relative_to(out)): sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    return result


def render_cross_report(result, out):
    lines = [
        "# Franka Panda 跨数据集解析验证",
        "",
        "没有执行真机，也没有在独立模拟器执行候选命令。本报告评估只读日志的解析几何与已保存 action 字段。",
        "",
        f"代码 `{result['code']['commit']}`；dirty=`{result['code']['dirty']}`；seed={result['seed']}。抽样名单在数值评估前冻结，不训练或调参。",
        "",
        "| Cohort | 数据证据 | 执行/选择 episode | 几何通过 | 转移数 | 单步最大 m/rad | 累计最大 m/rad | 原始观测最大 m/rad |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for cohort in result["cohorts"]:

        def pair(key):
            value = cohort.get(key)
            return f"{value['position_max_m']:.3g} / {value['rotation_max_rad']:.3g}" if value else "未执行"

        lines.append(
            f"| {cohort['id']} | {cohort['evidence_level']} | {cohort['executed_episodes']}/{cohort['selected_episodes']} | {cohort['geometry_passed_episodes']} | {cohort['valid_transitions']} | {pair('single_step')} | {pair('integrated')} | {pair('native_observation')} |"
        )
    lines += [
        "",
        "容差为 1e-5 m / 1e-4 rad，同时检查候选 native pose 与下一原始观测；完整逐 episode 结果、masks、媒体时间及 action 字段差异见 JSON/NPZ。",
        "",
        "## 这些结果能说明什么",
        "",
        "解析式为 T_BC_next=T_BC_now@Delta_T_C；候选原生位姿为 T_BC_next@inverse(H_native_tcp)。每个数据集单独固定 H，未使用 DROID 的 .075/.15 缩放。M1 另有已记录的 effector→link8 z+45°，A1 原生四元数为 wxyz。",
        "",
        "几何通过验证了发布表示、原始观测与坐标转换的一致性。目标来自日志未来观测，且发布状态本身源自这些原始观测，因此不能把低误差解释为控制器逆动力学正确、机器人执行成功或跨机器人控制泛化。",
        "",
        "## 未执行项及原因",
        "",
        "- 所有 cohort：没有真实硬件执行；序列号、历史控制配置及物理反馈时间未确认。",
        "- 不以媒体 fps 代替物理 dt，未跨数据集转移 DROID PID/B/C/归一化 velocity 接口。",
        "- M1 的夹爪缺少标定反馈，保留无效 mask，不生成夹爪命令。",
        "- RoboMIND-v1 的夹爪 target 为下一观测，未当作独立绝对命令；原生 master joint 不能直接当 Cartesian action。",
        "- A1 对应分支为仿真日志；与真实机器人日志分开，未据此声称真机或独立仿真通过。",
        "- RoboMIND2.0 涉及 Panda/FR3 和名义 FK，不作为已确认同型号比较。",
        "",
        "## 依据",
        "",
        "- 型号来源：[RoboMIND 官方项目](https://x-humanoid-robomind.github.io/)、[A1 官方项目](https://internrobotics.github.io/interndata-a1.github.io/)。M1 仅有本地发布标签，历史精确型号仍未验证。",
        "- 原始 native pose 约定：外部只读 eef_state_sources.py:89、:119、interndata_mobile_pose_probe.py:11，SHA256 存入 manifest；S2A 不导入发布器。",
        "- 运行：LD_LIBRARY_PATH='' .venv/bin/python scripts/run_cross_dataset.py --seed 42。",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
