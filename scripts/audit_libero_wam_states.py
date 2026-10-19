"""Compare integrated WAM references directly with published robot states.

Reads only frame/observation columns from R2. Expert actions and controller
commands are not ground truth. No inference, simulator, or training reader.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.geometry import integrate_local, pose9_to_transform


def errors(reference, predicted):
    return (
        np.linalg.norm(predicted[:, :3, 3] - reference[:, :3, 3], axis=1),
        Rotation.from_matrix(predicted[:, :3, :3] @ reference[:, :3, :3].swapaxes(-1, -2)).magnitude(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_wam_state_comparison_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen config")
    if any(output.iterdir()):
        raise ValueError("refusing to overwrite an existing audit")
    aligned = json.loads(Path(config["alignment"]).read_text())
    policy = json.loads(Path(config["policy_config"]).read_text())
    expected = str(Path(policy["checkpoint_run"]) / "checkpoints" / policy["checkpoint_iteration"] / "model")
    inputs = {p: sha256(p) for p in (args.config, config["alignment"], config["policy_config"])}
    data_paths = [
        p for p in aligned["inputs_sha256"] if Path(p).is_relative_to(Path(aligned["catalog_data"]["root"]) / "data")
    ]
    if len(data_paths) != 1:
        raise ValueError("this frozen audit expects exactly one source Parquet shard")
    path = data_paths[0]
    inputs[path] = sha256(path)
    if inputs[path] != aligned["inputs_sha256"][path]:
        raise ValueError("published states changed since camera pairing")
    episode_ids = [r["r2_episode"] for r in aligned["cases"] if r["trial"] in config["trials"]]
    published = pq.read_table(
        path,
        columns=["episode_index", "frame_index", "observation.state", "observation.state_mask"],
        filters=[("episode_index", "in", episode_ids), ("frame_index", "<=", max(config["horizons"]))],
    ).to_pylist()
    cases = []
    for horizon in config["horizons"]:
        for trial in config["trials"]:
            paired = next(r for r in aligned["cases"] if r["trial"] == trial)
            rows = sorted(
                [r for r in published if r["episode_index"] == paired["r2_episode"]], key=lambda r: r["frame_index"]
            )[: horizon + 1]
            if [r["frame_index"] for r in rows] != list(range(horizon + 1)):
                raise ValueError("missing or misaligned future state")
            states = np.asarray([r["observation.state"] for r in rows])
            masks = np.asarray([r["observation.state_mask"] for r in rows], dtype=bool)
            if not masks[:, :10].all() or not np.isfinite(states).all():
                raise ValueError("invalid observed state")
            if not np.array_equal(states, [r["observation.state"] for r in paired["rows"][: horizon + 1]]):
                raise ValueError("snapshot states differ from raw R2")
            directory = Path(config["query_root"]) / f"trial{trial}_h{horizon}"
            for name in ("manifest.json", "response_0000.json", "request_0000.json", "input_0000.png"):
                inputs[str(directory / name)] = sha256(directory / name)
            manifest = json.loads((directory / "manifest.json").read_text())
            content = json.loads((directory / "response_0000.json").read_text())["outputs"][0]["content"]
            if (
                manifest["server_info"]["checkpoint_path"] != expected
                or content["action128"]["checkpoint_path"] != expected
            ):
                raise ValueError("checkpoint differs from physical task evaluation")
            if manifest["config"]["seed"] != args.seed or manifest["config"]["control_fps"] != policy["control_fps"]:
                raise ValueError("prediction seed or clock differs")
            if manifest["server_info"]["ema_weights"]:
                raise ValueError("unexpected EMA weights")
            for key in ("sampling_steps", "guidance", "shift", "action_domain"):
                if manifest["config"][key] != policy[key]:
                    raise ValueError(f"inference setting differs: {key}")
            action = np.asarray(content["action"])
            if action.shape != (horizon, 128) or not np.isfinite(action).all() or np.any(action[:, 10:]):
                raise ValueError("invalid physical predicted reference")
            truth = pose9_to_transform(states[:, :9])
            predicted = integrate_local(truth[0], action[:, :9])[1:]
            # Independent R/t recurrence audits transform multiplication and anchor.
            r, t = truth[0, :3, :3].copy(), truth[0, :3, 3].copy()
            for index, delta in enumerate(pose9_to_transform(action[:, :9])):
                t = t + r @ delta[:3, 3]
                r = r @ delta[:3, :3]
                np.testing.assert_allclose(predicted[index, :3, 3], t, atol=1e-12, rtol=0)
                np.testing.assert_allclose(predicted[index, :3, :3], r, atol=1e-12, rtol=0)
            position, rotation = errors(truth[1:], predicted)
            hold_position, hold_rotation = errors(truth[1:], np.repeat(truth[:1], horizon, axis=0))
            points = []
            for step in sorted({1, 8, horizon}):
                points.append(
                    dict(
                        step=step,
                        time_s=step / policy["control_fps"],
                        position_mm=1000 * position[step - 1],
                        rotation_deg=np.rad2deg(rotation[step - 1]),
                        hold_position_mm=1000 * hold_position[step - 1],
                        hold_rotation_deg=np.rad2deg(hold_rotation[step - 1]),
                    )
                )
            cases.append(
                dict(
                    trial=trial,
                    r2_episode=paired["r2_episode"],
                    demo=paired["demo"],
                    horizon=horizon,
                    position_rmse_mm=1000 * np.sqrt(np.mean(position**2)),
                    rotation_rmse_deg=np.rad2deg(np.sqrt(np.mean(rotation**2))),
                    opening_mae_clipped=np.mean(abs(np.clip(action[:, 9], 0, 1) - states[1:, 9])),
                    opening_outside_fraction=np.mean((action[:, 9] < 0) | (action[:, 9] > 1)),
                    checkpoints=points,
                    position_errors_m=position,
                    rotation_errors_rad=rotation,
                )
            )
    groups = []
    for horizon in config["horizons"]:
        selected = [r for r in cases if r["horizon"] == horizon]
        groups.append(
            dict(
                horizon=horizon,
                cases=len(selected),
                position_rmse_mm=np.mean([r["position_rmse_mm"] for r in selected]),
                rotation_rmse_deg=np.mean([r["rotation_rmse_deg"] for r in selected]),
                opening_mae_clipped=np.mean([r["opening_mae_clipped"] for r in selected]),
                opening_outside_fraction=np.mean([r["opening_outside_fraction"] for r in selected]),
                checkpoints=[
                    {k: np.mean([r["checkpoints"][i][k] for r in selected]) for k in selected[0]["checkpoints"][i]}
                    for i in range(len(selected[0]["checkpoints"]))
                ],
            )
        )
    report = dict(
        seed=args.seed,
        code=code_fingerprint(),
        config=config,
        checkpoint=expected,
        ema_weights=False,
        inputs_sha256=inputs,
        ground_truth="R2 observation.state only; frame0 is anchor, future frames1..H are labels; no expert actions read",
        frame="world/canonical TCP, metres, column rot6d, local SE(3) right multiplication, SO(3) geodesic error",
        aggregation="arithmetic mean of five sequence metrics; checkpoints are pointwise errors, not trajectory RMSE",
        limits="Five known frame-zero contexts/seed42. One valid demonstration per context. No object/video-state metric or new task rollouts.",
        groups=groups,
        cases=cases,
    )
    write_json(output / "summary.json", report)
    lines = [
        "# WAM 与真实 state 的直接比较 — 2026-10-19",
        "",
        "只读当前 R2 Parquet 的 observation.state；没有读取专家 action/native command。复用此前同源 frame0 图像的10份保存预测，没有重新推理或新增仿真。",
        "frame0实测canonical TCP为积分锚点，物理delta按SE(3)右乘积分，对比frame1..H真实TCP；旋转为SO(3)测地角，开度对比下一帧实测归一化opening。",
        "所有数值为5条已知初始上下文的算术平均；点时刻误差与整段RMSE分开，不作未见场景结论。这里只评机器人TCP/开度，不评物体状态或生成视频。",
        "",
        f"Checkpoint（普通权重，非EMA）：`{expected}`。seed42，20Hz，35采样步，guidance1.5，shift5；与任务效果专项相同权重。",
        "",
        "| 请求长度 | 时刻 | WAM位置误差mm | WAM朝向误差° | 保持初态位置误差mm | 保持初态朝向误差° |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        for point in group["checkpoints"]:
            lines.append(
                f"| {group['horizon']}步 | {point['time_s']:.2f}s | {point['position_mm']:.3f} | {point['rotation_deg']:.3f} | {point['hold_position_mm']:.3f} | {point['hold_rotation_deg']:.3f} |"
            )
    lines += [
        "",
        "| 请求长度 | 整段位置RMSE mm | 整段朝向RMSE° | 裁切后开度MAE（0–1） | 原始开度越界 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            f"| {group['horizon']}步 | {group['position_rmse_mm']:.3f} | {group['rotation_rmse_deg']:.3f} | {group['opening_mae_clipped']:.5f} | {100 * group['opening_outside_fraction']:.2f}% |"
        )
    lines += [
        "",
        "16/40步是分别请求得到的不同预测，不能把它们拼成同一条连续轨迹。误差不含任何控制器跟踪误差；单条示范差异本身不证明任务失败，任务归因仍依靠独立物理正控/分量/几何证据。",
        "",
        "验证：当前原始Parquet SHA与图像配对时一致；重新读取的state与快照逐项相同；10份response/manifest checkpoint一致；280个预测endpoint通过独立R/t递推对照（1e-12）；主要输入SHA、每条误差曲线见同名JSON。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(groups, indent=2, default=float))


if __name__ == "__main__":
    main()
