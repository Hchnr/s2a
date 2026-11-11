# 因果绑定的两次 WAM 重观测

2026-11-11。按[登记](../../../plans/v0.0.1/tracking/references/wam_feedback_reobserve_20261111.md)
和[上下文修复](../../../plans/v0.0.1/tracking/references/wam_feedback_reobserve_context_repair_20261111.md)，
固定 iter_000008400、seed42、H40 和每请求 35 次去噪。先对已核验的 step80 当前
图像推理，raw 和 normalized 40×128 与先前实际驱动八周期的参考逐元素相同；只有
该门通过后，才以实际 WAM 反馈到达的 step88 当前图像生成第二参考。

修复后 GPU 工作进程在 `GPU-c0b598ba-0a6e-73af-bbcc-97bc89379bdf` 完成一次
初始化、两次请求和 70/70 次去噪；两条 initial-noise mask 记录均保存且无效槽
为零。CPU 独立审计复核实际请求、caption SHA、两图与状态绑定、响应有效槽、
逐次去噪顺序及第一参考精确相等。worker exit0、实际 wait/reap、无信号，设备
退出后显存 0 MiB、无计算进程。两次从请求准备到响应约 13.28 s 和 11.73 s，
不含模型初始化，不能据此宣称实时能力。

首次重观测曾完成一次初始化、step80 请求及 35 次去噪，但我在整理执行器时覆盖了
运行上下文变量，在线因果门之前抛出 AttributeError，未派发 step88。该失败及
exit1 收据保留；其保存的第一响应经离线复算与已执行参考相等，但不改记为成功。
修复后用模拟服务测试完整两请求及首响应不同时的提前停止，再在新目录执行。
含此前另一两请求 postflight 失败，本诊断链累计三次初始化、五次请求、175 次
去噪；没有新增物理或新初态。修复后 CPU 专项 34 项、独立审计专项 28 项通过，
合并前 64 路相关专项 36 项通过，均无跳过；全仓 Ruff 静态和格式检查通过。

第二参考来自第一参考实际反馈到达的状态，但前 80 周期使用正确参考历史，新的
step88 参考尚未驱动机器人；这不是完整 WAM 闭环或 18/12/24 最终验收。

| 仓库内证据 | SHA256 |
| --- | --- |
| 首次上下文失败 | `cfce5e1c14f58110b11533979fb96e4b404403b97b29915699639b6160590dd2` |
| 修复后准入 | `46289a2851269d24360adcc17d4ec6f4ee7fe00f4cad86ce0a7247d687ed2b43` |
| 修复后模型结果 | `becd62778765918df65e4bfaaa1b6a5d516eac20eccb1d44a81fae83ed42fa23` |
| 实际等待与 GPU 释放 | `c81c54fc5ad36184ee5c124fb5fce23e80d90ae342f844dea9df5c6f2e351498` |
| step80 精确一致性门 | `b3eb9f9b4c35e7e038b7e6867ddbb04c0be23a5724fbc61e754c8de0c01326b8` |
| step88 新响应 | `0e22981d338d7d9506ea77e4f5e324e6d0e7ed9b78436811bc2382c981e095bd` |
| CPU 独立审计 | `6d81b2c950f870bd10abcc47301cbe4b376e21b9f524d282c2a486b80ec4d76c` |
| 修复后 CPU 专项 | `50afb57d5c6c644b969ac9835ea4846e3686eb7de6898ae964aee55256d08ad9` |
| 独立审计 CPU 专项 | `9e73a1b17be77576d51f8978879810aee0dbe88740d5a90bf32c9ee7e2420fef` |
| 合并前 64 路 CPU 专项 | `d6e41ed056056020102e7fc0b5cc06c4434960ae3b5558fd783a1417e28345bd` |

后续更新：第二参考已在[两窗口物理反馈](../joint_control/causal_wam_two_window_feedback_20261111.md)
中执行并独立重放；上文“尚未驱动”描述的是本模型生成阶段结束时的状态。
