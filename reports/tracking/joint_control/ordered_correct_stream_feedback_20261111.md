# 正确参考受载点的双窗连续反馈

2026-11-11。按[登记](../../../plans/v0.0.1/tracking/joint_control/ordered_correct_stream_feedback_20261111.md)，
从 correct_08/task6/seed42 原初态重放 80 周期，然后使用同一质量约束机械臂和
正常 loaded 夹爪加暂缓闭合规则，连续反馈执行 step80–95。第 88 步切换到下一段
正确参考，夹爪 latch、实际上一命令、两套机械臂响应历史、warm start 和新反馈的
内环误差包络均持续保留；历史前缀未保存内环预测，因此起点不假称已有这项误差历史。

16/16 周期均完成，TCP 位置与旋转误差均满足 0.01 m 和 0.05 rad 门槛，16 次
质量 witness 审核通过。暂缓闭合发生在 step85–89。夹爪在 step90 后出现明显
opening 负误差，目标接触共 13/16 周期；step93–95 不再接触。任务成功仅出现在
最后一个周期，**不能认定稳定成功**。

原已审核正常控制器轨迹与本次有同一初态、相同前 80 个动作和相同的正确参考
绝对目标；step80 与 step88 的两段目标在重叠处最大差 `7.78e-16`。下表为这
16 周期的描述性比较；机械臂和夹爪策略同时变化，不能据此归因于单个组件。

| 指标 | 原正常控制器 | 新连续反馈 |
| --- | ---: | ---: |
| 位置 / 旋转门槛通过周期 | 16 / 16 | 16 / 16 |
| TCP 位置 RMSE（m） | 0.0000599 | 0.0001129 |
| TCP 旋转 RMSE（rad） | 0.0001814 | 0.0004279 |
| opening RMSE（归一化） | 0.1925 | 0.1775 |
| 最大绝对 opening 误差 | 0.3884 | 0.3587 |
| 目标 grasp 周期 | 14 / 16 | 13 / 16 |
| 任务成功周期 | 0 / 16 | 1 / 16 |

独立 checker 从相同初态重放了全部 16 条实际动作；16 个实际状态、400 个新周期
内环 q/qvel/时间逐元素一致，80 周期前缀的 2000 个内环样本也全部核验。独立
夹爪几何复算最大差为 0；checker 未导入候选策略。新周期最小关节余量 1.0446 rad，
前缀最小余量 0.6407 rad，均无低于 0.05 rad 的内环样本。producer 和 checker
各实际完成 96 控制周期、2400 内步，子进程均 exit0、wait/reap、无信号。

32 进程 CPU 专项 30 项通过、0 跳过；全仓 Ruff 和格式检查通过。本次无 WAM 调用、
训练或新初态，是一个局部正确参考正控；原 18 条完整正确参考、12 条 WAM 开发与
24 条留出初态的统一候选验收尚未完成。另一次[GPU 单样本模型复现](../references/gpu_resume_repro_20261111.md)
不能合并计算为本次模型闭环。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/ordered_correct_stream_feedback_20261111/producer/result.json` | `ff4efd0b687529cb0a93c5ba07dd6cfbfd2c9a9488f93e53e621b33a85b97998` |
| `outputs/ordered_correct_stream_feedback_20261111/producer/case_08_stream/result.json` | `054dad9bce3196b94c5620ee6f98825b934cdda249c6b3de069a0f071f9da91b` |
| `outputs/ordered_correct_stream_feedback_20261111/producer/workers.json` | `b0c1c3461a7f7a669ae0954936752048bd6694157b8b6229c3d5863c849af654` |
| `outputs/ordered_correct_stream_feedback_20261111/checker/result.json` | `c9000c2253f25d3dc1e7ba1146c79d1aab55647a1db86015a08094f2dc919b9d` |
| `outputs/ordered_correct_stream_feedback_20261111/checker/case_08_stream/result.json` | `2f58e689d659a3d2a425bd68fdcbb47ecccbd1739bd2898d2ba7279136693fa6` |
| `outputs/ordered_correct_stream_feedback_20261111/checker/workers.json` | `d202820d52f369952936ef84f23e628a4fc2950d863a60b466b9accae84cd4bc` |
| `outputs/ordered_correct_stream_feedback_dry_20261111/result.json` | `491c7aef299f3855c553d418716b16c0154257ece5559ac56f9b832bbc217746` |
| `outputs/qa/ordered_correct_stream_feedback_20261111/result.json` | `f7a54f9af050ba6837f3302476cb5bded04df97f91b6a59154f2d451cfb7d664` |
