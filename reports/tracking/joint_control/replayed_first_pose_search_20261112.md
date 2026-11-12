# 已重放停止状态的首点优先求解对照

2026-11-12。按[预登记](../../../plans/v0.0.1/tracking/joint_control/replayed_first_pose_search_20261112.md)，
在已独立重放的碗任务旧开发初态 step108，使用上一实际命令、完成的后移计划、
已完成的内环误差包络及 step88 已生成参考，单独运行已有两阶段首点优先求解器。
这不是原双点搜索之后同一预算内的追加求解，也没有派发动作。

| 指标 | 结果 |
| --- | --- |
| 原后移种子的模型首点位置 / SO(3)误差 | 0.011382 m / 0.048123 rad |
| 首点优先求解中的模型响应 | 320 / 320 |
| 求解器记录的线性安全尝试 / 首点双阈值尝试 | 66 / 0 |
| 最近线性安全尝试的模型首点位置 / SO(3)误差 | 0.010120 m / 0.050597 rad |
| 最近尝试的最坏首点归一化超限 | 0.011976 |

首阶段执行 31 次迭代后报告优化终止，但首点超限平方目标仍为 0.0001383，
不是合格 witness。第二阶段用尽响应预算，实际迭代数未记录；求解器未产生候选，
因此独立候选审核未执行。上表安全尝试数和最近误差来自求解器记录，不是独立
候选审核。原位姿、安全和动作阈值均未放宽；有限搜索没有找到解不证明无解。

实际创建 1 个静态模型，执行 1 次 `mj_forward`、1 次重建种子 rollout 和 320 次
求解器模型响应；初值已线性安全，未调用初值 LP。求解墙钟约 5.10 s，不具备
20 Hz 实时资格。worker 实际 wait/reap、exit 0、未发信号；没有物理积分、
渲染、WAM、GPU 或 fresh case。32 并发专项 15 项通过、0 skip，零模型 dry
通过。单个旧开发初态不能说明多任务或留出成功。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/replayed_first_pose_search_20261112/result.json` | `9858e6e107210369f344cd028ef0f5b12accb784458a9bdf07e60b7c3ea3364f` |
| `outputs/replayed_first_pose_search_20261112/worker/result.json` | `eafe51dc0f552a9b7ac6cd18faaa2e3ccf0601e370a7a67b3b1c1825da2a9d60` |
| `outputs/replayed_first_pose_search_dry_20261112/result.json` | `5989e8c6d71743637ad7cb67dd119d6a14b8a7f07bb7ac7befd1462ceac1c6aa` |
| `outputs/qa/replayed_first_pose_search_20261112/result.json` | `e83749447a4ec1ffd300783ac04f71d002fc1501577531991b01209cc209a6fc` |
