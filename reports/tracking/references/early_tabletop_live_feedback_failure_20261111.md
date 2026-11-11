# 分支 WAM 反馈首次执行：回放比较类型错误

2026-11-11。[step120 两份独立 WAM 响应](early_tabletop_live_wam_model_20261111.md)
已完成，随后按[计划](../../../plans/v0.0.1/tracking/references/early_tabletop_intervention_live_wam_20261111.md)
启动两个 CPU 物理 producer。两者均重放到各自 step120，控制周期各 120、各
3,000 个 2 ms 内步；各分支的端点状态及重采图像 SHA 均与模型输入一致。两次
运行都在**新 WAM 参考派发动作之前**退出，新增动作 0、checker 未启动，不能
报告后续反馈或控制成功。

错误发生在验证已重放提前动作的比较器：历史 JSON 使用列表，而当前未发布
的内存回放包含 NumPy 数组；比较器把二者直接用于布尔相等，触发数组真值不明确
的 `ValueError`。两次进程退出码均为 1，均实际 wait/reap、未收到终止信号，
原生仿真清理完成。此阶段共实际消耗 6,000 内步，不自动重试或覆盖失败产物。

修复将当前内存回放先按既有严格 JSON 序列化规则转换成与历史收据相同的值
类型，再使用原比较器和原数值容差；不改变原始数据或控制门槛。新增数组/列表
混合的回归测试后，32 并发专项 7 项通过。失败时的
`scripts/simulation/runtime/early_tabletop_live_continuation.py` 原 SHA256 为
`ba89b4e1091c2bff7dcecf333255584afee98bc332369b605ae766dc1ba62e42`，
保存在[原始源码快照](../../../docs/history/experiments/early_tabletop_live_continuation_6350a4e4.py.txt)；
对应测试原 SHA256 为 `0b3e2ddf4c6b7140cdc52713f55468eb1d99703ca6a77dd0966da34ec86a4464`，
保存在[历史测试快照](../../../docs/history/experiments/test_early_tabletop_live_continuation_6350a4e4.py.txt)。
后续若继续物理反馈，必须先另行登记额外预算，沿用这两份已生成响应并将本次
6,000 内步计入总成本；不重新生成模型响应。

| 仓库内证据 | SHA256 |
| --- | --- |
| [失败 owner](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b/result.json) | `6c34f5affca2ba55478ca9426698d28db69d6858656454b32e9a696cf378c14b` |
| [producer 生命周期](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b/producers_workers.json) | `138ef88477e25886a1a91519392cd0b8d6cf22724b9952e9ad4554bd05a5c5e3` |
| [原目标失败收据](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b/original/producer/simulation/result.json) | `f57348bca657ff3d107ca470341f4f736d10a9538c36083db5ce4ac7c3184d88` |
| [投影失败收据](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b/pose_bounded/producer/simulation/result.json) | `d7156c165d5c5dedd718eb6b876f4c69b3ec5fda274d62925495d55685230271` |
| [比较修复 32 并发专项](../../../outputs/qa/early_tabletop_live_comparison_repair_20261111/result.json) | `9dc7a27017995bd973bf251355049e3fd2f83ddef6164b6a2171cd9adccb025e` |
