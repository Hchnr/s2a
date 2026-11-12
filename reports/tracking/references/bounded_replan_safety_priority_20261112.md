# 连续因果重规划：预算边界的安全停止优先级

2026-11-12。软件审查发现，当部分段最后一个允许周期同时耗尽物理预算且实测
关节缓冲不足时，先前执行器会先记录 `period_budget_exhausted`，掩盖缓冲停止
原因；两种情况都未继续派发动作。现调整为先检查已完成周期的缓冲，再报告预算，
并用两周期合成场景验证最后周期同时触发时记录 `observed_joint_buffer_loss`。

32 并发相关回归 **46 项通过、0 skip、0 collection errors**；检查记录
`outputs/qa/adaptive_buffer_budget_priority_20261112/result.json` SHA-256
`04b8e55beb063a05926f8df7e85bc6ab8b6d1370c840630488e5b9c32b5ba406`。
无新真实模型、GPU、物理仿真或留出试验。
