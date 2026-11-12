# 因果部分换段与已重放轨迹影子检查

2026-11-12。依据[设计](../../../plans/v0.0.1/tracking/references/causal_partial_replan_20261112.md)，
新增已完成 1–8 周期后绑定当前图像参考的换段接口。切换前检查当前状态与 TCP
锚、已完成内环和命令历史、关节余量以及手臂和夹爪事务；切换后沿用原质量
求解器、实际上一命令、手臂/夹爪状态和内环误差包络。任何待完成事务或锚/时钟
不一致均拒绝；激活失败时恢复旧 refiner，且不追加运行收据。该接口还未接入
实际当前图像模型请求或物理控制循环。

只读影子检查对碗任务 step88 参考的 19 个已完成动作依次评估已审核的下一点预测：
首次触发发生在 step107，预测位置误差 `0.009500000000005317 m`、SO(3) 误差
`0.04285604012446784 rad`；位置触及原模型余量 `0.0095 m`。检查在首次触发处
停止，信号不读取任务成功或物体状态。该阈值和选择来自同一旧开发轨迹的事后
诊断，不证明新参考能产生更好动作，也不证明独立初态泛化。

32 并发 CPU 回归 49 项通过、0 skip；包括段内第 1、3、7、8 个完成周期、历史
延续、无完成动作、待处理事务、错误锚/时间、关节缓冲丢失和激活失败回滚。
影子 worker 实际 wait/reap、exit 0、未发信号；本阶段无物理积分、WAM、GPU
或 fresh case。原总目标、原质量门及留出验收不变。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/causal_partial_replan_shadow_20261112/result.json` | `1d5c64a878801858e02d0f8e456e1b2851203642d1b6bfd71361e3e82c32b71c` |
| `outputs/causal_partial_replan_shadow_20261112/worker/result.json` | `6b9827d463b80ba47b6fd1a4f79048558751dacae00236141658e391c68df880` |
| `outputs/causal_partial_replan_shadow_20261112/audit.py` | `e02dc2f43dc71a04791af0d466806272f2721b51e0ad4acf9837f0410e5162fd` |
| `outputs/qa/causal_partial_replan_v2_20261112/result.json` | `9afb5162611d027d085fac23cb3f2205aa40d0f290061bcd9c0d7b0ce372290c` |
