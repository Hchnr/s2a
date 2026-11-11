# 原质量门外的模型安全有界跟踪回退

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/model_safe_tracking_fallback_20261111.md)，
在三个任务的四个已研究开发起点比较统一的四级因果反馈。原严格三级无候选时，
允许已完成计划的后移结果在独立模型安全审核和额外误差界内作为降级回退；原
实际位姿、关节缓冲和任务成功标准均不变。三个原来停止的分支多执行了周期，但
新增回退周期**全部未通过原位姿双阈值**，也没有抓取或任务成功，不采用为完成
目标的统一候选。

| 案例 / 固定已生成参考 | 先前严格分支达标 / 执行 | 本次达标 / 执行 | 本次最大实际位置 / SO(3)误差 | 实际最小关节余量 | 停止 |
| --- | ---: | ---: | --- | ---: | --- |
| task0 seed43 raw，step16 | 8/8 | 8/8 | 0.009503 m / 0.047325 rad | 0.632306 rad | 完成8周期 |
| task1 seed43 raw，step16 | 1/1 | 1/3 | 0.016912 m / 0.074878 rad | 0.728850 rad | step19前回退超过额外界限 |
| task2 seed43 raw，step32 | 1/1 | 1/4 | 0.018784 m / 0.073185 rad | 0.199574 rad | step36前回退超过额外界限 |
| task2 seed43 terminal058，step80 | 5/5 | 5/6 | 0.012594 m / 0.061614 rad | 0.097992 rad | step86前严格搜索初值不可用 |

原实际位姿门为位置≤0.01 m、SO(3)≤0.05 rad；回退另设的模型首点上界
0.02 m / 0.10 rad只是本次预登记的运行边界，**不是**原跟踪合格条件。两种
task2变体使用同一开发初态和seed，不是独立泛化样本。各案例在首次回退前的
8、1、1、5个动作、完整状态和每周期25个内环样本与既有严格分支逐项一致；
之后由本分支实际反馈更新历史，不能把旧分支后续状态当作当前输入。

本次21个新周期中，15次选择严格候选、6次选择标为
`model_safe_tracking_degraded`的回退；6次独立模型审核均通过原接口box、
命令变化率、关节缓冲及制动约束，但6个对应的**实际**端点没有一个满足原位姿
双阈值。525个新增实际2 ms内环样本均未低于0.05 rad缓冲，也没有越关节范围；
这不是对未执行周期、完整前缀或未来行为的安全保证。四个短窗均无目标grasp和
任务成功端点；与原对照一致，夹爪输入保持0，本实验没有验证抓取与释放策略。

task1与task2 raw的实际误差随继续执行增大，下一次后移计划预测首点超过本次额外
界限而停止；task2 terminal058下一次严格搜索没有可用初值，故不进入回退。有限
搜索的停止不证明物理不可达。已派发臂决策约 `0.089–3.737 s`，超过20 Hz的
0.05 s；仿真等待时物理时间暂停，没有实时资格。本试验使用固定已生成chunk，
可跨原本重新查询的时刻，不构成新WAM闭环或完整任务评价。

## 独立重放与费用

checker没有加载候选策略或重新选择动作，从各自原初态重放21条实际命令。完整
状态、525个新增内环 q / qvel / time、TCP误差、实测opening / motor、grasp及
任务标签与producer逐值一致；task0/1的旧/新参考误差也逐值一致。checker核对
停止位置但未重新求解无候选原因；模型安全审核是producer独立标量审核，不等同
于独立物理安全预测。

producer和checker每阶段各165控制周期 / 4125内步、12模型 / 12个OSMesa软件
context；合计330周期 / 8250内步。producer有24次质量调用 / 1676次模型响应、
12次初值LP、16次后移rollout、4200审核端点查询 / 54600模型转换步；checker
新选命令为0。8个worker和两个监督进程均实际wait/reap、exit0、未发信号，输入
与源码前后一致。执行源码提交为 `110cdf11`；32进程CPU专项143项通过、0跳过，
零模型dry及全仓Ruff/格式检查通过；未重跑全量pytest。`complete/passed`表示
实验和复核闭合，不表示控制合格。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `model_safe_tracking_fallback_20261111/producer/result.json` | `0328973e954fc731721ce2aee61dc654b15fd7c65bf93ebc48dffed634a80c40` |
| `model_safe_tracking_fallback_20261111/producer/workers.json` | `bb05fc36db1bd132694e7ccbed297c8206652cd65b167db3b0867e0413dd0c4f` |
| `model_safe_tracking_fallback_20261111/checker/result.json` | `f1b6bed4ae12475d7b7cfd37a3007736b3beb9302e1985d2f0288d304582c2d7` |
| `model_safe_tracking_fallback_20261111/checker/workers.json` | `b5630aa1c413e709fe7c8f082270d43b23a2e8500a2e1ea1d5d6ba08ec1d5b23` |
| `model_safe_tracking_fallback_supervisor_20261111/producer_exit.json` | `c3aca3a20038a6a5695dedc788f38eb7d4f5e37eac8489c86764227ce0cbaf7b` |
| `model_safe_tracking_fallback_supervisor_20261111/checker_exit.json` | `bfa17aed15ac78d3376b4f3a47a1886d97037a7124f36f486d461b42c493c749` |
| `model_safe_tracking_fallback_dry_20261111/result.json` | `fa1c0d739672b5284e4f7a45bed65c733a06e4d6d8dafa76ffd6e081846becc8` |
| `qa/model_safe_tracking_fallback_20261111/result.json` | `7c20a5e7e27fccc04dfb7a706e997891221b2f6a1dcb642e1e8fc74cb293a2a3` |

本轮没有新WAM、GPU或最终留出实验。后续应研究在不绕过实际安全和原误差评价的
前提下，能否通过更早的参考衔接或有界重新优化阻止跟踪误差累积，并接入真实夹爪
控制、完整开发任务及正确参考正控。原18正确参考、12 WAM开发、24最终留出及
≥20/24、每任务≥4/8的稳定成功门槛不变。
