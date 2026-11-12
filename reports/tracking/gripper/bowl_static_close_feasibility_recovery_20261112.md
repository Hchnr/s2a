# 固定姿态下双 pad 静态闭合的恢复枚举

2026-11-12。根据[恢复计划](../../../plans/v0.0.1/tracking/gripper/bowl_static_close_feasibility_recovery_20261112.md)，
在[首次 LP 残差失败](bowl_static_close_feasibility_failure_20261112.md)和[单组合容差诊断](bowl_static_close_lp_diagnostic_20261112.md)
后，对原固定 H40 与 opening 提前四周期两条旧开发轨迹的第 48 个周期末分别枚举
1,600 种左右目标 box 组合。仅固定保存状态的物体与 pad 姿态、允许夹爪平移及两侧
手指对称向内闭合，要求原模型 TCP 位置误差不超过 0.010 m，且两个 pad–目标 box
均具有至少 `1e-5 m` 的静态重叠裕量。

| 指标 | 固定 H40 | opening 提前四周期 |
| --- | ---: | ---: |
| 实际 LP 数 | 1,600 | 1,600 |
| 求解器 optimal / infeasible 状态 | 700 / 900 | 675 / 925 |
| 原位置门内的独立验证几何 witness | 8 | 8 |
| witness 对应的实际几何 opening 范围 | 0.0465–0.1426 | 0.0571–0.1513 |
| 同一端点原模型 opening 参考 | 0.7062 | 0.7062 |

每个 witness 的平移、对称闭合量和各指关节边界经独立核验：对两个 pad 的实际 box
重新检查分离轴，并用线性规划验证严格内部点；TCP 的欧氏位置误差按原模型目标
计算。固定 H40 有 witness 将 TCP 平移到原模型位置目标，示例对称闭合约 0.02845 m；
提前组最小位置误差 witness 约 `6.95e-5 m`，对称闭合约 0.02429 m。

这些只是**找到的 witness**，不是所有可能解的开度范围，也不是力闭合或任务成功。
其中有些 witness 让两侧 pad 与同一个薄目标 box 重叠；假设还冻结物体和姿态，未
检查手掌、其他机器人几何、接触力、连续运动、夹爪动力学或可达性。实际执行中物体
和控制状态会改变。它们均比原模型 `0.7062` 的开度明显更闭合，不能声称跟踪了模型
原始完整状态。求解器返回 infeasible 的组合是数值求解结果，不据此排除其他姿态、
更小接触裕量或所有动态策略。离线分析使用碗真值，仅用于形成待验证的开发假设；
在线动作不得读取该真值。

## 数值核验与资源

恢复时显式设置 HiGHS primal/dual feasibility 容差为 `1e-9`，原 `1e-8 m` 解残差
检查及任务标准未放宽。两个分支分别进行 11 和 10 次欧氏细化。独立只读审核对
1,375 个返回解的两个 pad 共 2,750 次重新执行分离轴几何检查，并对 16 个 witness
的两个 pad 共 32 次使用独立线性规划检查内部点；全部报告的 witness 均满足原位置
门和原生手指范围。恢复 3,200 次 LP 加此前失败与诊断 13 次，以及审核 32 次独立
LP，累计 3,245 次相关 LP；这些是静态数值查询，没有新仿真模型、forward、物理步、
渲染 context、世界模型请求或 fresh 案例。恢复前 32 并发相关回归 100 项通过、0 skip。

恢复结果 `outputs/bowl_static_close_feasibility_recovery_20261112/result.json` SHA-256
`f90afbd74260b43dafe9851ed80ff596f1e977423ddfacac3c01fca7e58aaabb`；[独立审核](../../../scripts/simulation/runtime/audit_bowl_static_close_feasibility.py)
`outputs/bowl_static_close_feasibility_recovery_audit_20261112/result.json` SHA-256
`ca563b70af4bd0514dfc7dacd997acd95eb7aca9581e34e45d4a9bdf9c20ddd3`。此单初态结果不新增
抓取或任务成功，也不改变原留出、实时或真机验收。
