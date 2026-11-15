# 开发轨迹的质量刷新边界

2026-11-15。对[六条已执行开发轨迹](../references/reset_multitask_adaptive_complete_20261113.md)
的 891 次已记录段内刷新检查，依据对应已完成周期的独立 witness 重算第二点误差
范数与刷新判断。每条信号都绑定当前端点、前一完成动作及 25 个内步；未新增
WAM、物理步骤或新初态。这是观察现象后的回顾，三个任务各只有一个旧初态。

| 案例 | 余量可用 | 余量耗尽 | 无已审核第二点 |
| --- | ---: | ---: | ---: |
| bowl / seed 42 | 80 | 38 | 0 |
| bowl / seed 43 | 13 | 3 | 0 |
| wine / seed 42 | 130 | 1 | 0 |
| wine / seed 43 | 29 | 3 | 2 |
| cream / seed 42 | 379 | 32 | 0 |
| cream / seed 43 | 138 | 42 | 1 |
| 合计 | 769 | 119 | 3 |

119 次余量耗尽中，72 次的位置预测距 `0.0095 m` 边界不超过 `1e-10 m`，
101 次的旋转预测距 `0.047 rad` 边界不超过 `1e-10 rad`；其中 54 次两者
同时满足，故至少一项贴边共 119 次。规划候选接受到边界加 `1e-10`，刷新在
边界减 `1e-10` 时触发，详见
[选择条件](../../../src/s2a/research/trajectory_search/margin_two_step_search.py)和
[刷新条件](../../../src/s2a/research/joint_control/preview/causal_reference_replan.py)。
这是规划和刷新规则重叠的证据，但不能证明边界处一定存在满足更大余量的安全
候选，也不能把减少刷新等同于改善真实状态跟踪或任务完成。

核验直接从每次已完成 witness 的六维第二点误差独立计算范数和布尔条件，并检查
原始信号；不独立重算候选求解或物理停止。配置 32 并发的相关 CPU 回归实际运行
25 项、25 个 worker，全部通过并 wait/reap。原五条 owner postflight 失败维持原状。
后续预先登记的候选排序与未来质量约束见
[设计](../../../plans/v0.0.1/tracking/references/refresh_margin_preference_20261115.md)和
[补充](../../../plans/v0.0.1/tracking/references/refresh_margin_preference_addendum_20261115.md)。

| 本地证据 | SHA256 |
| --- | --- |
| `outputs/refresh_margin_events_20261115/result.json` | `dd266656d283d731227985eef7ca01f5861a92cddefa06b089209f562f73cadf` |
| `outputs/qa/refresh_margin_events_20261115/result.json` | `64054d6df6b336d692d8c5f861c8e2c0e7868c754836f49135c7bf1c28ccabd4` |

输出不随 Git 提交；结果中保存原始 producer、计划和核验脚本的指纹。
