# 同一重规划状态下的新旧参考对照

2026-11-10。依据[预登记方案](../../../plans/v0.0.1/tracking/joint_control/replan_reference_switch_20261110.md)，
两条原开发轨迹均从原初态重放16周期到相同的端点16；相同控制历史、控制器和夹爪保持
命令下，分别执行当时已生成的new chunk2前8点或old chunk1后8点。每个动作完成后，
同一实测TCP均分别与两套相同绝对时刻的参考比较。

| 案例 | 实际跟踪参考 | 对old双阈值达标 | 对new双阈值达标 | 对old最大位置 / SO(3)误差 | 对new最大位置 / SO(3)误差 | 2 ms最小关节余量 |
| --- | --- | ---: | ---: | --- | --- | ---: |
| task0 seed43 raw | new | 2/8 | 6/8 | 0.053750 m / 0.088076 rad | 0.010787 m / 0.041922 rad | 0.631751 rad |
| task0 seed43 raw | old | 8/8 | 1/8 | 0.000352 m / 0.008376 rad | 0.054773 m / 0.085292 rad | 0.632311 rad |
| task1 seed43 raw | new | 3/8 | 3/8 | 0.036261 m / 0.091445 rad | 0.018665 m / 0.056127 rad | 0.687236 rad |
| task1 seed43 raw | old | 8/8 | 1/8 | 0.000763 m / 0.004815 rad | 0.038753 m / 0.112155 rad | 0.728850 rad |

四个分支均实际执行8周期，所有2 ms内环样本均保持原0.05 rad缓冲，0关节越界；均没有
目标grasp或任务成功。这两个案例在该固定起点上，跟踪旧参考的自身位姿误差更小，但对
**共同的新参考**只分别有1/8点达标；因此不能把旧参考自身跟踪通过写成新参考跟踪成功，
也不能据此判断哪条参考更符合任务意图。两个chunk在同一时间首目标的位置/姿态差分别为
0.008095 m / 0.012993 rad、0.007368 m / 0.029205 rad；8点内最大差分别为
0.054758 m / 0.087129 rad、0.038696 m / 0.111944 rad。

new分支的实际臂命令变化率分别有7/8和8/8周期达到原0.25上限，old分支分别为6/8和
5/8；它是约束活跃的现象，不能单独证明超限的唯一原因。两条new分支的首机械臂命令与
原记录最大差分别约 `5.45e-16/2.50e-16`，完整计划最大差约 `1.78e-15/2.28e-15`，
route一致；原step16夹爪命令为1，本次两分支均用0保持motor目标，故不声称完整七维
动作或后续状态与原始WAM轨迹一致。

## 来源、复核和范围

producer与独立checker各4个worker均exit0且实际wait/reap，源码与输入前后相同。
checker不导入候选preview或refiner，重放32条实际动作；完整状态、动作、800个2 ms
内环q/qvel/time、实际TCP以及两套参考误差与producer最大差均为0。其两套误差在完成
动作后重算，不用于选择动作。每phase实际96控制周期/2400内步、12模型/12软件context，
合计192周期/4800内步；producer有32次原refiner、0额外优化响应，checker无新求解。
47项32并发CPU专项、全仓Ruff/格式及零模型dry通过；0新WAM/GPU/fresh。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `replan_reference_switch_20261110/producer/result.json` | `58bf371b022e9daded4929df1904cc6f15ea4866f7e34791cb359f4a5f65044e` |
| `replan_reference_switch_20261110/checker/result.json` | `ba09b7ca0a4ca39de3edeb545e09db580ff5a8a7cad4af77e71f3c368c87102b` |
| `replan_reference_switch_dry_20261110/result.json` | `460cea15463391bb5056e67e1cc835256e6523eb8ebf67622eae67417aea2965` |
| `qa/replan_reference_switch_20261110/result.json` | `4b36018d19fe4865d4af22ef69aaefd89625621f42c6e62d66635f0a5ff631a3` |

下一步应以同一端点16及new参考为输入，比较原反馈与不放宽动作/关节约束的质量反馈；
如果找不到并实际验证能跟踪new参考的候选，不应将换用old参考追认为解决世界模型跟踪。
原18正确参考、12开发、24最终留出门槛保持，GPU继续禁用。
