# 从复位执行完整正确参考的单案例正控

2026-11-16。按[预登记](../../../plans/v0.0.1/tracking/joint_control/reset_correct_reference_20261116.md)，
统一候选从 correct_08 / task6 / seed42 的原 reset 开始，使用完整离线正确参考，
实际选择并执行了 **182/182 个控制周期**；最后 40 个周期原生任务谓词均为真。
此前[该案例的连续反馈](ordered_correct_stream_feedback_20261111.md)先重放了
80 个历史动作，不能据此证明候选能从复位完成。本次没有派发历史动作。

原基线与本次的 182 个 TCP 和 opening 目标逐项最大差均为 0；两者从同一已验证
reset、固定控制器和 TCP 出发。原 23 个参考窗口的重叠位姿最大差为
`1.3323e-15`，末尾目标不足 40 点时保持最后目标。该参考是离线 oracle，
不代表世界模型从当前图像生成了这些目标。

| 指标 | 原正常控制器 | 当前统一候选 |
| --- | ---: | ---: |
| 完整执行周期 | 182 / 182 | 182 / 182 |
| 最后 40 周期任务成功 | 40 / 40 | 40 / 40 |
| TCP 位置 RMSE（m） | 0.00109046 | 0.00050554 |
| TCP 旋转 RMSE（rad） | 0.00299672 | 0.00148546 |
| opening RMSE（归一化） | 0.06443374 | 0.06471388 |

本次 182/182 个端点满足 0.01 m / 0.05 rad pose 门；位置与旋转最大误差
分别为 `0.00241903 m` 和 `0.00921228 rad`，opening 最大绝对误差为
`0.38941072`。原生任务谓词从 step 96 开始连续为真，共 86 周期；目标 grasp
谓词为真 41 周期。全部 4,550 个 2 ms 内步的最小关节余量为 `0.64065067 rad`，
没有低于 `0.05 rad` 缓冲的样本。结果为同一案例的描述性对照；候选和原控制器
的策略不同，不能将差异归因于某个组件或推广到其他初态。

## 重放和耗时

独立 checker 从相同 reset 重放全部 182 条实际动作，不加载候选策略、也不选择
新动作。动作、实际状态、TCP 误差、所有 4,550 个内步的 q / qvel / 时间以及
夹爪反馈的最大绝对差均为 0，任务和 grasp 标签一致。checker 保留 producer 的
参考终止记录，没有重新计算候选策略的决定。producer 与 checker 各使用 3 个
仿真模型和 3 个 OSMesa context，各完成 182 周期 / 4,550 内步；两个 worker
均正常退出并实际 wait/reap。reset 物理状态 SHA 为
`98c0b49db7e41f7de6e1c411cfd60dd403bbcdac2bc49c2124b2bc0d0fdef524`，
两次实际观测与记录的状态字节及相关端点字段一致。

producer 的机械臂 preview / refiner 计时中位数为 `77.80 ms`、p95 为
`115.64 ms`、最大值为 `149.85 ms`，182/182 次超过 20 Hz 周期的 `50 ms`。
该计时只覆盖代码中的部分动作选择，不含完整夹爪和端到端开销，且不是隔离机器
上的实时性基准；它表明本次运行不能宣称满足 20 Hz 实时要求。

本次没有 WAM 调用、GPU 任务或新初态。独立审计重新计算了目标一致性、重放差、
误差和耗时；候选运行前相关 CPU 回归为 96 项通过、0 skip，审计新增回归为
38 项通过、0 skip，worker 均 wait/reap；全仓 Ruff 和格式检查通过。
原 18 条正确参考的统一候选完整验收、12 条 WAM 开发及 24 条最终留出初态
仍未完成；此前六条 WAM 开发轨迹为 0/6 任务成功。本次成功不改变这些结论。

物理运行使用 `4a40ca082b80187311403181cc09aaba736de98f` 的源码字节；
运行前的 QA 快照保存了测试时 Git HEAD 及逐文件 SHA，物理阶段在启动和结束时
核对了相同源码字节。审计是之后运行的只读分析，未执行额外物理周期。

| 仓库内证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `reset_correct_reference_20261116/producer/result.json` | `7236573a008090168f619c8055c2c8a7af6efd0e51be6a427fa69a2a6e41dd4f` |
| `reset_correct_reference_20261116/checker/result.json` | `f6270a34d14f9339ca94dd2b50c1d48dc0c89822ad45a5edada9b5085477f511` |
| `reset_correct_reference_20261116/audit.json` | `9444bc7c66b314ae77909a41eb4e514257015956de84551ab525afe8a387b712` |
| `reset_correct_reference_dry_20261116.json` | `155b24ad9a7d0ddba2b5c1fb277dcd6e2fd5780dae4295f44b19038edae073f3` |
| `qa/reset_correct_reference_final_20261116/result.json` | `4de804ef137af092584ff00ddfb73c5da03db66e91e57755be9dfc052580504b` |
| `qa/reset_correct_reference_audit_20261116/result.json` | `560b46ec04ba897f24380ae613f84005a4a5deddfe12bc28bd88b9b0ad13e250` |
| `tracking_correct_reference_ordered_20261027/case08_candidate/result.json` | `885d02679fcdf75ab3067de9b1557cff4a5cb0c5671ca149aa9e0afef3da0da0` |

输出保存在本仓库 `outputs/`，不随 Git 提交；审计入口为
[`audit_reset_correct_reference.py`](../../../scripts/verification/reference_ports/audit_reset_correct_reference.py)。
