# 重规划新参考下的质量约束实际反馈

2026-11-10。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/replan_quality_feedback_20261110.md)，
两条案例均从原始端点16的同一已完成历史开始，使用step16已生成的new参考；夹爪保持
命令、固定控制器和动作/关节限制与[已独立重放的new基线](replan_reference_switch_20261110.md)
一致。质量分支只在首点模型质量和独立关节witness复算通过时派发，使用自身已完成的
预测—实际内环误差收紧下一次模型约束。

| 案例 | 分支 | 已执行周期 | 已执行位姿双阈值达标 | 2 ms最小关节余量 rad | 最大实际位置 / SO(3)误差 | 停止 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| task0 seed43 raw | new原反馈 | 8 | 6/8 | 0.631751 | 0.010787 m / 0.041922 rad | 无 |
| task0 seed43 raw | new质量约束 | 8 | 7/8 | 0.632307 | 0.010026 m / 0.042805 rad | 无 |
| task1 seed43 raw | new原反馈 | 8 | 3/8 | 0.687236 | 0.018665 m / 0.056127 rad | 无 |
| task1 seed43 raw | new质量约束 | 1 | 1/1 | 0.728863 | 0.008106 m / 0.018078 rad | step17派发前无witness |

两条质量分支已执行期间，实际内环均没有低于0.05 rad缓冲或越界，均无目标grasp或任务
成功。task0质量分支比原反馈少1个位姿超限周期，但仍未达到8/8，不能称完整短窗通过；
task1只完成1/8周期，原反馈的对应第1周期也达标，不能拿1/1与3/8作为改善比较。
后者是有限非线性搜索、原安全和额外制动条件下未找到witness，不证明全局无可行解。

task0第19步模型预测的位置误差为 `0.010000000000016326 m`，实际为
`0.01002599036428073 m`，超原阈值约26微米。模型约束贴边时，即使独立复算通过，
实际响应偏差仍使该点未通过；不能把模型witness当作实际质量保证。已派发决策的CPU
墙钟约1.15–1.35 s/周期，仿真等待期间物理暂停，不具实时资格。

## 复核与费用

producer与独立checker各2个worker均exit0并实际wait/reap，源码和输入前后相同。
checker不导入候选决策，重放9条已派发动作、9个后续完整状态、225个2 ms内环q/qvel/
time、实际TCP与两套参考误差，最大差均为0。task1在step17前停止的位置和动作数得到
核对；checker沿用producer记录的无witness原因，没有重新求解或独立证明无解。

producer与checker各41控制周期/1025内步、6模型/6软件context，合计82周期/2050内步；
producer实际10次原refiner、10次质量求解/934响应、9次witness审核计1800端点查询/
23400模型转换步，checker无新求解。44项32并发CPU专项、全仓Ruff/格式及零模型dry通过；
0新WAM/GPU/fresh。运行 `complete/passed` 表示执行与重放闭合，不代表跟踪或任务通过。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `replan_quality_feedback_20261110/producer/result.json` | `d6541e22dece07ca214b02f4df7b5b81ce613a632802116d933df8032deee63f` |
| `replan_quality_feedback_20261110/checker/result.json` | `0fca5c925a4477efa213f9578d2802eb4fd3d6724b35ea4d6123bd7916eb16f2` |
| `replan_quality_feedback_dry_20261110/result.json` | `813f4478289bd207c0d3ef36fb1151fc55570414d33b602b17cdb5a5e82472b5` |
| `qa/replan_quality_feedback_20261110/result.json` | `957b6aa37b794457f20063c4dfe589be916717d6d34539588d8e485697bec5f7` |

下一步应检查当前重规划起点的多周期跟踪和动作变化率可行性，以及针对实际模型误差的
因果质量余量；必要时改进候选规划，再用各分支自己的物理反馈验证。原18正确参考、
12开发、24最终留出和GPU禁用条件不变。
