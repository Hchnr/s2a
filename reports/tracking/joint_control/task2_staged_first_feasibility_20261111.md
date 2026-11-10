# 安全可行初值上的两阶段首点优先反馈

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_staged_first_feasibility_20261111.md)，
在[可行初值对照](task2_feasible_start_restoration_20261111.md)相同的task2 raw及
terminal058开发起点、已完成前缀和已生成参考上，先优化首点位姿超限，再在首点
硬约束下优化未来缺口。两阶段共用原320次响应预算及缓存，候选保留原动作、关节、
制动、终端和首点质量审核；分支后仅使用自身实际状态与已完成历史。

| 开发起点 | 执行周期 / 实际双阈值 | 实际位置 / SO(3)误差 | 内环最小余量 rad | 停止 |
| --- | --- | --- | ---: | --- |
| raw step32 | 1 / 0/1 | 0.01009471 m / 0.02147605 rad | 0.461285 | step33前无witness |
| terminal058 step80 | 1 / 1/1 | 0.00530873 m / 0.03879719 rad | 0.327534 | step81前无witness |

两个已执行周期的2 ms关节内环均未跌破0.05 rad缓冲、没有越界、目标grasp或
任务成功；两个分支都不足八周期。结果没有改善上一轮的执行周期数或实际双阈值。
两个变体属于同一开发初态和seed43，不能作为独立泛化样本。

## 第二次决策的有限搜索

| 停止状态 | 原线性安全候选 / 首点合格候选 | 最近安全尝试的模型首点位置 / SO(3) | 最坏归一化首点超限 | 响应 |
| --- | ---: | --- | ---: | ---: |
| raw step33 | 21 / 0 | 0.013686 m / 0.055657 rad | 0.368628 | 196 |
| terminal058 step81 | 12 / 0 | 0.010649 m / 0.063385 rad | 0.267701 | 320 |

两个停止状态的第一阶段优化器均报告成功终止，但首点超限平方目标仍分别为
0.074343和0.037526，不能解释为找到首点可行解。raw的第二阶段以正方向导数
线搜索失败停止；terminal058耗尽响应预算，第二阶段实际迭代数未知。最近尝试
没有通过首点质量，不是可派发动作或全局下界。未来最坏质量缺口单独记录，不把
首点审核通过称为八点硬质量witness；有限搜索失败也不证明物理不可达。

上一轮对应停止处评估到6/1个线性安全候选，最近尝试的最坏首点超限为
0.387837/0.414797。本轮覆盖更多安全候选且记录的最近超限较小，但raw首动作
最大差1.69e-6、随后状态最大差6.85e-7，因此不能把两次raw停止状态当作逐值相同；
terminal首动作和随后状态逐值相同。这些模型搜索差异没有转化为更多实际动作。

## 独立复核与费用

checker未导入候选控制策略，重放2条已派发动作；完整状态、50个2 ms内环q/qvel/time、
命令及实际TCP误差逐值一致，grasp和任务标签一致。两分支前缀末周期内环也逐值
一致。无witness停止只核对停止位置和未派发，checker未重新求解。两阶段各2个
CPU/OSMesa worker均实际wait/reap、exit0且未发信号；监督进程实际等待并回收
两个执行器，源码与输入前后一致。

每phase实际114控制周期、2850个2 ms步、6模型和6软件context；两phase合计
228周期、5700内步、12模型和12 context。producer实际4次原细化/283响应、
4次两阶段搜索/655个实际模型响应、4次初值LP、0次额外求解器资格LP及2次独立
审核计400端点查询/5200模型转换步；4次种子rollout复用，单次未超过320响应。
两条已派发决策约2.85/3.50秒，仿真等待时物理暂停，不构成20 Hz实时资格。

84项32并发CPU专项、全仓Ruff/格式和零模型dry通过；本轮未重跑全量pytest。
无新WAM、GPU或最终留出数据。原18正确参考、12开发和24最终留出门槛保持；
`complete/passed`只表示执行与重放闭合。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_staged_first_feasibility_20261111/producer/result.json` | `6010b4acbac2f489cf5720318408e8729085697a6e333d77b1d2a05cb0cec3d6` |
| `outputs/task2_staged_first_feasibility_20261111/checker/result.json` | `427010338d1f617b4cb8d6a479aefd0d314c91b42ba900f5a97428e219d4302d` |
| `outputs/task2_staged_first_feasibility_dry_20261111/result.json` | `fb58ff6f3c53595a2f18f4effadefcaf600a95f05040ded746ddc5712665b018` |
| `outputs/qa/task2_staged_first_feasibility_20261111/result.json` | `97828e4e916db416365d0a1a1399b606d0f8fa24479bc2b8a09abd10ee96d617` |
| `outputs/task2_staged_first_feasibility_supervisor_20261111/producer_exit.json` | `dd91ca6d48e954d38dca27b40b9b17ad43f132df4d95e0faf8da3c06adfa216e` |
| `outputs/task2_staged_first_feasibility_supervisor_20261111/checker_exit.json` | `713a288687e4d52bbf48b252982d65b310f54973f1e7e5d1f857d42926d23fb8` |

后续需要分析停止状态的首点质量与安全约束冲突、模型与实际首点误差的差别，
以及更早的因果参考或控制衔接；不能靠放宽原阈值、缩短验收窗口或读取未来观测
来声称完成目标。
