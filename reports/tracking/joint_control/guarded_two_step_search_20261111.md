# 双点可行性搜索的数值尺度对照

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/guarded_two_step_search_20261111.md)，
仅调整双点搜索第一阶段的目标尺度及内部搜索余量；原第二阶段、候选筛选、
0.01 m / 0.05 rad质量阈值、`1e-10`审核容差与原安全约束不变。四组输入指纹
均与[原双点静态对照](two_step_viability_20261111.md)逐项一致。

| 开发决策 | 原双点搜索 | 新搜索与独立审核 | 原 / 新响应 |
| --- | --- | --- | ---: |
| task0 raw step47 | 无witness | 无witness | 62 / 79 |
| task1 raw step56 | 无witness | 无witness | 63 / 105 |
| task2 raw step32 | 无witness | **双点模型witness，独立审核通过** | 39 / 52 |
| task2 terminal058 step80 | 双点模型witness | **双点模型witness，独立审核通过** | 64 / 67 |

raw新候选的前两点模型误差分别为0.008689 m / 0.013309 rad和约0.010000 m /
0.050000 rad，第二点分别超过标称上限约`5.6e-12 m`及`6.1e-12 rad`，均在原
`1e-10`审核容差内。它通过八周期原关节、动作、终端和制动约束及前两点质量的
独立模型审核；相对原已派发计划的首动作最大分量差约0.28944，后六点最坏归一化
质量缺口为0.771957，不能称完整八点质量witness。

terminal058新候选的前两点为0.006553 m / 0.017454 rad和约0.010000 m /
0.050000 rad，后六点最坏缺口1.295109，仍仅是双点模型witness。两个候选的
第二点均贴近原边界：第一阶段内部余量没有成为最终候选的额外硬条件，不能宣称
最终模型结果有该余量，更不能保证实际误差。

task0/1仍未找到双点witness，最近线性安全尝试的第二点分别为0.011270 m /
0.067036 rad及0.010949 m / 0.059532 rad；两次第一阶段均以线搜索正方向导数
终止。改变目标尺度并未解决这两个状态，有限搜索失败不证明全局或物理不可达。
task2两个变体属于同一开发初态和seed43，不能当作独立泛化样本。

## 独立复核与费用

新搜索4次共303个不同模型响应，原静态对照另为228个；本次没有调用初值LP。
两条候选经独立标量关节响应与八点SO(3)重建审核，共400端点查询及5200模型转换
步。4个静态模型、4次`mj_forward`和4次准备rollout均完成；4个CPU worker和
监督进程均实际wait/reap、exit0且未发信号，输入与源码前后一致。

32项32并发CPU专项、全仓Ruff/格式和零模型dry通过；本轮未重跑全量pytest。
0物理积分、新WAM、GPU或最终留出数据。原18正确参考、12开发和24最终留出门槛
不变；`complete/passed`仅表示静态对照及审核闭合。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/guarded_two_step_search_20261111/result.json` | `0f6ed1926a5091eda36e6bf3794e588aa62d7b3ffadfba82afa1b289750a404b` |
| `outputs/guarded_two_step_search_20261111/workers.json` | `0d515e7d278e110edbd0c0d5d172561f86877dbaaf50e746d54a88d59a63f969` |
| `outputs/guarded_two_step_search_dry_20261111/result.json` | `e3d892d477a863cd061688044606743080fbdf7bf0fdfa0d01625c298630f5cf` |
| `outputs/qa/guarded_two_step_search_20261111/result.json` | `25d462dae2b3b1869eef73be997dc59b54161dffec8ab06f6d35b36d97f9868f` |
| `outputs/guarded_two_step_search_supervisor_20261111/owner_exit.json` | `d85b445e950c0523b8e690f7c6a3c89e485771e898cc640c37c020e83ce107e9` |

下一步在相同task2开发前缀接入新搜索，分别使用本分支实际历史进行CPU物理反馈
和独立动作重放，检查它是否改变raw首动作和持续性；实际阈值与完整窗口不变。
