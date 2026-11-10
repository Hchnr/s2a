# 第三个任务早期预警的安全与跟踪反馈

2026-11-11。依据[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_warning_feedback_20261111.md)，
从已[逐端点复现](task2_warning_prefix_20261111.md)的task2/s43/raw端点34和
task2/s43/terminal058端点82出发，各比较原细化反馈与首点质量约束反馈。每对分支
使用相同原初态、已完成命令前缀及当时已生成的step32/80 chunk；分歧后只使用各自
实际状态与因果历史。两条是同一开发初态的不同生成变体，不是两个新初态。

| 案例 / 分支 | 已执行周期 | 实际TCP双阈值 | 实际内环最小关节余量 rad | 低于0.05 / 越界内环样本 | 最大位置 / SO(3)误差 | 停止 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| raw / 原细化 | 8 | 1/8 | 0.050248796 | 0 / 0 | 0.023113 m / 0.057669 rad | 无 |
| raw / 质量 | 0 | — | — | — | — | step34派发前无witness |
| terminal058 / 原细化 | 6 | 3/6 | 0.049440463 | 11 / 0 | 0.005644 m / 0.084102 rad | step88派发前缓冲丢失 |
| terminal058 / 质量 | 3 | 3/3 | 0.110226194 | 0 / 0 | 0.005582 m / 0.044098 rad | step85派发前无witness |

双阈值为实际TCP位置≤0.01 m及SO(3)误差≤0.05 rad；四分支均无目标grasp和任务
成功端点。两个质量分支均未完成8周期，不能以3/3的已执行部分宣称短窗通过。
两次停止均为`nonlinear_search_no_witness`，保留原关节、制动、动作和首点质量
约束；有限搜索失败不证明全局不可达。

terminal058原细化的第6个执行周期step87已有11个2 ms样本低于0.05 rad缓冲；
新门控在完成该周期并观察内环后阻止step88派发，不能把下一步停止说成本周期
没有失守。该周期没有越原XML关节范围。所有分支夹爪命令固定0以保持原motor
目标；本对照不验证抓取/释放控制，也没有新的世界模型重规划。臂决策约
1.37–3.34秒/周期，仿真等待期间物理暂停；并行共享机器上的墙钟不构成实时资格。

## 独立重放与实际费用

checker不导入或重新运行preview/细化/质量候选，从原初态重放17条已派发动作；
完整状态、425个2 ms内环q/qvel/time及TCP误差最大差均为0，grasp/任务标签一致。
质量停止仅核对位置和未派发，未独立重求；缓冲丢失停止由重放的实际样本和当前
状态复核。每分支的末次前缀25内步也与此前原前缀复现逐值一致。

producer和checker各4个CPU/OSMesa worker均实际wait/reap、exit0且未发信号，
两个phase的监督进程也实际等待并回收各自执行器；源码、输入及资产前后一致。
每phase实际249控制周期、6225个2 ms内步、12模型和12软件context；两phase合计
498周期、12450内步、24模型和24 context。producer实际选择17条动作，19次原细化
共1421响应、5次质量求解共397响应、3次witness审核共600端点查询和7800模型
转换步；checker未选择动作或重新求解。

首次32并发QA为34通过、1失败：成本测试夹具误将质量费用同时加到原细化分支，
当时未启动任何模型或物理。修正夹具并使用新独占QA目录后35/35通过；零模型dry
通过，旧失败保留。本次未重跑全量pytest，0新WAM/GPU/fresh。两种现有方案均未
通过两个完整短窗，不能直接提升为统一控制候选；原18正确参考、12开发和24最终
留出门槛不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_warning_feedback_20261111/producer/result.json` | `b3530db3cd9babc7f0c8431f59c14cee4809ca60217d17f92741b6361ee5210c` |
| `outputs/task2_warning_feedback_20261111/checker/result.json` | `d86848a947998f07b6763f6a41c1da7ce416a8d6074b3bd6075a3a442943689a` |
| `outputs/task2_warning_feedback_dry_20261111/result.json` | `71aef4bf836face0dfda835dc92d9ed0bf8ff67a41fdb4679c193ebcf20655b2` |
| `outputs/qa/task2_warning_feedback_20261111/result.json` | `914f6d318e7d052e08d5f77d90cc3ce0995635ee5d7a8890577c7f29f2b3badd` |
| `outputs/qa/task2_warning_feedback_20261111_v2/result.json` | `8df851c601874043e6ff58af955a4d573d93e675f7ab4d5dd1f7d1833ed8bc3f` |
| `outputs/task2_warning_feedback_supervisor_20261111/owner_exit.json` | `03a78472c78e80dad3cc8b8abe8603a8e3f98a7d57cdcb4b492e8672141adf7a` |

下一步需要针对跨任务的无witness、模型贴边与实际缓冲失守设计新候选，并在自身
已完成历史下验证，不能通过放宽关节缓冲或只报告较短成功片段绕过失败。
