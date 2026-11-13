# 固定 H40 的共同开发对照：前四案例

2026-11-13。按[统一协议](../../../plans/v0.0.1/tracking/references/reset_multitask_fixed_h40_20261113.md)，
在已经独立重放的前四个旧开发案例上，从 reset 执行原首 8 周期，再固定消费各自
step8 实际当前图像生成的 H40；未重新调用世界模型。四个任务/seed 以同一
控制器、分段反馈、质量搜索和关节保护运行，不依据任务结果改变策略。

| 案例 | 完成周期 / H40 消费 | pose 限值通过 | 最小参考 / 实测 opening | 最小内环关节余量 | 目标 grasp / 任务成功 | 停止 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| bowl，seed 42 | 48 / 40 | 48/48 | 0.706206 / 0.766871 | 0.504932 rad | 0 / 0 | H40 完成 |
| bowl，seed 43 | 36 / 28 | 34/36 | 0.946010 / 0.948099 | 0.586177 rad | 0 / 0 | `quality_no_witness` |
| wine，seed 42 | 48 / 40 | 48/48 | 0.972432 / 0.965292 | 0.658262 rad | 0 / 0 | H40 完成 |
| wine，seed 43 | 48 / 40 | 48/48 | 0.953220 / 0.956777 | 0.405828 rad | 0 / 0 | H40 完成 |

四条与对应原自适应分支从 reset 起的前 16 个周期，命令、状态、TCP 误差、
内步关节和夹爪反馈最大差均为 0。之后固定 H40 和重观测分支的参考不同，不能把
差异当成单因素反事实；两种协议中均无抓取或任务成功。固定参考虽然使 bowl/seed42
实际开度下降，但前 48 周期仍未抓取，与此前单案例一致。bowl/seed43 在消费
28/40 个生成目标后停止；其固定 H40 中更远期的闭合意图没有到达派发阶段。

bowl/seed43 的 step 34、35 实测 TCP 位置误差分别为 0.010800 m、0.015806 m，
超过 0.01 m 限值；step 36 因未找到质量 witness 在派发前停止。两次超限由独立
重放复现：当前模型预测质量门并不保证实测误差每次达标，不能将其视作已验证的
实测硬约束。搜索未找到 witness 也不证明真实系统不存在其他可行命令。

四条 producer 共 180 周期 / 4,500 内步；四条独立 checker 从 reset 重放同样
180 周期 / 4,500 内步，并重采 step8 图像。逐周期动作、完整状态、TCP 误差、
内步关节和夹爪反馈最大数值差均为 0，任务与 grasp 标签一致；checker 未加载
候选且不重算质量停止。两相位各使用 12 个仿真模型与 12 个 OSMesa context，
没有新模型初始化、推理请求或去噪，所有 worker 已实际 wait/reap、未发信号。

复用的有限参考、固定 H40 producer/checker 和尾段相关 32 并发回归 20 项通过，
0 skip / 收集错误；新统一入口在四个案例上完成输入预检及实际 producer/checker。
最初六案例的另两条原 producer 仍在运行或等待对应独立恢复，本报告不填补它们的
结果；48 周期不代表 600 周期稳定成功，正确参考正控、最终新初态、实时与真机
验收均未完成。

| 收据 | SHA-256 |
| --- | --- |
| `outputs/qa/multitask_fixed_h40_reuse_20261113/result.json` | `ae4fea67feaee3fecd8c5476b4d8b27c8f56ca0574835d0d6c918209d499ac69` |
| `outputs/reset_multitask_fixed_h40_20261113/preflight_wave_00.json` | `ce22e2ee5b4fcc7daf6db2f9321e13b4756dbbccbd03ad642bcfbdd40db9a637` |
| `outputs/reset_multitask_fixed_h40_20261113/producer_wave_00.json` | `ebe7f03ff22313e99eb856fac8270a9d29d9128e455090fefb4e07a6552a145a` |
| `outputs/reset_multitask_fixed_h40_20261113/checker_wave_00.json` | `77463ea4381e0f388c8591b3f9cf5e35b5d02c19cb03a4784c2ae6c03cc8da4f` |
| `outputs/reset_multitask_fixed_h40_20261113/summary_wave_00.json` | `a64a5e062abac5e82aa9821a7b7d0cdfa3d3627398d021e960a1a6b2f1cdfdd9` |
