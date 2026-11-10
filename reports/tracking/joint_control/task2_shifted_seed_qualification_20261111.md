# 停止状态后移计划的双点资格

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_shifted_seed_qualification_20261111.md)，
分别以已独立重放的task2 raw step33和terminal058 step84实际停止状态为起点，后移
该分支最后一次已派发的八步计划并重复末项；固定当前状态、已完成误差及已生成参考。
两条计划均未重新优化，也没有派发。

| 开发停止状态 | 首点位置 / SO(3) | 第二点位置 / SO(3) | 前两点原阈值 | 后六点最坏归一化缺口 | 模型安全审核 |
| --- | --- | --- | --- | ---: | --- |
| raw step33 | 0.010372 m / 0.052042 rad | 0.013141 m / 0.078789 rad | 均不通过 | 1.161716 | 通过 |
| terminal058 step84 | 0.009502 m / 0.046570 rad | 0.013043 m / 0.066521 rad | 首点通过，第二点不通过 | 1.495111 | 通过 |

原阈值为位置≤0.01 m、SO(3)≤0.05 rad，沿用`1e-10`审核容差。terminal058之前
只知道重锚后首点达标；本次第二点两个误差均超限，因此该后移计划**不是双点模型
候选**。raw的两点也都不合格。后六点归一化缺口以位置和姿态各自除以原阈值、减1
后取最大值并截断到非负；两条计划完整八点中仅terminal058首点达标。

两条计划的box、动作变化率、内环关节缓冲和0.1 s制动约束经独立标量响应复核；
终端方向来自后移计划，两个状态均无终端方向行。边界浮点误差在原容差内。这些
是冻结模型内约束，不证明真实动作安全或物理可达；也不等同于原停止求解使用的
细化种子。未搜索其他计划，因而不能由该负结果断定停止状态全局无双点候选。

## 复核与费用

首点误差与[先前模型偏差分析](task2_guarded_model_drift_20261111.md)一致；两次
独立审核共重建400个内环端点/5200个模型转换步。关节轨迹最大差`2.78e-17 rad`，
TCP位姿重建最大差`8.89e-16`。实际消耗2个静态模型、2次`mj_forward`、2次八步
rollout及2次独立审核；2个worker和监督进程均实际wait/reap、exit0、未发信号，
源码与输入前后一致。

47项32并发CPU专项、全仓Ruff/格式及零模型dry通过；本轮无新物理积分、新WAM、
GPU或最终留出样本。`complete/passed`仅表示检查和审核闭合，不表示质量候选、控制
或任务验收通过。可在另行登记后比较以后移计划为搜索初值是否找到其他合格计划，
或探索更早的因果参考衔接；当前证据不支持直接派发后移计划。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_shifted_seed_qualification_20261111/result.json` | `f5af0996b2efced760615c19dcbe143e26b1a2b18a020ff4fea7731e3fcd1279` |
| `outputs/task2_shifted_seed_qualification_20261111/workers.json` | `7a0a653efaeb3b1c609c066c091b233e30b724d32e6a3a97b03ec6683e2138c9` |
| `outputs/task2_shifted_seed_qualification_dry_20261111/result.json` | `d4826d3c869dd12aa147fc9db155369e16453e09714c7457ce2d837e557b76e9` |
| `outputs/qa/task2_shifted_seed_qualification_20261111/result.json` | `4e2a01d916dc039962d8b382615a9848405e11846a9a8a08db33874774ca9e35` |
| `outputs/task2_shifted_seed_qualification_supervisor_20261111/owner_exit.json` | `221126588617f68e904695889dc40182d033f431b8a41cffc44807e95a15bfbb` |
