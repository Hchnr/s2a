# 共享预算余量优先的实际反馈

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_margin_guarded_feedback_20261111.md)，
在同一task2开发初态与seed43下，从raw step32及terminal058 step80执行余量优先、
原双点回退的因果反馈。每次决策两段搜索共享320响应、60次第一阶段/120次总迭代
及最多2次初值LP；回退复用并重新验证同一线性域及初值，不再调用LP。

| 开发起点 | 原guarded | 已完成计划首点回退 | 严格余量 | 本次组合：实际达标 / 执行 | 本次最坏位置 / SO(3) | 停止 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| raw step32 | 1/1 | 1/1 | 0/0 | **1/1** | 0.008680 m / 0.014258 rad | step33前无候选 |
| terminal058 step80 | 4/4 | 5/5 | 4/4 | **4/4** | 0.009208 m / 0.046803 rad | step84前无候选 |

raw首轮余量搜索无候选后，原门回退经独立双点与全时域模型安全审核，派发1条动作；
动作、实际状态、TCP误差和25个2 ms内环样本与原guarded反馈逐值一致。该动作只
通过原双点门，不冒称具有新增模型余量。下一决策两段搜索仍无候选，因此停止。

terminal058四次均优先选择余量候选，并通过独立双点/安全及额外余量复核；4条动作、
实际状态、TCP误差与100个2 ms内环样本和严格余量反馈逐值一致。step84前两段搜索
均无候选，故停止；本次没有叠加已完成计划首点回退。其4周期最坏实际误差低于原
guarded的0.009656 m / 0.049547 rad，但执行长度相同，短于已有首点回退的5周期。
两个开发起点共享初态与seed，不是独立泛化样本；未出现grasp、任务成功或完整8周期。

5个新周期均通过原实际0.01 m / 0.05 rad误差门，125个内环样本均未出现0.05 rad
关节缓冲失守；raw和terminal最小关节余量分别约0.449920和0.142473 rad。单次选中
决策约2.52–3.28秒，仿真等待时物理暂停，不具备20 Hz实时资格。恢复raw的局部行为
及降低terminal最坏误差不等于跨初态、跨任务或WAM闭环任务成功。

## 独立重放与费用

checker未导入候选策略，5条动作、完整状态、125个2 ms内环q/qvel/time、实际TCP
误差、grasp和任务标签逐值一致；两个停止位置及未派发状态一致，未重求无候选原因。
每phase共117控制周期/2925内步、6模型和6个OSMesa context；两phase合计234周期/
5850内步、12模型和12 context。producer有7次原细化/500响应、7次组合决策/629
响应（其中回退202响应）、7次初值LP且回退0次额外LP；4个余量候选和1个原门候选
均经独立审核，共1000端点查询/13000模型转换步。raw首轮58+55=113响应，terminal
停止前69+55=124响应，均在单次320上限内。

四个worker及监督进程均实际wait/reap、exit0、未发信号，输入与源码前后一致。75项
32并发CPU专项、全仓Ruff/格式和零模型dry通过；本轮未重跑全量pytest。无新WAM、GPU
或最终留出样本；`complete/passed`仅表示执行及重放闭合，不表示任务或完整控制资格。
原18正确参考、12开发和24最终留出门槛保持不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_margin_guarded_feedback_20261111/producer/result.json` | `dd064000117f33e8d5342327d1b0751fb0f713708249e2f31bd845ea5669f687` |
| `outputs/task2_margin_guarded_feedback_20261111/checker/result.json` | `6d264b2c938fc396ef624a3e27073a6cb496a50653732f64234e099cd830ff1e` |
| `outputs/task2_margin_guarded_feedback_20261111/producer/workers.json` | `1121ba75aa4ff99e08ceb5552c775e496ba681ed2423a9d2336291ce0b2a4422` |
| `outputs/task2_margin_guarded_feedback_20261111/checker/workers.json` | `4fc007afefd216d5df50f7b6e56d75bfd05f38cd2946372daaa7a3c10d3d8a4c` |
| `outputs/task2_margin_guarded_feedback_dry_20261111/result.json` | `04bb1ab58446750dda5b0703eda57eac2605453b03cb632ee75ed2baba4ad93c` |
| `outputs/qa/task2_margin_guarded_feedback_20261111/result.json` | `adb3f079a12e78ec520c9980c0711ee46ce52e3ae10ad98123edac50c38e298d` |
| `outputs/task2_margin_guarded_feedback_supervisor_20261111/producer_exit.json` | `b2b0e09819789a3d1f954b302aca04ccd03e77426dfcf6bd3dddc253277c73c2` |
| `outputs/task2_margin_guarded_feedback_supervisor_20261111/checker_exit.json` | `b17cce667c5bc23ad57e2cb93016a647d134bcd8b1aad14fc0fbe96c0ab6d9f6` |
