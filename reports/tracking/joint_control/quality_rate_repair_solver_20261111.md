# 可注入质量求解器的自动数值修正

2026-11-11，按[预登记](../../../plans/v0.0.1/tracking/joint_control/quality_rate_repair_solver_20261111.md)，
新增可注入的研究型求解器：原求解器已有 witness 时直接返回；原求解没有 witness
时，仅对其成功优化且已缓存、只违反动作变化率的最终点尝试极小的 box/速率修正，
并重新计算局部轨迹与原约束。修正使响应总数仍受 160 上限约束；其他失败继续
返回无 witness。调用方仍必须在任何动作派发前独立核验。原在线默认路径没有更改。

在同一个 stop125 实测状态、已生成参考、过去内环误差和已完成计划移位种子上，
该封装自行完成原搜索 **58** 次模型响应及 **1** 次修正重算，随后独立审核通过
全部原质量、box、变化率、关节和制动检查。修正后的最大命令变化为 `4.96e-10`，
最大变化率为 `0.2499999999`；与[先前单候选修正](cached_live_stop125_rate_repair_20261111.md)
逐元素一致，先前候选仅用于求解后的比较，未作为求解器输入。

首点预测位置误差 `0.010000000000154` m、旋转误差 `0.0321320684991` rad；
位置仅在原 `1e-10` m 数值容差内通过，没有实质位置裕量。独立关节响应与预览
最大差 `2.78e-17`，最小收紧关节余量约 0.715421 rad。结果只证明一个已研究
停止状态的冻结模型候选；真实 WAM 原失败种子未保存，没有新物理动作、实时、
完整轨迹或留出资格。

实际一次模型初始化、一次静态 forward、一次求解及合计 59 次模型响应；独立
审核使用 200 端点查询、2,600 模型内转换步。未执行物理积分、WAM/GPU 调用或
新初态，工作进程正常退出并实际 wait/reap；运行前 32 并发专项 48 项通过、零
跳过和收集错误。下一阶段需要在不改变原阈值的前提下，将可注入候选生成接入
独立物理反馈与动作重放，再决定是否适用于更广泛开发案例。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零工作 dry](../../../outputs/quality_rate_repair_solver_dry_20261111/result.json) | `ee0fe3108d14d59d67380e8d0067193a40d73669af0a1bfe5006f3a944f0a572` |
| [32 并发专项](../../../outputs/qa/quality_rate_repair_solver_20261111/result.json) | `6491259540dabac24d48fbfd151ef685e405f824a76adc11f29c63293910d44d` |
| [运行登记](../../../outputs/quality_rate_repair_solver_20261111/registration.json) | `4ab69f4c50c9bebb041feea360e5ccfa530dba8a2858a30fc1865a7e6ea0b5d0` |
| [owner 结果](../../../outputs/quality_rate_repair_solver_20261111/result.json) | `5bef113e513a87c0a9b5f1424120841150285bb85ad0661307fdce619afd03d8` |
| [自动求解与独立核验](../../../outputs/quality_rate_repair_solver_20261111/worker/result.json) | `fb9824cba9ce18fc4acb52bcb25c336d8fa173522c3d27be66bc4acbfe7d9a26` |
| [进程收尾](../../../outputs/quality_rate_repair_solver_20261111/workers.json) | `6083b09e1d5e45dbf6dad96bce7fba3b52ff3794809dc26bd5131081fd693aff` |
