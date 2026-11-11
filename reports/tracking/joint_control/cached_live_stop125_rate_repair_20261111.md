# step125 静态候选的动作变化率修正

2026-11-11，按[登记](../../../plans/v0.0.1/tracking/joint_control/cached_live_stop125_rate_repair_20261111.md)，
只取[缓存审核](cached_live_stop125_constraint_audit_20261111.md)记录的优化器最终
候选，在原 box 和 `0.25 - 1e-10` 变化率范围内逐周期裁剪，最大命令改变量约
`4.96e-10`，低于预登记 `1e-8` 上限。没有调整原候选的质量、关节或制动阈值。

修正后以同一当前状态、已生成参考、控制器和过去内环误差重算一次冻结局部模型，
并用另一套逐内环响应独立核验：**通过原 box、变化率、关节、制动和首点质量
检查，得到一个模型 witness**。它的最大动作变化为 0.2499999999，最小收紧关节
余量约 0.715421 rad，制动约束亦通过；独立关节重算与预览的最大差约 `2.78e-17`。

预测第一周期位置误差为 `0.010000000000154` m，旋转误差为 `0.0321320684991`
rad。位置比名义 0.01 m 高约 `1.54e-13` m，处在原检查的 `1e-10` m 数值容差内；
不能把这个边界候选描述为具有实质位置裕量。该模型不含真实接触和时变动力学，
验证仅针对已保存的单个静态候选，原真实 WAM 停止时的种子未保存；没有派发动作，
尚无物理反馈或稳定成功证据。

本次实际一个模型初始化、一次静态 forward、一次预览响应、零优化和一次独立
审核（200 端点查询、2,600 模型内转换步）；没有物理积分、WAM/GPU 调用或新
初态。工作进程正常退出并实际 wait/reap。运行前 32 并发专项 38 项通过，零
跳过和收集错误；最终必须另行评估在线候选生成和独立物理执行与重放。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零工作 dry](../../../outputs/cached_live_stop125_rate_repair_dry_20261111/result.json) | `1171fc88b9e0b0c2ec0c36a978dfe2f32466c34d1feffe4e2dbf0af62ab94805` |
| [32 并发专项](../../../outputs/qa/cached_live_stop125_rate_repair_20261111/result.json) | `2656b308a2fd0d565739b0eb56fd0a5a9535e5f7996b451d3237345812f5d0b5` |
| [运行登记](../../../outputs/cached_live_stop125_rate_repair_20261111/registration.json) | `a3f887c22e28921915f7224129cd89f15f6473d3b5174e83e2b225774970c848` |
| [owner 结果](../../../outputs/cached_live_stop125_rate_repair_20261111/result.json) | `691f7bf8fdd3eb35b447f19fe0990acc5e28b9d53b24cb7b9dc453654fc1cd2c` |
| [修正与独立核验](../../../outputs/cached_live_stop125_rate_repair_20261111/worker/result.json) | `dc3b04563eec8cd97f46747507fcc5e609406642e61bf4dada6176453c7cb744` |
| [进程收尾](../../../outputs/cached_live_stop125_rate_repair_20261111/workers.json) | `5120eed20795badb9cf5b3b410b7ae5e200aacb33da665104e5ff7c4d901e08f` |
