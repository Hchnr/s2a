# step125 因果 TCP 偏差质量搜索

2026-11-11，按[预登记](../../../plans/v0.0.1/tracking/joint_control/causal_innovation_quality_search_20261111.md)，
在同一冻结 step125 状态、已生成参考、已完成计划移位种子和过去关节误差上，
同时要求原模型首点质量门和使用 step124 已完成 TCP 偏差后的质量门通过。步 124
偏差为位置约 0.013266 m、SO(3) 测地角约 0.045020 rad；step125 的实际结果没有
参与搜索。SO(3) 约束使用矩阵复合，解析 Jacobian 已通过非交换旋转中心差分测试。

作为对照，[原质量模型候选](quality_rate_repair_solver_20261111.md)预测首点
0.010000 m / 0.032132 rad；加入上述已完成偏差后为 **0.021585 m / 0.069216 rad**，
未通过 0.01 m / 0.05 rad 门。新的双质量门搜索在同一原关节、制动、box 和速率
条件下，线性约束求解成功，非线性优化经过 42 次迭代、69 次局部模型响应后以
`Positive directional derivative for linesearch` 结束；69 个缓存候选中没有同时
通过全部原门和偏差门的候选，也没有仅需极小速率修正的候选。没有候选，因此
没有执行独立 witness 核验或派发动作。

这是单个移位种子和冻结模型上的有限搜索，不证明全局不可行；原真实 WAM 停止
时的种子未保存，不能把该求解当作现场的精确复现。上一周期偏差是经验补偿，
不是未来误差保证；即使以后找到模型候选，也必须另行独立核验并进行物理对照。
不应因本次未找到候选而放宽原质量、安全或留出门槛。

实际一次模型初始化、一次不推进时间的 forward、一次优化、69 次模型响应；
没有物理积分、WAM/GPU 调用或新初态，子进程正常退出并实际 wait/reap。运行前
32 并发专项 42 项通过；另以真实优化器验证远距离补偿拒绝及原求解器命名空间
不被改写，专项 6 项通过，均无跳过和收集错误。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零工作 dry](../../../outputs/causal_innovation_quality_search_dry_20261111/result.json) | `63788bd440f3d3bca7ff44013d2fccfbda3ce58ef78b7e35c8bfe30944f5b868` |
| [32 并发运行前专项](../../../outputs/qa/causal_innovation_quality_search_20261111/result.json) | `3e3ab270e2a84a23ac78e0b295addbd5647b3a5ddb021d03f215c8d92b341d4e` |
| [真实优化器专项](../../../outputs/qa/causal_innovation_real_optimizer_20261111/result.json) | `94926b68ac733feaa5232726bdeb6eb24944c0767c864490de8cbe2d2930d4d1` |
| [运行登记](../../../outputs/causal_innovation_quality_search_20261111/registration.json) | `84fabd50de94262c11d392a84cbcbe311717cc4de3b10a96cd85d5d48e003f96` |
| [owner 结果](../../../outputs/causal_innovation_quality_search_20261111/result.json) | `2a35b69014cb0c02bd0ace87efed3f89dad4c7f8e66d095000d28a00ad51e3ec` |
| [求解与原候选对照](../../../outputs/causal_innovation_quality_search_20261111/worker/result.json) | `281da18b3013b299b443c47814101ab727b3dd44f9496c3fb2c0a73d3586d042` |
| [进程收尾](../../../outputs/causal_innovation_quality_search_20261111/workers.json) | `4df01c0a315babda9d004078cc95f1dd87e34d26870776c7a60faee3133e3fb0` |
