# 质量求解器对照的首次QA准入失败

2026-11-11。首次为[固定状态对照](../../../plans/v0.0.1/tracking/joint_control/batched_quality_solver_20261111.md)
加入可选展开接口时，修改了迁移清单中固定SHA的原质量求解器及其历史测试。统一CPU
QA入口在建立源码快照阶段报 `Current physical source SHA drift` 并停止；没有收集或
执行测试，也没有调用求解器、物理或WAM。本次失败收据保留为
`outputs/qa/batched_quality_solver_20261111/result.json`，SHA256为
`988ed2df20b9fee74d87ff2aed08c72cdbb79a47892b21ca4664bb32ebd7acc6`。

原求解器和历史测试已恢复到改动前字节，当前迁移身份清单重新通过。后续在独立
候选入口绑定原函数代码对象及私有展开依赖，不修改原模块全局变量或默认路径；
必须分别验证原代码对象、原全局依赖和异常恢复，并保留原独立witness审核。新的
CPU QA使用独占 `outputs/qa/batched_quality_solver_20261111_v2`；原失败收据不覆盖，
实验输入、预算、安全阈值及不自动重试约束不变。
