# 完整轨迹搜索研究

这里保存同一固定控制接口下的八个对照实现，不属于默认adapter：

| 实现 | 所研究的变化 |
| --- | --- |
| constraint_residual_repair | 原始约束残差修正 |
| interior_trajectory_search | 内点恢复与完整轨迹信赖域 |
| rejected_response_search | 安全拒绝响应的因果模型反馈 |
| global_envelope_search | 固定初始全局误差上限 |
| stage_budget_search | 有限阶段预算对照 |
| state_vector_gn_search | 归一化TCP误差向量GN目标 |
| state_vector_objective_scale | 整个QP目标统一缩放 |
| state_vector_dual_recovery | 固定紧活动子集上的单次乘子恢复 |

各实现的资格以各自预登记和完整前向评价为准。文件整理保持原数学、预算、安全
及执行caps；不据此选择默认胜者。

[实验入口](../../../../scripts/tracking/trajectory_search/README.md) ·
[独立审核与回归](../../../../tests/research/trajectory_search/README.md) ·
[当前目标与未完成项](../../../../docs/status.md)
