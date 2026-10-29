# 长窗口线性可行起点与残差准入（2026-10-28，CPU）

[计划](../plans/v0.0.1/feasible_start_trajectory_20261028.md)保持实际step368、已生成参考及完整原约束，仅在原warm不满足线性约束时选用复核后的LP点。

首版8点witness（2响应），12点改用线性可行LP起点后仍无witness（90响应）。16点LP虽报告success，但复核不满足原1e-10接收门槛而终止，probe/owner实际wait1/reap1，partial结果保留。对该LP命令重新冻结模型rollout得到最小全内环slack −1.38364821822e-8rad，不能接收；制动slack约−1.33e-15、box/rate/terminal无有意义违反。该失败不代表线性约束不可行，也未放宽原门槛。

v2把HiGHS primal/dual求解容差设为1e-10，并明确记录及复核LP残差；不认证的点返回linear_witness_not_certified，不进入非线性搜索。新独立模块及新outputs保留首版失败，不修改旧源码/注册。

| 预览 | 原warm满足线性约束 | 非线性起点 | 最终模型witness | 响应数 |
| --- | --- | --- | --- | ---: |
| 8 | 是 | 原warm | 有 | 2 |
| 12 | 否 | 复核LP点 | 无 | 122 |
| 16 | 否 | 复核LP点 | 无 | 103 |

12/16起点joint/braking最大超限分别1.67e-16/4.44e-16，rate约2.22e-16/1.11e-16，terminal为0/1.11e-16，均通过原准入。搜索后的最好已检查线性可行尝试最大归一化pose超限约1.51562/13.46104；这些只是有限局部搜索结果，不是全局下界。现在16点搜索确有线性可行尝试，但仍无完整pose witness，不能通过更多窗口/起点比较宣称控制成功。

v2三项共227响应、actual wait0/reap0，`outputs/tracking_v5_feasible_start_trajectory_numeric_v2_20261028/verified.json` SHA `063ca5370c69c77f1ac53f548219cc63a69192b502b7d5655626e665f07aa575`。相关6项CPU测试通过（4.83s），Ruff/format通过。新增物理/模型/fresh均0，原全部门槛和GPU禁用保持。

进一步[独立XML一致性检查](preview_fk_consistency_20261028.md)发现连被接收8点解的Cartesian/joint预测也存在明显几何差异。下一步先解决预测内部的运动学一致性，再评价预览长短；不是继续任意重启或放宽控制精度。
