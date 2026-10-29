# 全预览端点约束的更早状态检查（2026-10-28，CPU）

按[计划](../plans/v0.0.1/trajectory_capped_joint_plan_20261028.md)，在已独立复核的causal-error真实分支五个状态上分别做首点/完整8点精度约束搜索。每个状态使用自身当时实测M/J/q/qvel、过去已完成period的独立审计profile、对应原计划warm start和已生成chunk46。XML独立验证q/qvel和SITE速度合同，新增候选未执行，不能串接五个条件搜索为新rollout。

| 原实际step | 当前cursor | 首点策略 | 完整8点策略 | 首点/完整响应数 |
| --- | ---: | --- | --- | ---: |
| 368 | 0 | witness | witness | 2 / 2 |
| 370 | 2 | witness | 未找到witness，迭代上限 | 5 / 149 |
| 372 | 4 | witness | 未找到witness，linesearch退出 | 2 / 129 |
| 374 | 6 | witness | 未找到witness，响应预算耗尽 | 14 / 160 |
| 376 | 8 | witness | 未找到witness，linesearch退出 | 2 / 117 |

完整约束策略在step370/372/374/376的最好已检查线性可行尝试，最大归一化pose超限分别.444255734、1.999864392、3.611237469、5.008330648；它们不是全局下界或非线性不可行证明。此结果揭示原首点接收合同不能保证未来端点精度；新候选只是更早拒绝不满足完整预览的模型计划，尚未展示更好的实际控制或任务成功。

原固定8点窗口在step368仅包含cursor0..7，到step370才包含关键cursor9；但chunk46的40参考点在step368已经生成可用。可在不增加模型查询、不借未来观测的前提下进一步检查12/16点预览，判断是否能从更早实际状态形成满足完整精度与原关节约束的命令。不能将“更长预览”预先称为有效方案，仍需数值与固定接口实际前向验证。

10次条件搜索共582响应，owner actual wait0/reap0；`outputs/tracking_v5_trajectory_caps_numeric_20261028/verified.json` SHA `412858e83ccf920b476b1c02308e11a891ac68a5b8d6542659f7f7f343d133ba`。新增物理/世界模型/fresh均0。新模块及原首点模块4项CPU测试通过（5.14s），误差管/IK/XML FK另9项通过（5.29s）；新入口Ruff/format通过。未重复全量回归，不宣称全量通过。

GPU持续禁用，原18条4907步/完整12WAM开发/final24新初态任务门槛不变，尚无总目标完成证据。
