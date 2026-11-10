# 中断的非线性投影：安全恢复但整段跟踪退化

2026-11-10。原投影进程已不存在，保留236条连续生产响应，但最终结果、
求解器账本与退出收据缺失。不能重启原实验、推断原solver终态，或把本次
补充审核写成原实验完整通过。中断时可能有未落盘的部分调用，完整费用未知。

后验读取236条保存记录，核对原源码快照、命令唯一性、完整约束标签、全部
位置/姿态误差与prefix标签的内部自洽。生产侧有13条同时满足原joint audit
与1e-8 safety buffer，完整witness为0。这只是筛选，不是全部236条独立复算。

按[补充复核计划](../../../plans/v0.0.1/tracking/trajectory_search/interrupted_projection_candidate_replay_20261110.md)，
独立native实际复算固定的不安全对照、全部13条生产侧安全候选及一条重复，
共15响应；原严格响应/力矩容差及布尔分类保持，完整响应逐值exact。
13条均经独立复算满足原buffer，但没有完整witness，也没有一条24项误差
逐项支配已独立审核的安全基线。其余223条没有在本次做独立native复算。

## 完整轨迹比较

基线是同一实测状态、历史与参考的small-progress安全anchor23；它来自已关闭
独立审核，不称本次重跑。候选110是13条中最小最大归一化误差的后验选择，
不是已完成solver账本的selected，也不是可执行轨迹。

| 指标 | 已审核安全基线 | 候选110独立复算 |
| --- | ---: | ---: |
| 原位置/旋转caps | 12/24 | 10/24 |
| 连续达标prefix | 2 | 2 |
| 24项最大归一化平方误差 | 6.8730 | 24.8508 |
| cap-distance loss | 2.00889 | 21.70822 |
| 终点位置误差 | 16.90 mm | 49.85 mm |
| 终点旋转误差 | .13108 rad | .09571 rad |
| 全部原caps witness | 0 | 0 |

13条独立确认的安全候选全部为10/24、prefix2。局部恢复确实保住了该组的
模型安全余量，但改善终点旋转时，终点位置和整段最差误差明显恶化。
该结果来自保存状态上的模型响应，不是物理仿真、抓取或任务成功。不能仅凭
候选恢复安全或旋转误差下降而进入实际控制。

原因层面：本轮投影目标是接近固定不安全命令并恢复安全，并未显式保护整段
位置/姿态误差；但中断后缺少最终solver账本，不能推断完整迭代因果或宣称
替代方法一定成功。下一阶段应从已审核安全基线出发，保护已达标点并限制
未达标项继续变差，直接优化整段误差，同时用实际响应审核安全和原24caps。

## 成本与来源

本次新增15个独立完整响应/4500数值内步/4681查询/1model；0新生产响应、
显式QP/NNLS、物理/GPU/WAM/fresh/最终留出/实际参考消费。原运行已保存236
条完整生产响应，不能将这与未知中断部分调用合并成精确总费用。
46项jobs32专项全部通过，Ruff/format通过；所有worker实际回收。source/input
严格核验，原数值和控制源码未变；owner session97765退出0、实际wait/reap，
PID1270797不存在。原中断结果未覆盖或追认为完成。

以下路径相对outputs：

| 产物 | 路径 | SHA256 |
| --- | --- | --- |
| 236条保存记录后验筛查 | nonlinear_projection_interrupted_assessment_20261110/assessment.json | 4e1151091730a4da382a101a7be8442f7fb700a95e1ca9e17ad33e093f48da93 |
| 15条独立响应复核 | interrupted_projection_candidate_replay_20261110/result.json | 70968a498c1f7488368c44f01c12252ea124a2cd9a1cfd3b6db568676a1612e9 |
| owner终态 | interrupted_projection_candidate_replay_20261110/owner_observation.json | 03ea9d4e29b690e3c42f181e22a4895cf178b900ab2a4c6fd20d955febbd65d7 |
| 46项QA | qa/interrupted_projection_candidate_replay_20261110/result.json | 13edc768cffa64076e6a777d590da67c86e9130924e7b5eca1ebd4d33e7a9be3 |

实际控制仍cursor8/task0/grasp0；原18条正确参考、12条WAM开发及24条最终
留出门槛保持，总目标未完成。GPU继续禁用。
