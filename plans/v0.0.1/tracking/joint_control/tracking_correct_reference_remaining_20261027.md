# 原正确参考剩余17条候选回归

总目标及原18/18资格门槛不变。本次执行基于首条完整基线复现和首条候选独立动作回放已通过的证据，继续原case01..17。配置、数据、初态、控制接口、seed42、40步参考/每8步更新和原稳定40/位置0.01m/旋转0.05rad指标保持。case00不重复采集。

执行入口：`scripts/execute_tracking_correct_reference_remaining.py`。每个案例先运行已冻结的 `run_tracking_correct_reference.py --policy candidate`，再通过已冻结的 `check_tracking_correct_reference_v3.py` 从同一初态独立重放实际action。候选4743周期，独立回放4743周期，世界模型请求0、新初态0。全程CPU/OSMesa，CUDA与LD为空，三个CPU线程变量为8。不修改已登记来源；不覆盖产物、不自动重试。数值或执行错误终止并保留失败；性能失败单列但继续固定矩阵。

注册文件：`outputs/tracking_correct_reference_remaining_20261027/registration.json`。每案例 `caseXX/closed.json` 只在两阶段均正常完成并读取独立结果后写入；正常wait/reap由stage的exit/release凭据给出。最终result.json与原case00独立结果合并，候选18条总物理4907周期与回放4907周期分开报告。

本检查范围为原生动作回放和原性能门槛，仍不等同于规划器最优性/动态约束证书、规划全部输入因果性、图像/标签完整资格或WAM闭环成功。未完成项必须补齐，不能仅凭汇总performance字段开启WAM/fresh。原WAM开发和最终24条新初态门槛不变。
