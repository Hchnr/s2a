# 同warm H12信赖域对照

本轮按[执行前登记](../plans/v0.0.1/prefix_trust_region_control_20261028.md)比较冻结前缀求解器与有限信赖域候选。两组使用同一actual368状态、completed367控制历史及chunk46原时序H12参考；共同初始命令取自已完成独立核验的固定回缩响应`cases.prefix.entries[55]`。不使用GPU。

## 当前状态与证据范围

实现和合成验证正在收尾，两组真实数值搜索及完整独立检查尚未启动。不得将合成测试通过或程序准备就绪作为控制性能证据。实际仍止于原cursor8，task/grasp0；本轮新增物理、世界模型调用和fresh案例均为0。原18条正确参考、12条WAM开发及最终24条留出任务门槛保持，总目标未完成。

候选保留原完整响应、九组约束审计、位置/旋转caps和全轨迹目标；增加每阶段最多5迭代、固定三半径、完整响应验收后重新定中心及逐请求计费记录。普通下降可以在prefix不增长时获接纳；完整H12 witness仍必须通过全部原门槛。此对照比较整套控制流，不能单独归因于半径。

每臂最多4000唯一完整响应、60累计迭代和60阶段；每条响应300数值内步/312查询，另计每臂一次current查询。两臂不重置跨阶段预算，不免费评价warm或差分探针。独立检查重新构造全部唯一响应并核对compact trace；只有initial/selected/closest保存完整轨迹，因此仅对这三种记录比较完整q/v/FK/torque数组，候选接受anchor另逐条比较完整审计。

## 实现与准入

- 求解器：`src/s2a/prefix_trust_region_plan.py`，冻结基线仍为`prefix_preserving_trajectory_plan.py`。
- 来源准入：`src/s2a/prefix_trust_control_evidence.py`，绑定上轮96响应及完整独立检查的成功wait/reap、来源SHA和共同warm bytes。
- 生产、独立检查及进程管理分别为`scripts/probe_tracking_v5_prefix_trust_control.py`、`scripts/check_tracking_v5_prefix_trust_control.py`和`scripts/execute_tracking_v5_prefix_trust_control.py`。
- 三个组件测试分别覆盖求解器验收/差分/预算、独立分类/计费/排序及实际执行入口的重置与计费。首次同版本native QA为59通过/1失败：真实SLSQP合成例在60迭代预算内未达到测试所预期的完整witness。失败凭据保留于`outputs/prefix_trust_native_qa_20261029_0rg1svty/result.json`（SHA`7f6fff11aaf009453d3d1927ad790529535a27584ebd752e05532129674a90e7`），actual wait退出1、9份来源未变；正式搜索未启动，正在核对实现与该测试的预期。

## 结果与后续

待两臂实际退出后记录响应/迭代、原cursor9误差、全H12前缀、witness数、每次接受anchor及活动约束。独立检查未关闭前不授予局部数值资格；即便获得完整模型witness，也须另行登记同初态物理pilot并独立重放，不能据此宣称实际控制或留出任务成功。
