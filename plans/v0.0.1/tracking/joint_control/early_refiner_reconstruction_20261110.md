# 首次预警起点的原约束细化重建

2026-11-10。两条开发轨迹的首次0.10 rad警报处，简化preview及Jacobian退避均在实际短窗越界，而历史分支8周期内保持原0.05 rad缓冲，见[结果](../../../reports/tracking/joint_control/early_warning_tracking_20261110.md)。原记录显示预览之后还有约束细化和已完成历史；本阶段先验证能否从实时仿真端点及过去已完成信息重算该步骤，再决定真实反馈实验，不把历史后续命令作为新控制输入。

## 因果重建

固定原两案例、seed43、起点端点43和52及此前已生成的chunk5/cursor3和chunk6/cursor4。复用已核身份的原前缀、原控制器合同、资产和软件OSMesa环境，从原初态执行43/52条已保存命令，逐端点核对完整状态与已保存控制缓存。在各自首警报端点读取实际q/qvel、mass/J、site和控制器rotation goal；不推进后续物理、不执行新动作。

原step `t-1`已完成记录仅提取实际上一动作、预览耦合模型 `innovation_after`、细化历史 `tracking_history_after`、上一已完成8点计划及完成周期计数。验证该计划首项等于上一真实动作，并用当前实时site/twist初始化两个历史的expected endpoint；pending为空。保留当前运行时实际controller.initial_joint，不声称历史initial_joint已单独保存。使用当前已生成chunk与上一动作，在原接口、原坐标变换下依次调用 `dependency_safe_preview_runtime.solve_preview` 和 `TrackingPreviewTransaction.prepare`；后者保留原0.05 rad预测内环约束、终端条件及原分支决策，不调用commit或派发动作。

动作计算完成后，才解码原step `t`记录，比较原实时mass/J/q/qvel、site/twist、实际可用创新及细化前后命令、计划和route。比较是只读评估，原step `t`的动作和后续状态不得传入计算。以各项最大差和固定 `atol=1e-8, rtol=0` 给出是否重现；差异或无法重现必须保留原值，不以求解器成功等同于重现。内点route的首点精度原先并非硬约束，需如实记录其原模型预测和实际质量，不把模型witness当真实控制通过。

## 边界、预算和后续

两独立CPU worker各1次preview、1次原细化，原算法每阶段最多160响应，单案例最坏三阶段合计≤480个优化响应；不加大迭代或自动重试。合计最多95已保存控制周期/2375个2ms积分步、6模型/6软件context；重建计算阶段禁止调用环境step，记录计算前后sim state/控制与传感器不变。0新增动作、WAM/GPU/fresh；独占目录、空CUDA/LD、单线程数值库、真实wait/reap与失败/部分费用登记。源码、输入、资产SHA执行前后核对。

先测试上一完成计划和创新的恢复、篡改拒绝、禁止未来字段传入及历史状态重置；32并发CPU专项与零物理步dry通过后冻结来源。两点若重现仍只是一个规划周期的离线重建；必须另登记分支自有历史的实际闭环、原安全与跟踪质量、独立动作重放，才可评估控制改进。原18正确参考、12 WAM开发和最终24留出门槛保持。
