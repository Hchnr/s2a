# 研究计划：plans/v0.0.1/tracking/response_models

本页导航原始文件；正文、配置值和历史身份保持。目录迁移不改变实验或控制资格。

[当前状态](../../../../docs/status.md) · [研究索引](../../../../docs/research_index.md) · [路径与身份](../../../../docs/migrations/repository_layout_20261029.json)

| 领域 | 文件数 |
| --- | ---: |
| [causal](causal/index.md) | 9 |
| [diagnostics](diagnostics/index.md) | 9 |
| [forward](forward/index.md) | 4 |
| [identification](identification/index.md) | 22 |
| [inverse](inverse/index.md) | 6 |
| [nonlinear](nonlinear/index.md) | 1 |
| [wrench](wrench/index.md) | 3 |

| 文件 | 原声明路径 |
| --- | --- |
| [三任务因果 TCP 偏差回顾](multitask_causal_tcp_innovation_20261113.md) | 新增计划 |
| [038：有限真实 clean 残差训练与开发任务验证 — 2026-10-23](038_clean_residual_training_20261023.md) | `plans/v0.0.1/038_clean_residual_training_20261023.md` |
| [073 当前motor保持先验的单次残差对照](073_residual_motor_prior_20261025.md) | `plans/v0.0.1/073_residual_motor_prior_20261025.md` |
| [074 统一残差夹爪逆模型的正确参考物理资格](074_residual_motor_physical_qualification_20261025.md) | `plans/v0.0.1/074_residual_motor_physical_qualification_20261025.md` |
| [079 新响应数据的隔离验证与速度状态残差前向模型](079_velocity_residual_response_validation_20261025.md) | `plans/v0.0.1/079_velocity_residual_response_validation_20261025.md` |
| [089 固定成对差分监督的条件化arm前向模型](089_contrast_conditioned_arm_forward_20261025.md) | `plans/v0.0.1/089_contrast_conditioned_arm_forward_20261025.md` |
| [097 独立训练轨迹的首次闭合响应覆盖](097_training_response_coverage_20261025.md) | `plans/v0.0.1/097_training_response_coverage_20261025.md` |
| [098 增加独立训练干预覆盖的forward对照（待097完整资格）](098_coverage_conditioned_arm_forward_20261025.md) | `plans/v0.0.1/098_coverage_conditioned_arm_forward_20261025.md` |
| [101 干预斜率与自然轨迹偏置分阶段辨识](101_separated_response_identification_20261025.md) | `plans/v0.0.1/101_separated_response_identification_20261025.md` |
| [102 当前力/力矩反馈合同与baseline重放](102_current_wrench_sensor_contract_20261025.md) | `plans/v0.0.1/102_current_wrench_sensor_contract_20261025.md` |
| [103 原30轨迹的当前wrench数据](103_native_wrench_data_20261025.md) | `plans/v0.0.1/103_native_wrench_data_20261025.md` |
| [106 按fit当前wrench覆盖选择独立追加干预](106_wrench_coverage_interventions_20261025.md) | `plans/v0.0.1/106_wrench_coverage_interventions_20261025.md` |
| [107 固定模型的新增干预覆盖对照](107_wrench_coverage_forward_20261025.md) | `plans/v0.0.1/107_wrench_coverage_forward_20261025.md` |
| [108 训练响应残差只读审查](108_wrench_training_residual_audit_20261025.md) | `plans/v0.0.1/108_wrench_training_residual_audit_20261025.md` |
| [110 剩余失败只读归因](110_joint_wrench_failure_attribution_20261025.md) | `plans/v0.0.1/110_joint_wrench_failure_attribution_20261025.md` |
| [113 联合条件的干预斜率/自然偏置分阶段辨识](113_separated_joint_response_20261025.md) | `plans/v0.0.1/113_separated_joint_response_20261025.md` |
| [115 按完整条件特征行空间增量选择训练干预](115_condition_span_interventions_20261025.md) | `plans/v0.0.1/115_condition_span_interventions_20261025.md` |
| [116 固定109模型的条件方向覆盖数据对照](116_condition_span_forward_20261025.md) | `plans/v0.0.1/116_condition_span_forward_20261025.md` |
| [117 条件覆盖前后局部响应误差归因](117_condition_span_error_attribution_20261025.md) | `plans/v0.0.1/117_condition_span_error_attribution_20261025.md` |
| [122 时间残差训练方程](122_temporal_response_equations_20261025.md) | `plans/v0.0.1/122_temporal_response_equations_20261025.md` |
| [123 固定时间残差候选拟合](123_temporal_response_fit_20261025.md) | `plans/v0.0.1/123_temporal_response_fit_20261025.md` |
| [124 时间残差候选物理输出验证](124_temporal_response_validation_20261025.md) | `plans/v0.0.1/124_temporal_response_validation_20261025.md` |
| [125 追加条件覆盖的训练侧局部泛化](125_expanded_local_response_leaveout_20261025.md) | `plans/v0.0.1/125_expanded_local_response_leaveout_20261025.md` |
| [126 按物理组平衡的局部响应条件](126_group_balanced_response_leaveout_20261025.md) | `plans/v0.0.1/126_group_balanced_response_leaveout_20261025.md` |
| [131 腕部传感器与已知接触拓扑](131_wrench_contact_topology_20261025.md) | `plans/v0.0.1/131_wrench_contact_topology_20261025.md` |
| [166 待实现：训练支持约束下的逆模型使用](166_support_aware_inverse_control_20261025.md) | `plans/v0.0.1/166_support_aware_inverse_control_20261025.md` |
| [167 支持范围门控的物理执行与独立审查接入](167_support_aware_control_binding_20261025.md) | `plans/v0.0.1/167_support_aware_control_binding_20261025.md` |
| [174 实际命令残差的独立重建](174_executed_residual_replay_20261025.md) | `plans/v0.0.1/174_executed_residual_replay_20261025.md` |
| [175 待实现：保留因果反馈的支持范围组合对照](175_residual_supported_control_candidate_20261025.md) | `plans/v0.0.1/175_residual_supported_control_candidate_20261025.md` |
| [176 半权重学习与因果残差反馈的完整18条对照](176_residual_supported_execution_20261025.md) | `plans/v0.0.1/176_residual_supported_execution_20261025.md` |
| [208 待执行：直接平移逆的独立辨识对照](208_direct_arm_inverse_identification_20261026.md) | `plans/v0.0.1/208_direct_arm_inverse_identification_20261026.md` |
| [209 直接逆数据独立核验](209_direct_arm_inverse_data_verification_20261026.md) | `plans/v0.0.1/209_direct_arm_inverse_data_verification_20261026.md` |
| [210 固定直接逆残差拟合及动作诊断](210_direct_arm_inverse_fit_20261026.md) | `plans/v0.0.1/210_direct_arm_inverse_fit_20261026.md` |
| [213 直接逆真实控制接入与独立审查](213_direct_arm_inverse_control_20261026.md) | `plans/v0.0.1/213_direct_arm_inverse_control_20261026.md` |
| [冻结与状态更新机器人动力学诊断（执行前登记，CPU）](state_updated_joint_response_20261028.md) | `plans/v0.0.1/state_updated_joint_response_20261028.md` |

历史正文中的相对链接按原声明目录解释；不要改写原文或把旧运行登记当作新布局运行资格。
