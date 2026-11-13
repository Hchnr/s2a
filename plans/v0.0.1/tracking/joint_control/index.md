# 研究计划：plans/v0.0.1/tracking/joint_control

近期：[因果 TCP 补偿余量对照](causal_innovation_margin_20261113.md)

近期：[已重放停止状态的首点优先求解](replayed_first_pose_search_20261112.md)

近期：[已重放参考的首点位姿外界](replayed_reference_pose_bounds_20261112.md) · [软件库来源绑定修复](replayed_reference_pose_bounds_recovery_20261112.md)

近期：[三任务夹爪反馈独立重放修复](cached_calibration_replay_repair_20261111.md) · [三任务夹爪物理反馈](cached_calibration_gripper_feedback_20261111.md) · [原质量门外的模型安全有界跟踪回退](model_safe_tracking_fallback_20261111.md)

近期：[有限释放保持后的连续衔接反馈](bounded_release_bridge_feedback_20261111.md)

近期：[有限释放保持的匹配物理反馈](bounded_release_hold_feedback_20261111.md)

近期：[冲突释放参考的有限保持与衔接](bounded_release_revision_20261111.md)

近期：[跨任务重规划参考下的共享预算三级反馈](replan_shared_shift_feedback_20261111.md)

近期：[共享预算双点优先及已完成计划首点回退](task2_shared_shift_feedback_20261111.md)

近期：[组合反馈停止处的已完成计划资格](task2_shared_shift_qualification_20261111.md)

近期：[共享预算余量优先的实际反馈](task2_margin_guarded_feedback_20261111.md)

近期：[共享预算的余量优先与原双点回退](margin_guarded_fallback_20261111.md)

近期：[双点模型余量的实际反馈对照](task2_margin_two_step_feedback_20261111.md)

近期：[双点模型余量的预停止对照](margin_two_step_search_20261111.md)

近期：[terminal058 step85的首点优先搜索](task2_terminal85_first_search_20261111.md)

近期：[因果回退停止状态的首点位姿外界](task2_fallback_stop_pose_bound_20261111.md)

近期：[已评估双点搜索中的首点候选](task2_cached_first_pose_20261111.md)

近期：[双点优先、已完成计划后移的因果反馈](task2_causal_shift_fallback_20261111.md)

近期：[相同停止状态的双种子双点搜索](task2_shifted_seed_search_20261111.md)

近期：[停止状态的后移计划双点资格检查](task2_shifted_seed_qualification_20261111.md)

近期：[已执行双点计划的状态与模型偏差](task2_guarded_model_drift_20261111.md)

近期：[数值余量双点搜索的实际反馈](task2_guarded_two_step_feedback_20261111.md)

近期：[双点可行性搜索的数值尺度对照](guarded_two_step_search_20261111.md)

近期：[第三个任务的双点硬约束实际反馈](task2_two_step_feedback_20261111.md)

近期：[提前一拍的双点质量可行性](two_step_viability_20261111.md)

近期：[三任务停止状态的位姿与安全联合外界](stopped_pose_safety_bound_20261111.md)

近期：[安全可行初值上的首点优先恢复](task2_staged_first_feasibility_20261111.md)

近期：[原安全约束内的可行初值恢复反馈](task2_feasible_start_restoration_20261111.md)

近期：[保持首点硬约束的多点恢复反馈](task2_first_hard_restoration_20261111.md)

近期：[第三个任务的提前多点质量恢复反馈](task2_earlier_restoration_feedback_20261111.md)

近期：[第三个任务新参考生成时的前瞻反馈](task2_earlier_preview_feedback_20261111.md)

近期：[三任务首周期位置可达域外界](first_step_position_bound_20261111.md)

近期：[三任务停止/预警状态的可行性优先搜索对照](multitask_feasibility_first_20261111.md)

近期：[第三任务早期预警的安全与跟踪反馈](task2_warning_feedback_20261111.md)

近期：[第三任务的早期关节预警前缀资格](task2_warning_prefix_20261111.md)

近期：[四次打开事件的安全筛查与匹配反馈](causal_opening_event_feedback_20261111.md)

近期：[批量展开接入质量求解的固定状态对照](batched_quality_solver_20261111.md)

近期：[自由关节预览内环的等价批量计算](batched_joint_rollout_20261111.md)

近期：[质量约束求解的固定状态CPU耗时剖析](quality_solver_runtime_profile_20261111.md)

近期：[两因素对照的独占目录恢复](causal_event_factorial_recovery_20261111.md)

近期：[因果衔接中位姿与开度的固定两因素对照](causal_event_bridge_factorial_20261110.md)

近期：[已生成新旧参考的八周期因果衔接](causal_event_bridge_feedback_20261110.md)

近期：[闭合后切回已生成的新参考](causal_event_switch_feedback_20261110.md)

近期：[闭合事件记忆的16周期固定参考对照](causal_event_long_feedback_20261110.md)

近期：[旧/新闭合参考的匹配状态实际反馈对照](causal_event_feedback_20261110.md)

近期：[闭合机会的夹爪历史重建](causal_gripper_history_20261110.md)

近期：[两处闭合影子机会的原生命令前缀](causal_event_prefix_replay_20261110.md)

近期：[跨重规划因果事件记忆](causal_event_memory_20261110.md) · [影子回放独立复核](causal_event_memory_check_20261110.md)

近期：[开发参考阶段一致性与独立复核](replan_task_event_stability_20261110.md) · [逐点复核](replan_task_event_stability_check_20261110.md)

近期：[新参考重规划的安全优先质量恢复反馈](replan_restoration_feedback_20261110.md)

近期：[新参考重规划起点的八点约束实际反馈](replan_full_horizon_feedback_20261110.md)

近期：[新参考重规划起点的八点质量可行性](replan_full_horizon_quality_20261110.md)

近期：[重规划新参考下的质量约束实际反馈](replan_quality_feedback_20261110.md)

近期：[同一重规划状态下的新旧参考对照](replan_reference_switch_20261110.md)

近期：[组合预警的16周期固定参考实际反馈](combined_warning_long_feedback_20261110.md)

近期：[组合预警首次触发时的实际反馈对照](combined_warning_feedback_20261110.md)

近期：[跟踪质量与关节速度的组合预警回顾](combined_quality_warning_20261110.md)

本页导航原始文件；正文、配置值和历史身份保持。目录迁移不改变实验或控制资格。

[当前状态](../../../../docs/status.md) · [研究索引](../../../../docs/research_index.md) · [路径与身份](../../../../docs/migrations/repository_layout_20261029.json)

| 领域 | 文件数 |
| --- | ---: |
| [execution](execution/index.md) | 21 |
| [planning](planning/index.md) | 1 |
| [preview](preview/index.md) | 2 |
| [tracking](tracking/index.md) | 2 |

| 文件 | 原声明路径 |
| --- | --- |
| [071 当前电机目标保持＋位姿耦合预览组合pilot](071_motor_state_coupled_preview_20261025.md) | `plans/v0.0.1/071_motor_state_coupled_preview_20261025.md` |
| [080 079前向响应模型的有限命令逆控制资格](080_velocity_model_inverse_control_20261025.md) | `plans/v0.0.1/080_velocity_model_inverse_control_20261025.md` |
| [082 固定平移完整逆补偿对照](082_translation_inverse_qualification_20261025.md) | `plans/v0.0.1/082_translation_inverse_qualification_20261025.md` |
| [083 平移位置/速度一致性的单一名义逆对照](083_pose_velocity_consistent_inverse_20261025.md) | `plans/v0.0.1/083_pose_velocity_consistent_inverse_20261025.md` |
| [118 旋转增量与周期末角速度响应定义核验](118_rotation_velocity_consistency_20261025.md) | `plans/v0.0.1/118_rotation_velocity_consistency_20261025.md` |
| [153 合格forward受限逆控制绑定](153_qualified_inverse_control_binding_20261025.md) | `plans/v0.0.1/153_qualified_inverse_control_binding_20261025.md` |
| [154 原18条正确参考逆控制回归准入](154_qualified_inverse_regression_20261025.md) | `plans/v0.0.1/154_qualified_inverse_regression_20261025.md` |
| [155 有界逆解独立最优性与预测审查](155_independent_inverse_optimality_20261025.md) | `plans/v0.0.1/155_independent_inverse_optimality_20261025.md` |
| [157 原18条正确参考受限逆控制执行](157_qualified_inverse_execution_20261025.md) | `plans/v0.0.1/157_qualified_inverse_execution_20261025.md` |
| [160 数值接口修复后原18条控制回归](160_native_precision_control_execution_20261025.md) | `plans/v0.0.1/160_native_precision_control_execution_20261025.md` |
| [163 首条完整控制偏离诊断](163_inverse_control_departure_20261025.md) | `plans/v0.0.1/163_inverse_control_departure_20261025.md` |
| [170 位姿精度与任务结果不一致的同profile审查](170_task_tracking_disagreement_20261025.md) | `plans/v0.0.1/170_task_tracking_disagreement_20261025.md` |
| [202 待实现：预测跟踪约束下的加载限速](202_tracking_guarded_loading_rate_20261025.md) | `plans/v0.0.1/202_tracking_guarded_loading_rate_20261025.md` |
| [203 预测跟踪限速核心与保存状态数值核验](203_tracking_guard_saved_state_audit_20261025.md) | `plans/v0.0.1/203_tracking_guard_saved_state_audit_20261025.md` |
| [204 完整前向映射与限速决策独立绑定](204_guarded_loading_forward_binding_20261025.md) | `plans/v0.0.1/204_guarded_loading_forward_binding_20261025.md` |
| [205 预测跟踪约束加载限速的完整18条回归](205_guarded_loading_execution_20261025.md) | `plans/v0.0.1/205_guarded_loading_execution_20261025.md` |
| [215 固定直接逆的目标反馈与饱和顺序审查](215_direct_inverse_feedback_audit_20261026.md) | `plans/v0.0.1/215_direct_inverse_feedback_audit_20261026.md` |
| [216 唯一改变饱和顺序的直接逆对照](216_late_clipped_direct_inverse_20261026.md) | `plans/v0.0.1/216_late_clipped_direct_inverse_20261026.md` |
| [219 严格凸预览QP活动集版本](219_exact_preview_qp_20261026.md) | `plans/v0.0.1/219_exact_preview_qp_20261026.md` |
| [220 预览规划的在线输入合同](220_preview_runtime_binding_20261026.md) | `plans/v0.0.1/220_preview_runtime_binding_20261026.md` |
| [221 约束预览的真实控制对照](221_preview_physical_execution_20261026.md) | `plans/v0.0.1/221_preview_physical_execution_20261026.md` |
| [222 预览JSON检查修正与明确恢复执行](222_preview_serialization_recovery_20261026.md) | `plans/v0.0.1/222_preview_serialization_recovery_20261026.md` |
| [223 预览剩余失败的同状态/实轨迹审查](223_preview_remaining_failure_audit_20261026.md) | `plans/v0.0.1/223_preview_remaining_failure_audit_20261026.md` |
| [224 持有段的惯性耦合只读复核](224_preview_coupling_diagnosis_20261026.md) | `plans/v0.0.1/224_preview_coupling_diagnosis_20261026.md` |
| [228 当前惯性耦合的单步约束解析逆对照](228_current_coupled_inverse_20261026.md) | `plans/v0.0.1/228_current_coupled_inverse_20261026.md` |
| [229 当前M/J读取与旋转goal的原生环境准入](229_coupled_state_reader_admission_20261026.md) | `plans/v0.0.1/229_coupled_state_reader_admission_20261026.md` |
| [230 耦合逆和独立创新的完整记录链路](230_coupled_recorded_chain_20261026.md) | `plans/v0.0.1/230_coupled_recorded_chain_20261026.md` |
| [231 记录链路检查的持久监管](231_coupled_chain_durable_supervision_20261026.md) | `plans/v0.0.1/231_coupled_chain_durable_supervision_20261026.md` |
| [232 当前耦合逆的真实执行和日志绑定](232_current_coupled_control_binding_20261026.md) | `plans/v0.0.1/232_current_coupled_control_binding_20261026.md` |
| [234 当前耦合控制完整日志准入](234_coupled_control_log_binding_20261026.md) | `plans/v0.0.1/234_coupled_control_log_binding_20261026.md` |
| [235 完整日志准入返回值比较修正](235_coupled_log_comparison_repair_20261026.md) | `plans/v0.0.1/235_coupled_log_comparison_repair_20261026.md` |
| [236 JSON日志数值与元数据比较](236_coupled_log_metadata_comparison_20261026.md) | `plans/v0.0.1/236_coupled_log_metadata_comparison_20261026.md` |
| [237 耦合控制候选向世界模型开发对照交接](237_coupled_wam_development_handoff_20261026.md) | `plans/v0.0.1/237_coupled_wam_development_handoff_20261026.md` |
| [238 全耦合位姿反解的下一步验证](238_full_coupled_pose_inverse_diagnosis_20261026.md) | `plans/v0.0.1/238_full_coupled_pose_inverse_diagnosis_20261026.md` |
| [239 当前SO3 spring的六维联合逆](239_nonlinear_coupled_pose_inverse_20261026.md) | `plans/v0.0.1/239_nonlinear_coupled_pose_inverse_20261026.md` |
| [248 四步预览固定物理回归](248_coupled_preview_physical_20261026.md) | `plans/v0.0.1/248_coupled_preview_physical_20261026.md` |
| [八周期反馈预览与首步误差保护](guarded_long_joint_physical_20261027.md) | `plans/v0.0.1/guarded_long_joint_physical_20261027.md` |
| [明确内侧余量与软回退的真实反馈对照](interior_joint_feedback_20261027.md) | `plans/v0.0.1/interior_joint_feedback_20261027.md` |
| [限位反解的原生协议与固定局部分支](joint_limit_candidate_physical_20261027.md) | `plans/v0.0.1/joint_limit_candidate_physical_20261027.md` |
| [精确TCP位姿约束候选的固定局部物理比较](joint_limit_guarded_physical_20261027.md) | `plans/v0.0.1/joint_limit_guarded_physical_20261027.md` |
| [四周期关节约束反解的固定原生比较](joint_limit_preview_physical_20261027.md) | `plans/v0.0.1/joint_limit_preview_physical_20261027.md` |
| [已生成参考执行窗口的限定因果诊断](preview_chunk_event_diagnostic_20261027.md) | `plans/v0.0.1/preview_chunk_event_diagnostic_20261027.md` |
| [已接收模型解的关节/末端一致性（CPU）](preview_fk_consistency_20261028.md) | `plans/v0.0.1/preview_fk_consistency_20261028.md` |
| [固定失败分支的只读内环力重放](preview_limit_force_replay_20261027.md) | `plans/v0.0.1/preview_limit_force_replay_20261027.md` |
| [关节约束与首周期跟踪精度联合检查](quality_capped_tracking_20261028.md) | `plans/v0.0.1/quality_capped_tracking_20261028.md` |
| [捕获边界状态的反馈回入候选](tracking_buffer_reentry_20261028.md) | `plans/v0.0.1/tracking_buffer_reentry_20261028.md` |
| [原正确参考剩余17条候选回归](tracking_correct_reference_remaining_20261027.md) | `plans/v0.0.1/tracking_correct_reference_remaining_20261027.md` |
| [整段跟踪代价与首步硬限额的原生对照](tracking_margin_feedback_20261027.md) | `plans/v0.0.1/tracking_margin_feedback_20261027.md` |
| [原运算顺序候选完整18条验证](tracking_ordered_candidate_20261027.md) | `plans/v0.0.1/tracking_ordered_candidate_20261027.md` |
| [恢复原参考坐标变换运算顺序：case06基线复现](tracking_reference_ordered_port_20261027.md) | `plans/v0.0.1/tracking_reference_ordered_port_20261027.md` |
| [同参考的位置—速度制动候选](tracking_velocity_braking_20261028.md) | `plans/v0.0.1/tracking_velocity_braking_20261028.md` |
| [守护QP后的原完整WAM开发矩阵](tracking_wam_v4_20261027.md) | `plans/v0.0.1/tracking_wam_v4_20261027.md` |
| [释放参考阈值后的完整WAM开发矩阵](tracking_wam_v5_20261028.md) | `plans/v0.0.1/tracking_wam_v5_20261028.md` |

历史正文中的相对链接按原声明目录解释；不要改写原文或把旧运行登记当作新布局运行资格。
