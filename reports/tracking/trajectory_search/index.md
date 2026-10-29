# 研究报告：reports/tracking/trajectory_search

本页导航原始文件；正文、配置值和历史身份保持。目录迁移不改变实验或控制资格。

[当前状态](../../../docs/status.md) · [研究索引](../../../docs/research_index.md) · [路径与身份](../../../docs/migrations/repository_layout_20261029.json)

| 文件 | 原声明路径 |
| --- | --- |
| [实际接纳点曲率保留的显式 SQP：实现与组件验证](accepted_point_sqp_20261029.md) | `reports/accepted_point_sqp_20261029.md` |
| [保存方向3639的完整约束残差修正](constraint_residual_repair_20261029.md) | `reports/constraint_residual_repair_20261029.md` |
| [固定锚点方向导数一致性：CPU诊断](directional_derivative_consistency_20261029.md) | `reports/directional_derivative_consistency_20261029.md` |
| [首周期可行性优先搜索的恢复检查（2026-10-28，CPU）](feasibility_first_recovery_20261028.md) | `reports/feasibility_first_recovery_20261028.md` |
| [长窗口线性可行起点与残差准入（2026-10-28，CPU）](feasible_start_trajectory_20261028.md) | `reports/feasible_start_trajectory_20261028.md` |
| [同一停止状态的投影制动时间敏感性（2026-10-28，CPU）](fk_braking_time_numeric_20261028.md) | `reports/fk_braking_time_numeric_20261028.md` |
| [真实停止点的首周期/全预览精度对照（2026-10-28，CPU）](fk_cap_scope_20261028.md) | `reports/fk_cap_scope_20261028.md` |
| [FK一致预览的同状态比较（2026-10-28，CPU）](fk_consistent_trajectory_20261028.md) | `reports/fk_consistent_trajectory_20261028.md` |
| [FK一致预览的原生反馈与独立物理复核（2026-10-28，CPU）](fk_feedback_native_20261028.md) | `reports/fk_feedback_native_20261028.md` |
| [首周期FK反馈延续及独立仿真复核（2026-10-28，CPU）](fk_first_feedback_native_20261028.md) | `reports/fk_first_feedback_native_20261028.md` |
| [早期真实起点首周期FK长窗口结果（2026-10-28，CPU）](fk_first_long_20261028.md) | `reports/fk_first_long_20261028.md` |
| [同起点首周期FK长窗口实际对照（2026-10-28，CPU）](fk_first_long_native_20261028.md) | `reports/fk_first_long_native_20261028.md` |
| [实际377停止状态的约束归因（2026-10-28，CPU）](fk_stop_constraint_audit_20261028.md) | `reports/fk_stop_constraint_audit_20261028.md` |
| [SQP六维联合逆物理回归](full_coupled_sqp_control_execution_20261026.md) | `reports/full_coupled_sqp_control_execution_20261026.md` |
| [固定初始全局搜索上限的CPU对照](global_envelope_search_20261029.md) | `reports/global_envelope_search_20261029.md` |
| [第85步预算失败：因果重构与初值对照](interior_budget_replay_20261027.md) | `reports/interior_budget_replay_20261027.md` |
| [明确关节余量约束的模型可行性](interior_joint_plan_20261027.md) | `reports/interior_joint_plan_20261027.md` |
| [完整H12的反复内点恢复与信赖域搜索](interior_trajectory_search_20261029.md) | `reports/interior_trajectory_search_20261029.md` |
| [111 固定局部响应训练轨迹留一](local_response_trajectory_leaveout_20261025.md) | `reports/local_response_trajectory_leaveout_20261025.md` |
| [相同实际起点8/12/16点预览（2026-10-28，CPU）](long_trajectory_caps_20261028.md) | `reports/long_trajectory_caps_20261028.md` |
| [提前与临近参考相位分配（2026-10-28，CPU）](paired_reference_phase_20261028.md) | `reports/paired_reference_phase_20261028.md` |
| [中文版](pr_global_envelope_search_20261029.md) | `reports/pr_global_envelope_search_20261029.md` |
| [中文版](pr_rejected_response_search_20261029.md) | `reports/pr_rejected_response_search_20261029.md` |
| [中文版](pr_stage_budget_search_20261029.md) | `reports/pr_stage_budget_search_20261029.md` |
| [中文版](pr_state_vector_dual_recovery_20261029.md) | `reports/pr_state_vector_dual_recovery_20261029.md` |
| [中文版](pr_state_vector_gn_search_20261029.md) | `reports/pr_state_vector_gn_search_20261029.md` |
| [中文版](pr_state_vector_objective_scale_20261029.md) | `reports/pr_state_vector_objective_scale_20261029.md` |
| [H12 callback方向完整响应回缩](prefix_callback_backtracking_20261029.md) | `reports/prefix_callback_backtracking_20261029.md` |
| [固定前缀优化余量对照](prefix_optimization_reserve_20261029.md) | `reports/prefix_optimization_reserve_20261029.md` |
| [保留达标前缀的轨迹修复结果（2026-10-28，CPU）](prefix_preserving_trajectory_control_20261028.md) | `reports/prefix_preserving_trajectory_control_20261028.md` |
| [同warm H12信赖域对照](prefix_trust_region_control_20261028.md) | `reports/prefix_trust_region_control_20261028.md` |
| [保存H12候选的前缀线性化诊断](prefix_trust_saved_linearization_20261029.md) | `reports/prefix_trust_saved_linearization_20261029.md` |
| [原下一参考的静态姿态存在性（2026-10-28，CPU）](recovery_pose_ik_20261028.md) | `reports/recovery_pose_ik_20261028.md` |
| [实际状态锚定的误差管构型（2026-10-28，CPU）](recovery_pose_tube_20261028.md) | `reports/recovery_pose_tube_20261028.md` |
| [安全拒绝响应的同anchor割线反馈](rejected_response_search_20261029.md) | `reports/rejected_response_search_20261029.md` |
| [从已恢复终端余量的候选重新求解完整 H12](restored_anchor_sqp_20261029.md) | `reports/restored_anchor_sqp_20261029.md` |
| [152 真实原生前缀在线输入验证](runtime_prefix_parity_20261025.md) | `reports/runtime_prefix_parity_20261025.md` |
| [固定保存方向回缩检验（2026-10-28，CPU）](saved_direction_backtracking_20261028.md) | `reports/saved_direction_backtracking_20261028.md` |
| [保存差分与后续候选诊断（2026-10-28，CPU）](saved_trajectory_differences_20261028.md) | `reports/saved_trajectory_differences_20261028.md` |
| [保存预测的路径相位诊断（2026-10-28，CPU）](saved_trajectory_phase_diagnostic_20261028.md) | `reports/saved_trajectory_phase_diagnostic_20261028.md` |
| [SQP限速保持误差与多步预览诊断](sqp_constrained_hold_preview_20261026.md) | `reports/sqp_constrained_hold_preview_20261026.md` |
| [原生命令缩放和旋转目标的局部预览对照](sqp_native_hold_preview_20261026.md) | `reports/sqp_native_hold_preview_20261026.md` |
| [四步预览的目标相关SO(3)局部反馈复核](sqp_so3_hold_preview_20261026.md) | `reports/sqp_so3_hold_preview_20261026.md` |
| [60阶段与同122响应上限的CPU预算对照](stage_budget_search_20261029.md) | `reports/stage_budget_search_20261029.md` |
| [较早实际状态的原生目标多周期规划（2026-10-28，CPU）](state_updated_trajectory_control_20261028.md) | `reports/state_updated_trajectory_control_20261028.md` |
| [pose紧活动子集：实现与CPU准入](state_vector_dual_recovery_implementation_20261029.md) | `reports/state_vector_dual_recovery_implementation_20261029.md` |
| [TCP误差向量GN对照：停止于求解器资格门](state_vector_gn_search_20261029.md) | `reports/state_vector_gn_search_20261029.md` |
| [整个目标统一缩放：求解标志改善，原互补门仍阻止完整搜索](state_vector_objective_scale_20261029.md) | `reports/state_vector_objective_scale_20261029.md` |
| [末周期平移恢复终端余量](tail_translation_reserve_20261029.md) | `reports/tail_translation_reserve_20261029.md` |
| [保存3543的末周期平移修复](terminal_only_tail_repair_20261029.md) | `reports/terminal_only_tail_repair_20261029.md` |
| [终端单行割线修正的未拟合方向检验](terminal_secant_probe_20261029.md) | `reports/terminal_secant_probe_20261029.md` |
| [全预览端点约束的更早状态检查（2026-10-28，CPU）](trajectory_caps_numeric_20261028.md) | `reports/trajectory_caps_numeric_20261028.md` |
| [029：自由生成与 teacher 对照的完整结果](wam_free_trajectory_diagnostic_20261021.md) | `reports/wam_free_trajectory_diagnostic_20261021.md` |
| [057末节点候选准备](wam_terminal_trajectory_preparation_20261025.md) | `reports/wam_terminal_trajectory_preparation_20261025.md` |
| [轨迹目标实现准备](wam_trajectory_objective_20261025.md) | `reports/wam_trajectory_objective_20261025.md` |

历史正文中的相对链接按原声明目录解释；不要改写原文或把旧运行登记当作新布局运行资格。
