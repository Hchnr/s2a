# 研究计划：plans/v0.0.1/tracking/trajectory_search

本页导航原始文件；正文、配置值和历史身份保持。目录迁移不改变实验或控制资格。

[当前状态](../../../../docs/status.md) · [研究索引](../../../../docs/research_index.md) · [路径与身份](../../../../docs/migrations/repository_layout_20261029.json)

| 文件 | 原声明路径 |
| --- | --- |
| [029：自由轨迹与同实际时间的 teacher 输入诊断](029_free_trajectory_diagnostic_20261021.md) | `plans/v0.0.1/029_free_trajectory_diagnostic_20261021.md` |
| [057：固定052历史的末节点轨迹目标](057_terminal_trajectory_objective_20261025.md) | `plans/v0.0.1/057_terminal_trajectory_objective_20261025.md` |
| [111 固定局部响应训练轨迹留一](111_local_response_trajectory_leaveout_20261025.md) | `plans/v0.0.1/111_local_response_trajectory_leaveout_20261025.md` |
| [152 真实原生前缀在线输入一致性](152_runtime_prefix_parity_20261025.md) | `plans/v0.0.1/152_runtime_prefix_parity_20261025.md` |
| [245 SQP控制接入及运行来源范围](245_sqp_control_binding_and_provenance_20261026.md) | `plans/v0.0.1/245_sqp_control_binding_and_provenance_20261026.md` |
| [实际接纳点曲率保留的显式 SQP（CPU，执行前登记）](accepted_point_sqp_20261029.md) | `plans/v0.0.1/accepted_point_sqp_20261029.md` |
| [全约束实际残差修正：保存方向3639](constraint_residual_repair_20261029.md) | `plans/v0.0.1/constraint_residual_repair_20261029.md` |
| [固定方向诊断：隔离独立复核 v3（执行前登记）](directional_derivative_check_v3_20261029.md) | `plans/v0.0.1/directional_derivative_check_v3_20261029.md` |
| [固定锚点的方向导数一致性（CPU，执行前登记）](directional_derivative_consistency_20261029.md) | `plans/v0.0.1/directional_derivative_consistency_20261029.md` |
| [固定方向诊断入口修正（v2，执行前登记）](directional_derivative_v2_admission_20261029.md) | `plans/v0.0.1/directional_derivative_v2_admission_20261029.md` |
| [先首周期可行、后全预览性能的规划候选（CPU）](feasibility_first_joint_plan_20261028.md) | `plans/v0.0.1/feasibility_first_joint_plan_20261028.md` |
| [从线性可行点启动长预览搜索（CPU）](feasible_start_trajectory_20261028.md) | `plans/v0.0.1/feasible_start_trajectory_20261028.md` |
| [实际 cursor9 停止点的投影制动时间敏感性（CPU，2026-10-28）](fk_braking_time_numeric_20261028.md) | `plans/v0.0.1/fk_braking_time_numeric_20261028.md` |
| [同一真实停止点的FK首周期/全预览精度对照（CPU）](fk_cap_scope_20261028.md) | `plans/v0.0.1/fk_cap_scope_20261028.md` |
| [关节预测一致的FK预览（CPU）](fk_consistent_trajectory_20261028.md) | `plans/v0.0.1/fk_consistent_trajectory_20261028.md` |
| [FK一致预览的有限原生反馈（CPU）](fk_feedback_native_20261028.md) | `plans/v0.0.1/fk_feedback_native_20261028.md` |
| [从实际370延续首周期FK反馈（CPU）](fk_first_feedback_native_20261028.md) | `plans/v0.0.1/fk_first_feedback_native_20261028.md` |
| [早期真实起点的首周期FK长预览（CPU）](fk_first_long_20261028.md) | `plans/v0.0.1/fk_first_long_20261028.md` |
| [首周期长预览同起点实际反馈对照（CPU）](fk_first_long_native_20261028.md) | `plans/v0.0.1/fk_first_long_native_20261028.md` |
| [cursor9 停止状态的约束归因诊断（CPU）](fk_stop_constraint_audit_20261028.md) | `plans/v0.0.1/fk_stop_constraint_audit_20261028.md` |
| [固定初始全局搜索上限的独立对照](global_envelope_search_20261029.md) | `plans/v0.0.1/global_envelope_search_20261029.md` |
| [完整H12：反复内点恢复与实际改善比信赖域](interior_trajectory_search_20261029.md) | `plans/v0.0.1/interior_trajectory_search_20261029.md` |
| [同一实际起点的长预览（CPU）](long_trajectory_caps_20261028.md) | `plans/v0.0.1/long_trajectory_caps_20261028.md` |
| [同总时长的提前/临近参考相位分配（执行前登记，未执行）](paired_reference_phase_20261028.md) | `plans/v0.0.1/paired_reference_phase_20261028.md` |
| [H12首个callback方向的完整响应回缩（CPU、执行前登记）](prefix_callback_backtracking_20261029.md) | `plans/v0.0.1/prefix_callback_backtracking_20261029.md` |
| [原H12前缀优化余量对照（CPU、执行前登记）](prefix_optimization_reserve_20261029.md) | `plans/v0.0.1/prefix_optimization_reserve_20261029.md` |
| [保留已达标前缀的完整H12轨迹修复（执行前登记，CPU）](prefix_preserving_trajectory_control_20261028.md) | `plans/v0.0.1/prefix_preserving_trajectory_control_20261028.md` |
| [H12独立检查的完整资产路径迁移（执行前登记，CPU）](prefix_trust_asset_check_repair_20261029.md) | `plans/v0.0.1/prefix_trust_asset_check_repair_20261029.md` |
| [原H12时序的有限信赖域对照（执行前登记，CPU，待执行）](prefix_trust_region_control_20261028.md) | `plans/v0.0.1/prefix_trust_region_control_20261028.md` |
| [保存H12候选的前缀线性化诊断（CPU、零新响应）](prefix_trust_saved_linearization_20261029.md) | `plans/v0.0.1/prefix_trust_saved_linearization_20261029.md` |
| [恢复下一目标的静态可达性分离检查（CPU）](recovery_pose_ik_20261028.md) | `plans/v0.0.1/recovery_pose_ik_20261028.md` |
| [实际状态锚定的误差管过渡（CPU）](recovery_pose_tube_20261028.md) | `plans/v0.0.1/recovery_pose_tube_20261028.md` |
| [安全但越出搜索误差上限的拒绝响应反馈](rejected_response_search_20261029.md) | `plans/v0.0.1/rejected_response_search_20261029.md` |
| [从终端余量恢复候选重新开始完整 H12 SQP](restored_anchor_sqp_20261029.md) | `plans/v0.0.1/restored_anchor_sqp_20261029.md` |
| [保存方向的固定回缩检验（执行前登记，CPU，尚未执行）](saved_direction_backtracking_20261028.md) | `plans/v0.0.1/saved_direction_backtracking_20261028.md` |
| [固定回缩独立checker元数据修复（执行前登记，CPU）](saved_direction_backtracking_check_repair_20261028.md) | `plans/v0.0.1/saved_direction_backtracking_check_repair_20261028.md` |
| [保存差分组与后续候选的局部线性诊断（执行前登记，CPU）](saved_trajectory_differences_20261028.md) | `plans/v0.0.1/saved_trajectory_differences_20261028.md` |
| [保存轨迹的相位与路径偏差诊断（执行前登记，CPU）](saved_trajectory_phase_diagnostic_20261028.md) | `plans/v0.0.1/saved_trajectory_phase_diagnostic_20261028.md` |
| [同响应上限的60阶段预算对照](stage_budget_search_20261029.md) | `plans/v0.0.1/stage_budget_search_20261029.md` |
| [H12独立检查的等价高效实现（执行前登记，CPU）](state_updated_trajectory_check_v2_20261028.md) | `plans/v0.0.1/state_updated_trajectory_check_v2_20261028.md` |
| [早期实际状态的原生精度多周期控制规划（执行前登记，CPU）](state_updated_trajectory_control_20261028.md) | `plans/v0.0.1/state_updated_trajectory_control_20261028.md` |
| [pose乘子恢复活动子集的有限对照](state_vector_dual_recovery_20261029.md) | `plans/v0.0.1/state_vector_dual_recovery_20261029.md` |
| [dual recovery：输入路径恢复后的隔离check-only v2](state_vector_dual_recovery_checker_recovery_v2_20261029.md) | `plans/v0.0.1/state_vector_dual_recovery_checker_recovery_v2_20261029.md` |
| [TCP误差向量 Gauss–Newton 的前瞻对照](state_vector_gn_search_20261029.md) | `plans/v0.0.1/state_vector_gn_search_20261029.md` |
| [GN整个目标的统一缩放对照](state_vector_objective_scale_20261029.md) | `plans/v0.0.1/state_vector_objective_scale_20261029.md` |
| [末周期平移恢复：独立检查器v2的类型比较修复](tail_translation_check_v2_20261029.md) | `plans/v0.0.1/tail_translation_check_v2_20261029.md` |
| [末周期平移探针：在原安全门槛内恢复终端余量](tail_translation_reserve_20261029.md) | `plans/v0.0.1/tail_translation_reserve_20261029.md` |
| [保存3543的有界末周期修复](terminal_only_tail_repair_20261029.md) | `plans/v0.0.1/terminal_only_tail_repair_20261029.md` |
| [终端单行有限割线修正：未拟合方向外推检验](terminal_secant_probe_20261029.md) | `plans/v0.0.1/terminal_secant_probe_20261029.md` |
| [全预览端点精度约束候选（CPU）](trajectory_capped_joint_plan_20261028.md) | `plans/v0.0.1/trajectory_capped_joint_plan_20261028.md` |

历史正文中的相对链接按原声明目录解释；不要改写原文或把旧运行登记当作新布局运行资格。
