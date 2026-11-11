# 研究计划：plans/v0.0.1/tracking/gripper

本页导航原始文件；正文、配置值和历史身份保持。目录迁移不改变实验或控制资格。

[当前状态](../../../../docs/status.md) · [研究索引](../../../../docs/research_index.md) · [路径与身份](../../../../docs/migrations/repository_layout_20261029.json)

| 领域 | 文件数 |
| --- | ---: |
| [motor](motor/index.md) | 3 |

| 文件 | 原声明路径 |
| --- | --- |
| [夹爪连续开度与闭合保持 — 2026-10-21（Asia/Shanghai）](004_gripper_position_hold_20261021.md) | `plans/v0.0.1/004_gripper_position_hold_20261021.md` |
| [065 夹爪释放前向响应辨识](065_gripper_forward_identification_20261025.md) | `plans/v0.0.1/065_gripper_forward_identification_20261025.md` |
| [066 单一有限生成参考预览候选](066_gripper_reference_preview_20261025.md) | `plans/v0.0.1/066_gripper_reference_preview_20261025.md` |
| [067 六步预览的更广正确参考回归](067_gripper_preview_regression_20261025.md) | `plans/v0.0.1/067_gripper_preview_regression_20261025.md` |
| [068 位姿耦合的有限开度预览假设](068_coupled_gripper_preview_20261025.md) | `plans/v0.0.1/068_coupled_gripper_preview_20261025.md` |
| [072 当前电机状态条件的轻量夹爪逆模型](072_state_conditioned_gripper_inverse_20261025.md) | `plans/v0.0.1/072_state_conditioned_gripper_inverse_20261025.md` |
| [075 全阶段前向state响应辨识：单次固定对照](075_all_phase_gripper_forward_identification_20261025.md) | `plans/v0.0.1/075_all_phase_gripper_forward_identification_20261025.md` |
| [076 训练初态下的夹爪命令干预与因果响应采集](076_native_gripper_response_interventions_20261025.md) | `plans/v0.0.1/076_native_gripper_response_interventions_20261025.md` |
| [077 076速度传感器重复forward修复与完整新采集](077_native_gripper_response_sensor_repair_20261025.md) | `plans/v0.0.1/077_native_gripper_response_sensor_repair_20261025.md` |
| [078 077完整baseline后元数据序列化修复](078_native_gripper_response_metadata_repair_20261025.md) | `plans/v0.0.1/078_native_gripper_response_metadata_repair_20261025.md` |
| [084 固定083 arm的native夹爪命令诊断](084_native_gripper_ablation_20261025.md) | `plans/v0.0.1/084_native_gripper_ablation_20261025.md` |
| [179 arm命令对实测手指间距变化的训练标签](179_arm_finger_response_labels_20261025.md) | `plans/v0.0.1/179_arm_finger_response_labels_20261025.md` |
| [180 手指标签独立核验](180_arm_finger_labels_independent_20261025.md) | `plans/v0.0.1/180_arm_finger_labels_independent_20261025.md` |
| [181 固定146设计的手指响应单输出辨识](181_arm_finger_linear_fit_20261025.md) | `plans/v0.0.1/181_arm_finger_linear_fit_20261025.md` |
| [182 六留出轨迹的手指间距变化预测](182_arm_finger_holdout_20261025.md) | `plans/v0.0.1/182_arm_finger_holdout_20261025.md` |
| [183 原六anchor手指局部arm干预斜率](183_arm_finger_local_response_20261025.md) | `plans/v0.0.1/183_arm_finger_local_response_20261025.md` |
| [184 局部手指响应标签独立复核](184_arm_finger_local_independent_20261025.md) | `plans/v0.0.1/184_arm_finger_local_independent_20261025.md` |
| [185 待执行：手指响应斜率/自然偏置分阶段辨识](185_separated_finger_response_20261025.md) | `plans/v0.0.1/185_separated_finger_response_20261025.md` |
| [186 手指动作响应的真实非线性审查](186_finger_action_curvature_20261025.md) | `plans/v0.0.1/186_finger_action_curvature_20261025.md` |
| [187 待执行：独立训练干预的手指受载覆盖](187_finger_intervention_coverage_20261025.md) | `plans/v0.0.1/187_finger_intervention_coverage_20261025.md` |
| [188 未训练手指干预的条件邻近性诊断](188_finger_condition_neighbors_20261025.md) | `plans/v0.0.1/188_finger_condition_neighbors_20261025.md` |
| [189 训练内独立手指加载锚点选择](189_independent_finger_anchor_selection_20261025.md) | `plans/v0.0.1/189_independent_finger_anchor_selection_20261025.md` |
| [190 独立手指选点复核](190_independent_finger_selection_verification_20261025.md) | `plans/v0.0.1/190_independent_finger_selection_verification_20261025.md` |
| [191 训练内新颖手指加载状态的成对采集](191_finger_loading_interventions_20261025.md) | `plans/v0.0.1/191_finger_loading_interventions_20261025.md` |
| [192 追加采集收尾内环步数计数修正](192_finger_collection_substep_summary_20261025.md) | `plans/v0.0.1/192_finger_collection_substep_summary_20261025.md` |
| [193 资格关闭后的新增手指方程合并](193_finger_loading_equations_20261025.md) | `plans/v0.0.1/193_finger_loading_equations_20261025.md` |
| [194 追加9方程的独立复核](194_finger_loading_equations_verification_20261025.md) | `plans/v0.0.1/194_finger_loading_equations_verification_20261025.md` |
| [195 新增训练覆盖后的固定两阶段手指模型](195_finger_loading_fit_20261025.md) | `plans/v0.0.1/195_finger_loading_fit_20261025.md` |
| [196 可用手指接触历史输入](196_measured_finger_contact_history_20261025.md) | `plans/v0.0.1/196_measured_finger_contact_history_20261025.md` |
| [197 原30轨迹的手指历史可用性核验](197_measured_finger_history_audit_20261025.md) | `plans/v0.0.1/197_measured_finger_history_audit_20261025.md` |
| [198 待执行：固定响应基增加已审查手指历史](198_finger_history_response_candidate_20261025.md) | `plans/v0.0.1/198_finger_history_response_candidate_20261025.md` |
| [明确张开参考才释放：独立物理对照](gripper_release_floor_20261028.md) | `plans/v0.0.1/gripper_release_floor_20261028.md` |
| [即将释放时暂缓额外闭合：历史影子检查](defer_close_release_shadow_20261111.md) | 新登记 |

历史正文中的相对链接按原声明目录解释；不要改写原文或把旧运行登记当作新布局运行资格。
