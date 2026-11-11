# 已重放反馈中的因果 TCP 预测偏差

2026-11-11，按[预登记](../../../plans/v0.0.1/tracking/response_models/causal_tcp_prediction_audit_20261111.md)
只读分析[已独立重放的 46 个物理周期](../joint_control/quality_rate_repair_recorded_feedback_20261111.md)。
每周期先固定当时的模型首点预测，再在周期完成后读取实测 TCP 误差并更新历史；
首个周期无历史，不计入后两种诊断的分母。位置使用世界坐标系米，旋转使用 SO(3)
复合及测地距离；任务成功标签不作为输入。没有新模型调用、物理积分或新初态。

| 诊断（45 个有历史周期） | 预测达门 | 预测达门但实际未达门 | 预测未达门但实际达门 | 位置预测 RMSE | 旋转预测 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原局部模型 | 45 | 4 | 0 | 0.003835 m | 0.011744 rad |
| 上一已完成周期的有符号偏差补偿 | 41 | 0 | 0 | 0.001307 m | 0.004190 rad |
| 历史最大偏差叠加当前预测范数 | 39 | 0 | 2 | — | — |

预测侧沿用原模型门的 `1e-10` 数值容差，实际侧沿用物理评价的严格
0.01 m / 0.05 rad 门。原模型的四次假达门发生在 step122–125。最后 step125
的原模型预测为 0.010000 m / 0.032132 rad，上一周期偏差补偿后的预测为
0.021585 m / 0.069216 rad，实际为 0.024850 m / 0.081177 rad。两种预测都只是
对**已执行动作**的离线评估；若以其中一种改变策略，后续状态和动作可能改变，
不能直接把表中计数当作闭环效果。

历史最大偏差界仅覆盖 45 个周期中的 38 个位置误差和 40 个旋转误差，证实该
经验量不保证未来覆盖。原模型预测误差在 step120–125 窗口显著增加：位置 RMSE
0.010389 m、旋转 RMSE 0.032029 rad，该窗口实际仅 2/6 达位姿门；此前各窗口
均 8/8 达门。上一周期补偿值得作为候选继续研究，但这些结果来自单一已研究初态
和同一条已选择的动作轨迹，不能宣称泛化、实时或稳定任务成功。

运行前 32 并发专项 13 项通过、无跳过和收集错误。首次专项因引用不存在的测试
路径而收集失败，0 项运行，失败收据保留；当时测试入口把 `all_tests_passed`
错误标为真，但总体 `passed` 为假。现已修复空收集摘要，并用真实收集失败回归
验证；修复测试首次使用了不合规的临时目录，失败后改到 `outputs/qa` 并重跑
通过，两次收据均保留。

| 仓库内证据 | SHA256 |
| --- | --- |
| [因果偏差分析](../../../outputs/causal_tcp_prediction_audit_20261111/result.json) | `0a70c6505f69ba9e666b13ccb000e8eac2f418b995c515d4db9fcec5efd02c2c` |
| [32 并发运行前专项](../../../outputs/qa/causal_tcp_prediction_audit_20261111/result.json) | `b1f6a086b915bb460de647162c3becbab1df72cbb118aacba2c6ebb520502fcb` |
| [首次收集失败](../../../outputs/qa/causal_tcp_prediction_audit_unit_20261111/result.json) | `d9a73bb1c6972ad09235066e8b33ea8f8c28eb85fef5fd61027489c79f50beae` |
| [修正测试路径后通过](../../../outputs/qa/causal_tcp_prediction_audit_unit_v2_20261111/result.json) | `b7e75513530301703a1ae52c37470aed156d8f0809a1a8dc088868e4846cdd1b` |
| [运行器修复测试的目录失败](../../../outputs/qa/cpu_runner_empty_collection_fix_20261111/result.json) | `5618b942dd7b57caf2fcfac6213ca05719d9084d1d29b41275d99484111ba4b5` |
| [运行器修复回归通过](../../../outputs/qa/cpu_runner_empty_collection_fix_v2_20261111/result.json) | `18447c3ee5565d8a131e00b82349c00ce75be8b94fe3e521880bdc0cf2115df5` |
