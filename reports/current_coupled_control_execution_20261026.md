# 当前惯性耦合单步逆：完整执行结果

原18条4907周期完整结束，任务稳定17/18、综合达标17/18。全部18条的位置RMSE≤0.01m及SO3 RMSE≤0.05rad；唯一未稳定为case11（task1_native_demo_8，put the wine bottle on the rack）。因此原18/18资格失败，237世界模型开发交接与fresh24继续关闭，总目标未完成。

候选只替换平移逆：当前robot q/SITE只读M/J和native rotation goal，228单步耦合位置/速度目标，真实接口与previous±0.25约束；083/159旋转、079夹爪和173最终SOURCE命令commit保持。150与228 nominal/创新分开记录，独立episode隔离，只从完成周期更新，无新增fit、任务特判或目标区域容差修改。

## 完整来源和生命周期

123项CPU测试、18真实CLI dry和checker导入通过。18个collect和18个verify全部实际wait0/reap、无发送信号，owner正常wait0/reap与进程消失通过；36个stage、4907周期、源码不漂移以及逐条独立raw-state/ODE/KKT、机械标签、图像和任务审查完整关闭。收尾由独立持久监督自动执行，其closure child也实际wait0/reap。CPU/OSMesa均空CUDA/LD；0模型请求、0新增fit、0 fresh。

配置：`configs/current_coupled_control_registered_20261026.json`；执行：`outputs/current_coupled_control_20261026/execution_result.json`；完整汇总：同目录`regression_summary.json`；来源/生命周期关闭：`closure_verified.json`；收尾监督：`outputs/current_coupled_closure_supervision_20261026/execution_result.json`。关闭脚本为`scripts/close_current_coupled_control_checked.py`，逐条验证coupled日志周期数与KKT门槛。

## 固定逐条结果

| case | 位置RMSE (m) | SO3 RMSE (rad) | 末段连续成功周期 | 综合达标 |
| --- | --- | --- | --- | --- |
| 00 | 0.00110797793 | 0.00378261176 | 80 | 是 |
| 01 | 0.000291884243 | 0.00959077418 | 75 | 是 |
| 02 | 0.000420761098 | 0.00362710992 | 85 | 是 |
| 03 | 0.000468448016 | 0.00452903923 | 85 | 是 |
| 04 | 0.000602102334 | 0.0050252641 | 98 | 是 |
| 05 | 0.000265695461 | 0.00978380913 | 91 | 是 |
| 06 | 0.00189867368 | 0.0106204656 | 88 | 是 |
| 07 | 0.00264490625 | 0.00448776183 | 86 | 是 |
| 08 | 0.00305227311 | 0.00527529661 | 85 | 是 |
| 09 | 0.00689373014 | 0.0141046013 | 93 | 是 |
| 10 | 0.000406606862 | 0.0089772589 | 86 | 是 |
| 11 | 0.000323763325 | 0.00927817551 | 0 | 否 |
| 12 | 0.000705039609 | 0.0101339316 | 77 | 是 |
| 13 | 0.00563028597 | 0.00426128975 | 85 | 是 |
| 14 | 0.00531985967 | 0.00581498445 | 84 | 是 |
| 15 | 0.000196014059 | 0.00249630833 | 524 | 是 |
| 16 | 0.000300032035 | 0.00674344751 | 461 | 是 |
| 17 | 0.000335879693 | 0.00239056308 | 501 | 是 |

上一轮完整预览控制为16/18稳定、13/18综合达标。本轮修复其case06/09/10/12/15五条不达标，但引入case11任务退化，不能只报告改善。085因果残差基线为17/18稳定、15/18综合达标，失败构成也不同。原控制器、样本划分和验收门槛保持，不按case选择候选。

## case11失败证据

case11位置RMSE0.00032376332546262037m、SO3 RMSE0.009278175511682753rad，位姿数值达标，但只有5个成功周期、末段0。`outputs/current_coupled_rack_failure_20261026/verified.json`以实际XML/282个raw端点和manifest绑定的LIBERO predicate源码重建，281个任务标签全部一致。末态仍与酒架接触，但瓶中心超出原生目标region的x边界0.0015348200576833088m；旧预览成功轨迹在该方向余量为0.017223242085227755m。任务失败判定保持原样。

`outputs/current_coupled_grasp_departure_20261026/verified.json`核对相同期望轨迹和控制接口：首次瓶—机器人接触配对差异在端点76，成功预览出现hand_collision，本候选没有；端点83瓶位置差超过1mm；夹爪命令仅第90/91步不同，motor cache仅端点91不同；端点99瓶位置差超过1cm。接触差异发生在离散夹爪命令差异之前，不能把全部退化归因于后者。此为保存端点FK和碰撞几何，无新增integration、受力重建或因果干预，尚未证明唯一上游原因。

## 准入和下一步

233修正的164周期coupled记录链路及actual wait0/reap通过。完整日志234首版返回字典字段不同导致0周期失败，235版重复数值比较误处理字符串元数据导致TypeError，原失败/源码/actual wait1/reap保持；236新入口统一比较数值与元数据，164周期JSON往返、164次SOURCE commit及16项损坏拒绝全部通过，最大KKT stationarity residual为5.039687500896338e-13。证据：`outputs/current_coupled_control_log_serialized_20261026/independent_verified.json`和`outputs/current_coupled_log_supervision_serialized_20261026/execution_result.json`。这些准入不能代替上述物理结果。

预测诊断入口`scripts/diagnose_current_coupled_response.py`只读取独立通过机械审查的实际case，以实测SITE端点重算位移/SO3误差，产物在`outputs/current_coupled_response_diagnosis_20261026/`。同状态/实际命令的预测准确性是诊断证据，不是替代控制器的反事实轨迹或任务成功。

按238先检验统一平移/旋转反解的可预测误差：当前模型已经描述六维耦合，但只反解平移，旋转仍为独立旧逆，角度/角速度创新未进入规划。不提前声称可以修复case11，不修改原目标region、使用case专用偏移或在线object/task label。任何新候选仍需完整原18/18，然后才进入237的固定开发与原24条留出新初态验收。

后续239已实现六维联合逆及23项测试、首条164周期原生记录准入，详见`reports/full_coupled_pose_inverse_20261026.md`。尚无新的物理性能结论，本报告17/18资格失败保持。
