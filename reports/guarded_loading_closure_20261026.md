# 205 完整收尾结果

`outputs/qualified_inverse_guarded_loading_20261025/closure_verified.json`完成：原18条、4907控制周期、36个采集/独立检查stage与owner均正常实际wait0/reap，来源清单保持，机械/标签/图像检查全部通过。

任务稳定15/18，同时满足稳定、位置RMSE<=.01m及SO3 RMSE<=.05rad为13/18，未达18/18资格。

| case | record | 稳定 | 位置RMSE m | SO3 RMSE rad |
| --- | --- | --- | ---: | ---: |
| 03 | task0_native_demo_40 | 否 | .04337894 | .24899405 |
| 06 | task1_native_demo_46 | 是 | .03481158 | .25478767 |
| 12 | task1_native_demo_11 | 否 | .00140970 | .01022151 |
| 14 | task2_native_demo_9 | 否 | .04324499 | .18540434 |
| 15 | task0_init0 | 是 | .01183509 | .02526614 |

case12再次说明小位姿误差不等于任务稳定。206早期单步预测准确与最终失败并存，不能将预测5mm位置约束宣称为持续可行性保证。本候选不进入WAMdev或最终新初态验证。完整逐条证据见同目录`regression_summary.json`，失败记录全部保留。
