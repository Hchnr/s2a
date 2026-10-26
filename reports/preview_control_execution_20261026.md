# 221–222 六周期约束预览完整执行与收尾

证据等级为固定20Hz控制接口的LIBERO独立物理仿真（CPU/OSMesa）；参考为明确的oracle已生成轨迹，无新世界模型调用、拟合或fresh任务，真机尚未执行。原18profiles/4907周期、seed42、初态/参考时钟、控制器、083/159旋转与079夹爪策略和任务/误差门槛保持。

221真实仿真入口接入220当前状态合同和219活动集规划器：只读取当前实测世界SITE、实际previous、已完成150创新及当前已生成参考，最多预览6周期，只执行首条。最终实际命令回写goal与173 SOURCE commit，env.step完成后更新150 SITE创新。独立checker重新构造参考、标量物理rollout、目标、全部box/rate约束和非负乘子/KKT<=2e-6，并继续检查实际状态绑定、旋转/夹爪、标签、图像及任务。164条记录状态的完整命令/历史/SOURCE提交准入通过，10类损坏拒绝。

首次物理采集case00正常wait0/reap，164周期完整；原checker因JSON列表被二维NumPy索引而TypeError，checker与owner均wait1/reap。未将其计为正常完成，也未重采首条。222另存JSON数组规范化适配器和checker，原算法/容差不变；9项相关测试与164行完整JSON往返/10类损坏拒绝通过。

恢复owner在`outputs/preview_control_serialized_20261026`，case数据保持在`outputs/preview_control_20261026`。新配置与原配置仅增加序列化验证proof，采集使用原冻结入口。新owner先独立检查原case00，随后采集/检查case01–17。36个有效数据stage由原已成功case00采集加新owner35个stage组成；复用164周期，新采4743周期，共4907。原失败checker receipts保持原verification/case00，新checker receipts另存verification_serialized，不覆盖。

完整准入119项CPU tests、18实际CLI dry-run和checker导入通过。全部新实现/检查文件Ruff和格式检查通过；全仓库检查另发现历史`validate_temporal_response.py`导入顺序和`audit_rotation_goal_precision.py`格式两处差异，它们受既有SHA绑定，本轮未改动，未宣称全仓库检查通过。

独立收尾已完成：18条实际端点/图像/命令/历史检查，36个有效stage实际wait0/reap，新owner正常wait0/reap、所有进程PID/start_ticks消失及来源保持。原失败checker/owner的wait1/reap分别核对并保留。首版收尾脚本在故障日志检查处混用文本/bytes而失败于写结果前；另存`close_preview_control_serialized_checked.py`修正字节类型后完成核验，原脚本保持。证据为恢复根的`regression_summary.json`和`closure_verified.json`。

最终任务稳定16/18，综合资格13/18，未达原18/18。综合要求每条任务稳定且位置RMSE<=.01m、SO3 RMSE<=.05rad。

| 未达标case | 稳定 | 位置RMSE m | SO3 RMSE rad | 未满足项 |
| --- | --- | ---: | ---: | --- |
| 06 | 否 | .02022157 | .05234771 | 任务、位置、旋转 |
| 09 | 是 | .01063868 | .01466837 | 位置 |
| 10 | 是 | .01068162 | .02047697 | 位置 |
| 12 | 否 | .00094045 | .01042440 | 任务 |
| 15 | 是 | .01574325 | .02632567 | 位置 |

首条case00稳定，位置RMSE .00142971m、SO3 .00607218rad，末尾连续81周期成功。case03和14也完整达标，其中case14位置.00812209m/SO3 .01159125rad。case12再次表明亚毫米位姿误差不能替代稳定任务成功。

同原18条的闭合对照：085因果反馈稳定17/18、综合15/18；213早裁剪直接逆稳定1/18、综合0/18；216晚裁剪直接逆稳定10/18、综合1/18；221预览稳定16/18、综合13/18。213与216也已完成36stage+owner和来源closure。预览解决部分旧失败，同时引入其他失败，不能按单条或较小动作误差宣称胜出。WAMdev及最终24新初态仍关闭，总目标未完成。
