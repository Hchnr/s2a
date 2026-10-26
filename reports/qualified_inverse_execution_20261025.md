# 157 原18条正确参考控制回归已启动

固定18profiles/4907步、seed42与原位置/旋转/稳定门槛，合格146参数/148离线资格/152真实在线输入parity/156完整checker绑定。正式准入全部通过：70项CPU测试、18实际CLI dry、完整checker导入。冻结所有src/scripts/配置/计划/参考/模型来源，通过后启动持久supervisor与owner，无自动重试。

已实际核验supervisor PID68943、owner PID69003、case00 worker PID69062均/proc PID与start_ticks匹配存活，来源指纹未变。输出outputs/qualified_inverse_control_20261025。当前尚无完整单条或全量性能结果，不能从启动推断18/18通过；任意接口/机械审查失败停止并保留凭据。新增全局fit/世界模型/fresh0，物理步按实际执行日志统计。

全量必须18/18稳定、位置RMSE<=.01m、SO(3)RMSE<=.05rad并完整36stage及owner实际wait/reap/来源核验通过。WAM开发和最终24新初态仍关闭，总目标未完成。

首条执行终止记录：case00完成164步且worker正常wait0，独立checker action_goal_delta数值合同失败，checker/owner实际wait1/reap，任务已停止，未执行后17条。158只读诊断确认实际controller quat2mat的float32量化路径，旧151输入非零旋转按理想float64解码存在最多1.03e-7误差；详情reports/rotation_goal_precision_incident_20261025.md。原失败凭据与源码保持，未自动重试或放宽原门槛，不具备控制资格。
