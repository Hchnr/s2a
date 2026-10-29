# 158 首条控制审查失败数值路径诊断

157首条164物理步正常退出，独立checker goal_delta误差约1.17e-8超过1e-8而失败，owner实际wait1/reap后停止。不改冻结源码、不放宽容差、不重跑物理。只读日志/控制器源码，核查quat2mat内部float32量化是否复现controller实际goal，记录全部164周期的误差及源码SHA。后续修复用新namespace，旧失败不能改成通过。
