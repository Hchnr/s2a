# 158 首条控制审查失败定位

157 case00全部164物理步正常wait0，独立checker在model action_goal_delta与实际controller goal比较失败（首次1.16686781e-8超过原1e-8），owner实际wait1/reap、无信号和自动重试，按规则停止。旧失败、参数、源码与物理日志保留，未改门槛或重跑物理。

已读实际robosuite set_goal_orientation路径：axisangle2quat后quat2mat将quat转float32再计算旋转矩阵，而151非零旋转目标按理想float64轴角输入。用实际依赖转换函数复现全部164次实际controller goal矩阵，全部在1e-9矩阵检查内通过；记录的理想目标向量与实际Log(goal*currentR^T)最大差1.0303205522088632e-7。独立float32公式仅在float32误差界内核对，不伪称bit-exact；原实际goal矩阵检查容差仍1e-9。

证据outputs/rotation_goal_precision_incident_20261025/result.json保存每步误差与源码SHA。该诊断无新物理/fit/世界模型/fresh。应在新namespace让在线输入使用实际控制器数值解码（仍纯只读），保持原严格goal检查；旧157不能据此改写为合格。当前没有18/18控制资格，WAM/fresh仍关闭，总目标未完成。
