# Cap约束与旋转诊断（2026-11-02，Asia/Shanghai）

保存数据重建没有新增QP/NNLS、响应、模型查询或物理。输入是已独立审核通过的
cap记录与新stencil；重建所有已接受/反馈的raw Js/Jq/Je，并核对每个pose的B。
源数据与代码SHA前后保持。不能用这个局部模型分析证明全局可达性或物理成功。

59个已执行pose trial中29个被接受，27/29接受后reserve低于原.9门，之后回到
restore。最后合格pose为stage58/anchor85，radius=.005，QP步inf=.00475647。
最终命令最大偏移=.09999858044658，接近原.1，但近边界本身不足以证明主要瓶颈。

在stage58原单位KKT中，梯度inf=5.0253112，stationarity inf=5.10114e-8。各组
A_group.T@mu_group的inf分别：safety5.0972013、rate lower .1393911、rate upper
.1730675、command lower .0100818、command upper0、pose ceiling0。
这里比较的是同一归一化command梯度单位的向量贡献，不直接比较不同单位的mu，
也不将各组inf相加作百分比或把它们当成物理力。

最强行是关节7的braking_upper，0起period10/9/8的贡献inf分别3.1797925、
2.7847761、1.6930078，terminal贡献1.5348766。安全约束与目标梯度在该局部解上
显著竞争，直接扩大global盒并无充分依据；不放宽原安全门。

已收费pose trial的归一化signed E预测误差最大inf=.00643044。接受pose中仅1次
出现“预测最后旋转q下降、实际上升”，但这仅验证已采样方向，不保证完整Je正确，
也不能据此排除其他方向或控制器动态限制。当前证据不足以在目标取舍、约束和
方向模型之间确定唯一原因。

下一步单独登记组23旋转方向诊断，保持同一局部约束和原全部验收门，最多一个QP
及两侧共6完整响应，完整报告所有组变化。该单组试验不是完整方法胜出或任务成功。

证据：outputs/cap_distance_constraint_diagnostic_20261102/analyze.py、result.json；
result SHA256 f22f8e44e53d3dc7fd4f2f8f250feb6fafef02fe55fd4f9c10dc3f6c1245ef76。
代码/数据前后SHA和逐trial、逐非零dual行保存在该结果中；处理进程退出0并已回收。
