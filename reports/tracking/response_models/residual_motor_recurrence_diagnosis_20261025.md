# 074补充：单步监督与命令状态递推不一致

运行scripts/diagnose_residual_motor_recurrence.py，固定073参数和原30条轨迹，不拟合、不运行物理或模型。每周期当前实测q和前一q均取原日志，desired仍取本周期已提供的后40端点；仅把上一周期自身选出的native -1/0/+1经过25次0.01 clip累计的motor cache递推到下一周期。此为混合日志回放，q没有受新命令驱动，不能当作前向物理rollout、线上成功或因果反事实。

|内部留出|teacher可达motor RMSE|递推motor RMSE|偏离native motor>0.25的周期|
|---|---:|---:|---:|
|bowl/demo40|0.02712|0.23483|34|
|bowl/demo48|0.01917|0.55671|45|
|wine/demo39|0.05988|0.64098|93|
|wine/demo46|0.02369|0.30342|79|
|cream/demo40|0.01705|0.06379|0|
|cream/demo49|0.01853|0.10150|0|

六条teacher误差与073报告一致；独立标量重放不调用在线特征/适配器/量化函数。输入NPZ/参数/代码读取前后SHA一致，详细30条命令及递推motor保存outputs/residual_motor_recurrence_diagnosis_20261025/diagnosis.json。Ruff及实际执行通过，0新fit/模型请求/物理/fresh。

该诊断只支持“单步teacher条件误差不能约束自身命令状态递推”的判断，不能证明整个任务失败唯一来自motor递推；真实物理case03/case05已有失败证据。后续候选应把命令后的state响应与因果反馈作为模型/评价对象，而不是继续仅降低历史motor重建误差。原074完整批次仍继续，原18/18及WAM/fresh门槛不变。
