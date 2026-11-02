# 235 完整日志准入返回值比较修正

234首个独立比较在0周期失败，原failure.json、空日志、源码及监督actual wait1/reaped保留。生产转换器和独立replay的返回字段集合不同，不能整份字典要求同构。checked新入口按原221准入做法比较两者command和response_residual；SOURCE最终commit仍逐字段比较。独立完整日志检查继续核对原始状态/M/J、native goal、ODE/KKT、逆命令、双历史和165端点，16项损坏拒绝以及JSON164周期要求均不变。

新namespace只能在旧进程已消失、实际wait1/reap核验后一次执行。不是自动重试，不覆盖旧记录；0新增物理/拟合/模型。原18/18与最终WAM/fresh门槛保持。
