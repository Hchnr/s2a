# 236 JSON日志数值与元数据比较

235新入口已完成首周期独立raw-state/ODE/KKT验证和16项损坏拒绝，随后重复检查SOURCE response_residual时误将字符串元数据传给NumPy isfinite，TypeError退出。已实际wait1/reap，旧源码、failure和空日志保持。此错误发生在准入比较代码，不是控制性能失败或通过。

serialized新入口统一使用已有compare对字典递归比较：数值按原1e-9，字符串/None精确相等。删除同一字段的错误重复数值循环，其余全部验证保持。新root一次执行，先核实旧实际wait/reap和进程消失；不自动重试、不覆盖。正式控制仍须164周期完整JSON日志通过，原性能门槛不变。
