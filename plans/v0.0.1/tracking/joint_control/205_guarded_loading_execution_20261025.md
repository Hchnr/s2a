# 205 预测跟踪约束加载限速的完整18条回归

202/204固定策略，保留178混合候选，仅加载支持范围内从.125至.25选择满足146预测位置误差<=5mm的最小上限。无可行解或支持外保留178命令；不是实际误差保证。不调半径/模型，不使用未资格手指模型，不按case/task/未来成功切换。

新namespace outputs/qualified_inverse_guarded_loading_20261025。原18profiles/4907周期、原参考时钟、控制器、旋转、夹爪和完整稳定/误差门槛不变。独立重算当前前向映射、根/可行性及实际命令/预测，173/174按最终命令更新两套历史。完整CPU tests、18实际CLI dry、checker导入、204全部164状态proof与来源准入后持久supervisor/owner逐case collect/verify；机械失败停止、不自动重试。

须18/18稳定、位置RMSE<=.01m、SO3RMSE<=.05rad和36stage+owner实际wait/reap/来源资格才进入WAMdev，之后最终24新初态仍必需。200旧对照继续且源码不变，当前未证明新策略控制收益。
