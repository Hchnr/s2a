# 160 数值接口修复后原18条回归已启动

新namespace outputs/qualified_inverse_native_precision_20261025，原18profiles/4907步、146参数、旋转/夹爪策略和全部严格门槛保持。执行器仅把在线模型旋转goal读取改为159实际控制器数值解码，旧157首条审查失败记录及冻结源码未改。新准入完整通过：71项CPU测试、18实际CLI dry和完整独立checker导入，159全部164历史goal严格1e-9复现凭据绑定。

持久supervisor PID78820、owner PID78942、case00 worker PID79000均已实际/proc PID与start_ticks匹配存活，冻结来源不变。当前无完整新控制性能结论；不得把修复或启动当18/18通过。接口/机械失败停止，不自动重试。全局fit/世界模型/fresh0，物理按实际日志记录。

最终须原18/18稳定、位置RMSE<=.01m、SO(3)RMSE<=.05rad及完整36stage+owner实际wait/reap/来源审查通过。WAM开发与最终24新初态保持关闭，总目标未完成。

## 实际终止结果

首条case00完成90/164周期后，下一次平移逆求解抛出`RuntimeError('Bounded response inverse did not converge')`，物理worker及owner均实际wait1并回收，无signal/kill、无自动重试。凭据见`outputs/qualified_inverse_native_precision_20261025/orchestration/case00/exit.json`及`owner_wait/release.json`。后17条未执行，未进入独立完整case检查。部分任务成功和误差不算控制资格。旧冻结源码保留；161在新模块检查相同三变量目标的显式边界求解，不修改此失败记录或自动重启此namespace。
