# WAM预览控制的案例配置与采集接入

新增`src/s2a/coupled_preview_wam_control.py`与`scripts/collect_coupled_preview_wam_development.py`，并在新执行准入模块接入原生导入guard。没有启动模型、仿真或创建实际新执行登记。

配置生成保留固定12案例、seed42/43、600周期/H40/每8步重观测/末40步稳定，继承已获得物理资格的控制参数和固定079夹爪（gripper_preview_steps=0）。每案例输出目录隔离，checkpoint明确GPU UUID和HTTP31365；不把正确参考profiles或oracle_reference_root带入WAM配置。raw/terminal058采用同一个控制器。

采集入口先核对完整新登记、state资格、固定案例配置与原生preload证据，再导入实际预览WAM物理入口。当前endpoint observer仅记录实测端点及三张锚点图；完成后要求601个端点，保存raw-state/夹爪缓存和由实测数据计算的同步标签。同步标签只在执行结束后生成，不反馈到在线逆模型。保存完整HTTP请求摘要、实际600步/75请求结果及退出前环境；任一异常保留worker_result失败记录。

验证：16项采集/服务组合CPU测试通过（11.09秒），覆盖全部12案例配置、固定参数与UUID、无专家引用、错误case拒绝、当前图像/任务/seed/timeout/HTTP路由限制，以及dry采集不创建物理目录或执行preload。脚本实际`--help`退出0（日志`outputs/commit_preview_20261026/wam_collector_help.txt`）。相关文件Ruff与格式检查通过。

这些测试使用合成准入fixture，不能当成真实完整CLI dry。后续仍需独立WAM结果审查、总owner、metadata与prepare/register流程；当前新登记的角色清单不完整，尚不具备执行资格。

正确参考回归最新快照为前16条独立核验通过，case16核验worker PID1108821经/proc核实同身份存活。最后案例、完整退出与来源关闭尚未完成，WAM/fresh均未执行。总目标保持。
