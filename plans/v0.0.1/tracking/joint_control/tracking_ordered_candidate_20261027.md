# 原运算顺序候选完整18条验证

原case06移植入口任务退化后，已通过完整334步基线/原结果逐值一致证据验证原参考矩阵乘法顺序的恢复。此次只恢复输入SITE/anchor变换的运算顺序，旧候选失败及旧矩阵保留，不混合两个版本结果。

新owner `scripts/execute_tracking_correct_reference_ordered.py` 首先执行case06，随后case00..05、07..17，共原18条/4907周期；每条另从原初态独立重放已执行命令，共4907回放周期。原参考输入、seed42、固定20Hz控制器/2ms内环、夹爪、40步参考/每8步更新保持。门槛保持全部18条stable40且位置RMSE≤.01m、SO(3)RMSE≤.05rad；性能失败记录并继续完整矩阵，执行/数值失败终止，无自动重试、不覆盖产物。

所有模型引用、来源字节与入口冻结在 `outputs/tracking_correct_reference_ordered_execution_20261027/registration.json`；CPU/OSMesa运行，CUDA/LD均为空、三个CPU线程变量8。新增WAM/fresh均0。逐条closed.json须包含候选和独立回放的实际wait/reap阶段凭据。原输出写 `outputs/tracking_correct_reference_ordered_20261027/caseXX_candidate/`，独立结果写 `outputs/tracking_correct_reference_ordered_check_20261027/caseXX/verified.json`。

准备了适配该版本输出路径的 `audit_tracking_ordered_robot_inputs.py` 与 `audit_tracking_ordered_labels_images.py`，必须在实际结果产生后完成审查，不能以代码存在代替执行证据。整体规划输入因果性/模型约束及实时开销仍须如实评估；物理回放一致与原性能通过本身不能证明限位优化有效或最终24条新初态成功。完整资格关闭前不进入WAM/fresh，原总目标和所有后续门槛保持。
