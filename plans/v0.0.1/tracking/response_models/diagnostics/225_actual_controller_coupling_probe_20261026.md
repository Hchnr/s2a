# 225 同冻结入口重建实例的控制器字段与M/J核对

224条件模型在case10/15明显改善，但历史合同未直接记录uncoupling。此步在CPU/OSMesa下，按已冻结make_environment、实际record/XML及runtime_config重建case15/06/14三个任务实例，恢复已保存step550/220/60端点，直接读controller.uncoupling、impedance_mode/use_delta及实际kp/kd/接口。

禁止执行action或积分：在实例创建/重置前拦截mujoco.mj_step、mj_step1、mj_step2，有调用立即抛错并保留失败；只允许reset/restore/mj_forward和controller.update(force=True)。核对恢复sim_state、实际SITE pose/twist，并把controller.mass_matrix与J_full和独立mj_fullM/mj_jacSite的7关节子矩阵比较。前后完整状态不得改变，计数必须全零。

这能证明相同冻结入口重建实例的实际字段与局部矩阵，不能伪称死去的历史进程字段曾被记录。后续若采用耦合模型，真实执行仍须每case/当前实例显式记录和验证此字段及源。三任务字段核对不替代任何一步预测/闭环资格，无新拟合、世界模型调用或控制周期。
