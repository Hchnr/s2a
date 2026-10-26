# 093 原六baseline前缀实际控制器追踪

091/092显示无约束动力学遗漏group4桌面接触；实际uncoupling和扭矩饱和未直接测量。对086全部六baseline重放相同初态/seed42/命令/固定控制器与settled goal协议，不改命令、不拟合。只记录最后一个周期的25次内部控制调用：实际uncoupling、initial_joint、关节状态、goal、raw torque、robot clipped torque、实际ctrl、限幅、接触geom/距离、时间。接触记录为控制调用结束后/积分前的几何，不冒充传感器力。禁止使用追踪结果修改本次命令。

每一prefix endpoint完整sim_state及SITE pose/twist必须与086同case旧数据在1e-9内匹配；末周期实际ctrl必须等于clip(raw torque,actual robot limits)。旁路观察器不增加forward/step或修改返回torque。先通过观测不变和恢复测试、六dry-run准入；持久owner以每case worker/checker拥有实际wait/reap凭据，全部完成再关闭。新物理控制周期预算=原六baseline前缀共477（3339/7）；MuJoCo积分子步另计，世界模型/fresh/参数fit均0。env初始化内部行为与显式命令预算分开记录，不能宣称整进程零物理计算。

结果仅作历史协议一致条件下的运行时诊断，不计正确参考18/18或WAM/fresh成功。原门槛不变。生成产物新namespace outputs/native_controller_trace_20261025，不覆盖086/091/092。
