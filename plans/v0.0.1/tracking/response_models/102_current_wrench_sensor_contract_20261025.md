# 102 当前力/力矩反馈合同与baseline重放

101数值正确但性能5/24预测、18/24响应失败，不采用。核查机器人当前可读wrench是否补充约束信息，先确认可用性，不拟合模型/执行新控制。实际XML定义gripper0_force_ee/torque_ee，安装于gripper0_ft_frame；robosuite robot.ee_force/ee_torque只读取sensordata。MuJoCo2.3.7官方定义为传感器局部坐标系、子体向父体的相互作用力/力矩，包含多种动力学作用，不能称为纯外部接触力。参考 https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#sensor-force 。

转换为world方向并把moment参考点移至实际controller SITE：Fworld=Rws Flocal，Tworld_at_site=Rws Tlocal+(p_sensor-p_site)×Fworld。保留local原读数、sensor/site位姿、时刻和名称/类型/地址/维数；SI约定按固定机器人数据接口记录N/Nm，不改变符号。没有扣重力/载荷/惯性，也不声称sim读数已是真机传感器认证。

对086原六baseline按同初态/seed42/命令/固定控制器重放477显式控制周期；每endpoint在原endpoint刷新后、本周期新命令前读取，读函数本身不得forward/step或改变state/ctrl/sensordata。全部prefix完整state/SITE pose/twist仍与086匹配。直接sensor缓冲与getter一致，独立逐endpoint重算旋转与moment平移并审查有限值、时间和读取只读性。观察wrench不修改命令，不使用object/contact真值作为在线条件。

先做真实杠杆臂/旋转/非法输入测试、6CLI dry及原观察器测试准入。持久owner逐worker/checker及owner实际wait/reap，来源冻结。结果新namespace outputs/current_wrench_sensor_20261025；不得覆盖093。初始化内部步骤另列，世界模型/fresh/fit0。尚不扩充全部fit/holdout或probe数据，不提前授予模型/控制资格，原最终验收保持。
